"""Bybit V5 REST client — direct aiohttp with manual HMAC-SHA256 signing.

Replaces the pybit-based implementation to give full control over request
signing and eliminate pybit version incompatibilities.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from decimal import Decimal
from typing import Any
from urllib.parse import urlencode

import aiohttp
import structlog

from trader.domain.errors import (
    AuthenticationError,
    InsufficientFundsError,
    OrderRejectedError,
    RateLimitError,
    TradingSystemError,
)
from trader.exchange.endpoint_selector import EndpointSelector
from trader.exchange.rate_limiter import RateLimiter

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Bybit retCode → exception mapping
# ---------------------------------------------------------------------------

_BYBIT_ERROR_MAP: dict[int, type[TradingSystemError]] = {
    10003: AuthenticationError,
    10004: AuthenticationError,
    10006: RateLimitError,
    110007: InsufficientFundsError,
    110013: OrderRejectedError,
    110014: OrderRejectedError,
    110017: OrderRejectedError,
    110025: OrderRejectedError,
    110043: OrderRejectedError,
}

_NON_RETRYABLE_CODES = {10003, 10004, 110007, 110013, 110014, 110017, 110025}
_ALLOWED_NON_ZERO_CODES = {110043}  # "set leverage not modified" — not a real error


def _raise_for_ret_code(response: dict[str, Any], context: str = "") -> None:
    """Raise an appropriate exception if retCode != 0."""
    ret_code = response.get("retCode", 0)
    if ret_code == 0:
        return
    if ret_code in _ALLOWED_NON_ZERO_CODES:
        logger.debug("bybit_non_fatal_code", ret_code=ret_code, context=context)
        return

    msg = response.get("retMsg", f"Bybit error retCode={ret_code}")
    exc_class = _BYBIT_ERROR_MAP.get(ret_code)

    if exc_class is AuthenticationError:
        raise AuthenticationError(f"{msg} (code={ret_code})")
    if exc_class is RateLimitError:
        raise RateLimitError(f"{msg} (code={ret_code})")
    if exc_class is InsufficientFundsError:
        raise InsufficientFundsError(f"{msg} (code={ret_code})")
    if exc_class is OrderRejectedError:
        raise OrderRejectedError(
            f"{msg} (code={ret_code})",
            exchange_code=str(ret_code),
        )
    raise TradingSystemError(f"{msg} (code={ret_code})", code=str(ret_code))


class BybitRestClient:
    """Async REST client for Bybit V5 API using aiohttp with manual HMAC signing.

    All authenticated requests use the standard Bybit V5 signature:
        HMAC-SHA256(timestamp + apiKey + recvWindow + queryString, apiSecret)
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        endpoint_selector: EndpointSelector,
        rate_limiter: RateLimiter,
        use_testnet: bool,
        use_rsa: bool = False,
        rsa_private_key: str | None = None,
        recv_window: int = 5000,
        max_workers: int = 4,
    ) -> None:
        # Strip to guard against copy-paste whitespace in env vars
        self._api_key = api_key.strip()
        self._api_secret = api_secret.strip()
        self._endpoint_selector = endpoint_selector
        self._rate_limiter = rate_limiter
        self._recv_window = recv_window
        self._base_url = endpoint_selector.rest_base
        self._session: aiohttp.ClientSession | None = None

        if self._api_key:
            logger.info(
                "bybit_rest_client_init",
                base_url=self._base_url,
                credentials_configured=True,
                recv_window=self._recv_window,
            )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
            )
        return self._session

    # ------------------------------------------------------------------
    # Signature
    # ------------------------------------------------------------------

    def _sign(self, timestamp: str, params_str: str) -> str:
        """Generate HMAC-SHA256 signature for Bybit V5."""
        payload = f"{timestamp}{self._api_key}{self._recv_window}{params_str}"
        return hmac.new(
            self._api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _auth_headers(self, timestamp: str, signature: str) -> dict[str, str]:
        return {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-ALGO": "HMAC_SHA256",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": str(self._recv_window),
        }

    # ------------------------------------------------------------------
    # Core request
    # ------------------------------------------------------------------

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        authenticated: bool = True,
    ) -> dict[str, Any]:
        endpoint_hint = path
        await self._rate_limiter.acquire(endpoint_hint, "GET")

        t0 = time.monotonic()
        log = logger.bind(method="GET", endpoint=path)
        log.debug("bybit_rest_request")

        params = {k: v for k, v in (params or {}).items() if v is not None}
        query_string = urlencode(params)
        url = f"{self._base_url}{path}"

        headers: dict[str, str] = {}
        if authenticated and self._api_key:
            timestamp = str(int(time.time() * 1000))
            signature = self._sign(timestamp, query_string)
            headers = self._auth_headers(timestamp, signature)

        try:
            session = self._get_session()
            async with session.get(url, params=params, headers=headers) as resp:
                http_status = resp.status
                resp_headers = dict(resp.headers)
                response: Any = await resp.json(content_type=None)
        except Exception as exc:
            log.error("bybit_rest_exception", error=str(exc))
            raise

        self._rate_limiter.record_response(endpoint_hint, resp_headers, "GET")

        elapsed_ms = (time.monotonic() - t0) * 1000
        ret_code = response.get("retCode", 0) if isinstance(response, dict) else 0
        log.debug(
            "bybit_rest_response",
            http_status=http_status,
            ret_code=ret_code,
            elapsed_ms=round(elapsed_ms, 1),
        )

        if not isinstance(response, dict):
            raise TradingSystemError(
                f"Bybit returned non-dict response for {path}: "
                f"http_status={http_status} type={type(response).__name__} value={response!r}",
                code="INVALID_RESPONSE",
            )

        _raise_for_ret_code(response, context=path)
        return response

    async def _post(
        self,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        endpoint_hint = path
        await self._rate_limiter.acquire(endpoint_hint, "POST")

        t0 = time.monotonic()
        log = logger.bind(method="POST", endpoint=path)
        log.debug("bybit_rest_request")

        body = body or {}
        body_str = json.dumps(body)
        url = f"{self._base_url}{path}"

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            timestamp = str(int(time.time() * 1000))
            signature = self._sign(timestamp, body_str)
            headers.update(self._auth_headers(timestamp, signature))

        try:
            session = self._get_session()
            async with session.post(url, data=body_str, headers=headers) as resp:
                http_status = resp.status
                resp_headers = dict(resp.headers)
                response: Any = await resp.json(content_type=None)
        except Exception as exc:
            log.error("bybit_rest_exception", error=str(exc))
            raise

        self._rate_limiter.record_response(endpoint_hint, resp_headers, "POST")

        elapsed_ms = (time.monotonic() - t0) * 1000
        ret_code = response.get("retCode", 0) if isinstance(response, dict) else 0
        log.debug(
            "bybit_rest_response",
            http_status=http_status,
            ret_code=ret_code,
            elapsed_ms=round(elapsed_ms, 1),
        )

        if not isinstance(response, dict):
            raise TradingSystemError(
                f"Bybit returned non-dict response for {path}: "
                f"http_status={http_status} type={type(response).__name__} value={response!r}",
                code="INVALID_RESPONSE",
            )

        _raise_for_ret_code(response, context=path)
        return response

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------

    async def get_server_time(self) -> dict[str, Any]:
        return await self._get("/v5/market/time", authenticated=False)

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_wallet_balance(self, account_type: str = "UNIFIED") -> dict[str, Any]:
        return await self._get(
            "/v5/account/wallet-balance",
            params={"accountType": account_type},
        )

    async def get_account_info(self) -> dict[str, Any]:
        return await self._get("/v5/account/info")

    async def get_api_key_info(self) -> dict[str, Any]:
        return await self._get("/v5/user/query-api")

    # ------------------------------------------------------------------
    # Instruments / market data
    # ------------------------------------------------------------------

    async def get_instruments_info(self, category: str, symbol: str | None = None) -> dict[str, Any]:
        return await self._get(
            "/v5/market/instruments-info",
            params={"category": category, "symbol": symbol},
            authenticated=False,
        )

    async def get_tickers(self, category: str, symbol: str | None = None) -> dict[str, Any]:
        return await self._get(
            "/v5/market/tickers",
            params={"category": category, "symbol": symbol},
            authenticated=False,
        )

    async def get_conservative_market_price(
        self,
        category: str,
        symbol: str,
        side: str,
    ) -> Decimal:
        """Return a conservative market price for pre-order notional checks.

        Buy  → ask1Price (we'll pay at least this much)
        Sell → bid1Price (we'll receive at most this much)
        Falls back to lastPrice if ask/bid temporarily unavailable.
        """
        resp = await self.get_tickers(category=category, symbol=symbol)
        items = resp.get("result", {}).get("list", [])
        if not items:
            raise TradingSystemError(f"No ticker data for {symbol}", code="NO_TICKER")

        ticker = items[0]
        side_upper = side.upper()
        if side_upper in ("BUY", "LONG"):
            price_str = ticker.get("ask1Price") or ticker.get("lastPrice", "0")
        else:
            price_str = ticker.get("bid1Price") or ticker.get("lastPrice", "0")

        price = Decimal(str(price_str))
        if price <= Decimal("0"):
            raise TradingSystemError(
                f"Conservative market price is zero for {symbol} side={side}",
                code="ZERO_PRICE",
            )
        return price

    async def get_kline(
        self,
        category: str,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        return await self._get(
            "/v5/market/kline",
            params={
                "category": category,
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
                "start": start,
                "end": end,
            },
            authenticated=False,
        )

    async def get_orderbook(self, category: str, symbol: str, limit: int = 50) -> dict[str, Any]:
        return await self._get(
            "/v5/market/orderbook",
            params={"category": category, "symbol": symbol, "limit": limit},
            authenticated=False,
        )

    async def get_recent_trades(self, category: str, symbol: str, limit: int = 60) -> dict[str, Any]:
        return await self._get(
            "/v5/market/recent-trade",
            params={"category": category, "symbol": symbol, "limit": limit},
            authenticated=False,
        )

    async def get_funding_rate_history(self, category: str, symbol: str, limit: int = 200) -> dict[str, Any]:
        return await self._get(
            "/v5/market/funding/history",
            params={"category": category, "symbol": symbol, "limit": limit},
            authenticated=False,
        )

    async def get_open_interest(
        self,
        category: str,
        symbol: str,
        interval_time: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        return await self._get(
            "/v5/market/open-interest",
            params={
                "category": category,
                "symbol": symbol,
                "intervalTime": interval_time,
                "limit": limit,
            },
            authenticated=False,
        )

    async def get_long_short_ratio(
        self,
        category: str,
        symbol: str,
        period: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        return await self._get(
            "/v5/market/account-ratio",
            params={
                "category": category,
                "symbol": symbol,
                "period": period,
                "limit": limit,
            },
            authenticated=False,
        )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def place_order(self, **kwargs: Any) -> dict[str, Any]:
        return await self._post("/v5/order/create", body=kwargs)

    async def amend_order(self, **kwargs: Any) -> dict[str, Any]:
        return await self._post("/v5/order/amend", body=kwargs)

    async def cancel_order(
        self,
        category: str,
        symbol: str,
        order_id: str | None = None,
        order_link_id: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"category": category, "symbol": symbol}
        if order_id:
            body["orderId"] = order_id
        if order_link_id:
            body["orderLinkId"] = order_link_id
        return await self._post("/v5/order/cancel", body=body)

    async def get_open_orders(
        self,
        category: str,
        symbol: str | None = None,
        settle_coin: str | None = None,
        base_coin: str | None = None,
    ) -> dict[str, Any]:
        # Bybit /v5/order/realtime requires symbol, settleCoin, or baseCoin
        # when listing all orders (not filtered by a specific symbol).
        params: dict[str, Any] = {"category": category}
        if symbol:
            params["symbol"] = symbol
        elif settle_coin:
            params["settleCoin"] = settle_coin
        elif base_coin:
            params["baseCoin"] = base_coin
        elif category == "linear":
            params["settleCoin"] = "USDT"
        elif category == "inverse":
            params["settleCoin"] = "BTC"
        return await self._get("/v5/order/realtime", params=params)

    async def get_order_history(
        self,
        category: str,
        symbol: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        return await self._get(
            "/v5/order/history",
            params={"category": category, "symbol": symbol, "limit": limit},
        )

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def get_positions(self, category: str, symbol: str | None = None) -> dict[str, Any]:
        # Bybit requires symbol OR settleCoin when listing all positions
        params: dict[str, Any] = {"category": category}
        if symbol:
            params["symbol"] = symbol
        elif category == "linear":
            params["settleCoin"] = "USDT"
        elif category == "inverse":
            params["settleCoin"] = "BTC"
        return await self._get("/v5/position/list", params=params)

    async def set_leverage(
        self,
        category: str,
        symbol: str,
        buy_leverage: str,
        sell_leverage: str,
    ) -> dict[str, Any]:
        return await self._post(
            "/v5/position/set-leverage",
            body={
                "category": category,
                "symbol": symbol,
                "buyLeverage": buy_leverage,
                "sellLeverage": sell_leverage,
            },
        )

    async def set_trading_stop(self, category: str, symbol: str, **kwargs: Any) -> dict[str, Any]:
        return await self._post(
            "/v5/position/trading-stop",
            body={"category": category, "symbol": symbol, **kwargs},
        )

    # ------------------------------------------------------------------
    # Executions / PnL
    # ------------------------------------------------------------------

    async def get_executions(
        self,
        category: str,
        symbol: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        return await self._get(
            "/v5/execution/list",
            params={"category": category, "symbol": symbol, "limit": limit},
        )

    async def get_closed_pnl(
        self,
        category: str,
        symbol: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        return await self._get(
            "/v5/position/closed-pnl",
            params={"category": category, "symbol": symbol, "limit": limit},
        )

    async def get_fee_rate(self, category: str, symbol: str | None = None) -> dict[str, Any]:
        return await self._get(
            "/v5/account/fee-rate",
            params={"category": category, "symbol": symbol},
        )

    async def get_transaction_log(
        self,
        account_type: str = "UNIFIED",
        category: str = "linear",
        currency: str = "USDT",
        limit: int = 50,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "accountType": account_type,
            "category": category,
            "currency": currency,
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor
        return await self._get("/v5/account/transaction-log", params=params)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
