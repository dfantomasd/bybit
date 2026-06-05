"""Thin async REST client wrapping pybit's HTTP session.

pybit is a synchronous library; we wrap calls in a thread pool executor
so the rest of the async application is not blocked.
"""
from __future__ import annotations

import asyncio
import functools
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

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
    # Generic fallback
    raise TradingSystemError(f"{msg} (code={ret_code})", code=str(ret_code))


class BybitRestClient:
    """Async REST client for Bybit V5 API using pybit under the hood.

    All public methods are coroutines that run pybit's synchronous HTTP
    session in a thread-pool executor so the event loop is never blocked.
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
        self._api_key = api_key
        self._api_secret = api_secret
        self._endpoint_selector = endpoint_selector
        self._rate_limiter = rate_limiter
        self._use_testnet = use_testnet
        self._recv_window = recv_window
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="bybit_rest")

        # Build pybit HTTP session
        self._session = self._build_session(use_rsa, rsa_private_key)

    # ------------------------------------------------------------------
    # Session factory
    # ------------------------------------------------------------------

    def _build_session(self, use_rsa: bool, rsa_private_key: str | None) -> Any:
        """Instantiate a pybit HTTP session."""
        from pybit.unified_trading import HTTP

        kwargs: dict[str, Any] = {
            "testnet": self._use_testnet,
            "api_key": self._api_key,
            "api_secret": self._api_secret,
            "recv_window": self._recv_window,
        }
        if use_rsa and rsa_private_key:
            kwargs["rsa_authentication"] = True
            kwargs["private_key"] = rsa_private_key

        return HTTP(**kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call(self, method_name: str, endpoint_hint: str = "/", **kwargs: Any) -> dict[str, Any]:
        """Run a pybit session method in the thread pool, applying rate limiting.

        Args:
            method_name:   Name of the pybit HTTP session method to call.
            endpoint_hint: Short path used for rate limiter key (no real routing).
            **kwargs:      Arguments forwarded to the pybit method.
        """
        await self._rate_limiter.acquire(endpoint_hint, "GET")

        pybit_method = getattr(self._session, method_name)
        loop = asyncio.get_event_loop()

        t0 = time.monotonic()
        log = logger.bind(method=method_name, endpoint=endpoint_hint)
        log.debug("bybit_rest_request")

        try:
            response: dict[str, Any] = await loop.run_in_executor(
                self._executor,
                functools.partial(pybit_method, **kwargs),
            )
        except Exception as exc:
            log.error("bybit_rest_exception", error=str(exc))
            raise

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Feed headers back to rate limiter (pybit response dict has no headers, but
        # some pybit versions expose them via response["headers"] — handle gracefully)
        headers = response.get("headers", {}) if isinstance(response, dict) else {}
        if headers:
            self._rate_limiter.record_response(endpoint_hint, headers)

        ret_code = response.get("retCode", 0) if isinstance(response, dict) else 0
        log.debug(
            "bybit_rest_response",
            ret_code=ret_code,
            elapsed_ms=round(elapsed_ms, 1),
        )

        if isinstance(response, dict):
            _raise_for_ret_code(response, context=method_name)

        return response  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------

    async def get_server_time(self) -> dict[str, Any]:
        return await self._call("get_server_time", "/v5/market/time")

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_wallet_balance(self, account_type: str = "UNIFIED") -> dict[str, Any]:
        return await self._call(
            "get_wallet_balance",
            "/v5/account/wallet-balance",
            accountType=account_type,
        )

    async def get_account_info(self) -> dict[str, Any]:
        return await self._call("get_account_info", "/v5/account/info")

    async def get_api_key_info(self) -> dict[str, Any]:
        return await self._call("get_api_key_information", "/v5/user/query-api")

    # ------------------------------------------------------------------
    # Instruments / market data
    # ------------------------------------------------------------------

    async def get_instruments_info(
        self, category: str, symbol: str | None = None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"category": category}
        if symbol:
            kwargs["symbol"] = symbol
        return await self._call("get_instruments_info", "/v5/market/instruments-info", **kwargs)

    async def get_tickers(
        self, category: str, symbol: str | None = None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"category": category}
        if symbol:
            kwargs["symbol"] = symbol
        return await self._call("get_tickers", "/v5/market/tickers", **kwargs)

    async def get_kline(
        self,
        category: str,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start is not None:
            kwargs["start"] = start
        if end is not None:
            kwargs["end"] = end
        return await self._call("get_kline", "/v5/market/kline", **kwargs)

    async def get_orderbook(
        self, category: str, symbol: str, limit: int = 50
    ) -> dict[str, Any]:
        return await self._call(
            "get_orderbook",
            "/v5/market/orderbook",
            category=category,
            symbol=symbol,
            limit=limit,
        )

    async def get_recent_trades(
        self, category: str, symbol: str, limit: int = 60
    ) -> dict[str, Any]:
        return await self._call(
            "get_public_trade_history",
            "/v5/market/recent-trade",
            category=category,
            symbol=symbol,
            limit=limit,
        )

    async def get_funding_rate_history(
        self, category: str, symbol: str, limit: int = 200
    ) -> dict[str, Any]:
        return await self._call(
            "get_funding_rate_history",
            "/v5/market/funding/history",
            category=category,
            symbol=symbol,
            limit=limit,
        )

    async def get_open_interest(
        self,
        category: str,
        symbol: str,
        interval_time: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        return await self._call(
            "get_open_interest",
            "/v5/market/open-interest",
            category=category,
            symbol=symbol,
            intervalTime=interval_time,
            limit=limit,
        )

    async def get_long_short_ratio(
        self,
        category: str,
        symbol: str,
        period: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        return await self._call(
            "get_long_short_ratio",
            "/v5/market/account-ratio",
            category=category,
            symbol=symbol,
            period=period,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def place_order(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("place_order", "/v5/order/create", **kwargs)

    async def amend_order(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("amend_order", "/v5/order/amend", **kwargs)

    async def cancel_order(
        self,
        category: str,
        symbol: str,
        order_id: str | None = None,
        order_link_id: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"category": category, "symbol": symbol}
        if order_id:
            kwargs["orderId"] = order_id
        if order_link_id:
            kwargs["orderLinkId"] = order_link_id
        return await self._call("cancel_order", "/v5/order/cancel", **kwargs)

    async def get_open_orders(
        self, category: str, symbol: str | None = None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"category": category}
        if symbol:
            kwargs["symbol"] = symbol
        return await self._call("get_open_orders", "/v5/order/realtime", **kwargs)

    async def get_order_history(
        self,
        category: str,
        symbol: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"category": category, "limit": limit}
        if symbol:
            kwargs["symbol"] = symbol
        return await self._call("get_order_history", "/v5/order/history", **kwargs)

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def get_positions(
        self, category: str, symbol: str | None = None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"category": category}
        if symbol:
            kwargs["symbol"] = symbol
        return await self._call("get_positions", "/v5/position/list", **kwargs)

    async def set_leverage(
        self,
        category: str,
        symbol: str,
        buy_leverage: str,
        sell_leverage: str,
    ) -> dict[str, Any]:
        return await self._call(
            "set_leverage",
            "/v5/position/set-leverage",
            category=category,
            symbol=symbol,
            buyLeverage=buy_leverage,
            sellLeverage=sell_leverage,
        )

    async def set_trading_stop(
        self, category: str, symbol: str, **kwargs: Any
    ) -> dict[str, Any]:
        return await self._call(
            "set_trading_stop",
            "/v5/position/trading-stop",
            category=category,
            symbol=symbol,
            **kwargs,
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
        kwargs: dict[str, Any] = {"category": category, "limit": limit}
        if symbol:
            kwargs["symbol"] = symbol
        return await self._call("get_executions", "/v5/execution/list", **kwargs)

    async def get_closed_pnl(
        self,
        category: str,
        symbol: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"category": category, "limit": limit}
        if symbol:
            kwargs["symbol"] = symbol
        return await self._call("get_closed_pnl", "/v5/position/closed-pnl", **kwargs)

    async def get_fee_rate(
        self, category: str, symbol: str | None = None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"category": category}
        if symbol:
            kwargs["symbol"] = symbol
        return await self._call("get_fee_rate", "/v5/account/fee-rate", **kwargs)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shutdown the thread-pool executor."""
        self._executor.shutdown(wait=False)
