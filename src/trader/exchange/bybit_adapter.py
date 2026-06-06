"""High-level adapter composing REST client + state stores.

Primary interface between the trading system and Bybit.
Composes: EndpointSelector, RateLimiter, BybitRestClient, OrderMapper.
Provides typed, domain-level methods.
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

import structlog

from trader.domain.models import (
    Balance,
    InstrumentInfo,
    OrderIntent,
    Position,
    PreflightReport,
    ReconciliationResult,
)
from trader.exchange.bybit_rest import BybitRestClient
from trader.exchange.endpoint_selector import EndpointSelector
from trader.exchange.idempotency import IdempotencyManager
from trader.exchange.order_mapper import OrderMapper
from trader.exchange.preflight import PreflightChecker
from trader.exchange.rate_limiter import RateLimiter

logger = structlog.get_logger(__name__)


class BybitAdapter:
    """High-level adapter providing domain-level methods over the Bybit V5 API.

    Composes:
    - EndpointSelector  — correct URLs per region/testnet
    - RateLimiter       — adaptive token bucket
    - BybitRestClient   — async REST calls
    - OrderMapper       — domain ↔ API translation
    - IdempotencyManager — duplicate order prevention
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        region_code: str = "GLOBAL",
        use_testnet: bool = True,
        use_rsa: bool = False,
        rsa_private_key: str | None = None,
        default_category: str = "linear",
    ) -> None:
        from trader.domain.enums import BybitRegion

        region = BybitRegion(region_code)
        self._endpoint_selector = EndpointSelector(region=region, use_testnet=use_testnet)
        self._rate_limiter = RateLimiter()
        self._rest = BybitRestClient(
            api_key=api_key,
            api_secret=api_secret,
            endpoint_selector=self._endpoint_selector,
            rate_limiter=self._rate_limiter,
            use_testnet=use_testnet,
            use_rsa=use_rsa,
            rsa_private_key=rsa_private_key,
        )
        self._mapper = OrderMapper()
        self._idempotency = IdempotencyManager()
        self._default_category = default_category
        self._use_testnet = use_testnet

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> PreflightReport:
        """Run preflight checks and return a report.

        Raises ConfigurationError if critical checks fail.
        """
        checker = PreflightChecker(
            rest_client=self._rest,
            endpoint_selector=self._endpoint_selector,
            use_testnet=self._use_testnet,
        )
        report = await checker.run()
        logger.info(
            "bybit_adapter.initialized",
            passed=report.passed,
            checks=report.checks,
        )
        await self._check_clock_skew()
        return report

    async def _check_clock_skew(self) -> None:
        """Fetch Bybit server time and log clock offset; warn if > 2 s."""
        import time as _time

        try:
            resp = await self._rest.get_server_time()
            server_ms = int((resp.get("result") or {}).get("timeSecond", 0)) * 1000
            if not server_ms:
                server_ms = int((resp.get("result") or {}).get("timeNano", 0)) // 1_000_000
            local_ms = int(_time.time() * 1000)
            skew_ms = local_ms - server_ms
            if abs(skew_ms) > 2000:
                logger.warning(
                    "bybit_clock_skew_large",
                    skew_ms=skew_ms,
                    local_ms=local_ms,
                    server_ms=server_ms,
                )
            else:
                logger.info("bybit_clock_skew", skew_ms=skew_ms)
        except Exception as exc:
            logger.debug("bybit_clock_skew_check_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_balance(self) -> Balance:
        """Return the primary USDT balance from the UNIFIED account."""
        resp = await self._rest.get_wallet_balance(account_type="UNIFIED")
        accounts = (resp.get("result") or {}).get("list", [])
        for account in accounts:
            for coin_data in account.get("coin", []):
                if coin_data.get("coin") in ("USDT", "USDC"):
                    coin_data["accountType"] = "UNIFIED"
                    return self._mapper.rest_balance_to_model(coin_data)
        # Fallback: return zero balance
        return Balance(
            account_type="UNIFIED",
            currency="USDT",
            wallet_balance=0,  # type: ignore[arg-type]
            available_balance=0,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def get_positions(self, category: str) -> list[Position]:
        """Return all open positions for the given category."""
        resp = await self._rest.get_positions(category=category)
        items = (resp.get("result") or {}).get("list", [])
        positions = []
        for item in items:
            item["category"] = category
            pos = self._mapper.rest_position_to_model(item)
            positions.append(pos)
        return positions

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def get_open_orders(self, category: str, symbol: str | None = None) -> list[dict[str, Any]]:
        """Return open orders as raw dicts (mapper can be applied later)."""
        resp = await self._rest.get_open_orders(category=category, symbol=symbol)
        return (resp.get("result") or {}).get("list", [])

    async def place_order(self, intent: OrderIntent) -> dict[str, Any]:
        """Submit an order to Bybit after idempotency and mapper processing.

        Returns the raw Bybit response dict.
        """
        # Check duplicate
        if await self._idempotency.check_duplicate(intent.order_link_id):
            raise ValueError(f"Duplicate order detected: {intent.order_link_id}")

        # Register in idempotency store
        await self._idempotency.register_intent(intent)
        await self._idempotency.mark_submitted(intent.order_link_id)

        # Map to params
        params = self._mapper.intent_to_params(intent, self._default_category)

        try:
            resp = await self._rest.place_order(**params)
            exchange_id = (resp.get("result") or {}).get("orderId", "")
            await self._idempotency.mark_confirmed(intent.order_link_id, exchange_id)
            logger.info(
                "bybit_adapter.order_placed",
                order_link_id=intent.order_link_id,
                exchange_order_id=exchange_id,
                symbol=intent.symbol,
            )
            return resp
        except Exception as exc:
            logger.error(
                "bybit_adapter.order_failed",
                order_link_id=intent.order_link_id,
                error=str(exc),
            )
            raise

    async def cancel_order(self, category: str, symbol: str, order_link_id: str) -> dict[str, Any]:
        """Cancel an open order by order_link_id."""
        resp = await self._rest.cancel_order(
            category=category,
            symbol=symbol,
            order_link_id=order_link_id,
        )
        try:
            await self._idempotency.mark_cancelled(order_link_id)
        except Exception as exc:
            logger.debug(
                "bybit_adapter.cancelled_order_not_in_local_store",
                order_link_id=order_link_id,
                error=str(exc),
            )
        return resp

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_best_price(self, symbol: str) -> tuple[Decimal, Decimal]:
        """Return (bid, ask) for *symbol* using the current best prices.

        Fetches the V5 tickers endpoint for the adapter's default category.
        Returns ``(Decimal("0"), Decimal("0"))`` on any parse failure so the
        caller can decide how to handle degraded data.
        """
        resp = await self._rest.get_tickers(category=self._default_category, symbol=symbol)
        items = (resp.get("result") or {}).get("list", [])
        if not items:
            logger.warning("bybit_adapter.get_best_price_empty", symbol=symbol)
            return Decimal("0"), Decimal("0")
        item = items[0]
        try:
            bid = Decimal(str(item["bid1Price"]))
            ask = Decimal(str(item["ask1Price"]))
        except (KeyError, Exception) as exc:
            logger.warning("bybit_adapter.get_best_price_parse_error", symbol=symbol, error=str(exc))
            return Decimal("0"), Decimal("0")
        return bid, ask

    # ------------------------------------------------------------------
    # Instruments
    # ------------------------------------------------------------------

    async def get_instrument_info(self, category: str, symbol: str) -> InstrumentInfo:
        """Fetch and return parsed InstrumentInfo for a symbol."""
        resp = await self._rest.get_instruments_info(category=category, symbol=symbol)
        items = (resp.get("result") or {}).get("list", [])
        if not items:
            raise ValueError(f"No instrument info found for {symbol}")
        item = items[0]
        item["category"] = category
        return self._mapper.instruments_info_to_model(item)

    # ------------------------------------------------------------------
    # Trading stops
    # ------------------------------------------------------------------

    async def set_trading_stop(
        self,
        category: str,
        symbol: str,
        stop_loss: str | None = None,
        take_profit: str | None = None,
        trailing_stop: str | None = None,
        active_price: str | None = None,
        position_idx: int = 0,
        tpsl_mode: str = "Full",
    ) -> dict[str, Any]:
        """Set TP/SL/trailing-stop controls for an open position."""
        params: dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "positionIdx": position_idx,
            "tpslMode": tpsl_mode,
        }
        if stop_loss is not None:
            params["stopLoss"] = stop_loss
        if take_profit is not None:
            params["takeProfit"] = take_profit
        if trailing_stop is not None:
            params["trailingStop"] = trailing_stop
        if active_price is not None:
            params["activePrice"] = active_price
        return await self._rest.set_trading_stop(
            **params,
        )

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    async def reconcile(self) -> ReconciliationResult:
        """Basic reconciliation pass — compare local idempotency store with exchange."""
        local_pending = self._idempotency.pending_count()

        try:
            open_orders_resp = await self._rest.get_open_orders(category=self._default_category)
            exchange_open = (open_orders_resp.get("result") or {}).get("list", [])
            exchange_ids = {o.get("orderLinkId") for o in exchange_open}

            local_ids = set(self._idempotency.all_states().keys())
            mismatched = list(local_ids - exchange_ids)

            return ReconciliationResult(
                orders_checked=local_pending,
                positions_checked=0,
                discrepancies_found=len(mismatched),
                mismatched_order_ids=mismatched,
                summary=(f"Checked {local_pending} local pending orders; {len(mismatched)} not found on exchange"),
                success=True,
            )
        except Exception as exc:
            logger.error("bybit_adapter.reconcile_error", error=str(exc))
            return ReconciliationResult(
                orders_checked=0,
                success=False,
                summary=f"Reconciliation failed: {exc}",
            )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        """Lightweight health check — pings server time."""
        t0 = time.monotonic()
        try:
            resp = await self._rest.get_server_time()
            latency_ms = (time.monotonic() - t0) * 1000
            ok = resp.get("retCode", -1) == 0
            return {
                "healthy": ok,
                "latency_ms": round(latency_ms, 1),
                "rate_limit_status": self._rate_limiter.get_status(),
            }
        except Exception as exc:
            return {
                "healthy": False,
                "error": str(exc),
                "latency_ms": None,
            }

    async def close(self) -> None:
        """Release resources held by the underlying REST client."""
        await self._rest.close()
