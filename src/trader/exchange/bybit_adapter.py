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

from trader.domain.enums import OrderStatus
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
        trade_journal: Any = None,
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
        self._journal = trade_journal

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

        Durable state lifecycle:
          CREATED_LOCAL → SUBMITTING  (written before REST call)
          REST_ACCEPTED               (written on REST success)
          UNKNOWN_RECONCILIATION_REQUIRED (written on any exception — ambiguous)

        Returns the raw Bybit response dict.
        """
        # Check duplicate
        if await self._idempotency.check_duplicate(intent.order_link_id):
            raise ValueError(f"Duplicate order detected: {intent.order_link_id}")

        # Write CREATED_LOCAL to durable state before touching the exchange
        if self._journal is not None:
            try:
                await self._journal.upsert_durable_order_state(
                    order_link_id=intent.order_link_id,
                    symbol=intent.symbol,
                    side=intent.side.value,
                    qty=intent.qty,
                    state="CREATED_LOCAL",
                    proposal_id=intent.proposal_id,
                    decision_id=intent.decision_id,
                )
            except Exception as _j_exc:
                logger.debug("bybit_adapter.durable_created_local_failed", error=str(_j_exc))

        # Register in idempotency store
        await self._idempotency.register_intent(intent)
        await self._idempotency.mark_submitted(intent.order_link_id)

        # Write SUBMITTING to durable state before REST
        if self._journal is not None:
            try:
                await self._journal.upsert_durable_order_state(
                    order_link_id=intent.order_link_id,
                    symbol=intent.symbol,
                    side=intent.side.value,
                    qty=intent.qty,
                    state="SUBMITTING",
                )
            except Exception as _j_exc:
                logger.debug("bybit_adapter.durable_submitting_failed", error=str(_j_exc))

        # Map to params
        params = self._mapper.intent_to_params(intent, self._default_category)

        try:
            resp = await self._rest.place_order(**params)
            exchange_id = (resp.get("result") or {}).get("orderId", "")
            await self._idempotency.mark_confirmed(intent.order_link_id, exchange_id)

            # Write REST_ACCEPTED after confirmed response
            if self._journal is not None:
                try:
                    await self._journal.upsert_durable_order_state(
                        order_link_id=intent.order_link_id,
                        symbol=intent.symbol,
                        side=intent.side.value,
                        qty=intent.qty,
                        state="REST_ACCEPTED",
                        exchange_order_id=exchange_id,
                    )
                except Exception as _j_exc:
                    logger.debug("bybit_adapter.durable_rest_accepted_failed", error=str(_j_exc))

            logger.info(
                "bybit_adapter.order_placed",
                order_link_id=intent.order_link_id,
                exchange_order_id=exchange_id,
                symbol=intent.symbol,
            )
            return resp
        except Exception as exc:
            # Any exception (timeout, network, API error) is ambiguous — mark UNKNOWN.
            # Blind retry is forbidden; reconciliation must resolve this state.
            if self._journal is not None:
                try:
                    await self._journal.upsert_durable_order_state(
                        order_link_id=intent.order_link_id,
                        symbol=intent.symbol,
                        side=intent.side.value,
                        qty=intent.qty,
                        state="UNKNOWN_RECONCILIATION_REQUIRED",
                        last_error=str(exc)[:200],
                    )
                except Exception as _j_exc:
                    logger.debug("bybit_adapter.durable_unknown_failed", error=str(_j_exc))
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

    async def get_conservative_market_price(
        self,
        category: str,
        symbol: str,
        side: str,
    ) -> Decimal:
        """Return a conservative market price for pre-order notional checks.

        Delegates to the REST client (ask1Price for buys, bid1Price for sells).
        Raises TradingSystemError if ticker data is unavailable.
        """
        return await self._rest.get_conservative_market_price(
            category=category,
            symbol=symbol,
            side=side,
        )

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
        """Reconcile only PENDING in-memory states against exchange open orders.

        Terminal states (FILLED, CANCELLED, REJECTED, EXPIRED) are settled and
        never compared against exchange open orders — comparing them would always
        produce false mismatches.  Only pending states can legitimately appear on
        the exchange at all.
        """
        _pending_states = {
            OrderStatus.CREATED_LOCAL,
            OrderStatus.SUBMITTING,
            OrderStatus.REST_ACCEPTED,
            OrderStatus.WS_CONFIRMED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.CANCEL_REQUESTED,
            OrderStatus.UNKNOWN_RECONCILIATION_REQUIRED,
        }

        try:
            open_orders_resp = await self._rest.get_open_orders(category=self._default_category)
            exchange_open = (open_orders_resp.get("result") or {}).get("list", [])
            exchange_ids = {o.get("orderLinkId") for o in exchange_open}

            all_states = self._idempotency.all_states()
            pending_ids = {lid for lid, status_str in all_states.items() if OrderStatus(status_str) in _pending_states}

            mismatched = [lid for lid in pending_ids if lid not in exchange_ids]

            return ReconciliationResult(
                orders_checked=len(pending_ids),
                positions_checked=0,
                discrepancies_found=len(mismatched),
                mismatched_order_ids=mismatched,
                summary=(f"Checked {len(pending_ids)} pending orders; {len(mismatched)} not found on exchange"),
                success=True,
            )
        except Exception as exc:
            logger.error("bybit_adapter.reconcile_error", error=str(exc))
            return ReconciliationResult(
                orders_checked=0,
                success=False,
                summary=f"Reconciliation failed: {exc}",
            )

    async def handle_order_update(self, event: Any) -> bool:
        """Handle an OrderUpdateEvent: sync both idempotency and durable state.

        Updates the in-memory idempotency store and persists to durable order state.
        Returns True if the new status is terminal (caller should release pending count
        exactly once via a guard set).

        P0: Never use exchange orderId directly as pending-ID.
        Use order_link_id if present, otherwise reverse-lookup via exchange_order_id.
        """
        _terminal_states = {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }

        order_link_id = getattr(event, "order_link_id", None)
        exchange_order_id = getattr(event, "order_id", None)
        order_status: OrderStatus | None = getattr(event, "status", None) or getattr(event, "order_status", None)

        if order_status is None:
            return False

        # P0: Reverse lookup if order_link_id is missing
        if order_link_id is None and exchange_order_id:
            if self._journal is not None:
                order_link_id = await self._journal.find_order_link_id_by_exchange_order_id(exchange_order_id)
            # If lookup fails, we still process the event but can't tie it to a pending slot

        # If we still don't have an order_link_id, we can't update idempotency state
        # but we can still log/persist to durable state if journal is available
        has_order_link_id = order_link_id is not None

        is_terminal = order_status in _terminal_states

        # Update in-memory idempotency (only if we have order_link_id)
        if has_order_link_id:
            try:
                current = await self._idempotency.get_state(order_link_id)
                if current is not None and current not in _terminal_states:
                    if order_status == OrderStatus.FILLED:
                        await self._idempotency.mark_filled(order_link_id)
                    elif order_status == OrderStatus.CANCELLED:
                        await self._idempotency.mark_cancelled(order_link_id)
                    elif order_status == OrderStatus.WS_CONFIRMED and current in {
                        OrderStatus.REST_ACCEPTED,
                        OrderStatus.SUBMITTING,
                    }:
                        self._idempotency._store[order_link_id]["status"] = OrderStatus.WS_CONFIRMED
                    elif order_status == OrderStatus.PARTIALLY_FILLED and current in {
                        OrderStatus.WS_CONFIRMED,
                        OrderStatus.REST_ACCEPTED,
                    }:
                        self._idempotency._store[order_link_id]["status"] = OrderStatus.PARTIALLY_FILLED
            except Exception as exc:
                logger.debug("handle_order_update.idempotency_update_failed", error=str(exc))

        # Persist to durable state — use order_link_id if available, otherwise generate fallback
        durable_order_link_id = order_link_id or (f"unknown:{exchange_order_id}" if exchange_order_id else "unknown:no_exchange_id")
        if self._journal is not None:
            try:
                await self._journal.upsert_durable_order_state(
                    order_link_id=durable_order_link_id,
                    symbol=getattr(event, "symbol", ""),
                    side=event.side.value if getattr(event, "side", None) else "unknown",
                    qty=getattr(event, "qty", Decimal("0")),
                    state=order_status.value,
                    exchange_order_id=exchange_order_id,
                )
            except Exception as exc:
                logger.debug("handle_order_update.durable_write_failed", error=str(exc))

        return is_terminal

    async def load_pending_from_db(self) -> int:
        """Restore pending orders from PostgreSQL into the in-memory idempotency store.

        Called once at startup to recover orders that were in-flight before restart.
        Returns the count of orders loaded.
        """
        if self._journal is None:
            return 0
        _terminal_states = {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }
        try:
            pending_rows = await self._journal.get_pending_durable_orders()
            loaded = 0
            for row in pending_rows:
                lid = str(row["order_link_id"])
                if lid in self._idempotency._store:
                    continue
                status_str = str(row.get("state", "CREATED_LOCAL"))
                try:
                    status = OrderStatus(status_str)
                except ValueError:
                    status = OrderStatus.UNKNOWN_RECONCILIATION_REQUIRED
                if status in _terminal_states:
                    continue
                self._idempotency._store[lid] = {
                    "status": status,
                    "exchange_order_id": row.get("exchange_order_id"),
                    "intent": None,
                }
                loaded += 1
            if loaded:
                logger.info("bybit_adapter.pending_restored_from_db", count=loaded)
            return loaded
        except Exception as exc:
            logger.warning("bybit_adapter.load_pending_failed", error=str(exc))
            return 0

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
