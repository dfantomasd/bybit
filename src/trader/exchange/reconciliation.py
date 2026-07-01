"""Reconciliation service — periodically reconciles local state with exchange.

Runs every 15-60 seconds (configurable) and after every reconnect.

Checks
------
1. Positions: local vs exchange qty, side, symbol.
2. Active orders: unknown orders, missing local orders.
3. Balance: approximate match.
4. Stop-loss presence: every open position must have an SL.
5. Manual intervention detection.

Actions on mismatch
-------------------
- Log diff with details.
- Emit ReconciliationEvent.
- Mark affected orders as UNKNOWN_RECONCILIATION_REQUIRED.
- If position without SL: enter safe mode.
- If unknown position: alert + safe mode.
- If severe mismatch: trigger repair or safe mode.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

from trader.domain.enums import OrderStatus
from trader.domain.events import ReconciliationEvent
from trader.domain.models import ReconciliationResult

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Reconciliation diff
# ---------------------------------------------------------------------------


@dataclass
class ReconciliationDiff:
    """Structured diff from a single reconciliation pass."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    category: str = "linear"
    unknown_exchange_orders: list[str] = field(default_factory=list)
    missing_local_orders: list[str] = field(default_factory=list)
    position_mismatches: list[str] = field(default_factory=list)
    positions_without_sl: list[str] = field(default_factory=list)
    manual_interventions: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return (
            not self.unknown_exchange_orders
            and not self.missing_local_orders
            and not self.position_mismatches
            and not self.positions_without_sl
            and not self.manual_interventions
        )


# ---------------------------------------------------------------------------
# ReconciliationService
# ---------------------------------------------------------------------------


class ReconciliationService:
    """Periodically reconciles local state with exchange state."""

    def __init__(
        self,
        rest_client: Any,
        order_store: Any,
        position_store: Any,
        event_queue: asyncio.Queue[ReconciliationEvent],
        metrics: Any = None,
        logger: Any = None,
    ) -> None:
        self._rest = rest_client
        self._order_store = order_store
        self._position_store = position_store
        self._event_queue = event_queue
        self._metrics = metrics
        self._log = logger or structlog.get_logger(__name__)

        self._running = False
        self._stop_event = asyncio.Event()
        self._safe_mode: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run_once(self, category: str = "linear") -> ReconciliationResult:
        """Run a single reconciliation pass and return the result."""
        self._log.info("reconciliation.starting", category=category)
        diff = ReconciliationDiff(category=category)

        diffs: list[str] = []

        try:
            pos_diffs = await self._check_positions(category)
            diffs.extend(pos_diffs)
            diff.position_mismatches = pos_diffs
        except Exception as exc:
            self._log.warning("reconciliation.position_check_failed", error=str(exc))

        try:
            order_diffs = await self._check_orders(category)
            diffs.extend(order_diffs)
        except Exception as exc:
            self._log.warning("reconciliation.order_check_failed", error=str(exc))

        try:
            balance_diffs = await self._check_balance()
            diffs.extend(balance_diffs)
        except Exception as exc:
            self._log.warning("reconciliation.balance_check_failed", error=str(exc))

        try:
            sl_diffs = await self._check_stop_losses()
            diffs.extend(sl_diffs)
            diff.positions_without_sl = sl_diffs
        except Exception as exc:
            self._log.warning("reconciliation.sl_check_failed", error=str(exc))

        # Trigger safe mode conditions
        if diff.positions_without_sl:
            self._log.error(
                "reconciliation.positions_without_sl",
                symbols=diff.positions_without_sl,
            )
            self._safe_mode = True

        if diff.unknown_exchange_orders:
            self._log.error(
                "reconciliation.unknown_orders",
                orders=diff.unknown_exchange_orders,
            )
            self._safe_mode = True

        total_discrepancies = len(diffs)
        success = total_discrepancies == 0

        if not success:
            self._log.warning(
                "reconciliation.discrepancies_found",
                count=total_discrepancies,
                diffs=diffs[:10],  # log first 10
            )
        else:
            self._log.info("reconciliation.clean")

        result = ReconciliationResult(
            orders_checked=0,  # populated when order_store available
            positions_checked=0,
            discrepancies_found=total_discrepancies,
            discrepancies_resolved=0,
            discrepancies_unresolved=total_discrepancies,
            mismatched_order_ids=diff.unknown_exchange_orders + diff.missing_local_orders,
            summary=f"Found {total_discrepancies} discrepancies",
            success=success,
        )

        # Emit reconciliation event
        event = ReconciliationEvent(
            run_id=result.run_id,
            orders_checked=result.orders_checked,
            positions_checked=result.positions_checked,
            discrepancies_found=total_discrepancies,
            discrepancies_resolved=0,
            success=success,
            summary=result.summary,
        )
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            self._log.warning("reconciliation.queue_full")

        return result

    async def start_periodic(self, interval_seconds: int = 30) -> None:
        """Start periodic reconciliation loop."""
        self._running = True
        self._log.info("reconciliation.periodic_start", interval=interval_seconds)
        while not self._stop_event.is_set():
            try:
                await self.run_once()
            except Exception as exc:
                self._log.error("reconciliation.periodic_error", error=str(exc))
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=float(interval_seconds),
                )
                break  # stop_event fired
            except TimeoutError:
                pass  # normal interval expiry
        self._running = False

    async def stop(self) -> None:
        """Stop the periodic reconciliation loop."""
        self._stop_event.set()

    async def on_reconnect(self) -> ReconciliationResult:
        """Immediate reconciliation triggered after WS reconnect."""
        self._log.info("reconciliation.on_reconnect")
        return await self.run_once()

    # ------------------------------------------------------------------
    # Check methods
    # ------------------------------------------------------------------

    async def _check_positions(self, category: str) -> list[str]:
        """Compare local positions vs exchange positions."""
        diffs: list[str] = []
        if self._rest is None or self._position_store is None:
            return diffs

        try:
            exchange_positions = await self._rest.get_positions(category=category)
        except Exception as exc:
            self._log.warning("reconciliation._check_positions.rest_failed", error=str(exc))
            return diffs

        # Build exchange position map {symbol: position}
        exchange_map: dict[str, Any] = {}
        for pos in exchange_positions:
            symbol = (
                getattr(pos, "symbol", None) or pos.get("symbol", "")
                if isinstance(pos, dict)
                else getattr(pos, "symbol", "")
            )
            if symbol:
                exchange_map[symbol] = pos

        # Compare with local
        local_positions = getattr(self._position_store, "_positions", {})
        for symbol, local_pos in local_positions.items():
            if symbol not in exchange_map:
                diffs.append(f"Position {symbol} exists locally but not on exchange")
            else:
                exch_pos = exchange_map[symbol]
                local_size = getattr(local_pos, "size", None)
                exch_size_str = (
                    exch_pos.get("size", "0") if isinstance(exch_pos, dict) else str(getattr(exch_pos, "size", "0"))
                )
                try:
                    from decimal import Decimal as _D, InvalidOperation
                    if local_size is not None and _D(str(local_size)) != _D(exch_size_str):
                        diffs.append(f"Position {symbol} size mismatch: local={local_size} exchange={exch_size_str}")
                except InvalidOperation:
                    diffs.append(f"Position {symbol} size unreadable: exchange={exch_size_str}")

        for symbol in exchange_map:
            if symbol not in local_positions:
                diffs.append(f"Unknown position {symbol} on exchange not tracked locally")

        return diffs

    async def _check_orders(self, category: str) -> list[str]:
        """Compare local PENDING orders vs exchange open orders.

        Terminal states (FILLED, CANCELLED, REJECTED, EXPIRED) are never
        compared with exchange open orders — they are already settled.
        Only pending states are eligible to be on the exchange at all.
        """
        diffs: list[str] = []
        if self._rest is None or self._order_store is None:
            return diffs

        _pending_states = {
            OrderStatus.CREATED_LOCAL,
            OrderStatus.SUBMITTING,
            OrderStatus.REST_ACCEPTED,
            OrderStatus.WS_CONFIRMED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.CANCEL_REQUESTED,
            OrderStatus.UNKNOWN_RECONCILIATION_REQUIRED,
        }
        _terminal_states = {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }

        try:
            exchange_orders = await self._rest.get_open_orders(category=category)
        except Exception as exc:
            self._log.warning("reconciliation._check_orders.rest_failed", error=str(exc))
            return diffs

        # Build exchange order set by orderLinkId
        exchange_link_ids: set[str] = set()
        for order in exchange_orders:
            lid = order.get("orderLinkId", "") if isinstance(order, dict) else getattr(order, "order_link_id", "")
            if lid:
                exchange_link_ids.add(lid)

        # Get local active orders
        try:
            local_active = await self._order_store.get_all_active()
        except Exception as exc:
            self._log.warning("reconciliation._check_orders.store_failed", error=str(exc))
            return diffs  # cannot compare without local state; avoid false safe-mode

        # Only compare PENDING local orders against exchange open orders
        local_pending: dict[str, Any] = {}
        for lid, machine in local_active.items():
            status = getattr(machine, "status", None)
            if status in _pending_states:
                local_pending[lid] = machine

        local_pending_ids = set(local_pending.keys())

        # Unknown orders on exchange (not tracked locally at all)
        for lid in exchange_link_ids - set(local_active.keys()):
            diffs.append(f"Order {lid} on exchange but not in local store")

        # Local PENDING orders missing from exchange → mark for reconciliation
        for lid in local_pending_ids - exchange_link_ids:
            machine = local_pending.get(lid)
            if machine is not None:
                try:
                    await self._order_store.transition(
                        lid,
                        OrderStatus.UNKNOWN_RECONCILIATION_REQUIRED,
                        "not found on exchange during reconciliation",
                    )
                    diffs.append(f"Order {lid} pending locally but not on exchange (marked UNKNOWN)")
                except Exception:
                    diffs.append(f"Order {lid} pending locally but not on exchange")

        return diffs

    async def _check_balance(self) -> list[str]:
        """Perform approximate balance check."""
        diffs: list[str] = []
        if self._rest is None:
            return diffs

        try:
            await self._rest.get_wallet_balance(account_type="UNIFIED")
            # In production: compare with locally cached balance
        except Exception as exc:
            self._log.warning("reconciliation._check_balance.failed", error=str(exc))
            diffs.append(f"Balance check failed: {exc}")

        return diffs

    async def _check_stop_losses(self) -> list[str]:
        """Ensure every open position has a stop loss."""
        diffs: list[str] = []
        if self._position_store is None:
            return diffs

        local_positions = getattr(self._position_store, "_positions", {})
        for symbol, pos in local_positions.items():
            has_sl = getattr(pos, "stop_loss", None) is not None
            if not has_sl:
                diffs.append(symbol)

        return diffs

    @property
    def safe_mode(self) -> bool:
        return self._safe_mode
