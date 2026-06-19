"""Order idempotency manager — prevents duplicate orders.

Tracks orderLinkId → state mapping in memory.
Before submit: checks local log; can optionally check WS state or REST.

orderLinkId format: {env_short}-{date}-{strategy_id[:4]}-{proposal_id[:8]}-{random_hex_6}
Max 36 chars (Bybit limit).
"""

from __future__ import annotations

import secrets
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, cast

import structlog

from trader.domain.enums import OrderStatus
from trader.domain.errors import OrderRejectedError
from trader.domain.models import OrderIntent

logger = structlog.get_logger(__name__)

# Valid state transitions (from → allowed next states)
_VALID_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.CREATED_LOCAL: {OrderStatus.SUBMITTING, OrderStatus.CANCELLED},
    OrderStatus.SUBMITTING: {OrderStatus.REST_ACCEPTED, OrderStatus.REJECTED, OrderStatus.CANCELLED},
    OrderStatus.REST_ACCEPTED: {OrderStatus.WS_CONFIRMED, OrderStatus.CANCELLED, OrderStatus.REJECTED},
    OrderStatus.WS_CONFIRMED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCEL_REQUESTED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.PARTIALLY_FILLED: {
        OrderStatus.FILLED,
        OrderStatus.CANCEL_REQUESTED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.FILLED: set(),
    OrderStatus.CANCEL_REQUESTED: {OrderStatus.CANCELLED},
    OrderStatus.CANCELLED: set(),
    OrderStatus.REJECTED: set(),
    OrderStatus.EXPIRED: set(),
    OrderStatus.UNKNOWN_RECONCILIATION_REQUIRED: {
        OrderStatus.WS_CONFIRMED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
    },
}

_TERMINAL_STATES = {
    OrderStatus.FILLED,
    OrderStatus.CANCELLED,
    OrderStatus.REJECTED,
    OrderStatus.EXPIRED,
}
_DEFAULT_MAX_TERMINAL_RETAINED = 5_000

# Env-short labels for order link ID generation
_ENV_SHORT: dict[str, str] = {
    "LIVE": "LV",
    "CANARY_LIVE": "CL",
    "TESTNET": "TN",
    "SHADOW": "SH",
}


def _env_short(env: str) -> str:
    return _ENV_SHORT.get(env.upper(), env[:2].upper())


class IdempotencyManager:
    """In-memory idempotency store for order lifecycle tracking.

    Thread-safety note: this implementation is designed for use in a single-
    async-event-loop context.  For multi-process deployments, back this with
    Redis (Phase 3).
    """

    def __init__(self, max_terminal_retained: int = _DEFAULT_MAX_TERMINAL_RETAINED) -> None:
        # order_link_id → {"status": OrderStatus, "exchange_id": str | None, ...}
        self._store: dict[str, dict[str, Any]] = {}
        self._max_terminal_retained = max(0, int(max_terminal_retained))

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    def generate_order_link_id(
        self,
        env: str,
        strategy_id: str,
        proposal_id: str,
    ) -> str:
        """Generate a unique, ≤36-char orderLinkId.

        Format: {env_short}-{YYMMDD}-{strat[:4]}-{prop[:8]}-{hex6}

        Example: TN-260605-MOMO-12AB34CD-a1b2c3   (36 chars)
        """
        env_part = _env_short(env)  # 2 chars
        date_part = datetime.now(tz=UTC).strftime("%y%m%d")  # 6 chars
        strat_part = strategy_id[:4].upper().replace("-", "").replace("_", "")  # ≤4 chars
        prop_part = proposal_id.replace("-", "")[:8].upper()  # ≤8 chars
        rand_part = secrets.token_hex(3)  # 6 chars

        # Assemble: E-DDDDDD-SSSS-PPPPPPPP-RRRRRR → max 2+1+6+1+4+1+8+1+6 = 30 chars
        order_link_id = f"{env_part}-{date_part}-{strat_part}-{prop_part}-{rand_part}"

        # Safety: truncate to 36 chars if somehow longer
        if len(order_link_id) > 36:
            order_link_id = order_link_id[:36]

        logger.debug(
            "idempotency.generated_id",
            order_link_id=order_link_id,
            length=len(order_link_id),
        )
        return order_link_id

    # ------------------------------------------------------------------
    # State checks
    # ------------------------------------------------------------------

    async def check_duplicate(self, order_link_id: str) -> bool:
        """Return True if an order with this ID already exists in the local log."""
        exists = order_link_id in self._store
        if exists:
            state = self._store[order_link_id]
            logger.warning(
                "idempotency.duplicate_detected",
                order_link_id=order_link_id,
                current_status=state["status"].value,
            )
        return exists

    async def get_state(self, order_link_id: str) -> OrderStatus | None:
        """Return the current OrderStatus for the given ID, or None if unknown."""
        entry = self._store.get(order_link_id)
        if entry is None:
            return None
        return cast(OrderStatus, entry["status"])

    # ------------------------------------------------------------------
    # State mutations
    # ------------------------------------------------------------------

    async def register_intent(self, intent: OrderIntent) -> None:
        """Record a new OrderIntent in CREATED_LOCAL state.

        Raises OrderRejectedError if the ID already exists.
        """
        order_link_id = intent.order_link_id
        if order_link_id in self._store:
            existing = self._store[order_link_id]
            raise OrderRejectedError(
                f"orderLinkId {order_link_id!r} already registered with status {existing['status'].value}",
                order_link_id=order_link_id,
            )

        self._store[order_link_id] = {
            "status": OrderStatus.CREATED_LOCAL,
            "exchange_order_id": None,
            "intent": intent,
            "created_at": datetime.now(tz=UTC),
            "terminal_at": None,
        }
        logger.info(
            "idempotency.intent_registered",
            order_link_id=order_link_id,
            symbol=intent.symbol,
        )

    def _transition(self, order_link_id: str, new_status: OrderStatus) -> None:
        """Apply a state transition, enforcing valid paths."""
        if order_link_id not in self._store:
            raise KeyError(f"Unknown orderLinkId: {order_link_id!r}")
        current = self._store[order_link_id]["status"]
        allowed = _VALID_TRANSITIONS.get(current, set())
        if new_status not in allowed:
            raise OrderRejectedError(
                f"Invalid transition {current.value} → {new_status.value} for orderLinkId={order_link_id!r}",
                order_link_id=order_link_id,
            )
        self._store[order_link_id]["status"] = new_status
        if new_status in _TERMINAL_STATES:
            self._store[order_link_id]["terminal_at"] = datetime.now(tz=UTC)
            self._prune_terminal_orders()
        logger.debug(
            "idempotency.state_transition",
            order_link_id=order_link_id,
            from_status=current.value,
            to_status=new_status.value,
        )

    async def mark_submitted(self, order_link_id: str) -> None:
        """Mark order as in-flight (SUBMITTING)."""
        self._transition(order_link_id, OrderStatus.SUBMITTING)

    async def mark_confirmed(self, order_link_id: str, exchange_order_id: str) -> None:
        """Mark order as REST-accepted and record exchange order ID."""
        self._transition(order_link_id, OrderStatus.REST_ACCEPTED)
        self._store[order_link_id]["exchange_order_id"] = exchange_order_id

    async def mark_filled(self, order_link_id: str) -> None:
        """Mark order as fully filled."""
        current = self._store.get(order_link_id, {}).get("status")
        if current == OrderStatus.PARTIALLY_FILLED:
            self._transition(order_link_id, OrderStatus.FILLED)
        elif current == OrderStatus.WS_CONFIRMED:
            self._transition(order_link_id, OrderStatus.FILLED)
        elif current == OrderStatus.REST_ACCEPTED:
            # Fast-fill: jump straight to FILLED via WS_CONFIRMED intermediate
            self._store[order_link_id]["status"] = OrderStatus.WS_CONFIRMED
            self._transition(order_link_id, OrderStatus.FILLED)
        else:
            self._transition(order_link_id, OrderStatus.FILLED)

    async def mark_cancelled(self, order_link_id: str) -> None:
        """Mark order as cancelled."""
        current = self._store.get(order_link_id, {}).get("status")
        if current in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
            logger.warning(
                "idempotency.cancel_in_terminal_state",
                order_link_id=order_link_id,
                status=current.value if current else "None",
            )
            return
        # Move to CANCEL_REQUESTED first if needed
        if current == OrderStatus.WS_CONFIRMED or current == OrderStatus.PARTIALLY_FILLED:
            self._store[order_link_id]["status"] = OrderStatus.CANCEL_REQUESTED
        self._transition(order_link_id, OrderStatus.CANCELLED)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def all_states(self) -> dict[str, str]:
        """Return a snapshot of all tracked IDs and their statuses."""
        return {k: v["status"].value for k, v in self._store.items()}

    def pending_count(self) -> int:
        """Count orders not yet in a terminal state."""
        return sum(1 for v in self._store.values() if v["status"] not in _TERMINAL_STATES)

    def _terminal_order_ids_oldest_first(self) -> Iterable[str]:
        terminal_items = [
            (order_link_id, entry.get("terminal_at") or entry.get("created_at") or datetime.min.replace(tzinfo=UTC))
            for order_link_id, entry in self._store.items()
            if entry.get("status") in _TERMINAL_STATES
        ]
        return (order_link_id for order_link_id, _ in sorted(terminal_items, key=lambda item: item[1]))

    def _prune_terminal_orders(self) -> None:
        """Bound memory use by retaining only the newest terminal orders."""
        if self._max_terminal_retained <= 0:
            terminal_ids = list(self._terminal_order_ids_oldest_first())
        else:
            terminal_ids = list(self._terminal_order_ids_oldest_first())
            overflow = len(terminal_ids) - self._max_terminal_retained
            if overflow <= 0:
                return
            terminal_ids = terminal_ids[:overflow]
        for order_link_id in terminal_ids:
            self._store.pop(order_link_id, None)
