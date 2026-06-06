"""Order state machine — enforces valid lifecycle transitions for orders.

Every order is tracked by an OrderStateMachine instance.
The OrderStateStore holds all machines in memory (asyncio-safe via Lock).
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

from trader.domain.enums import OrderStatus
from trader.domain.errors import TradingSystemError

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class InvalidStateTransitionError(TradingSystemError):
    """Raised when an illegal state transition is attempted."""

    def __init__(
        self,
        order_link_id: str,
        from_status: OrderStatus,
        to_status: OrderStatus,
    ) -> None:
        super().__init__(
            f"Invalid transition {from_status.value} → {to_status.value} for order {order_link_id!r}",
            code="INVALID_STATE_TRANSITION",
            retryable=False,
        )
        self.order_link_id = order_link_id
        self.from_status = from_status
        self.to_status = to_status


# ---------------------------------------------------------------------------
# Valid transitions
# ---------------------------------------------------------------------------

VALID_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.CREATED_LOCAL: {OrderStatus.SUBMITTING, OrderStatus.CANCELLED},
    OrderStatus.SUBMITTING: {
        OrderStatus.REST_ACCEPTED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.REST_ACCEPTED: {
        OrderStatus.WS_CONFIRMED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
        OrderStatus.UNKNOWN_RECONCILIATION_REQUIRED,
    },
    OrderStatus.WS_CONFIRMED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCEL_REQUESTED,
        OrderStatus.CANCELLED,
        OrderStatus.EXPIRED,
    },
    OrderStatus.PARTIALLY_FILLED: {
        OrderStatus.FILLED,
        OrderStatus.CANCEL_REQUESTED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.CANCEL_REQUESTED: {
        OrderStatus.CANCELLED,
        OrderStatus.FILLED,
        OrderStatus.PARTIALLY_FILLED,
    },
    OrderStatus.FILLED: set(),  # terminal
    OrderStatus.CANCELLED: set(),  # terminal
    OrderStatus.REJECTED: set(),  # terminal
    OrderStatus.EXPIRED: set(),  # terminal
    OrderStatus.UNKNOWN_RECONCILIATION_REQUIRED: {
        OrderStatus.WS_CONFIRMED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    },
}

_TERMINAL_STATES: frozenset[OrderStatus] = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    }
)


# ---------------------------------------------------------------------------
# OrderStateMachine
# ---------------------------------------------------------------------------


class OrderStateMachine:
    """Tracks the lifecycle of a single order."""

    def __init__(
        self,
        order_link_id: str,
        initial_status: OrderStatus = OrderStatus.CREATED_LOCAL,
    ) -> None:
        self._order_link_id = order_link_id
        self._current: OrderStatus = initial_status
        self._history: list[tuple[OrderStatus, datetime, str]] = [(initial_status, datetime.now(tz=UTC), "initial")]
        self._entered_at: float = time.monotonic()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def transition(self, new_status: OrderStatus, reason: str = "") -> None:
        """Transition to *new_status*, raising InvalidStateTransitionError on bad path."""
        allowed = VALID_TRANSITIONS.get(self._current, set())
        if new_status not in allowed:
            raise InvalidStateTransitionError(self._order_link_id, self._current, new_status)
        self._current = new_status
        self._entered_at = time.monotonic()
        self._history.append((new_status, datetime.now(tz=UTC), reason))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def current_status(self) -> OrderStatus:
        return self._current

    @property
    def history(self) -> list[tuple[OrderStatus, datetime, str]]:
        return list(self._history)

    @property
    def order_link_id(self) -> str:
        return self._order_link_id

    def is_terminal(self) -> bool:
        return self._current in _TERMINAL_STATES

    def can_transition_to(self, status: OrderStatus) -> bool:
        return status in VALID_TRANSITIONS.get(self._current, set())

    def time_in_current_state(self) -> float:
        """Return seconds spent in the current state."""
        return time.monotonic() - self._entered_at

    def __repr__(self) -> str:
        return f"OrderStateMachine(order_link_id={self._order_link_id!r}, status={self._current.value})"


# ---------------------------------------------------------------------------
# OrderStateStore
# ---------------------------------------------------------------------------


class OrderStateStore:
    """In-memory store of all order state machines.  Thread-safe via asyncio.Lock."""

    def __init__(self) -> None:
        self._machines: dict[str, OrderStateMachine] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        order_link_id: str,
        initial_status: OrderStatus = OrderStatus.CREATED_LOCAL,
    ) -> OrderStateMachine:
        """Create and register a new OrderStateMachine."""
        async with self._lock:
            machine = OrderStateMachine(order_link_id, initial_status)
            self._machines[order_link_id] = machine
            return machine

    async def get(self, order_link_id: str) -> OrderStateMachine | None:
        """Return the machine for *order_link_id*, or None if unknown."""
        async with self._lock:
            return self._machines.get(order_link_id)

    async def transition(self, order_link_id: str, new_status: OrderStatus, reason: str = "") -> None:
        """Apply a state transition to the machine identified by *order_link_id*."""
        async with self._lock:
            machine = self._machines.get(order_link_id)
            if machine is None:
                raise KeyError(f"Unknown orderLinkId: {order_link_id!r}")
            machine.transition(new_status, reason)

    async def get_all_active(self) -> dict[str, OrderStateMachine]:
        """Return all non-terminal machines."""
        async with self._lock:
            return {k: v for k, v in self._machines.items() if not v.is_terminal()}

    async def get_by_status(self, status: OrderStatus) -> list[OrderStateMachine]:
        """Return all machines in the given status."""
        async with self._lock:
            return [m for m in self._machines.values() if m.current_status == status]

    def __len__(self) -> int:
        return len(self._machines)
