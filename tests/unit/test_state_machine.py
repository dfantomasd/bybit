"""Tests for the order state machine (state_machine.py)."""
from __future__ import annotations

import asyncio
import time

import pytest

from trader.domain.enums import OrderStatus
from trader.exchange.state_machine import (
    InvalidStateTransitionError,
    OrderStateStore,
    OrderStateMachine,
    VALID_TRANSITIONS,
)


# ---------------------------------------------------------------------------
# OrderStateMachine — valid transition tests
# ---------------------------------------------------------------------------


def test_created_local_to_submitting():
    """CREATED_LOCAL → SUBMITTING is valid."""
    m = OrderStateMachine("test-001")
    assert m.current_status == OrderStatus.CREATED_LOCAL
    m.transition(OrderStatus.SUBMITTING, "submitting order")
    assert m.current_status == OrderStatus.SUBMITTING


def test_submitting_to_rest_accepted():
    """SUBMITTING → REST_ACCEPTED is valid."""
    m = OrderStateMachine("test-002")
    m.transition(OrderStatus.SUBMITTING)
    m.transition(OrderStatus.REST_ACCEPTED, "exchange confirmed")
    assert m.current_status == OrderStatus.REST_ACCEPTED


def test_rest_accepted_to_ws_confirmed():
    """REST_ACCEPTED → WS_CONFIRMED is valid."""
    m = OrderStateMachine("test-003")
    m.transition(OrderStatus.SUBMITTING)
    m.transition(OrderStatus.REST_ACCEPTED)
    m.transition(OrderStatus.WS_CONFIRMED, "ws confirmed")
    assert m.current_status == OrderStatus.WS_CONFIRMED


def test_ws_confirmed_to_filled():
    """WS_CONFIRMED → FILLED is valid."""
    m = OrderStateMachine("test-004")
    m.transition(OrderStatus.SUBMITTING)
    m.transition(OrderStatus.REST_ACCEPTED)
    m.transition(OrderStatus.WS_CONFIRMED)
    m.transition(OrderStatus.FILLED, "fully filled")
    assert m.current_status == OrderStatus.FILLED


def test_ws_confirmed_to_cancelled():
    """WS_CONFIRMED → CANCELLED is valid."""
    m = OrderStateMachine("test-005")
    m.transition(OrderStatus.SUBMITTING)
    m.transition(OrderStatus.REST_ACCEPTED)
    m.transition(OrderStatus.WS_CONFIRMED)
    m.transition(OrderStatus.CANCELLED, "cancelled by user")
    assert m.current_status == OrderStatus.CANCELLED


def test_partially_filled_to_filled():
    """PARTIALLY_FILLED → FILLED is valid."""
    m = OrderStateMachine("test-006")
    m.transition(OrderStatus.SUBMITTING)
    m.transition(OrderStatus.REST_ACCEPTED)
    m.transition(OrderStatus.PARTIALLY_FILLED)
    m.transition(OrderStatus.FILLED)
    assert m.current_status == OrderStatus.FILLED


def test_filled_is_terminal():
    """FILLED is a terminal state; no further transitions allowed."""
    m = OrderStateMachine("test-007")
    m.transition(OrderStatus.SUBMITTING)
    m.transition(OrderStatus.REST_ACCEPTED)
    m.transition(OrderStatus.WS_CONFIRMED)
    m.transition(OrderStatus.FILLED)
    assert m.is_terminal()
    with pytest.raises(InvalidStateTransitionError):
        m.transition(OrderStatus.CANCELLED)


def test_cancelled_is_terminal():
    """CANCELLED is a terminal state; no further transitions allowed."""
    m = OrderStateMachine("test-008")
    m.transition(OrderStatus.SUBMITTING)
    m.transition(OrderStatus.REST_ACCEPTED)
    m.transition(OrderStatus.WS_CONFIRMED)
    m.transition(OrderStatus.CANCELLED)
    assert m.is_terminal()
    with pytest.raises(InvalidStateTransitionError):
        m.transition(OrderStatus.FILLED)


def test_invalid_transition_raises():
    """Attempting an invalid transition raises InvalidStateTransitionError."""
    m = OrderStateMachine("test-009")
    # Can't go directly from CREATED_LOCAL to FILLED
    with pytest.raises(InvalidStateTransitionError) as exc_info:
        m.transition(OrderStatus.FILLED)
    assert exc_info.value.from_status == OrderStatus.CREATED_LOCAL
    assert exc_info.value.to_status == OrderStatus.FILLED
    assert "test-009" in exc_info.value.order_link_id


def test_cannot_transition_from_terminal():
    """Rejected is terminal; all transitions from it raise."""
    m = OrderStateMachine("test-010", initial_status=OrderStatus.REJECTED)
    assert m.is_terminal()
    with pytest.raises(InvalidStateTransitionError):
        m.transition(OrderStatus.CANCELLED)


def test_history_records_transitions():
    """History list grows with each transition and includes reason."""
    m = OrderStateMachine("test-011")
    m.transition(OrderStatus.SUBMITTING, "test reason")
    m.transition(OrderStatus.REST_ACCEPTED, "exchange 200")

    history = m.history
    assert len(history) == 3  # initial + 2 transitions
    # Check last entry
    last_status, last_dt, last_reason = history[-1]
    assert last_status == OrderStatus.REST_ACCEPTED
    assert last_reason == "exchange 200"


def test_time_in_state():
    """time_in_current_state returns elapsed seconds."""
    m = OrderStateMachine("test-012")
    time.sleep(0.05)
    elapsed = m.time_in_current_state()
    assert elapsed >= 0.04


def test_can_transition_to():
    """can_transition_to returns correct bool based on valid transitions."""
    m = OrderStateMachine("test-013")
    assert m.can_transition_to(OrderStatus.SUBMITTING) is True
    assert m.can_transition_to(OrderStatus.FILLED) is False
    assert m.can_transition_to(OrderStatus.CANCELLED) is True


def test_rejected_is_terminal():
    """REJECTED is terminal."""
    m = OrderStateMachine("test-014", initial_status=OrderStatus.REJECTED)
    assert m.is_terminal()
    assert not m.can_transition_to(OrderStatus.FILLED)


def test_expired_is_terminal():
    """EXPIRED is terminal."""
    m = OrderStateMachine("test-015", initial_status=OrderStatus.EXPIRED)
    assert m.is_terminal()


def test_unknown_reconciliation_transitions():
    """UNKNOWN_RECONCILIATION_REQUIRED can transition to several valid states."""
    m = OrderStateMachine("test-016", initial_status=OrderStatus.UNKNOWN_RECONCILIATION_REQUIRED)
    assert m.can_transition_to(OrderStatus.FILLED)
    assert m.can_transition_to(OrderStatus.CANCELLED)
    assert m.can_transition_to(OrderStatus.REJECTED)
    assert m.can_transition_to(OrderStatus.WS_CONFIRMED)


def test_valid_transitions_dict_complete():
    """VALID_TRANSITIONS covers all OrderStatus values."""
    for status in OrderStatus:
        assert status in VALID_TRANSITIONS, f"{status} not in VALID_TRANSITIONS"


# ---------------------------------------------------------------------------
# OrderStateStore — async tests
# ---------------------------------------------------------------------------


def test_order_store_create():
    """Creating a machine registers it in the store."""
    async def _run():
        store = OrderStateStore()
        m = await store.create("order-A")
        assert m.current_status == OrderStatus.CREATED_LOCAL
        assert len(store) == 1
    asyncio.run(_run())


def test_order_store_get():
    """Getting a machine by ID returns the same instance."""
    async def _run():
        store = OrderStateStore()
        created = await store.create("order-B")
        fetched = await store.get("order-B")
        assert fetched is created
        missing = await store.get("nonexistent")
        assert missing is None
    asyncio.run(_run())


def test_order_store_transition():
    """store.transition updates machine status."""
    async def _run():
        store = OrderStateStore()
        await store.create("order-C")
        await store.transition("order-C", OrderStatus.SUBMITTING, "test")
        m = await store.get("order-C")
        assert m.current_status == OrderStatus.SUBMITTING
    asyncio.run(_run())


def test_order_store_get_active_excludes_terminal():
    """get_all_active excludes terminal-state machines."""
    async def _run():
        store = OrderStateStore()
        await store.create("active-1")
        m_done = await store.create("done-1")
        # Manually set terminal
        m_done.transition(OrderStatus.SUBMITTING)
        m_done.transition(OrderStatus.REST_ACCEPTED)
        m_done.transition(OrderStatus.WS_CONFIRMED)
        m_done.transition(OrderStatus.FILLED)

        active = await store.get_all_active()
        assert "active-1" in active
        assert "done-1" not in active
    asyncio.run(_run())


def test_order_store_get_by_status():
    """get_by_status returns only machines with matching status."""
    async def _run():
        store = OrderStateStore()
        await store.create("s-1")
        await store.create("s-2")
        await store.transition("s-1", OrderStatus.SUBMITTING)

        submitting = await store.get_by_status(OrderStatus.SUBMITTING)
        created_local = await store.get_by_status(OrderStatus.CREATED_LOCAL)

        assert len(submitting) == 1
        assert submitting[0].order_link_id == "s-1"
        assert len(created_local) == 1
    asyncio.run(_run())


def test_order_store_transition_unknown_raises():
    """Transitioning unknown order_link_id raises KeyError."""
    async def _run():
        store = OrderStateStore()
        with pytest.raises(KeyError):
            await store.transition("nonexistent", OrderStatus.SUBMITTING)
    asyncio.run(_run())
