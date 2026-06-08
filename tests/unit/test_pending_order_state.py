"""Tests for pending-order state invariant in ExecutionEngine.

Covers every scenario from the task specification:
  - duplicate mark_entry_submitted (idempotent)
  - duplicate terminal event (idempotent)
  - mark_entry_resolved with unknown ID
  - mark_entry_resolved with empty ID at 0 / 1 / 2 pending
  - restore after restart
  - restore with duplicates
  - restore with empty IDs
  - Cancelled / Rejected / Expired release the correct slot
  - Filled + ExecutionUpdateEvent does not double-decrement
  - API unavailable during reconcile preserves pending
  - stale pending cleared only on confirmed exchange miss
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.execution.engine import ExecutionEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _engine(shadow: bool = False) -> ExecutionEngine:
    """Minimal ExecutionEngine with stubs — no DB, no adapter needed for state tests."""
    adapter = MagicMock()
    adapter.get_open_orders = AsyncMock(return_value=[])
    risk_manager = MagicMock()
    exposure = MagicMock()
    exposure.total_exposure_pct = 0

    trade_journal = MagicMock()
    trade_journal.is_enabled = True
    trade_journal.get_pending_order_events = AsyncMock(return_value=[])
    trade_journal.get_pending_durable_orders = AsyncMock(return_value=[])
    trade_journal.mark_order_event_stale = AsyncMock()
    trade_journal.mark_durable_order_stale = AsyncMock()

    return ExecutionEngine(
        adapter=adapter,
        risk_manager=risk_manager,
        exposure_tracker=exposure,
        shadow_mode=shadow,
        trade_journal=trade_journal,
    )


def _assert_invariant(engine: ExecutionEngine) -> None:
    """_pending_entry_count must always equal len(_pending_entry_order_link_ids)."""
    assert engine._pending_entry_count == len(engine._pending_entry_order_link_ids), (
        f"Invariant violated: count={engine._pending_entry_count} "
        f"!= set size={len(engine._pending_entry_order_link_ids)} "
        f"ids={engine._pending_entry_order_link_ids}"
    )


# ---------------------------------------------------------------------------
# mark_entry_submitted
# ---------------------------------------------------------------------------


def test_submit_adds_id_and_syncs_count() -> None:
    eng = _engine(shadow=False)
    eng.mark_entry_submitted("order-1", symbol="BTCUSDT")
    assert "order-1" in eng._pending_entry_order_link_ids
    assert eng._pending_entry_symbols["order-1"] == "BTCUSDT"
    _assert_invariant(eng)


def test_submit_duplicate_id_does_not_double_count() -> None:
    eng = _engine(shadow=False)
    eng.mark_entry_submitted("order-1", symbol="BTCUSDT")
    eng.mark_entry_submitted("order-1", symbol="BTCUSDT")  # duplicate
    assert eng._pending_entry_count == 1
    assert len(eng._pending_entry_order_link_ids) == 1
    _assert_invariant(eng)


def test_submit_empty_id_live_mode_does_not_increment() -> None:
    eng = _engine(shadow=False)
    eng.mark_entry_submitted("", symbol="BTCUSDT")
    assert eng._pending_entry_count == 0
    assert len(eng._pending_entry_order_link_ids) == 0
    _assert_invariant(eng)


def test_submit_multiple_distinct_ids() -> None:
    eng = _engine(shadow=False)
    eng.mark_entry_submitted("order-1")
    eng.mark_entry_submitted("order-2")
    eng.mark_entry_submitted("order-3")
    assert eng._pending_entry_count == 3
    _assert_invariant(eng)


# ---------------------------------------------------------------------------
# mark_entry_resolved
# ---------------------------------------------------------------------------


def test_resolve_known_id_removes_slot() -> None:
    eng = _engine(shadow=False)
    eng.mark_entry_submitted("order-1")
    eng.mark_entry_resolved("order-1")
    assert "order-1" not in eng._pending_entry_order_link_ids
    assert eng._pending_entry_count == 0
    _assert_invariant(eng)


def test_resolve_known_id_is_idempotent() -> None:
    """Calling resolve twice for the same ID must not go negative."""
    eng = _engine(shadow=False)
    eng.mark_entry_submitted("order-1")
    eng.mark_entry_resolved("order-1")
    eng.mark_entry_resolved("order-1")  # second call — already resolved
    assert eng._pending_entry_count == 0
    _assert_invariant(eng)


def test_resolve_unknown_id_does_not_change_count() -> None:
    eng = _engine(shadow=False)
    eng.mark_entry_submitted("order-1")
    eng.mark_entry_resolved("order-unknown")
    # order-1 still pending; count must not drop
    assert eng._pending_entry_count == 1
    assert "order-1" in eng._pending_entry_order_link_ids
    _assert_invariant(eng)


def test_resolve_empty_id_zero_pending_is_noop() -> None:
    eng = _engine(shadow=False)
    eng.mark_entry_resolved("")
    assert eng._pending_entry_count == 0
    _assert_invariant(eng)


def test_resolve_empty_id_one_pending_fallback() -> None:
    """Safe backwards-compat: with exactly 1 pending, empty ID resolves it."""
    eng = _engine(shadow=False)
    eng.mark_entry_submitted("order-1")
    eng.mark_entry_resolved("")
    assert eng._pending_entry_count == 0
    assert "order-1" not in eng._pending_entry_order_link_ids
    _assert_invariant(eng)


def test_resolve_empty_id_two_pending_stays_failclosed() -> None:
    """With multiple pending entries, empty ID must NOT release anything."""
    eng = _engine(shadow=False)
    eng.mark_entry_submitted("order-1")
    eng.mark_entry_submitted("order-2")
    eng.mark_entry_resolved("")
    assert eng._pending_entry_count == 2
    assert "order-1" in eng._pending_entry_order_link_ids
    assert "order-2" in eng._pending_entry_order_link_ids
    _assert_invariant(eng)


# ---------------------------------------------------------------------------
# restore_pending_entries
# ---------------------------------------------------------------------------


def test_restore_pending_entries_basic() -> None:
    eng = _engine(shadow=False)
    eng.restore_pending_entries(["order-1", "order-2"])
    assert "order-1" in eng._pending_entry_order_link_ids
    assert "order-2" in eng._pending_entry_order_link_ids
    assert eng._pending_entry_count == 2
    _assert_invariant(eng)


def test_restore_pending_entries_deduplicates() -> None:
    eng = _engine(shadow=False)
    eng.restore_pending_entries(["order-1", "order-1", "order-2"])
    assert eng._pending_entry_count == 2
    _assert_invariant(eng)


def test_restore_pending_entries_filters_empty() -> None:
    eng = _engine(shadow=False)
    eng.restore_pending_entries(["", "order-1", "", "order-2"])
    assert eng._pending_entry_count == 2
    assert "" not in eng._pending_entry_order_link_ids
    _assert_invariant(eng)


def test_restore_pending_entries_empty_list() -> None:
    eng = _engine(shadow=False)
    eng.restore_pending_entries([])
    assert eng._pending_entry_count == 0
    _assert_invariant(eng)


# ---------------------------------------------------------------------------
# restore_pending_entries_with_symbols
# ---------------------------------------------------------------------------


def test_restore_with_symbols_basic() -> None:
    eng = _engine(shadow=False)
    now = datetime.now(tz=UTC)
    records = [
        {"order_link_id": "order-1", "symbol": "BTCUSDT", "created_at": now},
        {"order_link_id": "order-2", "symbol": "ETHUSDT", "created_at": now},
    ]
    eng.restore_pending_entries_with_symbols(records)
    assert eng._pending_entry_count == 2
    assert eng._pending_entry_symbols["order-1"] == "BTCUSDT"
    assert eng._pending_entry_symbols["order-2"] == "ETHUSDT"
    assert eng._pending_entry_created_at["order-1"] == now
    _assert_invariant(eng)


def test_restore_with_symbols_deduplicates() -> None:
    eng = _engine(shadow=False)
    now = datetime.now(tz=UTC)
    records = [
        {"order_link_id": "order-1", "symbol": "BTCUSDT", "created_at": now},
        {"order_link_id": "order-1", "symbol": "BTCUSDT", "created_at": now},
    ]
    eng.restore_pending_entries_with_symbols(records)
    assert eng._pending_entry_count == 1
    _assert_invariant(eng)


def test_restore_with_symbols_filters_empty_ids() -> None:
    eng = _engine(shadow=False)
    records = [
        {"order_link_id": "", "symbol": "BTCUSDT"},
        {"order_link_id": "order-1", "symbol": "BTCUSDT"},
    ]
    eng.restore_pending_entries_with_symbols(records)
    assert eng._pending_entry_count == 1
    assert "" not in eng._pending_entry_order_link_ids
    _assert_invariant(eng)


# ---------------------------------------------------------------------------
# Terminal event scenarios (Cancelled, Rejected, Expired)
# ---------------------------------------------------------------------------


def test_cancelled_releases_correct_slot() -> None:
    eng = _engine(shadow=False)
    eng.mark_entry_submitted("order-A")
    eng.mark_entry_submitted("order-B")
    eng.mark_entry_resolved("order-A")
    assert eng._pending_entry_count == 1
    assert "order-A" not in eng._pending_entry_order_link_ids
    assert "order-B" in eng._pending_entry_order_link_ids
    _assert_invariant(eng)


def test_rejected_releases_correct_slot() -> None:
    eng = _engine(shadow=False)
    eng.mark_entry_submitted("order-X")
    eng.mark_entry_submitted("order-Y")
    eng.mark_entry_resolved("order-X")
    assert eng._pending_entry_count == 1
    assert "order-X" not in eng._pending_entry_order_link_ids
    assert "order-Y" in eng._pending_entry_order_link_ids
    _assert_invariant(eng)


def test_expired_releases_correct_slot() -> None:
    eng = _engine(shadow=False)
    eng.mark_entry_submitted("order-Z")
    eng.mark_entry_resolved("order-Z")
    assert eng._pending_entry_count == 0
    _assert_invariant(eng)


# ---------------------------------------------------------------------------
# Filled + ExecutionUpdateEvent double-release protection
# ---------------------------------------------------------------------------


def test_filled_then_execution_update_does_not_double_decrement() -> None:
    """Simulates OrderUpdateEvent(FILLED) then ExecutionUpdateEvent for same order."""
    eng = _engine(shadow=False)
    eng.mark_entry_submitted("order-fill-1")

    # OrderUpdateEvent(FILLED) → first release
    eng.mark_entry_resolved("order-fill-1")
    assert eng._pending_entry_count == 0
    _assert_invariant(eng)

    # ExecutionUpdateEvent → second release attempt must be a no-op
    eng.mark_entry_resolved("order-fill-1")
    assert eng._pending_entry_count == 0
    _assert_invariant(eng)


# ---------------------------------------------------------------------------
# reconcile_restored_pending_entries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_api_unavailable_preserves_all() -> None:
    """API unavailable during reconcile → all pending entries preserved."""
    eng = _engine(shadow=False)
    eng.restore_pending_entries(["order-stale"])

    eng._adapter.get_open_orders = AsyncMock(side_effect=RuntimeError("API down"))
    eng._trade_journal.get_pending_order_events = AsyncMock(return_value=[])
    eng._trade_journal.get_pending_durable_orders = AsyncMock(return_value=[])

    await eng.reconcile_restored_pending_entries()

    assert "order-stale" in eng._pending_entry_order_link_ids
    assert eng._pending_entry_count == 1
    _assert_invariant(eng)


@pytest.mark.asyncio
async def test_reconcile_stale_pending_cleared() -> None:
    """Old pending without exchange order and no position is cleared."""
    eng = _engine(shadow=False)
    old_time = datetime.now(tz=UTC) - timedelta(seconds=700)
    eng.restore_pending_entries(["order-old"])
    eng._pending_entry_created_at["order-old"] = old_time

    eng._adapter.get_open_orders = AsyncMock(return_value=[])
    eng._trade_journal.get_pending_order_events = AsyncMock(
        return_value=[{"order_link_id": "order-old", "symbol": "BTCUSDT", "created_at": old_time}]
    )
    eng._trade_journal.get_pending_durable_orders = AsyncMock(return_value=[])

    await eng.reconcile_restored_pending_entries()

    assert "order-old" not in eng._pending_entry_order_link_ids
    assert eng._pending_entry_count == 0
    _assert_invariant(eng)


@pytest.mark.asyncio
async def test_reconcile_recent_pending_kept() -> None:
    """Recent pending (under threshold) must not be cleared by reconcile."""
    eng = _engine(shadow=False)
    recent_time = datetime.now(tz=UTC) - timedelta(seconds=60)
    eng.restore_pending_entries(["order-recent"])
    eng._pending_entry_created_at["order-recent"] = recent_time

    eng._adapter.get_open_orders = AsyncMock(return_value=[])
    eng._trade_journal.get_pending_order_events = AsyncMock(
        return_value=[{"order_link_id": "order-recent", "symbol": "BTCUSDT", "created_at": recent_time}]
    )
    eng._trade_journal.get_pending_durable_orders = AsyncMock(return_value=[])

    await eng.reconcile_restored_pending_entries()

    assert "order-recent" in eng._pending_entry_order_link_ids
    assert eng._pending_entry_count == 1
    _assert_invariant(eng)


@pytest.mark.asyncio
async def test_reconcile_syncs_count_after() -> None:
    """Count is correct after reconcile clears some entries."""
    eng = _engine(shadow=False)
    old_time = datetime.now(tz=UTC) - timedelta(seconds=700)
    eng.restore_pending_entries(["order-stale", "order-live"])
    eng._pending_entry_created_at["order-stale"] = old_time

    eng._adapter.get_open_orders = AsyncMock(return_value=[{"orderLinkId": "order-live"}])
    eng._trade_journal.get_pending_order_events = AsyncMock(
        return_value=[{"order_link_id": "order-stale", "symbol": "BTCUSDT", "created_at": old_time}]
    )
    eng._trade_journal.get_pending_durable_orders = AsyncMock(return_value=[])

    await eng.reconcile_restored_pending_entries()

    # order-live has open exchange order → kept; order-stale → cleared
    assert "order-live" in eng._pending_entry_order_link_ids
    _assert_invariant(eng)


# ---------------------------------------------------------------------------
# Restart simulation
# ---------------------------------------------------------------------------


def test_restart_restore_then_resolve() -> None:
    """Full restart flow: restore IDs then resolve one by one."""
    eng = _engine(shadow=False)
    eng.restore_pending_entries_with_symbols(
        [
            {"order_link_id": "A", "symbol": "BTCUSDT"},
            {"order_link_id": "B", "symbol": "ETHUSDT"},
        ]
    )
    _assert_invariant(eng)

    eng.mark_entry_resolved("A")
    assert eng._pending_entry_count == 1
    _assert_invariant(eng)

    eng.mark_entry_resolved("B")
    assert eng._pending_entry_count == 0
    _assert_invariant(eng)


def test_restart_no_phantom_slot_on_empty_restore() -> None:
    """Restarting with no pending entries must leave count at 0."""
    eng = _engine(shadow=False)
    eng.restore_pending_entries([])
    eng.restore_pending_entries_with_symbols([])
    assert eng._pending_entry_count == 0
    _assert_invariant(eng)
