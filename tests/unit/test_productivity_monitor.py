"""Tests for RuntimeProductivityMonitor."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trader.analytics.productivity import RuntimeProductivityMonitor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _monitor() -> RuntimeProductivityMonitor:
    return RuntimeProductivityMonitor()


# ---------------------------------------------------------------------------
# Basic event recording
# ---------------------------------------------------------------------------


def test_scanner_cycle_increments_counter():
    m = _monitor()
    m.record_scanner_cycle()
    m.record_scanner_cycle()
    snap = m.snapshot()
    assert snap.scanner_iterations_total == 2
    assert snap.last_scanner_cycle_at is not None


def test_strategy_loop_increments_counter():
    m = _monitor()
    m.record_strategy_loop()
    snap = m.snapshot()
    assert snap.strategy_iterations_total == 1
    assert snap.last_strategy_loop_at is not None


def test_signal_unapproved_not_counted_in_approved():
    m = _monitor()
    m.record_signal(approved=False)
    snap = m.snapshot()
    assert snap.signals_last_hour == 1
    assert snap.approved_last_hour == 0


def test_signal_approved_counted_in_both():
    m = _monitor()
    m.record_signal(approved=True)
    snap = m.snapshot()
    assert snap.signals_last_hour == 1
    assert snap.approved_last_hour == 1


def test_fill_recorded():
    m = _monitor()
    m.record_fill()
    snap = m.snapshot()
    assert snap.filled_last_hour == 1
    assert snap.last_order_filled_at is not None


def test_position_closed_increments_counter():
    m = _monitor()
    m.record_position_closed()
    snap = m.snapshot()
    assert snap.positions_closed_total == 1
    assert snap.last_position_closed_at is not None
    assert snap.reentries_total == 0


def test_reentry_increments_reentry_counter():
    m = _monitor()
    m.record_position_closed(reentry=True)
    snap = m.snapshot()
    assert snap.reentries_total == 1
    assert snap.last_position_reopened_at is not None


def test_order_submitted_increments_positions_opened():
    m = _monitor()
    m.record_order_submitted()
    snap = m.snapshot()
    assert snap.positions_opened_total == 1


# ---------------------------------------------------------------------------
# Rolling-hour window
# ---------------------------------------------------------------------------


def test_productivity_heartbeat_counts_events():
    """Events recorded within 1 hour appear in snapshot counts."""
    m = _monitor()
    for _ in range(5):
        m.record_signal(approved=False)
    for _ in range(3):
        m.record_signal(approved=True)
    m.record_fill()
    m.record_position_closed()

    snap = m.snapshot()
    assert snap.signals_last_hour == 8
    assert snap.approved_last_hour == 3
    assert snap.filled_last_hour == 1
    assert snap.closed_last_hour == 1


def test_old_events_expire_from_rolling_window():
    """Events older than 1 hour should not appear in the hourly counts."""
    m = _monitor()

    # Inject an old timestamp directly into the deque
    old_ts = datetime.now(tz=UTC) - timedelta(hours=2)
    m._signal_times.append(old_ts)
    m._approved_times.append(old_ts)

    # Add one recent event
    m.record_signal(approved=True)

    snap = m.snapshot()
    # Only the recent signal should count
    assert snap.signals_last_hour == 1
    assert snap.approved_last_hour == 1


# ---------------------------------------------------------------------------
# Top rejection reason
# ---------------------------------------------------------------------------


def test_top_rejection_reason():
    m = _monitor()
    m.record_rejection("low_volume")
    m.record_rejection("low_volume")
    m.record_rejection("high_spread")
    snap = m.snapshot()
    assert snap.top_rejection_reason == "low_volume"


def test_old_rejection_reasons_expire_from_rolling_window():
    m = _monitor()
    old_ts = datetime.now(tz=UTC) - timedelta(hours=2)
    m._rejection_times.append((old_ts, "old_risk"))
    m._rejection_counts["old_risk"] = 1
    m.record_rejection("fresh_risk")

    snap = m.snapshot()

    assert snap.top_rejection_reason == "fresh_risk"
    assert "old_risk" not in m._rejection_counts


def test_no_rejections_gives_none():
    m = _monitor()
    snap = m.snapshot()
    assert snap.top_rejection_reason is None


# ---------------------------------------------------------------------------
# Heartbeat logging (just verify it runs without error)
# ---------------------------------------------------------------------------


def test_heartbeat_does_not_raise():
    m = _monitor()
    m.record_scanner_cycle()
    m.record_strategy_loop()
    m.record_signal(approved=True)
    m.log_heartbeat(mode="SHADOW", open_positions=0)


# ---------------------------------------------------------------------------
# New trade after slot released (integration-style)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_shadow_trade_after_previous_tp():
    """Verify slot tracking: after a position closes, positions_opened can increment again."""
    m = _monitor()

    # Open position
    m.record_order_submitted()
    snap_open = m.snapshot()
    assert snap_open.positions_opened_total == 1

    # TP fill
    m.record_fill()
    m.record_position_closed()

    # New trade
    m.record_order_submitted()
    snap_new = m.snapshot()
    assert snap_new.positions_opened_total == 2
    assert snap_new.positions_closed_total == 1


@pytest.mark.asyncio
async def test_new_trade_after_slot_released():
    """Multiple open/close cycles accumulate correctly."""
    m = _monitor()
    for _ in range(3):
        m.record_order_submitted()
        m.record_fill()
        m.record_position_closed()

    snap = m.snapshot()
    assert snap.positions_opened_total == 3
    assert snap.positions_closed_total == 3


@pytest.mark.asyncio
async def test_same_symbol_reentry_after_cooldown():
    """Reentry tracking: close then reopen same symbol → reentry count increases."""
    m = _monitor()
    m.record_order_submitted()
    m.record_fill()
    m.record_position_closed()

    # Reentry (same symbol)
    m.record_order_submitted()
    m.record_fill()
    m.record_position_closed(reentry=True)

    snap = m.snapshot()
    assert snap.reentries_total == 1
    assert snap.positions_opened_total == 2


@pytest.mark.asyncio
async def test_different_symbol_not_blocked_by_symbol_cooldown():
    """Monitor tracks events independently; no symbol-level blocking logic here."""
    m = _monitor()
    # Signal on symbolA rejected (cooldown)
    m.record_rejection("cooldown")
    # Signal on symbolB approved
    m.record_signal(approved=True)

    snap = m.snapshot()
    assert snap.approved_last_hour == 1
    assert snap.top_rejection_reason == "cooldown"


@pytest.mark.asyncio
async def test_productivity_heartbeat():
    """Heartbeat snapshot reflects recorded events."""
    m = _monitor()
    m.record_scanner_cycle()
    m.record_feature_cycle()
    m.record_strategy_loop()
    m.record_signal(approved=True)
    m.record_order_submitted()
    m.record_fill()
    m.record_position_closed()

    snap = m.snapshot()
    assert snap.scanner_iterations_total == 1
    assert snap.strategy_iterations_total == 1
    assert snap.signals_last_hour == 1
    assert snap.approved_last_hour == 1
    assert snap.filled_last_hour == 1
    assert snap.closed_last_hour == 1
    assert snap.last_scanner_cycle_at is not None
    assert snap.last_feature_cycle_at is not None
    assert snap.last_strategy_loop_at is not None
    assert snap.last_order_filled_at is not None
    assert snap.last_position_closed_at is not None
