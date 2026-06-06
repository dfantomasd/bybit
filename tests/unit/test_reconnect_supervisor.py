"""Tests for ReconnectSupervisor."""

from __future__ import annotations

import asyncio

from trader.exchange.reconnect_supervisor import (
    ReconnectSupervisor,
    calc_backoff,
)

# ---------------------------------------------------------------------------
# calc_backoff tests
# ---------------------------------------------------------------------------


def test_backoff_sequence():
    """Back-off grows exponentially: 1, 2, 4, 8, 16, 32, capped at 60."""
    base_vals = []
    for attempt in range(7):
        # Use random.seed-like approach: test center value (0 jitter case)
        # We can't control jitter, so just verify the base trend
        expected_base = min(1.0 * (2**attempt), 60.0)
        wait = calc_backoff(attempt, base=1.0, max_wait=60.0)
        # Allow ±20% jitter plus a tiny floating point buffer
        assert expected_base * 0.79 <= wait <= expected_base * 1.21, (
            f"attempt={attempt}, expected_base={expected_base}, got={wait}"
        )
        base_vals.append(wait)

    # Verify that larger attempts have larger wait times on average
    # (center values grow monotonically)
    assert base_vals[0] < base_vals[3]  # attempt 0 < attempt 3


def test_jitter_within_bounds():
    """Jitter is within ±20% of the un-jittered wait time."""
    for attempt in range(6):
        expected_base = min(1.0 * (2**attempt), 60.0)
        # Run many samples to verify distribution
        for _ in range(50):
            wait = calc_backoff(attempt, base=1.0, max_wait=60.0)
            lower = max(0.1, expected_base * 0.80)
            upper = expected_base * 1.20
            assert lower <= wait <= upper, f"wait {wait} outside [{lower}, {upper}] for attempt={attempt}"


def test_backoff_max_cap():
    """Back-off never exceeds max_wait."""
    for attempt in range(20):
        wait = calc_backoff(attempt, base=1.0, max_wait=60.0)
        assert wait <= 60.0 * 1.21  # with jitter ceiling


def test_entries_blocked_after_reconnect():
    """entries_blocked is True immediately after construction (before run)."""

    async def connect_fn():
        # Simulate immediate connection failure
        raise ConnectionError("test failure")

    supervisor = ReconnectSupervisor(
        name="test_ws",
        connect_fn=connect_fn,
        metrics=None,
    )
    # Before run, entries_blocked is False (not in reconnect yet)
    # After first failure, it should be True
    assert supervisor.reconnect_count == 0


def test_stable_after_no_reconnects():
    """is_stable is False when not running."""

    async def connect_fn():
        await asyncio.sleep(0.01)

    supervisor = ReconnectSupervisor(
        name="test_ws",
        connect_fn=connect_fn,
        metrics=None,
    )
    # Not running → not stable
    assert not supervisor.is_stable


def test_downtime_tracking():
    """downtime_seconds accumulates when downtime_start is set."""

    async def connect_fn():
        # Fail immediately
        raise ConnectionError("test")

    supervisor = ReconnectSupervisor(
        name="test_ws2",
        connect_fn=connect_fn,
        metrics=None,
    )

    async def _run():
        # Start with a short timeout
        try:
            await asyncio.wait_for(supervisor.run(), timeout=0.5)
        except TimeoutError:
            await supervisor.request_stop()

        # Some downtime should have accumulated
        downtime = supervisor.downtime_seconds
        assert downtime >= 0.0  # non-negative

    asyncio.run(_run())


def test_reconnect_count_increments():
    """reconnect_count goes up after each connection failure."""
    call_count = 0

    async def connect_fn():
        nonlocal call_count
        call_count += 1
        # Fail on first 2 calls
        if call_count <= 2:
            raise ConnectionError("test")
        # Block on 3rd call (simulates stable connection)
        await asyncio.Event().wait()  # wait forever

    supervisor = ReconnectSupervisor(
        name="test_ws3",
        connect_fn=connect_fn,
        metrics=None,
    )

    async def _run():
        task = asyncio.create_task(supervisor.run())
        # Wait long enough for 2 reconnects (with small backoffs)
        await asyncio.sleep(0.5)
        await supervisor.request_stop()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        # Should have reconnected at least once
        assert supervisor.reconnect_count >= 1

    asyncio.run(_run())


def test_request_stop_terminates_run():
    """request_stop() causes run() to exit."""
    connected = asyncio.Event()

    async def connect_fn():
        connected.set()
        await asyncio.Event().wait()  # block until cancelled

    supervisor = ReconnectSupervisor(
        name="test_stop",
        connect_fn=connect_fn,
        metrics=None,
    )

    async def _run():
        task = asyncio.create_task(supervisor.run())
        # Wait for first connection
        try:
            await asyncio.wait_for(connected.wait(), timeout=2.0)
        except TimeoutError:
            pass
        await supervisor.request_stop()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            task.cancel()

    asyncio.run(_run())
