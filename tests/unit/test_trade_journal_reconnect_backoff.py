from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trader.storage.trade_journal import TradeJournal


def test_auth_circuit_breaker_error_detection() -> None:
    assert TradeJournal._is_auth_circuit_breaker_error(
        "(ECIRCUITBREAKER) too many authentication failures, new connections are temporarily blocked"
    )
    assert TradeJournal._is_auth_circuit_breaker_error("(EAUTHQUERY) authentication query failed")
    assert not TradeJournal._is_auth_circuit_breaker_error("connection was closed in the middle of operation")


def test_schedule_reconnect_backoff_uses_long_delay_for_auth_circuit_breaker() -> None:
    journal = TradeJournal("postgresql://example/db")
    journal._schedule_reconnect_backoff("(ECIRCUITBREAKER) too many authentication failures")

    assert journal._connect_failures == 1
    assert journal._reconnect_blocked_until is not None
    remaining = (journal._reconnect_blocked_until - datetime.now(tz=UTC)).total_seconds()
    assert remaining >= 299.0


@pytest.mark.asyncio
async def test_reconnect_if_needed_respects_backoff_window() -> None:
    journal = TradeJournal("postgresql://example/db")
    journal._reconnect_blocked_until = datetime.now(tz=UTC) + timedelta(seconds=300)

    allowed = await journal.reconnect_if_needed(min_interval=0.0, force=False)

    assert allowed is False
    assert journal._pool is None
