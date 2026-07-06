from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from trader.storage.trade_journal import TradeJournal


def test_auth_circuit_breaker_error_detection() -> None:
    assert TradeJournal._is_auth_circuit_breaker_error(
        "(ECIRCUITBREAKER) too many authentication failures, new connections are temporarily blocked"
    )
    assert TradeJournal._is_auth_circuit_breaker_error("(EAUTHQUERY) authentication query failed")
    assert not TradeJournal._is_auth_circuit_breaker_error("connection was closed in the middle of operation")


def test_schedule_reconnect_backoff_uses_long_delay_for_auth_circuit_breaker() -> None:
    journal = TradeJournal(
        "postgresql://example/db",
        reconnect_max_backoff_seconds=1800.0,
        auth_circuit_breaker_min_backoff_seconds=900.0,
    )
    journal._schedule_reconnect_backoff("(ECIRCUITBREAKER) too many authentication failures")

    assert journal._connect_failures == 1
    assert journal._last_backoff_was_auth is True
    assert journal._reconnect_blocked_until is not None
    remaining = (journal._reconnect_blocked_until - datetime.now(tz=UTC)).total_seconds()
    assert remaining >= 899.0


def test_auth_errors_are_not_treated_as_transient_schema_errors() -> None:
    assert not TradeJournal._is_transient_schema_error("(EAUTHQUERY) authentication query failed")
    assert not TradeJournal._is_transient_schema_error(
        "(ECIRCUITBREAKER) too many authentication failures, new connections are temporarily blocked"
    )


def test_pooler_decode_attribute_error_is_treated_as_transient_schema_error() -> None:
    assert TradeJournal._is_transient_schema_error("'NoneType' object has no attribute 'decode'")


@pytest.mark.asyncio
async def test_reconnect_if_needed_respects_backoff_window() -> None:
    journal = TradeJournal("postgresql://example/db")
    journal._reconnect_blocked_until = datetime.now(tz=UTC) + timedelta(seconds=300)

    allowed = await journal.reconnect_if_needed(min_interval=0.0, force=False)

    assert allowed is False
    assert journal._pool is None


@pytest.mark.asyncio
async def test_reconnect_if_needed_force_true_still_respects_auth_backoff() -> None:
    journal = TradeJournal("postgresql://example/db")
    journal._reconnect_blocked_until = datetime.now(tz=UTC) + timedelta(seconds=900)
    journal._last_backoff_was_auth = True

    with patch.object(journal, "connect", new_callable=AsyncMock) as connect_mock:
        allowed = await journal.reconnect_if_needed(min_interval=0.0, force=True)

    assert allowed is False
    connect_mock.assert_not_called()


@pytest.mark.asyncio
async def test_connect_skips_while_auth_backoff_active() -> None:
    journal = TradeJournal("postgresql://example/db")
    journal._reconnect_blocked_until = datetime.now(tz=UTC) + timedelta(seconds=900)
    journal._last_backoff_was_auth = True

    with patch("trader.storage.trade_journal.asyncpg.create_pool", new_callable=AsyncMock) as create_pool:
        allowed = await journal.reconnect_if_needed(min_interval=0.0, force=True)

    assert allowed is False
    create_pool.assert_not_called()
