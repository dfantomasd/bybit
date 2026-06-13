"""Tests for exchange_order_id → order_link_id reverse lookup (P0.4)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.storage.trade_journal import TradeJournal


def _make_journal(*, enabled: bool) -> TradeJournal:
    jnl = TradeJournal.__new__(TradeJournal)
    jnl._enabled = enabled
    jnl._pool = MagicMock() if enabled else None
    return jnl


@pytest.mark.asyncio
async def test_reverse_lookup_finds_via_durable_state() -> None:
    jnl = _make_journal(enabled=True)
    jnl._fetch = AsyncMock(
        side_effect=[
            [{"order_link_id": "my-link-id"}],
        ]
    )
    result = await jnl.find_order_link_id_by_exchange_order_id("ex-123")
    assert result == "my-link-id"


@pytest.mark.asyncio
async def test_reverse_lookup_falls_back_to_order_events() -> None:
    jnl = _make_journal(enabled=True)
    jnl._fetch = AsyncMock(
        side_effect=[
            [],
            [{"order_link_id": "fallback-link"}],
        ]
    )
    result = await jnl.find_order_link_id_by_exchange_order_id("ex-456")
    assert result == "fallback-link"


@pytest.mark.asyncio
async def test_reverse_lookup_returns_none_when_not_found() -> None:
    jnl = _make_journal(enabled=True)
    jnl._fetch = AsyncMock(return_value=[])
    result = await jnl.find_order_link_id_by_exchange_order_id("unknown")
    assert result is None


@pytest.mark.asyncio
async def test_reverse_lookup_returns_none_when_disabled() -> None:
    jnl = _make_journal(enabled=False)
    jnl._fetch = AsyncMock()
    result = await jnl.find_order_link_id_by_exchange_order_id("ex-789")
    assert result is None
    jnl._fetch.assert_not_called()
