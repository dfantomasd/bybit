"""Regression tests for DB write/retention/training alignment fixes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from trader.storage.trade_journal import TradeJournal, _parse_command_rowcount
from trader.training.train import _settings_label_bps


def test_parse_command_rowcount_delete_and_insert() -> None:
    assert _parse_command_rowcount("DELETE 42") == 42
    assert _parse_command_rowcount("INSERT 0 1") == 1
    assert _parse_command_rowcount("INSERT 0 0") == 0
    assert _parse_command_rowcount(None) == 0


@pytest.mark.asyncio
async def test_execute_returns_command_tag_for_retention() -> None:
    journal = TradeJournal(postgres_dsn="postgresql://u:p@localhost/db", enabled=True)
    journal._pool = MagicMock()
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="DELETE 7")
    journal._pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    journal._pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    result = await journal._execute("DELETE FROM market_candles WHERE interval = $1", "1")

    assert result == "DELETE 7"


@pytest.mark.asyncio
async def test_closed_pnl_counts_only_inserted_rows() -> None:
    journal = TradeJournal(postgres_dsn="postgresql://u:p@localhost/db", enabled=True)
    journal._execute = AsyncMock(side_effect=["INSERT 0 1", "INSERT 0 0"])

    inserted = await journal.record_closed_pnl_records(
        [
            {"symbol": "BTCUSDT", "closedPnl": "1.5", "updatedTime": "1700000000000"},
            {"symbol": "ETHUSDT", "closedPnl": "2.0", "updatedTime": "1700000001000"},
        ]
    )

    assert inserted == 1


@pytest.mark.asyncio
async def test_fetch_online_learning_batch_orders_by_resolved_at() -> None:
    journal = TradeJournal(postgres_dsn="postgresql://u:p@localhost/db", enabled=True)
    journal._fetch = AsyncMock(return_value=[])

    await journal.fetch_online_learning_batch(limit=10)

    query = journal._fetch.await_args.args[0]
    assert "po.resolved_at DESC" in query
    assert "po.created_at" not in query


def test_settings_label_bps_defaults_to_two(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_AUTO_TRAIN_LABEL_BPS", raising=False)
    assert _settings_label_bps() == 2.0


def test_journal_fallback_uuid_is_valid_nil() -> None:
    from trader.app import _JOURNAL_FALLBACK_UUID

    assert _JOURNAL_FALLBACK_UUID == UUID(int=0)
