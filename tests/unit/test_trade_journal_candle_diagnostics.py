"""Regression tests for candle diagnostics readiness inputs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest

from trader.storage.trade_journal import TradeJournal


class _FetchRecorder:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def __call__(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.calls.append((query, args))
        return self.rows


@pytest.mark.asyncio
async def test_candle_counts_use_only_confirmed_candles() -> None:
    journal = TradeJournal("postgresql://example/db")
    fetch = _FetchRecorder([{"interval": "1", "cnt": 42}])
    journal._fetch = fetch  # type: ignore[method-assign]

    counts = await journal.get_candle_counts()

    assert counts == {"1": 42}
    assert "WHERE confirmed = true" in fetch.calls[0][0]


@pytest.mark.asyncio
async def test_candle_readiness_counts_are_capped_per_interval() -> None:
    journal = TradeJournal("postgresql://example/db")
    fetch = _FetchRecorder([{"cnt": 7}])
    journal._fetch = fetch  # type: ignore[method-assign]

    counts = await journal.get_candle_readiness_counts()

    assert counts == {"1": 7, "5": 7, "15": 7, "60": 7}
    assert len(fetch.calls) == 4
    limits = [call[1][1] for call in fetch.calls]
    assert limits == [1000, 200, 200, 100]
    assert all("LIMIT $2" in call[0] for call in fetch.calls)
    assert all("WHERE interval = $1" in call[0] for call in fetch.calls)


@pytest.mark.asyncio
async def test_latest_candle_time_uses_only_confirmed_candles() -> None:
    journal = TradeJournal("postgresql://example/db")
    latest = datetime(2026, 6, 11, 10, 0, tzinfo=UTC)
    fetch = _FetchRecorder([{"ts": latest}])
    journal._fetch = fetch  # type: ignore[method-assign]

    result = await journal.get_latest_candle_time("1")

    assert result == latest
    query, args = fetch.calls[0]
    assert "MAX(open_time)" in query
    assert "AND confirmed = true" in query
    assert args == ("1",)


@pytest.mark.asyncio
async def test_db_diagnostics_reports_last_confirmed_candle_age() -> None:
    journal = TradeJournal("postgresql://example/db")
    journal._pool = object()  # type: ignore[assignment]
    journal.get_candle_readiness_counts = AsyncMock(return_value={"1": 10})  # type: ignore[method-assign]
    journal.get_latest_candle_time = AsyncMock(return_value=datetime.now(tz=UTC) - timedelta(seconds=45))  # type: ignore[method-assign]

    async def fake_fetch(query: str, *args: Any) -> list[dict[str, Any]]:
        del args
        if "FROM feature_snapshots" in query:
            return [{"cnt": 0}]
        if "FROM prediction_outcomes" in query and "horizon_minutes" not in query:
            return [{"cnt": 0}]
        return []

    journal._fetch = fake_fetch  # type: ignore[method-assign]

    diag = await journal.get_db_diagnostics()

    assert diag["latest_candle_1m"] is not None
    assert 0 <= diag["last_confirmed_candle_age_s"] <= 60


@pytest.mark.asyncio
async def test_db_diagnostics_uses_capped_readiness_counts() -> None:
    journal = TradeJournal("postgresql://example/db")
    journal._pool = object()  # type: ignore[assignment]
    journal.get_candle_readiness_counts = AsyncMock(return_value={"1": 1000, "5": 200})  # type: ignore[method-assign]
    journal.get_latest_candle_time = AsyncMock(return_value=None)  # type: ignore[method-assign]

    async def fake_fetch(query: str, *args: Any) -> list[dict[str, Any]]:
        if "FROM feature_snapshots" in query and "LIMIT $1" in query:
            assert "LIMIT $1" in query
            assert args == (1000,)
            return [{"cnt": 1000}]
        if "FROM prediction_outcomes" in query and "horizon_minutes = 15" in query and "LIMIT $1" in query:
            assert "LIMIT $1" in query
            assert args == (1000,)
            return [{"cnt": 1000}]
        if "FROM prediction_outcomes" in query and "LIMIT $1" in query:
            assert "LIMIT $1" in query
            assert args == (1000,)
            return [{"cnt": 1000}]
        return []

    journal._fetch = fake_fetch  # type: ignore[method-assign]

    diag = await journal.get_db_diagnostics()

    assert diag["candles_by_interval"] == {"1": 1000, "5": 200}
    assert diag["feature_snapshots"] == 1000
    assert diag["prediction_outcomes"] == 1000
    assert diag["training_eligible_15m"] == 1000
