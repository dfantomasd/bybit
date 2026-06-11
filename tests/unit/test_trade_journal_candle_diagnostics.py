"""Regression tests for candle diagnostics readiness inputs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

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
