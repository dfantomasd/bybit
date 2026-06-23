from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from trader.modules.market_data import MarketDataModule


class _FakeCandleStore:
    def __init__(self) -> None:
        self.added: list[tuple[str, str, object]] = []

    def add(self, symbol: str, interval: str, candle: object) -> None:
        self.added.append((symbol, interval, candle))


def _rows(count: int) -> list[dict[str, object]]:
    start = datetime(2026, 6, 23, tzinfo=UTC)
    return [
        {
            "open_time": start + timedelta(minutes=i),
            "open": "1.0",
            "high": "1.1",
            "low": "0.9",
            "close": "1.0",
            "volume": "100",
        }
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_seed_interval_from_db_uses_postgres_cache_when_ready() -> None:
    store = _FakeCandleStore()
    journal = SimpleNamespace(
        is_enabled=True,
        get_recent_market_candles=AsyncMock(return_value=_rows(220)),
    )
    app = SimpleNamespace(
        _settings=SimpleNamespace(CANDLE_SEED_USE_DB_CACHE=True, CANDLE_SEED_DB_MIN_BARS=200),
        _trade_journal=journal,
        _candle_store=store,
    )

    seeded = await MarketDataModule(app)._seed_interval_from_db("XRPUSDT", "1")

    assert seeded is True
    assert len(store.added) == 220
    journal.get_recent_market_candles.assert_awaited_once()


@pytest.mark.asyncio
async def test_seed_interval_from_db_falls_back_when_history_short() -> None:
    store = _FakeCandleStore()
    journal = SimpleNamespace(
        is_enabled=True,
        get_recent_market_candles=AsyncMock(return_value=_rows(50)),
    )
    app = SimpleNamespace(
        _settings=SimpleNamespace(CANDLE_SEED_USE_DB_CACHE=True, CANDLE_SEED_DB_MIN_BARS=200),
        _trade_journal=journal,
        _candle_store=store,
    )

    seeded = await MarketDataModule(app)._seed_interval_from_db("XRPUSDT", "1")

    assert seeded is False
    assert store.added == []
