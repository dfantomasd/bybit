"""Tests for CandleStore."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trader.data.candles import Candle, CandleStore, candle_from_kline_event


def _make_candle(offset_min: int = 0, close: float = 100.0, confirm: bool = True) -> Candle:
    return Candle(
        open_time=datetime(2024, 1, 1, 0, offset_min, tzinfo=UTC),
        open=close - 1,
        high=close + 2,
        low=close - 2,
        close=close,
        volume=1000.0,
        confirm=confirm,
    )


class TestCandleStore:
    def test_add_and_retrieve(self):
        store = CandleStore()
        c = _make_candle(0, 100.0)
        store.add("BTCUSDT", "1", c)
        assert store.count("BTCUSDT", "1") == 1
        assert store.closes("BTCUSDT", "1") == [100.0]

    def test_multiple_candles_ordering(self):
        store = CandleStore()
        for i in range(5):
            store.add("BTCUSDT", "1", _make_candle(i, float(100 + i)))
        closes = store.closes("BTCUSDT", "1")
        assert closes == [100.0, 101.0, 102.0, 103.0, 104.0]

    def test_live_bar_update(self):
        store = CandleStore()
        c1 = _make_candle(0, 100.0, confirm=False)
        c2 = _make_candle(0, 101.0, confirm=False)
        store.add("BTCUSDT", "1", c1)
        store.add("BTCUSDT", "1", c2)
        # Should still be only one bar (same timestamp → overwrite)
        assert store.count("BTCUSDT", "1", confirmed_only=False) == 1
        # Unconfirmed bar not in confirmed
        assert store.count("BTCUSDT", "1", confirmed_only=True) == 0

    def test_confirmed_filter(self):
        store = CandleStore()
        store.add("BTCUSDT", "1", _make_candle(0, 100.0, confirm=True))
        store.add("BTCUSDT", "1", _make_candle(1, 101.0, confirm=False))
        assert store.count("BTCUSDT", "1", confirmed_only=True) == 1
        assert store.count("BTCUSDT", "1", confirmed_only=False) == 2

    def test_is_ready(self):
        store = CandleStore()
        for i in range(29):
            store.add("BTCUSDT", "1", _make_candle(i))
        assert not store.is_ready("BTCUSDT", "1", 30)
        store.add("BTCUSDT", "1", _make_candle(29))
        assert store.is_ready("BTCUSDT", "1", 30)

    def test_maxlen_respected(self):
        store = CandleStore(max_bars=5)
        for i in range(10):
            store.add("BTCUSDT", "1", _make_candle(i, float(100 + i)))
        # Only last 5 should remain
        assert store.count("BTCUSDT", "1", confirmed_only=False) == 5
        closes = store.closes("BTCUSDT", "1")
        assert closes[-1] == 109.0

    def test_latest_n(self):
        store = CandleStore()
        for i in range(20):
            store.add("BTCUSDT", "1", _make_candle(i, float(100 + i)))
        latest = store.latest("BTCUSDT", "1", 5)
        assert len(latest) == 5
        assert latest[-1].close == 119.0

    def test_empty_symbol(self):
        store = CandleStore()
        assert store.closes("XYZUSDT", "1") == []
        assert store.count("XYZUSDT", "1") == 0
        assert not store.is_ready("XYZUSDT", "1", 1)

    def test_separate_symbols(self):
        store = CandleStore()
        store.add("BTCUSDT", "1", _make_candle(0, 50000.0))
        store.add("ETHUSDT", "1", _make_candle(0, 3000.0))
        assert store.closes("BTCUSDT", "1") == [50000.0]
        assert store.closes("ETHUSDT", "1") == [3000.0]

    def test_separate_intervals(self):
        store = CandleStore()
        store.add("BTCUSDT", "1", _make_candle(0, 100.0))
        store.add("BTCUSDT", "5", _make_candle(0, 200.0))
        assert store.closes("BTCUSDT", "1") == [100.0]
        assert store.closes("BTCUSDT", "5") == [200.0]

    def test_highs_lows_volumes(self):
        store = CandleStore()
        c = Candle(
            open_time=datetime(2024, 1, 1, tzinfo=UTC),
            open=99.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=5000.0,
            confirm=True,
        )
        store.add("BTCUSDT", "1", c)
        assert store.highs("BTCUSDT", "1") == [105.0]
        assert store.lows("BTCUSDT", "1") == [95.0]
        assert store.volumes("BTCUSDT", "1") == [5000.0]
