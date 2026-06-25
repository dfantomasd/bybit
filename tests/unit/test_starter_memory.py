from __future__ import annotations

from datetime import UTC, datetime

from trader.data.candles import Candle, CandleStore
from trader.data.orderbook_tracker import OrderbookTracker
from trader.features.pipeline import FeaturePipeline


def test_candle_store_remove_symbol_drops_all_intervals() -> None:
    store = CandleStore(max_bars=50)
    candle = Candle(
        open_time=datetime.now(tz=UTC),
        open=1.0,
        high=1.1,
        low=0.9,
        close=1.05,
        volume=100.0,
        confirm=True,
    )
    store.add("XRPUSDT", "1", candle)
    store.add("XRPUSDT", "5", candle)

    store.remove_symbol("XRPUSDT")

    assert store.count("XRPUSDT", "1") == 0
    assert store.count("XRPUSDT", "5") == 0


def test_orderbook_tracker_remove_symbol() -> None:
    tracker = OrderbookTracker()
    tracker.remove_symbol("DOGEUSDT")
    assert tracker.latest_imbalance("DOGEUSDT") is None


def test_feature_pipeline_evict_symbol() -> None:
    store = CandleStore(max_bars=50)
    pipeline = FeaturePipeline(candle_store=store)
    pipeline.evict_symbol("LINKUSDT")
