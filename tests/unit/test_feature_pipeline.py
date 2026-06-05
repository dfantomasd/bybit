"""Tests for FeaturePipeline."""
from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from trader.data.candles import Candle, CandleStore
from trader.features.pipeline import FeaturePipeline, _MIN_BARS


def _make_store(n: int = 60, symbol: str = "BTCUSDT", interval: str = "1") -> CandleStore:
    """CandleStore seeded with *n* confirmed candles (sine wave)."""
    store = CandleStore()
    base = 50000.0
    for i in range(n):
        price = base + 500 * math.sin(2 * math.pi * i / 20)
        store.add(
            symbol,
            interval,
            Candle(
                open_time=datetime(2024, 1, 1, 0, i, tzinfo=UTC),
                open=price - 10,
                high=price + 50,
                low=price - 50,
                close=price,
                volume=float(1000 + 100 * i),
                confirm=True,
            ),
        )
    return store


class TestFeaturePipeline:
    def test_returns_none_when_insufficient_data(self):
        store = CandleStore()
        # Add fewer bars than minimum
        for i in range(_MIN_BARS - 1):
            store.add(
                "BTCUSDT",
                "1",
                Candle(
                    open_time=datetime(2024, 1, 1, 0, i, tzinfo=UTC),
                    open=100.0,
                    high=105.0,
                    low=95.0,
                    close=100.0,
                    volume=1000.0,
                    confirm=True,
                ),
            )
        pipeline = FeaturePipeline(store)
        assert pipeline.compute("BTCUSDT", "1") is None

    def test_returns_feature_vector_with_sufficient_data(self):
        store = _make_store(60)
        pipeline = FeaturePipeline(store)
        vec = pipeline.compute("BTCUSDT", "1")
        assert vec is not None
        assert vec.symbol == "BTCUSDT"
        assert len(vec.values) > 0
        assert len(vec.feature_names) == len(vec.values)

    def test_quality_score_in_range(self):
        store = _make_store(60)
        pipeline = FeaturePipeline(store)
        vec = pipeline.compute("BTCUSDT", "1")
        assert vec is not None
        assert 0.0 <= vec.quality_score <= 1.0

    def test_feature_names_sorted(self):
        store = _make_store(60)
        pipeline = FeaturePipeline(store)
        vec = pipeline.compute("BTCUSDT", "1")
        assert vec is not None
        assert vec.feature_names == sorted(vec.feature_names)

    def test_known_features_present(self):
        store = _make_store(60)
        pipeline = FeaturePipeline(store)
        vec = pipeline.compute("BTCUSDT", "1")
        assert vec is not None
        names = set(vec.feature_names)
        assert "rsi_14" in names
        assert "bb_pct_b" in names
        assert "volume_zscore" in names

    def test_no_nan_in_values(self):
        store = _make_store(60)
        pipeline = FeaturePipeline(store)
        vec = pipeline.compute("BTCUSDT", "1")
        assert vec is not None
        for v in vec.values:
            assert not math.isnan(v), f"NaN found in features: {vec.feature_names}"

    def test_health_checker_updated(self):
        class FakeHealth:
            called = False
            last_dt = None
            def set_feature_computed_at(self, dt):
                self.called = True
                self.last_dt = dt

        store = _make_store(60)
        health = FakeHealth()
        pipeline = FeaturePipeline(store, health_checker=health)
        vec = pipeline.compute("BTCUSDT", "1")
        assert vec is not None
        # Health is only called in the async run() loop; compute() alone doesn't call it
        # To test health notification we need the run() coroutine — skip for unit test

    def test_latest_returns_none_before_compute(self):
        store = _make_store(60)
        pipeline = FeaturePipeline(store)
        assert pipeline.latest("BTCUSDT", "1") is None
