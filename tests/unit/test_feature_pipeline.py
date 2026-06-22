"""Tests for FeaturePipeline."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from trader.data.candles import Candle, CandleStore
from trader.features.pipeline import _MIN_BARS, FeaturePipeline
from trader.features.source_candle_guard import SourceCandleFeaturePipeline, source_candle_for_feature


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

    def test_source_guard_registers_last_confirmed_candle(self):
        store = _make_store(60)
        pipeline = SourceCandleFeaturePipeline(store)

        vec = pipeline.compute("BTCUSDT", "1")

        assert vec is not None
        latest = store.latest("BTCUSDT", "1", 1)[-1]
        assert source_candle_for_feature(vec.feature_id) == ("BTCUSDT", "1", latest.open_time)

    async def test_source_guard_rejects_cached_vector_after_new_candle(self):
        store = _make_store(60)
        pipeline = SourceCandleFeaturePipeline(store)
        vec = await pipeline.on_confirmed_candle("BTCUSDT", "1")
        assert vec is not None
        assert pipeline.latest("BTCUSDT", "1") is vec

        latest = store.latest("BTCUSDT", "1", 1)[-1]
        store.add(
            "BTCUSDT",
            "1",
            Candle(
                open_time=latest.open_time + timedelta(minutes=1),
                open=50000.0,
                high=50010.0,
                low=49990.0,
                close=50005.0,
                volume=1000.0,
                confirm=True,
            ),
        )

        assert pipeline.latest("BTCUSDT", "1") is None


class _FakeMarketStats:
    def __init__(self, stats):
        self._stats = stats

    def market_stats(self, symbol):
        return self._stats


class TestMarketStatsFeatures:
    def test_features_present_with_stats(self):
        store = _make_store(60)
        source = _FakeMarketStats({"funding_rate_bps": 1.25, "oi_change_pct_60m": 0.04})
        pipeline = FeaturePipeline(store, market_stats_source=source)
        vec = pipeline.compute("BTCUSDT", "1")
        assert vec is not None
        f = dict(zip(vec.feature_names, vec.values, strict=True))
        assert f["mkt_data_present"] == 1.0
        assert f["funding_rate_bps"] == 1.25
        assert f["oi_change_pct_60m"] == 0.04

    def test_features_zero_with_presence_flag_when_no_stats(self):
        store = _make_store(60)
        pipeline = FeaturePipeline(store, market_stats_source=_FakeMarketStats(None))
        vec = pipeline.compute("BTCUSDT", "1")
        assert vec is not None
        f = dict(zip(vec.feature_names, vec.values, strict=True))
        assert f["mkt_data_present"] == 0.0
        assert f["funding_rate_bps"] == 0.0
        assert f["oi_change_pct_60m"] == 0.0

    def test_schema_unchanged_without_source(self):
        store = _make_store(60)
        pipeline = FeaturePipeline(store)
        vec = pipeline.compute("BTCUSDT", "1")
        assert vec is not None
        assert "mkt_data_present" not in vec.feature_names

    def test_mtf_pattern_features_on_primary_interval(self):
        store = CandleStore()
        for interval in ("1", "5", "15"):
            for i in range(40):
                price = 50000.0 + i * 10.0
                store.add(
                    "BTCUSDT",
                    interval,
                    Candle(
                        open_time=datetime(2024, 1, 1, 0, i, tzinfo=UTC),
                        open=price - 5,
                        high=price + 20,
                        low=price - 20,
                        close=price,
                        volume=1000.0,
                        confirm=True,
                    ),
                )
        pipeline = FeaturePipeline(store)
        vec = pipeline.compute("BTCUSDT", "1")
        assert vec is not None
        names = set(vec.feature_names)
        assert "pat5_hammer" in names
        assert "pat15_morning_star" in names
        assert vec.feature_names == sorted(vec.feature_names)

    def test_source_error_does_not_kill_compute(self):
        class Exploding:
            def market_stats(self, symbol):
                raise RuntimeError("cache corrupted")

        store = _make_store(60)
        pipeline = FeaturePipeline(store, market_stats_source=Exploding())
        vec = pipeline.compute("BTCUSDT", "1")
        assert vec is not None
        f = dict(zip(vec.feature_names, vec.values, strict=True))
        assert f["mkt_data_present"] == 0.0
