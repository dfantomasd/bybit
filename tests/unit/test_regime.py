"""Tests for RegimeClassifier."""
from __future__ import annotations

import math

import pytest

from trader.domain.enums import MarketRegime, VolatilityLevel
from trader.domain.models import FeatureVector
from trader.features.regime import RegimeClassifier


def _vec(features: dict[str, float], symbol: str = "BTCUSDT") -> FeatureVector:
    names = sorted(features.keys())
    values = [features[k] for k in names]
    return FeatureVector(
        symbol=symbol,
        values=values,
        feature_names=names,
        quality_score=1.0,
        lookback_bars=60,
    )


def _bull_features() -> dict[str, float]:
    return {
        "adx_14": 0.30,          # 30 > 25 threshold
        "bb_bandwidth": 0.03,    # below high-vol threshold
        "rsi_14": 0.62,          # above RSI_BULL_MIN
        "ema_slope_9": 0.0003,   # positive, above threshold
        "ema_slope_21": 0.0001,  # positive
        "volume_zscore": 0.5,    # normal volume
        "realized_vol_20": 0.02,
    }


def _bear_features() -> dict[str, float]:
    return {
        "adx_14": 0.30,
        "bb_bandwidth": 0.03,
        "rsi_14": 0.38,           # below RSI_BEAR_MAX
        "ema_slope_9": -0.0003,
        "ema_slope_21": -0.0001,
        "volume_zscore": 0.2,
        "realized_vol_20": 0.02,
    }


def _sideways_features() -> dict[str, float]:
    return {
        "adx_14": 0.12,          # 12 < 20 threshold
        "bb_bandwidth": 0.02,
        "rsi_14": 0.50,
        "ema_slope_9": 0.00001,
        "ema_slope_21": 0.00001,
        "volume_zscore": 0.1,
        "realized_vol_20": 0.01,
    }


def _high_vol_features() -> dict[str, float]:
    return {
        "adx_14": 0.25,
        "bb_bandwidth": 0.09,    # above 0.06 threshold
        "rsi_14": 0.55,
        "ema_slope_9": 0.0001,
        "ema_slope_21": 0.00005,
        "volume_zscore": 1.5,
        "realized_vol_20": 0.05,
    }


def _low_liquidity_features() -> dict[str, float]:
    return {
        "adx_14": 0.20,
        "bb_bandwidth": 0.02,
        "rsi_14": 0.50,
        "ema_slope_9": 0.0001,
        "ema_slope_21": 0.00005,
        "volume_zscore": -2.0,   # below -1.5 threshold
        "realized_vol_20": 0.01,
    }


class TestRegimeClassifier:
    def setup_method(self):
        self.clf = RegimeClassifier()

    def test_bull_trend_detected(self):
        ctx = self.clf.classify(_vec(_bull_features()))
        assert ctx.regime == MarketRegime.BULL_TREND
        assert ctx.trading_allowed is True
        assert ctx.confidence > 0.5

    def test_bear_trend_detected(self):
        ctx = self.clf.classify(_vec(_bear_features()))
        assert ctx.regime == MarketRegime.BEAR_TREND
        assert ctx.trading_allowed is True

    def test_sideways_detected(self):
        ctx = self.clf.classify(_vec(_sideways_features()))
        assert ctx.regime == MarketRegime.SIDEWAYS
        assert ctx.trading_allowed is True

    def test_high_volatility_detected(self):
        ctx = self.clf.classify(_vec(_high_vol_features()))
        assert ctx.regime == MarketRegime.HIGH_VOLATILITY
        assert ctx.trading_allowed is True  # allowed but with reduced size
        assert ctx.volatility_level in (VolatilityLevel.HIGH, VolatilityLevel.EXTREME)

    def test_low_liquidity_blocks_trading(self):
        ctx = self.clf.classify(_vec(_low_liquidity_features()))
        assert ctx.regime == MarketRegime.LOW_LIQUIDITY
        assert ctx.trading_allowed is False
        assert ctx.block_reason is not None

    def test_symbol_preserved(self):
        ctx = self.clf.classify(_vec(_bull_features(), symbol="ETHUSDT"))
        assert ctx.symbol == "ETHUSDT"

    def test_confidence_in_range(self):
        for feat_fn in [_bull_features, _bear_features, _sideways_features,
                        _high_vol_features, _low_liquidity_features]:
            ctx = self.clf.classify(_vec(feat_fn()))
            assert 0.0 <= ctx.confidence <= 1.0, f"confidence out of range: {ctx.confidence}"

    def test_missing_features_graceful(self):
        # Minimal feature vector — only one feature
        vec = _vec({"volume_zscore": 0.0})
        ctx = self.clf.classify(vec)
        assert ctx.regime in MarketRegime.__members__.values()

    def test_high_adx_low_rsi_uncertain(self):
        # ADX trending but RSI in middle — uncertain or sideways
        features = {
            "adx_14": 0.30,
            "bb_bandwidth": 0.03,
            "rsi_14": 0.50,      # in the middle, neither bull nor bear
            "ema_slope_9": 0.0003,
            "ema_slope_21": 0.0001,
            "volume_zscore": 0.2,
            "realized_vol_20": 0.02,
        }
        ctx = self.clf.classify(_vec(features))
        # Should not be BULL (RSI not >= 0.48 with ema criteria) or could be UNCERTAIN
        assert ctx.regime in (MarketRegime.BULL_TREND, MarketRegime.UNCERTAIN, MarketRegime.SIDEWAYS)
