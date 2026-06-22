"""Tests for candlestick pattern feature scores."""

from __future__ import annotations

from dataclasses import dataclass

from trader.features.candle_patterns import (
    PATTERN_LOOKBACK,
    compute_pattern_features,
    score_engulfing,
    score_hammer,
    score_morning_star,
    zero_pattern_features,
)


@dataclass
class Bar:
    open: float
    high: float
    low: float
    close: float


def test_zero_pattern_features_has_stable_keys() -> None:
    keys = set(zero_pattern_features(prefix="pat5_").keys())
    assert "pat5_hammer" in keys
    assert "pat5_data_present" in keys
    assert zero_pattern_features(prefix="pat5_")["pat5_data_present"] == 0.0


def test_hammer_scores_high_on_long_lower_wick() -> None:
    bar = Bar(open=100.0, high=101.0, low=90.0, close=100.5)
    assert score_hammer(bar) >= 0.5


def test_engulfing_bull_scores_on_clear_pattern() -> None:
    bars = [
        Bar(open=105.0, high=106.0, low=100.0, close=101.0),
        Bar(open=100.0, high=108.0, low=99.0, close=107.0),
    ]
    assert score_engulfing(bars, bullish=True) > 0.3


def test_morning_star_needs_three_bars() -> None:
    bars = [
        Bar(open=110.0, high=111.0, low=105.0, close=106.0),
        Bar(open=106.0, high=106.5, low=105.5, close=106.0),
        Bar(open=106.5, high=112.0, low=106.0, close=111.5),
    ]
    assert score_morning_star(bars) > 0.4


def test_compute_pattern_features_respects_lookback() -> None:
    bars = [Bar(open=100.0 + i, high=101.0 + i, low=99.0 + i, close=100.5 + i) for i in range(PATTERN_LOOKBACK + 5)]
    out = compute_pattern_features(bars, prefix="pat15_")
    assert out["pat15_data_present"] == 1.0
    assert 0.0 <= out["pat15_hammer"] <= 1.0
