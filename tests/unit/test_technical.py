"""Tests for technical indicators."""

from __future__ import annotations

import math

import pytest

from trader.features.technical import (
    atr,
    bb_percent_b,
    bollinger_bands,
    candle_body_ratio,
    ema,
    ema_slope,
    ema_value,
    log_return,
    macd,
    realized_volatility,
    returns,
    rsi,
    sma,
    volume_zscore,
)


def _prices(n: int = 50, start: float = 100.0, step: float = 1.0) -> list[float]:
    """Monotonically increasing price series."""
    return [start + i * step for i in range(n)]


def _flat(n: int, val: float = 100.0) -> list[float]:
    return [val] * n


def _sine(n: int = 100, amplitude: float = 10.0, base: float = 100.0) -> list[float]:
    """Sine-wave prices for oscillator testing."""
    return [base + amplitude * math.sin(2 * math.pi * i / 20) for i in range(n)]


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


class TestReturns:
    def test_simple_return_positive(self):
        closes = [100.0, 110.0]
        assert returns(closes, 1) == pytest.approx(0.1)

    def test_simple_return_negative(self):
        closes = [100.0, 90.0]
        assert returns(closes, 1) == pytest.approx(-0.1)

    def test_insufficient_data(self):
        assert returns([100.0], 1) is None

    def test_period_2(self):
        closes = [100.0, 110.0, 121.0]
        assert returns(closes, 2) == pytest.approx(0.21)

    def test_log_return(self):
        val = log_return([100.0, math.e * 100.0], 1)
        assert val == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# EMA & SMA
# ---------------------------------------------------------------------------


class TestMovingAverages:
    def test_ema_length(self):
        series = list(range(1, 21))
        result = ema(series, 5)
        assert len(result) == 16  # len - period + 1

    def test_ema_first_value_is_sma(self):
        series = [10.0] * 10
        result = ema(series, 5)
        assert result[0] == pytest.approx(10.0)

    def test_ema_constant_series(self):
        series = _flat(30, 50.0)
        result = ema(series, 10)
        for v in result:
            assert v == pytest.approx(50.0)

    def test_sma_basic(self):
        assert sma([1.0, 2.0, 3.0, 4.0, 5.0], 3) == pytest.approx(4.0)

    def test_sma_insufficient(self):
        assert sma([1.0, 2.0], 5) is None

    def test_ema_value(self):
        series = _flat(20, 100.0)
        assert ema_value(series, 10) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


class TestRSI:
    def test_rsi_flat_returns_50_ish(self):
        # Alternating +1 / -1 should give ~50
        closes = []
        v = 100.0
        for i in range(30):
            v += 1 if i % 2 == 0 else -1
            closes.append(v)
        r = rsi(closes, 14)
        assert r is not None
        assert 40 < r < 60

    def test_rsi_all_up_is_100(self):
        closes = _prices(30, 100, 1.0)
        r = rsi(closes, 14)
        assert r is not None
        assert r > 90  # very strong uptrend → close to 100

    def test_rsi_all_down_is_near_0(self):
        closes = _prices(30, 130, -1.0)
        r = rsi(closes, 14)
        assert r is not None
        assert r < 10

    def test_rsi_insufficient(self):
        assert rsi([100.0, 101.0], 14) is None

    def test_rsi_range(self):
        closes = _sine(50)
        r = rsi(closes, 14)
        assert r is not None
        assert 0 <= r <= 100


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


class TestMACD:
    def test_macd_returns_tuple(self):
        closes = _prices(50, 100, 0.5)
        result = macd(closes)
        assert result is not None
        m, s, h = result
        assert h == pytest.approx(m - s)

    def test_macd_insufficient(self):
        closes = _prices(30, 100, 1.0)
        assert macd(closes) is None  # need 26 + 9 = 35

    def test_macd_hist_sign(self):
        # Strong uptrend: MACD line should be above signal → histogram ≥ 0
        # Use a larger step to generate a clear positive histogram
        closes = _prices(60, 100, 5.0)
        result = macd(closes)
        assert result is not None
        assert result[2] >= -1e-10  # histogram non-negative in uptrend


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


class TestBollingerBands:
    def test_upper_middle_lower_ordering(self):
        closes = _sine(40)
        result = bollinger_bands(closes, 20, 2.0)
        assert result is not None
        upper, middle, lower = result
        assert upper > middle > lower

    def test_flat_series_middle_equals_price(self):
        closes = _flat(30, 100.0)
        result = bollinger_bands(closes, 20)
        assert result is not None
        assert result[1] == pytest.approx(100.0)

    def test_bb_pct_b_range(self):
        closes = _sine(40)
        val = bb_percent_b(closes)
        assert val is not None  # may be outside [0,1] for extreme moves

    def test_bb_insufficient(self):
        assert bollinger_bands([100.0] * 10, 20) is None


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


class TestATR:
    def _ohlcv(self, n: int = 20) -> tuple[list[float], list[float], list[float]]:
        closes = _prices(n, 100, 1.0)
        highs = [c + 2 for c in closes]
        lows = [c - 2 for c in closes]
        return highs, lows, closes

    def test_atr_positive(self):
        h, lo, c = self._ohlcv(20)
        val = atr(h, lo, c, 14)
        assert val is not None
        assert val > 0

    def test_atr_constant_bars(self):
        # With constant prices and fixed HL range, ATR ≈ high - low
        n = 20
        highs = [105.0] * n
        lows = [95.0] * n
        closes = [100.0] * n
        val = atr(highs, lows, closes, 14)
        assert val is not None
        assert val == pytest.approx(10.0, rel=0.01)

    def test_atr_insufficient(self):
        h, lo, c = self._ohlcv(10)
        assert atr(h, lo, c, 14) is None


# ---------------------------------------------------------------------------
# Volume Z-score
# ---------------------------------------------------------------------------


class TestVolumeZscore:
    def test_zero_for_mean_volume(self):
        vols = _flat(25, 1000.0)
        assert volume_zscore(vols, 20) == pytest.approx(0.0)

    def test_positive_for_high_volume(self):
        vols = [1000.0] * 20 + [2000.0]  # last is 2x mean
        val = volume_zscore(vols, 20)
        assert val is not None
        assert val > 1.0

    def test_insufficient(self):
        assert volume_zscore([1000.0] * 5, 20) is None


# ---------------------------------------------------------------------------
# Candle patterns
# ---------------------------------------------------------------------------


class TestCandlePatterns:
    def test_doji(self):
        # Open == close → body ratio = 0
        ratio = candle_body_ratio(100.0, 105.0, 95.0, 100.0)
        assert ratio == pytest.approx(0.0)

    def test_full_body_candle(self):
        # Open at low, close at high → body ratio = 1
        ratio = candle_body_ratio(95.0, 105.0, 95.0, 105.0)
        assert ratio == pytest.approx(1.0)

    def test_partial_body(self):
        ratio = candle_body_ratio(100.0, 110.0, 90.0, 105.0)
        assert 0 < ratio < 1


# ---------------------------------------------------------------------------
# EMA slope
# ---------------------------------------------------------------------------


class TestEMASlope:
    def test_uptrend_positive_slope(self):
        closes = _prices(30, 100, 1.0)  # monotonically rising
        slope = ema_slope(closes, 9, 3)
        assert slope is not None
        assert slope > 0

    def test_downtrend_negative_slope(self):
        closes = _prices(30, 130, -1.0)
        slope = ema_slope(closes, 9, 3)
        assert slope is not None
        assert slope < 0

    def test_insufficient_data(self):
        closes = [100.0] * 5
        assert ema_slope(closes, 9, 3) is None


# ---------------------------------------------------------------------------
# Realised volatility
# ---------------------------------------------------------------------------


class TestRealisedVol:
    def test_flat_series_zero_vol(self):
        closes = _flat(30, 100.0)
        val = realized_volatility(closes, 20)
        assert val is not None
        assert val == pytest.approx(0.0)

    def test_positive_for_volatile_series(self):
        closes = _sine(30)
        val = realized_volatility(closes, 20)
        assert val is not None
        assert val > 0
