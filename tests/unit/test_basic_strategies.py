"""Tests for simple, proven basic strategies."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from trader.domain.enums import OrderSide
from trader.domain.models import FeatureVector
from trader.strategies.basic_strategies import (
    ATRBreakoutStrategy,
    MACDZeroCrossStrategy,
    MeanReversionStrategy,
)

_SYMBOL = "TESTUSDT"
_PRICE = 100.0


def _vector(
    rsi: float = 0.50,
    atr_pct: float = 0.008,
    vol_z: float = 0.5,
    macd_hist: float = 0.0001,
    adx: float = 0.22,
    log_return: float = 0.0002,
) -> FeatureVector:
    names = ["rsi_14", "atr_14_pct", "volume_zscore", "macd_hist", "adx_14", "log_return_1"]
    values = [rsi, atr_pct, vol_z, macd_hist, adx, log_return]
    return FeatureVector(
        feature_id=uuid.uuid4(),
        symbol=_SYMBOL,
        timestamp=datetime.now(tz=UTC),
        values=values,
        feature_names=names,
        quality_score=1.0,
        lookback_bars=60,
    )


class TestMeanReversionStrategy:
    def test_buy_on_oversold_rsi(self) -> None:
        strat = MeanReversionStrategy()
        proposal = strat.evaluate(_vector(rsi=0.25), _PRICE, 1000.0)
        assert proposal is not None
        assert proposal.side == OrderSide.BUY

    def test_sell_on_overbought_rsi(self) -> None:
        strat = MeanReversionStrategy()
        proposal = strat.evaluate(_vector(rsi=0.75), _PRICE, 1000.0)
        assert proposal is not None
        assert proposal.side == OrderSide.SELL

    def test_no_signal_on_neutral_rsi(self) -> None:
        strat = MeanReversionStrategy()
        assert strat.evaluate(_vector(rsi=0.50), _PRICE, 1000.0) is None

    def test_reject_dead_market(self) -> None:
        strat = MeanReversionStrategy()
        assert strat.evaluate(_vector(rsi=0.25, vol_z=-2.0), _PRICE, 1000.0) is None

    def test_reject_extreme_atr(self) -> None:
        strat = MeanReversionStrategy()
        assert strat.evaluate(_vector(rsi=0.25, atr_pct=0.04), _PRICE, 1000.0) is None


class TestMACDZeroCrossStrategy:
    def test_buy_on_histogram_cross_negative_to_positive(self) -> None:
        strat = MACDZeroCrossStrategy()
        # First eval to set last_hist
        strat.evaluate(_vector(macd_hist=-0.0001), _PRICE, 1000.0)
        # Cross happens
        proposal = strat.evaluate(_vector(macd_hist=0.0001), _PRICE, 1000.0)
        assert proposal is not None
        assert proposal.side == OrderSide.BUY

    def test_sell_on_histogram_cross_positive_to_negative(self) -> None:
        strat = MACDZeroCrossStrategy()
        # First eval to set last_hist
        strat.evaluate(_vector(macd_hist=0.0001), _PRICE, 1000.0)
        # Cross happens
        proposal = strat.evaluate(_vector(macd_hist=-0.0001), _PRICE, 1000.0)
        assert proposal is not None
        assert proposal.side == OrderSide.SELL

    def test_no_signal_on_same_side(self) -> None:
        strat = MACDZeroCrossStrategy()
        strat.evaluate(_vector(macd_hist=0.0001), _PRICE, 1000.0)
        assert strat.evaluate(_vector(macd_hist=0.0002), _PRICE, 1000.0) is None

    def test_no_signal_on_first_eval(self) -> None:
        strat = MACDZeroCrossStrategy()
        assert strat.evaluate(_vector(macd_hist=0.0001), _PRICE, 1000.0) is None

    def test_cooldown_blocks_second_signal(self) -> None:
        strat = MACDZeroCrossStrategy()
        strat.evaluate(_vector(macd_hist=-0.0001), _PRICE, 1000.0)
        first = strat.evaluate(_vector(macd_hist=0.0001), _PRICE, 1000.0)
        assert first is not None
        # Immediate next eval should be blocked by cooldown
        second = strat.evaluate(_vector(macd_hist=-0.0001), _PRICE, 1000.0)
        assert second is None


class TestATRBreakoutStrategy:
    def test_buy_on_breakout_above_high(self) -> None:
        strat = ATRBreakoutStrategy()
        # Build price history
        for i in range(1, 6):
            strat.evaluate(_vector(vol_z=1.0), _PRICE - 0.01 + i * 0.002, 1000.0)
        # Now break above with positive return and volume
        proposal = strat.evaluate(_vector(vol_z=1.0, log_return=0.0005), _PRICE + 0.05, 1000.0)
        assert proposal is not None
        assert proposal.side == OrderSide.BUY

    def test_sell_on_breakout_below_low(self) -> None:
        strat = ATRBreakoutStrategy()
        # Build price history
        for i in range(1, 6):
            strat.evaluate(_vector(vol_z=1.0), _PRICE - i * 0.002, 1000.0)
        # Now break below with negative return and volume
        proposal = strat.evaluate(_vector(vol_z=1.0, log_return=-0.0005), _PRICE - 0.05, 1000.0)
        assert proposal is not None
        assert proposal.side == OrderSide.SELL

    def test_no_signal_inside_range(self) -> None:
        strat = ATRBreakoutStrategy()
        for _i in range(1, 6):
            strat.evaluate(_vector(vol_z=1.0), _PRICE, 1000.0)
        # Stay inside range = no signal
        assert strat.evaluate(_vector(vol_z=1.0), _PRICE, 1000.0) is None

    def test_reject_low_volume(self) -> None:
        strat = ATRBreakoutStrategy()
        for _i in range(1, 6):
            strat.evaluate(_vector(vol_z=1.0), _PRICE, 1000.0)
        proposal = strat.evaluate(_vector(vol_z=0.3, log_return=0.0005), _PRICE + 0.05, 1000.0)
        assert proposal is None  # vol_z < 0.5

    def test_reject_established_trend(self) -> None:
        strat = ATRBreakoutStrategy()
        for _i in range(1, 6):
            strat.evaluate(_vector(vol_z=1.0), _PRICE, 1000.0)
        proposal = strat.evaluate(_vector(vol_z=1.0, adx=0.40, log_return=0.0005), _PRICE + 0.05, 1000.0)
        assert proposal is None  # adx > 0.35
