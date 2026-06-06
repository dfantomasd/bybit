"""Tests for EMA crossover strategy edge cases."""

from __future__ import annotations

from trader.domain.enums import OrderSide
from trader.domain.models import FeatureVector
from trader.strategies.trend import EMAcrossoverStrategy


def _features(symbol: str, values: list[float]) -> FeatureVector:
    return FeatureVector(
        symbol=symbol,
        feature_names=[
            "ema_9",
            "ema_21",
            "ema_slope_9",
            "rsi_14",
            "macd_hist",
            "volume_zscore",
            "atr_14_pct",
        ],
        values=values,
        quality_score=0.95,
        lookback_bars=100,
    )


def test_cheap_long_keeps_stop_below_entry() -> None:
    strategy = EMAcrossoverStrategy()
    proposal = strategy.evaluate(
        _features(
            "DOGEUSDT",
            [0.002, 0.001, 0.000414, 0.599, 0.000122, 0.1, 0.002],
        ),
        current_price=0.14235,
        available_balance_usd=23.52,
    )

    assert proposal is not None
    assert proposal.side == OrderSide.BUY
    assert proposal.stop_loss is not None
    assert proposal.entry_price is not None
    assert proposal.stop_loss < proposal.entry_price


def test_cheap_short_keeps_stop_above_entry() -> None:
    strategy = EMAcrossoverStrategy()
    proposal = strategy.evaluate(
        _features(
            "WLDUSDT",
            [-0.002, -0.001, -0.000414, 0.36, -0.0017, 0.1, 0.002],
        ),
        current_price=1.2345,
        available_balance_usd=23.52,
    )

    assert proposal is not None
    assert proposal.side == OrderSide.SELL
    assert proposal.stop_loss is not None
    assert proposal.entry_price is not None
    assert proposal.stop_loss > proposal.entry_price
