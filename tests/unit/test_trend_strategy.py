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
            "return_3",
            "return_5",
            "volume_zscore",
            "atr_14_pct",
            "adx_14",
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
            [-0.001, -0.002, 0.000414, 0.599, 0.000122, 0.002, 0.003, 0.1, 0.002, 0.30],
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
            [0.001, 0.002, -0.000414, 0.36, -0.0017, -0.002, -0.003, 0.1, 0.002, 0.30],
        ),
        current_price=1.2345,
        available_balance_usd=23.52,
    )

    assert proposal is not None
    assert proposal.side == OrderSide.SELL
    assert proposal.stop_loss is not None
    assert proposal.entry_price is not None
    assert proposal.stop_loss > proposal.entry_price


def test_low_adx_rejected() -> None:
    strategy = EMAcrossoverStrategy()
    proposal = strategy.evaluate(
        _features(
            "DOGEUSDT",
            [-0.001, -0.002, 0.000414, 0.599, 0.000122, 0.002, 0.003, 0.1, 0.002, 0.15],
        ),
        current_price=0.14235,
        available_balance_usd=23.52,
    )

    assert proposal is None


def test_long_rejects_negative_net_edge_after_costs() -> None:
    strategy = EMAcrossoverStrategy(
        taker_fee_pct=0.055,
        expected_slippage_pct=0.03,
        max_spread_bps=30.0,
        min_net_return_pct=0.05,
    )
    proposal = strategy.evaluate(
        _features(
            "DOGEUSDT",
            [-0.001, -0.002, 0.000414, 0.599, 0.000122, 0.002, 0.003, 0.1, 0.001, 0.30],
        ),
        current_price=0.14235,
        available_balance_usd=23.52,
    )

    assert proposal is None


def test_trend_net_edge_charges_round_trip_slippage() -> None:
    strategy = EMAcrossoverStrategy(
        taker_fee_pct=0.0,
        expected_slippage_pct=0.03,
        max_spread_bps=0.0,
        min_net_return_pct=0.05,
    )

    # ATR 0.00029 gives TP distance 0.116%. After 0.06% round-trip slippage
    # and 0.01% safety, net edge is 0.046%; with one-sided slippage it would
    # incorrectly pass at 0.076%.
    proposal = strategy.evaluate(
        _features(
            "DOGEUSDT",
            [-0.001, -0.002, 0.000414, 0.599, 0.000122, 0.002, 0.003, 0.1, 0.00029, 0.30],
        ),
        current_price=0.14235,
        available_balance_usd=23.52,
    )

    assert proposal is None


def test_long_rejects_price_below_fast_ema() -> None:
    strategy = EMAcrossoverStrategy()
    proposal = strategy.evaluate(
        _features(
            "XRPUSDT",
            [0.002, 0.001, 0.000414, 0.59, 0.0002, 0.002, 0.003, 0.1, 0.002, 0.30],
        ),
        current_price=1.23,
        available_balance_usd=23.52,
    )

    assert proposal is None


def test_weak_macd_hist_rejected_as_noise() -> None:
    strategy = EMAcrossoverStrategy()
    proposal = strategy.evaluate(
        _features(
            "XRPUSDT",
            [-0.001, -0.002, 0.000414, 0.59, 0.000081, 0.002, 0.003, 0.1, 0.002, 0.30],
        ),
        current_price=1.23,
        available_balance_usd=23.52,
    )

    assert proposal is None


def test_short_rejects_price_above_fast_ema() -> None:
    strategy = EMAcrossoverStrategy()
    proposal = strategy.evaluate(
        _features(
            "XRPUSDT",
            [-0.002, -0.001, -0.000414, 0.40, -0.0002, -0.002, -0.003, 0.1, 0.002, 0.30],
        ),
        current_price=1.23,
        available_balance_usd=23.52,
    )

    assert proposal is None
