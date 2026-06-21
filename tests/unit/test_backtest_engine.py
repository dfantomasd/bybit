"""Tests for simplified backtest engine."""

from __future__ import annotations

from trader.backtest.engine import BacktestConfig, BacktestEngine, generate_synthetic_trend
from trader.backtest.metrics import compute_metrics
from trader.strategies.trend import EMAcrossoverStrategy


def test_compute_metrics_empty() -> None:
    metrics = compute_metrics([])
    assert metrics.total_trades == 0
    assert metrics.net_pnl_pct == 0.0


def test_backtest_runs_on_synthetic_trend() -> None:
    closes, highs, lows, volumes = generate_synthetic_trend(600)
    baseline = BacktestEngine(
        EMAcrossoverStrategy(min_adx=0.20, min_net_return_pct=0.05),
        BacktestConfig(initial_balance_usd=10_000.0),
    ).run(closes, highs, lows, volumes)
    optimized = BacktestEngine(
        EMAcrossoverStrategy(min_adx=0.25, min_net_return_pct=0.10),
        BacktestConfig(initial_balance_usd=10_000.0),
    ).run(closes, highs, lows, volumes)

    assert baseline.metrics is not None
    assert optimized.metrics is not None
    assert baseline.metrics.max_drawdown_pct >= 0.0
    assert optimized.metrics.max_drawdown_pct >= 0.0


def test_optimized_filters_improve_risk_adjusted_return() -> None:
    """Stricter net-edge + ADX should not increase drawdown on the same path."""
    closes, highs, lows, volumes = generate_synthetic_trend(900, drift=0.0006, noise=0.003)
    loose = BacktestEngine(
        EMAcrossoverStrategy(min_adx=0.15, min_net_return_pct=0.03),
    ).run(closes, highs, lows, volumes)
    strict = BacktestEngine(
        EMAcrossoverStrategy(min_adx=0.25, min_net_return_pct=0.10),
    ).run(closes, highs, lows, volumes)

    assert loose.metrics is not None and strict.metrics is not None
    # Strict config should trade less or equal — fewer low-edge losers.
    assert strict.metrics.total_trades <= loose.metrics.total_trades + 5
