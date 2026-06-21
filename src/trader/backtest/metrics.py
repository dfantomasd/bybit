"""Backtest performance metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceMetrics:
    total_trades: int
    win_rate: float
    net_pnl_pct: float
    avg_trade_pnl_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float


def _max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            dd = (peak - value) / peak * 100.0
            max_dd = max(max_dd, dd)
    return max_dd


def compute_metrics(trade_pnls_pct: list[float], *, initial_capital_pct: float = 100.0) -> PerformanceMetrics:
    """Compute standard risk/return metrics from per-trade net PnL in percent."""
    if not trade_pnls_pct:
        return PerformanceMetrics(
            total_trades=0,
            win_rate=0.0,
            net_pnl_pct=0.0,
            avg_trade_pnl_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            profit_factor=0.0,
        )

    wins = [p for p in trade_pnls_pct if p > 0]
    losses = [p for p in trade_pnls_pct if p <= 0]
    equity = [initial_capital_pct]
    for pnl in trade_pnls_pct:
        equity.append(equity[-1] * (1.0 + pnl / 100.0))

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    mean = sum(trade_pnls_pct) / len(trade_pnls_pct)
    if len(trade_pnls_pct) > 1:
        variance = sum((p - mean) ** 2 for p in trade_pnls_pct) / (len(trade_pnls_pct) - 1)
        std = math.sqrt(variance)
        sharpe = (mean / std) * math.sqrt(len(trade_pnls_pct)) if std > 0 else 0.0
    else:
        sharpe = 0.0

    return PerformanceMetrics(
        total_trades=len(trade_pnls_pct),
        win_rate=len(wins) / len(trade_pnls_pct) * 100.0,
        net_pnl_pct=equity[-1] - initial_capital_pct,
        avg_trade_pnl_pct=mean,
        max_drawdown_pct=_max_drawdown(equity),
        sharpe_ratio=sharpe,
        profit_factor=profit_factor,
    )
