"""Historical regime-specific performance analysis.

Analyzes which strategies perform best in which market regimes,
enabling regime-aware position sizing and exit optimization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RegimePerformance:
    """Performance metrics for a strategy in a specific regime."""

    strategy_id: str
    regime: str
    win_count: int = 0
    loss_count: int = 0
    total_return_bps: float = 0.0
    avg_return_bps: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0  # sum_wins / abs(sum_losses)
    sharpe_ratio: float = 0.0
    max_drawdown_bps: float = 0.0
    confidence: float = 0.0  # 0-1, based on sample size


@dataclass
class RegimePerfMatrix:
    """Performance matrix: strategy x regime."""

    performance_data: dict[tuple[str, str], RegimePerformance] = field(default_factory=dict)
    strategies: list[str] = field(default_factory=list)
    regimes: list[str] = field(default_factory=list)
    overall_best_strategies: dict[str, str] = field(default_factory=dict)  # {regime: best_strategy}


def build_regime_performance_matrix(
    trades_history: list[dict[str, Any]],
    min_samples_per_regime: int = 5,
) -> RegimePerfMatrix:
    """Build performance matrix from historical trades.

    Args:
        trades_history: List of {strategy_id, regime, return_bps, is_win, ...}
        min_samples_per_regime: Minimum trades needed to trust metrics

    Returns:
        RegimePerfMatrix with strategy/regime performance data.
    """

    performance = {}
    strategies = set()
    regimes = set()

    # Group trades by (strategy, regime)
    grouped = {}
    for trade in trades_history:
        strategy = trade.get("strategy_id", "unknown")
        regime = trade.get("regime", "unknown")
        return_bps = trade.get("return_bps", 0.0)

        strategies.add(strategy)
        regimes.add(regime)

        key = (strategy, regime)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(return_bps)

    # Calculate metrics per group
    for (strategy, regime), returns in grouped.items():
        if len(returns) < min_samples_per_regime:
            confidence = len(returns) / min_samples_per_regime
        else:
            confidence = 1.0

        wins = sum(1 for r in returns if r > 0)
        losses = sum(1 for r in returns if r < 0)
        total_return = sum(returns)
        avg_return = total_return / len(returns) if returns else 0.0

        win_rate = wins / len(returns) if returns else 0.0

        # Calculate profit factor
        win_sum = sum(r for r in returns if r > 0)
        loss_sum = abs(sum(r for r in returns if r < 0))
        profit_factor = (win_sum / loss_sum) if loss_sum > 0 else (1.0 if win_sum > 0 else 0.0)

        # Calculate Sharpe (simple version: mean / std)
        import numpy as np

        returns_array = np.array(returns)
        std_dev = float(np.std(returns_array))
        sharpe = (avg_return / std_dev) if std_dev > 0 else 0.0

        # Max drawdown (simple: largest drop from peak)
        cumulative = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        perf = RegimePerformance(
            strategy_id=strategy,
            regime=regime,
            win_count=wins,
            loss_count=losses,
            total_return_bps=total_return,
            avg_return_bps=avg_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown_bps=max_drawdown,
            confidence=confidence,
        )

        performance[(strategy, regime)] = perf

    # Find best strategy per regime
    best_by_regime = {}
    for regime in regimes:
        regime_perfs = [p for (s, r), p in performance.items() if r == regime and p.confidence >= 0.5]
        if regime_perfs:
            best = max(regime_perfs, key=lambda p: p.profit_factor)
            best_by_regime[regime] = best.strategy_id

    return RegimePerfMatrix(
        performance_data=performance,
        strategies=sorted(strategies),
        regimes=sorted(regimes),
        overall_best_strategies=best_by_regime,
    )


def get_optimal_strategy_for_regime(
    regime: str,
    perf_matrix: RegimePerfMatrix,
    metric: str = "profit_factor",
) -> tuple[str | None, float]:
    """Find the best-performing strategy for current regime.

    Args:
        regime: Current market regime
        perf_matrix: Performance matrix
        metric: Optimization metric (profit_factor, win_rate, sharpe_ratio, avg_return_bps)

    Returns:
        (best_strategy_id, metric_value)
    """

    regime_data = [p for (s, r), p in perf_matrix.performance_data.items() if r == regime and p.confidence >= 0.3]

    if not regime_data:
        return None, 0.0

    best = max(regime_data, key=lambda p: getattr(p, metric, 0.0))

    return best.strategy_id, getattr(best, metric, 0.0)


def calculate_dynamic_tp_sl(
    entry_price: float,
    strategy_id: str,
    regime: str,
    perf_matrix: RegimePerfMatrix,
    base_atr: float = 1.0,
    conservative: bool = False,
) -> dict[str, float]:
    """Calculate dynamic TP/SL based on historical regime performance.

    Args:
        entry_price: Entry price
        strategy_id: Current strategy
        regime: Current regime
        perf_matrix: Historical performance matrix
        base_atr: Base ATR multiplier for initial calculation
        conservative: Use conservative levels (smaller TP/SL)

    Returns:
        {tp_price, sl_price, tp_pct, sl_pct, profit_target_bps, stop_loss_bps}
    """

    key = (strategy_id, regime)
    perf = perf_matrix.performance_data.get(key)

    if not perf or perf.confidence < 0.3:
        # Fallback: use base ATR only
        sl_pct = 0.5 if conservative else 1.0  # 0.5-1% stop loss
        tp_pct = 1.0 if conservative else 2.0  # 1-2% profit target
    else:
        # Use historical win/loss to calculate levels
        # Typical win is (avg_return / win_count), loss is abs(total_loss / loss_count)
        avg_win_pct = (perf.avg_return_bps / 10000.0) if perf.win_rate > 0 else 0.5
        avg_loss_pct = 0.5 if perf.loss_count == 0 else 0.5  # Conservative

        tp_pct = max(0.3, avg_win_pct * (0.75 if conservative else 1.0))
        sl_pct = min(1.0, avg_loss_pct * (0.5 if conservative else 1.0))

    # Calculate prices
    tp_price = entry_price * (1 + tp_pct / 100.0)
    sl_price = entry_price * (1 - sl_pct / 100.0)

    return {
        "tp_price": tp_price,
        "sl_price": sl_price,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "profit_target_bps": tp_pct * 100,  # convert % to bps
        "stop_loss_bps": sl_pct * 100,
        "strategy_id": strategy_id,
        "regime": regime,
    }


def get_regime_weighted_sizing(
    strategy_id: str,
    regime: str,
    perf_matrix: RegimePerfMatrix,
    base_size: float = 1.0,
    max_multiplier: float = 1.5,
) -> float:
    """Get position size multiplier based on strategy/regime fit.

    Args:
        strategy_id: Strategy ID
        regime: Current regime
        perf_matrix: Performance matrix
        base_size: Base position size (1.0 = 100%)
        max_multiplier: Maximum size multiplier

    Returns:
        Size multiplier (0.5-1.5 typical).
    """

    key = (strategy_id, regime)
    perf = perf_matrix.performance_data.get(key)

    if not perf:
        return base_size * 0.8  # Reduce if no data

    if perf.confidence < 0.3:
        return base_size * 0.7  # Reduce if low confidence

    # Calculate multiplier from win_rate
    # Perfect win_rate (100%) = 1.5x, poor (40%) = 0.7x
    win_rate_multiplier = 0.7 + (perf.win_rate * 0.8)

    # Apply confidence as dampening
    final_multiplier = base_size * win_rate_multiplier * perf.confidence

    return min(max_multiplier, max(0.5, final_multiplier))


def should_trade_in_regime(
    strategy_id: str,
    regime: str,
    perf_matrix: RegimePerfMatrix,
    min_win_rate: float = 0.45,
    min_confidence: float = 0.3,
) -> dict[str, Any]:
    """Determine if a strategy should trade in current regime.

    Args:
        strategy_id: Strategy ID
        regime: Current regime
        perf_matrix: Performance matrix
        min_win_rate: Minimum win rate required
        min_confidence: Minimum confidence in data

    Returns:
        {should_trade, reason, win_rate, confidence}
    """

    key = (strategy_id, regime)
    perf = perf_matrix.performance_data.get(key)

    if not perf:
        return {
            "should_trade": False,
            "reason": "No historical data for this strategy/regime combination",
            "win_rate": 0.0,
            "confidence": 0.0,
        }

    if perf.confidence < min_confidence:
        return {
            "should_trade": False,
            "reason": f"Insufficient samples ({int(perf.win_count + perf.loss_count)} trades, need >{int(1 / min_confidence * 5)})",
            "win_rate": perf.win_rate,
            "confidence": perf.confidence,
        }

    if perf.win_rate < min_win_rate:
        return {
            "should_trade": False,
            "reason": f"Win rate {perf.win_rate:.1%} < minimum {min_win_rate:.1%}",
            "win_rate": perf.win_rate,
            "confidence": perf.confidence,
        }

    return {
        "should_trade": True,
        "reason": f"Good performance in {regime}: {perf.win_rate:.1%} win rate, profit factor {perf.profit_factor:.2f}",
        "win_rate": perf.win_rate,
        "confidence": perf.confidence,
        "profit_factor": perf.profit_factor,
    }
