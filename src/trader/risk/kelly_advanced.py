"""Advanced Kelly Criterion with drawdown analysis and dynamic fractional Kelly.

Improvements over basic Kelly:
1. Analyzes drawdown sequences and volatility clustering
2. Dynamic fractional Kelly (0.1x-0.5x) based on recent performance
3. Detects fat tails in return distribution (kurtosis)
4. Risk-Parity alternative for portfolio-level sizing
5. Worst-case scenario sizing (Value-at-Risk aware)
6. Equity curve annealing (reduces size during drawdown recovery)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DrawdownAnalysis:
    """Analysis of drawdown patterns and recovery."""

    current_drawdown_pct: float  # Current drawdown from peak (negative)
    max_drawdown_pct: float      # Maximum historical drawdown
    avg_drawdown_pct: float      # Average drawdown magnitude
    recovery_time_avg: int       # Average bars to recover from drawdown
    drawdown_frequency: float    # How often drawdowns occur (0-1)
    in_drawdown: bool            # Currently in drawdown
    severity: str                # "none", "mild", "moderate", "severe", "extreme"


@dataclass
class ReturnDistribution:
    """Statistical properties of returns."""

    mean_bps: float
    std_dev_bps: float
    skewness: float          # -1 to 1, negative = left tail
    kurtosis: float          # >3 = fat tails (black swans)
    var_95_bps: float        # Value-at-Risk (95% confidence)
    cvar_95_bps: float       # Conditional VaR (expected loss if breached)
    max_loss_bps: float      # Worst single trade
    has_fat_tails: bool      # kurtosis > 4


@dataclass
class AdvancedKellyResult:
    """Result from advanced Kelly calculation."""

    kelly_fraction: float        # Base Kelly (0-0.25)
    dynamic_fractional: float    # Adjusted for current conditions (0.05-0.5)
    recommended_size: float      # Final recommended position size
    kelly_adjusted_drawdown: float
    kelly_adjusted_distribution: float
    kelly_adjusted_recovery: float
    sizing_reason: str          # Explanation
    risk_warnings: list[str] = None


def analyze_drawdown_sequence(
    returns_bps: list[float],
    lookback: int = 100,
) -> DrawdownAnalysis:
    """Analyze drawdown patterns in returns.

    Args:
        returns_bps: Historical returns in basis points
        lookback: How many recent trades to analyze

    Returns:
        DrawdownAnalysis with patterns and severity.
    """

    if not returns_bps or len(returns_bps) < 10:
        return DrawdownAnalysis(
            current_drawdown_pct=0.0,
            max_drawdown_pct=0.0,
            avg_drawdown_pct=0.0,
            recovery_time_avg=0,
            drawdown_frequency=0.0,
            in_drawdown=False,
            severity="none",
        )

    recent = returns_bps[-lookback:] if len(returns_bps) > lookback else returns_bps
    cumulative = np.cumsum(recent)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative

    # Current state
    current_dd = float(drawdown[-1]) / 10000.0 if drawdown[-1] > 0 else 0.0
    max_dd = float(np.max(drawdown)) / 10000.0
    avg_dd = float(np.mean(drawdown[drawdown > 0])) / 10000.0 if len(drawdown[drawdown > 0]) > 0 else 0.0

    # Drawdown frequency and recovery time
    in_dd = drawdown[-1] > 0
    dd_events = np.where(np.diff(np.sign(drawdown)) != 0)[0]
    dd_frequency = len(dd_events) / len(drawdown) if len(drawdown) > 0 else 0.0

    # Recovery time (bars to recover from each drawdown)
    recovery_times = []
    for i in range(len(drawdown) - 1):
        if drawdown[i] > 0 and drawdown[i + 1] == 0:
            # Found end of drawdown
            recovery_times.append(1)
        elif drawdown[i] > 0 and drawdown[i + 1] < drawdown[i]:
            recovery_times.append(1)

    avg_recovery = int(np.mean(recovery_times)) if recovery_times else 0

    # Severity
    if max_dd < 0.01:
        severity = "none"
    elif max_dd < 0.03:
        severity = "mild"
    elif max_dd < 0.07:
        severity = "moderate"
    elif max_dd < 0.15:
        severity = "severe"
    else:
        severity = "extreme"

    return DrawdownAnalysis(
        current_drawdown_pct=current_dd * 100,
        max_drawdown_pct=max_dd * 100,
        avg_drawdown_pct=avg_dd * 100,
        recovery_time_avg=avg_recovery,
        drawdown_frequency=float(dd_frequency),
        in_drawdown=in_dd,
        severity=severity,
    )


def analyze_return_distribution(
    returns_bps: list[float],
    min_samples: int = 30,
) -> ReturnDistribution:
    """Analyze statistical properties of returns.

    Args:
        returns_bps: Historical returns in basis points
        min_samples: Minimum samples needed

    Returns:
        ReturnDistribution with statistical analysis.
    """

    if not returns_bps or len(returns_bps) < min_samples:
        return ReturnDistribution(
            mean_bps=0.0,
            std_dev_bps=0.0,
            skewness=0.0,
            kurtosis=3.0,  # normal distribution
            var_95_bps=0.0,
            cvar_95_bps=0.0,
            max_loss_bps=0.0,
            has_fat_tails=False,
        )

    returns = np.array(returns_bps)
    mean = float(np.mean(returns))
    std = float(np.std(returns))

    # Skewness and kurtosis
    skew = float((np.mean((returns - mean) ** 3) / (std ** 3))) if std > 0 else 0.0
    kurt = float((np.mean((returns - mean) ** 4) / (std ** 4))) if std > 0 else 3.0

    # Value-at-Risk (95% confidence)
    var_95 = float(np.percentile(returns, 5))  # 5th percentile (worst 5%)
    losses = returns[returns < 0]
    cvar_95 = float(np.mean(losses)) if len(losses) > 0 else 0.0

    max_loss = float(np.min(returns))

    has_tails = kurt > 4.0  # Normal dist = 3, >4 = fat tails

    return ReturnDistribution(
        mean_bps=mean,
        std_dev_bps=std,
        skewness=skew,
        kurtosis=kurt,
        var_95_bps=var_95,
        cvar_95_bps=cvar_95,
        max_loss_bps=max_loss,
        has_fat_tails=has_tails,
    )


def calculate_kelly_with_drawdown_adjustment(
    win_rate: float,
    avg_win_bps: float,
    avg_loss_bps: float,
    drawdown_analysis: DrawdownAnalysis,
) -> float:
    """Calculate Kelly adjusted for current drawdown severity.

    Args:
        win_rate: Probability of winning
        avg_win_bps: Average win
        avg_loss_bps: Average loss
        drawdown_analysis: Current drawdown state

    Returns:
        Kelly fraction adjusted for drawdown (0-0.25).
    """

    # Base Kelly
    kelly = (win_rate * avg_win_bps - (1 - win_rate) * abs(avg_loss_bps)) / avg_win_bps
    kelly = max(0.0, min(kelly, 0.25))

    # Adjust for drawdown severity
    dd_multiplier = {
        "none": 1.0,
        "mild": 0.95,
        "moderate": 0.85,
        "severe": 0.65,
        "extreme": 0.40,
    }.get(drawdown_analysis.severity, 0.8)

    # Additional penalty if in active drawdown
    if drawdown_analysis.in_drawdown:
        dd_multiplier *= 0.8  # Further 20% reduction

    adjusted_kelly = kelly * dd_multiplier

    return max(0.0, min(adjusted_kelly, 0.25))


def calculate_kelly_with_fat_tail_adjustment(
    win_rate: float,
    avg_win_bps: float,
    avg_loss_bps: float,
    distribution: ReturnDistribution,
) -> float:
    """Adjust Kelly for fat tails in distribution.

    Args:
        win_rate: Win rate
        avg_win_bps: Average win
        avg_loss_bps: Average loss
        distribution: Return distribution analysis

    Returns:
        Kelly adjusted for tail risk (0-0.25).
    """

    # Base Kelly
    kelly = (win_rate * avg_win_bps - (1 - win_rate) * abs(avg_loss_bps)) / avg_win_bps
    kelly = max(0.0, min(kelly, 0.25))

    if not distribution.has_fat_tails:
        return kelly

    # For fat tails, use VaR-based adjustment
    # If worst case (VaR_95) is worse than expected, reduce Kelly
    worst_case_loss = distribution.var_95_bps
    normal_expected_loss = distribution.mean_bps - 2 * distribution.std_dev_bps

    if abs(worst_case_loss) > abs(normal_expected_loss):
        # Actual tail is worse than normal distribution predicts
        tail_multiplier = normal_expected_loss / worst_case_loss if worst_case_loss < 0 else 0.7
        tail_multiplier = max(0.5, tail_multiplier)  # Never reduce more than 50%
    else:
        tail_multiplier = 1.0

    adjusted_kelly = kelly * tail_multiplier

    return max(0.0, min(adjusted_kelly, 0.25))


def calculate_dynamic_fractional_kelly(
    kelly_fraction: float,
    drawdown_analysis: DrawdownAnalysis,
    distribution: ReturnDistribution,
    recent_win_rate: float,  # Last N trades
    recent_equity_trend: float,  # Up or down trend
) -> float:
    """Calculate dynamic fractional Kelly (0.1x - 0.5x).

    Args:
        kelly_fraction: Base Kelly from statistics
        drawdown_analysis: Drawdown state
        distribution: Return distribution
        recent_win_rate: Win rate in last 20-50 trades
        recent_equity_trend: Recent trend (>0 = up, <0 = down)

    Returns:
        Dynamic fractional Kelly multiplier (0.1 - 0.5).
    """

    # Base fractional is 0.25 (conservative)
    base_fractional = 0.25

    # Adjust for recent performance
    if recent_win_rate > 0.6:
        base_fractional = 0.35  # Recent good performance = higher
    elif recent_win_rate < 0.45:
        base_fractional = 0.15  # Recent poor performance = lower

    # Adjust for equity curve
    if recent_equity_trend < -0.02:  # Declining equity
        base_fractional *= 0.7  # Reduce by 30%
    elif recent_equity_trend > 0.05:  # Strong up trend
        base_fractional *= 1.1  # Increase by 10%

    # Adjust for volatility (kurtosis)
    if distribution.has_fat_tails:
        base_fractional *= 0.8  # Reduce for fat tails

    # Adjust for drawdown recovery
    if drawdown_analysis.in_drawdown:
        # Reduce more as we recover
        recovery_pct = drawdown_analysis.current_drawdown_pct / max(
            drawdown_analysis.max_drawdown_pct, 0.001
        )
        base_fractional *= (1 - recovery_pct)  # Goes to 0 as we recover

    # Clamp to reasonable range
    return max(0.1, min(base_fractional, 0.5))


def calculate_risk_parity_kelly(
    strategy_stats_list: list[dict[str, Any]],
    target_volatility: float = 10.0,  # bps per trade
) -> dict[str, float]:
    """Calculate Risk-Parity Kelly sizing across strategies.

    Instead of Kelly (maximize growth), target equal risk contribution
    from each strategy.

    Args:
        strategy_stats_list: [{win_rate, avg_win, avg_loss, std_dev, ...}, ...]
        target_volatility: Target volatility per trade in bps

    Returns:
        {strategy_id: kelly_fraction} for each strategy.
    """

    if not strategy_stats_list:
        return {}

    # Calculate expected return and volatility for each
    strategy_metrics = []
    for stats in strategy_stats_list:
        win_rate = stats.get("win_rate", 0.5)
        avg_win = stats.get("avg_win_bps", 10.0)
        avg_loss = stats.get("avg_loss_bps", -10.0)
        std_dev = stats.get("std_dev_bps", 15.0)

        # Expected return per trade
        exp_return = win_rate * avg_win + (1 - win_rate) * avg_loss

        strategy_metrics.append({
            "strategy_id": stats.get("strategy_id", "unknown"),
            "exp_return": exp_return,
            "volatility": std_dev,
            "sharpe": exp_return / std_dev if std_dev > 0 else 0,
        })

    # Risk parity: allocate based on inverse volatility
    total_inv_vol = sum(1.0 / max(0.1, m["volatility"]) for m in strategy_metrics)

    result = {}
    for metric in strategy_metrics:
        weight = (1.0 / max(0.1, metric["volatility"])) / total_inv_vol
        kelly = weight * (target_volatility / max(0.1, metric["volatility"]))
        kelly = max(0.01, min(kelly, 0.25))
        result[metric["strategy_id"]] = kelly

    return result


def calculate_worst_case_kelly(
    kelly_fraction: float,
    distribution: ReturnDistribution,
    max_acceptable_loss_pct: float = 2.0,
) -> float:
    """Adjust Kelly to ensure maximum loss doesn't exceed threshold.

    Args:
        kelly_fraction: Base Kelly
        distribution: Return distribution with VaR
        max_acceptable_loss_pct: Maximum acceptable loss as % of capital

    Returns:
        Kelly adjusted for worst case (0-0.25).
    """

    if distribution.var_95_bps <= 0:
        return kelly_fraction

    # Worst case loss at this Kelly level
    worst_case_loss_pct = (distribution.var_95_bps / 10000.0) * kelly_fraction

    if abs(worst_case_loss_pct) > max_acceptable_loss_pct / 100.0:
        # Scale back Kelly
        kelly_adjusted = kelly_fraction * (max_acceptable_loss_pct / 100.0) / abs(worst_case_loss_pct)
        return max(0.0, min(kelly_adjusted, 0.25))

    return kelly_fraction


def calculate_advanced_kelly(
    win_rate: float,
    avg_win_bps: float,
    avg_loss_bps: float,
    returns_bps: list[float],
    recent_returns_bps: list[float] | None = None,
    recent_equity_trend: float = 0.0,
) -> AdvancedKellyResult:
    """Complete advanced Kelly calculation.

    Args:
        win_rate: Overall win rate
        avg_win_bps: Average win
        avg_loss_bps: Average loss
        returns_bps: Full historical returns
        recent_returns_bps: Recent trades for trend analysis
        recent_equity_trend: Recent equity trend (-1 to 1)

    Returns:
        AdvancedKellyResult with detailed breakdown.
    """

    # Analyze conditions
    drawdown = analyze_drawdown_sequence(returns_bps)
    distribution = analyze_return_distribution(returns_bps)

    # Calculate adjusted Kelly values
    kelly_base = (win_rate * avg_win_bps - (1 - win_rate) * abs(avg_loss_bps)) / avg_win_bps
    kelly_base = max(0.0, min(kelly_base, 0.25))

    kelly_dd = calculate_kelly_with_drawdown_adjustment(
        win_rate, avg_win_bps, avg_loss_bps, drawdown
    )

    kelly_dist = calculate_kelly_with_fat_tail_adjustment(
        win_rate, avg_win_bps, avg_loss_bps, distribution
    )

    kelly_worst = calculate_worst_case_kelly(kelly_base, distribution)

    # Use the most conservative
    kelly_final = min(kelly_dd, kelly_dist, kelly_worst)

    # Calculate recent win rate for fractional
    recent_wr = win_rate
    if recent_returns_bps:
        recent_wins = sum(1 for r in recent_returns_bps if r > 0)
        recent_wr = recent_wins / len(recent_returns_bps) if recent_returns_bps else win_rate

    # Dynamic fractional
    fractional = calculate_dynamic_fractional_kelly(
        kelly_final, drawdown, distribution, recent_wr, recent_equity_trend
    )

    final_size = kelly_final * fractional

    # Build reason
    reasons = []
    if drawdown.severity != "none":
        reasons.append(f"Drawdown {drawdown.severity}: {drawdown.current_drawdown_pct:.1f}%")
    if distribution.has_fat_tails:
        reasons.append(f"Fat tails detected (kurtosis={distribution.kurtosis:.1f})")
    if recent_wr < 0.5:
        reasons.append(f"Recent win rate low: {recent_wr:.1%}")

    warnings = []
    if kelly_base > 0.25:
        warnings.append("Base Kelly >25% (capped for safety)")
    if distribution.cvar_95_bps < -100:
        warnings.append("Conditional VaR severe (tail risk)")

    return AdvancedKellyResult(
        kelly_fraction=kelly_final,
        dynamic_fractional=fractional,
        recommended_size=final_size,
        kelly_adjusted_drawdown=kelly_dd,
        kelly_adjusted_distribution=kelly_dist,
        kelly_adjusted_recovery=kelly_worst,
        sizing_reason="; ".join(reasons) if reasons else "Normal conditions",
        risk_warnings=warnings,
    )
