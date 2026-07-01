"""Kelly Criterion for optimal position sizing and adaptive risk management.

Kelly Criterion formula: f* = (bp - q) / b
where:
  f* = optimal fraction of capital to risk
  b = ratio of win amount to loss amount
  p = probability of winning
  q = probability of losing (1 - p)

Practical adjustments:
  - Fractional Kelly (0.25x) for conservative trading
  - Maximum leverage cap to prevent ruin
  - Per-strategy and regime-specific calculations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class StrategyStats:
    """Historical performance metrics for a strategy."""

    strategy_id: str
    win_count: int
    loss_count: int
    total_win_bps: float  # sum of positive returns in basis points
    total_loss_bps: float  # sum of negative returns in basis points (stored as negative)
    win_rate: float  # probability of win
    avg_win_bps: float
    avg_loss_bps: float
    profit_factor: float  # (total wins) / abs(total losses)


def calculate_kelly_fraction(
    win_rate: float,
    avg_win_bps: float,
    avg_loss_bps: float,
    min_samples: int = 20,
) -> float:
    """Calculate Kelly fraction from historical win/loss statistics.

    Args:
        win_rate: Probability of winning (0-1)
        avg_win_bps: Average win in basis points
        avg_loss_bps: Average loss in basis points (as negative value)
        min_samples: Minimum trades needed to trust the calculation

    Returns:
        Kelly fraction (0-1), or conservative 0.02 if insufficient data.

    Notes:
        - Returns 0 if strategy is expected to lose money
        - Capped at 0.25 (fractional Kelly) for safety
        - Returns conservative 0.02 if win_rate <= 0 or avg_win_bps <= 0
    """

    # Guard: insufficient data
    if win_rate < 0 or win_rate > 1 or avg_win_bps <= 0:
        logger.warning("kelly.calculate: invalid inputs win_rate=%s avg_win=%s", win_rate, avg_win_bps)
        return 0.02  # conservative default

    # Guard: strategy loses money on average
    expected_value = win_rate * avg_win_bps + (1 - win_rate) * avg_loss_bps
    if expected_value <= 0:
        logger.info("kelly.calculate: negative expected value, returning 0")
        return 0.0

    # Kelly formula: f* = (bp - q) / b
    # Rearranged: f* = (p * avg_win - (1-p) * abs(avg_loss)) / avg_win
    abs_avg_loss = abs(avg_loss_bps)

    kelly_fraction = (win_rate * avg_win_bps - (1 - win_rate) * abs_avg_loss) / avg_win_bps

    # Cap at 0.25 (fractional Kelly) for safety
    kelly_fraction = max(0.0, min(kelly_fraction, 0.25))

    return kelly_fraction


def calculate_adaptive_kelly(
    strategy_stats: StrategyStats,
    regime: str = "unknown",
    confidence: float = 1.0,
    fractional: float = 0.25,
) -> float:
    """Calculate regime-specific adaptive Kelly sizing.

    Args:
        strategy_stats: Historical stats for the strategy
        regime: Current market regime (SIDEWAYS, HIGH_VOLATILITY, TRENDING, etc.)
        confidence: How confident we are in current stats (0-1)
        fractional: Fractional Kelly multiplier (0-1), default 0.25 for safety

    Returns:
        Adaptive Kelly fraction adjusted for regime and confidence.
    """

    total_trades = strategy_stats.win_count + strategy_stats.loss_count
    if total_trades < 20:
        # Insufficient data - use very conservative sizing
        return 0.01

    # Base Kelly calculation
    kelly = calculate_kelly_fraction(
        win_rate=strategy_stats.win_rate,
        avg_win_bps=strategy_stats.avg_win_bps,
        avg_loss_bps=strategy_stats.avg_loss_bps,
        min_samples=20,
    )

    # Regime adjustment: reduce sizing in unfavorable regimes
    regime_multiplier = {
        "HIGH_VOLATILITY": 0.7,  # reduce by 30%
        "SIDEWAYS": 0.6,  # reduce by 40% (mean reversion regime)
        "UNCERTAIN": 0.5,  # reduce by 50%
        "TRENDING": 1.0,  # no adjustment
    }.get(regime, 0.8)  # default: reduce by 20%

    # Confidence adjustment: reduce if we're uncertain
    kelly = kelly * regime_multiplier * confidence

    # Apply fractional Kelly for additional safety
    kelly = kelly * fractional

    return max(0.0, min(kelly, 0.15))  # Cap at 15% for safety


def calculate_portfolio_kelly(
    strategy_stats_list: list[StrategyStats],
    weights: dict[str, float] | None = None,
) -> float:
    """Calculate portfolio-level Kelly sizing across multiple strategies.

    Args:
        strategy_stats_list: List of stats for each strategy
        weights: Optional custom weights per strategy (sum should be 1.0)

    Returns:
        Weighted Kelly fraction for portfolio.
    """

    if not strategy_stats_list:
        return 0.02

    total_kelly = 0.0

    for stats in strategy_stats_list:
        kelly = calculate_kelly_fraction(
            win_rate=stats.win_rate,
            avg_win_bps=stats.avg_win_bps,
            avg_loss_bps=stats.avg_loss_bps,
        )

        # Use custom weight if provided, otherwise equal weight
        weight = (
            weights.get(stats.strategy_id, 1.0 / len(strategy_stats_list))
            if weights
            else (1.0 / len(strategy_stats_list))
        )
        total_kelly += kelly * weight

    # Cap portfolio Kelly at 0.20 for safety
    return max(0.0, min(total_kelly, 0.20))


def calculate_position_size_kelly(
    capital_usd: Decimal,
    kelly_fraction: float,
    entry_price: Decimal,
    stop_price: Decimal,
    risk_per_trade_pct: float = 2.0,
) -> Decimal:
    """Calculate position size using Kelly Criterion.

    Args:
        capital_usd: Total capital available
        kelly_fraction: Kelly fraction (0-1)
        entry_price: Entry price
        stop_price: Stop loss price
        risk_per_trade_pct: Maximum risk per trade as % of capital (default 2%)

    Returns:
        Position size in contracts/units.
    """

    if entry_price <= 0 or stop_price <= 0:
        return Decimal("0")

    kelly_d = Decimal(str(kelly_fraction))
    risk_per_trade_d = Decimal(str(risk_per_trade_pct))

    # Risk amount from Kelly
    risk_amount = capital_usd * kelly_d

    # Cap at risk_per_trade_pct of capital
    max_risk = capital_usd * risk_per_trade_d / Decimal("100")
    risk_amount = min(risk_amount, max_risk)

    # Calculate position size
    price_diff = abs(entry_price - stop_price)
    if price_diff <= Decimal("0"):
        return Decimal("0")

    position_size = risk_amount / price_diff

    return max(Decimal("0"), position_size)
