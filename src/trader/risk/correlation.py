"""Cross-strategy correlation analysis and portfolio exposure management.

Analyzes correlations between open positions across strategies to prevent
concentrated risk and ensure healthy portfolio diversification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CorrelationMatrix:
    """Pairwise correlation between strategies."""

    strategies: list[str]
    correlations: dict[tuple[str, str], float]  # {(strat1, strat2): correlation}
    average_correlation: float
    max_correlation: float


def calculate_strategy_correlations(
    returns_history: dict[str, list[float]],
    min_samples: int = 20,
) -> CorrelationMatrix:
    """Calculate correlation matrix between strategy returns.

    Args:
        returns_history: {strategy_id: [returns_bps, ...]}
        min_samples: Minimum history needed to calculate

    Returns:
        CorrelationMatrix with pairwise correlations.
    """

    strategies = list(returns_history.keys())

    # Validate we have enough data
    valid_strategies = [s for s in strategies if returns_history[s] and len(returns_history[s]) >= min_samples]

    if len(valid_strategies) < 2:
        return CorrelationMatrix(
            strategies=valid_strategies,
            correlations={},
            average_correlation=0.0,
            max_correlation=0.0,
        )

    # Prepare data
    data_dict = {s: np.array(returns_history[s][-500:]) for s in valid_strategies}  # last 500 trades

    correlations = {}
    all_corrs = []

    for i, strat1 in enumerate(valid_strategies):
        for strat2 in valid_strategies[i + 1 :]:
            # Align arrays to same length
            arr1 = data_dict[strat1]
            arr2 = data_dict[strat2]
            min_len = min(len(arr1), len(arr2))
            arr1 = arr1[-min_len:]
            arr2 = arr2[-min_len:]

            # Calculate correlation
            if len(arr1) >= min_samples and len(arr2) >= min_samples:
                corr = float(np.corrcoef(arr1, arr2)[0, 1])
                if not np.isnan(corr):
                    correlations[tuple(sorted((strat1, strat2)))] = corr
                    all_corrs.append(corr)

    avg_corr = np.mean(all_corrs) if all_corrs else 0.0
    max_corr = max(all_corrs) if all_corrs else 0.0

    return CorrelationMatrix(
        strategies=valid_strategies,
        correlations=correlations,
        average_correlation=float(avg_corr),
        max_correlation=float(max_corr),
    )


def assess_position_concentration(
    open_positions: dict[str, dict[str, Any]],
    correlation_matrix: CorrelationMatrix,
    max_correlated_positions: int = 2,
) -> dict[str, Any]:
    """Assess if current positions are too correlated.

    Args:
        open_positions: {strategy_id: {symbol, notional_usd, side, ...}}
        correlation_matrix: Correlation matrix from calculate_strategy_correlations
        max_correlated_positions: Maximum positions allowed in same direction

    Returns:
        Assessment with risk score 0-1 and recommendations.
    """

    if not open_positions or len(open_positions) < 2:
        return {
            "concentration_score": 0.0,
            "is_concentrated": False,
            "risk_level": "LOW",
            "recommendations": ["No concentration issues: <2 positions"],
            "correlated_pairs": [],
        }

    # Check side concentration
    long_positions = [s for s, p in open_positions.items() if p.get("side") == "BUY"]
    short_positions = [s for s, p in open_positions.items() if p.get("side") == "SELL"]

    side_concentration = max(len(long_positions), len(short_positions))
    side_risk = side_concentration > max_correlated_positions

    # Check correlation concentration
    high_corr_pairs = [
        (s1, s2, corr)
        for (s1, s2), corr in correlation_matrix.correlations.items()
        if abs(corr) > 0.7  # >70% correlation = high
    ]

    # Calculate concentration score
    correlation_risk = (
        correlation_matrix.average_correlation * 0.5 if correlation_matrix.average_correlation > 0.5 else 0.0
    )
    side_risk_score = 0.3 if side_risk else 0.0
    concentration_score = min(1.0, correlation_risk + side_risk_score)

    recommendations = []
    if side_risk:
        recommendations.append(
            f"High side concentration: {side_concentration} {('long' if side_concentration == len(long_positions) else 'short')} "
            f"positions (max: {max_correlated_positions})"
        )

    if high_corr_pairs:
        recommendations.append(f"High correlation pairs: {len(high_corr_pairs)} pairs with >70% correlation")

    if correlation_matrix.average_correlation > 0.6:
        recommendations.append(
            f"Overall strategy correlation too high: {correlation_matrix.average_correlation:.2f} (target <0.4)"
        )

    if not recommendations:
        recommendations.append("Portfolio well-diversified")

    risk_level = "HIGH" if concentration_score > 0.7 else ("MEDIUM" if concentration_score > 0.4 else "LOW")

    return {
        "concentration_score": concentration_score,
        "is_concentrated": concentration_score > 0.6,
        "risk_level": risk_level,
        "side_concentration": side_concentration,
        "max_allowed": max_correlated_positions,
        "avg_correlation": correlation_matrix.average_correlation,
        "max_correlation": correlation_matrix.max_correlation,
        "high_correlation_pairs": high_corr_pairs,
        "recommendations": recommendations,
        "correlated_pairs": [(s1, s2, corr) for s1, s2, corr in high_corr_pairs],
    }


def filter_position_candidates(
    candidate_strategies: list[str],
    open_positions: dict[str, dict[str, Any]],
    correlation_matrix: CorrelationMatrix,
    max_new_correlated: int = 1,
) -> dict[str, Any]:
    """Filter which new positions are safe to open given current portfolio.

    Args:
        candidate_strategies: Strategies wanting to open new positions
        open_positions: Current open positions
        correlation_matrix: Strategy correlation matrix
        max_new_correlated: Max new positions allowed if already concentrated

    Returns:
        Filtering decision with approved/rejected candidates and reasons.
    """

    approved = []
    rejected = []

    for candidate in candidate_strategies:
        rejection_reason = None

        # Check correlation to existing positions
        max_correlation = 0.0
        most_correlated = None

        for open_strat in open_positions.keys():
            key = tuple(sorted([candidate, open_strat]))
            corr = correlation_matrix.correlations.get(key, 0.0)

            if abs(corr) > abs(max_correlation):
                max_correlation = corr
                most_correlated = open_strat

        # Reject if too correlated
        if abs(max_correlation) > 0.75:  # >75% correlation = reject
            rejection_reason = f"Too correlated with {most_correlated} (r={max_correlation:.2f} > 0.75)"
        elif abs(max_correlation) > 0.6 and len(approved) >= max_new_correlated:
            rejection_reason = (
                f"Already have {len(approved)} new positions, "
                f"this is correlated (r={max_correlation:.2f}) with {most_correlated}"
            )

        if rejection_reason:
            rejected.append((candidate, rejection_reason))
        else:
            approved.append(candidate)

    return {
        "approved": approved,
        "rejected": rejected,
        "total_candidates": len(candidate_strategies),
        "approved_count": len(approved),
        "rejection_rate": len(rejected) / len(candidate_strategies) if candidate_strategies else 0.0,
    }
