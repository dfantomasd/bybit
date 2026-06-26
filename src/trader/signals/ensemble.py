"""Ensemble signal combining and voting mechanism.

Combines signals from multiple strategies into a single ensemble decision
with configurable voting rules, confidence weighting, and regime filtering.

Voting methods:
  - Majority: Simple majority wins
  - Weighted: Weighted by strategy historical performance
  - Unanimous: All strategies must agree (strict)
  - Consensus: At least N strategies agree
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    """Signal types from strategies."""

    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"
    BLOCK = "BLOCK"


@dataclass
class StrategySignal:
    """Single strategy signal with metadata."""

    strategy_id: str
    signal: SignalType
    confidence: float  # 0-1, how confident the strategy is
    strength: float    # 0-1, signal strength (volume, momentum, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)
    regime_filter: str = "unknown"  # regime where this signal is valid


@dataclass
class EnsembleDecision:
    """Ensemble voting result."""

    final_signal: SignalType
    confidence: float  # 0-1, ensemble confidence
    votes_for: int
    votes_against: int
    votes_neutral: int
    agreement_pct: float  # percentage of strategies that agree
    component_signals: list[StrategySignal] = field(default_factory=list)
    reasoning: str = ""


class EnsembleVoter:
    """Combines multiple strategy signals into ensemble decisions."""

    def __init__(
        self,
        voting_method: str = "weighted",  # majority, weighted, unanimous, consensus
        min_agreement_pct: float = 60.0,  # 60% agreement threshold
        require_regime_alignment: bool = True,
    ):
        self.voting_method = voting_method
        self.min_agreement_pct = min_agreement_pct
        self.require_regime_alignment = require_regime_alignment
        self.strategy_weights: dict[str, float] = {}  # performance weights

    def set_strategy_weights(self, weights: dict[str, float]) -> None:
        """Set historical performance weights for each strategy.

        Args:
            weights: {strategy_id: weight} where higher = better performance
        """
        total = sum(weights.values()) or 1.0
        self.strategy_weights = {k: v / total for k, v in weights.items()}

    def vote(
        self,
        signals: list[StrategySignal],
        current_regime: str = "unknown",
    ) -> EnsembleDecision:
        """Combine multiple strategy signals into ensemble decision.

        Args:
            signals: List of signals from different strategies
            current_regime: Current market regime

        Returns:
            EnsembleDecision with voting results and recommendation.
        """

        if not signals:
            return EnsembleDecision(
                final_signal=SignalType.NEUTRAL,
                confidence=0.0,
                votes_for=0,
                votes_against=0,
                votes_neutral=0,
                agreement_pct=0.0,
                reasoning="No signals provided",
            )

        # Filter by regime if required
        filtered_signals = signals
        if self.require_regime_alignment:
            filtered_signals = [
                s for s in signals
                if s.regime_filter in ("unknown", current_regime)
            ]
            if not filtered_signals:
                # No signals align with current regime
                filtered_signals = signals  # fallback to all

        # Count votes
        buy_votes = 0.0
        sell_votes = 0.0
        neutral_votes = 0.0
        block_votes = 0.0

        for sig in filtered_signals:
            weight = self.strategy_weights.get(sig.strategy_id, 1.0 / len(filtered_signals))
            weighted_confidence = weight * sig.confidence

            if sig.signal == SignalType.BUY:
                buy_votes += weighted_confidence
            elif sig.signal == SignalType.SELL:
                sell_votes += weighted_confidence
            elif sig.signal == SignalType.BLOCK:
                block_votes += weighted_confidence
            else:
                neutral_votes += weighted_confidence

        total_votes = buy_votes + sell_votes + neutral_votes + block_votes

        if total_votes <= 0:
            return EnsembleDecision(
                final_signal=SignalType.NEUTRAL,
                confidence=0.0,
                votes_for=len([s for s in filtered_signals if s.signal in (SignalType.BUY, SignalType.SELL)]),
                votes_against=len([s for s in filtered_signals if s.signal == SignalType.BLOCK]),
                votes_neutral=len([s for s in filtered_signals if s.signal == SignalType.NEUTRAL]),
                agreement_pct=0.0,
                component_signals=filtered_signals,
                reasoning="No valid votes",
            )

        # Normalize votes
        buy_pct = (buy_votes / total_votes) * 100
        sell_pct = (sell_votes / total_votes) * 100
        block_pct = (block_votes / total_votes) * 100

        # Block if too many block votes
        if block_pct > 40:  # >40% block votes = veto
            return EnsembleDecision(
                final_signal=SignalType.BLOCK,
                confidence=min(0.9, block_pct / 100.0),
                votes_for=0,
                votes_against=len([s for s in filtered_signals if s.signal == SignalType.BLOCK]),
                votes_neutral=len([s for s in filtered_signals if s.signal != SignalType.BLOCK]),
                agreement_pct=block_pct,
                component_signals=filtered_signals,
                reasoning=f"Blocked by {block_pct:.1f}% of signals",
            )

        # Determine final signal based on voting method
        if self.voting_method == "unanimous":
            # All must agree
            final_signal = self._unanimous_vote(buy_pct, sell_pct)
            agreement_pct = max(buy_pct, sell_pct)
        elif self.voting_method == "consensus":
            # At least min_agreement_pct must agree
            final_signal = self._consensus_vote(buy_pct, sell_pct)
            agreement_pct = max(buy_pct, sell_pct)
        elif self.voting_method == "majority":
            # Simple majority
            final_signal = self._majority_vote(buy_pct, sell_pct)
            agreement_pct = max(buy_pct, sell_pct)
        else:  # weighted (default)
            final_signal = self._weighted_vote(buy_pct, sell_pct)
            agreement_pct = max(buy_pct, sell_pct)

        # Calculate ensemble confidence
        confidence = self._calculate_confidence(
            agreement_pct=agreement_pct,
            component_signals=filtered_signals,
            final_signal=final_signal,
        )

        # Check agreement threshold
        if agreement_pct < self.min_agreement_pct:
            reasoning = f"Insufficient agreement: {agreement_pct:.1f}% < {self.min_agreement_pct}%"
            final_signal = SignalType.NEUTRAL
        else:
            reasoning = f"{self.voting_method}: {agreement_pct:.1f}% agreement, {len(filtered_signals)} strategies"

        return EnsembleDecision(
            final_signal=final_signal,
            confidence=confidence,
            votes_for=len([s for s in filtered_signals if s.signal in (SignalType.BUY, SignalType.SELL)]),
            votes_against=len([s for s in filtered_signals if s.signal == SignalType.BLOCK]),
            votes_neutral=len([s for s in filtered_signals if s.signal == SignalType.NEUTRAL]),
            agreement_pct=agreement_pct,
            component_signals=filtered_signals,
            reasoning=reasoning,
        )

    @staticmethod
    def _majority_vote(buy_pct: float, sell_pct: float) -> SignalType:
        """Simple majority: whoever has more votes wins."""
        if buy_pct > sell_pct:
            return SignalType.BUY
        elif sell_pct > buy_pct:
            return SignalType.SELL
        else:
            return SignalType.NEUTRAL

    @staticmethod
    def _weighted_vote(buy_pct: float, sell_pct: float) -> SignalType:
        """Same as majority for binary choice."""
        return EnsembleVoter._majority_vote(buy_pct, sell_pct)

    def _consensus_vote(self, buy_pct: float, sell_pct: float) -> SignalType:
        """At least min_agreement_pct must agree."""
        if buy_pct >= self.min_agreement_pct:
            return SignalType.BUY
        elif sell_pct >= self.min_agreement_pct:
            return SignalType.SELL
        else:
            return SignalType.NEUTRAL

    @staticmethod
    def _unanimous_vote(buy_pct: float, sell_pct: float) -> SignalType:
        """All must agree (>=90%)."""
        if buy_pct >= 90:
            return SignalType.BUY
        elif sell_pct >= 90:
            return SignalType.SELL
        else:
            return SignalType.NEUTRAL

    @staticmethod
    def _calculate_confidence(
        agreement_pct: float,
        component_signals: list[StrategySignal],
        final_signal: SignalType,
    ) -> float:
        """Calculate ensemble confidence from agreement and component strengths.

        Args:
            agreement_pct: Percentage of strategies voting for final signal
            component_signals: List of component signals
            final_signal: Final ensemble signal

        Returns:
            Confidence score 0-1.
        """

        if final_signal == SignalType.NEUTRAL:
            return 0.0

        # Base confidence from agreement percentage
        agreement_confidence = agreement_pct / 100.0

        # Boost confidence if component signals are strong
        agreeing_signals = [
            s for s in component_signals
            if (final_signal == SignalType.BUY and s.signal == SignalType.BUY)
            or (final_signal == SignalType.SELL and s.signal == SignalType.SELL)
        ]

        if agreeing_signals:
            avg_strength = sum(s.strength for s in agreeing_signals) / len(agreeing_signals)
            agreement_confidence = (agreement_confidence + avg_strength) / 2.0

        return min(0.95, agreement_confidence)  # Cap at 0.95


def create_regime_aware_ensemble(
    regime: str,
    weights: dict[str, dict[str, float]] | None = None,
) -> EnsembleVoter:
    """Create an ensemble voter with regime-specific configuration.

    Args:
        regime: Current market regime
        weights: Optional {regime: {strategy_id: weight}} weights

    Returns:
        Configured EnsembleVoter.
    """

    # Regime-specific voting methods
    voting_config = {
        "HIGH_VOLATILITY": {
            "method": "unanimous",  # strict in volatile markets
            "min_agreement_pct": 80.0,
        },
        "SIDEWAYS": {
            "method": "consensus",
            "min_agreement_pct": 70.0,
        },
        "TRENDING": {
            "method": "majority",
            "min_agreement_pct": 50.0,
        },
        "UNCERTAIN": {
            "method": "consensus",
            "min_agreement_pct": 75.0,
        },
    }

    config = voting_config.get(regime, {
        "method": "weighted",
        "min_agreement_pct": 60.0,
    })

    voter = EnsembleVoter(
        voting_method=config.get("method", "weighted"),
        min_agreement_pct=config.get("min_agreement_pct", 60.0),
        require_regime_alignment=True,
    )

    # Set weights if provided
    if weights and regime in weights:
        voter.set_strategy_weights(weights[regime])

    return voter
