"""Strategy ensemble: runs all registered strategies and aggregates proposals.

Aggregation rules
-----------------
1. If multiple strategies agree on direction → increase confidence.
2. If strategies disagree → reduce confidence or skip.
3. Always return at most one proposal per symbol (highest confidence).
4. The Risk Manager has final authority.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog

from trader.domain.models import FeatureVector, TradeProposal
from trader.strategies.base import BaseStrategy

log = structlog.get_logger(__name__)


class StrategyEnsemble:
    """Aggregates signals from multiple BaseStrategy instances.

    Args:
        strategies:        List of strategy instances.
        health_checker:    Optional HealthChecker to notify on inference.
        min_confidence:    Minimum confidence to emit a proposal.
        agree_bonus:       Confidence bonus when multiple strategies agree.
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        health_checker: Any | None = None,
        min_confidence: float = 0.50,
        agree_bonus: float = 0.05,
    ) -> None:
        self._strategies = strategies
        self._health = health_checker
        self._min_confidence = min_confidence
        self._agree_bonus = agree_bonus

    def evaluate_all(
        self,
        feature_vector: FeatureVector,
        current_price: float,
        available_balance_usd: float,
    ) -> TradeProposal | None:
        """Run all strategies and return the best-combined proposal, or None."""
        proposals: list[TradeProposal] = []

        for strategy in self._strategies:
            try:
                proposal = strategy.evaluate(feature_vector, current_price, available_balance_usd)
                if proposal is not None:
                    proposals.append(proposal)
            except Exception as exc:
                log.warning(
                    "ensemble.strategy_error",
                    strategy_id=strategy.strategy_id,
                    symbol=feature_vector.symbol,
                    error=str(exc),
                )

        if self._health is not None:
            # Mark model inference even when no signal (strategy ran = inference happened)
            self._health.set_model_inference_at(datetime.now(tz=UTC))

        if not proposals:
            return None

        # Group by direction (side)
        from trader.domain.enums import OrderSide

        buys = [p for p in proposals if p.side == OrderSide.BUY]
        sells = [p for p in proposals if p.side == OrderSide.SELL]

        # Disagreement → skip
        if buys and sells:
            log.debug(
                "ensemble.conflicting_signals",
                symbol=feature_vector.symbol,
                buys=len(buys),
                sells=len(sells),
            )
            return None

        agreed = buys if buys else sells
        # Pick highest-confidence proposal and boost by agreement
        best = max(agreed, key=lambda p: p.confidence)
        agreement_bonus = self._agree_bonus * (len(agreed) - 1)

        # Rebuild with updated confidence (frozen model, need model_copy)
        new_conf = min(best.confidence + agreement_bonus, 0.95)
        if new_conf < self._min_confidence:
            return None

        if new_conf != best.confidence:
            best = best.model_copy(update={"confidence": new_conf})

        log.info(
            "ensemble.proposal_emitted",
            symbol=best.symbol,
            side=best.side,
            confidence=round(new_conf, 3),
            strategy_count=len(agreed),
        )
        return best
