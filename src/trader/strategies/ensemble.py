"""Strategy ensemble: runs all registered strategies and aggregates proposals.

Aggregation rules
-----------------
1. If multiple strategies agree on direction → increase confidence.
2. If strategies disagree → higher-priority strategy family wins; equal
   priority conflicts are skipped and logged.
3. Always return at most one proposal per symbol (highest confidence).
4. The Risk Manager has final authority.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog

from trader.domain.models import FeatureVector, RegimeContext, TradeProposal
from trader.strategies.base import BaseStrategy
from trader.strategies.regime_adapter import RegimeAwarePrioritizer

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
        strategy_priorities: dict[str, int] | None = None,
        confirmation_required_for: set[str] | None = None,
        confirmation_sources: set[str] | None = None,
        min_confirmation_sources: int = 1,
        diag_hook: Any | None = None,
    ) -> None:
        self._strategies = strategies
        self._health = health_checker
        self._min_confidence = min_confidence
        self._agree_bonus = agree_bonus
        self._strategy_priorities = strategy_priorities or {}
        self._confirmation_required_for = confirmation_required_for or set()
        self._confirmation_sources = confirmation_sources or set()
        self._min_confirmation_sources = max(1, int(min_confirmation_sources))
        self._diag_hook = diag_hook

    def _diag(self, event: str) -> None:
        if self._diag_hook is None:
            return
        try:
            self._diag_hook(event)
        except Exception as exc:
            log.debug("ensemble.diag_hook_failed", event=event, error=str(exc))

    def _priority(self, proposal: TradeProposal, priorities: dict[str, int] | None = None) -> int:
        p = priorities or self._strategy_priorities
        return p.get(proposal.strategy_id, 0)

    def evaluate_all(
        self,
        feature_vector: FeatureVector,
        current_price: float,
        available_balance_usd: float,
        regime_ctx: RegimeContext | None = None,
    ) -> TradeProposal | None:
        """Run all strategies and return the best-combined proposal, or None.

        Args:
            feature_vector: Current price/technical features
            current_price: Current market price
            available_balance_usd: Available capital for position sizing
            regime_ctx: Current market regime (optional, for adaptive prioritization)
        """
        # Adjust priorities based on current market regime
        priorities = self._strategy_priorities
        regime_adapted = False
        if regime_ctx is not None:
            try:
                priorities = RegimeAwarePrioritizer.compute_priorities(
                    regime_ctx=regime_ctx,
                    base_priorities=self._strategy_priorities,
                )
                regime_adapted = priorities != self._strategy_priorities
            except Exception as e:
                log.warning(
                    "ensemble.regime_adaptation_error",
                    symbol=feature_vector.symbol,
                    error=str(e),
                )
                # Fall back to base priorities if regime adaptation fails
                priorities = self._strategy_priorities
                regime_adapted = False

        proposals: list[TradeProposal] = []

        for strategy in self._strategies:
            try:
                proposal = strategy.evaluate(feature_vector, current_price, available_balance_usd)
                if proposal is not None:
                    self._diag(f"strategy_proposed:{strategy.strategy_id}")
                    proposals.append(proposal)
                else:
                    self._diag(f"strategy_no_signal:{strategy.strategy_id}")
                    log.debug(
                        "ensemble.strategy_no_signal",
                        strategy_id=strategy.strategy_id,
                        symbol=feature_vector.symbol,
                    )
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

        if buys and sells:
            buy_priority = max(self._priority(p, priorities) for p in buys)
            sell_priority = max(self._priority(p, priorities) for p in sells)
            if buy_priority == sell_priority:
                log.warning(
                    "ensemble.conflict_blocked_equal_priority",
                    symbol=feature_vector.symbol,
                    buy_count=len(buys),
                    sell_count=len(sells),
                    buy_strategies_with_confidence=[f"{p.strategy_id}:{round(p.confidence, 3)}" for p in buys],
                    sell_strategies_with_confidence=[f"{p.strategy_id}:{round(p.confidence, 3)}" for p in sells],
                    priority=buy_priority,
                )
                self._diag("ensemble_conflict_blocked")
                return None
            agreed = buys if buy_priority > sell_priority else sells
            suppressed = sells if buy_priority > sell_priority else buys
            log.info(
                "ensemble.conflict_resolved_by_priority",
                symbol=feature_vector.symbol,
                selected_side=agreed[0].side.value,
                selected_priority=max(self._priority(p, priorities) for p in agreed),
                selected_strategies=[p.strategy_id for p in agreed],
                selected_confidence=[round(p.confidence, 3) for p in agreed],
                suppressed_strategies=[p.strategy_id for p in suppressed],
                suppressed_confidence=[round(p.confidence, 3) for p in suppressed],
                suppressed_count=len(suppressed),
            )
        else:
            agreed = buys if buys else sells

        # Pick highest-confidence proposal and boost by agreement
        best = max(agreed, key=lambda p: (self._priority(p, priorities), p.confidence))
        if best.strategy_id in self._confirmation_required_for:
            confirming_sources = {
                proposal.strategy_id
                for proposal in agreed
                if proposal.strategy_id != best.strategy_id and proposal.strategy_id in self._confirmation_sources
            }
            if len(confirming_sources) < self._min_confirmation_sources:
                log.info(
                    "ensemble.confirmation_required_blocked",
                    symbol=feature_vector.symbol,
                    strategy_id=best.strategy_id,
                    side=best.side.value,
                    confirming_sources=sorted(confirming_sources),
                    min_confirmation_sources=self._min_confirmation_sources,
                )
                self._diag(f"ensemble_confirmation_blocked:{best.strategy_id}")
                return None

        agreement_bonus = self._agree_bonus * (len(agreed) - 1)

        # Apply regime-aware confidence adjustments
        alignment_multiplier = RegimeAwarePrioritizer.confidence_adjustment_for_alignment(
            proposal_side=best.side.value,
            regime_ctx=regime_ctx,
        )

        # Rebuild with updated confidence (frozen model, need model_copy)
        new_conf = min(best.confidence + agreement_bonus, 0.95)
        if alignment_multiplier != 1.0:
            new_conf = min(new_conf * alignment_multiplier, 0.95)

        if new_conf < self._min_confidence:
            log.info(
                "ensemble.proposal_below_min_confidence",
                symbol=best.symbol,
                strategy_id=best.strategy_id,
                confidence=round(new_conf, 3),
                min_confidence=self._min_confidence,
            )
            self._diag(f"ensemble_below_min_confidence:{best.strategy_id}")
            return None

        if new_conf != best.confidence:
            best = best.model_copy(update={"confidence": new_conf})

        log.info(
            "ensemble.proposal_emitted",
            symbol=best.symbol,
            side=best.side,
            confidence=round(new_conf, 3),
            strategy_count=len(agreed),
            strategy_id=best.strategy_id,
            priority=self._priority(best, priorities),
            regime=regime_ctx.regime.value if regime_ctx else "unknown",
            regime_confidence=round(regime_ctx.confidence, 3) if regime_ctx else None,
            alignment_multiplier=round(alignment_multiplier, 3),
            regime_adapted_priorities=regime_adapted,
        )
        self._diag(f"ensemble_emitted:{best.strategy_id}")
        return best
