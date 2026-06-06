"""Base strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from trader.domain.models import FeatureVector, TradeProposal


class BaseStrategy(ABC):
    """Strategy produces TradeProposals from FeatureVectors.

    Strategies never submit orders. They return a proposal or None.
    The RiskManager has final authority over execution.
    """

    @property
    @abstractmethod
    def strategy_id(self) -> str:
        """Unique identifier for this strategy."""

    @abstractmethod
    def evaluate(
        self,
        feature_vector: FeatureVector,
        current_price: float,
        available_balance_usd: float,
    ) -> TradeProposal | None:
        """Evaluate features and return a TradeProposal or None (no signal)."""
