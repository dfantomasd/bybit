from __future__ import annotations

import uuid
from decimal import Decimal

from trader.domain.enums import MarketType, OrderSide
from trader.domain.models import FeatureVector, TradeProposal
from trader.strategies.base import BaseStrategy
from trader.strategies.ensemble import StrategyEnsemble


class _Strategy(BaseStrategy):
    def __init__(self, strategy_id: str, side: OrderSide | None, confidence: float = 0.6) -> None:
        self._strategy_id = strategy_id
        self._side = side
        self._confidence = confidence

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    def evaluate(
        self,
        feature_vector: FeatureVector,
        current_price: float,
        available_balance_usd: float,
    ) -> TradeProposal | None:
        if self._side is None:
            return None
        return TradeProposal(
            proposal_id=uuid.uuid4(),
            strategy_id=self.strategy_id,
            symbol=feature_vector.symbol,
            market_type=MarketType.LINEAR,
            side=self._side,
            requested_qty=Decimal("1"),
            entry_price=Decimal(str(current_price)),
            take_profit=Decimal(str(current_price * 1.01))
            if self._side == OrderSide.BUY
            else Decimal(str(current_price * 0.99)),
            stop_loss=Decimal(str(current_price * 0.99))
            if self._side == OrderSide.BUY
            else Decimal(str(current_price * 1.01)),
            confidence=self._confidence,
        )


def _vector() -> FeatureVector:
    return FeatureVector(
        symbol="TESTUSDT",
        values=[1.0],
        feature_names=["x"],
        quality_score=1.0,
        lookback_bars=10,
    )


def test_higher_priority_strategy_wins_direction_conflict() -> None:
    ensemble = StrategyEnsemble(
        strategies=[
            _Strategy("ema_crossover_v1", OrderSide.BUY, 0.9),
            _Strategy("order_flow_v1", OrderSide.SELL, 0.6),
        ],
        min_confidence=0.5,
        strategy_priorities={"order_flow_v1": 10, "ema_crossover_v1": 1},
    )

    proposal = ensemble.evaluate_all(_vector(), 10.0, 1000.0)

    assert proposal is not None
    assert proposal.strategy_id == "order_flow_v1"
    assert proposal.side == OrderSide.SELL


def test_equal_priority_conflict_is_blocked() -> None:
    ensemble = StrategyEnsemble(
        strategies=[
            _Strategy("a", OrderSide.BUY, 0.8),
            _Strategy("b", OrderSide.SELL, 0.8),
        ],
        min_confidence=0.5,
        strategy_priorities={"a": 1, "b": 1},
    )

    assert ensemble.evaluate_all(_vector(), 10.0, 1000.0) is None


def test_confirmation_required_blocks_commodity_strategy_alone() -> None:
    ensemble = StrategyEnsemble(
        strategies=[
            _Strategy("ema_crossover_v1", OrderSide.BUY, 0.9),
        ],
        min_confidence=0.5,
        strategy_priorities={"ema_crossover_v1": 1},
        confirmation_required_for={"ema_crossover_v1"},
        confirmation_sources={"order_flow_v1"},
    )

    assert ensemble.evaluate_all(_vector(), 10.0, 1000.0) is None


def test_confirmation_required_allows_confirmed_commodity_strategy() -> None:
    ensemble = StrategyEnsemble(
        strategies=[
            _Strategy("ema_crossover_v1", OrderSide.BUY, 0.8),
            _Strategy("order_flow_v1", OrderSide.BUY, 0.6),
        ],
        min_confidence=0.5,
        strategy_priorities={"ema_crossover_v1": 10, "order_flow_v1": 1},
        confirmation_required_for={"ema_crossover_v1"},
        confirmation_sources={"order_flow_v1"},
    )

    proposal = ensemble.evaluate_all(_vector(), 10.0, 1000.0)

    assert proposal is not None
    assert proposal.strategy_id == "ema_crossover_v1"
    assert proposal.confidence > 0.8


def test_independent_alpha_strategy_can_trade_without_confirmation() -> None:
    ensemble = StrategyEnsemble(
        strategies=[
            _Strategy("order_flow_v1", OrderSide.SELL, 0.6),
        ],
        min_confidence=0.5,
        strategy_priorities={"order_flow_v1": 10},
        confirmation_required_for={"ema_crossover_v1", "scalp_micro_v1"},
        confirmation_sources={"order_flow_v1"},
    )

    proposal = ensemble.evaluate_all(_vector(), 10.0, 1000.0)

    assert proposal is not None
    assert proposal.strategy_id == "order_flow_v1"


def test_ensemble_emits_diagnostics_for_silent_and_emitted_strategies() -> None:
    events: list[str] = []
    ensemble = StrategyEnsemble(
        strategies=[
            _Strategy("quiet_v1", None),
            _Strategy("order_flow_v1", OrderSide.SELL, 0.6),
        ],
        min_confidence=0.5,
        strategy_priorities={"order_flow_v1": 10},
        diag_hook=events.append,
    )

    proposal = ensemble.evaluate_all(_vector(), 10.0, 1000.0)

    assert proposal is not None
    assert "strategy_no_signal:quiet_v1" in events
    assert "strategy_proposed:order_flow_v1" in events
    assert "ensemble_emitted:order_flow_v1" in events
