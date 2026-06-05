"""Unit tests for all Pydantic v2 domain models."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from trader.domain.enums import (
    MarketRegime,
    MarketType,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskDecisionStatus,
    VolatilityLevel,
)
from trader.domain.models import (
    AuditEvent,
    Balance,
    FeatureVector,
    Fill,
    HealthStatus,
    InstrumentInfo,
    MarketEvent,
    ModelMetadata,
    OrderIntent,
    Position,
    PreflightReport,
    ReconciliationResult,
    RegimeContext,
    RiskDecision,
    TradeProposal,
)


# ---------------------------------------------------------------------------
# MarketEvent
# ---------------------------------------------------------------------------


class TestMarketEvent:
    def test_valid_creation(self) -> None:
        ev = MarketEvent(
            symbol="btcusdt",  # lowercase → should be uppercased
            market_type=MarketType.LINEAR,
            close=Decimal("65000"),
            volume=Decimal("100"),
        )
        assert ev.symbol == "BTCUSDT"
        assert ev.market_type == MarketType.LINEAR
        assert ev.event_id is not None

    def test_symbol_is_uppercased(self) -> None:
        ev = MarketEvent(symbol="ethusdt", market_type=MarketType.SPOT)
        assert ev.symbol == "ETHUSDT"

    def test_optional_fields_default_to_none(self) -> None:
        ev = MarketEvent(symbol="BTCUSDT", market_type=MarketType.LINEAR)
        assert ev.close is None
        assert ev.bid is None
        assert ev.funding_rate is None

    def test_negative_latency_rejected(self) -> None:
        with pytest.raises(ValidationError, match="latency_ms"):
            MarketEvent(symbol="BTCUSDT", market_type=MarketType.LINEAR, latency_ms=-1.0)

    def test_zero_latency_accepted(self) -> None:
        ev = MarketEvent(symbol="BTCUSDT", market_type=MarketType.LINEAR, latency_ms=0.0)
        assert ev.latency_ms == 0.0

    def test_is_frozen(self) -> None:
        ev = MarketEvent(symbol="BTCUSDT", market_type=MarketType.LINEAR)
        with pytest.raises(ValidationError):
            ev.symbol = "ETHUSDT"  # type: ignore[misc]

    def test_has_event_id_uuid(self) -> None:
        ev = MarketEvent(symbol="BTCUSDT", market_type=MarketType.LINEAR)
        assert isinstance(ev.event_id, uuid.UUID)

    def test_timestamp_is_utc(self) -> None:
        ev = MarketEvent(symbol="BTCUSDT", market_type=MarketType.LINEAR)
        assert ev.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# FeatureVector
# ---------------------------------------------------------------------------


class TestFeatureVector:
    def test_valid_creation(self) -> None:
        fv = FeatureVector(
            symbol="BTCUSDT",
            values=[0.1, 0.2, 0.3],
            feature_names=["a", "b", "c"],
            quality_score=0.9,
            lookback_bars=50,
        )
        assert len(fv.values) == 3

    def test_quality_score_bounds_lower(self) -> None:
        with pytest.raises(ValidationError):
            FeatureVector(symbol="X", values=[1.0], quality_score=-0.1, lookback_bars=10)

    def test_quality_score_bounds_upper(self) -> None:
        with pytest.raises(ValidationError):
            FeatureVector(symbol="X", values=[1.0], quality_score=1.1, lookback_bars=10)

    def test_quality_score_edge_values(self) -> None:
        fv0 = FeatureVector(symbol="X", values=[1.0], quality_score=0.0, lookback_bars=10)
        fv1 = FeatureVector(symbol="X", values=[1.0], quality_score=1.0, lookback_bars=10)
        assert fv0.quality_score == 0.0
        assert fv1.quality_score == 1.0

    def test_empty_values_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureVector(symbol="X", values=[], quality_score=0.5, lookback_bars=10)

    def test_mismatched_feature_names_rejected(self) -> None:
        with pytest.raises(ValidationError, match="feature_names length"):
            FeatureVector(
                symbol="X",
                values=[1.0, 2.0],
                feature_names=["only_one"],
                quality_score=0.5,
                lookback_bars=10,
            )

    def test_empty_feature_names_ok(self) -> None:
        """Empty feature_names list is acceptable (names are optional)."""
        fv = FeatureVector(
            symbol="X",
            values=[1.0, 2.0],
            feature_names=[],
            quality_score=0.5,
            lookback_bars=10,
        )
        assert fv.feature_names == []

    def test_lookback_bars_minimum(self) -> None:
        with pytest.raises(ValidationError):
            FeatureVector(symbol="X", values=[1.0], quality_score=0.5, lookback_bars=0)


# ---------------------------------------------------------------------------
# TradeProposal
# ---------------------------------------------------------------------------


class TestTradeProposal:
    def test_valid_buy_proposal(self, sample_trade_proposal: TradeProposal) -> None:
        assert sample_trade_proposal.side == OrderSide.BUY
        assert sample_trade_proposal.confidence > 0

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            TradeProposal(
                strategy_id="s1",
                symbol="BTCUSDT",
                market_type=MarketType.LINEAR,
                side=OrderSide.BUY,
                requested_qty=Decimal("0.001"),
                confidence=1.1,
            )

    def test_confidence_zero_allowed(self) -> None:
        p = TradeProposal(
            strategy_id="s1",
            symbol="BTCUSDT",
            market_type=MarketType.LINEAR,
            side=OrderSide.BUY,
            requested_qty=Decimal("0.001"),
            confidence=0.0,
        )
        assert p.confidence == 0.0

    def test_negative_qty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TradeProposal(
                strategy_id="s1",
                symbol="BTCUSDT",
                market_type=MarketType.LINEAR,
                side=OrderSide.BUY,
                requested_qty=Decimal("-1"),
                confidence=0.5,
            )

    def test_buy_stop_loss_above_entry_rejected(self) -> None:
        with pytest.raises(ValidationError, match="stop_loss must be below"):
            TradeProposal(
                strategy_id="s1",
                symbol="BTCUSDT",
                market_type=MarketType.LINEAR,
                side=OrderSide.BUY,
                requested_qty=Decimal("0.001"),
                confidence=0.5,
                entry_price=Decimal("1000"),
                stop_loss=Decimal("1100"),  # above entry — invalid for BUY
            )

    def test_sell_stop_loss_below_entry_rejected(self) -> None:
        with pytest.raises(ValidationError, match="stop_loss must be above"):
            TradeProposal(
                strategy_id="s1",
                symbol="BTCUSDT",
                market_type=MarketType.LINEAR,
                side=OrderSide.SELL,
                requested_qty=Decimal("0.001"),
                confidence=0.5,
                entry_price=Decimal("1000"),
                stop_loss=Decimal("900"),  # below entry — invalid for SELL
            )

    def test_symbol_uppercased(self) -> None:
        p = TradeProposal(
            strategy_id="s1",
            symbol="btcusdt",
            market_type=MarketType.LINEAR,
            side=OrderSide.BUY,
            requested_qty=Decimal("0.001"),
            confidence=0.5,
        )
        assert p.symbol == "BTCUSDT"

    def test_has_unique_proposal_id(self) -> None:
        p1 = TradeProposal(
            strategy_id="s1",
            symbol="BTCUSDT",
            market_type=MarketType.LINEAR,
            side=OrderSide.BUY,
            requested_qty=Decimal("0.001"),
            confidence=0.5,
        )
        p2 = TradeProposal(
            strategy_id="s1",
            symbol="BTCUSDT",
            market_type=MarketType.LINEAR,
            side=OrderSide.BUY,
            requested_qty=Decimal("0.001"),
            confidence=0.5,
        )
        assert p1.proposal_id != p2.proposal_id


# ---------------------------------------------------------------------------
# RiskDecision
# ---------------------------------------------------------------------------


class TestRiskDecision:
    def test_approved_requires_qty(self) -> None:
        pid = uuid.uuid4()
        with pytest.raises(ValidationError, match="approved_qty is required"):
            RiskDecision(
                proposal_id=pid,
                status=RiskDecisionStatus.APPROVED,
                # approved_qty intentionally missing
            )

    def test_resized_requires_qty(self) -> None:
        pid = uuid.uuid4()
        with pytest.raises(ValidationError, match="approved_qty is required"):
            RiskDecision(
                proposal_id=pid,
                status=RiskDecisionStatus.RESIZED,
            )

    def test_rejected_does_not_require_qty(self) -> None:
        pid = uuid.uuid4()
        rd = RiskDecision(
            proposal_id=pid,
            status=RiskDecisionStatus.REJECTED,
            reason="Max daily drawdown exceeded",
        )
        assert rd.status == RiskDecisionStatus.REJECTED
        assert rd.approved_qty is None

    def test_approved_zero_qty_rejected(self) -> None:
        pid = uuid.uuid4()
        with pytest.raises(ValidationError, match="approved_qty must be positive"):
            RiskDecision(
                proposal_id=pid,
                status=RiskDecisionStatus.APPROVED,
                approved_qty=Decimal("0"),
            )

    def test_valid_approved_decision(self, sample_risk_decision: RiskDecision) -> None:
        assert sample_risk_decision.status == RiskDecisionStatus.APPROVED
        assert sample_risk_decision.approved_qty is not None
        assert sample_risk_decision.approved_qty > 0


# ---------------------------------------------------------------------------
# OrderIntent
# ---------------------------------------------------------------------------


class TestOrderIntent:
    def test_valid_market_order(self, sample_order_intent: OrderIntent) -> None:
        assert sample_order_intent.order_type == OrderType.MARKET
        assert sample_order_intent.price is None

    def test_limit_requires_price(self) -> None:
        did = uuid.uuid4()
        pid = uuid.uuid4()
        with pytest.raises(ValidationError, match="price is required"):
            OrderIntent(
                decision_id=did,
                proposal_id=pid,
                symbol="BTCUSDT",
                market_type=MarketType.LINEAR,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                qty=Decimal("0.001"),
                order_link_id="test-link-id",
                # price intentionally missing
            )

    def test_order_link_id_valid_format(self, sample_order_intent: OrderIntent) -> None:
        import re

        assert re.match(r"^[a-zA-Z0-9_-]{1,36}$", sample_order_intent.order_link_id)

    def test_order_link_id_too_long_rejected(self) -> None:
        did = uuid.uuid4()
        pid = uuid.uuid4()
        with pytest.raises(ValidationError):
            OrderIntent(
                decision_id=did,
                proposal_id=pid,
                symbol="BTCUSDT",
                market_type=MarketType.LINEAR,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                qty=Decimal("0.001"),
                order_link_id="a" * 37,  # 37 chars — exceeds 36 limit
            )

    def test_order_link_id_invalid_chars_rejected(self) -> None:
        did = uuid.uuid4()
        pid = uuid.uuid4()
        with pytest.raises(ValidationError):
            OrderIntent(
                decision_id=did,
                proposal_id=pid,
                symbol="BTCUSDT",
                market_type=MarketType.LINEAR,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                qty=Decimal("0.001"),
                order_link_id="invalid link id!",  # contains space and !
            )

    def test_zero_qty_rejected(self) -> None:
        did = uuid.uuid4()
        pid = uuid.uuid4()
        with pytest.raises(ValidationError):
            OrderIntent(
                decision_id=did,
                proposal_id=pid,
                symbol="BTCUSDT",
                market_type=MarketType.LINEAR,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                qty=Decimal("0"),
                order_link_id="valid-id",
            )


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


class TestPosition:
    def test_valid_position(self) -> None:
        p = Position(
            symbol="BTCUSDT",
            market_type=MarketType.LINEAR,
            side=OrderSide.BUY,
            size=Decimal("0.01"),
            entry_price=Decimal("65000"),
        )
        assert p.size == Decimal("0.01")

    def test_negative_size_rejected(self) -> None:
        with pytest.raises(ValidationError, match="size must be non-negative"):
            Position(
                symbol="BTCUSDT",
                market_type=MarketType.LINEAR,
                side=OrderSide.BUY,
                size=Decimal("-0.01"),
                entry_price=Decimal("65000"),
            )

    def test_zero_size_allowed(self) -> None:
        p = Position(
            symbol="BTCUSDT",
            market_type=MarketType.LINEAR,
            side=OrderSide.BUY,
            size=Decimal("0"),
            entry_price=Decimal("65000"),
        )
        assert p.size == Decimal("0")


# ---------------------------------------------------------------------------
# ReconciliationResult
# ---------------------------------------------------------------------------


class TestReconciliationResult:
    def test_default_values(self) -> None:
        r = ReconciliationResult()
        assert r.orders_checked == 0
        assert r.discrepancies_found == 0
        assert r.success is True

    def test_with_discrepancies(self) -> None:
        r = ReconciliationResult(
            orders_checked=10,
            discrepancies_found=2,
            discrepancies_resolved=1,
            discrepancies_unresolved=1,
            mismatched_order_ids=["order-1", "order-2"],
            success=False,
        )
        assert r.discrepancies_unresolved == 1
        assert len(r.mismatched_order_ids) == 2


# ---------------------------------------------------------------------------
# ModelMetadata
# ---------------------------------------------------------------------------


class TestModelMetadata:
    def test_valid_creation(self) -> None:
        now = datetime.now(tz=timezone.utc)
        m = ModelMetadata(
            model_id="ppo-btcusdt-v1",
            version="1.0.0",
            algorithm="PPO",
            strategy_id="ppo-v1",
            trained_at=now,
            train_sharpe=1.8,
            val_sharpe=1.5,
            train_steps=1000000,
        )
        assert m.is_stale is False
        assert m.inference_count == 0

    def test_drift_score_optional(self) -> None:
        now = datetime.now(tz=timezone.utc)
        m = ModelMetadata(
            model_id="m1",
            version="1",
            algorithm="SAC",
            strategy_id="sac-v1",
            trained_at=now,
        )
        assert m.drift_score is None
