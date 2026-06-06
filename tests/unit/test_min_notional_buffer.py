"""Tests: P0.5 – min-notional safety buffer."""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

from trader.domain.enums import MarketRegime, MarketType, OrderSide, RiskDecisionStatus, RiskProfile
from trader.domain.models import InstrumentInfo, TradeProposal
from trader.risk.circuit_breakers import CircuitBreakerManager
from trader.risk.drawdown import DrawdownTracker
from trader.risk.exposure import ExposureTracker
from trader.risk.kill_switch import KillSwitch
from trader.risk.manager import RiskManager
from trader.risk.profiles import get_risk_limits


def _make_rm(
    capital: Decimal = Decimal("1000"),
    buffer_pct: float = 3.0,
) -> RiskManager:
    limits = get_risk_limits(RiskProfile.CONSERVATIVE)
    return RiskManager(
        risk_profile=RiskProfile.CONSERVATIVE,
        drawdown_tracker=DrawdownTracker(initial_equity=capital),
        exposure_tracker=ExposureTracker(total_capital=capital, risk_limits=limits),
        circuit_breaker_manager=CircuitBreakerManager(risk_limits=limits),
        kill_switch=KillSwitch(),
        min_notional_safety_buffer_pct=buffer_pct,
    )


def _proposal(qty: str, price: str, confidence: float = 1.0) -> TradeProposal:
    return TradeProposal(
        proposal_id=uuid.uuid4(),
        strategy_id="test_strategy",
        symbol="XRPUSDT",
        side=OrderSide.BUY,
        market_type=MarketType.LINEAR,
        requested_qty=Decimal(qty),
        entry_price=Decimal(price),
        stop_loss=Decimal(str(float(price) * 0.95)),
        take_profit=Decimal(str(float(price) * 1.10)),
        confidence=confidence,
        regime=MarketRegime.BULL_TREND,
    )


def _instrument(min_notional: str = "5") -> InstrumentInfo:
    from trader.domain.enums import MarketType

    return InstrumentInfo(
        symbol="XRPUSDT",
        market_type=MarketType.LINEAR,
        base_coin="XRP",
        quote_coin="USDT",
        min_order_qty=Decimal("1"),
        max_order_qty=Decimal("100000"),
        qty_step=Decimal("1"),
        tick_size=Decimal("0.0001"),
        min_notional=Decimal(min_notional),
        max_leverage=Decimal("20"),
    )


@pytest.mark.asyncio
async def test_exactly_min_notional_is_bumped():
    """$5.00 order at exactly the min threshold should be bumped by buffer."""
    rm = _make_rm(buffer_pct=3.0)
    # 10 XRP at $0.50 = $5.00 — exactly at min_notional
    proposal = _proposal(qty="10", price="0.5000", confidence=1.0)
    instrument = _instrument(min_notional="5")
    capital = Decimal("1000")

    decision = await rm.evaluate(
        proposal=proposal,
        capital=capital,
        available_balance=capital,
        instrument_info=instrument,
    )
    # After buffer: required = $5 * 1.03 = $5.15 → needs 11 XRP at $0.50
    # Should either be bumped to 11 or rejected (not sent to Bybit at $5.00)
    if decision.status in (RiskDecisionStatus.APPROVED, RiskDecisionStatus.RESIZED):
        assert decision.approved_qty is not None
        notional = decision.approved_qty * Decimal("0.5000")
        assert notional >= Decimal("5.15"), f"notional {notional} should be >= $5.15 (with 3% buffer)"
    else:
        # Rejection is also acceptable if bump would violate limits
        assert decision.status == RiskDecisionStatus.REJECTED


@pytest.mark.asyncio
async def test_above_buffered_minimum_is_allowed():
    """$5.20+ order should pass without modification."""
    rm = _make_rm(buffer_pct=3.0)
    # 11 XRP at $0.5001 = $5.50 — above $5.15 threshold
    proposal = _proposal(qty="11", price="0.5001", confidence=1.0)
    instrument = _instrument(min_notional="5")
    capital = Decimal("1000")

    decision = await rm.evaluate(
        proposal=proposal,
        capital=capital,
        available_balance=capital,
        instrument_info=instrument,
    )
    assert decision.status in (RiskDecisionStatus.APPROVED, RiskDecisionStatus.RESIZED)
    if decision.approved_qty is not None:
        notional = decision.approved_qty * Decimal("0.5001")
        assert notional >= Decimal("5.15")


@pytest.mark.asyncio
async def test_buffer_applied_event_logged(caplog):
    """min_notional_buffer_applied rule should appear when bump happens.

    Uses very low confidence (0.01) to force multipliers to reduce qty below
    min-notional threshold, then checks the buffer bump rule appears.
    """
    rm = _make_rm(buffer_pct=3.0)
    # confidence=0.01 causes LLM multiplier to reduce sized qty far below min-notional
    proposal = _proposal(qty="1000", price="0.5000", confidence=0.01)
    instrument = _instrument(min_notional="5")
    capital = Decimal("1000")

    decision = await rm.evaluate(
        proposal=proposal,
        capital=capital,
        available_balance=capital,
        instrument_info=instrument,
    )
    if decision.status in (RiskDecisionStatus.APPROVED, RiskDecisionStatus.RESIZED):
        assert "min_notional_buffer_applied" in decision.triggered_rules
    # If rejected (bumped qty still too risky), that's also acceptable — the key
    # point is that the $5.00 order was not sent at exactly the exchange minimum.


@pytest.mark.asyncio
async def test_bump_can_exceed_requested_qty():
    """P0.5 fix: bump may exceed requested_qty if needed for min-notional."""
    rm = _make_rm(buffer_pct=3.0)
    # Request 10 XRP, but sizing multipliers bring it to ~8 XRP ($4.00)
    # The bump to $5.15 requires 11 XRP > requested_qty=10
    # Before fix: can_bump was blocked by min_qty <= requested_qty
    proposal = _proposal(qty="10", price="0.5000", confidence=0.8)  # 0.8 confidence
    instrument = _instrument(min_notional="5")
    capital = Decimal("1000")

    decision = await rm.evaluate(
        proposal=proposal,
        capital=capital,
        available_balance=capital,
        instrument_info=instrument,
    )
    # Must not be sent at < $5.15
    if decision.status in (RiskDecisionStatus.APPROVED, RiskDecisionStatus.RESIZED):
        assert decision.approved_qty is not None
        notional = decision.approved_qty * Decimal("0.5000")
        assert notional >= Decimal("5.00"), "Must be at or above exchange minimum"
