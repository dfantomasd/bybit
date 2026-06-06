"""Tests: P0.2 – oversized request must be RESIZED, not rejected."""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

from trader.domain.enums import (
    MarketRegime,
    MarketType,
    OrderSide,
    RiskDecisionStatus,
    RiskProfile,
)
from trader.domain.models import (
    InstrumentInfo,
    TradeProposal,
)
from trader.risk.circuit_breakers import CircuitBreakerManager
from trader.risk.drawdown import DrawdownTracker
from trader.risk.exposure import ExposureTracker
from trader.risk.kill_switch import KillSwitch
from trader.risk.manager import RiskManager
from trader.risk.profiles import get_risk_limits


def _make_instrument(
    min_qty: str = "0.001",
    max_qty: str = "1000",
    qty_step: str = "0.001",
    min_notional: str = "5",
    tick_size: str = "0.1",
    max_leverage: float = 10.0,
) -> InstrumentInfo:
    return InstrumentInfo(
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        base_coin="BTC",
        quote_coin="USDT",
        min_order_qty=Decimal(min_qty),
        max_order_qty=Decimal(max_qty),
        qty_step=Decimal(qty_step),
        tick_size=Decimal(tick_size),
        min_notional=Decimal(min_notional),
        max_leverage=Decimal(str(max_leverage)),
    )


def _make_proposal(
    requested_qty: str = "1.0",
    entry_price: str = "50000",
    confidence: float = 1.0,
) -> TradeProposal:
    return TradeProposal(
        proposal_id=uuid.uuid4(),
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        market_type=MarketType.LINEAR,
        requested_qty=Decimal(requested_qty),
        entry_price=Decimal(entry_price),
        stop_loss=Decimal("49000"),
        take_profit=Decimal("51000"),
        confidence=confidence,
        regime=MarketRegime.BULL_TREND,
    )


def _make_risk_manager(
    capital: Decimal = Decimal("10000"),
    exposure_pct: Decimal = Decimal("0"),
) -> tuple[RiskManager, ExposureTracker]:
    limits = get_risk_limits(RiskProfile.CONSERVATIVE)
    drawdown = DrawdownTracker(initial_equity=capital)
    exposure = ExposureTracker(total_capital=capital, risk_limits=limits)
    breakers = CircuitBreakerManager(risk_limits=limits)
    kill_switch = KillSwitch()

    rm = RiskManager(
        risk_profile=RiskProfile.CONSERVATIVE,
        drawdown_tracker=drawdown,
        exposure_tracker=exposure,
        circuit_breaker_manager=breakers,
        kill_switch=kill_switch,
    )
    return rm, exposure


@pytest.mark.asyncio
async def test_oversized_request_is_resized_not_rejected():
    """P0.2: requested_qty that exceeds budget should be RESIZED, not REJECTED."""
    capital = Decimal("1000")
    rm, exposure = _make_risk_manager(capital=capital)

    # Request 1 BTC at $50k = $50,000 notional — far exceeds $1000 capital
    proposal = _make_proposal(requested_qty="1.0", entry_price="50000", confidence=1.0)
    instrument = _make_instrument(min_qty="0.001", min_notional="5")

    decision = await rm.evaluate(
        proposal=proposal,
        capital=capital,
        available_balance=capital,
        instrument_info=instrument,
    )

    # Must not be rejected — should be approved or resized with smaller qty
    assert decision.status in (RiskDecisionStatus.APPROVED, RiskDecisionStatus.RESIZED), (
        f"Expected APPROVED or RESIZED, got {decision.status}: {decision.reason}"
    )
    assert decision.approved_qty is not None
    assert decision.approved_qty > Decimal("0")
    assert decision.approved_qty < Decimal("1.0"), "qty should be downsized"


@pytest.mark.asyncio
async def test_zero_remaining_budget_rejects():
    """P0.2: if total exposure is at 100% of cap, reject early."""
    capital = Decimal("1000")
    limits = get_risk_limits(RiskProfile.CONSERVATIVE)
    rm, exposure = _make_risk_manager(capital=capital)

    # Fill up the entire exposure budget
    await exposure.update_position(
        "ETHUSDT",
        "Buy",
        capital * limits.max_total_exposure_pct / Decimal("100"),
    )

    proposal = _make_proposal(requested_qty="0.01", entry_price="50000", confidence=1.0)
    instrument = _make_instrument()

    decision = await rm.evaluate(
        proposal=proposal,
        capital=capital,
        available_balance=capital,
        instrument_info=instrument,
    )

    assert decision.status == RiskDecisionStatus.REJECTED
    assert "exposure_cap_full" in decision.triggered_rules


@pytest.mark.asyncio
async def test_partial_budget_allows_smaller_order():
    """P0.2: partial remaining budget → order approved at reduced size."""
    capital = Decimal("1000")
    rm, exposure = _make_risk_manager(capital=capital)

    # Use 40% of budget already — leaves 30% remaining (within total cap of 70%)
    already_used_notional = capital * Decimal("0.40")
    await exposure.update_position("ETHUSDT", "Buy", already_used_notional)

    proposal = _make_proposal(requested_qty="0.5", entry_price="50000", confidence=1.0)
    instrument = _make_instrument(min_qty="0.001", min_notional="5")

    decision = await rm.evaluate(
        proposal=proposal,
        capital=capital,
        available_balance=capital,
        instrument_info=instrument,
    )

    # 30% remaining = $300; 0.5 BTC at $50k = $25,000 — too much
    # Should be resized, not rejected
    assert decision.status in (RiskDecisionStatus.APPROVED, RiskDecisionStatus.RESIZED), (
        f"Got {decision.status}: {decision.reason}"
    )
