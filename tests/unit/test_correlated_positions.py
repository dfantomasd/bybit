"""Tests for correlated position family cap."""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from trader.domain.enums import MarketRegime, MarketType, OrderSide, RiskProfile
from trader.domain.models import InstrumentInfo, TradeProposal
from trader.risk.circuit_breakers import CircuitBreakerManager
from trader.risk.drawdown import DrawdownTracker
from trader.risk.exposure import ExposureTracker
from trader.risk.kill_switch import KillSwitch
from trader.risk.manager import RiskManager
from trader.risk.profiles import get_risk_limits


def _instrument(symbol: str = "RBTCUSDT") -> InstrumentInfo:
    return InstrumentInfo(
        symbol=symbol,
        market_type=MarketType.LINEAR,
        base_coin="BTC",
        quote_coin="USDT",
        min_order_qty=Decimal("0.001"),
        max_order_qty=Decimal("1000"),
        qty_step=Decimal("0.001"),
        tick_size=Decimal("0.01"),
        min_notional=Decimal("5"),
        max_leverage=Decimal("10"),
    )


def _proposal(symbol: str = "BTCUSDT") -> TradeProposal:
    return TradeProposal(
        proposal_id=uuid4(),
        strategy_id="scalp_micro_v1",
        symbol=symbol,
        side=OrderSide.BUY,
        market_type=MarketType.LINEAR,
        confidence=0.8,
        entry_price=Decimal("100"),
        take_profit=Decimal("101"),
        stop_loss=Decimal("99"),
        requested_qty=Decimal("1"),
        regime=MarketRegime.BULL_TREND,
        rationale="test",
    )


@pytest.mark.asyncio
async def test_correlated_family_blocks_third_position() -> None:
    limits = get_risk_limits(RiskProfile.SCALP)
    exposure = ExposureTracker(total_capital=Decimal("1000"), risk_limits=limits)
    await exposure.update_position("BTCUSDT", "Buy", Decimal("100"), leverage=Decimal("10"))
    await exposure.update_position("WBTCUSDT", "Buy", Decimal("100"), leverage=Decimal("10"))

    manager = RiskManager(
        risk_profile=RiskProfile.SCALP,
        drawdown_tracker=DrawdownTracker(initial_equity=Decimal("1000")),
        exposure_tracker=exposure,
        circuit_breaker_manager=CircuitBreakerManager(risk_limits=limits),
        kill_switch=KillSwitch(),
        max_correlated_positions=2,
    )
    decision = await manager.evaluate(
        _proposal("RBTCUSDT"),
        capital=Decimal("1000"),
        available_balance=Decimal("1000"),
        instrument_info=_instrument("RBTCUSDT"),
    )
    assert decision.status.value == "REJECTED"
    assert "max_correlated_positions" in decision.triggered_rules
