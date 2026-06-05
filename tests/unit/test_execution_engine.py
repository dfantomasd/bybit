"""Tests for ExecutionEngine."""
from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trader.domain.enums import MarketRegime, MarketType, OrderSide, RiskDecisionStatus
from trader.domain.models import (
    InstrumentInfo,
    RiskDecision,
    TradeProposal,
)
from trader.execution.engine import ExecutionEngine


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _instrument_info(symbol: str = "BTCUSDT") -> InstrumentInfo:
    return InstrumentInfo(
        symbol=symbol,
        market_type=MarketType.LINEAR,
        base_coin="BTC",
        quote_coin="USDT",
        min_order_qty=Decimal("0.001"),
        max_order_qty=Decimal("100"),
        qty_step=Decimal("0.001"),
        tick_size=Decimal("0.5"),
        min_notional=Decimal("5"),
    )


def _proposal(
    symbol: str = "BTCUSDT",
    side: OrderSide = OrderSide.BUY,
    qty: Decimal = Decimal("0.01"),
    confidence: float = 0.70,
) -> TradeProposal:
    return TradeProposal(
        strategy_id="test_strategy",
        symbol=symbol,
        market_type=MarketType.LINEAR,
        side=side,
        requested_qty=qty,
        entry_price=Decimal("50000"),
        stop_loss=Decimal("49000"),
        take_profit=Decimal("52000"),
        confidence=confidence,
        regime=MarketRegime.BULL_TREND,
    )


def _approved_decision(proposal: TradeProposal, qty: Decimal | None = None) -> RiskDecision:
    return RiskDecision(
        proposal_id=proposal.proposal_id,
        status=RiskDecisionStatus.APPROVED,
        approved_qty=qty or proposal.requested_qty,
        portfolio_heat=0.05,
        current_drawdown_pct=0.0,
        open_positions_count=0,
    )


def _rejected_decision(proposal: TradeProposal) -> RiskDecision:
    return RiskDecision(
        proposal_id=proposal.proposal_id,
        status=RiskDecisionStatus.REJECTED,
        reason="daily_loss_limit",
        triggered_rules=["daily_loss_limit"],
        portfolio_heat=0.0,
        current_drawdown_pct=0.0,
        open_positions_count=0,
    )


def _make_engine(
    approved: bool = True,
    shadow_mode: bool = True,
    qty: Decimal | None = None,
) -> ExecutionEngine:
    adapter = MagicMock()
    adapter.get_positions = AsyncMock(return_value=[])
    adapter.get_instrument_info = AsyncMock(return_value=_instrument_info())
    adapter.place_order = AsyncMock(return_value={"result": {"orderId": "test-123"}})

    risk_manager = MagicMock()
    exposure = MagicMock()
    exposure.update_position = AsyncMock()
    exposure.remove_position = AsyncMock()

    def make_decision(proposal, **kwargs):
        if approved:
            return _approved_decision(proposal, qty)
        return _rejected_decision(proposal)

    risk_manager.evaluate = AsyncMock(side_effect=make_decision)

    engine = ExecutionEngine(
        adapter=adapter,
        risk_manager=risk_manager,
        exposure_tracker=exposure,
        shadow_mode=shadow_mode,
        cooldown_s=0,  # disable cooldown for tests
    )
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExecutionEngine:
    def test_no_open_position_initially(self):
        engine = _make_engine()
        assert not engine.has_open_position("BTCUSDT")
        assert engine.open_position_count() == 0

    @pytest.mark.asyncio
    async def test_approved_proposal_recorded(self):
        engine = _make_engine(approved=True, shadow_mode=True)
        proposal = _proposal()
        decision = await engine.submit(
            proposal=proposal,
            capital=Decimal("10000"),
            available_balance=Decimal("10000"),
        )
        assert decision is not None
        assert decision.status == RiskDecisionStatus.APPROVED
        assert engine.has_open_position("BTCUSDT")

    @pytest.mark.asyncio
    async def test_rejected_proposal_not_recorded(self):
        engine = _make_engine(approved=False)
        proposal = _proposal()
        decision = await engine.submit(
            proposal=proposal,
            capital=Decimal("10000"),
            available_balance=Decimal("10000"),
        )
        assert decision is not None
        assert decision.status == RiskDecisionStatus.REJECTED
        assert not engine.has_open_position("BTCUSDT")

    @pytest.mark.asyncio
    async def test_duplicate_position_skipped(self):
        engine = _make_engine(approved=True)
        proposal = _proposal()

        d1 = await engine.submit(proposal, Decimal("10000"), Decimal("10000"))
        assert d1 is not None

        # Second proposal for same symbol → skipped
        d2 = await engine.submit(proposal, Decimal("10000"), Decimal("10000"))
        assert d2 is None
        assert engine.open_position_count() == 1

    @pytest.mark.asyncio
    async def test_shadow_mode_does_not_call_place_order(self):
        engine = _make_engine(approved=True, shadow_mode=True)
        proposal = _proposal()
        await engine.submit(proposal, Decimal("10000"), Decimal("10000"))
        engine._adapter.place_order.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_live_mode_calls_place_order(self):
        engine = _make_engine(approved=True, shadow_mode=False)
        proposal = _proposal()
        await engine.submit(proposal, Decimal("10000"), Decimal("10000"))
        engine._adapter.place_order.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exposure_tracker_updated_on_approval(self):
        engine = _make_engine(approved=True, shadow_mode=True)
        proposal = _proposal()
        await engine.submit(proposal, Decimal("10000"), Decimal("10000"))
        engine._exposure.update_position.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_record_position_closed_clears_state(self):
        engine = _make_engine(approved=True)
        proposal = _proposal()
        await engine.submit(proposal, Decimal("10000"), Decimal("10000"))
        assert engine.has_open_position("BTCUSDT")

        engine.record_position_closed("BTCUSDT")
        assert not engine.has_open_position("BTCUSDT")

    @pytest.mark.asyncio
    async def test_different_symbols_independent(self):
        engine = _make_engine(approved=True)
        engine._instrument_cache["ETHUSDT"] = _instrument_info("ETHUSDT")

        btc = _proposal("BTCUSDT")
        eth = _proposal("ETHUSDT")

        await engine.submit(btc, Decimal("10000"), Decimal("10000"))
        await engine.submit(eth, Decimal("10000"), Decimal("10000"))

        assert engine.has_open_position("BTCUSDT")
        assert engine.has_open_position("ETHUSDT")
        assert engine.open_position_count() == 2

    @pytest.mark.asyncio
    async def test_sync_positions_populates_registry(self):
        from trader.domain.models import Position

        pos = Position(
            symbol="BTCUSDT",
            market_type=MarketType.LINEAR,
            side=OrderSide.BUY,
            size=Decimal("0.01"),
            entry_price=Decimal("50000"),
        )
        engine = _make_engine()
        engine._adapter.get_positions = AsyncMock(return_value=[pos])

        await engine.sync_positions()
        assert engine.has_open_position("BTCUSDT")

    def test_get_status_returns_dict(self):
        engine = _make_engine()
        status = engine.get_status()
        assert "shadow_mode" in status
        assert "open_positions" in status
        assert "cooldown_s" in status
