"""Tests for P0.6: mandatory stop-loss validation in live mode."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.domain.enums import MarketRegime, MarketType, OrderSide, RiskDecisionStatus
from trader.domain.models import InstrumentInfo, RiskDecision, TradeProposal
from trader.execution.engine import ExecutionEngine


def _instrument() -> InstrumentInfo:
    return InstrumentInfo(
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        base_coin="BTC",
        quote_coin="USDT",
        min_order_qty=Decimal("0.001"),
        max_order_qty=Decimal("100"),
        qty_step=Decimal("0.001"),
        tick_size=Decimal("0.1"),
        min_notional=Decimal("5"),
    )


def _proposal_no_sl(symbol: str = "BTCUSDT") -> TradeProposal:
    return TradeProposal(
        strategy_id="test",
        symbol=symbol,
        market_type=MarketType.LINEAR,
        side=OrderSide.BUY,
        requested_qty=Decimal("0.01"),
        entry_price=Decimal("50000"),
        stop_loss=None,  # <-- no SL
        take_profit=Decimal("52000"),
        confidence=0.70,
        regime=MarketRegime.BULL_TREND,
    )


def _proposal_with_sl(
    symbol: str = "BTCUSDT",
    side: OrderSide = OrderSide.BUY,
    entry: Decimal = Decimal("50000"),
    sl: Decimal = Decimal("49000"),
) -> TradeProposal:
    tp = entry * Decimal("1.04") if side == OrderSide.BUY else entry * Decimal("0.96")
    return TradeProposal(
        strategy_id="test",
        symbol=symbol,
        market_type=MarketType.LINEAR,
        side=side,
        requested_qty=Decimal("0.01"),
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        confidence=0.70,
        regime=MarketRegime.BULL_TREND,
    )


def _make_engine(shadow_mode: bool = True) -> ExecutionEngine:
    adapter = MagicMock()
    adapter.get_positions = AsyncMock(return_value=[])
    adapter.get_instrument_info = AsyncMock(return_value=_instrument())
    adapter.place_order = AsyncMock(return_value={"result": {"orderId": "x"}})
    adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("50000"))
    adapter._rest = MagicMock()
    adapter._rest.set_leverage = AsyncMock()

    risk_manager = MagicMock()
    risk_manager._limits = MagicMock()
    risk_manager._limits.max_leverage = Decimal("5")

    def _approve(proposal, **_kw):
        return RiskDecision(
            proposal_id=proposal.proposal_id,
            status=RiskDecisionStatus.APPROVED,
            approved_qty=Decimal("0.01"),
            portfolio_heat=0.05,
            current_drawdown_pct=0.0,
            open_positions_count=0,
        )

    risk_manager.evaluate = AsyncMock(side_effect=_approve)

    exposure = MagicMock()
    exposure.update_position = AsyncMock()
    exposure.remove_position = AsyncMock()

    return ExecutionEngine(
        adapter=adapter,
        risk_manager=risk_manager,
        exposure_tracker=exposure,
        shadow_mode=shadow_mode,
        cooldown_s=0,
        failure_cooldown_s=0,
    )


class TestSLValidation:
    @pytest.mark.asyncio
    async def test_live_mode_rejects_proposal_without_sl(self):
        """In live mode, a proposal with no stop_loss must be rejected."""
        engine = _make_engine(shadow_mode=False)
        result = await engine.submit(_proposal_no_sl(), Decimal("10000"), Decimal("10000"))
        assert result is None
        engine._adapter.place_order.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_shadow_mode_allows_proposal_without_sl(self):
        """In shadow mode, a proposal with no stop_loss is allowed for simulation."""
        engine = _make_engine(shadow_mode=True)
        result = await engine.submit(_proposal_no_sl(), Decimal("10000"), Decimal("10000"))
        # Shadow mode: should reach risk evaluation (result is a RiskDecision)
        assert result is not None

    @pytest.mark.asyncio
    async def test_live_mode_rejects_sl_on_wrong_side_buy(self):
        """BUY order: SL above entry price must be rejected by engine."""
        engine = _make_engine(shadow_mode=False)
        # Bypass model validation to feed a bad proposal directly to the engine
        proposal = TradeProposal.model_construct(
            strategy_id="test",
            symbol="BTCUSDT",
            market_type=MarketType.LINEAR,
            side=OrderSide.BUY,
            requested_qty=Decimal("0.01"),
            entry_price=Decimal("50000"),
            stop_loss=Decimal("51000"),  # wrong side: above entry for BUY
            take_profit=Decimal("52000"),
            confidence=0.70,
            regime=MarketRegime.BULL_TREND,
        )
        result = await engine.submit(proposal, Decimal("10000"), Decimal("10000"))
        assert result is None
        engine._adapter.place_order.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_live_mode_rejects_sl_on_wrong_side_sell(self):
        """SELL order: SL below entry price must be rejected by engine."""
        engine = _make_engine(shadow_mode=False)
        proposal = TradeProposal.model_construct(
            strategy_id="test",
            symbol="BTCUSDT",
            market_type=MarketType.LINEAR,
            side=OrderSide.SELL,
            requested_qty=Decimal("0.01"),
            entry_price=Decimal("50000"),
            stop_loss=Decimal("49000"),  # wrong side: below entry for SELL
            take_profit=Decimal("48000"),
            confidence=0.70,
            regime=MarketRegime.BULL_TREND,
        )
        result = await engine.submit(proposal, Decimal("10000"), Decimal("10000"))
        assert result is None
        engine._adapter.place_order.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_live_mode_accepts_valid_buy_sl(self):
        """BUY order with SL below entry proceeds to risk evaluation."""
        engine = _make_engine(shadow_mode=False)
        proposal = _proposal_with_sl(
            side=OrderSide.BUY,
            entry=Decimal("50000"),
            sl=Decimal("49000"),  # correct side
        )
        result = await engine.submit(proposal, Decimal("10000"), Decimal("10000"))
        assert result is not None
        engine._adapter.place_order.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_live_mode_accepts_valid_sell_sl(self):
        """SELL order with SL above entry proceeds to risk evaluation."""
        engine = _make_engine(shadow_mode=False)
        proposal = _proposal_with_sl(
            side=OrderSide.SELL,
            entry=Decimal("50000"),
            sl=Decimal("51000"),  # correct side for SELL
        )
        result = await engine.submit(proposal, Decimal("10000"), Decimal("10000"))
        assert result is not None
