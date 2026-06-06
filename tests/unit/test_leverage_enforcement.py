"""Tests for P0.7: leverage enforcement before live order placement."""

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


def _proposal() -> TradeProposal:
    return TradeProposal(
        strategy_id="test",
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        side=OrderSide.BUY,
        requested_qty=Decimal("0.01"),
        entry_price=Decimal("50000"),
        stop_loss=Decimal("49000"),
        take_profit=Decimal("52000"),
        confidence=0.70,
        regime=MarketRegime.BULL_TREND,
    )


def _make_engine(max_leverage: Decimal = Decimal("5")) -> ExecutionEngine:
    adapter = MagicMock()
    adapter.get_positions = AsyncMock(return_value=[])
    adapter.get_instrument_info = AsyncMock(return_value=_instrument())
    adapter.place_order = AsyncMock(return_value={"result": {"orderId": "x"}})
    adapter._rest = MagicMock()
    adapter._rest.set_leverage = AsyncMock()

    risk_manager = MagicMock()
    risk_manager._limits = MagicMock()
    risk_manager._limits.max_leverage = max_leverage

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
        shadow_mode=False,
        cooldown_s=0,
        failure_cooldown_s=0,
    )


class TestLeverageEnforcement:
    @pytest.mark.asyncio
    async def test_leverage_set_before_first_order(self):
        """set_leverage is called before placing the first order for a symbol."""
        engine = _make_engine(max_leverage=Decimal("3"))
        await engine.submit(_proposal(), Decimal("10000"), Decimal("10000"))
        engine._adapter._rest.set_leverage.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_leverage_not_reset_on_second_order(self):
        """set_leverage is NOT called again after being confirmed for a symbol."""
        engine = _make_engine(max_leverage=Decimal("3"))
        # First order — sets leverage
        await engine.submit(_proposal(), Decimal("10000"), Decimal("10000"))
        first_call_count = engine._adapter._rest.set_leverage.await_count

        # Reset open positions so a second order can go through
        engine._open_positions.clear()
        engine._last_entry_at.clear()

        await engine.submit(_proposal(), Decimal("10000"), Decimal("10000"))
        # Same call count — leverage not set again
        assert engine._adapter._rest.set_leverage.await_count == first_call_count

    @pytest.mark.asyncio
    async def test_leverage_not_called_in_shadow_mode(self):
        """set_leverage is never called in shadow mode."""
        engine = _make_engine(max_leverage=Decimal("5"))
        engine._shadow_mode = True
        await engine.submit(_proposal(), Decimal("10000"), Decimal("10000"))
        engine._adapter._rest.set_leverage.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_leverage_confirmed_cached(self):
        """Confirmed leverage is cached in _leverage_confirmed dict."""
        engine = _make_engine(max_leverage=Decimal("5"))
        await engine.submit(_proposal(), Decimal("10000"), Decimal("10000"))
        assert "BTCUSDT" in engine._leverage_confirmed
        assert engine._leverage_confirmed["BTCUSDT"] == Decimal("5")

    @pytest.mark.asyncio
    async def test_leverage_failure_does_not_block_order(self):
        """If set_leverage fails, the order still proceeds (best-effort enforcement)."""
        engine = _make_engine(max_leverage=Decimal("3"))
        engine._adapter._rest.set_leverage = AsyncMock(side_effect=RuntimeError("API error"))
        result = await engine.submit(_proposal(), Decimal("10000"), Decimal("10000"))
        # Order should still go through despite leverage failure
        assert result is not None
        engine._adapter.place_order.assert_awaited_once()
