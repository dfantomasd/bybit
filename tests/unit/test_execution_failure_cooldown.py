"""Tests for execution engine failure cooldown and sub-minimum notional guard."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.domain.enums import MarketRegime, MarketType, OrderSide, RiskDecisionStatus
from trader.domain.models import InstrumentInfo, RiskDecision, TradeProposal
from trader.execution.engine import _DEFAULT_FAILURE_COOLDOWN_S, ExecutionEngine

# ---------------------------------------------------------------------------
# Helpers (mirrors test_execution_engine.py helpers)
# ---------------------------------------------------------------------------


def _instrument() -> InstrumentInfo:
    return InstrumentInfo(
        symbol="DOGEUSDT",
        market_type=MarketType.LINEAR,
        base_coin="DOGE",
        quote_coin="USDT",
        min_order_qty=Decimal("1"),
        max_order_qty=Decimal("10000"),
        qty_step=Decimal("1"),
        tick_size=Decimal("0.0001"),
        min_notional=Decimal("5"),
    )


def _proposal(symbol: str = "DOGEUSDT") -> TradeProposal:
    return TradeProposal(
        strategy_id="test",
        symbol=symbol,
        market_type=MarketType.LINEAR,
        side=OrderSide.BUY,
        requested_qty=Decimal("50"),
        entry_price=Decimal("0.20"),
        stop_loss=Decimal("0.196"),
        take_profit=Decimal("0.210"),
        confidence=0.70,
        regime=MarketRegime.BULL_TREND,
    )


def _approved_decision(proposal: TradeProposal) -> RiskDecision:
    return RiskDecision(
        proposal_id=proposal.proposal_id,
        status=RiskDecisionStatus.APPROVED,
        approved_qty=Decimal("50"),
        portfolio_heat=0.05,
        current_drawdown_pct=0.0,
        open_positions_count=0,
    )


def _make_engine(shadow_mode: bool = False, failure_cooldown_s: int = 30) -> ExecutionEngine:
    adapter = MagicMock()
    adapter.get_positions = AsyncMock(return_value=[])
    adapter.get_instrument_info = AsyncMock(return_value=_instrument())
    adapter.place_order = AsyncMock(return_value={"result": {"orderId": "live-001"}})
    adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("10"))

    risk_manager = MagicMock()
    exposure = MagicMock()
    exposure.update_position = AsyncMock()
    exposure.remove_position = AsyncMock()

    def _decide(proposal, **_kw):
        return _approved_decision(proposal)

    risk_manager.evaluate = AsyncMock(side_effect=_decide)

    return ExecutionEngine(
        adapter=adapter,
        risk_manager=risk_manager,
        exposure_tracker=exposure,
        shadow_mode=shadow_mode,
        cooldown_s=0,  # disable entry cooldown for these tests
        failure_cooldown_s=failure_cooldown_s,
        live_armed=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFailureCooldown:
    @pytest.mark.asyncio
    async def test_api_failure_does_not_set_entry_cooldown(self):
        """A Bybit API rejection must NOT populate _last_entry_at."""
        engine = _make_engine(shadow_mode=False)
        engine._adapter.place_order = AsyncMock(side_effect=RuntimeError("110094 min notional"))
        proposal = _proposal()

        result = await engine.submit(proposal, Decimal("100"), Decimal("100"))

        assert result is None
        # Entry cooldown must NOT be set
        assert "DOGEUSDT" not in engine._last_entry_at
        # Failure cooldown MUST be set
        assert "DOGEUSDT" in engine._last_failure_at

    @pytest.mark.asyncio
    async def test_api_failure_creates_failure_cooldown(self):
        """After an API failure the symbol is blocked for failure_cooldown_s seconds."""
        engine = _make_engine(shadow_mode=False, failure_cooldown_s=300)
        engine._adapter.place_order = AsyncMock(side_effect=RuntimeError("exchange error"))
        proposal = _proposal()

        # First submission fails
        await engine.submit(proposal, Decimal("100"), Decimal("100"))
        assert "DOGEUSDT" in engine._last_failure_at

        # Second submission immediately → skipped due to failure cooldown
        result2 = await engine.submit(proposal, Decimal("100"), Decimal("100"))
        assert result2 is None
        # place_order called only once
        engine._adapter.place_order.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_successful_order_sets_entry_cooldown_not_failure(self):
        """A successful live order sets _last_entry_at, NOT _last_failure_at."""
        engine = _make_engine(shadow_mode=False)
        proposal = _proposal()

        await engine.submit(proposal, Decimal("100"), Decimal("100"))

        # Entry cooldown set after successful submission
        assert "DOGEUSDT" in engine._last_entry_at
        # Failure cooldown must NOT be set on success
        assert "DOGEUSDT" not in engine._last_failure_at

    @pytest.mark.asyncio
    async def test_default_failure_cooldown_constant(self):
        """_DEFAULT_FAILURE_COOLDOWN_S must be 60 (1 minute)."""
        assert _DEFAULT_FAILURE_COOLDOWN_S == 60

    @pytest.mark.asyncio
    async def test_api_not_called_after_failure_cooldown(self):
        """API is not called again while failure cooldown is active."""
        engine = _make_engine(shadow_mode=False, failure_cooldown_s=9999)
        engine._adapter.place_order = AsyncMock(side_effect=RuntimeError("error"))
        proposal = _proposal()

        # Trigger failure
        await engine.submit(proposal, Decimal("100"), Decimal("100"))
        call_count_after_failure = engine._adapter.place_order.await_count

        # Try again immediately — should be blocked by failure cooldown
        await engine.submit(proposal, Decimal("100"), Decimal("100"))
        assert engine._adapter.place_order.await_count == call_count_after_failure
