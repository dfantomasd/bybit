"""Tests for post-multiplier min-notional guard in RiskManager."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from trader.domain.enums import MarketRegime, MarketType, OrderSide, RiskDecisionStatus
from trader.domain.models import InstrumentInfo, TradeProposal
from trader.risk.manager import RiskManager
from trader.risk.profiles import RiskProfile, get_risk_limits

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _instrument(
    min_notional: str = "5",
    min_order_qty: str = "1",
    max_order_qty: str = "1000",
    qty_step: str = "1",
    tick_size: str = "0.0001",
) -> InstrumentInfo:
    return InstrumentInfo(
        symbol="DOGEUSDT",
        market_type=MarketType.LINEAR,
        base_coin="DOGE",
        quote_coin="USDT",
        min_order_qty=Decimal(min_order_qty),
        max_order_qty=Decimal(max_order_qty),
        qty_step=Decimal(qty_step),
        tick_size=Decimal(tick_size),
        min_notional=Decimal(min_notional),
    )


def _proposal(
    qty: str = "10",
    entry: str = "0.20",
    confidence: float = 0.30,  # low → multiplier shrinks qty
    expected_risk: float | None = None,
) -> TradeProposal:
    entry_d = Decimal(entry)
    return TradeProposal(
        strategy_id="test",
        symbol="DOGEUSDT",
        market_type=MarketType.LINEAR,
        side=OrderSide.BUY,
        requested_qty=Decimal(qty),
        entry_price=entry_d,
        stop_loss=entry_d * Decimal("0.98"),  # 2% stop
        take_profit=entry_d * Decimal("1.04"),
        confidence=confidence,
        expected_risk=expected_risk,
        regime=MarketRegime.BULL_TREND,
    )


def _make_manager(profile: RiskProfile = RiskProfile.MODERATE) -> tuple[RiskManager, MagicMock, MagicMock]:
    get_risk_limits(profile)
    drawdown = MagicMock()
    drawdown.drawdown_pct = Decimal("0")
    drawdown.is_at_hard_stop = MagicMock(return_value=False)

    exposure = MagicMock()
    exposure.position_count = 0
    exposure.total_exposure_pct = Decimal("0")
    exposure.can_add_position = MagicMock(return_value=(True, ""))
    # New methods added by P0.2/P0.3 fix
    exposure.remaining_position_exposure_usd = MagicMock(return_value=Decimal("1000"))
    exposure.remaining_total_exposure_usd = MagicMock(return_value=Decimal("1000"))

    breakers = MagicMock()
    breakers.should_emergency = MagicMock(return_value=False)
    breakers.should_block_entries = MagicMock(return_value=False)
    breakers.should_safe_mode = MagicMock(return_value=False)

    kill_switch = MagicMock()
    kill_switch.is_active = False

    manager = RiskManager(
        risk_profile=profile,
        drawdown_tracker=drawdown,
        exposure_tracker=exposure,
        circuit_breaker_manager=breakers,
        kill_switch=kill_switch,
    )
    return manager, drawdown, exposure


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPostMultiplierMinNotional:
    @pytest.mark.asyncio
    async def test_qty_below_min_notional_triggers_floor_bump(self):
        """After multipliers reduce qty, manager bumps it to meet min_notional.

        Setup:
          capital=$100, risk_pct≈1% (MODERATE), stop_distance=2%
          raw_qty = (100 * 0.01) / (0.02 * 0.20) = 250 units
          confidence=0.06 → qty_after_llm = 250 * 0.06 = 15 → round_down → 15
          notional = 15 * 0.20 = $3.00 < $5 min_notional → floor applied
          min_qty = ceil(5 / 0.20) = 25 units
          notional_after_bump = 25 * 0.20 = $5 ≥ min_notional ✓
        """
        manager, _, _ = _make_manager()
        info = _instrument(min_notional="5", min_order_qty="1", qty_step="1")

        # confidence=0.06 → multiplier 0.06; qty=15 → notional=$3 < $5
        proposal = _proposal(qty="300", entry="0.20", confidence=0.06)

        decision = await manager.evaluate(
            proposal=proposal,
            capital=Decimal("100"),
            available_balance=Decimal("100"),
            instrument_info=info,
        )

        assert decision.status in (RiskDecisionStatus.APPROVED, RiskDecisionStatus.RESIZED)
        assert decision.approved_qty is not None
        assert decision.approved_qty * Decimal("0.20") >= Decimal("5"), (
            f"notional {decision.approved_qty * Decimal('0.20')} should be >= 5"
        )
        assert "min_notional_buffer_applied" in (decision.triggered_rules or [])

    @pytest.mark.asyncio
    async def test_expected_risk_multiplies_signal_confidence(self):
        """LLM risk multiplier must not replace the signal confidence."""
        manager, _, _ = _make_manager()
        info = _instrument(
            min_notional="1",
            min_order_qty="1",
            max_order_qty="1000",
            qty_step="1",
        )
        proposal = _proposal(qty="300", entry="10", confidence=0.50, expected_risk=0.40)

        decision = await manager.evaluate(
            proposal=proposal,
            capital=Decimal("1000"),
            available_balance=Decimal("1000"),
            instrument_info=info,
        )

        assert decision.status in (RiskDecisionStatus.APPROVED, RiskDecisionStatus.RESIZED)
        assert decision.approved_qty == Decimal("10")

    @pytest.mark.asyncio
    async def test_expected_risk_defaults_to_one_when_missing(self):
        """Missing LLM multiplier should leave the signal confidence multiplier intact."""
        manager, _, _ = _make_manager()
        info = _instrument(
            min_notional="1",
            min_order_qty="1",
            max_order_qty="1000",
            qty_step="1",
        )
        proposal = _proposal(qty="300", entry="10", confidence=0.50, expected_risk=None)

        decision = await manager.evaluate(
            proposal=proposal,
            capital=Decimal("1000"),
            available_balance=Decimal("1000"),
            instrument_info=info,
        )

        assert decision.status in (RiskDecisionStatus.APPROVED, RiskDecisionStatus.RESIZED)
        assert decision.approved_qty == Decimal("25")

    @pytest.mark.asyncio
    async def test_ceil_to_step_rounds_up_correctly(self):
        """min_notional floor qty is rounded UP to qty_step, not down."""
        from trader.risk.manager import _ceil_to_step

        # 25 / 7 = 3.57… → ceil to step=1 → 4
        result = _ceil_to_step(Decimal("25") / Decimal("7"), Decimal("1"))
        assert result == Decimal("4")

        # 5.001 with step=0.01 → 5.01
        result2 = _ceil_to_step(Decimal("5.001"), Decimal("0.01"))
        assert result2 == Decimal("5.01")

    @pytest.mark.asyncio
    async def test_floor_bump_rejected_when_hard_cap_would_be_violated(self):
        """Bumping qty to meet min_notional is rejected if it violates hard cap risk."""
        manager, _, _ = _make_manager()

        # Entry price=1.0, min_notional=$5 → min_qty=5; hard cap at tiny capital
        info = _instrument(min_notional="5", min_order_qty="1", qty_step="1")
        # Very small capital → hard cap risk = capital * hard_cap_pct / 100 will be tiny
        proposal = _proposal(qty="5", entry="1.0", confidence=0.10)

        decision = await manager.evaluate(
            proposal=proposal,
            capital=Decimal("1"),  # only $1 capital
            available_balance=Decimal("1"),
            instrument_info=info,
        )

        # Should be rejected — can't bump without violating hard cap
        assert decision.status == RiskDecisionStatus.REJECTED

    @pytest.mark.asyncio
    async def test_floor_bump_rejected_when_portfolio_exposure_exceeded(self):
        """Bumping qty to meet min_notional is rejected if it exceeds portfolio exposure budget."""
        manager, _, exposure = _make_manager()
        # Simulate near-full exposure
        exposure.total_exposure_pct = Decimal("99")

        info = _instrument(min_notional="5", min_order_qty="1", qty_step="1")
        proposal = _proposal(qty="100", entry="0.20", confidence=0.10)

        decision = await manager.evaluate(
            proposal=proposal,
            capital=Decimal("1000"),
            available_balance=Decimal("1000"),
            instrument_info=info,
        )

        # exposure cap check (step 8-9) should reject before we even get to min_notional
        # but if it doesn't, min_notional bump should also be rejected
        assert decision.status == RiskDecisionStatus.REJECTED

    @pytest.mark.asyncio
    async def test_floor_bump_rejected_when_available_balance_insufficient(self):
        """Bumping qty is rejected when min_notional exceeds available_balance."""
        manager, _, _ = _make_manager()

        # min_notional=$5, entry=1.0 → need 5 units; but balance is only $4
        info = _instrument(min_notional="5", min_order_qty="1", qty_step="1")
        proposal = _proposal(qty="5", entry="1.0", confidence=0.10)

        decision = await manager.evaluate(
            proposal=proposal,
            capital=Decimal("100"),
            available_balance=Decimal("4"),  # less than min_notional
            instrument_info=info,
        )

        assert decision.status == RiskDecisionStatus.REJECTED

    @pytest.mark.asyncio
    async def test_notional_ok_no_bump_needed(self):
        """When final notional is already >= min_notional, no floor rule is triggered."""
        manager, _, _ = _make_manager()

        # confidence=0.80, entry=1.0, qty=10 → notional=$8 → above $5
        info = _instrument(min_notional="5", min_order_qty="1", qty_step="1")
        proposal = _proposal(qty="10", entry="1.0", confidence=0.80)

        decision = await manager.evaluate(
            proposal=proposal,
            capital=Decimal("1000"),
            available_balance=Decimal("1000"),
            instrument_info=info,
        )

        assert decision.status in (RiskDecisionStatus.APPROVED, RiskDecisionStatus.RESIZED)
        assert "min_notional_buffer_applied" not in (decision.triggered_rules or [])
        assert "post_multiplier_min_notional_rejected" not in (decision.triggered_rules or [])
