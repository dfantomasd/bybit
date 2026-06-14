"""Tests for RiskManager sizing order — premature rejection fix and final validation.

Key invariants verified:
- Step 8 only blocks when portfolio exposure is ALREADY at/above the cap.
- PositionSizer is allowed to reduce qty to fit the remaining budget (no pre-reject).
- Step 15.7 is the definitive exposure gate after all sizing and multipliers.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from trader.domain.enums import MarketRegime, MarketType, OrderSide, RiskDecisionStatus, RiskProfile
from trader.domain.models import InstrumentInfo, TradeProposal
from trader.risk.exposure import ExposureTracker
from trader.risk.manager import RiskManager
from trader.risk.profiles import get_risk_limits

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _instrument(
    symbol: str = "BTCUSDT",
    min_notional: str = "5",
    min_order_qty: str = "1",
    max_order_qty: str = "10000",
    qty_step: str = "1",
    tick_size: str = "0.0001",
) -> InstrumentInfo:
    return InstrumentInfo(
        symbol=symbol,
        market_type=MarketType.LINEAR,
        base_coin=symbol.replace("USDT", ""),
        quote_coin="USDT",
        min_order_qty=Decimal(min_order_qty),
        max_order_qty=Decimal(max_order_qty),
        qty_step=Decimal(qty_step),
        tick_size=Decimal(tick_size),
        min_notional=Decimal(min_notional),
    )


def _proposal(
    symbol: str = "BTCUSDT",
    qty: str = "100",
    entry: str = "10",
    stop_loss_pct: str = "0.02",
    confidence: float = 1.0,
) -> TradeProposal:
    entry_d = Decimal(entry)
    stop_d = entry_d * (Decimal("1") - Decimal(stop_loss_pct))
    return TradeProposal(
        strategy_id="test",
        symbol=symbol,
        market_type=MarketType.LINEAR,
        side=OrderSide.BUY,
        requested_qty=Decimal(qty),
        entry_price=entry_d,
        stop_loss=stop_d,
        take_profit=entry_d * Decimal("1.04"),
        confidence=confidence,
        regime=MarketRegime.BULL_TREND,
    )


def _make_manager_with_real_exposure(
    capital: Decimal,
    profile: RiskProfile = RiskProfile.SCALP,
) -> tuple[RiskManager, ExposureTracker]:
    limits = get_risk_limits(profile)
    exposure = ExposureTracker(total_capital=capital, risk_limits=limits)

    drawdown = MagicMock()
    drawdown.drawdown_pct = Decimal("0")
    drawdown.is_at_hard_stop = MagicMock(return_value=False)

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
    return manager, exposure


# ---------------------------------------------------------------------------
# Main bug-fix test: premature rejection
# ---------------------------------------------------------------------------


class TestPrematureRejectionFix:
    @pytest.mark.asyncio
    async def test_requested_qty_above_remaining_budget_is_resized_not_rejected(self):
        """Large requested_qty that exceeds exposure budget must be RESIZED, not REJECTED.

        Setup (SCALP profile, capital=$1000, max_total_exposure=90%):
          - ETHUSDT already open at $600 notional → 60% exposure used
          - Remaining budget: 30% = $300
          - Proposal: BTCUSDT, requested_qty=100 @ $10 = $1000 notional (100% — way above cap)

        OLD behaviour (steps 8-9 used requested_qty): would reject immediately because
        100 * $10 = $1000 > 90% cap → REJECTED.

        NEW behaviour (step 8 guards only when already at cap):
          - 60% < 90% → step 8 passes
          - PositionSizer caps qty to $300 / $10 = 30 units
          - Step 15.7 validates: $600 + $300 = $900 = 90% ≤ cap → approved
          - Result: RESIZED with approved_qty=30
        """
        capital = Decimal("1000")
        manager, exposure = _make_manager_with_real_exposure(capital)

        # Pre-load ETHUSDT at $600 notional
        await exposure.update_position("ETHUSDT", "Buy", Decimal("600"), leverage=Decimal("7"))
        assert exposure.total_exposure_pct == Decimal("60")

        info = _instrument("BTCUSDT", min_notional="5", min_order_qty="1", qty_step="1")
        proposal = _proposal("BTCUSDT", qty="100", entry="10", confidence=1.0)

        decision = await manager.evaluate(
            proposal=proposal,
            capital=capital,
            available_balance=Decimal("1000"),
            instrument_info=info,
        )

        assert decision.status == RiskDecisionStatus.RESIZED, (
            f"Expected RESIZED but got {decision.status}: {decision.reason}"
        )
        assert decision.approved_qty == Decimal("30"), (
            f"Expected 30 units (remaining budget), got {decision.approved_qty}"
        )
        assert "exposure_cap_reached" not in (decision.triggered_rules or [])

    @pytest.mark.asyncio
    async def test_exposure_already_at_cap_is_rejected_at_step_8(self):
        """Step 8 must reject immediately when portfolio is already at the exposure cap.

        SCALP max_total_exposure_pct=90%. If existing positions fill exactly 90%,
        no new entry is possible.
        """
        capital = Decimal("1000")
        manager, exposure = _make_manager_with_real_exposure(capital)

        # Fill to exactly 90%
        await exposure.update_position("ETHUSDT", "Buy", Decimal("900"), leverage=Decimal("7"))
        assert exposure.total_exposure_pct == Decimal("90")

        info = _instrument("BTCUSDT", min_notional="5", min_order_qty="1", qty_step="1")
        proposal = _proposal("BTCUSDT", qty="10", entry="10", confidence=1.0)

        decision = await manager.evaluate(
            proposal=proposal,
            capital=capital,
            available_balance=Decimal("1000"),
            instrument_info=info,
        )

        assert decision.status == RiskDecisionStatus.REJECTED
        assert "exposure_cap_reached" in (decision.triggered_rules or [])

    @pytest.mark.asyncio
    async def test_exposure_just_below_cap_passes_step_8_then_sized_down(self):
        """89% existing exposure: step 8 allows through; sizer caps qty to remaining 1%."""
        capital = Decimal("1000")
        manager, exposure = _make_manager_with_real_exposure(capital)

        await exposure.update_position("ETHUSDT", "Buy", Decimal("890"), leverage=Decimal("7"))
        assert exposure.total_exposure_pct == Decimal("89")

        # Proposal with large requested_qty — sizer will cap to remaining 1% = $10
        info = _instrument("BTCUSDT", min_notional="5", min_order_qty="1", qty_step="1")
        proposal = _proposal("BTCUSDT", qty="1000", entry="1", confidence=1.0)

        decision = await manager.evaluate(
            proposal=proposal,
            capital=capital,
            available_balance=Decimal("1000"),
            instrument_info=info,
        )

        # Should not be rejected for exposure — either RESIZED or APPROVED
        assert decision.status != RiskDecisionStatus.REJECTED or (
            "exposure_cap_reached" not in (decision.triggered_rules or [])
        ), f"Step 8 should not fire at 89%; got {decision.reason}"

    @pytest.mark.asyncio
    async def test_two_positions_near_cap_third_is_resized_not_rejected(self):
        """Reproduce the original production bug: 3rd position rejected prematurely.

        Simulates the ZECUSDT+HYPEUSDT scenario from the Render logs.
        Capital=$50, SCALP profile, two small positions open.
        """
        capital = Decimal("50")
        manager, exposure = _make_manager_with_real_exposure(capital)

        # Two small existing positions totalling ~34% exposure
        await exposure.update_position("ZECUSDT", "Buy", Decimal("8"), leverage=Decimal("7"))
        await exposure.update_position("HYPEUSDT", "Buy", Decimal("9"), leverage=Decimal("7"))
        total_pct = exposure.total_exposure_pct
        assert total_pct < Decimal("90"), f"Setup error: {total_pct}% >= 90%"

        # Third symbol with a large requested_qty (strategy sends raw qty before sizing)
        remaining_usd = exposure.remaining_gross_notional_usd(capital)
        assert remaining_usd > Decimal("0"), "Remaining budget should be positive"

        info = _instrument("SOMEUSDT", min_notional="5", min_order_qty="1", qty_step="1")
        # entry=$0.10, 1000 units = $100 requested notional (clearly exceeds remaining budget)
        proposal = _proposal("SOMEUSDT", qty="1000", entry="0.10", confidence=1.0)

        decision = await manager.evaluate(
            proposal=proposal,
            capital=capital,
            available_balance=Decimal("50"),
            instrument_info=info,
        )

        # Must not be rejected with exposure_cap_reached — should be resized or approved
        assert "exposure_cap_reached" not in (decision.triggered_rules or []), (
            f"Premature rejection at step 8: {decision.reason}"
        )


# ---------------------------------------------------------------------------
# Final exposure validation (step 15.7)
# ---------------------------------------------------------------------------


class TestFinalExposureValidation:
    @pytest.mark.asyncio
    async def test_final_exposure_allows_when_sized_qty_fits(self):
        """Step 15.7 approves a position when the sized qty fits within all limits."""
        capital = Decimal("1000")
        manager, exposure = _make_manager_with_real_exposure(capital)

        # 30% existing exposure; new position sized to fit in remaining 60%
        await exposure.update_position("ETHUSDT", "Buy", Decimal("300"), leverage=Decimal("7"))

        info = _instrument("BTCUSDT", min_notional="5", min_order_qty="1", qty_step="1")
        proposal = _proposal("BTCUSDT", qty="50", entry="10", confidence=1.0)

        decision = await manager.evaluate(
            proposal=proposal,
            capital=capital,
            available_balance=Decimal("1000"),
            instrument_info=info,
        )

        assert decision.status in (RiskDecisionStatus.APPROVED, RiskDecisionStatus.RESIZED)
        assert decision.approved_qty is not None and decision.approved_qty > Decimal("0")
        assert "exposure_rejected_after_final_resize" not in (decision.triggered_rules or [])

    @pytest.mark.asyncio
    async def test_final_exposure_resize_succeeds_when_remaining_fits_min_notional(self):
        """When step 15.7 triggers but there is still usable budget, qty is resized."""
        capital = Decimal("1000")

        # Use a mock exposure to precisely control what can_add_position returns
        exposure = MagicMock()
        exposure.position_count = 1
        exposure.total_exposure_pct = Decimal("80")
        # First call to can_add_position returns False; remaining = $100
        exposure.can_add_position = MagicMock(return_value=(False, "total exposure 95.00% would exceed cap 90%"))
        exposure.remaining_gross_notional_usd = MagicMock(return_value=Decimal("50"))

        drawdown = MagicMock()
        drawdown.drawdown_pct = Decimal("0")
        drawdown.is_at_hard_stop = MagicMock(return_value=False)

        breakers = MagicMock()
        breakers.should_emergency = MagicMock(return_value=False)
        breakers.should_block_entries = MagicMock(return_value=False)
        breakers.should_safe_mode = MagicMock(return_value=False)

        kill_switch = MagicMock()
        kill_switch.is_active = False

        manager = RiskManager(
            risk_profile=RiskProfile.SCALP,
            drawdown_tracker=drawdown,
            exposure_tracker=exposure,
            circuit_breaker_manager=breakers,
            kill_switch=kill_switch,
        )

        # entry=$1, remaining=$50 → resized_qty = 50 units
        info = _instrument("BTCUSDT", min_notional="5", min_order_qty="1", qty_step="1")
        proposal = _proposal("BTCUSDT", qty="200", entry="1", confidence=1.0)

        decision = await manager.evaluate(
            proposal=proposal,
            capital=capital,
            available_balance=Decimal("1000"),
            instrument_info=info,
        )

        # Should be resized to remaining budget, not rejected
        assert decision.status in (RiskDecisionStatus.RESIZED, RiskDecisionStatus.APPROVED)
        assert "exposure_rejected_after_final_resize" not in (decision.triggered_rules or [])

    @pytest.mark.asyncio
    async def test_final_exposure_rejects_when_resized_qty_below_min_order_qty(self):
        """Step 15.7 rejects when the resized qty would be below min_order_qty."""
        capital = Decimal("1000")

        exposure = MagicMock()
        exposure.position_count = 1
        exposure.total_exposure_pct = Decimal("80")
        exposure.can_add_position = MagicMock(return_value=(False, "total exposure 91.00% would exceed cap 90%"))
        # Only $0.50 remaining — not enough for min_order_qty=1 at entry=$1
        exposure.remaining_gross_notional_usd = MagicMock(return_value=Decimal("0.5"))

        drawdown = MagicMock()
        drawdown.drawdown_pct = Decimal("0")
        drawdown.is_at_hard_stop = MagicMock(return_value=False)

        breakers = MagicMock()
        breakers.should_emergency = MagicMock(return_value=False)
        breakers.should_block_entries = MagicMock(return_value=False)
        breakers.should_safe_mode = MagicMock(return_value=False)

        kill_switch = MagicMock()
        kill_switch.is_active = False

        manager = RiskManager(
            risk_profile=RiskProfile.SCALP,
            drawdown_tracker=drawdown,
            exposure_tracker=exposure,
            circuit_breaker_manager=breakers,
            kill_switch=kill_switch,
        )

        info = _instrument("BTCUSDT", min_notional="5", min_order_qty="1", qty_step="1")
        proposal = _proposal("BTCUSDT", qty="200", entry="1", confidence=1.0)

        decision = await manager.evaluate(
            proposal=proposal,
            capital=capital,
            available_balance=Decimal("1000"),
            instrument_info=info,
        )

        assert decision.status == RiskDecisionStatus.REJECTED
        assert "exposure_rejected_after_final_resize" in (decision.triggered_rules or [])
