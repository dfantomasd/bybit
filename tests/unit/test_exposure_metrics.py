"""Tests: P0.3/P0.4 – ExposureTracker per-position cap and notional tracking."""

from __future__ import annotations

from decimal import Decimal

import pytest

from trader.domain.enums import RiskProfile
from trader.risk.exposure import ExposureTracker
from trader.risk.profiles import get_risk_limits


def _tracker(
    capital: Decimal = Decimal("10000"),
    profile: RiskProfile = RiskProfile.CONSERVATIVE,
) -> ExposureTracker:
    limits = get_risk_limits(profile)
    return ExposureTracker(total_capital=capital, risk_limits=limits)


@pytest.mark.asyncio
async def test_single_position_cannot_exceed_cap():
    """P0.3: single position must be capped by max_capital_per_position_pct."""
    capital = Decimal("10000")
    t = _tracker(capital=capital)
    limits = get_risk_limits(RiskProfile.CONSERVATIVE)

    # This notional would bring per-position pct above cap
    over_cap_notional = capital * limits.max_capital_per_position_pct / Decimal("100") + Decimal("500")

    can_add, reason = t.can_add_position("BTCUSDT", over_cap_notional)
    assert not can_add, "Should reject notional that exceeds per-position cap"
    assert "per-position cap" in reason.lower() or "position exposure" in reason.lower()


@pytest.mark.asyncio
async def test_existing_symbol_position_reduces_remaining_symbol_budget():
    """P0.3: adding more to an existing position uses cumulative notional for cap check."""
    capital = Decimal("10000")
    t = _tracker(capital=capital)
    limits = get_risk_limits(RiskProfile.CONSERVATIVE)

    # Add half the per-position cap
    half_cap = capital * limits.max_capital_per_position_pct / Decimal("200")
    await t.update_position("BTCUSDT", "Buy", half_cap)

    # Try to add another full per-position cap amount
    # Combined should exceed cap
    can_add, reason = t.can_add_position("BTCUSDT", half_cap * Decimal("2"))
    assert not can_add, "Combined position should exceed cap"


@pytest.mark.asyncio
async def test_total_portfolio_cap_still_applies():
    """P0.4: total portfolio exposure cap must be respected."""
    capital = Decimal("10000")
    t = _tracker(capital=capital)
    limits = get_risk_limits(RiskProfile.CONSERVATIVE)

    # Fill up to max_total_exposure_pct with multiple symbols
    per_sym = capital * limits.max_total_exposure_pct / Decimal("100") / Decimal("2")
    await t.update_position("BTCUSDT", "Buy", per_sym)
    await t.update_position("ETHUSDT", "Buy", per_sym)

    # Another position would exceed total cap
    can_add, reason = t.can_add_position("XRPUSDT", Decimal("1"))
    assert not can_add, "Should reject when total portfolio cap is full"
    assert "total exposure" in reason.lower() or "cap" in reason.lower()


@pytest.mark.asyncio
async def test_pending_reservation_counts_towards_position_limit():
    """Pending exposure must close the race between risk approval and order registration."""
    t = _tracker(capital=Decimal("10000"))
    limits = get_risk_limits(RiskProfile.CONSERVATIVE)

    for idx in range(limits.max_simultaneous_positions):
        can_add, reason = t.can_add_position(f"COIN{idx}USDT", Decimal("10"), order_id=f"order-{idx}")
        assert can_add, reason

    can_add, reason = t.can_add_position("EXTRAUSDT", Decimal("10"), order_id="order-extra")
    assert not can_add
    assert "max simultaneous positions" in reason

    t.release_reservation("order-0")
    can_add, reason = t.can_add_position("EXTRAUSDT", Decimal("10"), order_id="order-extra")
    assert can_add, reason


@pytest.mark.asyncio
async def test_notional_is_not_multiplied_by_leverage():
    """P0.4: stored notional must equal qty × price (gross, not margin)."""
    capital = Decimal("10000")
    t = _tracker(capital=capital)

    # 0.1 BTC at $50,000 = $5,000 notional (regardless of 10x leverage)
    notional = Decimal("5000")
    await t.update_position("BTCUSDT", "Buy", notional)

    exposure_pct = t.get_position_exposure_pct("BTCUSDT")
    expected_pct = notional / capital * Decimal("100")
    assert exposure_pct == expected_pct, (
        f"Expected {expected_pct}% but got {exposure_pct}% — leverage must NOT be applied"
    )


@pytest.mark.asyncio
async def test_total_exposure_pct_aggregates_all_positions():
    """Total exposure should be sum of all position notionals."""
    capital = Decimal("10000")
    t = _tracker(capital=capital)

    await t.update_position("BTCUSDT", "Buy", Decimal("2000"))
    await t.update_position("ETHUSDT", "Buy", Decimal("1000"))

    expected_pct = Decimal("30")  # (2000+1000)/10000 * 100
    assert t.total_exposure_pct == expected_pct


@pytest.mark.asyncio
async def test_remove_position_reduces_exposure():
    """Removing a position must decrease total exposure."""
    capital = Decimal("10000")
    t = _tracker(capital=capital)

    await t.update_position("BTCUSDT", "Buy", Decimal("3000"))
    await t.update_position("ETHUSDT", "Buy", Decimal("1000"))

    await t.remove_position("BTCUSDT")

    assert t.total_exposure_pct == Decimal("10")  # 1000/10000*100
    assert t.position_count == 1
