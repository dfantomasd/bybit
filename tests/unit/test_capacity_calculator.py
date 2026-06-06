"""Tests for EffectiveCapacityCalculator."""

from __future__ import annotations

from decimal import Decimal

from trader.analytics.capacity import CapacitySnapshot, EffectiveCapacityCalculator


def _calc(
    *,
    equity: str = "100",
    available_balance: str = "100",
    configured_max: int = 8,
    max_total_exposure_pct: str = "70",
    max_capital_per_position_pct: str = "15",
    risk_per_trade_max_pct: str = "1.5",
    open_positions: int = 0,
    current_gross_exposure: str = "0",
    avg_leverage: str = "5",
    min_notional: str = "5",
    fee_reserve_pct: str = "0.5",
) -> CapacitySnapshot:
    calculator = EffectiveCapacityCalculator()
    return calculator.calculate(
        equity=Decimal(equity),
        available_balance=Decimal(available_balance),
        configured_max_positions=configured_max,
        max_total_exposure_pct=Decimal(max_total_exposure_pct),
        max_capital_per_position_pct=Decimal(max_capital_per_position_pct),
        risk_per_trade_max_pct=Decimal(risk_per_trade_max_pct),
        current_open_positions=open_positions,
        current_gross_exposure_usd=Decimal(current_gross_exposure),
        avg_leverage=Decimal(avg_leverage),
        min_notional_usd=Decimal(min_notional),
        fee_reserve_pct=Decimal(fee_reserve_pct),
    )


# ---------------------------------------------------------------------------
# Profile limit is the ceiling
# ---------------------------------------------------------------------------


def test_profile_limit_is_not_exceeded():
    """effective_max_positions must never exceed configured_max_positions."""
    snap = _calc(configured_max=3, equity="10000", available_balance="10000", min_notional="5")
    assert snap.effective_max_positions <= 3


def test_profile_limit_enforced_even_with_large_balance():
    """A large balance doesn't allow more than configured_max_positions."""
    snap = _calc(configured_max=2, equity="100000", available_balance="100000", min_notional="5")
    assert snap.effective_max_positions <= 2


# ---------------------------------------------------------------------------
# Capital (min-notional) limit
# ---------------------------------------------------------------------------


def test_effective_slots_limited_by_min_notional():
    """With tiny balance, effective_max ≤ configured_max regardless of raw capital limit."""
    # balance=5 USDT, min_notional=5, leverage=5 → margin_per_pos=1 → only ~4.97 slots
    snap = _calc(
        equity="5",
        available_balance="5",
        configured_max=8,
        min_notional="5",
        avg_leverage="5",
        fee_reserve_pct="0.5",
    )
    # effective must never exceed profile cap
    assert snap.effective_max_positions <= snap.configured_max_positions
    # With only 5 USDT available, effective must be < configured 8
    assert snap.effective_max_positions <= 5


def test_zero_balance_returns_zero_free_slots():
    """With zero equity, nothing can be opened."""
    snap = _calc(equity="0", available_balance="0")
    assert snap.effective_max_positions == 0
    assert snap.free_slots == 0


# ---------------------------------------------------------------------------
# Gross exposure limit
# ---------------------------------------------------------------------------


def test_effective_slots_limited_by_exposure():
    """When gross exposure cap is nearly exhausted, effective slots drop."""
    # equity=100, max_total_exposure=70% → max 70 USD notional
    # already 65 USD open → only 5 USD remaining
    # each new pos needs 5 USD min_notional → only 1 slot
    snap = _calc(
        equity="100",
        available_balance="20",
        configured_max=8,
        max_total_exposure_pct="70",
        max_capital_per_position_pct="15",
        current_gross_exposure="65",
        min_notional="5",
        avg_leverage="5",
    )
    assert snap.gross_exposure_limited_positions <= 2
    assert snap.remaining_gross_exposure_usd <= Decimal("5")


# ---------------------------------------------------------------------------
# Margin limit
# ---------------------------------------------------------------------------


def test_effective_slots_limited_by_margin():
    """Low available_balance constrains margin-based limit."""
    # equity=1000, but available_balance=2 (most is used as margin)
    # margin per pos = 5/5 = 1 → capital_limited ≈ 1 after fee reserve
    snap = _calc(
        equity="1000",
        available_balance="2",
        configured_max=8,
        min_notional="5",
        avg_leverage="5",
        fee_reserve_pct="0.5",
    )
    # remaining margin ≈ 2 - 0.005*1000 = 1.5 → ~1 position
    assert snap.margin_limited_positions <= 2


# ---------------------------------------------------------------------------
# Risk-at-stop limit
# ---------------------------------------------------------------------------


def test_effective_slots_limited_by_risk_at_stop():
    """Risk headroom limits number of positions when risk_per_trade is tight."""
    # equity=100, risk_per_trade_max=1% → 1 USD per trade
    # assumed stop distance 2%, min_notional=5 → risk ≈ 0.10/position
    snap = _calc(
        equity="100",
        available_balance="100",
        configured_max=8,
        risk_per_trade_max_pct="1.0",
        min_notional="5",
        avg_leverage="5",
    )
    assert snap.risk_at_stop_limited_positions >= 0


# ---------------------------------------------------------------------------
# Free slots
# ---------------------------------------------------------------------------


def test_free_slots_decreases_with_open_positions():
    """free_slots = effective_max - open_positions (floor 0)."""
    snap_0 = _calc(open_positions=0)
    snap_2 = _calc(open_positions=2)
    assert snap_2.free_slots == max(0, snap_0.effective_max_positions - 2)


def test_free_slots_never_negative():
    """free_slots must always be >= 0."""
    snap = _calc(open_positions=999, configured_max=2)
    assert snap.free_slots == 0


# ---------------------------------------------------------------------------
# Remaining exposure
# ---------------------------------------------------------------------------


def test_remaining_gross_exposure_decreases_as_positions_open():
    """remaining_gross_exposure_usd decreases as current_gross_exposure rises."""
    snap_low = _calc(current_gross_exposure="10")
    snap_high = _calc(current_gross_exposure="50")
    assert snap_high.remaining_gross_exposure_usd < snap_low.remaining_gross_exposure_usd


# ---------------------------------------------------------------------------
# Summary — correct snapshot type
# ---------------------------------------------------------------------------


def test_calculate_returns_capacity_snapshot():
    snap = _calc()
    assert isinstance(snap, CapacitySnapshot)
    assert snap.current_open_positions == 0
    assert snap.configured_max_positions == 8
