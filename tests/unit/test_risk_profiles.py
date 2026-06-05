"""Tests for risk profile definitions."""
from __future__ import annotations

import pytest
from decimal import Decimal

from trader.domain.enums import MarketType, RiskProfile
from trader.risk.profiles import RISK_PROFILES, RiskLimits, get_risk_limits


def test_conservative_limits():
    """Test conservative profile has correct values from spec."""
    limits = RISK_PROFILES[RiskProfile.CONSERVATIVE]
    assert limits.risk_per_trade_min_pct == Decimal("0.25")
    assert limits.risk_per_trade_max_pct == Decimal("0.50")
    assert limits.risk_per_trade_hard_cap_pct == Decimal("1.00")
    assert limits.max_leverage == Decimal("1")
    assert limits.daily_loss_limit_pct == Decimal("1.50")
    assert limits.daily_loss_hard_stop_pct == Decimal("2.00")
    assert limits.max_drawdown_pct == Decimal("8.00")
    assert limits.hard_stop_drawdown_pct == Decimal("10.00")
    assert limits.max_simultaneous_positions == 2
    assert limits.max_capital_per_position_pct == Decimal("10")
    assert limits.max_total_exposure_pct == Decimal("30")


def test_moderate_limits():
    """Test moderate profile has correct values from spec."""
    limits = RISK_PROFILES[RiskProfile.MODERATE]
    assert limits.risk_per_trade_min_pct == Decimal("0.50")
    assert limits.risk_per_trade_max_pct == Decimal("1.00")
    assert limits.risk_per_trade_hard_cap_pct == Decimal("2.00")
    assert limits.max_leverage == Decimal("3")
    assert limits.daily_loss_limit_pct == Decimal("3.00")
    assert limits.daily_loss_hard_stop_pct == Decimal("5.00")
    assert limits.max_drawdown_pct == Decimal("12.00")
    assert limits.hard_stop_drawdown_pct == Decimal("15.00")
    assert limits.max_simultaneous_positions == 4
    assert limits.max_capital_per_position_pct == Decimal("20")
    assert limits.max_total_exposure_pct == Decimal("60")


def test_aggressive_limits():
    """Test aggressive profile has correct values from spec."""
    limits = RISK_PROFILES[RiskProfile.AGGRESSIVE]
    assert limits.risk_per_trade_min_pct == Decimal("1.00")
    assert limits.risk_per_trade_max_pct == Decimal("2.00")
    assert limits.risk_per_trade_hard_cap_pct == Decimal("4.00")
    assert limits.max_leverage == Decimal("10")
    assert limits.daily_loss_limit_pct == Decimal("5.00")
    assert limits.daily_loss_hard_stop_pct == Decimal("8.00")
    assert limits.max_drawdown_pct == Decimal("18.00")
    assert limits.hard_stop_drawdown_pct == Decimal("25.00")
    assert limits.max_simultaneous_positions == 6
    assert limits.max_capital_per_position_pct == Decimal("33")
    assert limits.max_total_exposure_pct == Decimal("100")


def test_conservative_no_short():
    """Conservative profile must not allow short selling."""
    limits = RISK_PROFILES[RiskProfile.CONSERVATIVE]
    assert limits.short_allowed is False


def test_conservative_no_derivatives():
    """Conservative profile must not allow derivatives."""
    limits = RISK_PROFILES[RiskProfile.CONSERVATIVE]
    assert limits.derivatives_allowed is False
    assert MarketType.SPOT in limits.allowed_market_types
    assert MarketType.LINEAR not in limits.allowed_market_types
    assert MarketType.INVERSE not in limits.allowed_market_types


def test_aggressive_has_higher_limits_than_moderate():
    """Aggressive profile should have higher limits than moderate."""
    moderate = RISK_PROFILES[RiskProfile.MODERATE]
    aggressive = RISK_PROFILES[RiskProfile.AGGRESSIVE]

    assert aggressive.risk_per_trade_max_pct > moderate.risk_per_trade_max_pct
    assert aggressive.risk_per_trade_hard_cap_pct > moderate.risk_per_trade_hard_cap_pct
    assert aggressive.max_leverage > moderate.max_leverage
    assert aggressive.daily_loss_limit_pct > moderate.daily_loss_limit_pct
    assert aggressive.max_simultaneous_positions > moderate.max_simultaneous_positions
    assert aggressive.max_total_exposure_pct > moderate.max_total_exposure_pct


def test_auto_resume_always_false():
    """CRITICAL: auto_resume_after_hard_stop must always be False for ALL profiles."""
    for profile in RiskProfile:
        limits = RISK_PROFILES[profile]
        assert limits.auto_resume_after_hard_stop is False, (
            f"auto_resume_after_hard_stop must be False for {profile.value}"
        )


def test_auto_resume_forced_false_even_if_passed_true():
    """CRITICAL: RiskLimits.__post_init__ must force auto_resume to False even if True is passed."""
    limits = RiskLimits(
        risk_per_trade_min_pct=Decimal("0.25"),
        risk_per_trade_max_pct=Decimal("0.50"),
        risk_per_trade_hard_cap_pct=Decimal("1.00"),
        max_leverage=Decimal("1"),
        daily_loss_limit_pct=Decimal("1.50"),
        daily_loss_hard_stop_pct=Decimal("2.00"),
        max_drawdown_pct=Decimal("8.00"),
        hard_stop_drawdown_pct=Decimal("10.00"),
        max_simultaneous_positions=2,
        max_capital_per_position_pct=Decimal("10"),
        max_total_exposure_pct=Decimal("30"),
        short_allowed=False,
        derivatives_allowed=False,
        auto_resume_after_hard_stop=True,  # <- attempting to set True
        allowed_market_types=[MarketType.SPOT],
    )
    # Must be forced to False
    assert limits.auto_resume_after_hard_stop is False


def test_hard_cap_higher_than_soft_cap():
    """Hard cap must be >= soft max for all profiles."""
    for profile in RiskProfile:
        limits = RISK_PROFILES[profile]
        assert limits.risk_per_trade_hard_cap_pct >= limits.risk_per_trade_max_pct, (
            f"hard_cap must be >= soft_max for {profile.value}"
        )
        assert limits.hard_stop_drawdown_pct >= limits.max_drawdown_pct, (
            f"hard_stop_drawdown must be >= soft_warning for {profile.value}"
        )


def test_get_risk_limits_returns_correct_profile():
    """get_risk_limits helper returns correct limits."""
    for profile in RiskProfile:
        limits = get_risk_limits(profile)
        assert limits is RISK_PROFILES[profile]


def test_all_profiles_present():
    """All three profiles must be present in RISK_PROFILES."""
    assert RiskProfile.CONSERVATIVE in RISK_PROFILES
    assert RiskProfile.MODERATE in RISK_PROFILES
    assert RiskProfile.AGGRESSIVE in RISK_PROFILES


def test_risk_limits_immutable():
    """RiskLimits is a frozen dataclass — mutation must raise."""
    limits = RISK_PROFILES[RiskProfile.CONSERVATIVE]
    with pytest.raises((AttributeError, TypeError)):
        limits.max_leverage = Decimal("5")  # type: ignore[misc]


def test_hard_cap_validation():
    """RiskLimits rejects hard_cap < soft_max."""
    with pytest.raises(ValueError, match="risk_per_trade_hard_cap_pct"):
        RiskLimits(
            risk_per_trade_min_pct=Decimal("0.25"),
            risk_per_trade_max_pct=Decimal("1.00"),
            risk_per_trade_hard_cap_pct=Decimal("0.50"),  # < max_pct!
            max_leverage=Decimal("1"),
            daily_loss_limit_pct=Decimal("1.50"),
            daily_loss_hard_stop_pct=Decimal("2.00"),
            max_drawdown_pct=Decimal("8.00"),
            hard_stop_drawdown_pct=Decimal("10.00"),
            max_simultaneous_positions=2,
            max_capital_per_position_pct=Decimal("10"),
            max_total_exposure_pct=Decimal("30"),
            short_allowed=False,
            derivatives_allowed=False,
            auto_resume_after_hard_stop=False,
            allowed_market_types=[MarketType.SPOT],
        )
