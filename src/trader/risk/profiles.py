"""Risk profile definitions for the Bybit AI trading system.

Each profile defines strict limits that the RiskManager enforces.
Profiles are immutable dataclasses; override nothing at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from trader.domain.enums import MarketType, RiskProfile


@dataclass(frozen=True)
class RiskLimits:
    """Complete risk limits for a given risk profile.

    All percentage values are expressed as plain percentage numbers,
    e.g. ``Decimal("2.0")`` means 2%.
    """

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    risk_per_trade_min_pct: Decimal
    """Minimum % of capital to risk per trade."""

    risk_per_trade_max_pct: Decimal
    """Maximum % of capital to risk per trade (soft limit)."""

    risk_per_trade_hard_cap_pct: Decimal
    """Absolute maximum % of capital that can be risked per trade."""

    # ------------------------------------------------------------------
    # Leverage
    # ------------------------------------------------------------------
    max_leverage: Decimal
    """Maximum allowed leverage (1 = no leverage)."""

    # ------------------------------------------------------------------
    # Daily limits
    # ------------------------------------------------------------------
    daily_loss_limit_pct: Decimal
    """Stop opening new entries if daily loss exceeds this % of capital."""

    daily_loss_hard_stop_pct: Decimal
    """Close all positions if daily loss exceeds this % of capital."""

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------
    max_drawdown_pct: Decimal
    """Soft warning threshold; risk is reduced when exceeded."""

    hard_stop_drawdown_pct: Decimal
    """No new entries; reduce existing exposure when this is exceeded."""

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------
    max_simultaneous_positions: int
    """Maximum number of open positions at any one time."""

    max_capital_per_position_pct: Decimal
    """Maximum % of total capital allocated to a single position (notional)."""

    max_total_exposure_pct: Decimal
    """Total open exposure as % of capital (sum of all positions)."""

    # ------------------------------------------------------------------
    # Permissions
    # ------------------------------------------------------------------
    short_allowed: bool
    """Whether short (SELL) positions are permitted."""

    derivatives_allowed: bool
    """Whether LINEAR or INVERSE market types are permitted."""

    auto_resume_after_hard_stop: bool
    """MUST always be False — system requires manual restart after hard stop."""

    # ------------------------------------------------------------------
    # Market types
    # ------------------------------------------------------------------
    allowed_market_types: list[MarketType] = field(default_factory=list)
    """Exhaustive list of market types permitted for this profile."""

    max_total_margin_usage_pct: Decimal = Decimal("0")
    """Total margin used across all positions as % of capital. 0 = not enforced."""

    max_total_risk_at_stop_pct: Decimal = Decimal("0")
    """Total portfolio risk-at-stop as % of capital. 0 = not enforced."""

    max_margin_usage_per_position_pct: Decimal = Decimal("0")
    """Per-position margin usage as % of capital. 0 = not enforced."""

    def __post_init__(self) -> None:
        # CRITICAL INVARIANT: auto_resume_after_hard_stop is ALWAYS False.
        # Enforce it here regardless of what was passed.
        if self.auto_resume_after_hard_stop:
            # Use object.__setattr__ because the dataclass is frozen.
            object.__setattr__(self, "auto_resume_after_hard_stop", False)

        # Sanity: hard cap must be >= soft max
        if self.risk_per_trade_hard_cap_pct < self.risk_per_trade_max_pct:
            raise ValueError("risk_per_trade_hard_cap_pct must be >= risk_per_trade_max_pct")

        # Sanity: hard stop drawdown >= soft warning
        if self.hard_stop_drawdown_pct < self.max_drawdown_pct:
            raise ValueError("hard_stop_drawdown_pct must be >= max_drawdown_pct")

        # Sanity: risk_per_trade bounds must be ordered
        if self.risk_per_trade_min_pct > self.risk_per_trade_max_pct:
            raise ValueError("risk_per_trade_min_pct must be <= risk_per_trade_max_pct")

        # Sanity: daily loss soft limit must fire before the hard stop
        if self.daily_loss_limit_pct > self.daily_loss_hard_stop_pct:
            raise ValueError("daily_loss_limit_pct must be <= daily_loss_hard_stop_pct")


# ---------------------------------------------------------------------------
# Profile definitions — values from spec table
# ---------------------------------------------------------------------------

RISK_PROFILES: dict[RiskProfile, RiskLimits] = {
    RiskProfile.CONSERVATIVE: RiskLimits(
        # Position sizing — calibrated for small balance ($20-100)
        risk_per_trade_min_pct=Decimal("0.50"),
        risk_per_trade_max_pct=Decimal("1.50"),
        risk_per_trade_hard_cap_pct=Decimal("2.00"),
        # Leverage — 5x so $5 notional needs only $1 margin
        max_leverage=Decimal("5"),
        # Daily limits
        daily_loss_limit_pct=Decimal("3.00"),
        daily_loss_hard_stop_pct=Decimal("5.00"),
        # Drawdown
        max_drawdown_pct=Decimal("10.00"),
        hard_stop_drawdown_pct=Decimal("15.00"),
        # Portfolio
        max_simultaneous_positions=3,
        max_capital_per_position_pct=Decimal("30"),
        max_total_exposure_pct=Decimal("70"),
        # Permissions
        short_allowed=True,
        derivatives_allowed=True,
        auto_resume_after_hard_stop=False,
        # Market types
        allowed_market_types=[MarketType.LINEAR],
    ),
    RiskProfile.MODERATE: RiskLimits(
        # Position sizing
        risk_per_trade_min_pct=Decimal("0.50"),
        risk_per_trade_max_pct=Decimal("1.00"),
        risk_per_trade_hard_cap_pct=Decimal("2.00"),
        # Leverage
        max_leverage=Decimal("3"),
        # Daily limits
        daily_loss_limit_pct=Decimal("3.00"),
        daily_loss_hard_stop_pct=Decimal("5.00"),
        # Drawdown
        max_drawdown_pct=Decimal("12.00"),
        hard_stop_drawdown_pct=Decimal("15.00"),
        # Portfolio
        max_simultaneous_positions=4,
        max_capital_per_position_pct=Decimal("20"),
        max_total_exposure_pct=Decimal("60"),
        # Permissions
        short_allowed=True,
        derivatives_allowed=True,
        auto_resume_after_hard_stop=False,
        # Market types
        allowed_market_types=[MarketType.SPOT, MarketType.LINEAR],
    ),
    RiskProfile.AGGRESSIVE: RiskLimits(
        # Position sizing
        risk_per_trade_min_pct=Decimal("1.00"),
        risk_per_trade_max_pct=Decimal("2.00"),
        risk_per_trade_hard_cap_pct=Decimal("4.00"),
        # Leverage
        max_leverage=Decimal("10"),
        # Daily limits
        daily_loss_limit_pct=Decimal("5.00"),
        daily_loss_hard_stop_pct=Decimal("8.00"),
        # Drawdown
        max_drawdown_pct=Decimal("18.00"),
        hard_stop_drawdown_pct=Decimal("25.00"),
        # Portfolio
        max_simultaneous_positions=6,
        max_capital_per_position_pct=Decimal("33"),
        max_total_exposure_pct=Decimal("100"),
        # Permissions
        short_allowed=True,
        derivatives_allowed=True,
        auto_resume_after_hard_stop=False,
        # Market types
        allowed_market_types=[MarketType.SPOT, MarketType.LINEAR, MarketType.INVERSE],
    ),
    RiskProfile.SCALP: RiskLimits(
        # Position sizing — more frequent entries, but small risk per idea.
        risk_per_trade_min_pct=Decimal("0.25"),
        risk_per_trade_max_pct=Decimal("0.75"),
        risk_per_trade_hard_cap_pct=Decimal("1.25"),
        # Leverage
        max_leverage=Decimal("7"),
        # Daily limits
        daily_loss_limit_pct=Decimal("2.50"),
        daily_loss_hard_stop_pct=Decimal("4.00"),
        # Drawdown
        max_drawdown_pct=Decimal("8.00"),
        hard_stop_drawdown_pct=Decimal("12.00"),
        # Portfolio
        max_simultaneous_positions=8,
        max_capital_per_position_pct=Decimal("35"),
        max_total_exposure_pct=Decimal("90"),
        # Permissions
        short_allowed=True,
        derivatives_allowed=True,
        auto_resume_after_hard_stop=False,
        # Market types
        allowed_market_types=[MarketType.LINEAR],
    ),
}


def get_risk_limits(profile: RiskProfile) -> RiskLimits:
    """Return the ``RiskLimits`` for the given profile."""
    return RISK_PROFILES[profile]
