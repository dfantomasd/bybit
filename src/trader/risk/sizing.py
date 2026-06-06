"""Position sizing calculator.

All arithmetic uses ``Decimal`` — no float is used for financial calculations.
Quantities are ALWAYS rounded DOWN (ROUND_DOWN) to the nearest qty_step.
"""

from __future__ import annotations

import logging
from decimal import ROUND_DOWN, Decimal

from trader.domain.models import InstrumentInfo
from trader.risk.profiles import RiskLimits

logger = logging.getLogger(__name__)

# Minimum spread threshold (expressed as %) above which we reject the trade.
_DEFAULT_SPREAD_THRESHOLD_PCT = Decimal("0.30")

# Minimum ATR multiple that stop distance must satisfy.
_DEFAULT_MIN_ATR_MULTIPLE = Decimal("0.5")

# Max position size as fraction of typical volume (liquidity filter).
_DEFAULT_MAX_VOLUME_FRACTION = Decimal("0.05")  # 5%


class PositionSizer:
    """Calculates approved position size from risk parameters.

    Formula (core):
        allowed_risk_amount = capital * desired_risk_pct / 100
        raw_size = allowed_risk_amount / (entry_price * stop_distance_pct)

    Then applies constraints in order:
    1. Exchange min/max qty.
    2. qty_step rounding (ROUND_DOWN, never up).
    3. Spread filter — reject if spread > threshold.
    4. ATR filter — stop must be > min_atr_multiple * ATR.
    5. Total portfolio exposure cap.
    6. Current drawdown reduction.
    7. Event risk reduction.
    8. Data quality reduction.
    9. Available balance cap.
    10. Final hard cap check.
    """

    def __init__(self, risk_limits: RiskLimits, instrument_info: InstrumentInfo) -> None:
        self._limits = risk_limits
        self._info = instrument_info

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        capital: Decimal,
        stop_distance_pct: Decimal,
        desired_risk_pct: Decimal,
        current_exposure_pct: Decimal,
        drawdown_pct: Decimal,
        event_risk_score: float,
        data_quality_score: float,
        spread: Decimal | None,
        atr: Decimal | None,
        available_balance: Decimal,
        entry_price: Decimal | None = None,
        remaining_position_budget_usd: Decimal | None = None,
    ) -> tuple[Decimal, str]:
        """Compute approved quantity.

        Returns:
            (approved_quantity, rejection_reason_or_empty_string)
            Returns (Decimal("0"), reason) if the trade should be rejected.
        """
        # ----------------------------------------------------------------
        # Guard: stop distance must be positive
        # ----------------------------------------------------------------
        if stop_distance_pct <= Decimal("0"):
            return Decimal("0"), "stop_distance_pct must be positive"

        # ----------------------------------------------------------------
        # Guard: capital must be positive
        # ----------------------------------------------------------------
        if capital <= Decimal("0") or available_balance <= Decimal("0"):
            return Decimal("0"), "insufficient capital or balance"

        # ----------------------------------------------------------------
        # Spread filter
        # ----------------------------------------------------------------
        if spread is not None and spread > _DEFAULT_SPREAD_THRESHOLD_PCT:
            return Decimal("0"), f"spread {spread}% exceeds threshold {_DEFAULT_SPREAD_THRESHOLD_PCT}%"

        # ----------------------------------------------------------------
        # ATR filter — stop must be at least MIN_ATR_MULTIPLE * ATR
        # ----------------------------------------------------------------
        if atr is not None and atr > Decimal("0"):
            stop_price_distance = stop_distance_pct * (entry_price or Decimal("1"))
            min_stop_distance = _DEFAULT_MIN_ATR_MULTIPLE * atr
            if stop_price_distance < min_stop_distance:
                return Decimal("0"), (f"stop distance {stop_price_distance} < min ATR multiple {min_stop_distance}")

        # ----------------------------------------------------------------
        # Clamp desired_risk_pct to profile range
        # ----------------------------------------------------------------
        clamped_risk_pct = min(
            max(desired_risk_pct, self._limits.risk_per_trade_min_pct),
            self._limits.risk_per_trade_max_pct,
        )

        # ----------------------------------------------------------------
        # Core size calculation
        # raw_size = (capital * risk_pct / 100) / stop_distance_pct
        # stop_distance_pct here is fractional (e.g. 0.02 = 2%)
        # ----------------------------------------------------------------
        allowed_risk_amount = capital * clamped_risk_pct / Decimal("100")
        # stop_distance_pct is a fraction of price (e.g. 0.02)
        raw_size = allowed_risk_amount / stop_distance_pct

        # ----------------------------------------------------------------
        # Convert to qty units if we have an entry price
        # ----------------------------------------------------------------
        if entry_price is not None and entry_price > Decimal("0"):
            raw_qty = raw_size / entry_price
        else:
            # Without entry price, raw_size is already in qty terms
            raw_qty = raw_size

        # ----------------------------------------------------------------
        # Apply drawdown multiplier (linear scale between 0 and hard_stop)
        # ----------------------------------------------------------------
        drawdown_multiplier = self._drawdown_multiplier(drawdown_pct)
        raw_qty = raw_qty * drawdown_multiplier

        # ----------------------------------------------------------------
        # Apply event risk reduction
        # ----------------------------------------------------------------
        if event_risk_score > 0:
            event_multiplier = Decimal(str(1.0 - event_risk_score * 0.5))
            event_multiplier = max(Decimal("0"), min(Decimal("1"), event_multiplier))
            raw_qty = raw_qty * event_multiplier

        # ----------------------------------------------------------------
        # Apply data quality reduction
        # ----------------------------------------------------------------
        if data_quality_score < 1.0:
            quality_multiplier = Decimal(str(data_quality_score))
            quality_multiplier = max(Decimal("0.1"), min(Decimal("1"), quality_multiplier))
            raw_qty = raw_qty * quality_multiplier

        # ----------------------------------------------------------------
        # Exposure cap check — don't push total exposure over limit
        # ----------------------------------------------------------------
        remaining_exposure_pct = self._limits.max_total_exposure_pct - current_exposure_pct
        if remaining_exposure_pct <= Decimal("0"):
            return Decimal("0"), "portfolio exposure cap reached"

        if entry_price is not None and entry_price > Decimal("0"):
            max_qty_from_exposure = (capital * remaining_exposure_pct / Decimal("100")) / entry_price
            raw_qty = min(raw_qty, max_qty_from_exposure)

        # ----------------------------------------------------------------
        # Per-position cap — cap this single position's notional
        # ----------------------------------------------------------------
        if (
            remaining_position_budget_usd is not None
            and remaining_position_budget_usd >= Decimal("0")
            and entry_price is not None
            and entry_price > Decimal("0")
        ):
            if remaining_position_budget_usd <= Decimal("0"):
                return Decimal("0"), "per-position exposure cap fully reached"
            max_qty_from_position_cap = remaining_position_budget_usd / entry_price
            raw_qty = min(raw_qty, max_qty_from_position_cap)

        # ----------------------------------------------------------------
        # Available balance cap
        # ----------------------------------------------------------------
        if entry_price is not None and entry_price > Decimal("0"):
            max_qty_from_balance = available_balance / entry_price
            raw_qty = min(raw_qty, max_qty_from_balance)

        # ----------------------------------------------------------------
        # Hard cap check — risk amount must not exceed hard_cap * capital / 100
        # ----------------------------------------------------------------
        hard_cap_risk = capital * self._limits.risk_per_trade_hard_cap_pct / Decimal("100")
        if entry_price is not None and entry_price > Decimal("0"):
            max_qty_from_hard_cap = hard_cap_risk / (stop_distance_pct * entry_price)
            raw_qty = min(raw_qty, max_qty_from_hard_cap)
        else:
            max_qty_from_hard_cap = hard_cap_risk / stop_distance_pct
            raw_qty = min(raw_qty, max_qty_from_hard_cap)

        # ----------------------------------------------------------------
        # Exchange min/max qty constraints
        # ----------------------------------------------------------------
        if raw_qty < self._info.min_order_qty:
            return Decimal("0"), (f"calculated qty {raw_qty} < min_order_qty {self._info.min_order_qty}")

        raw_qty = min(raw_qty, self._info.max_order_qty)

        # ----------------------------------------------------------------
        # Round down to qty_step — NEVER round up
        # ----------------------------------------------------------------
        approved_qty = self.round_to_step(raw_qty, self._info.qty_step)

        if approved_qty <= Decimal("0"):
            return Decimal("0"), "rounded qty is zero after step rounding"

        if approved_qty < self._info.min_order_qty:
            return Decimal("0"), (f"rounded qty {approved_qty} < min_order_qty {self._info.min_order_qty}")

        # ----------------------------------------------------------------
        # Min notional check
        # ----------------------------------------------------------------
        if self._info.min_notional is not None and entry_price is not None:
            notional = approved_qty * entry_price
            if notional < self._info.min_notional:
                return Decimal("0"), (f"notional {notional} < min_notional {self._info.min_notional}")

        return approved_qty, ""

    def round_to_step(self, qty: Decimal, step: Decimal) -> Decimal:
        """Round DOWN to nearest qty_step. Never rounds up."""
        if step <= Decimal("0"):
            return qty
        # Integer multiples of step, rounded down
        steps = (qty / step).to_integral_value(rounding=ROUND_DOWN)
        return steps * step

    def round_price_to_tick(self, price: Decimal, tick_size: Decimal) -> Decimal:
        """Round price to nearest tick_size (standard rounding)."""
        if tick_size <= Decimal("0"):
            return price
        ticks = (price / tick_size).to_integral_value(rounding=ROUND_DOWN)
        return ticks * tick_size

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _drawdown_multiplier(self, drawdown_pct: Decimal) -> Decimal:
        """Linear scale from 1.0 (no drawdown) to 0.0 (at hard stop)."""
        if drawdown_pct <= Decimal("0"):
            return Decimal("1")
        hard_stop = self._limits.hard_stop_drawdown_pct
        if hard_stop <= Decimal("0"):
            return Decimal("0")
        if drawdown_pct >= hard_stop:
            return Decimal("0")
        multiplier = Decimal("1") - (drawdown_pct / hard_stop)
        return max(Decimal("0"), min(Decimal("1"), multiplier))
