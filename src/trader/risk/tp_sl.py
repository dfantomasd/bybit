"""TP/SL calculator for the Bybit AI trading system.

All arithmetic uses Decimal. Prices are rounded to tick_size.
"""

from __future__ import annotations

import logging
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal

logger = logging.getLogger(__name__)

# Minimum stop-loss distance as % of entry (0.1%)
_MIN_SL_DISTANCE_PCT = Decimal("0.001")

# Maximum stop-loss distance as % of entry (20%)
_MAX_SL_DISTANCE_PCT = Decimal("0.20")

# Minimum reward-to-risk ratio (TP distance / SL distance)
_MIN_RR_RATIO = Decimal("1.0")


class TPSLCalculator:
    """Calculates TP and SL prices from proposal parameters.

    Validates:
    - SL is on the correct side (below entry for long, above for short)
    - SL distance is reasonable (not too tight, not too wide)
    - TP is on correct side
    - TP/SL ratio is reasonable (TP distance >= SL distance)
    - Prices are rounded to tick_size
    """

    def calculate(
        self,
        side: str,
        entry_price: Decimal,
        stop_distance_pct: Decimal,
        take_profit_distance_pct: Decimal,
        tick_size: Decimal,
        atr: Decimal | None = None,
    ) -> tuple[Decimal, Decimal]:
        """Calculate (stop_loss, take_profit) prices.

        Args:
            side: "Buy" (long) or "Sell" (short).
            entry_price: Trade entry price.
            stop_distance_pct: Stop loss distance as fraction (e.g. 0.02 = 2%).
            take_profit_distance_pct: TP distance as fraction (e.g. 0.04 = 4%).
            tick_size: Exchange tick size for rounding.
            atr: Optional ATR for ATR-based floor.

        Returns:
            (stop_loss_price, take_profit_price) both rounded to tick_size.
        """
        is_long = side.lower() in ("buy", "long")

        sl_distance = entry_price * stop_distance_pct
        tp_distance = entry_price * take_profit_distance_pct

        # If ATR available, use the larger of ATR-based and pct-based distance
        if atr is not None and atr > Decimal("0"):
            sl_distance = max(sl_distance, atr * Decimal("1.0"))

        if is_long:
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        # Round conservatively to tick size: never improve exits on paper.
        sl_price = self._round_exit_to_tick(sl_price, tick_size, side=side, is_stop_loss=True)
        tp_price = self._round_exit_to_tick(tp_price, tick_size, side=side, is_stop_loss=False)

        # Ensure SL is strictly on the correct side after rounding
        if is_long:
            if sl_price >= entry_price:
                sl_price = self._round_to_tick(entry_price - tick_size, tick_size)
            if tp_price <= entry_price:
                tp_price = self._round_exit_to_tick(entry_price + tick_size, tick_size, side=side, is_stop_loss=False)
        else:
            if sl_price <= entry_price:
                sl_price = self._round_exit_to_tick(entry_price + tick_size, tick_size, side=side, is_stop_loss=True)
            if tp_price >= entry_price:
                tp_price = self._round_exit_to_tick(entry_price - tick_size, tick_size, side=side, is_stop_loss=False)

        return sl_price, tp_price

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate_sl_present(self, stop_loss_price: Decimal | None) -> bool:
        """Return True if stop loss price is present and positive."""
        return stop_loss_price is not None and stop_loss_price > Decimal("0")

    def validate_sl_direction(
        self,
        side: str,
        entry: Decimal,
        sl: Decimal,
    ) -> bool:
        """Return True if SL is on the correct side of entry.

        - Long: SL must be strictly below entry.
        - Short: SL must be strictly above entry.
        """
        is_long = side.lower() in ("buy", "long")
        if is_long:
            return sl < entry
        return sl > entry

    def validate_tp_direction(
        self,
        side: str,
        entry: Decimal,
        tp: Decimal,
    ) -> bool:
        """Return True if TP is on the correct side of entry.

        - Long: TP must be strictly above entry.
        - Short: TP must be strictly below entry.
        """
        is_long = side.lower() in ("buy", "long")
        if is_long:
            return tp > entry
        return tp < entry

    def validate_sl_distance(
        self,
        entry: Decimal,
        sl: Decimal,
        side: str,
    ) -> tuple[bool, str]:
        """Return (valid, reason) for stop loss distance."""
        if entry <= Decimal("0"):
            return False, "entry_price must be positive"

        distance = abs(entry - sl) / entry
        if distance < _MIN_SL_DISTANCE_PCT:
            return (
                False,
                f"SL distance {distance:.4%} is below minimum {_MIN_SL_DISTANCE_PCT:.4%}",
            )
        if distance > _MAX_SL_DISTANCE_PCT:
            return (
                False,
                f"SL distance {distance:.4%} exceeds maximum {_MAX_SL_DISTANCE_PCT:.4%}",
            )
        if not self.validate_sl_direction(side, entry, sl):
            return False, f"SL on wrong side for {side}"
        return True, ""

    def validate_rr_ratio(
        self,
        entry: Decimal,
        sl: Decimal,
        tp: Decimal,
    ) -> tuple[bool, str]:
        """Return (valid, reason) for reward-to-risk ratio."""
        sl_dist = abs(entry - sl)
        tp_dist = abs(entry - tp)
        if sl_dist <= Decimal("0"):
            return False, "SL distance is zero"
        ratio = tp_dist / sl_dist
        if ratio < _MIN_RR_RATIO:
            return (
                False,
                f"R:R ratio {ratio:.2f} is below minimum {_MIN_RR_RATIO}",
            )
        return True, ""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _round_to_tick(price: Decimal, tick_size: Decimal) -> Decimal:
        """Round price DOWN to nearest tick_size."""
        if tick_size <= Decimal("0"):
            return price
        ticks = (price / tick_size).to_integral_value(rounding=ROUND_DOWN)
        return ticks * tick_size

    @staticmethod
    def _round_exit_to_tick(
        price: Decimal,
        tick_size: Decimal,
        *,
        side: str,
        is_stop_loss: bool,
    ) -> Decimal:
        """Round exit prices conservatively for the trade direction.

        Both SL and TP use the same conservative direction per side:
          - Long  (buy)  → ROUND_DOWN:    keeps SL below entry; keeps TP at or below target
          - Short (sell) → ROUND_CEILING: keeps SL above entry; keeps TP at or above target

        `is_stop_loss` is accepted for API compatibility but does not change the
        rounding direction because the conservative rounding is identical for SL and TP.
        """
        _ = is_stop_loss  # same rounding direction for both SL and TP — see docstring
        if tick_size <= Decimal("0"):
            return price
        is_short = side.lower() in ("sell", "short")
        rounding = ROUND_CEILING if is_short else ROUND_DOWN
        ticks = (price / tick_size).to_integral_value(rounding=rounding)
        return ticks * tick_size
