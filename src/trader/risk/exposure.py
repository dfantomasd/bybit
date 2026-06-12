"""Portfolio exposure tracker for the Bybit AI trading system.

Thread-safe via asyncio.Lock. All financial arithmetic uses Decimal.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from trader.risk.profiles import RiskLimits

# Base-asset families for correlation heuristic
_CRYPTO_FAMILIES: dict[str, list[str]] = {
    "BTC": ["BTC", "WBTC", "RBTC", "BTCB"],
    "ETH": ["ETH", "WETH", "STETH", "RETH", "CBETH"],
    "BNB": ["BNB", "WBNB"],
    "SOL": ["SOL", "MSOL", "JSOL", "BSOL"],
}


def _get_family(symbol: str) -> str | None:
    """Return the family name for a symbol's base asset, or None."""
    upper = symbol.upper()
    for family, members in _CRYPTO_FAMILIES.items():
        for member in members:
            if upper.startswith(member):
                return family
    return None


class ExposureTracker:
    """Tracks current portfolio exposure.

    Exposure is tracked per symbol as a notional value. The tracker is
    thread-safe via asyncio.Lock and uses Decimal throughout.
    """

    def __init__(self, total_capital: Decimal, risk_limits: RiskLimits) -> None:
        if total_capital <= Decimal("0"):
            raise ValueError("total_capital must be positive")
        self._capital = total_capital
        self._limits = risk_limits
        self._positions: dict[str, dict[str, Any]] = {}
        self._pending_exposure: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    async def update_position(
        self,
        symbol: str,
        side: str,
        notional_value: Decimal,
    ) -> None:
        """Add or update a position's notional value."""
        async with self._lock:
            self._positions[symbol] = {
                "side": side,
                "notional": notional_value,
            }
            self._release_pending_for_symbol_unlocked(symbol)

    async def remove_position(self, symbol: str) -> None:
        """Remove a closed position."""
        async with self._lock:
            self._positions.pop(symbol, None)

    def release_reservation(self, order_id: str) -> None:
        """Release a pending exposure reservation.

        Idempotent by design: terminal order callbacks and local abort paths can
        both try to release the same reservation without causing a state error.
        """
        if order_id:
            self._pending_exposure.pop(order_id, None)

    def update_capital(self, new_capital: Decimal) -> None:
        """Update total capital (e.g. after deposit/withdrawal)."""
        if new_capital > Decimal("0"):
            self._capital = new_capital

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_exposure_pct(self) -> Decimal:
        """Total open exposure as % of capital."""
        if self._capital <= Decimal("0"):
            return Decimal("0")
        total = self._total_notional()
        return Decimal(str(total)) / self._capital * Decimal("100")

    @property
    def position_count(self) -> int:
        """Number of open positions."""
        return len(self._symbols_with_exposure())

    def get_position_exposure_pct(self, symbol: str) -> Decimal:
        """Return exposure of a single position as % of capital."""
        if symbol not in self._positions:
            pending = self._pending_symbol_notional(symbol)
            if pending <= Decimal("0"):
                return Decimal("0")
            if self._capital <= Decimal("0"):
                return Decimal("0")
            return pending / self._capital * Decimal("100")
        if self._capital <= Decimal("0"):
            return Decimal("0")
        notional = Decimal(str(self._positions[symbol]["notional"])) + self._pending_symbol_notional(symbol)
        return notional / self._capital * Decimal("100")

    def get_position_notional(self, symbol: str) -> Decimal:
        """Return the current notional value of an existing position (0 if none)."""
        if symbol not in self._positions:
            return self._pending_symbol_notional(symbol)
        return Decimal(str(self._positions[symbol]["notional"])) + self._pending_symbol_notional(symbol)

    def remaining_total_exposure_usd(self) -> Decimal:
        """Remaining portfolio budget in USD before hitting max_total_exposure_pct."""
        if self._capital <= Decimal("0"):
            return Decimal("0")
        current_total = self._total_notional()
        max_total = self._capital * self._limits.max_total_exposure_pct / Decimal("100")
        return max(Decimal("0"), max_total - current_total)

    def remaining_position_exposure_usd(self, symbol: str) -> Decimal:
        """Remaining per-symbol budget in USD before hitting max_capital_per_position_pct."""
        if self._capital <= Decimal("0"):
            return Decimal("0")
        existing = self.get_position_notional(symbol)
        max_per_position = self._capital * self._limits.max_capital_per_position_pct / Decimal("100")
        return max(Decimal("0"), max_per_position - existing)

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------

    def can_add_position(
        self,
        symbol: str,
        additional_notional: Decimal,
        order_id: str | None = None,
    ) -> tuple[bool, str]:
        """Check whether a new/increased position is within risk limits.

        Returns:
            (allowed, reason_if_not_allowed)
        """
        if order_id and order_id in self._pending_exposure:
            return False, f"order {order_id} already has pending exposure reserved"

        # Check position count (only if it's a brand-new symbol)
        is_new = symbol not in self._symbols_with_exposure()
        if is_new and self.position_count >= self._limits.max_simultaneous_positions:
            return (
                False,
                f"max simultaneous positions ({self._limits.max_simultaneous_positions}) reached",
            )

        # Check per-position cap
        existing_notional = self.get_position_notional(symbol)
        new_notional = existing_notional + additional_notional
        new_position_pct = new_notional / self._capital * Decimal("100")
        if new_position_pct > self._limits.max_capital_per_position_pct:
            return (
                False,
                f"position exposure {new_position_pct:.2f}% exceeds per-position cap "
                f"{self._limits.max_capital_per_position_pct}%",
            )

        # Check total exposure cap
        current_total = self._total_notional()
        new_total = current_total + additional_notional
        new_total_pct = new_total / self._capital * Decimal("100")
        if new_total_pct > self._limits.max_total_exposure_pct:
            return (
                False,
                f"total exposure {new_total_pct:.2f}% would exceed cap {self._limits.max_total_exposure_pct}%",
            )

        if order_id:
            self._pending_exposure[order_id] = {
                "symbol": symbol,
                "notional": additional_notional,
            }
        return True, ""

    def get_correlation_adjustment(
        self,
        symbol: str,
        existing_symbols: list[str],
    ) -> Decimal:
        """Return a size multiplier based on correlation with existing positions.

        Heuristic: if the new symbol belongs to the same base-asset family as
        one or more existing positions, reduce allowed size proportionally.

        Returns:
            Decimal multiplier in [0.0, 1.0].
        """
        if not existing_symbols:
            return Decimal("1")

        new_family = _get_family(symbol)
        if new_family is None:
            return Decimal("1")

        # Count existing positions in the same family
        same_family_count = sum(1 for s in existing_symbols if _get_family(s) == new_family)

        if same_family_count == 0:
            return Decimal("1")

        # Each same-family position reduces allowed size by 20%
        reduction = Decimal(str(same_family_count)) * Decimal("0.20")
        multiplier = Decimal("1") - reduction
        return max(Decimal("0"), min(Decimal("1"), multiplier))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_capital": str(self._capital),
            "position_count": self.position_count,
            "total_exposure_pct": str(self.total_exposure_pct),
            "positions": {
                sym: {
                    "side": pos["side"],
                    "notional": str(pos["notional"]),
                    "exposure_pct": str(self.get_position_exposure_pct(sym)),
                }
                for sym, pos in self._positions.items()
            },
            "pending_exposure": {
                oid: {
                    "symbol": str(pos["symbol"]),
                    "notional": str(pos["notional"]),
                }
                for oid, pos in self._pending_exposure.items()
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pending_symbol_notional(self, symbol: str) -> Decimal:
        return sum(
            (
                Decimal(str(p["notional"]))
                for p in self._pending_exposure.values()
                if str(p.get("symbol", "")).upper() == symbol.upper()
            ),
            Decimal("0"),
        )

    def _total_notional(self) -> Decimal:
        open_notional = sum((Decimal(str(p["notional"])) for p in self._positions.values()), Decimal("0"))
        pending_notional = sum((Decimal(str(p["notional"])) for p in self._pending_exposure.values()), Decimal("0"))
        return open_notional + pending_notional

    def _symbols_with_exposure(self) -> set[str]:
        symbols = set(self._positions)
        symbols.update(str(p.get("symbol", "")).upper() for p in self._pending_exposure.values() if p.get("symbol"))
        return symbols

    def _release_pending_for_symbol_unlocked(self, symbol: str) -> None:
        to_release = [
            oid
            for oid, pending in self._pending_exposure.items()
            if str(pending.get("symbol", "")).upper() == symbol.upper()
        ]
        for oid in to_release:
            self._pending_exposure.pop(oid, None)
