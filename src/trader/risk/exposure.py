"""Portfolio exposure tracker for the Bybit AI trading system.

Thread-safe via a re-entrant lock. All financial arithmetic uses Decimal.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from threading import RLock
from typing import Any

from trader.risk.profiles import RiskLimits

# Base-asset families for correlation heuristic
_CRYPTO_FAMILIES: dict[str, list[str]] = {
    "BTC": ["BTC", "WBTC", "RBTC", "BTCB"],
    "ETH": ["ETH", "WETH", "STETH", "RETH", "CBETH"],
    "BNB": ["BNB", "WBNB"],
    "SOL": ["SOL", "MSOL", "JSOL", "BSOL"],
}

_QUOTE_SUFFIXES = (
    "USDT",
    "USDC",
    "USD",
    "BTC",
    "ETH",
    "EUR",
    "TRY",
)


def _base_asset(symbol: str) -> str:
    """Extract the base asset without matching partial token prefixes."""
    upper = symbol.upper().strip()
    for separator in ("-", "_", "/"):
        if separator in upper:
            return upper.split(separator, 1)[0]
    for suffix in _QUOTE_SUFFIXES:
        if upper.endswith(suffix) and len(upper) > len(suffix):
            return upper[: -len(suffix)]
    return upper


def _get_family(symbol: str) -> str | None:
    """Return the family name for a symbol's base asset, or None."""
    base = _base_asset(symbol)
    for family, members in _CRYPTO_FAMILIES.items():
        if base in members:
            return family
    return None


class ExposureTracker:
    """Tracks current portfolio exposure.

    Exposure is tracked per symbol as a notional value. The tracker is
    thread-safe via a re-entrant lock and uses Decimal throughout.
    """

    def __init__(self, total_capital: Decimal, risk_limits: RiskLimits) -> None:
        if total_capital <= Decimal("0"):
            raise ValueError("total_capital must be positive")
        self._capital = total_capital
        self._limits = risk_limits
        self._positions: dict[str, dict[str, Any]] = {}
        self._pending_exposure: dict[str, dict[str, Any]] = {}
        self._lock = RLock()
        self._capital_updated_at: datetime = datetime.now(UTC)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    async def update_position(
        self,
        symbol: str,
        side: str,
        notional_value: Decimal,
        leverage: Decimal | None = None,
        stop_distance_pct: Decimal | None = None,
    ) -> None:
        """Add or update a position's notional and derived risk metrics."""
        lev = leverage if leverage is not None and leverage > Decimal("0") else Decimal("1")
        margin_used = notional_value / lev
        sdp = (
            stop_distance_pct if stop_distance_pct is not None and stop_distance_pct > Decimal("0") else Decimal("0.02")
        )
        risk_at_stop = notional_value * sdp
        with self._lock:
            self._positions[symbol] = {
                "side": side,
                "notional": notional_value,
                "leverage": lev,
                "margin_used": margin_used,
                "stop_distance_pct": sdp,
                "risk_at_stop": risk_at_stop,
            }
            self._release_pending_for_symbol_unlocked(symbol)

    async def remove_position(self, symbol: str) -> None:
        """Remove a closed position."""
        with self._lock:
            self._positions.pop(symbol, None)

    def release_reservation(self, order_id: str) -> None:
        """Release a pending exposure reservation.

        Idempotent by design: terminal order callbacks and local abort paths can
        both try to release the same reservation without causing a state error.
        """
        with self._lock:
            if order_id:
                self._pending_exposure.pop(order_id, None)

    def update_capital(self, new_capital: Decimal, updated_at: datetime | None = None) -> None:
        """Update total capital, ignoring stale values.

        updated_at lets callers pass the authoritative timestamp (e.g. REST fetch
        time or WS event time) so that a delayed WS push cannot overwrite a more
        recent REST response.
        """
        with self._lock:
            if new_capital <= Decimal("0"):
                return
            ts = updated_at or datetime.now(UTC)
            if ts < self._capital_updated_at:
                return
            self._capital = new_capital
            self._capital_updated_at = ts

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_exposure_pct(self) -> Decimal:
        """Total open exposure as % of capital."""
        with self._lock:
            if self._capital <= Decimal("0"):
                return Decimal("0")
            total = self._total_notional()
            return Decimal(str(total)) / self._capital * Decimal("100")

    @property
    def total_gross_notional_pct(self) -> Decimal:
        """Alias for total_exposure_pct (three-metric model)."""
        return self.total_exposure_pct

    @property
    def total_margin_usage_pct(self) -> Decimal:
        """Total margin used across all positions as % of capital."""
        with self._lock:
            if self._capital <= Decimal("0"):
                return Decimal("0")
            total = sum(Decimal(str(p.get("margin_used", p["notional"]))) for p in self._positions.values())
            return Decimal(str(total)) / self._capital * Decimal("100")

    @property
    def total_risk_at_stop_pct(self) -> Decimal:
        """Total portfolio risk-at-stop as % of capital."""
        with self._lock:
            if self._capital <= Decimal("0"):
                return Decimal("0")
            total = sum(
                Decimal(str(p.get("risk_at_stop", p["notional"] * Decimal("0.02")))) for p in self._positions.values()
            )
            return Decimal(str(total)) / self._capital * Decimal("100")

    @property
    def position_count(self) -> int:
        """Number of open positions."""
        with self._lock:
            return len(self._symbols_with_exposure())

    def get_position_exposure_pct(self, symbol: str) -> Decimal:
        """Return exposure of a single position as % of capital."""
        with self._lock:
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
        with self._lock:
            if symbol not in self._positions:
                return self._pending_symbol_notional(symbol)
            return Decimal(str(self._positions[symbol]["notional"])) + self._pending_symbol_notional(symbol)

    def remaining_total_exposure_usd(self) -> Decimal:
        """Remaining portfolio budget in USD before hitting max_total_exposure_pct."""
        with self._lock:
            if self._capital <= Decimal("0"):
                return Decimal("0")
            current_total = self._total_notional()
            max_total = self._capital * self._limits.max_total_exposure_pct / Decimal("100")
            return max(Decimal("0"), max_total - current_total)

    def remaining_position_exposure_usd(self, symbol: str) -> Decimal:
        """Remaining per-symbol budget in USD before hitting max_capital_per_position_pct."""
        with self._lock:
            if self._capital <= Decimal("0"):
                return Decimal("0")
            existing = self.get_position_notional(symbol)
            max_per_position = self._capital * self._limits.max_capital_per_position_pct / Decimal("100")
            return max(Decimal("0"), max_per_position - existing)

    def remaining_gross_notional_usd(self, capital: Decimal, symbol: str | None = None) -> Decimal:
        """Remaining gross notional budget in USD before portfolio cap is hit.

        When *symbol* is given, the caller's current position for that symbol
        is excluded so re-entry scenarios compute the correct remaining budget.
        """
        with self._lock:
            current_total = sum(Decimal(str(p["notional"])) for p in self._positions.values())
            if symbol is not None and symbol in self._positions:
                current_total -= Decimal(str(self._positions[symbol]["notional"]))
            max_total = capital * self._limits.max_total_exposure_pct / Decimal("100")
            return max(Decimal("0"), max_total - current_total)

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------

    def can_add_position(
        self,
        symbol: str,
        additional_notional: Decimal,
        order_id: str | None = None,
        leverage: Decimal | None = None,
        stop_distance_pct: Decimal | None = None,
    ) -> tuple[bool, str]:
        """Check whether a new/increased position is within risk limits.

        Returns:
            (allowed, reason_if_not_allowed)
        """
        with self._lock:
            if order_id and order_id in self._pending_exposure:
                return False, f"order {order_id} already has pending exposure reserved"

            # Check position count (only if it's a brand-new symbol)
            is_new = symbol not in self._symbols_with_exposure()
            if is_new and len(self._symbols_with_exposure()) >= self._limits.max_simultaneous_positions:
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

            # Optional: per-position margin cap
            lev = leverage if leverage is not None and leverage > Decimal("0") else None
            if lev is not None and self._limits.max_margin_usage_per_position_pct > Decimal("0"):
                new_margin_pct = new_notional / lev / self._capital * Decimal("100")
                if new_margin_pct > self._limits.max_margin_usage_per_position_pct:
                    return (
                        False,
                        f"position margin {new_margin_pct:.2f}% exceeds per-position margin cap "
                        f"{self._limits.max_margin_usage_per_position_pct}%",
                    )

            # Optional: total margin cap
            if lev is not None and self._limits.max_total_margin_usage_pct > Decimal("0"):
                existing_margin = (
                    Decimal(str(self._positions[symbol]["margin_used"])) if symbol in self._positions else Decimal("0")
                )
                current_margin = (
                    sum(Decimal(str(p.get("margin_used", p["notional"]))) for p in self._positions.values())
                    - existing_margin
                )
                new_total_margin_pct = (current_margin + new_notional / lev) / self._capital * Decimal("100")
                if new_total_margin_pct > self._limits.max_total_margin_usage_pct:
                    return (
                        False,
                        f"total margin usage {new_total_margin_pct:.2f}% would exceed cap "
                        f"{self._limits.max_total_margin_usage_pct}%",
                    )

            # Optional: total risk-at-stop cap
            sdp = stop_distance_pct if stop_distance_pct is not None and stop_distance_pct > Decimal("0") else None
            if sdp is not None and self._limits.max_total_risk_at_stop_pct > Decimal("0"):
                existing_risk = (
                    Decimal(str(self._positions[symbol]["risk_at_stop"])) if symbol in self._positions else Decimal("0")
                )
                current_risk = (
                    sum(
                        Decimal(str(p.get("risk_at_stop", p["notional"] * Decimal("0.02"))))
                        for p in self._positions.values()
                    )
                    - existing_risk
                )
                new_total_risk_pct = (current_risk + new_notional * sdp) / self._capital * Decimal("100")
                if new_total_risk_pct > self._limits.max_total_risk_at_stop_pct:
                    return (
                        False,
                        f"total risk-at-stop {new_total_risk_pct:.2f}% would exceed cap "
                        f"{self._limits.max_total_risk_at_stop_pct}%",
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
        with self._lock:
            return {
                "total_capital": str(self._capital),
                "position_count": len(self._symbols_with_exposure()),
                "gross_notional_exposure_pct": str(self.total_gross_notional_pct),
                "total_exposure_pct": str(self.total_exposure_pct),
                "margin_usage_pct": str(self.total_margin_usage_pct),
                "risk_at_stop_pct": str(self.total_risk_at_stop_pct),
                "positions": {
                    sym: {
                        "side": pos["side"],
                        "notional": str(pos["notional"]),
                        "leverage": str(pos.get("leverage", "1")),
                        "margin_used": str(pos.get("margin_used", pos["notional"])),
                        "stop_distance_pct": str(pos.get("stop_distance_pct", "0.02")),
                        "risk_at_stop": str(pos.get("risk_at_stop", Decimal(str(pos["notional"])) * Decimal("0.02"))),
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
