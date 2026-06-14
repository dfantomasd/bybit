"""Portfolio exposure tracker for the Bybit AI trading system.

Tracks three independent risk metrics per position:
  - gross_notional  : qty * entry_price  (raw position size in USD)
  - margin_used     : gross_notional / leverage
  - risk_at_stop    : gross_notional * stop_distance_pct

Portfolio aggregates: total_gross_notional_pct, total_margin_usage_pct,
total_risk_at_stop_pct — all as % of total capital.

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
    """Tracks current portfolio exposure across three risk dimensions.

    Per-position state:
        gross_notional    = qty * entry_price
        leverage          = confirmed exchange leverage (default 1)
        margin_used       = gross_notional / leverage
        stop_distance_pct = fractional distance to stop loss (default 0.02)
        risk_at_stop      = gross_notional * stop_distance_pct

    Portfolio limits enforced by can_add_position():
        max_total_exposure_pct          (gross notional / capital)
        max_capital_per_position_pct    (per-position gross notional / capital)
        max_total_margin_usage_pct      (total margin / capital)  — if > 0
        max_total_risk_at_stop_pct      (total risk / capital)    — if > 0
        max_margin_usage_per_position_pct (per-pos margin / cap)  — if > 0
    """

    def __init__(self, total_capital: Decimal, risk_limits: RiskLimits) -> None:
        if total_capital <= Decimal("0"):
            raise ValueError("total_capital must be positive")
        self._capital = total_capital
        self._limits = risk_limits
        self._positions: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

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
        """Add or update a position's notional and derived risk metrics.

        Parameters
        ----------
        symbol:
            Trading pair, e.g. "BTCUSDT".
        side:
            "Buy" or "Sell".
        notional_value:
            Gross notional = qty * entry_price.
        leverage:
            Actual exchange leverage; defaults to 1 when unknown.
        stop_distance_pct:
            Fractional distance from entry to stop loss (e.g. 0.02 = 2%).
            Defaults to 0.02 when unknown (conservative estimate).
        """
        gross_notional = notional_value
        lev = leverage if leverage is not None and leverage > Decimal("0") else Decimal("1")
        margin_used = gross_notional / lev
        sdp = (
            stop_distance_pct if stop_distance_pct is not None and stop_distance_pct > Decimal("0") else Decimal("0.02")
        )
        risk_at_stop = gross_notional * sdp

        async with self._lock:
            self._positions[symbol] = {
                "side": side,
                "notional": gross_notional,
                "leverage": lev,
                "margin_used": margin_used,
                "stop_distance_pct": sdp,
                "risk_at_stop": risk_at_stop,
            }

    async def remove_position(self, symbol: str) -> None:
        """Remove a closed position."""
        async with self._lock:
            self._positions.pop(symbol, None)

    def update_capital(self, new_capital: Decimal) -> None:
        """Update total capital (e.g. after deposit/withdrawal)."""
        if new_capital > Decimal("0"):
            self._capital = new_capital

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_gross_notional_pct(self) -> Decimal:
        """Total open gross notional as % of capital."""
        if self._capital <= Decimal("0"):
            return Decimal("0")
        total = sum(p["notional"] for p in self._positions.values())
        return Decimal(str(total)) / self._capital * Decimal("100")

    @property
    def total_exposure_pct(self) -> Decimal:
        """Backward-compat alias for total_gross_notional_pct."""
        return self.total_gross_notional_pct

    @property
    def total_margin_usage_pct(self) -> Decimal:
        """Total margin used across all positions as % of capital."""
        if self._capital <= Decimal("0"):
            return Decimal("0")
        total = sum(p["margin_used"] for p in self._positions.values())
        return Decimal(str(total)) / self._capital * Decimal("100")

    @property
    def total_risk_at_stop_pct(self) -> Decimal:
        """Total portfolio risk-at-stop as % of capital."""
        if self._capital <= Decimal("0"):
            return Decimal("0")
        total = sum(p["risk_at_stop"] for p in self._positions.values())
        return Decimal(str(total)) / self._capital * Decimal("100")

    @property
    def position_count(self) -> int:
        """Number of open positions."""
        return len(self._positions)

    def get_position_exposure_pct(self, symbol: str) -> Decimal:
        """Return gross notional exposure of a single position as % of capital."""
        if symbol not in self._positions or self._capital <= Decimal("0"):
            return Decimal("0")
        return self._positions[symbol]["notional"] / self._capital * Decimal("100")

    def remaining_gross_notional_usd(self, capital: Decimal, symbol: str | None = None) -> Decimal:
        """Remaining gross notional budget in USD before the portfolio cap is hit.

        Parameters
        ----------
        capital:
            Current total capital (may differ from initialisation value if
            refreshed by the risk manager).
        symbol:
            When provided, exclude this symbol's current position from the
            'already used' total (useful when re-entering the same symbol).
        """
        current_total = sum(p["notional"] for p in self._positions.values())
        if symbol is not None and symbol in self._positions:
            current_total -= self._positions[symbol]["notional"]
        max_total = capital * self._limits.max_total_exposure_pct / Decimal("100")
        return max(Decimal("0"), max_total - current_total)

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------

    def can_add_position(
        self,
        symbol: str,
        additional_notional: Decimal,
        leverage: Decimal | None = None,
        stop_distance_pct: Decimal | None = None,
    ) -> tuple[bool, str]:
        """Check whether adding *additional_notional* is within all risk limits.

        Parameters
        ----------
        symbol:
            Trading pair being added/updated.
        additional_notional:
            New gross notional for this position (total, not incremental).
        leverage:
            Leverage to use for margin checks; skipped when None.
        stop_distance_pct:
            Fractional stop distance for risk-at-stop checks; skipped when None.

        Returns
        -------
        (allowed, reason_if_denied)
        """
        if self._capital <= Decimal("0"):
            return False, "capital is zero"

        # -- Position count (new symbols only) --------------------------
        is_new = symbol not in self._positions
        if is_new and len(self._positions) >= self._limits.max_simultaneous_positions:
            return (
                False,
                f"max simultaneous positions ({self._limits.max_simultaneous_positions}) reached",
            )

        # -- Per-position gross notional cap ----------------------------
        existing_notional = Decimal("0")
        if symbol in self._positions:
            existing_notional = self._positions[symbol]["notional"]
        new_notional = existing_notional + additional_notional
        new_position_pct = new_notional / self._capital * Decimal("100")
        if new_position_pct > self._limits.max_capital_per_position_pct:
            return (
                False,
                f"position exposure {new_position_pct:.2f}% exceeds per-position cap "
                f"{self._limits.max_capital_per_position_pct}%",
            )

        # -- Total gross notional cap -----------------------------------
        current_total = sum(p["notional"] for p in self._positions.values())
        if symbol in self._positions:
            current_total -= self._positions[symbol]["notional"]
        total_notional = current_total + new_notional
        total_notional_pct = total_notional / self._capital * Decimal("100")
        if total_notional_pct > self._limits.max_total_exposure_pct:
            return (
                False,
                f"total exposure {total_notional_pct:.2f}% would exceed cap {self._limits.max_total_exposure_pct}%",
            )

        # -- Per-position margin cap (optional) -------------------------
        lev = leverage if leverage is not None and leverage > Decimal("0") else None
        if lev is not None and self._limits.max_margin_usage_per_position_pct > Decimal("0"):
            new_margin = new_notional / lev
            new_margin_pct = new_margin / self._capital * Decimal("100")
            if new_margin_pct > self._limits.max_margin_usage_per_position_pct:
                return (
                    False,
                    f"position margin {new_margin_pct:.2f}% exceeds per-position margin cap "
                    f"{self._limits.max_margin_usage_per_position_pct}%",
                )

        # -- Total margin usage cap (optional) --------------------------
        if lev is not None and self._limits.max_total_margin_usage_pct > Decimal("0"):
            existing_margin = self._positions[symbol]["margin_used"] if symbol in self._positions else Decimal("0")
            current_margin_total = sum(p["margin_used"] for p in self._positions.values()) - existing_margin
            new_total_margin = current_margin_total + new_notional / lev
            new_total_margin_pct = new_total_margin / self._capital * Decimal("100")
            if new_total_margin_pct > self._limits.max_total_margin_usage_pct:
                return (
                    False,
                    f"total margin usage {new_total_margin_pct:.2f}% would exceed cap "
                    f"{self._limits.max_total_margin_usage_pct}%",
                )

        # -- Total risk-at-stop cap (optional) --------------------------
        sdp = stop_distance_pct if stop_distance_pct is not None and stop_distance_pct > Decimal("0") else None
        if sdp is not None and self._limits.max_total_risk_at_stop_pct > Decimal("0"):
            existing_risk = self._positions[symbol]["risk_at_stop"] if symbol in self._positions else Decimal("0")
            current_risk_total = sum(p["risk_at_stop"] for p in self._positions.values()) - existing_risk
            new_total_risk = current_risk_total + new_notional * sdp
            new_total_risk_pct = new_total_risk / self._capital * Decimal("100")
            if new_total_risk_pct > self._limits.max_total_risk_at_stop_pct:
                return (
                    False,
                    f"total risk-at-stop {new_total_risk_pct:.2f}% would exceed cap "
                    f"{self._limits.max_total_risk_at_stop_pct}%",
                )

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
            "gross_notional_exposure_pct": str(self.total_gross_notional_pct),
            "total_exposure_pct": str(self.total_exposure_pct),  # backward-compat key
            "margin_usage_pct": str(self.total_margin_usage_pct),
            "risk_at_stop_pct": str(self.total_risk_at_stop_pct),
            "positions": {
                sym: {
                    "side": pos["side"],
                    "gross_notional": str(pos["notional"]),
                    "leverage": str(pos["leverage"]),
                    "margin_used": str(pos["margin_used"]),
                    "stop_distance_pct": str(pos["stop_distance_pct"]),
                    "risk_at_stop": str(pos["risk_at_stop"]),
                    "exposure_pct": str(self.get_position_exposure_pct(sym)),
                }
                for sym, pos in self._positions.items()
            },
        }
