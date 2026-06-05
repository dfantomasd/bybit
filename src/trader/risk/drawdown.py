"""Drawdown tracker for the Bybit AI trading system.

Thread-safe via asyncio.Lock. All financial arithmetic uses Decimal.
"""
from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from typing import Any


class DrawdownTracker:
    """Tracks peak equity, current equity, and drawdown metrics.

    Thread-safe via asyncio.Lock.
    All percentage values returned are in 0.0–100.0 range.
    """

    def __init__(self, initial_equity: Decimal) -> None:
        if initial_equity <= Decimal("0"):
            raise ValueError("initial_equity must be positive")
        self._peak_equity: Decimal = initial_equity
        self._current_equity: Decimal = initial_equity
        self._peak_set_at: float = time.monotonic()
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    async def update(self, current_equity: Decimal) -> None:
        """Update current equity, adjusting peak if a new high is reached."""
        async with self._lock:
            self._current_equity = current_equity
            if current_equity > self._peak_equity:
                self._peak_equity = current_equity
                self._peak_set_at = time.monotonic()

    def reset_peak(self) -> None:
        """Reset peak to current equity (e.g. after capital injection).

        Not async — intended for use during system initialisation.
        """
        self._peak_equity = self._current_equity
        self._peak_set_at = time.monotonic()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def peak_equity(self) -> Decimal:
        return self._peak_equity

    @property
    def current_equity(self) -> Decimal:
        return self._current_equity

    @property
    def drawdown_pct(self) -> Decimal:
        """Current drawdown as % of peak equity (0.0 to 100.0)."""
        if self._peak_equity <= Decimal("0"):
            return Decimal("0")
        if self._current_equity >= self._peak_equity:
            return Decimal("0")
        return (
            (self._peak_equity - self._current_equity)
            / self._peak_equity
            * Decimal("100")
        )

    @property
    def drawdown_amount(self) -> Decimal:
        """Absolute drawdown in currency units."""
        return max(Decimal("0"), self._peak_equity - self._current_equity)

    @property
    def time_underwater(self) -> float:
        """Seconds since the last peak was set.

        Returns 0.0 if currently at the peak.
        """
        if self._current_equity >= self._peak_equity:
            return 0.0
        return time.monotonic() - self._peak_set_at

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------

    def is_at_hard_stop(self, hard_stop_pct: Decimal) -> bool:
        """True if current drawdown has reached or exceeded the hard stop."""
        return self.drawdown_pct >= hard_stop_pct

    def is_at_soft_warning(self, warning_pct: Decimal) -> bool:
        """True if current drawdown has reached or exceeded the soft warning."""
        return self.drawdown_pct >= warning_pct

    def get_risk_multiplier(self, hard_stop_pct: Decimal) -> Decimal:
        """Position size multiplier based on current drawdown.

        Returns:
            Decimal in [0.0, 1.0].
            1.0  → no drawdown.
            0.0  → at or beyond hard stop.
            Linear interpolation in between.
        """
        dd = self.drawdown_pct
        if dd <= Decimal("0"):
            return Decimal("1")
        if hard_stop_pct <= Decimal("0"):
            return Decimal("0")
        if dd >= hard_stop_pct:
            return Decimal("0")
        multiplier = Decimal("1") - (dd / hard_stop_pct)
        return max(Decimal("0"), min(Decimal("1"), multiplier))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_equity": str(self._peak_equity),
            "current_equity": str(self._current_equity),
            "drawdown_pct": str(self.drawdown_pct),
            "drawdown_amount": str(self.drawdown_amount),
            "time_underwater_seconds": self.time_underwater,
        }
