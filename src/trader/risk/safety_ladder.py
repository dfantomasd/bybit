"""Safety Mode Ladder — graduated response to portfolio drawdown.

Inspired by Krypto-trading-bot's PingPong/Boomerang/AK-47 strategy modes.

Levels
------
NORMAL   (0): drawdown ≤ soft_warning_pct       — full operation
PINGPONG (1): drawdown in (soft, 50% of hard]   — slow down; skip non-trend entries
BOOMERANG(2): drawdown in (50%, 80% of hard]    — halve sizing; warn loudly
AK47     (3): drawdown ≥ 80% of hard_stop_pct   — block all new entries; preserve capital

Thread-safe: all state is read-only after construction; DrawdownTracker is
already protected by its own asyncio.Lock.
"""

from __future__ import annotations

import time
from enum import IntEnum

from trader.risk.drawdown import DrawdownTracker
from trader.risk.profiles import RiskLimits


class SafetyLevel(IntEnum):
    NORMAL = 0
    PINGPONG = 1
    BOOMERANG = 2
    AK47 = 3


_PINGPONG_THRESHOLD = 0.0  # above soft_warning
_BOOMERANG_THRESHOLD = 0.50  # 50% of hard_stop distance
_AK47_THRESHOLD = 0.80  # 80% of hard_stop distance

# Maximum seconds a position should be held before a ladder warning is emitted
_DEFAULT_MAX_HOLD_S: float = 3600.0  # 1 hour


class SafetyModeLadder:
    """Graduated drawdown response provider.

    Args:
        drawdown_tracker: shared DrawdownTracker (read-only access).
        limits:           profile risk limits for threshold calibration.
        max_hold_s:       stale-position threshold (seconds).
    """

    def __init__(
        self,
        drawdown_tracker: DrawdownTracker,
        limits: RiskLimits,
        max_hold_s: float = _DEFAULT_MAX_HOLD_S,
    ) -> None:
        self._drawdown = drawdown_tracker
        self._limits = limits
        self._max_hold_s = max_hold_s

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def current_level(self) -> SafetyLevel:
        """Return the current ladder level based on portfolio drawdown."""
        dd = float(self._drawdown.drawdown_pct)
        hard = float(self._limits.hard_stop_drawdown_pct)
        soft = float(self._limits.max_drawdown_pct)

        if hard <= 0:
            return SafetyLevel.AK47  # misconfigured hard stop → fail closed
        if dd <= soft:
            return SafetyLevel.NORMAL

        # Scale: how far between soft_warning and hard_stop
        distance_to_hard = hard - soft
        if distance_to_hard <= 0:
            return SafetyLevel.AK47

        fraction = (dd - soft) / distance_to_hard

        if fraction >= _AK47_THRESHOLD:
            return SafetyLevel.AK47
        if fraction >= _BOOMERANG_THRESHOLD:
            return SafetyLevel.BOOMERANG
        return SafetyLevel.PINGPONG

    def size_multiplier(self) -> float:
        """Position size multiplier implied by current ladder level.

        Returns:
            1.0  → NORMAL (no restriction)
            0.75 → PINGPONG (mild reduction)
            0.50 → BOOMERANG (halve sizing)
            0.0  → AK47 (block all entries)
        """
        level = self.current_level()
        if level == SafetyLevel.NORMAL:
            return 1.0
        if level == SafetyLevel.PINGPONG:
            return 0.75
        if level == SafetyLevel.BOOMERANG:
            return 0.50
        return 0.0  # AK47

    def blocks_new_entries(self) -> bool:
        """True when the ladder level prohibits all new entries."""
        return self.current_level() == SafetyLevel.AK47

    def position_is_stale(self, opened_at_timestamp: float) -> bool:
        """True when a position has been open longer than max_hold_s.

        Args:
            opened_at_timestamp: ``time.monotonic()``-compatible timestamp.
        """
        return (time.monotonic() - opened_at_timestamp) > self._max_hold_s

    def describe(self) -> dict[str, object]:
        """Serialisable snapshot for logging and health endpoints."""
        level = self.current_level()
        return {
            "level": level.name,
            "level_int": int(level),
            "drawdown_pct": str(self._drawdown.drawdown_pct),
            "size_multiplier": self.size_multiplier(),
            "blocks_new_entries": self.blocks_new_entries(),
        }
