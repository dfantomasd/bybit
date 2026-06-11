"""Kill switch for the Bybit AI trading system.

Provides multiple activation methods: file flag, programmatic, and operator
command. Modes escalate from PAUSE_NEW_ENTRIES through FULL_STOP.

CRITICAL: Once activated beyond PAUSE_NEW_ENTRIES, only manual operator
action can deactivate the kill switch.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trader.domain.enums import KillSwitchMode

logger = logging.getLogger(__name__)


class KillSwitch:
    """Emergency stop mechanism with multiple activation methods.

    Modes (in escalating order of severity):
    - PAUSE_NEW_ENTRIES: block new positions, keep existing
    - CANCEL_OPEN_ORDERS: cancel pending orders, keep positions
    - REDUCE_RISK: tighten stops, reduce sizes
    - CLOSE_ALL_IF_CONFIGURED: close all positions (configurable)
    - FULL_STOP: stop everything, requires manual restart
    """

    DEFAULT_FLAG_FILE = Path.home() / ".bybit_trader_kill.flag"

    # Modes that can be deactivated without manual restart
    _DEACTIVATABLE_MODES = {KillSwitchMode.PAUSE_NEW_ENTRIES}

    def __init__(
        self,
        execution_engine_ref: Any = None,
        event_bus: Any = None,
        metrics: Any = None,
        flag_file: str | os.PathLike[str] | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self._engine = execution_engine_ref
        self._event_bus = event_bus
        self._metrics = metrics
        self._log = log or logger
        self._lock = asyncio.Lock()
        self._flag_file = Path(flag_file or os.getenv("KILL_SWITCH_FLAG_FILE") or self.DEFAULT_FLAG_FILE)

        self._active: bool = False
        self._mode: KillSwitchMode | None = None
        self._activated_at: datetime | None = None
        self._activated_by: str | None = None
        self._reason: str = ""

    # ------------------------------------------------------------------
    # Activation / deactivation
    # ------------------------------------------------------------------

    async def activate(
        self,
        mode: KillSwitchMode,
        reason: str,
        operator: str = "system",
    ) -> None:
        """Activate the kill switch in the given mode.

        If already active in an equal or higher mode, the call is a no-op.
        """
        async with self._lock:
            if self._active and self._mode is not None:
                # Only escalate, never de-escalate
                current_order = list(KillSwitchMode).index(self._mode)
                new_order = list(KillSwitchMode).index(mode)
                if new_order <= current_order:
                    return

            self._active = True
            self._mode = mode
            self._activated_at = datetime.now(tz=UTC)
            self._activated_by = operator
            self._reason = reason

            self._log.warning(
                "Kill switch activated",
                extra={
                    "mode": mode.value,
                    "reason": reason,
                    "operator": operator,
                },
            )

    async def deactivate(self, operator: str = "operator") -> None:
        """Deactivate the kill switch.

        Only PAUSE_NEW_ENTRIES mode can be deactivated programmatically.
        All other modes require manual system restart.
        """
        async with self._lock:
            if not self._active:
                return

            if self._mode not in self._deactivatable_modes():
                self._log.error(
                    "Kill switch deactivation refused: mode requires manual restart",
                    extra={"mode": self._mode.value if self._mode else "unknown"},
                )
                return

            self._log.info(
                "Kill switch deactivated",
                extra={
                    "previous_mode": self._mode.value if self._mode else "unknown",
                    "operator": operator,
                },
            )
            self._active = False
            self._mode = None
            self._activated_at = None
            self._activated_by = None
            self._reason = ""

    def _deactivatable_modes(self) -> set[KillSwitchMode]:
        return self._DEACTIVATABLE_MODES

    # ------------------------------------------------------------------
    # File flag polling
    # ------------------------------------------------------------------

    async def check_file_flag(self) -> None:
        """Check if the kill flag file exists and activate if so.

        The file content (if any) is used as the reason string.
        """
        if self._flag_file.exists():
            try:
                with self._flag_file.open(encoding="utf-8") as fh:
                    content = fh.read().strip()
                reason = content or "kill flag file present"
            except OSError:
                reason = "kill flag file present"
            # activate() handles escalation; always call it when flag file exists
            await self.activate(KillSwitchMode.FULL_STOP, reason=reason, operator="file_flag")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True if the kill switch is currently activated."""
        return self._active

    @property
    def current_mode(self) -> KillSwitchMode | None:
        """Current kill switch mode, or None if inactive."""
        return self._mode

    @property
    def activated_at(self) -> datetime | None:
        """UTC timestamp when kill switch was last activated."""
        return self._activated_at

    @property
    def activated_by(self) -> str | None:
        """Operator/source that activated the kill switch."""
        return self._activated_by

    @property
    def reason(self) -> str:
        """Reason for activation."""
        return self._reason

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------

    def new_entries_allowed(self) -> bool:
        """Return False if the kill switch blocks new position entries."""
        if not self._active:
            return True
        # All modes block new entries
        return False

    def orders_allowed(self) -> bool:
        """Return False if new orders of any type are blocked."""
        if not self._active:
            return True
        # PAUSE_NEW_ENTRIES allows reduce-only orders
        if self._mode == KillSwitchMode.PAUSE_NEW_ENTRIES:
            return True
        return False

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self._active,
            "mode": self._mode.value if self._mode else None,
            "activated_at": self._activated_at.isoformat() if self._activated_at else None,
            "activated_by": self._activated_by,
            "reason": self._reason,
        }
