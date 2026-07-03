"""Circuit breaker manager for the Bybit AI trading system.

Implements all circuit breakers from spec section 11.4.
Thread-safe via asyncio.Lock.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from trader.risk.profiles import RiskLimits

logger = logging.getLogger(__name__)


class CircuitBreakerType(StrEnum):
    """All supported circuit breaker types."""

    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MAX_DRAWDOWN = "max_drawdown"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    SLIPPAGE_EXCEEDED = "slippage_exceeded"
    ANOMALOUS_SPREAD = "anomalous_spread"
    LOW_LIQUIDITY = "low_liquidity"
    MISSING_STOP_LOSS = "missing_stop_loss"
    POSITION_DESYNC = "position_desync"
    WEBSOCKET_STALE = "websocket_stale"
    REST_ERRORS = "rest_errors"
    RATE_LIMIT_PRESSURE = "rate_limit_pressure"
    HIGH_LATENCY = "high_latency"
    ANOMALOUS_ORDER_COUNT = "anomalous_order_count"
    UNKNOWN_MANUAL_ORDER = "unknown_manual_order"
    CORRUPTED_FEATURES = "corrupted_features"
    DRIFT_ALERT = "drift_alert"
    DATABASE_UNAVAILABLE = "database_unavailable"
    NTP_DRIFT = "ntp_drift"
    STALE_BALANCE = "stale_balance"
    UNEXPECTED_LEVERAGE = "unexpected_leverage"
    REGION_MISMATCH = "region_mismatch"
    API_KEY_ANOMALY = "api_key_anomaly"


# Severity ordering (higher index = more severe)
_SEVERITY_ORDER = ["WARNING", "STOP_ENTRIES", "SAFE_MODE", "EMERGENCY"]


@dataclass
class CircuitBreakerState:
    """State of an individual circuit breaker."""

    breaker_type: CircuitBreakerType
    triggered: bool = False
    triggered_at: datetime | None = None
    reason: str = ""
    severity: str = "WARNING"
    auto_reset_after_seconds: int | None = None


# Default configuration for each circuit breaker type
_BREAKER_DEFAULTS: dict[CircuitBreakerType, dict[str, Any]] = {
    CircuitBreakerType.DAILY_LOSS_LIMIT: {
        "severity": "STOP_ENTRIES",
        "auto_reset_after_seconds": None,
    },
    CircuitBreakerType.MAX_DRAWDOWN: {
        "severity": "STOP_ENTRIES",
        "auto_reset_after_seconds": None,
    },
    CircuitBreakerType.CONSECUTIVE_LOSSES: {
        "severity": "STOP_ENTRIES",
        "auto_reset_after_seconds": 3600,
    },
    CircuitBreakerType.SLIPPAGE_EXCEEDED: {
        "severity": "WARNING",
        "auto_reset_after_seconds": 300,
    },
    CircuitBreakerType.ANOMALOUS_SPREAD: {
        "severity": "STOP_ENTRIES",
        "auto_reset_after_seconds": 300,
    },
    CircuitBreakerType.LOW_LIQUIDITY: {
        "severity": "STOP_ENTRIES",
        "auto_reset_after_seconds": 600,
    },
    CircuitBreakerType.MISSING_STOP_LOSS: {
        "severity": "EMERGENCY",
        "auto_reset_after_seconds": None,
    },
    CircuitBreakerType.POSITION_DESYNC: {
        "severity": "SAFE_MODE",
        "auto_reset_after_seconds": None,
    },
    CircuitBreakerType.WEBSOCKET_STALE: {
        "severity": "SAFE_MODE",
        "auto_reset_after_seconds": 120,
    },
    CircuitBreakerType.REST_ERRORS: {
        "severity": "STOP_ENTRIES",
        "auto_reset_after_seconds": 300,
    },
    CircuitBreakerType.RATE_LIMIT_PRESSURE: {
        "severity": "WARNING",
        "auto_reset_after_seconds": 60,
    },
    CircuitBreakerType.HIGH_LATENCY: {
        "severity": "WARNING",
        "auto_reset_after_seconds": 120,
    },
    CircuitBreakerType.ANOMALOUS_ORDER_COUNT: {
        "severity": "SAFE_MODE",
        "auto_reset_after_seconds": None,
    },
    CircuitBreakerType.UNKNOWN_MANUAL_ORDER: {
        "severity": "SAFE_MODE",
        "auto_reset_after_seconds": None,
    },
    CircuitBreakerType.CORRUPTED_FEATURES: {
        "severity": "STOP_ENTRIES",
        "auto_reset_after_seconds": 300,
    },
    CircuitBreakerType.DRIFT_ALERT: {
        "severity": "WARNING",
        "auto_reset_after_seconds": 3600,
    },
    CircuitBreakerType.DATABASE_UNAVAILABLE: {
        "severity": "EMERGENCY",
        "auto_reset_after_seconds": None,
    },
    CircuitBreakerType.NTP_DRIFT: {
        "severity": "STOP_ENTRIES",
        "auto_reset_after_seconds": 300,
    },
    CircuitBreakerType.STALE_BALANCE: {
        "severity": "STOP_ENTRIES",
        "auto_reset_after_seconds": 120,
    },
    CircuitBreakerType.UNEXPECTED_LEVERAGE: {
        "severity": "EMERGENCY",
        "auto_reset_after_seconds": None,
    },
    CircuitBreakerType.REGION_MISMATCH: {
        "severity": "EMERGENCY",
        "auto_reset_after_seconds": None,
    },
    CircuitBreakerType.API_KEY_ANOMALY: {
        "severity": "EMERGENCY",
        "auto_reset_after_seconds": None,
    },
}


class CircuitBreakerManager:
    """Manages all circuit breakers.

    When triggered: logs event, updates state, applies action.
    Thread-safe via asyncio.Lock.
    """

    def __init__(
        self,
        risk_limits: RiskLimits,
        metrics: Any = None,
        event_bus: Any = None,
        log: logging.Logger | None = None,
    ) -> None:
        self._limits = risk_limits
        self._metrics = metrics
        self._event_bus = event_bus
        self._log = log or logger
        self._lock = asyncio.Lock()

        # Initialise all circuit breakers in un-triggered state
        self._breakers: dict[CircuitBreakerType, CircuitBreakerState] = {}
        for bt, cfg in _BREAKER_DEFAULTS.items():
            self._breakers[bt] = CircuitBreakerState(
                breaker_type=bt,
                triggered=False,
                severity=cfg["severity"],
                auto_reset_after_seconds=cfg["auto_reset_after_seconds"],
            )

    # ------------------------------------------------------------------
    # Check methods (each returns True if triggered)
    # ------------------------------------------------------------------

    async def check_daily_loss(
        self,
        daily_pnl: Decimal,
        capital: Decimal,
    ) -> bool:
        """Trigger if daily loss exceeds the limit.

        Fails closed when capital is zero or unknown: trigger the breaker rather
        than bypassing it, because a zero capital reading most likely indicates a
        stale/failed balance query, not a genuinely zero account.
        """
        if capital <= Decimal("0"):
            await self.trigger(
                CircuitBreakerType.DAILY_LOSS_LIMIT,
                "capital reported as zero or negative — failing closed on daily loss check",
            )
            return True
        loss_pct = abs(daily_pnl) / capital * Decimal("100") if daily_pnl < 0 else Decimal("0")
        if loss_pct >= self._limits.daily_loss_limit_pct:
            await self.trigger(
                CircuitBreakerType.DAILY_LOSS_LIMIT,
                f"daily loss {loss_pct:.2f}% >= limit {self._limits.daily_loss_limit_pct}%",
            )
            return True
        return False

    async def check_drawdown(self, drawdown_pct: Decimal) -> bool:
        """Trigger if drawdown exceeds the hard stop."""
        if drawdown_pct >= self._limits.hard_stop_drawdown_pct:
            await self.trigger(
                CircuitBreakerType.MAX_DRAWDOWN,
                f"drawdown {drawdown_pct:.2f}% >= hard stop {self._limits.hard_stop_drawdown_pct}%",
            )
            return True
        return False

    async def check_consecutive_losses(
        self,
        loss_streak: int,
        threshold: int = 5,
    ) -> bool:
        """Trigger if consecutive losses exceed threshold."""
        if loss_streak >= threshold:
            await self.trigger(
                CircuitBreakerType.CONSECUTIVE_LOSSES,
                f"consecutive losses {loss_streak} >= threshold {threshold}",
            )
            return True
        return False

    async def check_slippage(
        self,
        actual_slippage_pct: Decimal,
        threshold_pct: Decimal = Decimal("0.5"),
    ) -> bool:
        """Trigger if slippage exceeds threshold."""
        if actual_slippage_pct > threshold_pct:
            await self.trigger(
                CircuitBreakerType.SLIPPAGE_EXCEEDED,
                f"slippage {actual_slippage_pct:.4f}% > threshold {threshold_pct}%",
            )
            return True
        return False

    async def check_spread(
        self,
        spread_pct: Decimal,
        threshold_pct: Decimal = Decimal("0.3"),
    ) -> bool:
        """Trigger if spread is anomalously wide."""
        if spread_pct > threshold_pct:
            await self.trigger(
                CircuitBreakerType.ANOMALOUS_SPREAD,
                f"spread {spread_pct:.4f}% > threshold {threshold_pct}%",
            )
            return True
        return False

    async def check_websocket_staleness(
        self,
        last_message_age_seconds: float,
    ) -> bool:
        """Trigger if websocket last message is too old."""
        threshold = 30.0
        if last_message_age_seconds > threshold:
            await self.trigger(
                CircuitBreakerType.WEBSOCKET_STALE,
                f"websocket stale: {last_message_age_seconds:.1f}s since last message",
            )
            return True
        return False

    async def check_rest_error_rate(
        self,
        error_count: int,
        window_seconds: int = 60,
    ) -> bool:
        """Trigger if REST error count is too high."""
        threshold = 5
        if error_count >= threshold:
            await self.trigger(
                CircuitBreakerType.REST_ERRORS,
                f"{error_count} REST errors in {window_seconds}s window",
            )
            return True
        return False

    async def check_feature_quality(
        self,
        quality_score: float,
        threshold: float = 0.5,
    ) -> bool:
        """Trigger if feature quality is below threshold."""
        if quality_score < threshold:
            await self.trigger(
                CircuitBreakerType.CORRUPTED_FEATURES,
                f"feature quality {quality_score:.3f} < threshold {threshold}",
            )
            return True
        return False

    async def check_ntp_drift(
        self,
        drift_seconds: float,
        threshold: float = 5.0,
    ) -> bool:
        """Trigger if NTP clock drift is too large."""
        if abs(drift_seconds) > threshold:
            await self.trigger(
                CircuitBreakerType.NTP_DRIFT,
                f"NTP drift {drift_seconds:.2f}s > threshold {threshold}s",
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Core trigger / reset
    # ------------------------------------------------------------------

    async def trigger(
        self,
        breaker_type: CircuitBreakerType,
        reason: str,
    ) -> None:
        """Trigger a circuit breaker."""
        async with self._lock:
            state = self._breakers[breaker_type]
            if not state.triggered:
                self._breakers[breaker_type] = CircuitBreakerState(
                    breaker_type=breaker_type,
                    triggered=True,
                    triggered_at=datetime.now(tz=UTC),
                    reason=reason,
                    severity=state.severity,
                    auto_reset_after_seconds=state.auto_reset_after_seconds,
                )
                self._log.warning(
                    "Circuit breaker triggered",
                    extra={
                        "breaker": breaker_type.value,
                        "reason": reason,
                        "severity": state.severity,
                    },
                )

    async def reset(self, breaker_type: CircuitBreakerType) -> None:
        """Manually reset a circuit breaker."""
        async with self._lock:
            state = self._breakers[breaker_type]
            self._breakers[breaker_type] = CircuitBreakerState(
                breaker_type=breaker_type,
                triggered=False,
                triggered_at=None,
                reason="",
                severity=state.severity,
                auto_reset_after_seconds=state.auto_reset_after_seconds,
            )
            self._log.info(
                "Circuit breaker reset",
                extra={"breaker": breaker_type.value},
            )

    async def reset_all_auto(self) -> None:
        """Reset all circuit breakers that have auto_reset_after_seconds configured."""
        now = datetime.now(tz=UTC)

        async with self._lock:
            for bt, state in list(self._breakers.items()):
                if state.triggered and state.auto_reset_after_seconds is not None and state.triggered_at is not None:
                    age = (now - state.triggered_at).total_seconds()
                    if age >= state.auto_reset_after_seconds:
                        self._breakers[bt] = CircuitBreakerState(
                            breaker_type=bt,
                            triggered=False,
                            triggered_at=None,
                            reason="",
                            severity=state.severity,
                            auto_reset_after_seconds=state.auto_reset_after_seconds,
                        )
                        self._log.info(
                            "Circuit breaker reset",
                            extra={"breaker": bt.value},
                        )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def is_triggered(self, breaker_type: CircuitBreakerType) -> bool:
        """Return True if the given circuit breaker is currently triggered."""
        return self._breakers[breaker_type].triggered

    def any_triggered(self) -> bool:
        """Return True if any circuit breaker is triggered."""
        return any(state.triggered for state in self._breakers.values())

    def get_triggered(self) -> list[CircuitBreakerState]:
        """Return a list of all currently triggered circuit breaker states."""
        return [state for state in self._breakers.values() if state.triggered]

    def should_block_entries(self) -> bool:
        """Return True if any STOP_ENTRIES or higher breaker is triggered."""
        block_severities = {"STOP_ENTRIES", "SAFE_MODE", "EMERGENCY"}
        return any(state.triggered and state.severity in block_severities for state in self._breakers.values())

    def should_safe_mode(self) -> bool:
        """Return True if any SAFE_MODE or higher breaker is triggered."""
        safe_severities = {"SAFE_MODE", "EMERGENCY"}
        return any(state.triggered and state.severity in safe_severities for state in self._breakers.values())

    def should_emergency(self) -> bool:
        """Return True if any EMERGENCY breaker is triggered."""
        return any(state.triggered and state.severity == "EMERGENCY" for state in self._breakers.values())

    def to_dict(self) -> dict[str, Any]:
        """Serialise all breaker states."""
        return {
            bt.value: {
                "triggered": state.triggered,
                "triggered_at": state.triggered_at.isoformat() if state.triggered_at else None,
                "reason": state.reason,
                "severity": state.severity,
                "auto_reset_after_seconds": state.auto_reset_after_seconds,
            }
            for bt, state in self._breakers.items()
        }
