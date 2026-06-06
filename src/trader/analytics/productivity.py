"""Runtime productivity monitor.

Tracks the trading system's activity: when each subsystem last ran,
how many events occurred in the last hour, and overall counters.

Intended use:
- Call record_* from the relevant subsystem after each event.
- Call snapshot() to read the current state (e.g. for Telegram dashboard).
- Call log_heartbeat() once per minute from the main loop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_HOUR = timedelta(hours=1)


def _now() -> datetime:
    return datetime.now(tz=UTC)


@dataclass
class ProductivitySnapshot:
    """Point-in-time snapshot of trading system activity."""

    last_scanner_cycle_at: datetime | None
    last_feature_cycle_at: datetime | None
    last_strategy_loop_at: datetime | None
    last_signal_at: datetime | None
    last_approved_signal_at: datetime | None
    last_order_submitted_at: datetime | None
    last_order_filled_at: datetime | None
    last_position_closed_at: datetime | None
    last_position_reopened_at: datetime | None

    strategy_iterations_total: int
    scanner_iterations_total: int
    positions_opened_total: int
    positions_closed_total: int
    reentries_total: int

    signals_last_hour: int
    approved_last_hour: int
    filled_last_hour: int
    closed_last_hour: int

    top_rejection_reason: str | None


@dataclass
class RuntimeProductivityMonitor:
    """Tracks trading system activity with rolling-hour counters.

    All record_* methods are synchronous and non-blocking; they just
    update in-memory state. thread-safety is not required because the
    trading system runs in a single asyncio event loop.
    """

    _last_scanner_cycle_at: datetime | None = field(default=None, init=False)
    _last_feature_cycle_at: datetime | None = field(default=None, init=False)
    _last_strategy_loop_at: datetime | None = field(default=None, init=False)
    _last_signal_at: datetime | None = field(default=None, init=False)
    _last_approved_signal_at: datetime | None = field(default=None, init=False)
    _last_order_submitted_at: datetime | None = field(default=None, init=False)
    _last_order_filled_at: datetime | None = field(default=None, init=False)
    _last_position_closed_at: datetime | None = field(default=None, init=False)
    _last_position_reopened_at: datetime | None = field(default=None, init=False)

    _strategy_iterations_total: int = field(default=0, init=False)
    _scanner_iterations_total: int = field(default=0, init=False)
    _positions_opened_total: int = field(default=0, init=False)
    _positions_closed_total: int = field(default=0, init=False)
    _reentries_total: int = field(default=0, init=False)

    # Rolling deques: each element is a timestamp of that event
    _signal_times: deque[datetime] = field(default_factory=deque, init=False)
    _approved_times: deque[datetime] = field(default_factory=deque, init=False)
    _fill_times: deque[datetime] = field(default_factory=deque, init=False)
    _close_times: deque[datetime] = field(default_factory=deque, init=False)

    # Rejection reason counts (last hour; reset on each heartbeat if needed)
    _rejection_counts: dict[str, int] = field(default_factory=dict, init=False)

    def record_scanner_cycle(self) -> None:
        self._last_scanner_cycle_at = _now()
        self._scanner_iterations_total += 1

    def record_feature_cycle(self) -> None:
        self._last_feature_cycle_at = _now()

    def record_strategy_loop(self) -> None:
        self._last_strategy_loop_at = _now()
        self._strategy_iterations_total += 1

    def record_signal(self, *, approved: bool = False) -> None:
        t = _now()
        self._last_signal_at = t
        self._signal_times.append(t)
        if approved:
            self._last_approved_signal_at = t
            self._approved_times.append(t)

    def record_order_submitted(self) -> None:
        self._last_order_submitted_at = _now()
        self._positions_opened_total += 1

    def record_fill(self) -> None:
        t = _now()
        self._last_order_filled_at = t
        self._fill_times.append(t)

    def record_position_closed(self, *, reentry: bool = False) -> None:
        t = _now()
        self._last_position_closed_at = t
        self._close_times.append(t)
        self._positions_closed_total += 1
        if reentry:
            self._last_position_reopened_at = t
            self._reentries_total += 1

    def record_rejection(self, reason: str) -> None:
        self._rejection_counts[reason] = self._rejection_counts.get(reason, 0) + 1

    def snapshot(self) -> ProductivitySnapshot:
        cutoff = _now() - _HOUR
        self._prune(cutoff)

        top_reason: str | None = None
        if self._rejection_counts:
            top_reason = max(self._rejection_counts, key=lambda k: self._rejection_counts[k])

        return ProductivitySnapshot(
            last_scanner_cycle_at=self._last_scanner_cycle_at,
            last_feature_cycle_at=self._last_feature_cycle_at,
            last_strategy_loop_at=self._last_strategy_loop_at,
            last_signal_at=self._last_signal_at,
            last_approved_signal_at=self._last_approved_signal_at,
            last_order_submitted_at=self._last_order_submitted_at,
            last_order_filled_at=self._last_order_filled_at,
            last_position_closed_at=self._last_position_closed_at,
            last_position_reopened_at=self._last_position_reopened_at,
            strategy_iterations_total=self._strategy_iterations_total,
            scanner_iterations_total=self._scanner_iterations_total,
            positions_opened_total=self._positions_opened_total,
            positions_closed_total=self._positions_closed_total,
            reentries_total=self._reentries_total,
            signals_last_hour=len(self._signal_times),
            approved_last_hour=len(self._approved_times),
            filled_last_hour=len(self._fill_times),
            closed_last_hour=len(self._close_times),
            top_rejection_reason=top_reason,
        )

    def log_heartbeat(self, **extra: Any) -> None:
        """Log a structured productivity heartbeat. Call once per minute."""
        snap = self.snapshot()

        def _age(ts: datetime | None) -> float | None:
            if ts is None:
                return None
            return round((_now() - ts).total_seconds(), 1)

        log.info(
            "runtime.productivity_heartbeat",
            last_strategy_loop_age_s=_age(snap.last_strategy_loop_at),
            last_fill_age_s=_age(snap.last_order_filled_at),
            last_close_age_s=_age(snap.last_position_closed_at),
            strategy_iterations=snap.strategy_iterations_total,
            scanner_iterations=snap.scanner_iterations_total,
            positions_opened=snap.positions_opened_total,
            positions_closed=snap.positions_closed_total,
            reentries=snap.reentries_total,
            signals_last_hour=snap.signals_last_hour,
            approved_last_hour=snap.approved_last_hour,
            filled_last_hour=snap.filled_last_hour,
            closed_last_hour=snap.closed_last_hour,
            top_rejection_reason=snap.top_rejection_reason,
            **extra,
        )

    def _prune(self, cutoff: datetime) -> None:
        """Remove events older than the rolling window."""
        for dq in (
            self._signal_times,
            self._approved_times,
            self._fill_times,
            self._close_times,
        ):
            while dq and dq[0] < cutoff:
                dq.popleft()
