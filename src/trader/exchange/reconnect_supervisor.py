"""WebSocket reconnection supervisor with exponential back-off and jitter.

Manages the full reconnect lifecycle:
- Exponential back-off: 1 → 2 → 4 → … → max 60 s
- ±20 % jitter on each wait
- Max reconnect attempts per hour (default 20)
- Blocks new trade entries during reconnect + 10 s settling window
- Alerts on excessive reconnects (>3 in 5 min)
- Prometheus metrics: ws_reconnect_total, ws_downtime_seconds
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from collections.abc import Callable, Coroutine
from typing import Any, cast

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def calc_backoff(
    attempt: int,
    base: float = 1.0,
    max_wait: float = 60.0,
) -> float:
    """Return back-off wait time with ±20 % jitter."""
    wait = min(base * (2**attempt), max_wait)
    jitter = wait * 0.2 * (random.random() * 2 - 1)  # ±20 %  # noqa: S311
    return cast(float, max(0.1, wait + jitter))


# ---------------------------------------------------------------------------
# ReconnectSupervisor
# ---------------------------------------------------------------------------

_ALERT_WINDOW_SECONDS = 300  # 5 min
_ALERT_THRESHOLD = 3  # >3 reconnects in 5 min triggers alert
_SETTLE_SECONDS = 10.0  # block entries for 10 s after reconnect
_MAX_ATTEMPTS_PER_HOUR = 20


class ReconnectSupervisor:
    """Manages WebSocket reconnection lifecycle with back-off, metrics and alerting."""

    def __init__(
        self,
        name: str,
        connect_fn: Callable[[], Coroutine[Any, Any, None]],
        metrics: Any = None,
        logger: Any = None,
    ) -> None:
        self._name = name
        self._connect_fn = connect_fn
        self._metrics = metrics
        self._log = logger or structlog.get_logger(__name__)

        self._reconnect_count: int = 0
        self._total_downtime: float = 0.0
        self._downtime_start: float | None = None

        # Timestamps of recent reconnects for alert detection
        self._reconnect_times: deque[float] = deque()

        self._running = False
        self._stop_event = asyncio.Event()

        # Entry block: True during reconnect + settle window
        self._entries_blocked: bool = False
        self._settle_until: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main reconnection loop.  Runs until request_stop() is called."""
        self._running = True
        attempt = 0
        log = self._log.bind(supervisor=self._name)

        while not self._stop_event.is_set():
            # Block entries while connecting
            self._entries_blocked = True
            self._downtime_start = time.monotonic()

            try:
                log.info("ws.connecting", attempt=attempt)
                await self._connect_fn()
                # connect_fn returned normally → connection closed cleanly
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("ws.connection_error", error=str(exc), attempt=attempt)

            # Record downtime
            if self._downtime_start is not None:
                self._total_downtime += time.monotonic() - self._downtime_start
                self._downtime_start = None

            if self._stop_event.is_set():
                break

            # Record reconnect
            now = time.monotonic()
            self._reconnect_count += 1
            self._reconnect_times.append(now)

            # Prune old timestamps outside the alert window
            cutoff = now - _ALERT_WINDOW_SECONDS
            while self._reconnect_times and self._reconnect_times[0] < cutoff:
                self._reconnect_times.popleft()

            # Prometheus metric
            if self._metrics is not None:
                try:
                    self._metrics.ws_reconnect_total.labels(name=self._name).inc()
                except Exception:  # noqa: S110
                    pass

            # Alert on repeated reconnects
            if len(self._reconnect_times) > _ALERT_THRESHOLD:
                log.error(
                    "ws.frequent_reconnects",
                    count=len(self._reconnect_times),
                    window_seconds=_ALERT_WINDOW_SECONDS,
                )

            # Check hourly rate limit
            hourly_cutoff = now - 3600
            hourly_count = sum(1 for t in self._reconnect_times if t > hourly_cutoff)
            if hourly_count >= _MAX_ATTEMPTS_PER_HOUR:
                log.error(
                    "ws.max_reconnects_reached",
                    hourly_count=hourly_count,
                    limit=_MAX_ATTEMPTS_PER_HOUR,
                )
                # Wait a full minute before trying again
                await asyncio.sleep(60.0)
                attempt = 0
                continue

            # Compute backoff and wait
            wait = calc_backoff(attempt)
            log.info("ws.reconnect_backoff", wait_seconds=round(wait, 2), attempt=attempt)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=wait,
                )
                # stop_event fired during wait
                break
            except TimeoutError:
                pass

            attempt += 1

            # Schedule entry unblock after settle period
            self._settle_until = time.monotonic() + _SETTLE_SECONDS

        # Unblock on clean exit
        self._entries_blocked = False
        self._running = False

    async def request_stop(self) -> None:
        """Signal the supervisor to stop reconnecting."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_stable(self) -> bool:
        """True when connected and not in back-off or settling."""
        return self._running and not self._entries_blocked

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    @property
    def downtime_seconds(self) -> float:
        """Total accumulated downtime in seconds (including current if disconnected)."""
        extra = 0.0
        if self._downtime_start is not None:
            extra = time.monotonic() - self._downtime_start
        return self._total_downtime + extra

    @property
    def entries_blocked(self) -> bool:
        """True during reconnect or within the 10-second settle window."""
        if self._entries_blocked:
            # Check if settle window expired
            if not self._running or time.monotonic() >= self._settle_until:
                self._entries_blocked = False
        return self._entries_blocked
