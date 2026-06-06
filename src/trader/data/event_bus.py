"""In-process async event bus using asyncio.Queue.

Separate queues for different event categories with backpressure handling,
dead letter queue for critical events, and graceful shutdown support.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import Any

import structlog

from trader.domain.events import BaseEvent

logger = structlog.get_logger(__name__)


class EventBus:
    """In-process async event bus using asyncio.Queue.

    Separate queues for:
    - market_data: orderbook, trades, ticker, kline events
    - execution: order, fill, position, balance events
    - risk: risk decisions, circuit breaker events
    - persistence: events to be saved to DB
    - system: health, alerts, status changes

    Features
    --------
    - Bounded queues (configurable maxsize, default 10000).
    - Backpressure: if queue full, drop non-critical events.
    - Dropped event counter (Prometheus-compatible).
    - Dead letter queue for critical dropped events.
    - Graceful shutdown: drain queues before exit.
    - Idempotent consumer design: consumers handle duplicate events.
    - Append-only journal for critical events (via structlog).
    """

    QUEUE_NAMES = ["market_data", "execution", "risk", "persistence", "system"]

    def __init__(self, maxsize: int = 10000, metrics: Any = None) -> None:
        self._maxsize = maxsize
        self._metrics = metrics

        # Main queues
        self._queues: dict[str, asyncio.Queue] = {name: asyncio.Queue(maxsize=maxsize) for name in self.QUEUE_NAMES}

        # Dead letter queue for critical dropped events (unbounded)
        self._dead_letter: asyncio.Queue = asyncio.Queue()

        # Drop counters per queue
        self._dropped: dict[str, int] = defaultdict(int)

        # Shutdown flag
        self._shutdown = False

        self._log = structlog.get_logger(__name__)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(
        self,
        queue_name: str,
        event: BaseEvent,
        critical: bool = False,
    ) -> bool:
        """Publish an event to the named queue.

        Parameters
        ----------
        queue_name:
            One of QUEUE_NAMES.
        event:
            The event to publish.
        critical:
            If True and the queue is full, the event goes to the dead
            letter queue instead of being silently dropped.

        Returns
        -------
        bool
            True if published to main queue, False if dropped/dead-lettered.
        """
        if queue_name not in self._queues:
            self._log.warning("event_bus.unknown_queue", queue_name=queue_name)
            return False

        queue = self._queues[queue_name]
        try:
            queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            self._dropped[queue_name] += 1

            # Update Prometheus if available
            if self._metrics is not None:
                try:
                    self._metrics.events_dropped_total.labels(queue=queue_name).inc()
                except Exception:  # noqa: S110
                    pass

            if critical:
                # Send to dead letter queue (never drop critical events)
                await self._dead_letter.put(event)
                self._log.warning(
                    "event_bus.critical_to_dead_letter",
                    queue=queue_name,
                    event_type=type(event).__name__,
                )
            else:
                self._log.debug(
                    "event_bus.dropped",
                    queue=queue_name,
                    event_type=type(event).__name__,
                    total_dropped=self._dropped[queue_name],
                )
            return False

    # ------------------------------------------------------------------
    # Consuming
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        queue_name: str,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Async generator yielding events from the named queue.

        Stops when shutdown() is called.
        """
        if queue_name not in self._queues:
            raise ValueError(f"Unknown queue: {queue_name!r}")

        queue = self._queues[queue_name]
        while not self._shutdown:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield event
                queue.task_done()
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def get(self, queue_name: str, timeout: float | None = None) -> BaseEvent | None:
        """Get a single event from the named queue."""
        if queue_name not in self._queues:
            raise ValueError(f"Unknown queue: {queue_name!r}")
        queue = self._queues[queue_name]
        try:
            if timeout is not None:
                return await asyncio.wait_for(queue.get(), timeout=timeout)
            return await queue.get()
        except TimeoutError:
            return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def drain(self, timeout: float = 5.0) -> None:
        """Wait for all queues to be processed, with a timeout."""
        tasks = []
        for _name, queue in self._queues.items():
            if queue.qsize() > 0:
                tasks.append(asyncio.wait_for(queue.join(), timeout=timeout))
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:  # noqa: S110
                pass

    async def shutdown(self) -> None:
        """Signal shutdown and drain queues gracefully."""
        self._shutdown = True
        await self.drain(timeout=5.0)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_queue_sizes(self) -> dict[str, int]:
        """Return current size of each queue."""
        return {name: q.qsize() for name, q in self._queues.items()}

    def get_dropped_count(self) -> dict[str, int]:
        """Return dropped event counts per queue."""
        return dict(self._dropped)

    def get_dead_letter_size(self) -> int:
        """Return number of events in the dead letter queue."""
        return self._dead_letter.qsize()

    async def drain_dead_letter(self) -> list[BaseEvent]:
        """Drain and return all dead letter events."""
        events: list[BaseEvent] = []
        while not self._dead_letter.empty():
            try:
                events.append(self._dead_letter.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events
