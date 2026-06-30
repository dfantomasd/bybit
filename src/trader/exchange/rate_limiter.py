"""Centralized adaptive token-bucket rate limiter with per-endpoint tracking."""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from trader.domain.errors import RateLimitError

try:
    from prometheus_client import Gauge

    _RATE_LIMIT_REMAINING = Gauge(
        "bybit_rate_limit_remaining",
        "Remaining API call capacity for a given endpoint",
        ["endpoint"],
    )
    _RATE_LIMIT_USAGE_PCT = Gauge(
        "bybit_rate_limit_usage_pct",
        "Rate limit usage percentage for a given endpoint",
        ["endpoint"],
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

logger = structlog.get_logger(__name__)

# Warning thresholds (usage %)
_WARN_THRESHOLDS = (70.0, 85.0, 95.0)

# Default token bucket configuration
_DEFAULT_CAPACITY = 120  # requests per window
_DEFAULT_REFILL_RATE = 2.0  # tokens per second
_DEFAULT_WINDOW_SECONDS = 60

# Backoff configuration
_BASE_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 60.0
_BACKOFF_MULTIPLIER = 2.0
_JITTER_FRACTION = 0.25


@dataclass
class _EndpointState:
    """Per-endpoint token bucket and limit state."""

    capacity: float = _DEFAULT_CAPACITY
    tokens: float = _DEFAULT_CAPACITY
    refill_rate: float = _DEFAULT_REFILL_RATE  # tokens/s
    last_refill_ts: float = field(default_factory=time.monotonic)

    # From exchange headers
    limit_total: int | None = None
    limit_remaining: int | None = None
    limit_reset_ts_ms: int | None = None  # Unix ms

    # Backoff state
    consecutive_429s: int = 0

    # Warning bookkeeping: last threshold logged
    last_warned_threshold: float = 0.0

    def refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill_ts
        added = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + added)
        self.last_refill_ts = now

    @property
    def usage_pct(self) -> float:
        """Usage percentage (0–100). Higher = more constrained."""
        if self.limit_total and self.limit_remaining is not None:
            used = self.limit_total - self.limit_remaining
            return (used / self.limit_total) * 100.0
        # Fall back to local token bucket
        if self.capacity > 0:
            return ((self.capacity - self.tokens) / self.capacity) * 100.0
        return 0.0

    @property
    def remaining_capacity(self) -> int:
        """Remaining capacity in absolute terms."""
        if self.limit_remaining is not None:
            return self.limit_remaining
        return max(0, int(self.tokens))


class RateLimiter:
    """Adaptive rate limiter with per-endpoint token buckets.

    Reads X-Bapi-Limit-Status, X-Bapi-Limit, X-Bapi-Limit-Reset-Timestamp
    response headers to calibrate limits in real time.

    Emits Prometheus metrics for remaining capacity.
    Warns at 70%, 85%, and 95% usage.
    Applies exponential backoff with jitter on 429 / 10006 errors.
    """

    def __init__(
        self,
        default_capacity: int = _DEFAULT_CAPACITY,
        default_refill_rate: float = _DEFAULT_REFILL_RATE,
    ) -> None:
        self._default_capacity = default_capacity
        self._default_refill_rate = default_refill_rate
        self._states: dict[str, _EndpointState] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key(self, endpoint: str, method: str) -> str:
        return f"{method.upper()}:{endpoint}"

    def _get_or_create(self, endpoint: str, method: str) -> _EndpointState:
        k = self._key(endpoint, method)
        if k not in self._states:
            self._states[k] = _EndpointState(
                capacity=self._default_capacity,
                tokens=self._default_capacity,
                refill_rate=self._default_refill_rate,
            )
        return self._states[k]

    def _emit_metrics(self, key: str, state: _EndpointState) -> None:
        if not _PROMETHEUS_AVAILABLE:
            return
        try:
            _RATE_LIMIT_REMAINING.labels(endpoint=key).set(state.remaining_capacity)
            _RATE_LIMIT_USAGE_PCT.labels(endpoint=key).set(state.usage_pct)
        except Exception:  # pragma: no cover  # noqa: S110
            pass

    def _check_thresholds(self, key: str, state: _EndpointState) -> None:
        usage = state.usage_pct
        for threshold in sorted(_WARN_THRESHOLDS, reverse=True):
            if usage >= threshold and state.last_warned_threshold < threshold:
                state.last_warned_threshold = threshold
                logger.warning(
                    "rate_limit_threshold_reached",
                    endpoint=key,
                    usage_pct=round(usage, 1),
                    threshold_pct=threshold,
                    remaining=state.remaining_capacity,
                )
                break
        # Reset warning if usage drops below all thresholds
        if usage < _WARN_THRESHOLDS[0]:
            state.last_warned_threshold = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def acquire(self, endpoint: str, method: str = "GET") -> None:
        """Wait until a token is available for the given endpoint/method pair.

        Uses an iterative loop instead of recursion to avoid stack overflow
        under prolonged exchange-side rate limiting.
        """
        key = self._key(endpoint, method)
        while True:
            async with self._lock:
                state = self._get_or_create(endpoint, method)
                state.refill()

                if state.tokens >= 1.0:
                    state.tokens -= 1.0
                    self._emit_metrics(key, state)
                    self._check_thresholds(key, state)
                    return

                # Compute wait while still holding the lock so refill_rate is stable.
                wait_seconds = max(0.01, (1.0 - state.tokens) / max(state.refill_rate, 1e-9))

            logger.debug(
                "rate_limiter_waiting",
                endpoint=key,
                wait_seconds=round(wait_seconds, 3),
            )
            await asyncio.sleep(wait_seconds)

    def record_response(self, endpoint: str, headers: dict[str, Any], method: str = "GET") -> None:
        """Update internal state from Bybit response headers.

        Headers parsed:
        - X-Bapi-Limit-Status: remaining calls
        - X-Bapi-Limit: total capacity
        - X-Bapi-Limit-Reset-Timestamp: reset epoch ms

        Note: this method is synchronous and called from within the same asyncio
        event loop as acquire(). In asyncio there is no concurrent interleaving
        between two synchronous code paths (no await means no yield), so the
        updates are atomic within a single event-loop turn. The lock is NOT
        acquired here to avoid deadlock (callers may already hold it indirectly).
        All mutations are therefore safe in a single-event-loop context.
        """
        key = self._key(endpoint, method)
        state = self._get_or_create(endpoint, method)

        # Parse headers (case-insensitive lookup)
        lower_headers = {k.lower(): v for k, v in headers.items()}

        try:
            if "x-bapi-limit-status" in lower_headers:
                state.limit_remaining = int(lower_headers["x-bapi-limit-status"])
            if "x-bapi-limit" in lower_headers:
                limit_total = int(lower_headers["x-bapi-limit"])
                state.limit_total = limit_total
                # Sync local bucket capacity with server-reported limit atomically.
                if limit_total != state.capacity:
                    state.capacity = float(limit_total)
                    state.refill_rate = limit_total / _DEFAULT_WINDOW_SECONDS
            if "x-bapi-limit-reset-timestamp" in lower_headers:
                state.limit_reset_ts_ms = int(lower_headers["x-bapi-limit-reset-timestamp"])
            # Reset consecutive 429 counter on successful response
            state.consecutive_429s = 0
        except (ValueError, TypeError) as exc:
            logger.warning("rate_limiter_header_parse_error", endpoint=key, error=str(exc))

        self._emit_metrics(key, state)
        self._check_thresholds(key, state)

    def handle_rate_limit_error(
        self,
        endpoint: str,
        retry_after: float | None = None,
        method: str = "GET",
    ) -> float:
        """Record a rate limit hit and return the number of seconds to wait.

        Uses exponential backoff with jitter:
            wait = min(base * 2^n + jitter, max_backoff)
        """
        state = self._get_or_create(endpoint, method)
        state.consecutive_429s += 1
        n = state.consecutive_429s - 1  # 0-indexed for first hit

        if retry_after is not None and retry_after > 0:
            wait = retry_after
        else:
            wait = min(
                _BASE_BACKOFF_SECONDS * math.pow(_BACKOFF_MULTIPLIER, n),
                _MAX_BACKOFF_SECONDS,
            )

        # Add jitter: ±JITTER_FRACTION of computed wait
        jitter = wait * _JITTER_FRACTION * (random.random() * 2 - 1)  # noqa: S311
        wait = max(0.5, wait + jitter)

        key = self._key(endpoint, method)
        logger.warning(
            "rate_limit_hit_backing_off",
            endpoint=key,
            consecutive_hits=state.consecutive_429s,
            wait_seconds=round(wait, 2),
        )
        return wait

    def get_status(self) -> dict[str, Any]:
        """Return current rate limit status for all tracked endpoints."""
        result: dict[str, Any] = {}
        for key, state in self._states.items():
            result[key] = {
                "capacity": state.capacity,
                "tokens": round(state.tokens, 2),
                "limit_total": state.limit_total,
                "limit_remaining": state.limit_remaining,
                "limit_reset_ts_ms": state.limit_reset_ts_ms,
                "usage_pct": round(state.usage_pct, 1),
                "consecutive_429s": state.consecutive_429s,
            }
        return result

    def get_usage_pct(self, endpoint: str, method: str = "GET") -> float:
        """Return usage percentage for a specific endpoint."""
        key = self._key(endpoint, method)
        if key not in self._states:
            return 0.0
        return self._states[key].usage_pct

    async def wait_for_rate_limit_reset(self, endpoint: str, method: str = "GET") -> None:
        """Wait until rate limit resets, then raise RateLimitError if still limited."""
        state = self._get_or_create(endpoint, method)
        wait = self.handle_rate_limit_error(endpoint, method=method)
        await asyncio.sleep(wait)
        if state.usage_pct >= 100.0:
            raise RateLimitError(
                f"Rate limit still exceeded for {endpoint} after backoff",
                retry_after_ms=int(wait * 1000),
            )
