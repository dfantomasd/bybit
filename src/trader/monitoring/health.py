"""Health check service.

Provides individual component health checks and an aggregate ``overall_health``
method that returns a ``HealthStatus`` domain model.

Designed to be instantiated once and called periodically or on HTTP /health
requests.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import UTC, datetime
from typing import Any

import asyncpg
import redis.asyncio as aioredis

from trader.domain.enums import SystemStatus, TradingMode
from trader.domain.models import HealthStatus
from trader.monitoring.logging import get_logger
from trader.storage.trade_journal import asyncpg_pool_connect_kwargs

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_DB_TIMEOUT_S = 3.0
_REDIS_TIMEOUT_S = 2.0
_BYBIT_TIMEOUT_S = 5.0
_MODEL_STALE_THRESHOLD_S = 3600.0  # 1 hour
_FEATURE_STALE_THRESHOLD_S = 120.0  # 2 minutes — allows pipeline warm-up on startup
_OVERALL_HEALTH_CACHE_TTL_S = 2.0  # debounce repeated unauthenticated /readyz hits
_CIRCUIT_BREAKER_MARKERS = (
    "ECIRCUITBREAKER",
    "EAUTHQUERY",
    "authentication query failed",
    "failed to retrieve database credentials",
    "too many authentication failures",
)


_DSN_CREDENTIALS_RE = re.compile(r"//[^/@\s]+@")


def _scrub_dsn_credentials(text: str) -> str:
    """Strip embedded user:pass@ credentials from a connection string that may
    appear inside a driver exception message before it is logged."""
    return _DSN_CREDENTIALS_RE.sub("//***@", text)


def _postgres_retry_wait_seconds(
    *,
    attempt: int,
    base_delay_seconds: float,
    error_text: str | None,
) -> float:
    """Linear backoff; longer waits when Supabase auth circuit breaker is open."""
    linear = base_delay_seconds * attempt
    if error_text and any(marker in error_text for marker in _CIRCUIT_BREAKER_MARKERS):
        return max(linear, 15.0 * attempt)
    return linear


class HealthChecker:
    """Aggregates health checks for all system components.

    Args:
        postgres_dsn:  asyncpg-compatible connection string.
        redis_url:     Redis URL string.
        bybit_rest_url: Base URL for the Bybit REST API (used for ping).
    """

    def __init__(
        self,
        postgres_dsn: str,
        redis_url: str,
        redis_required: bool = False,
        bybit_required: bool = False,
        bybit_rest_url: str = "https://api.bybit.com",
        trading_mode: TradingMode = TradingMode.TESTNET,
        system_status: SystemStatus = SystemStatus.STOPPED,
        model_enabled: bool = False,
        postgres_retry_attempts: int = 6,
        postgres_retry_delay_s: float = 2.0,
        postgres_required: bool = True,
        postgres_optional_max_attempts: int = 3,
    ) -> None:
        self._postgres_connect_kwargs = asyncpg_pool_connect_kwargs(postgres_dsn) if postgres_dsn else {}
        self._redis_url = redis_url.strip().strip("\"'")
        self._redis_required = redis_required
        self._bybit_required = bybit_required
        self._bybit_rest_url = bybit_rest_url
        self._trading_mode = trading_mode
        self._system_status = system_status
        self._model_enabled = model_enabled
        self._postgres_retry_attempts = max(1, int(postgres_retry_attempts))
        self._postgres_retry_delay_s = max(0.0, float(postgres_retry_delay_s))
        self._postgres_required = postgres_required
        self._postgres_optional_max_attempts = max(1, int(postgres_optional_max_attempts))

        # Mutable state updated by the trading system
        self._ws_connected: bool = False
        self._last_ws_message_at: datetime | None = None
        self._last_model_inference_at: datetime | None = None
        self._last_feature_computed_at: datetime | None = None
        self._health_cache: HealthStatus | None = None
        self._health_cache_at: float = 0.0
        self._health_cache_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # State setters (called by other components)
    # ------------------------------------------------------------------

    def set_ws_status(self, connected: bool, last_message_at: datetime | None = None) -> None:
        self._ws_connected = connected
        if last_message_at:
            self._last_ws_message_at = last_message_at

    def set_model_inference_at(self, dt: datetime) -> None:
        self._last_model_inference_at = dt

    def set_feature_computed_at(self, dt: datetime) -> None:
        self._last_feature_computed_at = dt

    def set_system_status(self, status: SystemStatus) -> None:
        self._system_status = status

    def set_trading_mode(self, mode: TradingMode) -> None:
        self._trading_mode = mode

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    async def _postgres_ping(self) -> tuple[bool, float | None, str | None]:
        if not self._postgres_connect_kwargs:
            return True, None, None

        start = time.monotonic()
        try:
            conn = await asyncio.wait_for(
                asyncpg.connect(**self._postgres_connect_kwargs, statement_cache_size=0),
                timeout=_DB_TIMEOUT_S,
            )
            try:
                await conn.fetchval("SELECT 1")
            finally:
                await conn.close()
            latency = (time.monotonic() - start) * 1000
            return True, latency, None
        except Exception as exc:
            return False, None, _scrub_dsn_credentials(str(exc))

    async def check_postgres(self) -> tuple[bool, float | None]:
        """Ping PostgreSQL.

        Returns:
            (is_healthy, latency_ms)
        """
        ok, latency, error = await self._postgres_ping()
        if not ok:
            log.warning("postgres_health_check_failed", error=error)
        return ok, latency

    async def check_postgres_with_retries(
        self,
        *,
        max_attempts: int | None = None,
    ) -> tuple[bool, float | None]:
        """Ping PostgreSQL with startup retries for transient cloud DB blips."""
        attempts = max(1, int(max_attempts or self._postgres_retry_attempts))
        last_error: str | None = None
        for attempt in range(1, attempts + 1):
            ok, latency, error = await self._postgres_ping()
            if ok:
                if attempt > 1:
                    log.info("postgres_health_check_recovered", attempt=attempt)
                return True, latency
            last_error = error
            if attempt < attempts:
                wait_s = _postgres_retry_wait_seconds(
                    attempt=attempt,
                    base_delay_seconds=self._postgres_retry_delay_s,
                    error_text=error,
                )
                log.warning(
                    "postgres_health_check_retrying",
                    attempt=attempt,
                    max_attempts=attempts,
                    wait_seconds=round(wait_s, 2),
                    error=error,
                )
                await asyncio.sleep(wait_s)
        log.warning("postgres_health_check_failed", error=last_error)
        return False, None

    async def check_redis(self) -> tuple[bool, float | None]:
        """Ping Redis.

        Returns:
            (is_healthy, latency_ms)

        When REDIS_URL is empty and Redis is optional (``redis_required=False``),
        the check is silently skipped (returns healthy/None) so no warning fires.
        When the URL is empty but Redis is required, returns unhealthy immediately.
        """
        if not self._redis_url:
            if self._redis_required:
                log.warning("redis_health_check_failed", error="REDIS_URL is not configured but REDIS_REQUIRED=true")
                return False, None
            # Optional and not configured → silently skip
            return True, None

        start = time.monotonic()
        try:
            client = aioredis.from_url(self._redis_url, socket_connect_timeout=_REDIS_TIMEOUT_S)
            try:
                await asyncio.wait_for(client.ping(), timeout=_REDIS_TIMEOUT_S)
            finally:
                await client.close()
            latency = (time.monotonic() - start) * 1000
            return True, latency
        except Exception as exc:
            log.warning("redis_health_check_failed", error=str(exc))
            return False, None

    async def check_bybit_connectivity(self) -> tuple[bool, float | None]:
        """Check Bybit REST API reachability via /v5/market/time.

        Returns:
            (is_healthy, latency_ms)
        """
        import aiohttp  # lazy import

        start = time.monotonic()
        url = f"{self._bybit_rest_url}/v5/market/time"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=_BYBIT_TIMEOUT_S)) as resp:
                    if resp.status == 200:
                        latency = (time.monotonic() - start) * 1000
                        return True, latency
                    log.warning("bybit_rest_unhealthy", status=resp.status)
                    return False, None
        except Exception as exc:
            log.warning("bybit_rest_check_failed", error=str(exc))
            return False, None

    async def check_websocket_status(self) -> bool:
        """Return whether the WebSocket connection is considered live.

        A connection is live if:
        - The internal flag is True, AND
        - A message was received within the last 30 seconds.
        """
        if not self._ws_connected:
            return False
        if self._last_ws_message_at is None:
            return False
        age = (datetime.now(tz=UTC) - self._last_ws_message_at).total_seconds()
        return age < 30.0

    async def check_model_freshness(self) -> bool:
        """Return True if a model inference happened within the threshold."""
        if self._last_model_inference_at is None:
            return False
        age = (datetime.now(tz=UTC) - self._last_model_inference_at).total_seconds()
        return age < _MODEL_STALE_THRESHOLD_S

    async def check_feature_freshness(self) -> bool:
        """Return True if features were computed within the threshold."""
        if self._last_feature_computed_at is None:
            return False
        age = (datetime.now(tz=UTC) - self._last_feature_computed_at).total_seconds()
        return age < _FEATURE_STALE_THRESHOLD_S

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    async def overall_health(self) -> HealthStatus:
        """Run all checks concurrently and return an aggregate HealthStatus.

        This method is designed to be called from a FastAPI route handler.
        Results are cached for a short TTL so that repeated unauthenticated
        calls (e.g. /readyz hit in a tight loop) cannot force a fresh
        Postgres/Redis/Bybit round-trip on every request.
        """
        async with self._health_cache_lock:
            now = time.monotonic()
            if self._health_cache is not None and (now - self._health_cache_at) < _OVERALL_HEALTH_CACHE_TTL_S:
                return self._health_cache
            result = await self._compute_overall_health()
            self._health_cache = result
            self._health_cache_at = time.monotonic()
            return result

    async def _compute_overall_health(self) -> HealthStatus:
        (
            (pg_ok, pg_lat),
            (redis_ok, redis_lat),
            (bybit_ok, bybit_lat),
            ws_ok,
            model_ok,
            feat_ok,
        ) = await asyncio.gather(
            self.check_postgres(),
            self.check_redis(),
            self.check_bybit_connectivity(),
            self.check_websocket_status(),
            self.check_model_freshness(),
            self.check_feature_freshness(),
        )

        messages: list[str] = []
        if not pg_ok:
            messages.append("PostgreSQL is unreachable")
        if not redis_ok and self._redis_required:
            messages.append("Redis is unreachable")
        # Only add optional-Redis warning when a URL was configured but is down;
        # an empty optional URL is intentionally unconfigured — no warning needed.
        if not redis_ok and not self._redis_required and self._redis_url:
            messages.append("Redis unavailable (optional)")
        if not bybit_ok and self._bybit_required:
            messages.append("Bybit REST API is unreachable")
        if not bybit_ok and not self._bybit_required:
            messages.append("Bybit REST unavailable (geo-restricted or optional)")
        if not ws_ok:
            messages.append("WebSocket feed is stale or disconnected")
        if self._model_enabled and not model_ok:
            messages.append("Model has not produced inferences recently")
        if not feat_ok:
            messages.append("Feature pipeline has not produced features recently")

        # Determine overall status
        # Non-required components (redis, bybit_rest) and model (when disabled)
        # are excluded from the critical path.
        redis_critical_ok = redis_ok or not self._redis_required
        bybit_critical_ok = bybit_ok or not self._bybit_required
        model_critical_ok = model_ok if self._model_enabled else True
        critical_ok = pg_ok and redis_critical_ok and bybit_critical_ok
        if critical_ok and ws_ok and feat_ok and model_critical_ok:
            overall = "healthy"
        elif critical_ok:
            overall = "degraded"
        else:
            overall = "unhealthy"

        return HealthStatus(
            overall=overall,
            postgres=pg_ok,
            redis=redis_ok,
            bybit_rest=bybit_ok,
            bybit_ws=ws_ok,
            model_fresh=model_ok,
            features_fresh=feat_ok,
            system_status=self._system_status,
            trading_mode=self._trading_mode,
            postgres_latency_ms=pg_lat,
            redis_latency_ms=redis_lat,
            bybit_rest_latency_ms=bybit_lat,
            messages=messages,
        )

    async def run_preflight(self) -> dict[str, Any]:
        """Run all checks and return a dict of check_name → bool.

        Suitable for the startup preflight sequence.
        """
        if self._postgres_required:
            pg_ok, _ = await self.check_postgres_with_retries()
        else:
            pg_ok, _ = await self.check_postgres_with_retries(
                max_attempts=self._postgres_optional_max_attempts,
            )
            if not pg_ok:
                log.warning(
                    "preflight_postgres_optional_failed",
                    trading_mode=self._trading_mode.value,
                    hint="service will start; trade journal reconnects in background",
                )
        redis_ok, _ = await self.check_redis()
        bybit_ok, _ = await self.check_bybit_connectivity()

        checks = {
            "postgres": pg_ok,
            "redis": redis_ok,
            "bybit_connectivity": bybit_ok,
        }
        postgres_blocks = self._postgres_required and not pg_ok
        passed = (
            not postgres_blocks and (redis_ok or not self._redis_required) and (bybit_ok or not self._bybit_required)
        )
        return {
            "passed": passed,
            "checks": checks,
            "postgres_required": self._postgres_required,
        }
