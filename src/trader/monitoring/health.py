"""Health check service.

Provides individual component health checks and an aggregate ``overall_health``
method that returns a ``HealthStatus`` domain model.

Designed to be instantiated once and called periodically or on HTTP /health
requests.
"""
from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

import asyncpg
import redis.asyncio as aioredis

from trader.domain.enums import SystemStatus, TradingMode
from trader.domain.models import HealthStatus
from trader.monitoring.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_DB_TIMEOUT_S = 3.0
_REDIS_TIMEOUT_S = 2.0
_BYBIT_TIMEOUT_S = 5.0
_MODEL_STALE_THRESHOLD_S = 3600.0  # 1 hour
_FEATURE_STALE_THRESHOLD_S = 10.0  # 10 seconds


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
    ) -> None:
        self._postgres_dsn = postgres_dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
        self._redis_url = redis_url.strip().strip("\"'")
        self._redis_required = redis_required
        self._bybit_required = bybit_required
        self._bybit_rest_url = bybit_rest_url
        self._trading_mode = trading_mode
        self._system_status = system_status

        # Mutable state updated by the trading system
        self._ws_connected: bool = False
        self._last_ws_message_at: datetime | None = None
        self._last_model_inference_at: datetime | None = None
        self._last_feature_computed_at: datetime | None = None

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

    async def check_postgres(self) -> tuple[bool, float | None]:
        """Ping PostgreSQL.

        Returns:
            (is_healthy, latency_ms)
        """
        if not self._postgres_dsn:
            return True, None

        start = time.monotonic()
        try:
            conn = await asyncio.wait_for(
                asyncpg.connect(self._postgres_dsn, statement_cache_size=0),
                timeout=_DB_TIMEOUT_S,
            )
            try:
                await conn.fetchval("SELECT 1")
            finally:
                await conn.close()
            latency = (time.monotonic() - start) * 1000
            return True, latency
        except Exception as exc:
            log.warning("postgres_health_check_failed", error=str(exc))
            return False, None

    async def check_redis(self) -> tuple[bool, float | None]:
        """Ping Redis.

        Returns:
            (is_healthy, latency_ms)
        """
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
        """
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
        if not bybit_ok and self._bybit_required:
            messages.append("Bybit REST API is unreachable")
        if not ws_ok:
            messages.append("WebSocket feed is stale or disconnected")
        if not model_ok:
            messages.append("Model has not produced inferences recently")
        if not feat_ok:
            messages.append("Feature pipeline has not produced features recently")

        # Determine overall status
        redis_critical_ok = redis_ok or not self._redis_required
        bybit_critical_ok = bybit_ok or not self._bybit_required
        critical_ok = pg_ok and redis_critical_ok and bybit_critical_ok
        if critical_ok and ws_ok and model_ok and feat_ok:
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
        pg_ok, _ = await self.check_postgres()
        redis_ok, _ = await self.check_redis()
        bybit_ok, _ = await self.check_bybit_connectivity()

        checks = {
            "postgres": pg_ok,
            "redis": redis_ok,
            "bybit_connectivity": bybit_ok,
        }
        passed = (
            pg_ok
            and (redis_ok or not self._redis_required)
            and (bybit_ok or not self._bybit_required)
        )
        return {"passed": passed, "checks": checks}
