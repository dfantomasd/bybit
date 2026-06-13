"""Tests for monitoring health checks."""

from __future__ import annotations

from trader.monitoring.health import HealthChecker


class TestHealthCheckerOptionalServices:
    async def test_optional_empty_redis_is_healthy(self) -> None:
        checker = HealthChecker(
            postgres_dsn="postgresql://u:p@localhost:5432/db",
            redis_url="",
            redis_required=False,
        )
        ok, latency = await checker.check_redis()
        assert ok is True
        assert latency is None

    async def test_required_empty_redis_is_unhealthy(self) -> None:
        checker = HealthChecker(
            postgres_dsn="postgresql://u:p@localhost:5432/db",
            redis_url="",
            redis_required=True,
        )
        ok, latency = await checker.check_redis()
        assert ok is False
        assert latency is None

    async def test_empty_postgres_dsn_is_unhealthy_even_when_redis_empty(self) -> None:
        checker = HealthChecker(
            postgres_dsn="",
            redis_url="",
            redis_required=False,
        )
        ok, latency = await checker.check_postgres()
        assert ok is False
        assert latency is None
