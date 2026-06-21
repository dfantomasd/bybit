"""Tests for HealthChecker Postgres startup retry behaviour."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from trader.monitoring.health import HealthChecker


class TestHealthCheckerPostgresRetries:
    @pytest.mark.asyncio
    async def test_postgres_preflight_recovers_on_second_attempt(self) -> None:
        checker = HealthChecker(
            postgres_dsn="postgresql://user:pw@localhost/db",
            redis_url="",
            postgres_retry_attempts=3,
            postgres_retry_delay_s=0.0,
        )
        ping = AsyncMock(side_effect=[(False, None, "connection was closed"), (True, 12.5, None)])

        with patch.object(checker, "_postgres_ping", ping):
            ok, latency = await checker.check_postgres_with_retries()

        assert ok is True
        assert latency == 12.5
        assert ping.await_count == 2

    @pytest.mark.asyncio
    async def test_postgres_preflight_fails_after_all_retries(self) -> None:
        checker = HealthChecker(
            postgres_dsn="postgresql://user:pw@localhost/db",
            redis_url="",
            postgres_retry_attempts=2,
            postgres_retry_delay_s=0.0,
        )
        ping = AsyncMock(return_value=(False, None, "connection was closed"))

        with patch.object(checker, "_postgres_ping", ping):
            ok, latency = await checker.check_postgres_with_retries()

        assert ok is False
        assert latency is None
        assert ping.await_count == 2

    @pytest.mark.asyncio
    async def test_run_preflight_uses_postgres_retries(self) -> None:
        checker = HealthChecker(
            postgres_dsn="postgresql://user:pw@localhost/db",
            redis_url="",
            postgres_retry_attempts=2,
            postgres_retry_delay_s=0.0,
        )
        checker.check_postgres_with_retries = AsyncMock(return_value=(True, 1.0))  # type: ignore[method-assign]
        checker.check_redis = AsyncMock(return_value=(True, None))  # type: ignore[method-assign]
        checker.check_bybit_connectivity = AsyncMock(return_value=(True, None))  # type: ignore[method-assign]

        result = await checker.run_preflight()

        checker.check_postgres_with_retries.assert_awaited_once()
        assert result["passed"] is True
        assert result["checks"]["postgres"] is True
