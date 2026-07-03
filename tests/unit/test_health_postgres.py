"""Tests for HealthChecker Postgres startup retry behaviour."""

from __future__ import annotations

import ssl
from unittest.mock import AsyncMock, patch

import pytest

from trader.domain.enums import TradingMode
from trader.monitoring.health import HealthChecker, _postgres_retry_wait_seconds


class TestPostgresRetryBackoff:
    def test_circuit_breaker_uses_longer_backoff(self) -> None:
        wait = _postgres_retry_wait_seconds(
            attempt=2,
            base_delay_seconds=2.0,
            error_text="(ECIRCUITBREAKER) failed to retrieve database credentials",
        )
        assert wait == 30.0

    def test_transient_error_uses_linear_backoff(self) -> None:
        wait = _postgres_retry_wait_seconds(
            attempt=2,
            base_delay_seconds=2.0,
            error_text="connection was closed in the middle of operation",
        )
        assert wait == 4.0


class TestHealthCheckerPostgresRetries:
    @pytest.mark.asyncio
    async def test_postgres_ping_uses_normalized_asyncpg_kwargs_for_sslmode_require(self) -> None:
        checker = HealthChecker(
            postgres_dsn=(
                "postgresql+asyncpg://postgres.projectref:secret@"
                "aws-0-eu-west-1.pooler.supabase.com:6543/postgres?sslmode=require"
            ),
            redis_url="",
        )
        fake_conn = AsyncMock()

        with patch("trader.monitoring.health.asyncpg.connect", AsyncMock(return_value=fake_conn)) as connect:
            ok, latency, error = await checker._postgres_ping()

        assert ok is True
        assert latency is not None
        assert error is None
        connect.assert_awaited_once()
        kwargs = connect.await_args.kwargs
        # Credentials are passed via separate user/password kwargs, not
        # embedded in the dsn string, so a logged/traced dsn never leaks the
        # password (asyncpg may log connection kwargs on error).
        assert kwargs["dsn"] == "postgresql://aws-0-eu-west-1.pooler.supabase.com:6543/postgres"
        assert kwargs["user"] == "postgres.projectref"
        assert kwargs["password"] == "secret"
        assert isinstance(kwargs["ssl"], ssl.SSLContext)
        assert kwargs["ssl"].verify_mode == ssl.CERT_NONE
        assert kwargs["statement_cache_size"] == 0
        fake_conn.fetchval.assert_awaited_once_with("SELECT 1")
        fake_conn.close.assert_awaited_once()

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

    @pytest.mark.asyncio
    async def test_run_preflight_allows_optional_postgres_failure(self) -> None:
        checker = HealthChecker(
            postgres_dsn="postgresql://user:pw@localhost/db",
            redis_url="",
            postgres_required=False,
            postgres_optional_max_attempts=2,
            postgres_retry_delay_s=0.0,
            trading_mode=TradingMode.SHADOW,
        )
        checker.check_postgres_with_retries = AsyncMock(return_value=(False, None))  # type: ignore[method-assign]
        checker.check_redis = AsyncMock(return_value=(True, None))  # type: ignore[method-assign]
        checker.check_bybit_connectivity = AsyncMock(return_value=(True, None))  # type: ignore[method-assign]

        result = await checker.run_preflight()

        checker.check_postgres_with_retries.assert_awaited_once_with(max_attempts=2)
        assert result["passed"] is True
        assert result["checks"]["postgres"] is False
        assert result["postgres_required"] is False

    @pytest.mark.asyncio
    async def test_run_preflight_blocks_when_postgres_required(self) -> None:
        checker = HealthChecker(
            postgres_dsn="postgresql://user:pw@localhost/db",
            redis_url="",
            postgres_required=True,
            postgres_retry_delay_s=0.0,
        )
        checker.check_postgres_with_retries = AsyncMock(return_value=(False, None))  # type: ignore[method-assign]
        checker.check_redis = AsyncMock(return_value=(True, None))  # type: ignore[method-assign]
        checker.check_bybit_connectivity = AsyncMock(return_value=(True, None))  # type: ignore[method-assign]

        result = await checker.run_preflight()

        assert result["passed"] is False
        assert result["checks"]["postgres"] is False
        assert result["postgres_required"] is True
