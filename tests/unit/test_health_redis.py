"""Tests for HealthChecker Redis behaviour with empty / optional URL."""

from __future__ import annotations

import pytest

from trader.monitoring.health import HealthChecker


class TestHealthCheckerRedis:
    @pytest.mark.asyncio
    async def test_empty_optional_redis_url_is_healthy(self):
        """Empty Redis URL + REDIS_REQUIRED=False → healthy, no exception."""
        checker = HealthChecker(
            postgres_dsn="postgresql://user:pw@localhost/db",
            redis_url="",
            redis_required=False,
        )
        ok, latency = await checker.check_redis()
        assert ok is True
        assert latency is None

    @pytest.mark.asyncio
    async def test_empty_required_redis_url_is_unhealthy(self):
        """Empty Redis URL + REDIS_REQUIRED=True → unhealthy."""
        checker = HealthChecker(
            postgres_dsn="postgresql://user:pw@localhost/db",
            redis_url="",
            redis_required=True,
        )
        ok, latency = await checker.check_redis()
        assert ok is False

    @pytest.mark.asyncio
    async def test_empty_optional_redis_no_warning_message(self):
        """overall_health() must not add a Redis warning when URL is empty & optional."""
        checker = HealthChecker(
            postgres_dsn="postgresql://user:pw@localhost/db",
            redis_url="",
            redis_required=False,
        )
        # We can't call overall_health without a live Postgres, so just verify
        # check_redis returns True (healthy) so overall_health won't add a message.
        ok, _ = await checker.check_redis()
        assert ok is True, "optional empty Redis should be silently skipped (healthy)"

    @pytest.mark.asyncio
    async def test_whitespace_only_redis_url_treated_as_empty(self):
        """A URL that is only whitespace should be treated the same as empty."""
        checker = HealthChecker(
            postgres_dsn="postgresql://user:pw@localhost/db",
            redis_url="   ",
            redis_required=False,
        )
        # After stripping, _redis_url should be ""
        assert checker._redis_url == ""
        ok, _ = await checker.check_redis()
        assert ok is True
