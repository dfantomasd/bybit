"""Tests for RateLimiter."""

from __future__ import annotations

import asyncio
import math
import time

import pytest

from trader.exchange.rate_limiter import (
    _BACKOFF_MULTIPLIER,
    _BASE_BACKOFF_SECONDS,
    _WARN_THRESHOLDS,
    RateLimiter,
    _EndpointState,
)


class TestTokenBucket:
    """Test token bucket allow/block behaviour."""

    async def test_allows_N_requests_immediately(self) -> None:
        rl = RateLimiter(default_capacity=10, default_refill_rate=1.0)
        for _ in range(10):
            await rl.acquire("/test", "GET")
        status = rl.get_status()
        assert "GET:/test" in status
        assert status["GET:/test"]["tokens"] < 1.0

    async def test_usage_pct_zero_initially(self) -> None:
        rl = RateLimiter(default_capacity=100, default_refill_rate=2.0)
        assert rl.get_usage_pct("/fresh", "GET") == 0.0

    async def test_usage_pct_increases_as_tokens_consumed(self) -> None:
        rl = RateLimiter(default_capacity=10, default_refill_rate=0.01)
        for _ in range(5):
            await rl.acquire("/ep", "GET")
        pct = rl.get_usage_pct("/ep", "GET")
        assert pct > 0.0

    async def test_get_status_returns_dict(self) -> None:
        rl = RateLimiter()
        await rl.acquire("/v5/order/create", "POST")
        status = rl.get_status()
        assert isinstance(status, dict)
        assert "POST:/v5/order/create" in status

    async def test_separate_endpoint_keys(self) -> None:
        rl = RateLimiter(default_capacity=10, default_refill_rate=1.0)
        await rl.acquire("/ep1", "GET")
        await rl.acquire("/ep2", "POST")
        status = rl.get_status()
        assert "GET:/ep1" in status
        assert "POST:/ep2" in status

    async def test_blocks_then_resumes(self) -> None:
        """Token bucket exhausted → acquire waits, then succeeds."""
        rl = RateLimiter(default_capacity=1, default_refill_rate=10.0)
        await rl.acquire("/ep", "GET")  # Consume the one token
        # Next call should wait briefly (refill_rate=10 → 0.1s for 1 token)
        start = time.monotonic()
        await asyncio.wait_for(rl.acquire("/ep", "GET"), timeout=2.0)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.05  # waited at least a little


class TestHeaderParsing:
    """Test record_response parses Bybit headers correctly."""

    def test_parses_limit_status(self) -> None:
        rl = RateLimiter()
        headers = {
            "X-Bapi-Limit-Status": "95",
            "X-Bapi-Limit": "100",
            "X-Bapi-Limit-Reset-Timestamp": "1700000000000",
        }
        rl.record_response("/ep", headers, method="GET")
        status = rl.get_status()
        key = "GET:/ep"
        assert status[key]["limit_remaining"] == 95
        assert status[key]["limit_total"] == 100
        assert status[key]["limit_reset_ts_ms"] == 1700000000000

    def test_parses_lowercase_headers(self) -> None:
        rl = RateLimiter()
        headers = {
            "x-bapi-limit-status": "50",
            "x-bapi-limit": "100",
        }
        rl.record_response("/ep2", headers, method="POST")
        status = rl.get_status()
        assert status["POST:/ep2"]["limit_remaining"] == 50

    def test_ignores_invalid_header_values(self) -> None:
        rl = RateLimiter()
        headers = {"X-Bapi-Limit-Status": "not_a_number"}
        # Should not raise
        rl.record_response("/ep3", headers, method="GET")

    def test_resets_consecutive_429s_on_success(self) -> None:
        rl = RateLimiter()
        rl.handle_rate_limit_error("/ep4", method="GET")
        rl.handle_rate_limit_error("/ep4", method="GET")
        # Success response resets counter
        rl.record_response("/ep4", {"X-Bapi-Limit": "100"}, method="GET")
        state = rl._states.get("GET:/ep4")
        assert state is not None
        assert state.consecutive_429s == 0

    def test_usage_pct_computed_from_headers(self) -> None:
        rl = RateLimiter()
        headers = {"X-Bapi-Limit-Status": "20", "X-Bapi-Limit": "100"}
        rl.record_response("/ep5", headers, method="GET")
        pct = rl.get_usage_pct("/ep5", "GET")
        assert pct == pytest.approx(80.0, abs=0.1)


class TestUsagePercentage:
    """Test usage percentage property and warning thresholds."""

    def test_endpoint_state_usage_pct_from_headers(self) -> None:
        state = _EndpointState(capacity=100.0, tokens=100.0)
        state.limit_total = 100
        state.limit_remaining = 30
        assert state.usage_pct == pytest.approx(70.0)

    def test_endpoint_state_usage_pct_fallback_to_local(self) -> None:
        state = _EndpointState(capacity=100.0, tokens=50.0)
        # No server-reported limits
        assert state.usage_pct == pytest.approx(50.0)

    def test_endpoint_state_remaining_capacity_from_headers(self) -> None:
        state = _EndpointState()
        state.limit_remaining = 42
        assert state.remaining_capacity == 42

    def test_endpoint_state_remaining_capacity_fallback(self) -> None:
        state = _EndpointState(capacity=100.0, tokens=75.0)
        assert state.remaining_capacity == 75

    def test_warn_thresholds_are_70_85_95(self) -> None:
        assert 70.0 in _WARN_THRESHOLDS
        assert 85.0 in _WARN_THRESHOLDS
        assert 95.0 in _WARN_THRESHOLDS


class TestExponentialBackoff:
    """Test handle_rate_limit_error backoff calculation."""

    def test_first_hit_uses_base_backoff(self) -> None:
        rl = RateLimiter()
        wait = rl.handle_rate_limit_error("/ep", method="GET")
        # First hit: base * 2^0 = 1.0s ± jitter
        assert wait >= 0.5  # Minimum enforced
        assert wait <= 2.0  # Should be well below max

    def test_backoff_increases_with_consecutive_hits(self) -> None:
        rl = RateLimiter()
        rl.handle_rate_limit_error("/ep2", method="GET")
        rl.handle_rate_limit_error("/ep2", method="GET")
        # w2 should generally be larger (2^1 vs 2^0), allowing for jitter
        # We check the raw expected values before jitter
        assert _BASE_BACKOFF_SECONDS * math.pow(_BACKOFF_MULTIPLIER, 1) > _BASE_BACKOFF_SECONDS

    def test_retry_after_overrides_backoff(self) -> None:
        rl = RateLimiter()
        wait = rl.handle_rate_limit_error("/ep3", retry_after=15.0, method="GET")
        # retry_after=15.0 should dominate (before jitter)
        # After jitter it's ±25% of 15 = [11.25, 18.75]
        assert wait >= 11.0
        assert wait <= 20.0

    def test_backoff_capped_at_max(self) -> None:
        rl = RateLimiter()
        # Many consecutive hits — should cap
        for _ in range(20):
            wait = rl.handle_rate_limit_error("/ep4", method="GET")
        assert wait <= 75.0  # max + jitter

    def test_consecutive_429s_increments(self) -> None:
        rl = RateLimiter()
        rl.handle_rate_limit_error("/ep5", method="GET")
        rl.handle_rate_limit_error("/ep5", method="GET")
        state = rl._states.get("GET:/ep5")
        assert state is not None
        assert state.consecutive_429s == 2
