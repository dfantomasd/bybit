"""Tests for SafetyModeLadder, queue-aware escalation, and latency logging."""

from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.domain.enums import MarketType, OrderSide, OrderType
from trader.domain.models import OrderIntent
from trader.risk.drawdown import DrawdownTracker
from trader.risk.profiles import get_risk_limits
from trader.risk.safety_ladder import SafetyLevel, SafetyModeLadder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _async_update(tracker: DrawdownTracker, value: Decimal) -> None:
    await tracker.update(value)


def _make_tracker(peak: str = "1000", current: str = "1000") -> DrawdownTracker:
    tracker = DrawdownTracker(Decimal(peak))
    asyncio.run(_async_update(tracker, Decimal(current)))
    return tracker


def _make_ladder(peak: str = "1000", current: str = "1000", max_hold_s: float = 3600.0) -> SafetyModeLadder:
    limits = get_risk_limits_scalp()
    tracker = _make_tracker(peak, current)
    return SafetyModeLadder(tracker, limits, max_hold_s=max_hold_s)


def get_risk_limits_scalp():
    from trader.domain.enums import RiskProfile

    return get_risk_limits(RiskProfile.SCALP)


def _make_order_intent(symbol: str = "BTCUSDT", side: str = "Buy") -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        market_type=MarketType.LINEAR,
        side=OrderSide.BUY if side == "Buy" else OrderSide.SELL,
        qty=Decimal("1"),
        price=Decimal("100"),
        order_type=OrderType.LIMIT,
        order_link_id="test-link-id",
        proposal_id=uuid.uuid4(),
        decision_id=uuid.uuid4(),
    )


# ---------------------------------------------------------------------------
# SafetyModeLadder tests
# ---------------------------------------------------------------------------


class TestSafetyModeLadder:
    def test_normal_level_no_drawdown(self):
        ladder = _make_ladder("1000", "1000")
        assert ladder.current_level() == SafetyLevel.NORMAL
        assert ladder.size_multiplier() == 1.0
        assert not ladder.blocks_new_entries()

    def test_pingpong_on_soft_warning(self):
        """SCALP: max_drawdown=8%, hard_stop=12%. At 9% → PINGPONG (above soft)."""
        limits = get_risk_limits_scalp()
        tracker = DrawdownTracker(Decimal("1000"))
        # 9% drawdown: current = 910
        asyncio.run(_async_update(tracker, Decimal("910")))
        ladder = SafetyModeLadder(tracker, limits)
        level = ladder.current_level()
        assert level == SafetyLevel.PINGPONG
        assert ladder.size_multiplier() == 0.75

    def test_boomerang_at_midpoint(self):
        """At 50%+ of distance between soft (8%) and hard (12%) → BOOMERANG.
        Threshold: 8 + 0.5*(12-8) = 10%.
        """
        limits = get_risk_limits_scalp()
        tracker = DrawdownTracker(Decimal("1000"))
        # 10.5% drawdown → current = 895
        asyncio.run(_async_update(tracker, Decimal("895")))
        ladder = SafetyModeLadder(tracker, limits)
        assert ladder.current_level() == SafetyLevel.BOOMERANG
        assert ladder.size_multiplier() == 0.50

    def test_ak47_near_hard_stop(self):
        """At 80%+ of distance to hard_stop → AK47 (blocks new entries)."""
        limits = get_risk_limits_scalp()
        tracker = DrawdownTracker(Decimal("1000"))
        # 87.5% of (12-8) above 8% → 8 + 0.875*4 = 11.5% → current = 885
        asyncio.run(_async_update(tracker, Decimal("885")))
        ladder = SafetyModeLadder(tracker, limits)
        assert ladder.current_level() == SafetyLevel.AK47
        assert ladder.size_multiplier() == 0.0
        assert ladder.blocks_new_entries()

    def test_describe_returns_serialisable_dict(self):
        ladder = _make_ladder()
        d = ladder.describe()
        assert "level" in d
        assert "size_multiplier" in d
        assert "blocks_new_entries" in d
        assert isinstance(d["level"], str)

    def test_position_not_stale_when_fresh(self):
        import time

        ladder = _make_ladder()
        now = time.monotonic()
        assert not ladder.position_is_stale(now)

    def test_position_stale_when_old(self):
        import time

        ladder = _make_ladder(max_hold_s=1.0)
        # Simulate a position opened 10 seconds ago
        old_ts = time.monotonic() - 10
        assert ladder.position_is_stale(old_ts)


# ---------------------------------------------------------------------------
# Queue-aware escalation tests
# ---------------------------------------------------------------------------


class TestQueueAwareEscalation:
    def _make_engine_mock(self) -> MagicMock:
        from trader.execution.engine import ExecutionEngine

        engine = MagicMock(spec=ExecutionEngine)
        engine._maker_allow_escalation = True
        engine._maker_ttl_s = 5.0
        engine._imbalance_provider = None
        engine._category = "linear"
        engine._maker_escalation_allowed = ExecutionEngine._maker_escalation_allowed.__get__(engine)
        return engine

    @pytest.mark.asyncio
    async def test_escalation_allowed_when_no_imbalance_and_price_stable(self):
        engine = self._make_engine_mock()
        engine._adapter = MagicMock()
        engine._adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("100"))
        intent = _make_order_intent()
        allowed, reason = await engine._maker_escalation_allowed(intent, Decimal("100"), time_waited_s=0)
        assert allowed

    @pytest.mark.asyncio
    async def test_escalation_blocked_by_price_drift(self):
        engine = self._make_engine_mock()
        engine._adapter = MagicMock()
        # Current price drifted 5% from maker price
        engine._adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("105"))
        intent = _make_order_intent()
        allowed, reason = await engine._maker_escalation_allowed(intent, Decimal("100"), time_waited_s=0)
        assert not allowed
        assert "price_drifted" in reason

    @pytest.mark.asyncio
    async def test_adverse_imbalance_blocks_early_escalation(self):
        engine = self._make_engine_mock()
        engine._imbalance_provider = MagicMock(return_value=-0.7)  # adverse for BUY
        engine._adapter = MagicMock()
        engine._adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("100"))
        intent = _make_order_intent(side="Buy")
        # time_waited_s = 0 → early, queue override not active
        allowed, reason = await engine._maker_escalation_allowed(intent, Decimal("100"), time_waited_s=0)
        assert not allowed
        assert "imbalance_against" in reason

    @pytest.mark.asyncio
    async def test_queue_depth_override_bypasses_adverse_imbalance(self):
        """When 75%+ of maker window and >= 2s elapsed, escalate despite adverse imbalance."""
        engine = self._make_engine_mock()
        engine._imbalance_provider = MagicMock(return_value=-0.7)  # adverse for BUY
        engine._adapter = MagicMock()
        engine._adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("100"))
        intent = _make_order_intent(side="Buy")
        # time_waited_s = 4.0 out of ttl=5.0 → 80% and >= 2s → override kicks in
        allowed, reason = await engine._maker_escalation_allowed(intent, Decimal("100"), time_waited_s=4.0)
        assert allowed
        assert reason == "queue_depth_override"

    @pytest.mark.asyncio
    async def test_queue_depth_override_not_triggered_below_min_wait(self):
        """Short wait (< 2s) should NOT trigger queue override even at high fraction."""
        engine = self._make_engine_mock()
        engine._imbalance_provider = MagicMock(return_value=-0.7)  # adverse for BUY
        engine._adapter = MagicMock()
        engine._adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("100"))
        intent = _make_order_intent(side="Buy")
        # 1.9s wait (< 2s minimum) → override should NOT fire
        allowed, reason = await engine._maker_escalation_allowed(intent, Decimal("100"), time_waited_s=1.9)
        assert not allowed
        assert "imbalance_against" in reason

    @pytest.mark.asyncio
    async def test_queue_depth_override_still_respects_price_drift(self):
        """Even with queue override, price drift check still applies."""
        engine = self._make_engine_mock()
        engine._imbalance_provider = MagicMock(return_value=-0.7)
        engine._adapter = MagicMock()
        # Price drifted 5% — still blocks regardless of queue override
        engine._adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("105"))
        intent = _make_order_intent(side="Buy")
        allowed, reason = await engine._maker_escalation_allowed(intent, Decimal("100"), time_waited_s=4.0)
        assert not allowed
        assert "price_drifted" in reason

    @pytest.mark.asyncio
    async def test_escalation_disabled_config_overrides_all(self):
        engine = self._make_engine_mock()
        engine._maker_allow_escalation = False
        intent = _make_order_intent()
        allowed, reason = await engine._maker_escalation_allowed(intent, Decimal("100"), time_waited_s=4.0)
        assert not allowed
        assert reason == "escalation_disabled"


# ---------------------------------------------------------------------------
# Safety ladder integration in ExecutionEngine
# ---------------------------------------------------------------------------


class TestSafetyLadderIntegration:
    def _make_ladder_at_drawdown(self, drawdown_pct: float) -> SafetyModeLadder:
        limits = get_risk_limits_scalp()
        peak = Decimal("1000")
        current = peak * Decimal(str(1.0 - drawdown_pct / 100.0))
        tracker = DrawdownTracker(peak)
        asyncio.run(_async_update(tracker, current))
        return SafetyModeLadder(tracker, limits)

    def test_ak47_ladder_sets_multiplier_to_zero(self):
        ladder = self._make_ladder_at_drawdown(11.5)
        assert ladder.current_level() == SafetyLevel.AK47
        assert ladder.size_multiplier() == 0.0

    def test_boomerang_ladder_halves_size(self):
        ladder = self._make_ladder_at_drawdown(10.5)
        assert ladder.current_level() == SafetyLevel.BOOMERANG
        assert ladder.size_multiplier() == 0.50

    def test_normal_ladder_no_restriction(self):
        ladder = self._make_ladder_at_drawdown(0.0)
        assert ladder.current_level() == SafetyLevel.NORMAL
        assert not ladder.blocks_new_entries()
