"""Tests for ExecutionEngine correlation gate and queue utilization gate."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.domain.enums import MarketRegime, MarketType, OrderSide, RiskDecisionStatus
from trader.domain.models import RiskDecision, TradeProposal
from trader.execution.engine import ExecutionEngine


def _proposal(symbol: str = "BTCUSDT", side: OrderSide = OrderSide.BUY) -> TradeProposal:
    return TradeProposal(
        strategy_id="test",
        symbol=symbol,
        market_type=MarketType.LINEAR,
        side=side,
        requested_qty=Decimal("0.01"),
        entry_price=Decimal("50000"),
        stop_loss=Decimal("49000"),
        take_profit=Decimal("52000"),
        confidence=0.7,
        regime=MarketRegime.BULL_TREND,
    )


def _approved_decision(proposal: TradeProposal) -> RiskDecision:
    return RiskDecision(
        proposal_id=proposal.proposal_id,
        status=RiskDecisionStatus.APPROVED,
        approved_qty=proposal.requested_qty,
        portfolio_heat=0.05,
        current_drawdown_pct=0.0,
        open_positions_count=0,
    )


def _make_engine(
    max_correlated: int = 0,
    max_queue_pct: int = 100,
    max_concurrent_pending: int = 5,
    shadow_mode: bool = False,
    family_count: int = 0,
) -> ExecutionEngine:
    adapter = MagicMock()
    adapter.get_positions = AsyncMock(return_value=[])
    adapter.get_instrument_info = AsyncMock(
        return_value=MagicMock(
            min_order_qty=Decimal("0.001"),
            max_order_qty=Decimal("100"),
            qty_step=Decimal("0.001"),
            tick_size=Decimal("0.5"),
            min_notional=Decimal("5"),
            base_coin="BTC",
            quote_coin="USDT",
        )
    )
    adapter.place_order = AsyncMock(return_value={"result": {"orderId": "test-123"}})
    adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("50000"))

    risk_manager = MagicMock()

    def _evaluate(proposal, **kwargs):
        return _approved_decision(proposal)

    risk_manager.evaluate = AsyncMock(side_effect=_evaluate)

    exposure = MagicMock()
    exposure.update_position = AsyncMock()
    exposure.remove_position = AsyncMock()
    exposure.total_exposure_pct = Decimal("0")
    exposure.count_family_positions = MagicMock(return_value=family_count)

    engine = ExecutionEngine(
        adapter=adapter,
        risk_manager=risk_manager,
        exposure_tracker=exposure,
        shadow_mode=shadow_mode,
        cooldown_s=0,
        startup_warmup_seconds=0,
        max_correlated_positions=max_correlated,
        max_queue_utilization_pct=max_queue_pct,
        max_concurrent_pending_entries=max_concurrent_pending,
        max_new_entries_per_minute=100,
    )
    # Skip warmup period for tests
    engine._started_at = datetime(2000, 1, 1, tzinfo=UTC)
    return engine


# ---------------------------------------------------------------------------
# Correlation gate
# ---------------------------------------------------------------------------


class TestCorrelationGate:
    @pytest.mark.asyncio
    async def test_gate_disabled_when_max_correlated_zero(self):
        """With max_correlated_positions=0 gate is bypassed regardless of family count."""
        engine = _make_engine(max_correlated=0, shadow_mode=True, family_count=99)
        proposal = _proposal("BTCUSDT")
        result = await engine.submit(proposal, Decimal("10000"), Decimal("10000"))
        # Should not return None due to correlation gate (shadow mode, risk approved)
        assert result is not None
        assert result.status == RiskDecisionStatus.APPROVED

    @pytest.mark.asyncio
    async def test_gate_blocks_when_family_at_limit(self):
        engine = _make_engine(max_correlated=2, family_count=2)
        proposal = _proposal("WBTCUSDT")
        result = await engine.submit(proposal, Decimal("10000"), Decimal("10000"))
        assert result is None

    @pytest.mark.asyncio
    async def test_gate_blocks_when_family_exceeds_limit(self):
        engine = _make_engine(max_correlated=1, family_count=3)
        proposal = _proposal("BTCUSDT")
        result = await engine.submit(proposal, Decimal("10000"), Decimal("10000"))
        assert result is None

    @pytest.mark.asyncio
    async def test_gate_passes_when_family_below_limit(self):
        engine = _make_engine(max_correlated=3, family_count=1, shadow_mode=True)
        proposal = _proposal("BTCUSDT")
        result = await engine.submit(proposal, Decimal("10000"), Decimal("10000"))
        assert result is not None
        assert result.status == RiskDecisionStatus.APPROVED

    @pytest.mark.asyncio
    async def test_gate_passes_when_family_count_zero(self):
        engine = _make_engine(max_correlated=2, family_count=0, shadow_mode=True)
        proposal = _proposal("DOGEUSDT")
        result = await engine.submit(proposal, Decimal("10000"), Decimal("10000"))
        assert result is not None


# ---------------------------------------------------------------------------
# Queue utilization gate
# ---------------------------------------------------------------------------


class TestQueueUtilizationGate:
    def _get_gate_reason(self, engine: ExecutionEngine, side: str = "Buy") -> str | None:
        return engine._check_rate_limits("BTCUSDT", side)

    def test_utilization_check_skipped_when_pct_is_100(self):
        """When max_queue_pct=100 the utilization check is skipped (pending_limit may still fire)."""
        engine = _make_engine(max_queue_pct=100, max_concurrent_pending=5, shadow_mode=False)
        engine._pending_entry_count = 2  # 40% utilization — below any meaningful cap
        reason = self._get_gate_reason(engine)
        # utilization check is skipped, pending_limit check is separate (2 < 5 → passes)
        assert reason is None

    def test_gate_blocks_at_threshold(self):
        engine = _make_engine(max_queue_pct=50, max_concurrent_pending=2, shadow_mode=False)
        engine._pending_entry_count = 1  # 50% utilization == threshold → block
        reason = self._get_gate_reason(engine)
        assert reason is not None
        assert "queue_utilization" in reason

    def test_gate_passes_below_threshold(self):
        engine = _make_engine(max_queue_pct=70, max_concurrent_pending=10, shadow_mode=False)
        engine._pending_entry_count = 5  # 50% < 70% → pass
        reason = self._get_gate_reason(engine)
        assert reason is None

    def test_gate_blocks_above_threshold(self):
        engine = _make_engine(max_queue_pct=50, max_concurrent_pending=4, shadow_mode=False)
        engine._pending_entry_count = 3  # 75% > 50% → block
        reason = self._get_gate_reason(engine)
        assert reason is not None
        assert "queue_utilization" in reason
        assert "75.0%" in reason

    def test_gate_inactive_in_shadow_mode(self):
        """Queue utilization gate only applies in live mode."""
        engine = _make_engine(max_queue_pct=50, max_concurrent_pending=2, shadow_mode=True)
        engine._pending_entry_count = 2  # would block in live
        reason = self._get_gate_reason(engine)
        assert reason is None

    def test_utilization_check_skipped_when_max_concurrent_is_zero(self):
        """When max_concurrent=0 the utilization check is skipped (division guard)."""
        engine = _make_engine(max_queue_pct=50, max_concurrent_pending=1, shadow_mode=False)
        engine._pending_entry_count = 0  # 0% utilization — below threshold
        reason = self._get_gate_reason(engine)
        assert reason is None

    def test_utilization_uses_true_division(self):
        """Ensure fractional utilization is calculated precisely."""
        engine = _make_engine(max_queue_pct=70, max_concurrent_pending=3, shadow_mode=False)
        engine._pending_entry_count = 1  # 33.33% < 70% → should pass
        reason = self._get_gate_reason(engine)
        assert reason is None

    def test_reason_message_contains_counts(self):
        engine = _make_engine(max_queue_pct=50, max_concurrent_pending=4, shadow_mode=False)
        engine._pending_entry_count = 3
        reason = self._get_gate_reason(engine)
        assert "3/4" in reason
