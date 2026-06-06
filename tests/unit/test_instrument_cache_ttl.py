"""Tests for P1.3: Instrument info cache with TTL."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.domain.enums import MarketType, RiskDecisionStatus
from trader.domain.models import InstrumentInfo, RiskDecision
from trader.execution.engine import _INSTRUMENT_CACHE_TTL_S, ExecutionEngine


def _instrument() -> InstrumentInfo:
    return InstrumentInfo(
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        base_coin="BTC",
        quote_coin="USDT",
        min_order_qty=Decimal("0.001"),
        max_order_qty=Decimal("100"),
        qty_step=Decimal("0.001"),
        tick_size=Decimal("0.1"),
        min_notional=Decimal("5"),
    )


def _make_engine() -> ExecutionEngine:
    adapter = MagicMock()
    adapter.get_positions = AsyncMock(return_value=[])
    adapter.get_instrument_info = AsyncMock(return_value=_instrument())
    adapter._rest = MagicMock()
    adapter._rest.set_leverage = AsyncMock()

    risk_manager = MagicMock()
    risk_manager._limits = MagicMock()
    risk_manager._limits.max_leverage = Decimal("5")
    risk_manager.evaluate = AsyncMock(
        return_value=RiskDecision(
            proposal_id=uuid.uuid4(),
            status=RiskDecisionStatus.APPROVED,
            approved_qty=Decimal("0.001"),
            portfolio_heat=0.0,
            current_drawdown_pct=0.0,
            open_positions_count=0,
        )
    )

    exposure = MagicMock()
    exposure.update_position = AsyncMock()
    exposure.remove_position = AsyncMock()

    return ExecutionEngine(
        adapter=adapter,
        risk_manager=risk_manager,
        exposure_tracker=exposure,
        shadow_mode=True,
    )


class TestInstrumentCacheTTL:
    @pytest.mark.asyncio
    async def test_cache_miss_fetches_from_adapter(self):
        """First call fetches from adapter and caches."""
        engine = _make_engine()
        info = await engine.get_instrument_info("BTCUSDT")
        engine._adapter.get_instrument_info.assert_awaited_once()
        assert info is not None

    @pytest.mark.asyncio
    async def test_cache_hit_uses_cached_value(self):
        """Second call within TTL uses cached value, no new fetch."""
        engine = _make_engine()
        await engine.get_instrument_info("BTCUSDT")
        await engine.get_instrument_info("BTCUSDT")
        engine._adapter.get_instrument_info.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cache_miss_after_ttl_expires(self):
        """After TTL expires, info is re-fetched."""
        engine = _make_engine()
        # Populate cache with an expired entry
        expired_at = datetime.now(tz=UTC) - timedelta(seconds=_INSTRUMENT_CACHE_TTL_S + 10)
        engine._instrument_cache["BTCUSDT"] = (_instrument(), expired_at)

        await engine.get_instrument_info("BTCUSDT")
        engine._adapter.get_instrument_info.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cache_fresh_within_ttl(self):
        """Entry within TTL is not re-fetched."""
        engine = _make_engine()
        fresh_at = datetime.now(tz=UTC) - timedelta(seconds=_INSTRUMENT_CACHE_TTL_S - 60)
        engine._instrument_cache["BTCUSDT"] = (_instrument(), fresh_at)

        await engine.get_instrument_info("BTCUSDT")
        engine._adapter.get_instrument_info.assert_not_awaited()

    def test_ttl_constant_is_one_hour(self):
        """_INSTRUMENT_CACHE_TTL_S is exactly 3600 seconds."""
        assert _INSTRUMENT_CACHE_TTL_S == 3600
