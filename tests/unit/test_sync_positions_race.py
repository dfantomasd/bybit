"""Tests for the sync_positions vs submit race fix.

The registry snapshot merge must never erase a freshly opened position that
the exchange snapshot does not list yet (REST lag / partial fill window), and
the maker-flow variant must not deadlock on the already-held submit lock.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.domain.enums import OrderSide
from trader.execution.engine import _SYNC_REMOVAL_GRACE_SECONDS, ExecutionEngine


def _make_engine(positions: list | None = None) -> ExecutionEngine:
    adapter = MagicMock()
    adapter.get_positions = AsyncMock(return_value=positions or [])
    risk_manager = MagicMock()
    exposure = MagicMock()
    exposure.update_position = AsyncMock()
    exposure.remove_position = AsyncMock()
    return ExecutionEngine(
        adapter=adapter,
        risk_manager=risk_manager,
        exposure_tracker=exposure,
        shadow_mode=True,
        cooldown_s=0,
    )


def _exchange_position(symbol: str = "BTCUSDT") -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        side=OrderSide.BUY,
        size=Decimal("0.01"),
        entry_price=Decimal("50000"),
    )


class TestSyncPositionsRace:
    @pytest.mark.asyncio
    async def test_recent_entry_survives_stale_snapshot(self) -> None:
        """A position opened moments ago must not be wiped by an empty snapshot."""
        engine = _make_engine(positions=[])
        engine._open_positions["BTCUSDT"] = {"side": OrderSide.BUY, "size": Decimal("0.01")}
        engine._last_entry_at["BTCUSDT"] = datetime.now(tz=UTC)

        await engine.sync_positions()

        assert engine.has_open_position("BTCUSDT")
        engine._exposure.remove_position.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stale_position_is_removed(self) -> None:
        """A position past the grace period and absent on exchange is closed."""
        engine = _make_engine(positions=[])
        engine._open_positions["BTCUSDT"] = {"side": OrderSide.BUY, "size": Decimal("0.01")}
        engine._last_entry_at["BTCUSDT"] = datetime.now(tz=UTC) - timedelta(seconds=_SYNC_REMOVAL_GRACE_SECONDS + 1)

        await engine.sync_positions()

        assert not engine.has_open_position("BTCUSDT")
        engine._exposure.remove_position.assert_awaited_once_with("BTCUSDT")

    @pytest.mark.asyncio
    async def test_position_without_entry_timestamp_is_removed(self) -> None:
        """Registry entries with no recorded entry time (restart) sync normally."""
        engine = _make_engine(positions=[])
        engine._open_positions["ETHUSDT"] = {"side": OrderSide.BUY, "size": Decimal("0.1")}

        await engine.sync_positions()

        assert not engine.has_open_position("ETHUSDT")

    @pytest.mark.asyncio
    async def test_snapshot_merges_exchange_positions(self) -> None:
        engine = _make_engine(positions=[_exchange_position("BTCUSDT")])
        engine._open_positions["XRPUSDT"] = {"side": OrderSide.BUY, "size": Decimal("10")}
        engine._last_entry_at["XRPUSDT"] = datetime.now(tz=UTC)  # recent → kept

        await engine.sync_positions()

        assert engine.has_open_position("BTCUSDT")
        assert engine.has_open_position("XRPUSDT")
        engine._exposure.update_position.assert_awaited()

    @pytest.mark.asyncio
    async def test_sync_serialised_with_submit_lock(self) -> None:
        """Public sync_positions waits for the submit lock holder."""
        engine = _make_engine(positions=[])
        order: list[str] = []

        async def hold_lock() -> None:
            async with engine._submit_lock:
                order.append("submit_start")
                await asyncio.sleep(0.05)
                order.append("submit_end")

        async def syncer() -> None:
            await asyncio.sleep(0.01)  # let hold_lock acquire first
            await engine.sync_positions()
            order.append("sync_done")

        await asyncio.gather(hold_lock(), syncer())
        assert order == ["submit_start", "submit_end", "sync_done"]

    @pytest.mark.asyncio
    async def test_locked_variant_does_not_deadlock_under_held_lock(self) -> None:
        """Maker flow calls the locked variant while holding the lock."""
        engine = _make_engine(positions=[])
        async with engine._submit_lock:
            result = await asyncio.wait_for(engine._sync_positions_locked(), timeout=1.0)
        assert result == []
