"""Tests for BybitAdapter helper methods."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from trader.exchange.bybit_adapter import BybitAdapter


@pytest.mark.asyncio
async def test_set_trading_stop_sends_trailing_stop_params() -> None:
    adapter = object.__new__(BybitAdapter)
    rest = AsyncMock()
    rest.set_trading_stop.return_value = {"retCode": 0, "result": {}}
    adapter._rest = rest

    await adapter.set_trading_stop(
        category="linear",
        symbol="BTCUSDT",
        stop_loss="50100",
        trailing_stop="150",
        active_price="50250",
        position_idx=0,
    )

    rest.set_trading_stop.assert_awaited_once_with(
        category="linear",
        symbol="BTCUSDT",
        positionIdx=0,
        tpslMode="Full",
        stopLoss="50100",
        trailingStop="150",
        activePrice="50250",
    )
