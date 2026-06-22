from __future__ import annotations

import asyncio
import json
import time

import pytest

from trader.domain.events import LiquidationEvent
from trader.exchange.bybit_ws_public import BybitPublicWebSocket


@pytest.mark.asyncio
async def test_pong_does_not_refresh_market_watchdog_timestamp() -> None:
    queue: asyncio.Queue = asyncio.Queue()
    ws = BybitPublicWebSocket(
        endpoint="wss://example.invalid",
        subscriptions=[],
        event_queue=queue,
    )
    ws._last_market_message_ts = time.monotonic() - 5.0

    await ws._handle_message(json.dumps({"op": "pong", "success": True}))

    assert ws.last_market_message_age_s is not None
    assert ws.last_market_message_age_s >= 4.0


@pytest.mark.asyncio
async def test_market_message_refreshes_watchdog_timestamp() -> None:
    queue: asyncio.Queue = asyncio.Queue()
    ws = BybitPublicWebSocket(
        endpoint="wss://example.invalid",
        subscriptions=[],
        event_queue=queue,
    )
    ws._last_market_message_ts = time.monotonic() - 5.0

    await ws._handle_message(
        json.dumps(
            {
                "topic": "allLiquidation.BTCUSDT",
                "type": "snapshot",
                "data": [{"s": "BTCUSDT", "S": "Sell", "p": "65000", "v": "0.2"}],
            }
        )
    )

    event = await queue.get()
    assert isinstance(event, LiquidationEvent)
    assert ws.last_market_message_age_s is not None
    assert ws.last_market_message_age_s < 1.0
