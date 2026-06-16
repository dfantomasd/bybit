from __future__ import annotations

import asyncio
import json

import pytest

from trader.domain.events import LiquidationEvent
from trader.exchange.bybit_ws_public import BybitPublicWebSocket


@pytest.mark.asyncio
async def test_all_liquidation_topic_emits_liquidation_event() -> None:
    queue: asyncio.Queue = asyncio.Queue()
    ws = BybitPublicWebSocket(
        endpoint="wss://example.invalid",
        subscriptions=[],
        event_queue=queue,
    )

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
    assert event.symbol == "BTCUSDT"
    assert event.price > 0
    assert event.qty > 0
