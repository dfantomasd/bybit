"""Tests: P0.8 – private WebSocket must emit ExecutionUpdateEvent for fills."""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal

import pytest

from trader.domain.enums import OrderSide, OrderStatus
from trader.domain.events import ExecutionUpdateEvent, OrderUpdateEvent, PositionUpdateEvent
from trader.exchange.bybit_ws_private import _MAX_SEEN_EVENTS, BybitPrivateWebSocket


def _make_ws(queue: asyncio.Queue) -> BybitPrivateWebSocket:
    return BybitPrivateWebSocket(
        endpoint="wss://fake",
        api_key="key",
        api_secret="secret",
        event_queue=queue,
    )


@pytest.mark.asyncio
async def test_execution_event_emitted_on_fill():
    """P0.8: execution topic must emit ExecutionUpdateEvent (not just log)."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    ws = _make_ws(queue)

    execution_data = [
        {
            "execId": "exec123",
            "orderId": "order456",
            "orderLinkId": "link789",
            "symbol": "BTCUSDT",
            "category": "linear",
            "side": "Buy",
            "orderType": "Market",
            "execPrice": "50000",
            "execQty": "0.001",
            "execFee": "0.05",
            "isMaker": False,
            "closedSize": "0",
        }
    ]

    await ws._handle_execution(execution_data)

    assert not queue.empty(), "ExecutionUpdateEvent should have been emitted"
    event = queue.get_nowait()
    assert isinstance(event, ExecutionUpdateEvent), f"Expected ExecutionUpdateEvent, got {type(event)}"
    assert event.exec_id == "exec123"
    assert event.symbol == "BTCUSDT"
    assert event.exec_price == Decimal("50000")
    assert event.exec_qty == Decimal("0.001")
    assert event.side == OrderSide.BUY


@pytest.mark.asyncio
async def test_execution_deduplication():
    """Same exec_id + order_id must not emit twice."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    ws = _make_ws(queue)

    data = [{"execId": "e1", "orderId": "o1", "symbol": "X", "execPrice": "100", "execQty": "1"}]
    await ws._handle_execution(data)
    await ws._handle_execution(data)  # duplicate

    assert queue.qsize() == 1, "Duplicate execution must be deduplicated"


@pytest.mark.asyncio
async def test_private_ws_seen_event_cache_is_bounded():
    """Dedup cache must not grow forever on a long-running private stream."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    ws = _make_ws(queue)

    for idx in range(_MAX_SEEN_EVENTS + 25):
        duplicate = ws._mark_seen(f"event-{idx}")
        assert duplicate is False

    assert len(ws._seen_events) == _MAX_SEEN_EVENTS
    assert "event-0" not in ws._seen_events
    assert ws._mark_seen(f"event-{_MAX_SEEN_EVENTS + 24}") is True


def test_private_ws_auth_expiry_has_network_latency_budget():
    """Auth expiry should leave more than a tiny 1s margin for Render/network jitter."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    ws = _make_ws(queue)

    expires_ms = int(ws._build_auth_msg()["args"][1])
    margin_s = expires_ms / 1000 - time.time()

    assert margin_s >= 8.0


@pytest.mark.asyncio
async def test_order_update_event_emitted():
    """Order topic must emit OrderUpdateEvent."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    ws = _make_ws(queue)

    order_data = [
        {
            "orderId": "ord001",
            "orderLinkId": "lnk001",
            "symbol": "ETHUSDT",
            "category": "linear",
            "side": "Sell",
            "orderType": "Limit",
            "orderStatus": "Filled",
            "qty": "1",
            "cumExecQty": "1",
            "avgPrice": "3000",
            "price": "3000",
            "cumExecFee": "0.3",
            "updatedTime": "123456",
        }
    ]

    await ws._handle_order(order_data)

    assert not queue.empty()
    event = queue.get_nowait()
    assert isinstance(event, OrderUpdateEvent)
    assert event.status == OrderStatus.FILLED
    assert event.symbol == "ETHUSDT"


@pytest.mark.asyncio
async def test_position_update_event_emitted():
    """Position topic must emit PositionUpdateEvent."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    ws = _make_ws(queue)

    position_data = [
        {
            "symbol": "BTCUSDT",
            "category": "linear",
            "side": "Buy",
            "size": "0.05",
            "entryPrice": "50000",
            "markPrice": "50100",
            "unrealisedPnl": "5",
            "cumRealisedPnl": "0",
            "leverage": "10",
        }
    ]

    await ws._handle_position(position_data)

    assert not queue.empty()
    event = queue.get_nowait()
    assert isinstance(event, PositionUpdateEvent)
    assert event.symbol == "BTCUSDT"
    assert event.size == Decimal("0.05")
    assert event.entry_price == Decimal("50000")
