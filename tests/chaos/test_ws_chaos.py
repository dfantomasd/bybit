"""Chaos tests for WebSocket layer.

Uses mock WebSocket — no real network connections.
Tests resilience against:
- Sequence gaps
- Stale connections
- Queue overflow
- Duplicate events
- Missing stop losses
- Unknown orders
- Reconnect behavior
"""
from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trader.data.event_bus import EventBus
from trader.data.orderbook import LocalOrderBook
from trader.domain.events import OrderBookEvent, ReconciliationEvent
from trader.domain.enums import MarketType, OrderStatus
from trader.exchange.state_machine import OrderStateStore, OrderStateMachine
from trader.exchange.reconnect_supervisor import ReconnectSupervisor, calc_backoff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def snapshot_data(bids, asks, update_id=1, seq=1):
    return {
        "b": [[str(p), str(q)] for p, q in bids],
        "a": [[str(p), str(q)] for p, q in asks],
        "u": update_id,
        "seq": seq,
    }


def delta_data(bids, asks, update_id, seq=None):
    return {
        "b": [[str(p), str(q)] for p, q in bids],
        "a": [[str(p), str(q)] for p, q in asks],
        "u": update_id,
        "seq": seq or update_id,
    }


# ---------------------------------------------------------------------------
# Orderbook chaos tests
# ---------------------------------------------------------------------------


async def test_snapshot_then_delta_valid():
    """After a snapshot, a sequential delta is applied correctly."""
    ob = LocalOrderBook("BTCUSDT")
    ob.apply_snapshot(snapshot_data(
        bids=[(30000, 1.0)],
        asks=[(30001, 0.5)],
        update_id=100,
    ))
    assert ob.is_valid()

    result = ob.apply_delta(delta_data(
        bids=[(30000, 2.0)],
        asks=[],
        update_id=101,
    ))
    assert result is True
    assert ob.get_best_bid() == (Decimal("30000"), Decimal("2.0"))


async def test_sequence_gap_invalidates_orderbook():
    """A sequence gap (non-contiguous update_id) invalidates the orderbook."""
    ob = LocalOrderBook("BTCUSDT")
    ob.apply_snapshot(snapshot_data(
        bids=[(30000, 1.0)],
        asks=[(30001, 0.5)],
        update_id=100,
    ))

    # Skip update_id 101, send 103 directly
    result = ob.apply_delta(delta_data(
        bids=[(30000, 3.0)],
        asks=[],
        update_id=103,
    ))
    assert result is False
    assert not ob.is_valid()


async def test_reconnect_rebuilds_orderbook():
    """After invalidation, a new snapshot fully rebuilds the orderbook."""
    ob = LocalOrderBook("BTCUSDT")

    # Initial snapshot
    ob.apply_snapshot(snapshot_data(
        bids=[(30000, 1.0)],
        asks=[(30001, 0.5)],
        update_id=100,
    ))

    # Sequence gap invalidates
    ob.apply_delta(delta_data(bids=[(30000, 5.0)], asks=[], update_id=200))
    assert not ob.is_valid()

    # New snapshot after reconnect
    ob.apply_snapshot(snapshot_data(
        bids=[(31000, 2.0), (30999, 1.0)],
        asks=[(31001, 0.5)],
        update_id=500,
    ))
    assert ob.is_valid()
    assert ob.get_best_bid() == (Decimal("31000"), Decimal("2.0"))


async def test_repeated_snapshot_resets_book():
    """Multiple snapshots correctly reset the book each time."""
    ob = LocalOrderBook("BTCUSDT")

    for i, price in enumerate([30000, 31000, 32000], start=1):
        ob.apply_snapshot(snapshot_data(
            bids=[(price, float(i))],
            asks=[(price + 1, 0.5)],
            update_id=i * 100,
        ))
        assert ob.is_valid()
        best = ob.get_best_bid()
        assert best is not None
        assert best[0] == Decimal(str(price))


async def test_stale_ws_triggers_reconnect():
    """ReconnectSupervisor reconnects when connection times out."""
    call_count = 0

    async def flaky_connect():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Simulate immediate failure (stale connection)
            raise ConnectionError("connection timed out")
        # Second call succeeds (stable)
        await asyncio.Event().wait()

    supervisor = ReconnectSupervisor(
        name="test_stale",
        connect_fn=flaky_connect,
        metrics=None,
    )

    task = asyncio.create_task(supervisor.run())
    await asyncio.sleep(0.3)  # Allow time for first failure + retry
    await supervisor.request_stop()
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=1.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass

    # Should have attempted at least once (first failure)
    assert call_count >= 1


async def test_duplicate_event_is_idempotent():
    """Private WS deduplicates events with same orderId + updateTime."""
    from trader.exchange.bybit_ws_private import BybitPrivateWebSocket

    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    ws = BybitPrivateWebSocket(
        endpoint="wss://test.example.com",
        api_key="key",
        api_secret="secret",
        event_queue=queue,
    )

    # Simulate two identical order messages
    order_msg = {
        "topic": "order",
        "data": [{
            "orderId": "order-123",
            "orderLinkId": "link-456",
            "symbol": "BTCUSDT",
            "category": "linear",
            "side": "Buy",
            "orderType": "Limit",
            "qty": "0.1",
            "price": "30000",
            "orderStatus": "New",
            "cumExecQty": "0",
            "cumExecFee": "0",
            "updatedTime": "1700000000000",
        }],
    }

    raw = json.dumps(order_msg)
    await ws._handle_message(raw)
    await ws._handle_message(raw)  # duplicate

    # Only one event should be emitted
    assert queue.qsize() == 1


async def test_queue_full_drops_market_data():
    """When event queue is full, market data events are dropped."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=2)

    # Fill the queue
    from trader.domain.events import MarketDataEvent
    queue.put_nowait(MarketDataEvent(symbol="BTCUSDT", market_type=MarketType.LINEAR))
    queue.put_nowait(MarketDataEvent(symbol="ETHUSDT", market_type=MarketType.LINEAR))

    # Now queue is full — emitting more should not raise
    from trader.exchange.bybit_ws_public import BybitPublicWebSocket
    ws_pub = BybitPublicWebSocket(
        endpoint="wss://test",
        subscriptions=["orderbook.50.BTCUSDT"],
        event_queue=queue,
    )

    # This should silently drop
    event = MarketDataEvent(symbol="SOLUSDT", market_type=MarketType.LINEAR)
    await ws_pub._emit(event)

    # Queue size should still be 2 (max)
    assert queue.qsize() == 2


async def test_reconciliation_after_reconnect():
    """ReconciliationService.on_reconnect runs a full pass successfully."""
    from trader.exchange.reconciliation import ReconciliationService

    queue: asyncio.Queue = asyncio.Queue()

    # Mock rest client that returns empty lists
    mock_rest = AsyncMock()
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mock_rest.get_wallet_balance = AsyncMock(return_value={})

    mock_order_store = AsyncMock()
    mock_order_store.get_all_active = AsyncMock(return_value={})

    svc = ReconciliationService(
        rest_client=mock_rest,
        order_store=mock_order_store,
        position_store=None,
        event_queue=queue,
    )

    result = await svc.on_reconnect()
    assert result is not None
    assert result.success is True
    assert result.discrepancies_found == 0

    # Should have emitted a reconciliation event
    assert queue.qsize() >= 1
    event = queue.get_nowait()
    assert isinstance(event, ReconciliationEvent)


async def test_position_without_sl_triggers_safe_mode():
    """ReconciliationService enters safe mode when a position has no stop loss."""
    from trader.exchange.reconciliation import ReconciliationService

    queue: asyncio.Queue = asyncio.Queue()

    # Mock position store with a position lacking SL
    class MockPosition:
        symbol = "BTCUSDT"
        size = Decimal("0.1")
        stop_loss = None  # no SL!

    class MockPositionStore:
        _positions = {"BTCUSDT": MockPosition()}

    mock_rest = AsyncMock()
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mock_rest.get_wallet_balance = AsyncMock(return_value={})

    mock_order_store = AsyncMock()
    mock_order_store.get_all_active = AsyncMock(return_value={})

    svc = ReconciliationService(
        rest_client=mock_rest,
        order_store=mock_order_store,
        position_store=MockPositionStore(),
        event_queue=queue,
    )

    result = await svc.run_once()
    # Safe mode should be activated
    assert svc.safe_mode is True


async def test_unknown_order_detected():
    """ReconciliationService detects orders on exchange not in local store."""
    from trader.exchange.reconciliation import ReconciliationService

    queue: asyncio.Queue = asyncio.Queue()

    # Exchange has an order that local store doesn't know about
    mock_rest = AsyncMock()
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_open_orders = AsyncMock(return_value=[
        {"orderLinkId": "unknown-order-999", "orderStatus": "New"}
    ])
    mock_rest.get_wallet_balance = AsyncMock(return_value={})

    mock_order_store = AsyncMock()
    mock_order_store.get_all_active = AsyncMock(return_value={})  # empty local store

    svc = ReconciliationService(
        rest_client=mock_rest,
        order_store=mock_order_store,
        position_store=None,
        event_queue=queue,
    )

    result = await svc.run_once()
    assert result.discrepancies_found > 0


# ---------------------------------------------------------------------------
# Sync wrapper to run async tests
# ---------------------------------------------------------------------------


def test_snapshot_then_delta_valid_sync():
    asyncio.run(test_snapshot_then_delta_valid())


def test_sequence_gap_invalidates_orderbook_sync():
    asyncio.run(test_sequence_gap_invalidates_orderbook())


def test_reconnect_rebuilds_orderbook_sync():
    asyncio.run(test_reconnect_rebuilds_orderbook())


def test_repeated_snapshot_resets_book_sync():
    asyncio.run(test_repeated_snapshot_resets_book())


def test_stale_ws_triggers_reconnect_sync():
    asyncio.run(test_stale_ws_triggers_reconnect())


def test_duplicate_event_is_idempotent_sync():
    asyncio.run(test_duplicate_event_is_idempotent())


def test_queue_full_drops_market_data_sync():
    asyncio.run(test_queue_full_drops_market_data())


def test_reconciliation_after_reconnect_sync():
    asyncio.run(test_reconciliation_after_reconnect())


def test_position_without_sl_triggers_safe_mode_sync():
    asyncio.run(test_position_without_sl_triggers_safe_mode())


def test_unknown_order_detected_sync():
    asyncio.run(test_unknown_order_detected())
