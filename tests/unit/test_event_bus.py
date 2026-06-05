"""Tests for the EventBus."""
from __future__ import annotations

import asyncio

import pytest

from trader.data.event_bus import EventBus
from trader.domain.events import BaseEvent, MarketDataEvent, SystemEvent
from trader.domain.enums import MarketType, SystemStatus, TradingMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_event(topic: str = "test") -> BaseEvent:
    return BaseEvent(topic=topic)


def make_market_event(symbol: str = "BTCUSDT") -> MarketDataEvent:
    return MarketDataEvent(
        symbol=symbol,
        market_type=MarketType.LINEAR,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_publish_and_consume():
    """Publishing an event and consuming it via get() works correctly."""
    async def _run():
        bus = EventBus(maxsize=100)
        event = make_event("market_data")
        published = await bus.publish("market_data", event)
        assert published is True

        sizes = bus.get_queue_sizes()
        assert sizes["market_data"] == 1

        received = await bus.get("market_data", timeout=1.0)
        assert received is not None
        assert received.event_id == event.event_id

    asyncio.run(_run())


async def test_bounded_queue_drops_when_full():
    """When queue is full, non-critical events are dropped and counter increments."""
    bus = EventBus(maxsize=2)
    # Fill the queue
    e1 = make_event()
    e2 = make_event()
    e3 = make_event()

    r1 = await bus.publish("market_data", e1)
    r2 = await bus.publish("market_data", e2)
    r3 = await bus.publish("market_data", e3)  # should be dropped

    assert r1 is True
    assert r2 is True
    assert r3 is False  # dropped

    dropped = bus.get_dropped_count()
    assert dropped["market_data"] == 1


async def test_critical_event_goes_to_dead_letter():
    """A critical event dropped from full queue goes to dead letter queue."""
    bus = EventBus(maxsize=1)
    e1 = make_event()
    e2 = make_event()

    await bus.publish("system", e1)  # fills the queue
    result = await bus.publish("system", e2, critical=True)  # goes to DLQ

    assert result is False
    assert bus.get_dead_letter_size() == 1

    dlq_events = await bus.drain_dead_letter()
    assert len(dlq_events) == 1
    assert dlq_events[0].event_id == e2.event_id


async def test_drain_empties_queues():
    """drain() waits for queues to be processed."""
    bus = EventBus(maxsize=10)
    event = make_event()
    await bus.publish("execution", event)

    # Consume it manually
    await bus.get("execution", timeout=1.0)

    # drain should complete quickly since queues are empty
    await bus.drain(timeout=2.0)
    sizes = bus.get_queue_sizes()
    assert sizes["execution"] == 0


async def test_queue_sizes():
    """get_queue_sizes returns correct sizes for all queues."""
    bus = EventBus(maxsize=100)
    await bus.publish("market_data", make_event())
    await bus.publish("market_data", make_event())
    await bus.publish("execution", make_event())

    sizes = bus.get_queue_sizes()
    assert sizes["market_data"] == 2
    assert sizes["execution"] == 1
    assert sizes["risk"] == 0
    assert sizes["system"] == 0
    assert sizes["persistence"] == 0


async def test_dropped_counter():
    """Dropped counter accumulates across multiple drops."""
    bus = EventBus(maxsize=1)
    await bus.publish("risk", make_event())  # fills queue

    # Drop 3 more
    for _ in range(3):
        await bus.publish("risk", make_event())

    dropped = bus.get_dropped_count()
    assert dropped["risk"] == 3


def test_unknown_queue_returns_false():
    """Publishing to unknown queue returns False."""
    async def _run():
        bus = EventBus()
        result = await bus.publish("nonexistent_queue", make_event())
        assert result is False
    asyncio.run(_run())


def test_queue_names_constant():
    """QUEUE_NAMES contains expected queues."""
    assert "market_data" in EventBus.QUEUE_NAMES
    assert "execution" in EventBus.QUEUE_NAMES
    assert "risk" in EventBus.QUEUE_NAMES
    assert "persistence" in EventBus.QUEUE_NAMES
    assert "system" in EventBus.QUEUE_NAMES


def test_publish_sync_wrapper():
    """Test sync wrapper for running async publish tests."""
    asyncio.run(test_bounded_queue_drops_when_full())
    asyncio.run(test_critical_event_goes_to_dead_letter())
    asyncio.run(test_drain_empties_queues())
    asyncio.run(test_queue_sizes())
    asyncio.run(test_dropped_counter())
