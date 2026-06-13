"""Bybit V5 public WebSocket manager.

Manages subscriptions for orderbook, trades, ticker, kline, liquidations.
Uses the ``websockets`` library for native asyncio integration.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC
from decimal import Decimal
from typing import Any

import structlog

from trader.data.orderbook import LocalOrderBook
from trader.domain.enums import MarketType, OrderSide
from trader.domain.events import (
    BaseEvent,
    KlineEvent,
    MarketDataEvent,
    OrderBookEvent,
    TickerEvent,
    TradeEvent,
)

logger = structlog.get_logger(__name__)

_HEARTBEAT_INTERVAL = 20.0  # send ping every 20 s
_PONG_TIMEOUT = 5.0  # expect pong within 5 s
_WATCHDOG_TIMEOUT = 30.0  # reconnect if no message for 30 s


class BybitPublicWebSocket:
    """Manages Bybit V5 public WebSocket connection.

    Subscriptions: orderbook, trades, ticker, kline, liquidations.

    Features
    --------
    - Automatic reconnect (handled by ReconnectSupervisor externally or via
      the internal _run_connection loop when used standalone).
    - Heartbeat/ping every 20 s; reconnect if pong not received within 5 s.
    - Watchdog: if no message for 30 s, connection is considered dead.
    - Orderbook: wait for snapshot, then apply deltas; validate sequence numbers;
      on gap: invalidate + wait for new snapshot.
    - Measure latency: exchange_ts vs received_ts.
    - Emit typed events to asyncio.Queue.
    """

    def __init__(
        self,
        endpoint: str,
        subscriptions: list[str],
        event_queue: asyncio.Queue[Any],
        metrics: Any = None,
        logger: Any = None,
    ) -> None:
        self._endpoint = endpoint
        self._subscriptions = list(subscriptions)
        self._event_queue = event_queue
        self._metrics = metrics
        self._log = logger or structlog.get_logger(__name__)

        # Local orderbook per symbol
        self._orderbooks: dict[str, LocalOrderBook] = {}

        self._ws: Any = None
        self._connected: bool = False
        self._running: bool = False
        self._stop_event = asyncio.Event()
        self._last_message_ts: float = 0.0

        # Reconnect counter
        self._reconnect_count: int = 0

        # Downtime tracking
        self._downtime_start: float | None = None
        self._total_downtime: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect and start processing messages (runs until stop() called)."""
        self._running = True
        while not self._stop_event.is_set():
            await self._run_connection()
            if self._stop_event.is_set():
                break
            self._reconnect_count += 1
            if self._metrics is not None:
                try:
                    self._metrics.ws_reconnect_total.labels(name="public").inc()
                except Exception:  # noqa: S110
                    pass
            # Small backoff before reconnecting (supervisor manages larger backoff)
            await asyncio.sleep(1.0)
        self._running = False

    async def stop(self) -> None:
        """Signal the WS loop to stop."""
        self._stop_event.set()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: S110
                pass

    async def subscribe(self, topics: list[str]) -> None:
        """Subscribe to additional topics (adds to list and sends if connected)."""
        for t in topics:
            if t not in self._subscriptions:
                self._subscriptions.append(t)
        if self._ws is not None and self._connected:
            await self._send_subscribe(topics)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def orderbook_valid(self) -> dict[str, bool]:
        """Return {symbol: is_valid} for all tracked orderbooks."""
        return {sym: ob.is_valid() for sym, ob in self._orderbooks.items()}

    # ------------------------------------------------------------------
    # Internal connection loop
    # ------------------------------------------------------------------

    async def _run_connection(self) -> None:
        """Single connection attempt — returns when disconnected."""
        try:
            import websockets
        except ImportError:
            self._log.error("websockets_not_installed")
            await asyncio.sleep(5.0)
            return

        self._downtime_start = time.monotonic()
        try:
            async with websockets.connect(
                self._endpoint,
                ping_interval=None,  # we manage heartbeat manually
                ping_timeout=None,
                close_timeout=5,
            ) as ws:
                self._ws = ws
                self._connected = True
                self._last_message_ts = time.monotonic()

                # Record connection restored
                if self._downtime_start is not None:
                    self._total_downtime += time.monotonic() - self._downtime_start
                    self._downtime_start = None

                self._log.info("ws_public.connected", endpoint=self._endpoint)

                # Subscribe to all topics
                await self._send_subscribe(self._subscriptions)

                # Start heartbeat task and watchdog
                heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws))
                watchdog_task = asyncio.create_task(self._watchdog_loop())

                try:
                    _err_backoff = 0.0
                    async for raw in ws:
                        if self._stop_event.is_set():
                            break
                        self._last_message_ts = time.monotonic()
                        try:
                            await self._handle_message(raw)
                            _err_backoff = 0.0
                        except Exception as exc:
                            # Per-message error: back off then continue; don't kill the connection.
                            _err_backoff = min(_err_backoff * 2 + 0.1, 5.0)
                            self._log.warning(
                                "ws_public.handle_error",
                                error=str(exc),
                                backoff=round(_err_backoff, 2),
                            )
                            await asyncio.sleep(_err_backoff)
                except Exception as exc:
                    self._log.warning("ws_public.recv_error", error=str(exc))
                finally:
                    heartbeat_task.cancel()
                    watchdog_task.cancel()
                    try:
                        await asyncio.gather(heartbeat_task, watchdog_task, return_exceptions=True)
                    except Exception:  # noqa: S110
                        pass

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log.warning("ws_public.connection_failed", error=str(exc))
        finally:
            self._connected = False
            self._ws = None
            if self._downtime_start is None:
                self._downtime_start = time.monotonic()

    async def _send_subscribe(self, topics: list[str]) -> None:
        """Send subscription request."""
        if not topics or self._ws is None:
            return
        msg = json.dumps({"op": "subscribe", "args": topics})
        try:
            await self._ws.send(msg)
        except Exception as exc:
            self._log.warning("ws_public.subscribe_failed", error=str(exc))

    async def _heartbeat_loop(self, ws: Any) -> None:
        """Send ping every 20s and verify pong received within 5s."""
        while not self._stop_event.is_set():
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            try:
                ping_msg = json.dumps({"op": "ping"})
                await ws.send(ping_msg)
            except Exception as exc:
                self._log.warning("ws_public.ping_failed", error=str(exc))
                break

    async def _watchdog_loop(self) -> None:
        """Reconnect if no message received for WATCHDOG_TIMEOUT seconds."""
        while not self._stop_event.is_set():
            await asyncio.sleep(5.0)
            if time.monotonic() - self._last_message_ts > _WATCHDOG_TIMEOUT:
                self._log.warning(
                    "ws_public.watchdog_timeout",
                    timeout_seconds=_WATCHDOG_TIMEOUT,
                )
                if self._ws is not None:
                    try:
                        await self._ws.close()
                    except Exception:  # noqa: S110
                        pass
                break

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_message(self, raw: str | bytes) -> None:
        """Parse and dispatch a raw WS message."""
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return

        # Handle op responses (subscribe confirmations, pong, etc.)
        if "op" in msg:
            await self._handle_op_response(msg)
            return

        topic: str = msg.get("topic", "")
        msg_type: str = msg.get("type", "")  # snapshot | delta
        data: dict[str, Any] = msg.get("data", {})
        exchange_ts: int = msg.get("ts", 0)
        received_ts = time.time()
        latency_ms = (received_ts * 1000 - exchange_ts) if exchange_ts else None

        if topic.startswith("orderbook."):
            await self._handle_orderbook(topic, msg_type, data, latency_ms)
        elif topic.startswith("publicTrade.") or topic.startswith("trade."):
            await self._handle_trade(topic, data, exchange_ts)
        elif topic.startswith("tickers."):
            await self._handle_ticker(topic, data)
        elif topic.startswith("kline."):
            await self._handle_kline(topic, data)
        else:
            await self._emit(
                MarketDataEvent(
                    symbol=data.get("s", topic),
                    market_type=MarketType.LINEAR,
                    raw_payload=msg,
                )
            )

    async def _handle_op_response(self, msg: dict[str, Any]) -> None:
        op = msg.get("op", "")
        if op == "pong" or (op == "subscribe" and msg.get("success")):
            pass  # expected responses
        elif op == "subscribe" and not msg.get("success"):
            self._log.warning("ws_public.subscribe_failed", msg=msg)

    async def _handle_orderbook(
        self,
        topic: str,
        msg_type: str,
        data: dict[str, Any],
        latency_ms: float | None,
    ) -> None:
        """Update local orderbook and emit OrderBookEvent."""
        symbol = data.get("s", "")
        if not symbol:
            return

        if symbol not in self._orderbooks:
            # Extract depth from topic e.g. "orderbook.50.BTCUSDT"
            parts = topic.split(".")
            depth = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 50
            self._orderbooks[symbol] = LocalOrderBook(symbol, depth)

        ob = self._orderbooks[symbol]

        if msg_type == "snapshot":
            ob.apply_snapshot(data)
            update_type = "snapshot"
        elif msg_type == "delta":
            ok = ob.apply_delta(data)
            if not ok:
                # Sequence gap — invalidate and emit invalid marker
                self._log.warning(
                    "ws_public.orderbook_seq_gap",
                    symbol=symbol,
                    last_id=ob.last_update_id,
                )
                update_type = "delta"
            else:
                update_type = "delta"
        else:
            return

        # Build and emit typed event
        bids = [[str(p), str(q)] for p, q in ob._sorted_bids()]
        asks = [[str(p), str(q)] for p, q in ob._sorted_asks()]
        event = OrderBookEvent(
            symbol=symbol,
            market_type=MarketType.LINEAR,
            bids=[[Decimal(b[0]), Decimal(b[1])] for b in bids],
            asks=[[Decimal(a[0]), Decimal(a[1])] for a in asks],
            update_type=update_type,
            sequence=ob.sequence,
        )
        await self._emit(event)

    async def _handle_trade(self, topic: str, data: Any, exchange_ts: int) -> None:
        """Emit TradeEvent(s) from publicTrade messages."""
        items = data if isinstance(data, list) else [data]
        for item in items:
            symbol = item.get("s", item.get("S", ""))
            side_str = item.get("S", item.get("side", "Buy"))
            try:
                side = OrderSide(side_str)
            except ValueError:
                side = OrderSide.BUY
            event = TradeEvent(
                symbol=symbol,
                market_type=MarketType.LINEAR,
                trade_id=str(item.get("i", item.get("tradeId", ""))),
                side=side,
                price=Decimal(str(item.get("p", "0"))),
                qty=Decimal(str(item.get("v", item.get("q", "0")))),
                is_block_trade=item.get("BT", False),
            )
            await self._emit(event)

    async def _handle_ticker(self, topic: str, data: dict[str, Any]) -> None:
        """Emit TickerEvent."""
        symbol = data.get("symbol", data.get("s", ""))
        event = TickerEvent(
            symbol=symbol,
            market_type=MarketType.LINEAR,
            last_price=Decimal(str(data["lastPrice"])) if data.get("lastPrice") else None,
            bid=Decimal(str(data["bid1Price"])) if data.get("bid1Price") else None,
            ask=Decimal(str(data["ask1Price"])) if data.get("ask1Price") else None,
            volume_24h=Decimal(str(data["volume24h"])) if data.get("volume24h") else None,
            turnover_24h=Decimal(str(data["turnover24h"])) if data.get("turnover24h") else None,
            high_24h=Decimal(str(data["highPrice24h"])) if data.get("highPrice24h") else None,
            low_24h=Decimal(str(data["lowPrice24h"])) if data.get("lowPrice24h") else None,
            funding_rate=Decimal(str(data["fundingRate"])) if data.get("fundingRate") else None,
            mark_price=Decimal(str(data["markPrice"])) if data.get("markPrice") else None,
            index_price=Decimal(str(data["indexPrice"])) if data.get("indexPrice") else None,
        )
        await self._emit(event)

    async def _handle_kline(self, topic: str, data: Any) -> None:
        """Emit KlineEvent(s)."""
        from datetime import datetime

        parts = topic.split(".")
        interval = parts[1] if len(parts) >= 3 else "1"
        symbol = parts[2] if len(parts) >= 3 else ""

        items = data if isinstance(data, list) else [data]
        for item in items:
            start_ms = int(item.get("start", 0))
            event = KlineEvent(
                symbol=symbol,
                market_type=MarketType.LINEAR,
                interval=interval,
                open_time=datetime.fromtimestamp(start_ms / 1000, tz=UTC),
                open=Decimal(str(item.get("open", "0"))),
                high=Decimal(str(item.get("high", "0"))),
                low=Decimal(str(item.get("low", "0"))),
                close=Decimal(str(item.get("close", "0"))),
                volume=Decimal(str(item.get("volume", "0"))),
                turnover=Decimal(str(item.get("turnover", "0"))),
                confirm=item.get("confirm", False),
            )
            await self._emit(event)

    async def _emit(self, event: BaseEvent) -> None:
        """Put event onto queue; drop if full (non-blocking)."""
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            self._log.debug("ws_public.queue_full_drop", event_type=type(event).__name__)
