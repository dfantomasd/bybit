"""Bybit V5 private WebSocket manager.

Handles authenticated subscriptions: order, execution, position, wallet.
Uses HMAC-SHA256 for authentication on connect.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from collections import OrderedDict
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import structlog

from trader.domain.enums import MarketType, OrderSide, OrderStatus, OrderType
from trader.domain.events import (
    BalanceUpdateEvent,
    BaseEvent,
    ExecutionUpdateEvent,
    OrderUpdateEvent,
    PositionUpdateEvent,
)

logger = structlog.get_logger(__name__)

_HEARTBEAT_INTERVAL = 20.0
_WATCHDOG_TIMEOUT = 30.0
_AUTH_EXPIRES_SECONDS = 10
_MAX_SEEN_EVENTS = 20_000
_PRIVATE_TOPICS = ["order", "execution", "position", "wallet"]


def _d(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("0")


def _event_time(data: dict[str, Any], fallback: dict[str, Any] | None = None) -> datetime:
    raw = data.get("updatedTime") or data.get("creationTime") or data.get("createdTime")
    if raw is None and fallback is not None:
        raw = fallback.get("updatedTime") or fallback.get("creationTime") or fallback.get("createdTime")
    if raw is None:
        return datetime.now(tz=UTC)
    try:
        millis = int(raw)
        if millis > 0:
            return datetime.fromtimestamp(millis / 1000, tz=UTC)
    except (TypeError, ValueError, OSError):
        return datetime.now(tz=UTC)
    return datetime.now(tz=UTC)


class BybitPrivateWebSocket:
    """Manages Bybit V5 private WebSocket connection.

    Subscriptions: order, execution, position, wallet.

    Features
    --------
    - Auth via HMAC-SHA256 on connect.
    - Reconnect with re-auth.
    - Emit OrderUpdateEvent, PositionUpdateEvent, BalanceUpdateEvent.
    - Deduplicate events by orderId + updateTime.
    - Log all events to audit journal (structlog).
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_secret: str,
        event_queue: asyncio.Queue[BaseEvent],
        metrics: Any = None,
        logger: Any = None,
    ) -> None:
        self._endpoint = endpoint
        self._api_key = api_key
        self._api_secret = api_secret
        self._event_queue = event_queue
        self._metrics = metrics
        self._log = logger or structlog.get_logger(__name__)

        self._ws: Any = None
        self._connected: bool = False
        self._authenticated: bool = False
        self._running: bool = False
        self._stop_event = asyncio.Event()
        self._last_message_ts: float = 0.0

        # Deduplication keys are retained as a bounded LRU so long-running
        # private streams cannot grow memory without limit.
        self._seen_events: OrderedDict[str, None] = OrderedDict()

        self._reconnect_count: int = 0
        # Per-symbol last updatedTime (ms) — used to drop out-of-order position frames
        self._last_position_ts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect, authenticate and start processing messages."""
        self._running = True
        while not self._stop_event.is_set():
            await self._run_connection()
            if self._stop_event.is_set():
                break
            self._reconnect_count += 1
            if self._metrics is not None:
                try:
                    self._metrics.ws_reconnect_total.labels(name="private").inc()
                except Exception:  # noqa: S110
                    pass
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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

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

        try:
            async with websockets.connect(
                self._endpoint,
                ping_interval=None,
                ping_timeout=None,
                close_timeout=5,
            ) as ws:
                self._ws = ws
                self._connected = True
                self._authenticated = False
                self._last_message_ts = time.monotonic()

                self._log.info("ws_private.connected", endpoint=self._endpoint)

                # Authenticate
                if not await self._authenticate(ws):
                    self._log.error("ws_private.auth_failed")
                    return

                # Subscribe to private topics
                await self._send_subscribe(ws, _PRIVATE_TOPICS)

                # Start heartbeat + watchdog
                hb_task = asyncio.create_task(self._heartbeat_loop(ws))
                wd_task = asyncio.create_task(self._watchdog_loop(ws))

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
                                "ws_private.handle_error",
                                error=str(exc),
                                backoff=round(_err_backoff, 2),
                            )
                            await asyncio.sleep(_err_backoff)
                except Exception as exc:
                    self._log.warning("ws_private.recv_error", error=str(exc))
                finally:
                    hb_task.cancel()
                    wd_task.cancel()
                    try:
                        await asyncio.gather(hb_task, wd_task, return_exceptions=True)
                    except Exception:  # noqa: S110
                        pass

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log.warning("ws_private.connection_failed", error=str(exc))
        finally:
            self._connected = False
            self._authenticated = False
            self._ws = None

    def _build_auth_msg(self) -> dict[str, Any]:
        """Build Bybit V5 WebSocket auth message using HMAC-SHA256."""
        expires = int((time.time() + _AUTH_EXPIRES_SECONDS) * 1000)
        sign_str = f"GET/realtime{expires}"
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            sign_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {
            "op": "auth",
            "args": [self._api_key, expires, signature],
        }

    async def _authenticate(self, ws: Any) -> bool:
        """Send auth message and wait for confirmation."""
        auth_msg = json.dumps(self._build_auth_msg())
        await ws.send(auth_msg)
        # Wait up to 5 seconds for auth response
        try:
            async with asyncio.timeout(5.0):
                async for raw in ws:
                    self._last_message_ts = time.monotonic()
                    try:
                        msg = json.loads(raw)
                    except Exception:  # noqa: S112
                        continue
                    if msg.get("op") == "auth":
                        if msg.get("success"):
                            self._authenticated = True
                            self._log.info("ws_private.authenticated")
                            return True
                        else:
                            self._log.error("ws_private.auth_rejected", msg=msg)
                            return False
        except TimeoutError:
            self._log.error("ws_private.auth_timeout")
        return False

    async def _send_subscribe(self, ws: Any, topics: list[str]) -> None:
        msg = json.dumps({"op": "subscribe", "args": topics})
        try:
            await ws.send(msg)
        except Exception as exc:
            self._log.warning("ws_private.subscribe_failed", error=str(exc))

    async def _heartbeat_loop(self, ws: Any) -> None:
        while not self._stop_event.is_set():
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            try:
                await ws.send(json.dumps({"op": "ping"}))
            except Exception as exc:
                self._log.warning("ws_private.ping_failed", error=str(exc))
                break

    async def _watchdog_loop(self, ws: Any) -> None:
        while not self._stop_event.is_set():
            await asyncio.sleep(5.0)
            if time.monotonic() - self._last_message_ts > _WATCHDOG_TIMEOUT:
                self._log.warning("ws_private.watchdog_timeout")
                try:
                    await ws.close()
                except Exception:  # noqa: S110
                    pass
                break

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_message(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
        except Exception:
            return

        # Op responses
        if "op" in msg:
            return

        topic: str = msg.get("topic", "")
        data: Any = msg.get("data", [])

        self._log.debug("ws_private.message", topic=topic)

        if topic == "order":
            await self._handle_order(data)
        elif topic == "execution":
            await self._handle_execution(data)
        elif topic == "position":
            await self._handle_position(data)
        elif topic == "wallet":
            await self._handle_wallet(data)

    async def _handle_order(self, data: Any) -> None:
        """Parse order update messages."""
        items = data if isinstance(data, list) else [data]
        for item in items:
            order_id = item.get("orderId", "")
            update_time = item.get("updatedTime", "")
            dedup_key = f"{order_id}_{update_time}"
            if self._mark_seen(dedup_key):
                continue

            # Map exchange status to OrderStatus
            exchange_status = item.get("orderStatus", "")
            order_status = self._map_order_status(exchange_status)

            try:
                side = OrderSide(item.get("side", "Buy"))
            except ValueError:
                side = OrderSide.BUY
            try:
                order_type = OrderType(item.get("orderType", "Limit"))
            except ValueError:
                order_type = OrderType.LIMIT
            try:
                market_type = MarketType(item.get("category", "linear"))
            except ValueError:
                market_type = MarketType.LINEAR

            event = OrderUpdateEvent(
                symbol=item.get("symbol", ""),
                market_type=market_type,
                order_id=order_id,
                order_link_id=item.get("orderLinkId", ""),
                side=side,
                order_type=order_type,
                status=order_status,
                qty=_d(item.get("qty", "0")),
                filled_qty=_d(item.get("cumExecQty", "0")),
                price=_d(item.get("price")) if item.get("price") else None,
                avg_fill_price=_d(item.get("avgPrice")) if item.get("avgPrice") else None,
                fee=_d(item.get("cumExecFee", "0")),
                cancel_type=item.get("cancelType"),
                reject_reason=item.get("rejectReason"),
            )
            self._log.info(
                "ws_private.order_update",
                order_link_id=event.order_link_id,
                status=order_status.value,
            )
            await self._emit(event)

    async def _handle_execution(self, data: Any) -> None:
        """Parse execution (fill) messages and emit ExecutionUpdateEvent."""
        items = data if isinstance(data, list) else [data]
        for item in items:
            exec_id = item.get("execId", "")
            order_id = item.get("orderId", "")
            dedup_key = f"exec_{exec_id}_{order_id}"
            if self._mark_seen(dedup_key):
                continue

            try:
                side = OrderSide(item.get("side", "Buy"))
            except ValueError:
                side = OrderSide.BUY
            try:
                order_type = OrderType(item.get("orderType", "Market"))
            except ValueError:
                order_type = OrderType.MARKET
            try:
                market_type = MarketType(item.get("category", "linear"))
            except ValueError:
                market_type = MarketType.LINEAR

            exec_price = _d(item.get("execPrice", "0"))
            exec_qty = _d(item.get("execQty", "0"))
            exec_value = exec_price * exec_qty

            event = ExecutionUpdateEvent(
                symbol=item.get("symbol", ""),
                market_type=market_type,
                order_id=order_id,
                order_link_id=item.get("orderLinkId", ""),
                exec_id=exec_id,
                side=side,
                order_type=order_type,
                exec_price=exec_price,
                exec_qty=exec_qty,
                exec_fee=_d(item.get("execFee", "0")),
                exec_value=exec_value,
                is_maker=item.get("isMaker", False),
                closed_size=_d(item.get("closedSize", "0")),
            )
            self._log.info(
                "ws_private.execution",
                exec_id=exec_id,
                symbol=event.symbol,
                exec_price=str(exec_price),
                exec_qty=str(exec_qty),
            )
            await self._emit(event)

    async def _handle_position(self, data: Any) -> None:
        """Parse position update messages."""
        items = data if isinstance(data, list) else [data]
        for item in items:
            symbol = item.get("symbol", "")
            try:
                updated_ms = int(item.get("updatedTime", 0) or 0)
            except (TypeError, ValueError):
                updated_ms = 0
            if updated_ms > 0 and symbol:
                prev = self._last_position_ts.get(symbol, 0)
                if updated_ms < prev:
                    self._log.debug(
                        "ws_position_out_of_order_dropped",
                        symbol=symbol,
                        event_ts_ms=updated_ms,
                        last_ts_ms=prev,
                    )
                    continue
                self._last_position_ts[symbol] = updated_ms
            try:
                side = OrderSide(item.get("side", "Buy"))
            except ValueError:
                side = OrderSide.BUY
            try:
                market_type = MarketType(item.get("category", "linear"))
            except ValueError:
                market_type = MarketType.LINEAR

            event = PositionUpdateEvent(
                symbol=item.get("symbol", ""),
                market_type=market_type,
                side=side,
                size=_d(item.get("size", "0")),
                entry_price=_d(item.get("entryPrice", item.get("avgPrice", "0"))),
                mark_price=_d(item.get("markPrice")) if item.get("markPrice") else None,
                liquidation_price=_d(item.get("liqPrice")) if item.get("liqPrice") else None,
                unrealised_pnl=_d(item.get("unrealisedPnl", "0")),
                realised_pnl=_d(item.get("cumRealisedPnl", "0")),
                leverage=_d(item.get("leverage", "1")),
            )
            await self._emit(event)

    async def _handle_wallet(self, data: Any) -> None:
        """Parse wallet balance update messages."""
        items = data if isinstance(data, list) else [data]
        for item in items:
            account_type = item.get("accountType", "UNIFIED")
            for coin in item.get("coin", []):
                event = BalanceUpdateEvent(
                    account_type=account_type,
                    currency=coin.get("coin", "USDT"),
                    wallet_balance=_d(coin.get("walletBalance", "0")),
                    available_balance=_d(coin.get("availableToWithdraw", coin.get("availableBalance", "0"))),
                    unrealised_pnl=_d(coin.get("unrealisedPnl", "0")),
                    timestamp=_event_time(coin, item),
                )
                await self._emit(event)

    def _map_order_status(self, exchange_status: str) -> OrderStatus:
        """Map Bybit exchange order status string to OrderStatus enum."""
        mapping: dict[str, OrderStatus] = {
            "New": OrderStatus.WS_CONFIRMED,
            "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "Rejected": OrderStatus.REJECTED,
            "Expired": OrderStatus.EXPIRED,
            "Untriggered": OrderStatus.WS_CONFIRMED,
            "Triggered": OrderStatus.WS_CONFIRMED,
            "Active": OrderStatus.WS_CONFIRMED,
            "PendingCancel": OrderStatus.CANCEL_REQUESTED,
        }
        return mapping.get(exchange_status, OrderStatus.UNKNOWN_RECONCILIATION_REQUIRED)

    def _mark_seen(self, key: str) -> bool:
        """Return True when ``key`` is a duplicate; otherwise remember it."""
        if key in self._seen_events:
            self._seen_events.move_to_end(key)
            return True
        self._seen_events[key] = None
        while len(self._seen_events) > _MAX_SEEN_EVENTS:
            self._seen_events.popitem(last=False)
        return False

    _CRITICAL_EVENT_TYPES = (PositionUpdateEvent, OrderUpdateEvent, ExecutionUpdateEvent)

    async def _emit(self, event: BaseEvent) -> None:
        """Put event onto queue.

        For critical safety events (position/order/execution), if the queue is
        full we drop the oldest entry to make room rather than silently
        discarding the new event.  Dropping stale data is safer than dropping
        fresh data — the consumer will re-sync via the next REST reconciliation.
        """
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            if isinstance(event, self._CRITICAL_EVENT_TYPES):
                try:
                    dropped = self._event_queue.get_nowait()
                    self._log.warning(
                        "ws_private.queue_full_dropped_oldest",
                        dropped_type=type(dropped).__name__,
                        new_type=type(event).__name__,
                    )
                    self._event_queue.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    self._log.error(
                        "ws_private.critical_event_lost",
                        event_type=type(event).__name__,
                    )
            else:
                self._log.debug("ws_private.queue_full_drop", event_type=type(event).__name__)
