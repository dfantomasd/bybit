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
from decimal import Decimal
from typing import Any

import structlog

from trader.domain.enums import MarketType, OrderSide, OrderStatus, OrderType
from trader.domain.events import (
    BalanceUpdateEvent,
    BaseEvent,
    OrderUpdateEvent,
    PositionUpdateEvent,
)

logger = structlog.get_logger(__name__)

_HEARTBEAT_INTERVAL = 20.0
_WATCHDOG_TIMEOUT = 30.0
_PRIVATE_TOPICS = ["order", "execution", "position", "wallet"]


def _d(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("0")


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
        event_queue: asyncio.Queue,
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

        # Deduplication: {orderId_updateTime} → True
        self._seen_events: set[str] = set()

        self._reconnect_count: int = 0

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
                except Exception:
                    pass
            await asyncio.sleep(1.0)
        self._running = False

    async def stop(self) -> None:
        """Signal the WS loop to stop."""
        self._stop_event.set()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
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
            import websockets  # type: ignore
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
                    async for raw in ws:
                        if self._stop_event.is_set():
                            break
                        self._last_message_ts = time.monotonic()
                        await self._handle_message(raw)
                except Exception as exc:
                    self._log.warning("ws_private.recv_error", error=str(exc))
                finally:
                    hb_task.cancel()
                    wd_task.cancel()
                    try:
                        await asyncio.gather(hb_task, wd_task, return_exceptions=True)
                    except Exception:
                        pass

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log.warning("ws_private.connection_failed", error=str(exc))
        finally:
            self._connected = False
            self._authenticated = False
            self._ws = None

    def _build_auth_msg(self) -> dict:
        """Build Bybit V5 WebSocket auth message using HMAC-SHA256."""
        expires = int((time.time() + 1) * 1000)  # 1 second in future
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
                    except Exception:
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
        except asyncio.TimeoutError:
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
                except Exception:
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
            if dedup_key in self._seen_events:
                continue
            self._seen_events.add(dedup_key)

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
        """Parse execution (fill) messages."""
        items = data if isinstance(data, list) else [data]
        for item in items:
            exec_id = item.get("execId", "")
            order_id = item.get("orderId", "")
            dedup_key = f"exec_{exec_id}_{order_id}"
            if dedup_key in self._seen_events:
                continue
            self._seen_events.add(dedup_key)
            # Emit as generic MarketDataEvent (execution stream)
            # In production you'd emit a FillEvent
            self._log.info(
                "ws_private.execution",
                exec_id=exec_id,
                symbol=item.get("symbol", ""),
            )

    async def _handle_position(self, data: Any) -> None:
        """Parse position update messages."""
        items = data if isinstance(data, list) else [data]
        for item in items:
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

    async def _emit(self, event: BaseEvent) -> None:
        """Put event onto queue non-blocking."""
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            self._log.debug("ws_private.queue_full_drop", event_type=type(event).__name__)
