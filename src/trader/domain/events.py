"""Event types for the internal event bus.

All events inherit from ``BaseEvent`` which provides a unique ``event_id``,
``timestamp``, and optional ``correlation_id`` for request tracing.
Events are immutable (frozen Pydantic models) and JSON-serialisable.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from trader.domain.enums import (
    KillSwitchMode,
    MarketRegime,
    MarketType,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskDecisionStatus,
    SystemStatus,
    TradingMode,
)


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class BaseEvent(BaseModel):
    """Immutable base event with identity and tracing fields."""

    model_config = ConfigDict(frozen=True)

    event_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    timestamp: datetime = Field(default_factory=_now_utc)
    correlation_id: uuid.UUID | None = None
    # Topic hint for the event bus router
    topic: str = "base"


# ---------------------------------------------------------------------------
# Market data events
# ---------------------------------------------------------------------------


class MarketDataEvent(BaseEvent):
    """Generic market data update (catch-all when type is not known)."""

    topic: str = "market.data"
    symbol: str
    market_type: MarketType
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class OrderBookEvent(BaseEvent):
    """Level-2 order book snapshot or delta."""

    topic: str = "market.orderbook"
    symbol: str
    market_type: MarketType

    # Bid / ask levels: list of [price, size]
    bids: list[list[Decimal]] = Field(default_factory=list)
    asks: list[list[Decimal]] = Field(default_factory=list)

    # "snapshot" or "delta"
    update_type: str = "snapshot"
    sequence: int | None = None


class TradeEvent(BaseEvent):
    """Public trade (tape) event from the exchange."""

    topic: str = "market.trade"
    symbol: str
    market_type: MarketType

    trade_id: str
    side: OrderSide
    price: Decimal
    qty: Decimal
    is_block_trade: bool = False
    executed_at: datetime = Field(default_factory=_now_utc)


class TickerEvent(BaseEvent):
    """24-hour ticker snapshot."""

    topic: str = "market.ticker"
    symbol: str
    market_type: MarketType

    last_price: Decimal | None = None
    bid: Decimal | None = None
    ask: Decimal | None = None
    volume_24h: Decimal | None = None
    turnover_24h: Decimal | None = None
    high_24h: Decimal | None = None
    low_24h: Decimal | None = None
    price_change_pct_24h: float | None = None
    funding_rate: Decimal | None = None
    next_funding_at: datetime | None = None
    mark_price: Decimal | None = None
    index_price: Decimal | None = None
    open_interest: Decimal | None = None


class KlineEvent(BaseEvent):
    """Candlestick (OHLCV) bar event."""

    topic: str = "market.kline"
    symbol: str
    market_type: MarketType
    interval: str  # "1", "5", "15", "60", "D", etc.

    open_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    turnover: Decimal
    confirm: bool = False  # True when bar is closed / confirmed


# ---------------------------------------------------------------------------
# Account / order events
# ---------------------------------------------------------------------------


class OrderUpdateEvent(BaseEvent):
    """Order lifecycle update received from the exchange (WS or REST poll)."""

    topic: str = "account.order"
    symbol: str
    market_type: MarketType

    order_id: str  # exchange order ID
    order_link_id: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus

    qty: Decimal
    filled_qty: Decimal = Decimal(0)
    remaining_qty: Decimal | None = None
    price: Decimal | None = None
    avg_fill_price: Decimal | None = None

    fee: Decimal = Decimal(0)
    fee_currency: str = "USDT"

    cancel_type: str | None = None
    reject_reason: str | None = None


class PositionUpdateEvent(BaseEvent):
    """Position state update (from WS or reconciliation)."""

    topic: str = "account.position"
    symbol: str
    market_type: MarketType
    side: OrderSide

    size: Decimal
    entry_price: Decimal
    mark_price: Decimal | None = None
    liquidation_price: Decimal | None = None
    unrealised_pnl: Decimal = Decimal(0)
    realised_pnl: Decimal = Decimal(0)
    leverage: Decimal = Decimal(1)
    margin_type: str = "cross"


class BalanceUpdateEvent(BaseEvent):
    """Account balance update."""

    topic: str = "account.balance"
    account_type: str
    currency: str
    wallet_balance: Decimal
    available_balance: Decimal
    unrealised_pnl: Decimal = Decimal(0)


class ExecutionUpdateEvent(BaseEvent):
    """Trade execution (fill) event from private WebSocket."""

    topic: str = "account.execution"
    symbol: str
    market_type: MarketType
    order_id: str
    order_link_id: str = ""
    exec_id: str
    side: OrderSide
    order_type: OrderType
    exec_price: Decimal
    exec_qty: Decimal
    exec_fee: Decimal = Decimal(0)
    exec_value: Decimal = Decimal(0)
    is_maker: bool = False
    closed_size: Decimal = Decimal(0)


# ---------------------------------------------------------------------------
# Strategy / risk pipeline events
# ---------------------------------------------------------------------------


class TradeProposalEvent(BaseEvent):
    """Strategy emitted a new trade proposal."""

    topic: str = "strategy.proposal"
    strategy_id: str
    proposal_id: uuid.UUID
    symbol: str
    side: OrderSide
    requested_qty: Decimal
    confidence: float
    regime: MarketRegime = MarketRegime.UNCERTAIN


class RiskDecisionEvent(BaseEvent):
    """Risk Manager issued a decision on a trade proposal."""

    topic: str = "risk.decision"
    proposal_id: uuid.UUID
    decision_id: uuid.UUID
    status: RiskDecisionStatus
    approved_qty: Decimal | None = None
    reason: str = ""
    triggered_rules: list[str] = Field(default_factory=list)


class OrderIntentEvent(BaseEvent):
    """An order intent has been constructed and is ready for submission."""

    topic: str = "execution.intent"
    intent_id: uuid.UUID
    decision_id: uuid.UUID
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: Decimal
    order_link_id: str


class OrderConfirmedEvent(BaseEvent):
    """Exchange confirmed order placement (WS or REST)."""

    topic: str = "execution.confirmed"
    order_link_id: str
    exchange_order_id: str
    symbol: str
    side: OrderSide
    qty: Decimal
    price: Decimal | None = None
    confirmed_via: str = "websocket"  # "websocket" | "rest"


class ReconciliationEvent(BaseEvent):
    """Reconciliation loop completed a pass."""

    topic: str = "reconciliation.result"
    run_id: uuid.UUID
    orders_checked: int = 0
    positions_checked: int = 0
    discrepancies_found: int = 0
    discrepancies_resolved: int = 0
    success: bool = True
    summary: str = ""


# ---------------------------------------------------------------------------
# System events
# ---------------------------------------------------------------------------


class SystemEvent(BaseEvent):
    """System lifecycle state change."""

    topic: str = "system.status"
    previous_status: SystemStatus | None = None
    new_status: SystemStatus
    trading_mode: TradingMode
    reason: str = ""


class AlertEvent(BaseEvent):
    """Alert raised by a monitoring component."""

    topic: str = "system.alert"
    severity: str  # "info" | "warning" | "critical"
    component: str  # which component raised the alert
    title: str
    body: str
    symbol: str | None = None
    kill_switch_mode: KillSwitchMode | None = None
    # Whether this alert has been sent to the operator (Telegram, etc.)
    notified: bool = False
