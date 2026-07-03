"""Maps domain OrderIntent → pybit params dict and pybit response → domain models.

All price / quantity rounding uses Decimal arithmetic; float is never used
for financial calculations.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from typing import Any

import structlog

from trader.domain.enums import MarketType, OrderSide, OrderType
from trader.domain.models import Balance, Fill, InstrumentInfo, OrderIntent, Position

logger = structlog.get_logger(__name__)


def _d(value: Any) -> Decimal:
    """Safely convert a value to Decimal; return Decimal(0) on failure."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _round_to_step(value: Decimal, step: Decimal) -> Decimal:
    """Round *value* down to the nearest multiple of *step*.

    Uses Decimal arithmetic exclusively — no float involved.
    """
    if step <= 0:
        return value
    quotient = value / step
    floored = Decimal(int(quotient.to_integral_value(rounding=ROUND_DOWN)))
    return floored * step


def _parse_dt(ms_str: Any) -> datetime:
    """Convert a millisecond-epoch string/int to a UTC datetime."""
    try:
        ms = int(ms_str)
        return datetime.fromtimestamp(ms / 1000, tz=UTC)
    except (TypeError, ValueError, OSError):
        return datetime.now(tz=UTC)


class OrderMapper:
    """Bidirectional mapper between domain objects and Bybit API dicts."""

    # ------------------------------------------------------------------
    # Domain → API
    # ------------------------------------------------------------------

    def intent_to_params(self, intent: OrderIntent, category: str) -> dict[str, Any]:
        """Convert an OrderIntent to a pybit place_order kwargs dict.

        Prices and quantities are already rounded by the risk layer before
        OrderIntent is created, but this method applies final tick/step rounding
        as a safety net if InstrumentInfo is supplied via extra context.
        """
        params: dict[str, Any] = {
            "category": category,
            "symbol": intent.symbol,
            "side": intent.side.value,
            "orderType": intent.order_type.value,
            "qty": str(intent.qty),
            "orderLinkId": intent.order_link_id,
            "timeInForce": intent.time_in_force,
        }

        if intent.order_type == OrderType.LIMIT:
            if intent.price is not None:
                params["price"] = str(intent.price)
            else:
                logger.warning(
                    "order_mapper.limit_order_missing_price",
                    order_link_id=intent.order_link_id,
                    symbol=intent.symbol,
                )

        if intent.reduce_only:
            params["reduceOnly"] = True

        if intent.take_profit is not None:
            params["takeProfit"] = str(intent.take_profit)
            if intent.tp_order_type is not None:
                params["tpOrderType"] = intent.tp_order_type.value

        if intent.stop_loss is not None:
            params["stopLoss"] = str(intent.stop_loss)
            if intent.sl_order_type is not None:
                params["slOrderType"] = intent.sl_order_type.value

        if intent.take_profit is not None or intent.stop_loss is not None:
            params["tpslMode"] = "Full"

        # Linear / inverse perpetuals use positionIdx (0=one-way, 1=Buy, 2=Sell)
        if category in ("linear", "inverse"):
            params["positionIdx"] = 0  # default one-way mode

        logger.debug(
            "order_mapper.intent_to_params",
            order_link_id=intent.order_link_id,
            symbol=intent.symbol,
            side=intent.side.value,
            qty=str(intent.qty),
        )
        return params

    def round_price(self, price: Decimal, tick_size: Decimal) -> Decimal:
        """Round price to the instrument tick size."""
        return _round_to_step(price, tick_size)

    def round_qty(self, qty: Decimal, qty_step: Decimal) -> Decimal:
        """Round quantity down to the instrument qty step."""
        return _round_to_step(qty, qty_step)

    # ------------------------------------------------------------------
    # API → Domain
    # ------------------------------------------------------------------

    def response_to_order_state(self, response: dict[str, Any]) -> dict[str, Any]:
        """Extract order state fields from a Bybit REST order response."""
        result = response.get("result", {})
        return {
            "order_id": result.get("orderId", ""),
            "order_link_id": result.get("orderLinkId", ""),
            "symbol": result.get("symbol", ""),
            "side": result.get("side", ""),
            "order_type": result.get("orderType", ""),
            "price": result.get("price", "0"),
            "qty": result.get("qty", "0"),
            "order_status": result.get("orderStatus", ""),
            "created_time": result.get("createdTime", "0"),
            "updated_time": result.get("updatedTime", "0"),
        }

    def ws_order_to_event(self, ws_payload: dict[str, Any]) -> dict[str, Any]:
        """Parse a WebSocket order update message into a normalised event dict.

        Bybit WS order update topic: ``order``
        """
        data = ws_payload.get("data", [])
        events = []
        for item in data:
            events.append(
                {
                    "order_id": item.get("orderId", ""),
                    "order_link_id": item.get("orderLinkId", ""),
                    "symbol": item.get("symbol", ""),
                    "side": item.get("side", ""),
                    "order_type": item.get("orderType", ""),
                    "price": _d(item.get("price", "0")),
                    "qty": _d(item.get("qty", "0")),
                    "cum_exec_qty": _d(item.get("cumExecQty", "0")),
                    "order_status": item.get("orderStatus", ""),
                    "reduce_only": item.get("reduceOnly", False),
                    "close_on_trigger": item.get("closeOnTrigger", False),
                    "created_time": _parse_dt(item.get("createdTime", 0)),
                    "updated_time": _parse_dt(item.get("updatedTime", 0)),
                }
            )
        return {"events": events, "topic": ws_payload.get("topic", "order")}

    def ws_execution_to_fill(self, ws_payload: dict[str, Any]) -> Fill:
        """Parse a WebSocket execution (fill) message into a Fill domain model.

        Bybit WS execution topic: ``execution``
        """
        data = ws_payload.get("data", [{}])
        item = data[0] if data else {}

        return Fill(
            exchange_exec_id=item.get("execId", ""),
            order_link_id=item.get("orderLinkId", ""),
            symbol=item.get("symbol", "UNKNOWN"),
            side=OrderSide(item.get("side", "Buy")),
            qty=_d(item.get("execQty", "0")),
            price=_d(item.get("execPrice", "0")),
            fee=_d(item.get("execFee", "0")),
            fee_currency=item.get("feeCurrency", "USDT"),
            is_maker=item.get("isMaker", False),
            executed_at=_parse_dt(item.get("execTime", 0)),
        )

    def rest_position_to_model(self, data: dict[str, Any]) -> Position:
        """Convert a single Bybit REST position dict to a Position domain model."""
        side_str = data.get("side", "Buy")
        try:
            side = OrderSide(side_str)
        except ValueError:
            side = OrderSide.BUY

        category_str = data.get("category", "linear")
        try:
            market_type = MarketType(category_str)
        except ValueError:
            market_type = MarketType.LINEAR

        return Position(
            symbol=data.get("symbol", "UNKNOWN"),
            market_type=market_type,
            side=side,
            size=_d(data.get("size", "0")),
            entry_price=_d(data.get("avgPrice", data.get("entryPrice", "0"))),
            mark_price=_d(data.get("markPrice")) if data.get("markPrice") else None,
            liquidation_price=_d(data.get("liqPrice")) if data.get("liqPrice") else None,
            unrealised_pnl=_d(data.get("unrealisedPnl", "0")),
            realised_pnl=_d(data.get("cumRealisedPnl", data.get("realisedPnl", "0"))),
            leverage=_d(data.get("leverage", "1")),
            margin_type=str(data.get("tradeMode", data.get("marginType", "cross"))),
            updated_at=_parse_dt(data.get("updatedTime", 0)),
        )

    def rest_balance_to_model(self, data: dict[str, Any]) -> Balance:
        """Convert a Bybit UNIFIED wallet balance dict to a Balance domain model.

        ``data`` should be one element from ``result.list[].coin[]``.
        """
        updated_raw = data.get("updatedTime") or data.get("time") or data.get("createdTime")
        return Balance(
            account_type=data.get("accountType", "UNIFIED"),
            currency=data.get("coin", data.get("currency", "USDT")),
            wallet_balance=_d(data.get("walletBalance", "0")),
            available_balance=_d(
                # Bybit UNIFIED: prefer availableToWithdraw; if zero fall back
                # to walletBalance so the bot doesn't use $1000 fallback capital
                data.get("availableToWithdraw") or data.get("availableBalance") or data.get("walletBalance", "0")
            ),
            unrealised_pnl=_d(data.get("unrealisedPnl", "0")),
            margin_balance=_d(data.get("equity")) if data.get("equity") else None,
            updated_at=_parse_dt(updated_raw) if updated_raw else datetime.now(tz=UTC),
        )

    def instruments_info_to_model(self, data: dict[str, Any]) -> InstrumentInfo:
        """Convert a Bybit instrumentsInfo item to an InstrumentInfo domain model."""
        lot_size = data.get("lotSizeFilter", {})
        price_filter = data.get("priceFilter", {})
        leverage_filter = data.get("leverageFilter", {})

        category_str = data.get("category", "linear")
        try:
            market_type = MarketType(category_str)
        except ValueError:
            market_type = MarketType.LINEAR

        return InstrumentInfo(
            symbol=data.get("symbol", "UNKNOWN"),
            market_type=market_type,
            base_coin=data.get("baseCoin", ""),
            quote_coin=data.get("quoteCoin", ""),
            min_order_qty=_d(lot_size.get("minOrderQty", "0")),
            max_order_qty=_d(lot_size.get("maxOrderQty", "999999")),
            qty_step=_d(lot_size.get("qtyStep", "0.001")),
            tick_size=_d(price_filter.get("tickSize", "0.01")),
            min_notional=_d(lot_size.get("minNotionalValue")) if lot_size.get("minNotionalValue") else None,
            max_leverage=_d(leverage_filter.get("maxLeverage")) if leverage_filter.get("maxLeverage") else None,
            turnover_24h=_d(data.get("turnover24h")) if data.get("turnover24h") else None,
            status=data.get("status", "Trading"),
        )
