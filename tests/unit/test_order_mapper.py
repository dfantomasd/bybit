"""Tests for OrderMapper — domain ↔ API dict translation."""
from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

from trader.domain.enums import MarketType, OrderSide, OrderType
from trader.domain.models import Fill, InstrumentInfo, OrderIntent, Position
from trader.exchange.order_mapper import OrderMapper, _d, _round_to_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_intent(
    qty: str = "0.1",
    price: str | None = "30000",
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.LIMIT,
    reduce_only: bool = False,
    tp: str | None = None,
    sl: str | None = None,
    order_link_id: str = "TN-260605-MOMO-TEST1234-abc123",
) -> OrderIntent:
    kwargs = dict(
        decision_id=uuid.uuid4(),
        proposal_id=uuid.uuid4(),
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        side=side,
        order_type=order_type,
        qty=Decimal(qty),
        order_link_id=order_link_id,
        reduce_only=reduce_only,
    )
    if price is not None:
        kwargs["price"] = Decimal(price)
    if tp is not None:
        kwargs["take_profit"] = Decimal(tp)
    if sl is not None:
        kwargs["stop_loss"] = Decimal(sl)
    return OrderIntent(**kwargs)


# ---------------------------------------------------------------------------
# _round_to_step tests
# ---------------------------------------------------------------------------

class TestRoundToStep:
    def test_rounds_down_to_step(self) -> None:
        result = _round_to_step(Decimal("0.123456"), Decimal("0.001"))
        assert result == Decimal("0.123")

    def test_exact_multiple_unchanged(self) -> None:
        result = _round_to_step(Decimal("0.100"), Decimal("0.001"))
        assert result == Decimal("0.100")

    def test_price_rounded_to_tick(self) -> None:
        result = _round_to_step(Decimal("29999.7"), Decimal("0.5"))
        assert result == Decimal("29999.5")

    def test_zero_step_returns_value(self) -> None:
        # Edge case: step of 0 returns value unchanged
        result = _round_to_step(Decimal("5"), Decimal("0"))
        assert result == Decimal("5")

    def test_uses_decimal_not_float(self) -> None:
        # Floating-point representation issues should not affect result
        qty = Decimal("0.3")
        step = Decimal("0.1")
        result = _round_to_step(qty, step)
        assert result == Decimal("0.3")


# ---------------------------------------------------------------------------
# intent_to_params tests
# ---------------------------------------------------------------------------

class TestIntentToParams:
    def setup_method(self) -> None:
        self.mapper = OrderMapper()

    def test_basic_limit_buy(self) -> None:
        intent = _make_intent(qty="0.05", price="29000", side=OrderSide.BUY)
        params = self.mapper.intent_to_params(intent, "linear")
        assert params["symbol"] == "BTCUSDT"
        assert params["side"] == "Buy"
        assert params["orderType"] == "Limit"
        assert params["qty"] == "0.05"
        assert params["price"] == "29000"
        assert params["category"] == "linear"

    def test_market_sell_no_price(self) -> None:
        intent = _make_intent(qty="0.1", price=None, side=OrderSide.SELL, order_type=OrderType.MARKET)
        params = self.mapper.intent_to_params(intent, "linear")
        assert params["orderType"] == "Market"
        assert "price" not in params

    def test_order_link_id_preserved(self) -> None:
        intent = _make_intent(order_link_id="TN-260605-MOMO-TEST1234-abc123")
        params = self.mapper.intent_to_params(intent, "linear")
        assert params["orderLinkId"] == "TN-260605-MOMO-TEST1234-abc123"

    def test_reduce_only_flag_true(self) -> None:
        intent = _make_intent(reduce_only=True, price=None, order_type=OrderType.MARKET)
        params = self.mapper.intent_to_params(intent, "linear")
        assert params["reduceOnly"] is True

    def test_reduce_only_flag_false_not_included(self) -> None:
        intent = _make_intent(reduce_only=False)
        params = self.mapper.intent_to_params(intent, "linear")
        assert not params.get("reduceOnly", False)

    def test_take_profit_and_stop_loss_included(self) -> None:
        intent = _make_intent(tp="32000", sl="28000")
        params = self.mapper.intent_to_params(intent, "linear")
        assert params["takeProfit"] == "32000"
        assert params["stopLoss"] == "28000"

    def test_time_in_force_preserved(self) -> None:
        intent = _make_intent()
        params = self.mapper.intent_to_params(intent, "linear")
        assert params["timeInForce"] == "GTC"

    def test_spot_category_has_no_position_idx(self) -> None:
        intent = _make_intent()
        params = self.mapper.intent_to_params(intent, "spot")
        # spot category — positionIdx is only added for linear/inverse
        assert "positionIdx" not in params

    def test_linear_has_position_idx_zero(self) -> None:
        intent = _make_intent()
        params = self.mapper.intent_to_params(intent, "linear")
        assert params["positionIdx"] == 0


# ---------------------------------------------------------------------------
# round_price / round_qty tests
# ---------------------------------------------------------------------------

class TestRounding:
    def setup_method(self) -> None:
        self.mapper = OrderMapper()

    def test_round_price_to_tick_size(self) -> None:
        result = self.mapper.round_price(Decimal("29999.73"), Decimal("0.5"))
        assert result == Decimal("29999.5")

    def test_round_qty_to_step(self) -> None:
        result = self.mapper.round_qty(Decimal("0.12345"), Decimal("0.001"))
        assert result == Decimal("0.123")

    def test_round_qty_exact_multiple(self) -> None:
        result = self.mapper.round_qty(Decimal("1.000"), Decimal("0.001"))
        assert result == Decimal("1.000")

    def test_no_float_used_in_rounding(self) -> None:
        # All Decimal operations — should be exact
        price = Decimal("1234567.891234")
        tick = Decimal("0.01")
        result = self.mapper.round_price(price, tick)
        assert isinstance(result, Decimal)
        assert result == Decimal("1234567.89")


# ---------------------------------------------------------------------------
# WS order update parser
# ---------------------------------------------------------------------------

class TestWsOrderToEvent:
    def setup_method(self) -> None:
        self.mapper = OrderMapper()

    def _make_payload(self) -> dict:
        return {
            "topic": "order",
            "data": [
                {
                    "orderId": "abc123",
                    "orderLinkId": "TN-260605-MOMO-TEST1234-abc123",
                    "symbol": "BTCUSDT",
                    "side": "Buy",
                    "orderType": "Limit",
                    "price": "30000",
                    "qty": "0.1",
                    "cumExecQty": "0",
                    "orderStatus": "New",
                    "reduceOnly": False,
                    "closeOnTrigger": False,
                    "createdTime": "1700000000000",
                    "updatedTime": "1700000001000",
                }
            ],
        }

    def test_event_topic_preserved(self) -> None:
        result = self.mapper.ws_order_to_event(self._make_payload())
        assert result["topic"] == "order"

    def test_event_has_events_list(self) -> None:
        result = self.mapper.ws_order_to_event(self._make_payload())
        assert isinstance(result["events"], list)
        assert len(result["events"]) == 1

    def test_event_price_is_decimal(self) -> None:
        result = self.mapper.ws_order_to_event(self._make_payload())
        evt = result["events"][0]
        assert isinstance(evt["price"], Decimal)
        assert evt["price"] == Decimal("30000")

    def test_event_reduce_only_preserved(self) -> None:
        payload = self._make_payload()
        payload["data"][0]["reduceOnly"] = True
        result = self.mapper.ws_order_to_event(payload)
        assert result["events"][0]["reduce_only"] is True

    def test_event_close_on_trigger_preserved(self) -> None:
        payload = self._make_payload()
        payload["data"][0]["closeOnTrigger"] = True
        result = self.mapper.ws_order_to_event(payload)
        assert result["events"][0]["close_on_trigger"] is True


# ---------------------------------------------------------------------------
# WS execution / fill parser
# ---------------------------------------------------------------------------

class TestWsExecutionToFill:
    def setup_method(self) -> None:
        self.mapper = OrderMapper()

    def _make_execution_payload(self) -> dict:
        return {
            "topic": "execution",
            "data": [
                {
                    "execId": "exec-001",
                    "orderLinkId": "TN-260605-MOMO-TEST1234-abc123",
                    "symbol": "BTCUSDT",
                    "side": "Buy",
                    "execQty": "0.05",
                    "execPrice": "30100",
                    "execFee": "1.5",
                    "feeCurrency": "USDT",
                    "isMaker": False,
                    "execTime": "1700000005000",
                }
            ],
        }

    def test_fill_exchange_exec_id(self) -> None:
        fill = self.mapper.ws_execution_to_fill(self._make_execution_payload())
        assert fill.exchange_exec_id == "exec-001"

    def test_fill_qty_is_decimal(self) -> None:
        fill = self.mapper.ws_execution_to_fill(self._make_execution_payload())
        assert isinstance(fill.qty, Decimal)
        assert fill.qty == Decimal("0.05")

    def test_fill_price_is_decimal(self) -> None:
        fill = self.mapper.ws_execution_to_fill(self._make_execution_payload())
        assert fill.price == Decimal("30100")

    def test_fill_fee_is_decimal(self) -> None:
        fill = self.mapper.ws_execution_to_fill(self._make_execution_payload())
        assert fill.fee == Decimal("1.5")

    def test_fill_side(self) -> None:
        fill = self.mapper.ws_execution_to_fill(self._make_execution_payload())
        assert fill.side == OrderSide.BUY


# ---------------------------------------------------------------------------
# REST position parser
# ---------------------------------------------------------------------------

class TestRestPositionToModel:
    def setup_method(self) -> None:
        self.mapper = OrderMapper()

    def _make_position_data(self) -> dict:
        return {
            "symbol": "BTCUSDT",
            "category": "linear",
            "side": "Buy",
            "size": "0.5",
            "avgPrice": "29500",
            "markPrice": "29600",
            "liqPrice": "25000",
            "unrealisedPnl": "50",
            "cumRealisedPnl": "20",
            "leverage": "5",
            "tradeMode": "cross",
            "updatedTime": "1700000000000",
        }

    def test_position_symbol(self) -> None:
        pos = self.mapper.rest_position_to_model(self._make_position_data())
        assert pos.symbol == "BTCUSDT"

    def test_position_size_decimal(self) -> None:
        pos = self.mapper.rest_position_to_model(self._make_position_data())
        assert pos.size == Decimal("0.5")

    def test_position_entry_price(self) -> None:
        pos = self.mapper.rest_position_to_model(self._make_position_data())
        assert pos.entry_price == Decimal("29500")

    def test_position_side(self) -> None:
        pos = self.mapper.rest_position_to_model(self._make_position_data())
        assert pos.side == OrderSide.BUY

    def test_position_leverage(self) -> None:
        pos = self.mapper.rest_position_to_model(self._make_position_data())
        assert pos.leverage == Decimal("5")

    def test_position_unrealised_pnl(self) -> None:
        pos = self.mapper.rest_position_to_model(self._make_position_data())
        assert pos.unrealised_pnl == Decimal("50")
