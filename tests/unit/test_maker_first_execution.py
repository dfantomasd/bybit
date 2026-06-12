"""Tests for MAKER_FIRST entry execution with escalation."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.domain.enums import (
    MarketRegime,
    MarketType,
    OrderSide,
    OrderType,
    RiskDecisionStatus,
)
from trader.domain.models import (
    InstrumentInfo,
    Position,
    RiskDecision,
    TradeProposal,
)
from trader.execution.engine import ExecutionEngine

_SYMBOL = "BTCUSDT"


def _instrument_info() -> InstrumentInfo:
    return InstrumentInfo(
        symbol=_SYMBOL,
        market_type=MarketType.LINEAR,
        base_coin="BTC",
        quote_coin="USDT",
        min_order_qty=Decimal("0.001"),
        max_order_qty=Decimal("100"),
        qty_step=Decimal("0.001"),
        tick_size=Decimal("0.5"),
        min_notional=Decimal("5"),
    )


def _proposal(side: OrderSide = OrderSide.BUY) -> TradeProposal:
    is_buy = side == OrderSide.BUY
    return TradeProposal(
        strategy_id="test_strategy",
        symbol=_SYMBOL,
        market_type=MarketType.LINEAR,
        side=side,
        requested_qty=Decimal("0.01"),
        entry_price=Decimal("50000"),
        stop_loss=Decimal("49000") if is_buy else Decimal("51000"),
        take_profit=Decimal("52000") if is_buy else Decimal("48000"),
        confidence=0.70,
        regime=MarketRegime.BULL_TREND if is_buy else MarketRegime.BEAR_TREND,
    )


def _position(side: OrderSide = OrderSide.BUY) -> Position:
    return Position(
        symbol=_SYMBOL,
        market_type=MarketType.LINEAR,
        side=side,
        size=Decimal("0.01"),
        entry_price=Decimal("50000"),
    )


def _make_engine(
    *,
    maker_allow_escalation: bool = True,
    imbalance_provider=None,
    maker_timeout_s: float = 0.5,
) -> ExecutionEngine:
    adapter = MagicMock()
    adapter.get_positions = AsyncMock(return_value=[])
    adapter.get_instrument_info = AsyncMock(return_value=_instrument_info())
    adapter.place_order = AsyncMock(return_value={"result": {"orderId": "ex-1"}})
    adapter.cancel_order = AsyncMock(return_value={"result": {}})
    adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("50000"))
    adapter.get_best_bid_ask = AsyncMock(return_value=(Decimal("50000"), Decimal("50010")))
    adapter.get_open_orders = AsyncMock(return_value=[])

    risk_manager = MagicMock()

    def make_decision(proposal, **kwargs):
        return RiskDecision(
            proposal_id=proposal.proposal_id,
            status=RiskDecisionStatus.APPROVED,
            approved_qty=proposal.requested_qty,
            portfolio_heat=0.05,
            current_drawdown_pct=0.0,
            open_positions_count=0,
        )

    risk_manager.evaluate = AsyncMock(side_effect=make_decision)

    exposure = MagicMock()
    exposure.update_position = AsyncMock()
    exposure.remove_position = AsyncMock()

    return ExecutionEngine(
        adapter=adapter,
        risk_manager=risk_manager,
        exposure_tracker=exposure,
        shadow_mode=False,
        cooldown_s=0,
        entry_order_mode="MAKER_FIRST",
        maker_timeout_s=maker_timeout_s,
        maker_ttl_s=maker_timeout_s,
        maker_allow_escalation=maker_allow_escalation,
        imbalance_provider=imbalance_provider,
    )


class TestMakerFirstExecution:
    def test_unsupported_mode_still_rejected(self) -> None:
        with pytest.raises(ValueError):
            ExecutionEngine(
                adapter=MagicMock(),
                risk_manager=MagicMock(),
                exposure_tracker=MagicMock(),
                entry_order_mode="POST_ONLY_LIMIT",
            )

    @pytest.mark.asyncio
    async def test_maker_order_is_post_only_limit_at_improved_bid(self) -> None:
        engine = _make_engine()
        # Immediate fill: order is gone from open orders, position appears
        engine._adapter.get_positions = AsyncMock(return_value=[_position()])

        decision = await engine.submit(_proposal(), Decimal("10000"), Decimal("10000"))

        assert decision is not None
        first_intent = engine._adapter.place_order.await_args_list[0].args[0]
        assert first_intent.order_type == OrderType.LIMIT
        assert first_intent.time_in_force == "PostOnly"
        # bid 50000 + tick 0.5 < ask 50010 → improved bid
        assert first_intent.price == Decimal("50000.5")
        assert engine.get_diag_counts()["maker_filled"] == 1

    @pytest.mark.asyncio
    async def test_sell_maker_prices_inside_ask(self) -> None:
        engine = _make_engine()
        engine._adapter.get_positions = AsyncMock(return_value=[_position(OrderSide.SELL)])

        decision = await engine.submit(_proposal(OrderSide.SELL), Decimal("10000"), Decimal("10000"))

        assert decision is not None
        first_intent = engine._adapter.place_order.await_args_list[0].args[0]
        assert first_intent.side == OrderSide.SELL
        assert first_intent.price == Decimal("50009.5")  # ask 50010 - tick 0.5

    @pytest.mark.asyncio
    async def test_timeout_escalates_to_market(self) -> None:
        engine = _make_engine()
        # Order rests in the book the whole time, never fills
        engine._adapter.get_open_orders = AsyncMock(
            side_effect=lambda *_a, **_k: [
                {"orderLinkId": engine._adapter.place_order.await_args_list[0].args[0].order_link_id}
            ]
        )

        decision = await engine.submit(_proposal(), Decimal("10000"), Decimal("10000"))

        assert decision is not None
        engine._adapter.cancel_order.assert_awaited_once()
        assert engine._adapter.place_order.await_count == 2
        second_intent = engine._adapter.place_order.await_args_list[1].args[0]
        first_intent = engine._adapter.place_order.await_args_list[0].args[0]
        assert second_intent.order_type == OrderType.MARKET
        assert second_intent.order_link_id == first_intent.order_link_id[:35] + "E"
        assert engine.get_diag_counts()["maker_escalated"] == 1

    @pytest.mark.asyncio
    async def test_escalation_blocked_by_imbalance_against(self) -> None:
        engine = _make_engine(imbalance_provider=lambda _s: -0.5)  # book against BUY
        engine._adapter.get_open_orders = AsyncMock(
            side_effect=lambda *_a, **_k: [
                {"orderLinkId": engine._adapter.place_order.await_args_list[0].args[0].order_link_id}
            ]
        )

        decision = await engine.submit(_proposal(), Decimal("10000"), Decimal("10000"))

        assert decision is None
        engine._adapter.cancel_order.assert_awaited_once()
        assert engine._adapter.place_order.await_count == 1  # no taker order
        assert engine.get_diag_counts()["maker_aborted"] == 1
        assert not engine.has_pending_entries()  # slot released

    @pytest.mark.asyncio
    async def test_escalation_blocked_by_price_drift(self) -> None:
        engine = _make_engine()
        engine._adapter.get_open_orders = AsyncMock(
            side_effect=lambda *_a, **_k: [
                {"orderLinkId": engine._adapter.place_order.await_args_list[0].args[0].order_link_id}
            ]
        )
        # Price ran 1% away from the maker price → abort instead of taker
        engine._adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("50500"))

        decision = await engine.submit(_proposal(), Decimal("10000"), Decimal("10000"))

        assert decision is None
        assert engine._adapter.place_order.await_count == 1
        assert engine.get_diag_counts()["maker_aborted"] == 1

    @pytest.mark.asyncio
    async def test_escalation_disabled_aborts_after_ttl(self) -> None:
        engine = _make_engine(maker_allow_escalation=False)
        engine._adapter.get_open_orders = AsyncMock(
            side_effect=lambda *_a, **_k: [
                {"orderLinkId": engine._adapter.place_order.await_args_list[0].args[0].order_link_id}
            ]
        )

        decision = await engine.submit(_proposal(), Decimal("10000"), Decimal("10000"))

        assert decision is None
        engine._adapter.cancel_order.assert_awaited_once()
        assert engine._adapter.place_order.await_count == 1
        assert engine.get_diag_counts()["maker_aborted"] == 1

    @pytest.mark.asyncio
    async def test_postonly_rejected_no_position_goes_to_escalation(self) -> None:
        # Order vanishes immediately (exchange cancelled the crossing PostOnly)
        # and no position exists → escalation path (allowed here)
        engine = _make_engine()
        decision = await engine.submit(_proposal(), Decimal("10000"), Decimal("10000"))

        assert decision is not None
        assert engine._adapter.place_order.await_count == 2
        assert engine.get_diag_counts()["maker_escalated"] == 1

    @pytest.mark.asyncio
    async def test_partial_fill_after_cancel_counts_as_filled(self) -> None:
        engine = _make_engine()
        link_holder: dict = {}

        def open_orders(*_a, **_k):
            link_holder["id"] = engine._adapter.place_order.await_args_list[0].args[0].order_link_id
            return [{"orderLinkId": link_holder["id"]}]

        engine._adapter.get_open_orders = AsyncMock(side_effect=open_orders)
        # After cancel, the partial fill shows up as a position
        positions: list = []
        engine._adapter.get_positions = AsyncMock(side_effect=lambda *_a, **_k: positions)
        engine._adapter.cancel_order = AsyncMock(
            side_effect=lambda *_a, **_k: positions.append(_position()) or {"result": {}}
        )

        decision = await engine.submit(_proposal(), Decimal("10000"), Decimal("10000"))

        assert decision is not None
        assert engine._adapter.place_order.await_count == 1  # no escalation needed
        assert engine.get_diag_counts()["maker_filled"] == 1

    @pytest.mark.asyncio
    async def test_cancel_failed_order_still_live_fails_closed(self) -> None:
        engine = _make_engine()
        engine._adapter.get_open_orders = AsyncMock(
            side_effect=lambda *_a, **_k: [
                {"orderLinkId": engine._adapter.place_order.await_args_list[0].args[0].order_link_id}
            ]
        )
        engine._adapter.cancel_order = AsyncMock(side_effect=RuntimeError("network"))

        decision = await engine.submit(_proposal(), Decimal("10000"), Decimal("10000"))

        assert decision is None
        assert engine._adapter.place_order.await_count == 1  # no taker on top of a live limit
        # The pending slot must stay blocked until WS/reconciliation resolves it
        assert engine.has_pending_entries()
        assert _SYMBOL in engine._last_failure_at

    @pytest.mark.asyncio
    async def test_late_fill_during_escalation_gate_prevents_taker(self) -> None:
        engine = _make_engine()
        # Order vanishes immediately without a position ("gone" → escalation path)
        positions: list = []
        engine._adapter.get_positions = AsyncMock(side_effect=lambda *_a, **_k: positions)
        # The fill lands while the escalation gate fetches the current price
        original_price_check = engine._adapter.get_conservative_market_price

        async def price_with_racing_fill(*a, **k):
            positions.append(_position())
            return await original_price_check(*a, **k)

        engine._adapter.get_conservative_market_price = AsyncMock(side_effect=price_with_racing_fill)

        decision = await engine.submit(_proposal(), Decimal("10000"), Decimal("10000"))

        assert decision is not None
        assert engine._adapter.place_order.await_count == 1  # no doubled entry
        assert engine.get_diag_counts()["maker_filled"] == 1
        assert engine.get_diag_counts()["maker_escalated"] == 0

    @pytest.mark.asyncio
    async def test_no_quote_aborts_entry(self) -> None:
        engine = _make_engine()
        engine._adapter.get_best_bid_ask = AsyncMock(side_effect=RuntimeError("no ticker"))

        decision = await engine.submit(_proposal(), Decimal("10000"), Decimal("10000"))

        assert decision is None
        engine._adapter.place_order.assert_not_awaited()
        assert engine.get_diag_counts()["maker_aborted"] == 1
        assert not engine.has_pending_entries()

    @pytest.mark.asyncio
    async def test_market_mode_unaffected(self) -> None:
        engine = _make_engine()
        engine._entry_order_mode = "MARKET"
        engine._adapter.get_positions = AsyncMock(return_value=[_position()])

        decision = await engine.submit(_proposal(), Decimal("10000"), Decimal("10000"))

        assert decision is not None
        assert engine._adapter.place_order.await_count == 1
        intent = engine._adapter.place_order.await_args.args[0]
        assert intent.order_type == OrderType.MARKET
        engine._adapter.get_best_bid_ask.assert_not_awaited()
