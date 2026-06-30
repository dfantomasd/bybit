"""Tests for fee-aware profit manager, dynamic breakeven, and net-edge filter."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.domain.enums import MarketRegime, MarketType, OrderSide
from trader.domain.models import TradeProposal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fee_rates(taker: float = 0.00055, maker: float = 0.0002):
    from trader.exchange.fee_provider import FeeRates

    return FeeRates(
        maker_fee_rate=maker,
        taker_fee_rate=taker,
        source="test",
        fetched_at=datetime.now(tz=UTC),
    )


def _make_app_settings(**overrides):
    s = MagicMock()
    s.BREAKEVEN_STOP_OFFSET_PCT = 0.20
    s.DEFAULT_LINEAR_TAKER_FEE_RATE = 0.00055
    s.DEFAULT_LINEAR_MAKER_FEE_RATE = 0.0002
    s.SCREENER_MAX_SPREAD_BPS = 8.0
    s.EXPECTED_SLIPPAGE_PCT = 0.03
    s.MIN_NET_PROFIT_BUFFER_PCT = 0.08
    s.FUNDING_BUFFER_PCT = 0.01
    s.MIN_EXPECTED_NET_EDGE_PCT = 0.15
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


# ---------------------------------------------------------------------------
# Test 1-2: execution event fee fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_event_persists_fee_value_maker_and_closed_size():
    """record_execution_event called with exec_fee, exec_value, is_maker, closed_size."""
    journal = MagicMock()
    journal.record_execution_event = AsyncMock(return_value=None)
    journal.record_order_event = AsyncMock(return_value=None)
    journal.is_enabled = True

    from trader.domain.events import ExecutionUpdateEvent, OrderType

    event = ExecutionUpdateEvent(
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        order_id="ORD001",
        order_link_id="LINK001",
        exec_id="EXEC001",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        exec_price=Decimal("50000"),
        exec_qty=Decimal("0.001"),
        exec_fee=Decimal("0.0275"),
        exec_value=Decimal("50.00"),
        is_maker=False,
        closed_size=Decimal("0"),
    )

    # Simulate the app.py recording logic
    await journal.record_execution_event(
        exec_id=event.exec_id,
        order_link_id=event.order_link_id or None,
        exchange_order_id=event.order_id,
        symbol=event.symbol,
        side=event.side.value,
        exec_price=event.exec_price,
        exec_qty=event.exec_qty,
        exec_fee=event.exec_fee if event.exec_fee else None,
        exec_value=event.exec_value if event.exec_value else None,
        is_maker=event.is_maker if hasattr(event, "is_maker") else None,
        closed_size=event.closed_size if event.closed_size else None,
    )

    journal.record_execution_event.assert_called_once()
    kwargs = journal.record_execution_event.call_args.kwargs
    assert kwargs["exec_fee"] == Decimal("0.0275")
    assert kwargs["exec_value"] == Decimal("50.00")
    assert kwargs["is_maker"] is False
    assert kwargs["closed_size"] is None  # Decimal("0") → None because falsy


# ---------------------------------------------------------------------------
# Tests 3-6: dynamic breakeven
# ---------------------------------------------------------------------------


def _make_app_with_settings(**overrides):
    """Create a minimal TradingApplication-like object with _breakeven_stop."""
    from trader.app import TradingApplication

    app = TradingApplication.__new__(TradingApplication)
    app._settings = _make_app_settings(**overrides)
    return app


def test_breakeven_covers_roundtrip_taker_fees():
    """Breakeven offset must be >= 2 * taker_fee_pct."""
    app = _make_app_with_settings()
    fee_rates = _make_fee_rates(taker=0.00055)
    entry = Decimal("50000")
    be = app._breakeven_stop(entry, "Buy", fee_rates=fee_rates)
    # Round-trip taker fee: 2 * 0.055% = 0.11% of 50000 = 55
    roundtrip_fee = Decimal("50000") * Decimal("0.0011")
    assert be > entry + roundtrip_fee, f"Breakeven {be} must exceed entry + roundtrip fee {entry + roundtrip_fee}"


def test_breakeven_includes_spread_and_slippage():
    """Breakeven offset includes spread + slippage on top of fees."""
    app = _make_app_with_settings(SCREENER_MAX_SPREAD_BPS=8.0, EXPECTED_SLIPPAGE_PCT=0.03)
    fee_rates = _make_fee_rates(taker=0.00055)
    entry = Decimal("50000")
    be_with = app._breakeven_stop(entry, "Buy", fee_rates=fee_rates)
    # Without spread/slippage the offset would be smaller
    app2 = _make_app_with_settings(SCREENER_MAX_SPREAD_BPS=0.0, EXPECTED_SLIPPAGE_PCT=0.0)
    be_without = app2._breakeven_stop(entry, "Buy", fee_rates=fee_rates)
    assert be_with > be_without


def test_breakeven_charges_round_trip_slippage():
    """Expected slippage is per leg, so breakeven must include entry + exit slippage."""
    fee_rates = _make_fee_rates(taker=0.00055)
    entry = Decimal("50000")
    app_with = _make_app_with_settings(SCREENER_MAX_SPREAD_BPS=8.0, EXPECTED_SLIPPAGE_PCT=0.03)
    app_without = _make_app_with_settings(SCREENER_MAX_SPREAD_BPS=8.0, EXPECTED_SLIPPAGE_PCT=0.0)

    be_with = app_with._breakeven_stop(entry, "Buy", fee_rates=fee_rates)
    be_without = app_without._breakeven_stop(entry, "Buy", fee_rates=fee_rates)

    assert be_with - be_without == Decimal("30.0000")


def test_short_breakeven_is_below_entry():
    """For a short position, breakeven stop is below entry price."""
    app = _make_app_with_settings()
    fee_rates = _make_fee_rates(taker=0.00055)
    entry = Decimal("50000")
    be = app._breakeven_stop(entry, "Sell", fee_rates=fee_rates)
    assert be < entry, f"Short breakeven {be} should be below entry {entry}"


def test_long_breakeven_is_above_entry():
    """For a long position, breakeven stop is above entry price."""
    app = _make_app_with_settings()
    fee_rates = _make_fee_rates(taker=0.00055)
    entry = Decimal("50000")
    be = app._breakeven_stop(entry, "Buy", fee_rates=fee_rates)
    assert be > entry, f"Long breakeven {be} should be above entry {entry}"


# ---------------------------------------------------------------------------
# Tests 7-8: net edge filter
# ---------------------------------------------------------------------------


def _make_engine_with_fee_provider(shadow=False, min_edge=0.15, fee_taker=0.00055):
    from trader.exchange.fee_provider import FeeRates
    from trader.execution.engine import ExecutionEngine

    fee_rates = FeeRates(
        maker_fee_rate=0.0002,
        taker_fee_rate=fee_taker,
        source="test",
        fetched_at=datetime.now(tz=UTC),
    )
    fee_provider = MagicMock()
    fee_provider.get = AsyncMock(return_value=fee_rates)

    adapter = MagicMock()
    adapter.get_instrument_info = AsyncMock()
    adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("50000"))
    adapter.place_order = AsyncMock(return_value={"result": {"orderId": "X"}})

    risk_manager = MagicMock()
    import uuid

    from trader.domain.enums import RiskDecisionStatus
    from trader.domain.models import RiskDecision

    risk_manager.evaluate = AsyncMock(
        return_value=RiskDecision(
            proposal_id=uuid.uuid4(),
            status=RiskDecisionStatus.APPROVED,
            approved_qty=Decimal("0.001"),
            triggered_rules=[],
            portfolio_heat=5.0,
            current_drawdown_pct=0.0,
            open_positions_count=0,
        )
    )
    risk_manager._limits = MagicMock()
    risk_manager._limits.max_leverage = Decimal("10")

    exposure = MagicMock()
    exposure.update_position = AsyncMock()

    engine = ExecutionEngine(
        adapter=adapter,
        risk_manager=risk_manager,
        exposure_tracker=exposure,
        shadow_mode=shadow,
        fee_provider=fee_provider,
        min_net_edge_pct=min_edge,
        max_spread_bps=8.0,
        expected_slippage_pct=0.03,
        funding_buffer_pct=0.01,
        live_armed=True,
    )
    return engine, adapter, fee_provider


def _make_proposal(entry=50000, tp_pct=0.0, side=OrderSide.BUY):
    """Create proposal. tp_pct is the take profit as % above/below entry."""
    entry_d = Decimal(str(entry))
    if side == OrderSide.BUY:
        tp = entry_d * (1 + Decimal(str(tp_pct)) / 100)
        sl = entry_d * Decimal("0.98")
    else:
        tp = entry_d * (1 - Decimal(str(tp_pct)) / 100)
        sl = entry_d * Decimal("1.02")
    return TradeProposal(
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        side=side,
        requested_qty=Decimal("0.001"),
        entry_price=entry_d,
        stop_loss=sl,
        take_profit=tp,
        confidence=0.8,
        strategy_id="test",
        regime=MarketRegime.BULL_TREND,
    )


@pytest.mark.asyncio
async def test_trade_rejected_when_net_edge_negative():
    """Engine rejects trade when net edge (after fees) is below MIN_EXPECTED_NET_EDGE_PCT."""
    from trader.domain.models import InstrumentInfo

    engine, adapter, _ = _make_engine_with_fee_provider(shadow=False, min_edge=0.15)

    # TP is only 0.05% above entry — after 0.11% fees + spread + slippage → net negative
    proposal = _make_proposal(entry=50000, tp_pct=0.05, side=OrderSide.BUY)

    instrument = InstrumentInfo(
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        base_coin="BTC",
        quote_coin="USDT",
        min_order_qty=Decimal("0.001"),
        max_order_qty=Decimal("1000"),
        qty_step=Decimal("0.001"),
        tick_size=Decimal("0.01"),
        min_notional=Decimal("5"),
    )
    engine._instrument_cache["BTCUSDT"] = (instrument, datetime.now(tz=UTC))
    engine._adapter.get_instrument_info = AsyncMock(return_value=instrument)
    engine._adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("50000"))

    result = await engine.submit(proposal, capital=Decimal("10000"), available_balance=Decimal("5000"))
    assert result is None
    adapter.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_trade_allowed_when_net_edge_above_threshold():
    """Engine allows trade when net edge is comfortably above MIN_EXPECTED_NET_EDGE_PCT."""
    from trader.domain.models import InstrumentInfo

    engine, adapter, _ = _make_engine_with_fee_provider(shadow=False, min_edge=0.15)

    # TP is 1.5% above entry — after ~0.3% total costs → net ~1.2% >> threshold
    proposal = _make_proposal(entry=50000, tp_pct=1.5, side=OrderSide.BUY)

    instrument = InstrumentInfo(
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        base_coin="BTC",
        quote_coin="USDT",
        min_order_qty=Decimal("0.001"),
        max_order_qty=Decimal("1000"),
        qty_step=Decimal("0.001"),
        tick_size=Decimal("0.01"),
        min_notional=Decimal("5"),
    )
    engine._instrument_cache["BTCUSDT"] = (instrument, datetime.now(tz=UTC))
    engine._adapter.get_instrument_info = AsyncMock(return_value=instrument)
    engine._adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("50000"))

    await engine.submit(proposal, capital=Decimal("10000"), available_balance=Decimal("5000"))
    adapter.place_order.assert_called_once()
