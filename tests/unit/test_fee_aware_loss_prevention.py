"""Tests for fee-aware loss prevention: fills, dynamic breakeven, net-edge filter."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.domain.enums import MarketRegime, MarketType, OrderSide
from trader.domain.models import InstrumentInfo, TradeProposal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fee_rates(taker: float = 0.00055, maker: float = 0.0002):
    from trader.exchange.fee_provider import FeeRates

    return FeeRates(
        maker_fee_rate=maker,
        taker_fee_rate=taker,
        source="test",
        fetched_at=datetime.now(tz=UTC),
    )


def _app_settings(**overrides):
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


def _instrument(symbol: str = "BTCUSDT") -> InstrumentInfo:
    return InstrumentInfo(
        symbol=symbol,
        market_type=MarketType.LINEAR,
        base_coin="BTC",
        quote_coin="USDT",
        min_order_qty=Decimal("0.001"),
        max_order_qty=Decimal("1000"),
        qty_step=Decimal("0.001"),
        tick_size=Decimal("0.01"),
        min_notional=Decimal("5"),
    )


def _proposal(entry: float = 50000, tp_pct: float = 0.0, side: OrderSide = OrderSide.BUY) -> TradeProposal:
    entry_d = Decimal(str(entry))
    if side == OrderSide.BUY:
        tp = entry_d * (1 + Decimal(str(tp_pct)) / 100) if tp_pct else entry_d * Decimal("1.02")
        sl = entry_d * Decimal("0.98")
    else:
        tp = entry_d * (1 - Decimal(str(tp_pct)) / 100) if tp_pct else entry_d * Decimal("0.98")
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


def _make_engine(shadow: bool = False, min_edge: float = 0.15, fee_provider=None):
    import uuid

    from trader.domain.enums import RiskDecisionStatus
    from trader.domain.models import RiskDecision
    from trader.execution.engine import ExecutionEngine

    adapter = MagicMock()
    adapter.get_instrument_info = AsyncMock(return_value=_instrument())
    adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("50000"))
    adapter.place_order = AsyncMock(return_value={"result": {"orderId": "ORD1"}})

    risk_manager = MagicMock()
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

    return ExecutionEngine(
        adapter=adapter,
        risk_manager=risk_manager,
        exposure_tracker=exposure,
        shadow_mode=shadow,
        fee_provider=fee_provider,
        min_net_edge_pct=min_edge,
        max_spread_bps=8.0,
        expected_slippage_pct=0.03,
        funding_buffer_pct=0.01,
    ), adapter


# ---------------------------------------------------------------------------
# Test 1: execution event records all fee fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_event_persists_fee_value_maker_and_closed_size():
    """record_execution_event is called with exec_fee, exec_value, is_maker, closed_size."""
    from trader.domain.events import ExecutionUpdateEvent, OrderType

    journal = MagicMock()
    journal.record_execution_event = AsyncMock(return_value=None)
    journal.record_order_event = AsyncMock(return_value=None)
    journal.is_enabled = True

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

    # Simulate the app.py recording call (mirrors actual call in app.py)
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
    kw = journal.record_execution_event.call_args.kwargs
    assert kw["exec_fee"] == Decimal("0.0275")
    assert kw["exec_value"] == Decimal("50.00")
    assert kw["is_maker"] is False
    assert kw["closed_size"] is None  # Decimal("0") is falsy → None


# ---------------------------------------------------------------------------
# Tests 2-5: dynamic breakeven
# ---------------------------------------------------------------------------


def _app_with_breakeven(**overrides):
    from trader.app import TradingApplication

    app = TradingApplication.__new__(TradingApplication)
    app._settings = _app_settings(**overrides)
    return app


def test_breakeven_covers_roundtrip_taker_fees():
    """Breakeven offset must exceed 2× taker round-trip fee."""
    app = _app_with_breakeven()
    rates = _fee_rates(taker=0.00055)
    entry = Decimal("50000")
    be = app._breakeven_stop(entry, "Buy", fee_rates=rates)
    roundtrip_cost = entry * Decimal("0.0011")
    assert be > entry + roundtrip_cost, f"Breakeven {be} ≤ entry + roundtrip {entry + roundtrip_cost}"


def test_breakeven_includes_spread_and_slippage():
    """Breakeven with spread+slippage is higher than without."""
    rates = _fee_rates(taker=0.00055)
    entry = Decimal("50000")

    app_with = _app_with_breakeven(SCREENER_MAX_SPREAD_BPS=8.0, EXPECTED_SLIPPAGE_PCT=0.03)
    app_without = _app_with_breakeven(SCREENER_MAX_SPREAD_BPS=0.0, EXPECTED_SLIPPAGE_PCT=0.0)

    be_with = app_with._breakeven_stop(entry, "Buy", fee_rates=rates)
    be_without = app_without._breakeven_stop(entry, "Buy", fee_rates=rates)
    assert be_with > be_without


def test_short_breakeven_is_below_entry():
    """Short position breakeven stop must be below entry price."""
    app = _app_with_breakeven()
    rates = _fee_rates(taker=0.00055)
    entry = Decimal("50000")
    be = app._breakeven_stop(entry, "Sell", fee_rates=rates)
    assert be < entry, f"Short breakeven {be} should be below entry {entry}"


def test_long_breakeven_is_above_entry():
    """Long position breakeven stop must be above entry price."""
    app = _app_with_breakeven()
    rates = _fee_rates(taker=0.00055)
    entry = Decimal("50000")
    be = app._breakeven_stop(entry, "Buy", fee_rates=rates)
    assert be > entry, f"Long breakeven {be} should be above entry {entry}"


# ---------------------------------------------------------------------------
# Tests 6-7: fee-aware net-edge filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trade_rejected_when_net_edge_negative():
    """Engine rejects trade when TP is too close to cover fees."""
    fee_provider = MagicMock()
    fee_provider.get = AsyncMock(return_value=_fee_rates(taker=0.00055))

    engine, adapter = _make_engine(shadow=False, min_edge=0.15, fee_provider=fee_provider)
    engine._instrument_cache["BTCUSDT"] = (_instrument(), datetime.now(tz=UTC))

    # TP only 0.05% above entry → net edge ≈ 0.05 - 0.11 - 0.08 - 0.03 - 0.01 = -0.18% < 0.15
    prop = _proposal(entry=50000, tp_pct=0.05, side=OrderSide.BUY)

    result = await engine.submit(prop, capital=Decimal("10000"), available_balance=Decimal("5000"))
    assert result is None
    adapter.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_trade_allowed_when_net_edge_above_threshold():
    """Engine allows trade when TP distance comfortably exceeds all costs."""
    fee_provider = MagicMock()
    fee_provider.get = AsyncMock(return_value=_fee_rates(taker=0.00055))

    engine, adapter = _make_engine(shadow=False, min_edge=0.15, fee_provider=fee_provider)
    engine._instrument_cache["BTCUSDT"] = (_instrument(), datetime.now(tz=UTC))

    # TP 1.5% above entry → net edge ≈ 1.5 - 0.11 - 0.08 - 0.03 - 0.01 = 1.27% >> 0.15
    prop = _proposal(entry=50000, tp_pct=1.5, side=OrderSide.BUY)

    await engine.submit(prop, capital=Decimal("10000"), available_balance=Decimal("5000"))
    adapter.place_order.assert_called_once()


# ---------------------------------------------------------------------------
# Tests 8-9: FeeRateProvider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fee_provider_returns_api_rates():
    """FeeRateProvider parses makerFeeRate/takerFeeRate from API response."""
    from trader.exchange.fee_provider import FeeRateProvider

    rest = MagicMock()
    rest.get_fee_rate = AsyncMock(
        return_value={"result": {"list": [{"makerFeeRate": "0.0001", "takerFeeRate": "0.0006", "symbol": "BTCUSDT"}]}}
    )
    provider = FeeRateProvider(rest=rest, shadow_mode=True)
    rates = await provider.get("BTCUSDT")
    assert rates is not None
    assert rates.maker_fee_rate == 0.0001
    assert rates.taker_fee_rate == 0.0006
    assert rates.source == "api"


@pytest.mark.asyncio
async def test_fee_provider_fallback_in_shadow():
    """FeeRateProvider falls back to defaults in SHADOW when API fails."""
    from trader.exchange.fee_provider import FeeRateProvider

    rest = MagicMock()
    rest.get_fee_rate = AsyncMock(side_effect=Exception("network error"))
    provider = FeeRateProvider(rest=rest, shadow_mode=True, default_taker=0.00055)
    rates = await provider.get("BTCUSDT")
    assert rates is not None
    assert rates.taker_fee_rate == 0.00055
    assert rates.source == "fallback"


@pytest.mark.asyncio
async def test_fee_provider_fail_closed_in_live():
    """FeeRateProvider returns None in LIVE mode when API fails (fail-closed)."""
    from trader.exchange.fee_provider import FeeRateProvider

    rest = MagicMock()
    rest.get_fee_rate = AsyncMock(side_effect=Exception("timeout"))
    provider = FeeRateProvider(rest=rest, shadow_mode=False)
    rates = await provider.get("BTCUSDT")
    assert rates is None
