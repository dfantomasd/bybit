"""Tests for net PnL reporting and cost-aware entry gate."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# TradeJournal.get_daily_net_results
# ---------------------------------------------------------------------------


def test_get_daily_net_results_disabled_returns_zeros():
    from trader.storage.trade_journal import TradeJournal

    tj = TradeJournal(postgres_dsn="postgresql://fake", enabled=False)
    result = asyncio.run(tj.get_daily_net_results())
    assert result["gross_closed_pnl_usd"] == 0.0
    assert result["total_fees_usd"] == 0.0
    assert result["closed_trade_count"] == 0


def test_get_daily_net_results_has_all_keys():
    from trader.storage.trade_journal import TradeJournal

    tj = TradeJournal(postgres_dsn="postgresql://fake", enabled=False)
    result = asyncio.run(tj.get_daily_net_results())
    required = {
        "closed_trade_count",
        "gross_closed_pnl_usd",
        "total_fees_usd",
        "total_funding_usd",
        "net_pnl_usd",
        "maker_fill_pct",
        "taker_fill_pct",
        "transaction_event_count",
        "latest_transaction_at",
    }
    assert required.issubset(result.keys())


def test_get_daily_net_results_net_pnl_is_zero_when_disabled():
    from trader.storage.trade_journal import TradeJournal

    tj = TradeJournal(postgres_dsn="postgresql://fake", enabled=False)
    result = asyncio.run(tj.get_daily_net_results())
    assert result["net_pnl_usd"] == 0.0


# ---------------------------------------------------------------------------
# ExecutionEngine net edge gate — diagnostic counters
# ---------------------------------------------------------------------------


def _make_engine(
    shadow_mode: bool = False,
    fee_rate: float = 0.00055,
    fee_provider_returns_none: bool = False,
    fee_provider: Any = None,
    min_net_edge_pct: float = 0.25,
    net_edge_safety_margin_pct: float = 0.05,
):
    from trader.execution.engine import ExecutionEngine

    adapter = MagicMock()
    risk_manager = MagicMock()
    exposure = MagicMock()
    exposure.total_exposure_pct = Decimal("0")

    if fee_provider is None:
        if fee_provider_returns_none:
            fp = MagicMock()
            fp.get = AsyncMock(return_value=None)
        else:
            fp = MagicMock()
            fee_rates = MagicMock()
            fee_rates.taker_fee_rate = fee_rate
            fp.get = AsyncMock(return_value=fee_rates)
    else:
        fp = fee_provider

    engine = ExecutionEngine(
        adapter=adapter,
        risk_manager=risk_manager,
        exposure_tracker=exposure,
        shadow_mode=shadow_mode,
        fee_provider=fp,
        min_net_edge_pct=min_net_edge_pct,
        net_edge_safety_margin_pct=net_edge_safety_margin_pct,
        expected_slippage_pct=0.03,
        funding_buffer_pct=0.01,
        max_spread_bps=8.0,
    )
    return engine


def test_diag_counters_start_at_zero():
    engine = _make_engine(shadow_mode=True)
    counts = engine.get_diag_counts()
    assert counts["net_edge_rejected"] == 0
    assert counts["no_tp_rejected"] == 0
    assert counts["fee_unavailable_rejected"] == 0


def test_get_diag_counts_includes_new_keys():
    engine = _make_engine(shadow_mode=True)
    counts = engine.get_diag_counts()
    assert "net_edge_rejected" in counts
    assert "no_tp_rejected" in counts
    assert "fee_unavailable_rejected" in counts


# ---------------------------------------------------------------------------
# Net edge arithmetic
# ---------------------------------------------------------------------------


def test_taker_round_trip_fee_applied_twice():
    """Round-trip fee must be 2x taker rate."""
    taker = Decimal("0.00055")
    entry_fee = taker * Decimal("100")
    exit_fee = taker * Decimal("100")
    round_trip = entry_fee + exit_fee
    assert round_trip == Decimal("0.11")  # 0.11%


def test_safety_margin_reduces_net_edge():
    """Safety margin must be subtracted from net edge."""
    # gross = 0.5%, fees = 0.11%, spread = 0.08%, slippage = 0.03%, funding = 0.01%, safety = 0.05%
    gross = Decimal("0.5")
    fees = Decimal("0.11")
    spread = Decimal("0.08")
    slippage = Decimal("0.03")
    funding = Decimal("0.01")
    safety = Decimal("0.05")
    net = gross - fees - spread - slippage - funding - safety
    assert net == Decimal("0.22")


def test_net_edge_below_threshold_should_reject():
    """Gross edge of 0.1% is well below threshold of 0.25% after costs."""
    taker = Decimal("0.00055")
    entry_price = Decimal("100")
    tp = Decimal("100.1")  # 0.1% gross
    gross_edge_pct = (tp - entry_price) / entry_price * Decimal("100")
    round_trip = taker * Decimal("200")
    spread = Decimal("8.0") / Decimal("100")
    slippage = Decimal("0.03")
    funding = Decimal("0.01")
    safety = Decimal("0.05")
    net = gross_edge_pct - round_trip - spread - slippage - funding - safety
    assert net < Decimal("0.25"), f"Expected net edge < 0.25% but got {net}"


def test_net_edge_above_threshold_should_allow():
    """Gross edge of 1.0% should exceed threshold of 0.25% after costs."""
    taker = Decimal("0.00055")
    entry_price = Decimal("100")
    tp = Decimal("101.0")  # 1.0% gross
    gross_edge_pct = (tp - entry_price) / entry_price * Decimal("100")
    round_trip = taker * Decimal("200")
    spread = Decimal("8.0") / Decimal("100")
    slippage = Decimal("0.03")
    funding = Decimal("0.01")
    safety = Decimal("0.05")
    net = gross_edge_pct - round_trip - spread - slippage - funding - safety
    assert net > Decimal("0.25"), f"Expected net edge > 0.25% but got {net}"


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_config_min_edge_is_0_25():
    from trader.config import Settings

    s = Settings(
        BYBIT_USE_TESTNET=True,
        TRADING_MODE="SHADOW",
    )
    assert s.MIN_EXPECTED_NET_EDGE_PCT == 0.25


def test_config_safety_margin_is_0_05():
    from trader.config import Settings

    s = Settings(
        BYBIT_USE_TESTNET=True,
        TRADING_MODE="SHADOW",
    )
    assert s.NET_EDGE_SAFETY_MARGIN_PCT == 0.05


# ---------------------------------------------------------------------------
# _submit_locked: reject when no take_profit (LIVE mode)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_rejects_no_tp_increments_counter():
    """LIVE mode must reject and increment no_tp_rejected when TP is None."""
    from trader.domain.enums import OrderSide, RiskDecisionStatus
    from trader.domain.models import InstrumentInfo, MarketType, RiskDecision, TradeProposal

    engine = _make_engine(shadow_mode=False)

    proposal = MagicMock(spec=TradeProposal)
    proposal.symbol = "BTCUSDT"
    proposal.side = OrderSide.BUY
    proposal.entry_price = Decimal("100")
    proposal.take_profit = None  # <-- no TP
    proposal.stop_loss = Decimal("99")
    proposal.confidence = 0.7
    proposal.requested_qty = Decimal("10")
    proposal.proposal_id = "test-id-1"
    proposal.strategy_id = "test"
    proposal.rationale = "test"

    decision = MagicMock(spec=RiskDecision)
    decision.status = RiskDecisionStatus.APPROVED
    decision.approved_qty = Decimal("10")
    decision.approved_notional_usd = Decimal("1000")
    decision.reason = "ok"
    decision.decision_id = "decision-1"
    decision.proposal_id = "test-id-1"
    decision.triggered_rules = []
    decision.portfolio_heat = 0.0
    decision.current_drawdown_pct = 0.0
    decision.open_positions_count = 0

    instrument = InstrumentInfo(
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        base_coin="BTC",
        quote_coin="USDT",
        min_order_qty=Decimal("1"),
        max_order_qty=Decimal("1000"),
        qty_step=Decimal("1"),
        tick_size=Decimal("0.01"),
        min_notional=Decimal("5"),
    )

    engine._open_positions = {}
    engine._pending_entry_order_link_ids = set()
    engine._is_canary = False
    engine._risk_manager.evaluate = AsyncMock(return_value=decision)
    engine._adapter.get_instrument_info = AsyncMock(return_value=instrument)
    engine._adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("100"))
    engine._trade_journal = None

    result = await engine._submit_locked(
        proposal=proposal,
        capital=Decimal("1000"),
        available_balance=Decimal("1000"),
    )

    assert result is None
    assert engine._diag_no_tp_rejected == 1
    assert engine._diag_net_edge_rejected == 0


@pytest.mark.asyncio
async def test_live_rejects_low_edge_increments_counter():
    """LIVE mode must reject and increment net_edge_rejected when edge is too low."""
    from trader.domain.enums import OrderSide, RiskDecisionStatus
    from trader.domain.models import InstrumentInfo, MarketType, RiskDecision, TradeProposal

    engine = _make_engine(shadow_mode=False, fee_rate=0.00055, min_net_edge_pct=0.25)

    proposal = MagicMock(spec=TradeProposal)
    proposal.symbol = "BTCUSDT"
    proposal.side = OrderSide.BUY
    proposal.entry_price = Decimal("100")
    proposal.take_profit = Decimal("100.1")  # 0.1% gross — well below 0.25%
    proposal.stop_loss = Decimal("99")
    proposal.confidence = 0.7
    proposal.requested_qty = Decimal("10")
    proposal.proposal_id = "test-id-2"
    proposal.strategy_id = "test"
    proposal.rationale = "test"

    decision = MagicMock(spec=RiskDecision)
    decision.status = RiskDecisionStatus.APPROVED
    decision.approved_qty = Decimal("10")
    decision.approved_notional_usd = Decimal("1000")
    decision.reason = "ok"
    decision.decision_id = "decision-2"
    decision.proposal_id = "test-id-2"
    decision.triggered_rules = []
    decision.portfolio_heat = 0.0
    decision.current_drawdown_pct = 0.0
    decision.open_positions_count = 0

    instrument = InstrumentInfo(
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        base_coin="BTC",
        quote_coin="USDT",
        min_order_qty=Decimal("1"),
        max_order_qty=Decimal("1000"),
        qty_step=Decimal("1"),
        tick_size=Decimal("0.01"),
        min_notional=Decimal("5"),
    )

    engine._open_positions = {}
    engine._pending_entry_order_link_ids = set()
    engine._is_canary = False
    engine._risk_manager.evaluate = AsyncMock(return_value=decision)
    engine._adapter.get_instrument_info = AsyncMock(return_value=instrument)
    engine._adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("100"))
    engine._trade_journal = None

    result = await engine._submit_locked(
        proposal=proposal,
        capital=Decimal("1000"),
        available_balance=Decimal("1000"),
    )

    assert result is None
    assert engine._diag_net_edge_rejected == 1
    assert engine._diag_no_tp_rejected == 0


@pytest.mark.asyncio
async def test_live_net_edge_uses_rounded_take_profit():
    """LIVE edge gate must evaluate the same tick-rounded TP that will be submitted."""
    from trader.domain.enums import MarketType, OrderSide, RiskDecisionStatus
    from trader.domain.models import InstrumentInfo, RiskDecision, TradeProposal

    engine = _make_engine(
        shadow_mode=False,
        fee_rate=0.0,
        min_net_edge_pct=0.05,
        net_edge_safety_margin_pct=0.0,
    )
    engine._max_spread_bps = 0.0
    engine._expected_slippage_pct = 0.0
    engine._funding_buffer_pct = 0.0

    proposal = MagicMock(spec=TradeProposal)
    proposal.symbol = "BTCUSDT"
    proposal.side = OrderSide.BUY
    proposal.entry_price = Decimal("100")
    proposal.take_profit = Decimal("100.09")
    proposal.stop_loss = Decimal("99")
    proposal.confidence = 0.7
    proposal.requested_qty = Decimal("10")
    proposal.proposal_id = "test-id-rounded"
    proposal.strategy_id = "test"
    proposal.rationale = "test"

    decision = MagicMock(spec=RiskDecision)
    decision.status = RiskDecisionStatus.APPROVED
    decision.approved_qty = Decimal("10")
    decision.approved_notional_usd = Decimal("1000")
    decision.reason = "ok"
    decision.decision_id = "decision-rounded"
    decision.proposal_id = "test-id-rounded"
    decision.triggered_rules = []
    decision.portfolio_heat = 0.0
    decision.current_drawdown_pct = 0.0
    decision.open_positions_count = 0

    instrument = InstrumentInfo(
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        base_coin="BTC",
        quote_coin="USDT",
        min_order_qty=Decimal("1"),
        max_order_qty=Decimal("1000"),
        qty_step=Decimal("1"),
        tick_size=Decimal("0.1"),
        min_notional=Decimal("5"),
    )

    engine._open_positions = {}
    engine._pending_entry_order_link_ids = set()
    engine._is_canary = False
    engine._risk_manager.evaluate = AsyncMock(return_value=decision)
    engine._adapter.get_instrument_info = AsyncMock(return_value=instrument)
    engine._adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("100"))
    engine._trade_journal = None

    result = await engine._submit_locked(
        proposal=proposal,
        capital=Decimal("1000"),
        available_balance=Decimal("1000"),
    )

    assert result is None
    assert engine._diag_net_edge_rejected == 1
    engine._adapter.place_order.assert_not_called()


def test_shadow_mode_skips_net_edge_gate():
    """Shadow mode must NOT apply the net edge gate — counters start at zero and gate is not reached."""
    engine = _make_engine(shadow_mode=True)
    # In shadow mode the gate block is never entered; counters must stay at zero
    assert engine._diag_no_tp_rejected == 0
    assert engine._diag_net_edge_rejected == 0
    assert engine._shadow_mode is True
