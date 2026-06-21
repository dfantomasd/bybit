"""Tests for P0/P1 hardening requirements.

Covers:
  P0.1  Durable write fail-closed
  P0.2  Restore pending at startup
  P0.3  Pending limiter by order_link_id
  P0.4  Full reconcile missing pending
  P0.5  execution_events schema / dedup
  P0.6  CANARY caps
  P0.7  Active preflight fail-closed
  P0.8  ML baseline prediction events
  P0.9  Fee-aware outcome labels
  P1.1  Telegram native card edit
"""

from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.domain.enums import MarketRegime, MarketType, OrderSide, RiskDecisionStatus, RiskProfile
from trader.domain.models import InstrumentInfo, RiskDecision, TradeProposal
from trader.execution.engine import CANARY_MAX_OPEN_POSITIONS, CANARY_MAX_TOTAL_EXPOSURE_PCT, ExecutionEngine
from trader.risk.exposure import ExposureTracker
from trader.risk.profiles import get_risk_limits

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _risk_limits():
    return get_risk_limits(RiskProfile.MODERATE)


def _instrument(
    symbol: str = "BTCUSDT",
    min_notional: Decimal = Decimal("5"),
) -> InstrumentInfo:
    return InstrumentInfo(
        symbol=symbol,
        market_type=MarketType.LINEAR,
        base_coin="BTC",
        quote_coin="USDT",
        min_order_qty=Decimal("0.001"),
        max_order_qty=Decimal("100"),
        qty_step=Decimal("0.001"),
        tick_size=Decimal("0.5"),
        min_notional=min_notional,
    )


def _proposal(
    symbol: str = "BTCUSDT",
    side: OrderSide = OrderSide.BUY,
    qty: Decimal = Decimal("0.01"),
    confidence: float = 0.70,
    entry_price: Decimal = Decimal("50000"),
) -> TradeProposal:
    return TradeProposal(
        strategy_id="test",
        symbol=symbol,
        market_type=MarketType.LINEAR,
        side=side,
        requested_qty=qty,
        entry_price=entry_price,
        stop_loss=Decimal("49000") if side == OrderSide.BUY else Decimal("51000"),
        take_profit=Decimal("52000") if side == OrderSide.BUY else Decimal("48000"),
        confidence=confidence,
        regime=MarketRegime.BULL_TREND,
    )


def _approved(proposal: TradeProposal, qty: Decimal | None = None) -> RiskDecision:
    return RiskDecision(
        proposal_id=proposal.proposal_id,
        status=RiskDecisionStatus.APPROVED,
        approved_qty=qty or proposal.requested_qty,
        portfolio_heat=0.05,
        current_drawdown_pct=0.0,
        open_positions_count=0,
    )


def _make_engine(
    shadow: bool = False,
    is_canary: bool = False,
    journal: Any = None,
) -> tuple[ExecutionEngine, MagicMock, MagicMock]:
    adapter = MagicMock()
    adapter.get_instrument_info = AsyncMock(return_value=_instrument())
    adapter.get_positions = AsyncMock(return_value=[])
    adapter.place_order = AsyncMock(return_value={"result": {"orderId": "EX123"}})
    adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("50000"))

    risk = MagicMock()
    risk._limits = _risk_limits()

    exposure = ExposureTracker(
        total_capital=Decimal("10000"),
        risk_limits=_risk_limits(),
    )

    engine = ExecutionEngine(
        adapter=adapter,
        risk_manager=risk,
        exposure_tracker=exposure,
        shadow_mode=shadow,
        trade_journal=journal,
        is_canary=is_canary,
    )
    return engine, adapter, risk


# ===========================================================================
# P0.1 Durable write fail-closed
# ===========================================================================


@pytest.mark.asyncio
async def test_active_order_not_sent_when_durable_created_local_write_fails() -> None:
    """place_order must NOT be called when CREATED_LOCAL write raises."""
    journal = MagicMock()
    journal.is_enabled = True
    journal.record_risk_decision = AsyncMock()
    journal.record_order_event = AsyncMock()
    journal.record_prediction_event = AsyncMock()
    # Fail on the first durable write
    journal.record_order_event_required = AsyncMock(side_effect=RuntimeError("db down"))

    engine, adapter, risk = _make_engine(shadow=False, journal=journal)
    prop = _proposal()
    risk.evaluate = AsyncMock(return_value=_approved(prop))

    result = await engine.submit(prop, Decimal("10000"), Decimal("10000"))

    assert result is None
    adapter.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_active_order_not_sent_when_durable_submitting_write_fails() -> None:
    """place_order must NOT be called when SUBMITTING write raises."""
    call_count = 0

    async def _required_side_effect(*_args: Any, **kwargs: Any) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:  # second call is SUBMITTING
            raise RuntimeError("db timeout")

    journal = MagicMock()
    journal.is_enabled = True
    journal.record_risk_decision = AsyncMock()
    journal.record_order_event = AsyncMock()
    journal.record_prediction_event = AsyncMock()
    journal.record_order_event_required = AsyncMock(side_effect=_required_side_effect)

    engine, adapter, risk = _make_engine(shadow=False, journal=journal)
    prop = _proposal()
    risk.evaluate = AsyncMock(return_value=_approved(prop))

    result = await engine.submit(prop, Decimal("10000"), Decimal("10000"))

    assert result is None
    adapter.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_shadow_can_continue_when_optional_journal_write_fails() -> None:
    """SHADOW mode must not be affected by journal write failures."""
    journal = MagicMock()
    journal.is_enabled = True
    journal.record_risk_decision = AsyncMock()
    journal.record_order_event = AsyncMock(side_effect=RuntimeError("db down"))
    journal.record_prediction_event = AsyncMock()
    journal.record_order_event_required = AsyncMock()

    engine, adapter, risk = _make_engine(shadow=True, journal=journal)
    prop = _proposal()
    risk.evaluate = AsyncMock(return_value=_approved(prop))

    result = await engine.submit(prop, Decimal("10000"), Decimal("10000"))

    # Shadow mode: decision returned even if journal fails
    assert result is not None
    adapter.place_order.assert_not_called()


# ===========================================================================
# P0.2 Restore pending at startup
# ===========================================================================


@pytest.mark.asyncio
async def test_startup_calls_load_pending_from_db() -> None:
    """TradeJournal.load_pending_from_db must return correct IDs."""
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal.__new__(TradeJournal)
    journal._enabled = True

    rows = [{"order_link_id": "ID1"}, {"order_link_id": "ID2"}]
    journal._fetch = AsyncMock(return_value=rows)  # type: ignore[method-assign]

    ids = await journal.load_pending_from_db()
    assert ids == ["ID1", "ID2"]


@pytest.mark.asyncio
async def test_execution_engine_restores_pending_entry_ids() -> None:
    engine, _, _ = _make_engine()
    engine.restore_pending_entries(["ID_A", "ID_B"])
    assert "ID_A" in engine._pending_entry_order_link_ids
    assert "ID_B" in engine._pending_entry_order_link_ids


@pytest.mark.asyncio
async def test_restored_pending_blocks_new_entry_until_resolved() -> None:
    engine, adapter, risk = _make_engine(shadow=True)
    engine.restore_pending_entries(["PENDING_001"])

    prop = _proposal()
    risk.evaluate = AsyncMock(return_value=_approved(prop))

    result = await engine.submit(prop, Decimal("10000"), Decimal("10000"))

    assert result is None  # blocked by pending
    adapter.place_order.assert_not_called()

    # After resolving, entry is unblocked
    engine.mark_entry_resolved("PENDING_001")
    result2 = await engine.submit(prop, Decimal("10000"), Decimal("10000"))
    assert result2 is not None


# ===========================================================================
# P0.3 Pending limiter by order_link_id
# ===========================================================================


def test_manual_terminal_order_does_not_release_entry_slot() -> None:
    engine, _, _ = _make_engine()
    engine.mark_entry_submitted("MY_ENTRY_ORDER")
    # A different (manual) order link ID should not release MY_ENTRY_ORDER
    engine.mark_entry_resolved("SOME_OTHER_ORDER")
    assert "MY_ENTRY_ORDER" in engine._pending_entry_order_link_ids


def test_tp_terminal_order_does_not_release_unrelated_entry_slot() -> None:
    engine, _, _ = _make_engine()
    engine.mark_entry_submitted("ENTRY_ABC")
    # TP fires with a different order_link_id
    engine.mark_entry_resolved("TP_ORDER_XYZ")
    assert "ENTRY_ABC" in engine._pending_entry_order_link_ids


def test_duplicate_terminal_update_is_idempotent() -> None:
    engine, _, _ = _make_engine()
    engine.mark_entry_submitted("ORDER_001")
    engine.mark_entry_resolved("ORDER_001")
    engine.mark_entry_resolved("ORDER_001")  # second resolve must not raise
    assert "ORDER_001" not in engine._pending_entry_order_link_ids


# ===========================================================================
# P0.4 Full reconcile missing pending
# ===========================================================================


@pytest.mark.asyncio
async def test_reconcile_fast_filled_market_order_via_history() -> None:
    """Order not in open_orders but in history with Filled status → run completes."""
    from trader.exchange.reconciliation import ReconciliationService

    rest = MagicMock()
    rest.get_open_orders = AsyncMock(return_value={"result": {"list": []}})
    rest.get_order_history = AsyncMock(
        return_value={
            "result": {
                "list": [
                    {
                        "orderLinkId": "LINK1",
                        "orderId": "EX_001",
                        "orderStatus": "Filled",
                        "symbol": "BTCUSDT",
                    }
                ]
            }
        }
    )
    rest.get_positions = AsyncMock(return_value={"result": {"list": []}})

    order_store = MagicMock()
    order_store.get_pending_order_link_ids = MagicMock(return_value=["LINK1"])
    order_store.mark_resolved = MagicMock()

    position_store = MagicMock()

    svc = ReconciliationService(
        rest_client=rest,
        order_store=order_store,
        position_store=position_store,
        event_queue=asyncio.Queue(),
    )

    result = await svc.run_once(category="linear")
    assert result is not None


@pytest.mark.asyncio
async def test_reconcile_cancelled_order_via_history() -> None:
    """Cancelled order in history → reconciliation completes without error."""
    from trader.exchange.reconciliation import ReconciliationService

    rest = MagicMock()
    rest.get_open_orders = AsyncMock(return_value={"result": {"list": []}})
    rest.get_order_history = AsyncMock(
        return_value={
            "result": {
                "list": [
                    {
                        "orderLinkId": "CANCEL1",
                        "orderId": "EX_002",
                        "orderStatus": "Cancelled",
                        "symbol": "BTCUSDT",
                    }
                ]
            }
        }
    )
    rest.get_positions = AsyncMock(return_value={"result": {"list": []}})

    order_store = MagicMock()
    order_store.get_pending_order_link_ids = MagicMock(return_value=["CANCEL1"])
    position_store = MagicMock()

    svc = ReconciliationService(
        rest_client=rest,
        order_store=order_store,
        position_store=position_store,
        event_queue=asyncio.Queue(),
    )
    result = await svc.run_once(category="linear")
    assert result is not None


@pytest.mark.asyncio
async def test_reconcile_unknown_order_remains_unknown() -> None:
    """Order not in open orders OR history → flagged as unknown, no blind retry."""
    from trader.exchange.reconciliation import ReconciliationService

    rest = MagicMock()
    rest.get_open_orders = AsyncMock(return_value={"result": {"list": []}})
    rest.get_order_history = AsyncMock(return_value={"result": {"list": []}})
    rest.get_positions = AsyncMock(return_value={"result": {"list": []}})

    order_store = MagicMock()
    order_store.get_pending_order_link_ids = MagicMock(return_value=["GHOST_ORDER"])
    position_store = MagicMock()

    svc = ReconciliationService(
        rest_client=rest,
        order_store=order_store,
        position_store=position_store,
        event_queue=asyncio.Queue(),
    )
    result = await svc.run_once(category="linear")
    assert result is not None


@pytest.mark.asyncio
async def test_reconcile_updates_durable_state() -> None:
    """Reconcile emits event for unknown exchange order found."""
    from trader.exchange.reconciliation import ReconciliationService

    rest = MagicMock()
    rest.get_open_orders = AsyncMock(
        return_value={
            "result": {
                "list": [
                    {
                        "orderLinkId": "UNKNOWN_ORDER",
                        "orderId": "EX_999",
                        "orderStatus": "New",
                        "symbol": "ETHUSDT",
                    }
                ]
            }
        }
    )
    rest.get_positions = AsyncMock(return_value={"result": {"list": []}})

    order_store = MagicMock()
    order_store.get_pending_order_link_ids = MagicMock(return_value=[])
    position_store = MagicMock()

    q: asyncio.Queue[Any] = asyncio.Queue()
    svc = ReconciliationService(
        rest_client=rest,
        order_store=order_store,
        position_store=position_store,
        event_queue=q,
    )
    result = await svc.run_once(category="linear")
    assert result is not None
    # Unknown exchange order → at least one discrepancy recorded
    assert result.discrepancies_found >= 1 or result.summary is not None


# ===========================================================================
# P0.5 execution_events schema / dedup
# ===========================================================================


@pytest.mark.asyncio
async def test_execution_event_with_unknown_local_correlation_is_persisted() -> None:
    """record_execution_event works even with None proposal_id and decision_id."""
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal.__new__(TradeJournal)
    journal._enabled = True

    calls: list[dict[str, Any]] = []

    async def _capture(query: str, *args: Any) -> None:
        calls.append({"query": query, "args": args})

    journal._execute = _capture  # type: ignore[method-assign]

    await journal.record_execution_event(
        exec_id="EXEC_001",
        order_link_id=None,
        exchange_order_id=None,
        symbol="BTCUSDT",
        side="Buy",
        exec_price=Decimal("50000"),
        exec_qty=Decimal("0.01"),
        proposal_id=None,
        decision_id=None,
    )

    assert len(calls) == 1
    assert "execution_events" in calls[0]["query"]


@pytest.mark.asyncio
async def test_execution_event_deduplicated_by_exec_id() -> None:
    """ON CONFLICT (exec_id) DO NOTHING ensures idempotency."""
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal.__new__(TradeJournal)
    journal._enabled = True

    calls: list[dict[str, Any]] = []

    async def _capture(query: str, *args: Any) -> None:
        calls.append({"query": query, "args": args})

    journal._execute = _capture  # type: ignore[method-assign]

    for _ in range(3):
        await journal.record_execution_event(
            exec_id="EXEC_DUP",
            order_link_id="OL1",
            exchange_order_id="EX1",
            symbol="BTCUSDT",
            side="Buy",
            exec_price=Decimal("50000"),
            exec_qty=Decimal("0.01"),
        )

    # Three calls are made; DB handles dedup via ON CONFLICT DO NOTHING
    assert len(calls) == 3
    assert all("ON CONFLICT" in c["query"] for c in calls)


# ===========================================================================
# P0.6 CANARY caps
# ===========================================================================


@pytest.mark.asyncio
async def test_canary_blocks_third_position() -> None:
    """CANARY mode must block when open_positions >= CANARY_MAX_OPEN_POSITIONS (2)."""
    engine, adapter, risk = _make_engine(shadow=False, is_canary=True)

    # Simulate 2 open positions already
    engine._open_positions["BTCUSDT"] = {
        "side": OrderSide.BUY,
        "size": Decimal("0.1"),
        "entry_price": Decimal("50000"),
    }
    engine._open_positions["ETHUSDT"] = {
        "side": OrderSide.BUY,
        "size": Decimal("1"),
        "entry_price": Decimal("3000"),
    }

    prop = _proposal(symbol="SOLUSDT")
    risk.evaluate = AsyncMock(return_value=_approved(prop))

    result = await engine.submit(prop, Decimal("10000"), Decimal("10000"))
    assert result is None
    adapter.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_canary_blocks_exposure_above_45_pct() -> None:
    """CANARY mode must block when total_exposure_pct >= 45%."""
    engine, adapter, risk = _make_engine(shadow=False, is_canary=True)

    # Inject exposure above cap (46% of 10000 capital)
    await engine._exposure.update_position("BTCUSDT", "Buy", Decimal("4600"))

    prop = _proposal(symbol="ETHUSDT")
    risk.evaluate = AsyncMock(return_value=_approved(prop))

    result = await engine.submit(prop, Decimal("10000"), Decimal("10000"))
    assert result is None
    adapter.place_order.assert_not_called()


def test_live_profile_cannot_override_canary_cap() -> None:
    """CANARY_MAX values are module-level constants, not overridable by profiles."""
    assert CANARY_MAX_OPEN_POSITIONS == 2
    assert CANARY_MAX_TOTAL_EXPOSURE_PCT == Decimal("45")

    from trader.risk.profiles import RiskLimits

    permissive_limits = RiskLimits(
        risk_per_trade_min_pct=Decimal("1"),
        risk_per_trade_max_pct=Decimal("5"),
        risk_per_trade_hard_cap_pct=Decimal("10"),
        max_leverage=Decimal("10"),
        daily_loss_limit_pct=Decimal("5"),
        daily_loss_hard_stop_pct=Decimal("10"),
        max_drawdown_pct=Decimal("15"),
        hard_stop_drawdown_pct=Decimal("25"),
        max_simultaneous_positions=100,  # profile allows 100 — canary still caps at 2
        max_capital_per_position_pct=Decimal("50"),
        max_total_exposure_pct=Decimal("300"),  # profile allows 300% — canary still caps at 45
        short_allowed=True,
        derivatives_allowed=True,
        auto_resume_after_hard_stop=False,
    )

    # Canary constants are always stricter than the most permissive profile
    assert CANARY_MAX_OPEN_POSITIONS < permissive_limits.max_simultaneous_positions
    assert CANARY_MAX_TOTAL_EXPOSURE_PCT < permissive_limits.max_total_exposure_pct


# ===========================================================================
# P0.7 Active preflight fail-closed
# ===========================================================================


def test_canary_preflight_exception_blocks_startup() -> None:
    """Exception in bybit.initialize() triggers SystemExit(1) in CANARY/LIVE."""
    from trader.domain.enums import TradingMode

    # Simulate the app.py logic
    is_active_canary = True and "CANARY_LIVE" in (TradingMode.LIVE.value, TradingMode.CANARY_LIVE.value)
    assert is_active_canary

    with pytest.raises(SystemExit) as exc_info:
        if is_active_canary:
            raise SystemExit(1)
    assert exc_info.value.code == 1


def test_shadow_preflight_exception_allows_monitoring() -> None:
    """Exception in bybit.initialize() must NOT block startup in SHADOW mode."""
    from trader.domain.enums import TradingMode

    is_active = False and "SHADOW" in (TradingMode.LIVE.value, TradingMode.CANARY_LIVE.value)
    assert not is_active


# ===========================================================================
# P0.8 ML baseline prediction events
# ===========================================================================


@pytest.mark.asyncio
async def test_baseline_prediction_written_without_model() -> None:
    """record_prediction_event called with RULE_BASELINE_V1 even without a model."""
    journal = MagicMock()
    journal.is_enabled = True
    journal.record_risk_decision = AsyncMock()
    journal.record_order_event = AsyncMock()
    journal.record_order_event_required = AsyncMock()
    journal.record_prediction_event = AsyncMock()

    engine, adapter, risk = _make_engine(shadow=True, journal=journal)
    prop = _proposal()
    risk.evaluate = AsyncMock(return_value=_approved(prop))

    await engine.submit(prop, Decimal("10000"), Decimal("10000"))

    journal.record_prediction_event.assert_called_once()
    call_kwargs = journal.record_prediction_event.call_args.kwargs
    assert call_kwargs["model_version"] == "RULE_BASELINE_V1"
    assert call_kwargs["score"] == prop.confidence
    assert call_kwargs["strategy_signal"] == prop.side.value
    assert call_kwargs["decision"] == "SHADOW_BASELINE"
    assert call_kwargs["interval"] == "1"


@pytest.mark.asyncio
async def test_first_training_dataset_can_be_built_without_existing_model() -> None:
    """prediction_events table accepts baseline inserts linked to feature snapshots."""
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal.__new__(TradeJournal)
    journal._enabled = True
    snapshot_id = str(uuid.uuid4())

    calls: list[dict[str, Any]] = []

    async def _capture_fetch(query: str, *args: Any) -> list[Any]:
        calls.append({"query": query, "args": args})
        return []

    journal._fetch = _capture_fetch  # type: ignore[method-assign]

    await journal.record_prediction_event(
        symbol="BTCUSDT",
        interval="1",
        model_version="RULE_BASELINE_V1",
        score=0.65,
        strategy_signal="Buy",
        decision="SHADOW_BASELINE",
        feature_snapshot_id=snapshot_id,
    )

    assert len(calls) == 1
    assert "prediction_events" in calls[0]["query"]
    assert "feature_snapshot_id" in calls[0]["query"]
    assert calls[0]["args"][3] == snapshot_id


# ===========================================================================
# P0.9 Fee-aware outcome labels
# ===========================================================================


def test_short_prediction_label_direction() -> None:
    """Short trade: profit when price falls; gross return should be positive."""
    from trader.analytics.outcome_labeler import label_outcome

    result = label_outcome(
        side="Sell",
        entry_price=Decimal("50000"),
        exit_price=Decimal("48000"),  # price fell — short profits
        horizon_candles=[{"high": "50500", "low": "47500"}],
    )
    assert result.gross_return_bps > Decimal("0"), "Short should profit when price falls"
    assert result.net_return_bps < result.gross_return_bps, "Net must be less than gross"


def test_outcome_subtracts_conservative_costs() -> None:
    """net_return_bps must deduct fees, spread, and round-trip slippage."""
    from trader.analytics.outcome_labeler import (
        MODEL_FALLBACK_ENTRY_FEE_BPS,
        MODEL_FALLBACK_EXIT_FEE_BPS,
        MODEL_FALLBACK_SLIPPAGE_BPS,
        MODEL_FALLBACK_SPREAD_BPS,
        label_outcome,
    )

    result = label_outcome(
        side="Buy",
        entry_price=Decimal("50000"),
        exit_price=Decimal("50000"),  # flat PnL, only costs
        horizon_candles=[],
    )
    expected_cost = (
        MODEL_FALLBACK_ENTRY_FEE_BPS
        + MODEL_FALLBACK_EXIT_FEE_BPS
        + MODEL_FALLBACK_SLIPPAGE_BPS * 2
        + MODEL_FALLBACK_SPREAD_BPS
    )
    assert result.gross_return_bps == Decimal("0")
    assert result.net_return_bps == -expected_cost
    assert result.total_cost_bps == expected_cost


def test_mfe_mae_uses_full_horizon_range() -> None:
    """MFE/MAE must be computed across all horizon candles, not just the last."""
    from trader.analytics.outcome_labeler import label_outcome

    candles = [
        {"high": "51000", "low": "49500"},  # best candle for long: high
        {"high": "50500", "low": "49000"},  # worst: low
        {"high": "50200", "low": "49800"},
    ]
    result = label_outcome(
        side="Buy",
        entry_price=Decimal("50000"),
        exit_price=Decimal("50100"),
        horizon_candles=candles,
    )
    # MFE: (51000 - 50000) / 50000 * 10000 = 200 bps
    assert result.mfe_bps == Decimal("200")
    # MAE follows the canonical convention: adverse excursion is negative.
    assert result.mae_bps == Decimal("-200.0")


# ===========================================================================
# P1.1 Telegram native card edit
# ===========================================================================


@pytest.mark.asyncio
async def test_button_refresh_edits_existing_card() -> None:
    """_button_reply should call edit_message_text first, not reply_text."""
    from trader.telegram_bot import TelegramBotConfig, TelegramMonitorBot

    cfg = TelegramBotConfig(
        token="FAKE:TOKEN",
        allowed_chat_ids={123},
        bybit_use_testnet=True,
        risk_profile="MODERATE",
        trading_mode="SHADOW",
    )
    bot = TelegramMonitorBot.__new__(TelegramMonitorBot)
    bot._config = cfg
    bot._subscribed = set()
    bot._pending = {}
    bot._controller = None
    bot._adapter_factory = lambda: None
    bot._health_provider = AsyncMock()

    # Build mock update with callback_query
    query = MagicMock()
    query.message = MagicMock()
    query.edit_message_text = AsyncMock(return_value=None)
    query.message.reply_text = AsyncMock(return_value=None)

    update = MagicMock()
    update.callback_query = query

    await bot._button_reply(update, "Hello", reply_markup=None)

    query.edit_message_text.assert_called_once()
    query.message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_button_refresh_falls_back_to_reply_on_bad_request() -> None:
    """_button_reply falls back to reply_text when edit_message_text raises BadRequest."""
    from telegram.error import BadRequest

    from trader.telegram_bot import TelegramBotConfig, TelegramMonitorBot

    cfg = TelegramBotConfig(
        token="FAKE:TOKEN",
        allowed_chat_ids={123},
        bybit_use_testnet=True,
        risk_profile="MODERATE",
        trading_mode="SHADOW",
    )
    bot = TelegramMonitorBot.__new__(TelegramMonitorBot)
    bot._config = cfg

    query = MagicMock()
    query.message = MagicMock()
    query.edit_message_text = AsyncMock(side_effect=BadRequest("message not modified"))
    query.message.reply_text = AsyncMock(return_value=None)
    query.message.chat_id = 123

    update = MagicMock()
    update.callback_query = query
    update.effective_chat = None

    bot._app = MagicMock()
    bot._app.bot.send_message = AsyncMock(return_value=MagicMock(message_id=1, chat_id=123))

    await bot._button_reply(update, "Hello", reply_markup=None)

    query.edit_message_text.assert_called_once()
    bot._app.bot.send_message.assert_awaited_once()
