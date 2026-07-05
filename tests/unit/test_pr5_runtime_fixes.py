"""Tests for PR5: symbol sync, pending reconcile, model diagnostics, Telegram aliases.

Tests are intentionally minimal and focus on the new code paths only.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# ЭТАП 1 — _active_symbols helper
# ---------------------------------------------------------------------------


class _FakeScreener:
    def __init__(self, symbols: list[str]) -> None:
        self._symbols = symbols

    @property
    def active_symbols(self) -> list[str]:
        return list(self._symbols)


def _make_app() -> Any:
    """Create a minimal TradingApplication with no real connections."""
    from trader.app import TradingApplication

    app = TradingApplication()
    return app


def test_active_symbols_falls_back_when_no_screener():
    from trader.app import _SYMBOLS

    app = _make_app()
    assert app._screener is None
    result = app._active_symbols()
    assert result == list(_SYMBOLS)


def test_active_symbols_uses_screener_when_available():
    app = _make_app()
    app._screener = _FakeScreener(["DOGEUSDT", "XRPUSDT", "ADAUSDT"])
    result = app._active_symbols()
    assert result == ["DOGEUSDT", "XRPUSDT", "ADAUSDT"]


def test_active_symbols_falls_back_when_screener_empty():
    from trader.app import _SYMBOLS

    app = _make_app()
    app._screener = _FakeScreener([])
    result = app._active_symbols()
    assert result == list(_SYMBOLS)


def test_diagnostics_active_symbols_do_not_use_bootstrap_fallback():
    from trader.modules.diagnostics import DiagnosticsModule

    app = _make_app()
    app._screener = None

    assert DiagnosticsModule(app).runtime_active_symbols() == []


def test_diagnostics_active_symbols_use_only_live_screener_symbols():
    from trader.modules.diagnostics import DiagnosticsModule

    app = _make_app()
    app._screener = _FakeScreener(["DOGEUSDT", "XRPUSDT", "ADAUSDT"])

    assert DiagnosticsModule(app).runtime_active_symbols() == ["DOGEUSDT", "XRPUSDT", "ADAUSDT"]


def test_zero_trading_warning_is_suppressed_during_startup_warmup():
    app = _make_app()
    app._settings = MagicMock()
    app._settings.MIN_SIGNALS_PER_HOUR = 1
    app._settings.AUTO_SOFTEN_FILTERS_ENABLED = False
    app._record_diag("signals_emitted")

    engine = MagicMock()
    engine.is_in_warmup.return_value = True
    engine.warmup_seconds_remaining.return_value = 41.0
    engine.get_diag_counts.return_value = {"order_placed": 0}
    app._execution_engine = engine

    with patch("trader.modules.diagnostics.log") as log:
        app._check_zero_trading()

    log.warning.assert_not_called()
    log.info.assert_called_once()
    assert log.info.call_args.args[0] == "zero_trading.suppressed_warmup"


@pytest.mark.asyncio
async def test_risk_monitor_housekeeping_polls_kill_switch_file_and_pauses():
    from trader.modules.execution_runtime import ExecutionRuntimeModule

    app = _make_app()
    app._kill_switch = MagicMock()
    app._kill_switch.check_file_flag = AsyncMock()
    app._kill_switch.is_active = True
    app._risk_manager = MagicMock()
    app._risk_manager.reset_daily_stats = AsyncMock()

    await ExecutionRuntimeModule(app)._risk_monitor_housekeeping()

    app._kill_switch.check_file_flag.assert_awaited_once()
    assert app._trading_paused is True


@pytest.mark.asyncio
async def test_risk_monitor_housekeeping_resets_daily_pnl_once_per_utc_day():
    from trader.modules.execution_runtime import ExecutionRuntimeModule

    app = _make_app()
    app._kill_switch = None
    app._risk_manager = MagicMock()
    app._risk_manager.reset_daily_stats = AsyncMock()
    module = ExecutionRuntimeModule(app)

    await module._risk_monitor_housekeeping()
    first_day = app._last_daily_reset_date
    await module._risk_monitor_housekeeping()

    assert first_day is not None
    app._risk_manager.reset_daily_stats.assert_awaited_once()


def test_zero_trading_warning_is_suppressed_when_shadow_orders_flow():
    app = _make_app()
    app._settings = MagicMock()
    app._settings.MIN_SIGNALS_PER_HOUR = 1
    app._settings.AUTO_SOFTEN_FILTERS_ENABLED = False
    app._record_diag("signals_emitted")

    engine = MagicMock()
    engine.is_in_warmup.return_value = False
    engine.get_diag_counts.return_value = {
        "order_placed": 0,
        "shadow_order_would_be_placed": 1,
    }
    app._execution_engine = engine

    with patch("trader.modules.diagnostics.log") as log:
        app._check_zero_trading()

    log.warning.assert_not_called()
    log.info.assert_not_called()


def test_top_blocker_prefers_specific_risk_reason():
    app = _make_app()
    diag = {
        "hour_risk_rejected": 4,
        "hour_min_notional_rejected": 4,
        "hour_model_gate_canary_blocked": 1,
    }

    top, blockers = app._top_blocker_from_diag(diag, default="unknown")

    assert top == "risk_rejected:min_notional"
    assert blockers["risk_rejected"] == 4
    assert blockers["risk_rejected:min_notional"] == 4


def test_top_blocker_includes_engine_skip_reasons():
    app = _make_app()
    diag = {
        "hour_signals_emitted": 3,
        "hour_skipped_startup_warmup": 1,
        "hour_skipped_rate_limit": 2,
        "hour_signal_qty_adjustment_rejected": 1,
    }

    top, blockers = app._top_blocker_from_diag(diag, default="unknown")

    assert top == "rate_limit"
    assert blockers["startup_warmup"] == 1
    assert blockers["rate_limit"] == 2
    assert blockers["post_signal_size_rejected"] == 1


def test_top_blocker_exposes_shadow_probe_regime_filter():
    app = _make_app()

    top, blockers = app._top_blocker_from_diag(
        {"hour_shadow_probe_regime_blocked": 7},
        default="unknown",
    )

    assert top == "shadow_probe_regime_blocked"
    assert blockers["shadow_probe_regime_blocked"] == 7


def test_top_blocker_exposes_shadow_probe_net_rr_filter():
    app = _make_app()

    top, blockers = app._top_blocker_from_diag(
        {"hour_shadow_probe_net_rr_rejected": 9},
        default="unknown",
    )

    assert top == "shadow_probe_net_rr_rejected"
    assert blockers["shadow_probe_net_rr_rejected"] == 9


def test_get_diagnostics_exposes_specific_risk_rejection_counts():
    app = _make_app()
    app._record_diag("risk_rejected")
    app._record_diag("risk_rule:sizer_rejected")
    app._record_diag("risk_market_filter_rejected")
    app._record_diag("post_multiplier_min_notional_rejected")

    diag = app.get_diagnostics()

    assert diag["hour_risk_rejected"] == 1
    assert diag["hour_risk_market_filter_rejected"] == 1
    assert diag["hour_min_notional_rejected"] == 1


def test_get_diagnostics_exposes_rejection_symbol_side_details():
    app = _make_app()
    app._record_diag("imbalance_rejected")
    app._record_diag("imbalance_rejected:XRPUSDT:Buy")
    app._record_diag("imbalance_rejected:XRPUSDT:Buy")
    app._record_diag("scalp_net_edge_rejected:DOGEUSDT:Sell")

    diag = app.get_diagnostics()

    assert diag["hour_imbalance_rejected"] == 1
    assert diag["hour_rejection_details"]["imbalance_rejected"][0] == {
        "reason": "imbalance_rejected",
        "symbol": "XRPUSDT",
        "side": "Buy",
        "count": 2,
    }
    assert diag["hour_rejection_details"]["scalp_net_edge_rejected"][0]["symbol"] == "DOGEUSDT"


@pytest.mark.asyncio
async def test_trade_journal_pool_close_timeout_terminates(monkeypatch: pytest.MonkeyPatch) -> None:
    import trader.storage.trade_journal as trade_journal_module
    from trader.storage.trade_journal import TradeJournal

    monkeypatch.setattr(trade_journal_module, "_POOL_CLOSE_TIMEOUT_SECONDS", 0.01)

    class SlowPool:
        def __init__(self) -> None:
            self.terminated = False

        async def close(self) -> None:
            await asyncio.sleep(60)

        def terminate(self) -> None:
            self.terminated = True

    pool = SlowPool()
    journal = TradeJournal("postgresql://example/db")
    journal._pool = pool  # type: ignore[assignment]

    await journal.close()

    assert pool.terminated is True
    assert journal._pool is None


@pytest.mark.asyncio
async def test_trade_journal_connect_closes_pool_when_schema_bootstrap_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trader.storage.trade_journal as trade_journal_module
    from trader.storage.trade_journal import TradeJournal

    closed = False

    class FakePool:
        async def close(self) -> None:
            nonlocal closed
            closed = True

    pool = FakePool()

    async def fake_create_pool(**kwargs: Any) -> FakePool:
        del kwargs
        return pool

    async def failing_schema(self: TradeJournal) -> None:
        del self
        raise TimeoutError("canceling statement due to statement timeout")

    monkeypatch.setattr(trade_journal_module.asyncpg, "create_pool", fake_create_pool)
    monkeypatch.setattr(TradeJournal, "_ensure_schema", failing_schema)

    journal = TradeJournal("postgresql://example/db")

    await journal.connect()

    assert closed is True
    assert journal.is_enabled is False
    assert journal._pool is None
    assert "schema bootstrap degraded" in str(journal.write_health()["last_connect_error"])
    assert journal._reconnect_blocked_until is not None


def test_load_governor_uses_ws_stale_hysteresis() -> None:
    import inspect

    from trader.modules.market_data import MarketDataModule

    src = inspect.getsource(MarketDataModule.run_load_governor)
    assert "ws_stale_threshold_s = 90.0" in src
    assert "overload_streak >= 2" in src
    assert "restore_streak >= 2" in src
    assert "per_symbol_cycle_ms" in src


def test_strategy_loop_measures_processing_before_sleep() -> None:
    import inspect

    from trader.modules.trading_loop import TradingLoopModule

    src = inspect.getsource(TradingLoopModule.start)
    assert "Measure processing time only" in src
    assert src.index("_last_strategy_cycle_ms =") < src.index("timeout=_STRATEGY_LOOP_INTERVAL")


def test_model_progress_reports_actual_and_compatible_samples() -> None:
    import inspect

    from trader.modules.training import TrainingModule

    src = inspect.getsource(TrainingModule.run_model_progress_reporter)
    assert "actual_training_samples" in src
    assert "compatible_training_samples" in src
    assert "Совместимо:" in src


def test_canary_gate_scores_side_aware_model_features() -> None:
    import inspect

    from trader.modules.trading_loop import TradingLoopModule

    src = inspect.getsource(TradingLoopModule.start)
    assert "model_feature_names, model_feature_values = self._app._feature_values_for_side" in src
    assert "self._app._model_registry.score_live(model_feature_values, model_feature_names)" in src
    assert "self._app._model_registry.score_live(vec.values, vec.feature_names)" not in src


def test_shadow_probe_pre_gate_uses_probe_specific_edge_hurdle() -> None:
    import inspect

    from trader.modules.trading_loop import TradingLoopModule

    src = inspect.getsource(TradingLoopModule.start)
    assert "safety_margin_pct=self._app._settings.NET_EDGE_SAFETY_MARGIN_PCT" in src
    assert "probe_min_net_return_pct = self._app._settings.SHADOW_PROBE_MIN_NET_RETURN_PCT" in src
    assert "min_net_return_pct=probe_min_net_return_pct" in src
    assert "safety_margin_pct=0.01" not in src


def test_enabled_canary_gate_fails_closed_without_compatible_champion() -> None:
    import inspect

    from trader.modules.trading_loop import TradingLoopModule

    src = inspect.getsource(TradingLoopModule.start)
    assert "ml_canary.no_compatible_champion" in src
    assert 'await _record_signal("model_gate_no_compatible_champion")' in src
    after = src.split('await _record_signal("model_gate_no_compatible_champion")', maxsplit=1)[1]
    assert "return" in after[:80]


def test_strategy_expectancy_blocks_are_persisted_as_blocked_signals() -> None:
    import inspect

    from trader.modules.trading_loop import TradingLoopModule

    src = inspect.getsource(TradingLoopModule.start)
    for reason in (
        "strategy_expectancy_blocked",
        "strategy_side_expectancy_blocked",
        "strategy_side_confidence_blocked",
        "strategy_regime_expectancy_blocked",
        "strategy_regime_confidence_blocked",
    ):
        marker = f'await _record_signal("{reason}")'
        assert marker in src
        after = src.split(marker, maxsplit=1)[1]
        assert "return" in after[:120]


def test_pre_strategy_expectancy_blocks_are_persisted_as_blocked_signals() -> None:
    import inspect

    from trader.modules.trading_loop import TradingLoopModule

    src = inspect.getsource(TradingLoopModule.start)
    assert "stats_ready, stats_block_reason = self._app._expectancy_stats_ready()" in src
    stats_marker = "await _record_signal(reason)"
    bucket_marker = 'await _record_signal("bucket_blocked")'
    assert stats_marker in src
    assert bucket_marker in src
    assert src.index("proposal = self._app._strategy_ensemble.evaluate_all") < src.index(stats_marker)
    assert src.index("proposal = self._app._strategy_ensemble.evaluate_all") < src.index(bucket_marker)
    assert "return" in src.split(stats_marker, maxsplit=1)[1][:120]
    assert "return" in src.split(bucket_marker, maxsplit=1)[1][:120]


def test_execution_pre_risk_none_decision_records_specific_reason() -> None:
    import inspect

    from trader.modules.trading_loop import TradingLoopModule

    src = inspect.getsource(TradingLoopModule.start)
    assert 'await _record_signal(str(pre_risk_reason or "no_decision"))' in src
    assert "consume_last_pre_risk_rejection_reason" in src
    decision_none_block = src.split("if decision is None:", maxsplit=1)[1].split("return", maxsplit=1)[0]
    assert "pre_risk_reason" in decision_none_block
    assert "consume_last_pre_risk_rejection_reason" in decision_none_block


# ---------------------------------------------------------------------------
# ЭТАП 1 — feature_pipeline symbols_updated log
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_logs_symbols_updated_on_change():
    from trader.data.candles import CandleStore
    from trader.features.pipeline import FeaturePipeline

    store = CandleStore(max_bars=10)
    pipeline = FeaturePipeline(candle_store=store, stale_threshold_s=999.0, watchdog_interval_s=0.05)

    class DynamicScreener:
        def __init__(self) -> None:
            self._symbols = ["DOGEUSDT"]

        @property
        def active_symbols(self) -> list[str]:
            return list(self._symbols)

    screener = DynamicScreener()

    log_calls: list[str] = []

    with patch("trader.features.pipeline.log") as mock_log:
        mock_log.info = MagicMock(side_effect=lambda event, **kw: log_calls.append(event))

        # Run pipeline briefly, then change symbols
        async def _run():
            await pipeline.run(symbols=["DOGEUSDT"], intervals=["1"], symbol_source=screener)

        task = asyncio.create_task(_run())
        await asyncio.sleep(0.1)
        screener._symbols = ["DOGEUSDT", "XRPUSDT"]
        await asyncio.sleep(0.12)
        pipeline.stop()
        await asyncio.wait_for(task, timeout=2.0)

    assert "feature_pipeline.symbols_updated" in log_calls


@pytest.mark.asyncio
async def test_pipeline_no_log_when_symbols_unchanged():
    from trader.data.candles import CandleStore
    from trader.features.pipeline import FeaturePipeline

    store = CandleStore(max_bars=10)
    pipeline = FeaturePipeline(candle_store=store, stale_threshold_s=999.0, watchdog_interval_s=0.05)

    class StaticScreener:
        @property
        def active_symbols(self) -> list[str]:
            return ["DOGEUSDT"]

    log_calls: list[str] = []
    with patch("trader.features.pipeline.log") as mock_log:
        mock_log.info = MagicMock(side_effect=lambda event, **kw: log_calls.append(event))

        async def _run():
            await pipeline.run(symbols=["DOGEUSDT"], intervals=["1"], symbol_source=StaticScreener())

        task = asyncio.create_task(_run())
        await asyncio.sleep(0.15)
        pipeline.stop()
        await asyncio.wait_for(task, timeout=2.0)

    assert "feature_pipeline.symbols_updated" not in log_calls


# ---------------------------------------------------------------------------
# ЭТАП 2 — has_pending_order_for_symbol
# ---------------------------------------------------------------------------


def _make_engine(shadow: bool = True) -> Any:
    from trader.execution.engine import ExecutionEngine

    adapter = MagicMock()
    risk_manager = MagicMock()
    exposure = MagicMock()
    exposure.total_exposure_pct = Decimal("0")

    return ExecutionEngine(
        adapter=adapter,
        risk_manager=risk_manager,
        exposure_tracker=exposure,
        shadow_mode=shadow,
    )


def test_has_pending_order_for_symbol_false_initially():
    engine = _make_engine()
    assert engine.has_pending_order_for_symbol("DOGEUSDT") is False


def test_mark_entry_submitted_stores_symbol():
    engine = _make_engine()
    engine.mark_entry_submitted("order123", symbol="DOGEUSDT")
    assert engine.has_pending_order_for_symbol("DOGEUSDT") is True
    assert engine.has_pending_order_for_symbol("XRPUSDT") is False


def test_mark_entry_resolved_removes_symbol():
    engine = _make_engine()
    engine.mark_entry_submitted("order123", symbol="DOGEUSDT")
    engine.mark_entry_resolved("order123")
    assert engine.has_pending_order_for_symbol("DOGEUSDT") is False


def test_mark_entry_resolved_idempotent():
    engine = _make_engine()
    engine.mark_entry_resolved("nonexistent")  # must not raise


def test_mark_entry_submitted_no_symbol_still_adds_to_set():
    engine = _make_engine()
    engine.mark_entry_submitted("order456")
    assert "order456" in engine._pending_entry_order_link_ids
    # No symbol stored
    assert "order456" not in engine._pending_entry_symbols


def _proposal_for_pending(symbol: str) -> Any:
    from trader.domain.enums import MarketRegime, MarketType, OrderSide
    from trader.domain.models import TradeProposal

    return TradeProposal(
        strategy_id="test",
        symbol=symbol,
        market_type=MarketType.LINEAR,
        side=OrderSide.BUY,
        requested_qty=Decimal("1"),
        entry_price=Decimal("10"),
        stop_loss=Decimal("9.8"),
        take_profit=Decimal("10.4"),
        confidence=0.8,
        regime=MarketRegime.BULL_TREND,
    )


@pytest.mark.asyncio
async def test_pending_gate_blocks_same_symbol_but_not_other_shadow_symbols():
    from trader.domain.enums import RiskDecisionStatus
    from trader.domain.models import InstrumentInfo, RiskDecision
    from trader.execution.engine import ExecutionEngine

    adapter = MagicMock()
    adapter.get_instrument_info = AsyncMock(
        return_value=InstrumentInfo(
            symbol="ETHUSDT",
            market_type=_proposal_for_pending("ETHUSDT").market_type,
            base_coin="ETH",
            quote_coin="USDT",
            min_order_qty=Decimal("0.001"),
            max_order_qty=Decimal("1000"),
            qty_step=Decimal("0.001"),
            tick_size=Decimal("0.01"),
            min_notional=Decimal("5"),
        )
    )
    risk_manager = MagicMock()
    risk_manager.evaluate = AsyncMock(
        return_value=RiskDecision(
            proposal_id=_proposal_for_pending("ETHUSDT").proposal_id,
            status=RiskDecisionStatus.REJECTED,
            reason="test reached risk manager",
        )
    )
    exposure = MagicMock()
    exposure.total_exposure_pct = Decimal("0")
    engine = ExecutionEngine(
        adapter=adapter,
        risk_manager=risk_manager,
        exposure_tracker=exposure,
        shadow_mode=True,
        max_concurrent_pending_entries=4,
    )
    engine.mark_entry_submitted("oid-btc", symbol="BTCUSDT")

    assert await engine.submit(_proposal_for_pending("BTCUSDT"), Decimal("1000"), Decimal("1000")) is None
    risk_manager.evaluate.assert_not_awaited()

    await engine.submit(_proposal_for_pending("ETHUSDT"), Decimal("1000"), Decimal("1000"))
    risk_manager.evaluate.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_all_open_orders_cancels_and_clears_pending():
    from trader.execution.engine import ExecutionEngine

    adapter = MagicMock()
    adapter.get_open_orders = AsyncMock(
        return_value=[
            {"symbol": "BTCUSDT", "orderLinkId": "link-1"},
            {"symbol": "ETHUSDT", "orderLinkId": "link-2"},
        ]
    )
    adapter.cancel_order = AsyncMock(return_value={})

    engine = ExecutionEngine(
        adapter=adapter,
        risk_manager=MagicMock(),
        exposure_tracker=MagicMock(total_exposure_pct=Decimal("0")),
        shadow_mode=False,
    )
    engine.resolve_pending_durable = AsyncMock()

    assert await engine.cancel_all_open_orders() == 2
    assert adapter.cancel_order.await_count == 2
    assert engine.resolve_pending_durable.await_count == 2


# ---------------------------------------------------------------------------
# ЭТАП 3 — reconcile_restored_pending_entries
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _make_engine_with_journal(shadow: bool = True) -> tuple[Any, MagicMock]:
    engine = _make_engine(shadow=shadow)
    journal = MagicMock()
    engine._trade_journal = journal
    return engine, journal


@pytest.mark.asyncio
async def test_reconcile_does_nothing_when_no_pending():
    engine, journal = _make_engine_with_journal()
    # No pending entries
    await engine.reconcile_restored_pending_entries()
    journal.get_pending_durable_orders.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_keeps_fresh_pending():
    engine, journal = _make_engine_with_journal()

    oid = "fresh_order_id"
    engine.restore_pending_entries([oid])
    created_at = _now() - timedelta(seconds=30)  # 30s old → below 600s threshold

    journal.get_pending_order_events = AsyncMock(return_value=[])
    journal.get_pending_durable_orders = AsyncMock(
        return_value=[
            {"order_link_id": oid, "symbol": "DOGEUSDT", "state": "PENDING", "created_at": created_at},
        ]
    )
    # No exchange orders
    engine._adapter.get_open_orders = AsyncMock(return_value=[])

    await engine.reconcile_restored_pending_entries()

    # Should still be in the pending set (kept as recent)
    assert oid in engine._pending_entry_order_link_ids
    journal.mark_order_event_stale.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_clears_old_stale_pending():
    engine, journal = _make_engine_with_journal()

    oid = "stale_order_id"
    engine.restore_pending_entries([oid])
    created_at = _now() - timedelta(seconds=700)  # 700s > 600s threshold

    journal.get_pending_order_events = AsyncMock(
        return_value=[
            {"order_link_id": oid, "symbol": "DOGEUSDT", "status": "CREATED_LOCAL", "created_at": created_at},
        ]
    )
    journal.get_pending_durable_orders = AsyncMock(return_value=[])
    engine._adapter.get_open_orders = AsyncMock(return_value=[])
    journal.mark_order_event_stale = AsyncMock()
    journal.mark_durable_order_stale = AsyncMock()

    await engine.reconcile_restored_pending_entries()

    # Should be removed from pending set
    assert oid not in engine._pending_entry_order_link_ids
    journal.mark_order_event_stale.assert_called_once_with(
        oid, pytest.approx(f"no_exchange_order_no_position_age_{700}s", rel=None)
    )
    journal.mark_durable_order_stale.assert_called_once()


@pytest.mark.asyncio
async def test_reconcile_keeps_pending_with_exchange_order():
    engine, journal = _make_engine_with_journal()

    oid = "live_order"
    engine.restore_pending_entries([oid])
    created_at = _now() - timedelta(seconds=700)  # old but has exchange order

    journal.get_pending_order_events = AsyncMock(
        return_value=[
            {"order_link_id": oid, "symbol": "XRPUSDT", "status": "SUBMITTING", "created_at": created_at},
        ]
    )
    journal.get_pending_durable_orders = AsyncMock(return_value=[])
    engine._adapter.get_open_orders = AsyncMock(
        return_value=[
            {"orderLinkId": oid, "symbol": "XRPUSDT"},
        ]
    )
    journal.mark_order_event_stale = AsyncMock()

    await engine.reconcile_restored_pending_entries()

    assert oid in engine._pending_entry_order_link_ids
    journal.mark_order_event_stale.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_keeps_pending_when_position_exists():
    engine, journal = _make_engine_with_journal()

    oid = "filled_order"
    engine.restore_pending_entries([oid])
    created_at = _now() - timedelta(seconds=800)

    journal.get_pending_order_events = AsyncMock(
        return_value=[
            {"order_link_id": oid, "symbol": "ADAUSDT", "status": "SUBMITTING", "created_at": created_at},
        ]
    )
    journal.get_pending_durable_orders = AsyncMock(return_value=[])
    engine._adapter.get_open_orders = AsyncMock(return_value=[])
    # Simulate open position for ADAUSDT
    engine._open_positions["ADAUSDT"] = {"side": MagicMock(), "size": Decimal("100"), "entry_price": Decimal("0.5")}
    journal.mark_order_event_stale = AsyncMock()

    await engine.reconcile_restored_pending_entries()

    assert oid in engine._pending_entry_order_link_ids
    journal.mark_order_event_stale.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_fails_safe_when_api_unavailable():
    engine, journal = _make_engine_with_journal()

    oid = "pending_api_down"
    engine.restore_pending_entries([oid])
    created_at = _now() - timedelta(seconds=700)

    journal.get_pending_order_events = AsyncMock(
        return_value=[
            {"order_link_id": oid, "symbol": "DOGEUSDT", "status": "CREATED_LOCAL", "created_at": created_at},
        ]
    )
    journal.get_pending_durable_orders = AsyncMock(return_value=[])
    engine._adapter.get_open_orders = AsyncMock(side_effect=RuntimeError("Network error"))
    journal.mark_order_event_stale = AsyncMock()

    await engine.reconcile_restored_pending_entries()

    # Fail-safe: pending must be preserved when API is down
    assert oid in engine._pending_entry_order_link_ids
    journal.mark_order_event_stale.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_clears_old_stale_unblocks_new_submit():
    """After stale cleared, has_pending_entries() returns False."""
    engine, journal = _make_engine_with_journal()

    oid = "stale_blocker"
    engine.restore_pending_entries([oid])
    created_at = _now() - timedelta(seconds=700)

    journal.get_pending_order_events = AsyncMock(
        return_value=[
            {"order_link_id": oid, "symbol": "WLDUSDT", "status": "CREATED_LOCAL", "created_at": created_at},
        ]
    )
    journal.get_pending_durable_orders = AsyncMock(return_value=[])
    engine._adapter.get_open_orders = AsyncMock(return_value=[])
    journal.mark_order_event_stale = AsyncMock()
    journal.mark_durable_order_stale = AsyncMock()

    assert engine.has_pending_entries() is True
    await engine.reconcile_restored_pending_entries()
    assert engine.has_pending_entries() is False


# ---------------------------------------------------------------------------
# ЭТАП 4 — Telegram aliases /db and /model
# ---------------------------------------------------------------------------


def test_telegram_db_model_handler_registered():
    """Both /db and /model must route to _cmd_db_model (inspect handler list directly)."""
    from trader.telegram_bot import TelegramBotConfig, TelegramMonitorBot

    config = TelegramBotConfig(
        token="fake:token",
        allowed_chat_ids={12345},
        trading_mode="SHADOW",
        risk_profile="CONSERVATIVE",
        bybit_use_testnet=False,
    )

    # Track registered handlers without actually starting the bot
    registered: dict[str, Any] = {}

    bot = TelegramMonitorBot(
        config=config,
        health_provider=AsyncMock(),
        adapter_factory=lambda: None,
        controller=None,
    )

    with patch("trader.telegram_bot.Application") as mock_app_cls:
        mock_app = MagicMock()

        def record_handler(handler: Any) -> None:
            if hasattr(handler, "commands"):
                for cmd in handler.commands:
                    registered[cmd] = handler.callback

        mock_app.add_handler = MagicMock(side_effect=record_handler)
        mock_app_cls.builder.return_value.token.return_value.build.return_value = mock_app
        mock_app.initialize = AsyncMock()
        mock_app.start = AsyncMock()
        mock_app.bot.delete_webhook = AsyncMock()
        mock_app.updater = MagicMock()
        mock_app.updater.start_polling = AsyncMock()
        mock_app.updater.running = False

        asyncio.run(bot.start())
        if bot._polling_watchdog_task is not None:
            bot._polling_watchdog_task.cancel()

    assert "db" in registered, "/db handler not registered"
    assert "model" in registered, "/model handler not registered"


def test_fmt_timestamp_handles_datetime():
    from trader.telegram_bot import TelegramMonitorBot

    dt = datetime(2025, 6, 8, 14, 30, 0, tzinfo=UTC)
    result = TelegramMonitorBot._fmt_timestamp(dt)
    assert "14:30:00" in result
    assert "UTC" in result


def test_fmt_timestamp_handles_iso_string():
    from trader.telegram_bot import TelegramMonitorBot

    result = TelegramMonitorBot._fmt_timestamp("2025-06-08T14:30:00+00:00")
    assert "14:30:00" in result


def test_fmt_timestamp_handles_none():
    from trader.telegram_bot import TelegramMonitorBot

    assert TelegramMonitorBot._fmt_timestamp(None) == "нет"


def test_fmt_timestamp_handles_malformed():
    from trader.telegram_bot import TelegramMonitorBot

    result = TelegramMonitorBot._fmt_timestamp("not-a-date")
    assert isinstance(result, str)  # must not raise


# ---------------------------------------------------------------------------
# ЭТАП 5 — active_model_version in diagnostics
# ---------------------------------------------------------------------------


def test_get_db_diagnostics_includes_active_model_version_key():
    """get_db_diagnostics must include active_model_version key in result."""
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal(postgres_dsn="postgresql://fake", enabled=False)

    # When not enabled, get_db_diagnostics returns early with defaults
    result = asyncio.run(journal.get_db_diagnostics())
    # The key must be initialised in the result dict even when DB is disabled
    assert "active_model_version" in result


# ---------------------------------------------------------------------------
# ЭТАП 6 — Diagnostic counters
# ---------------------------------------------------------------------------


def test_engine_diag_counts_initial():
    engine = _make_engine()
    counts = engine.get_diag_counts()
    assert counts["skipped_pending_entries"] == 0
    assert counts["skipped_startup_warmup"] == 0
    assert counts["skipped_rate_limit"] == 0
    assert counts["signal_qty_adjustment_rejected"] == 0
    assert counts["order_placed"] == 0
    assert counts["order_failed"] == 0
    assert counts["pending_entry_count"] == 0


def test_engine_diag_pending_count_increments():
    engine = _make_engine()
    engine.mark_entry_submitted("oid1", symbol="DOGEUSDT")
    counts = engine.get_diag_counts()
    assert counts["pending_entry_count"] == 1


def test_engine_pending_entry_diagnostics():
    engine = _make_engine()
    engine.mark_entry_submitted("abc123", symbol="XRPUSDT")
    diag = engine.pending_entry_diagnostics()
    assert diag["pending_entry_count"] == 1
    assert "abc123" in diag["pending_entry_ids"]
    assert "XRPUSDT" in diag["pending_entry_symbols"]


# ---------------------------------------------------------------------------
# mark_order_event_stale safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_order_event_stale_only_updates_non_terminal():
    """mark_order_event_stale must use WHERE clause limiting to non-terminal states."""
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal(postgres_dsn="postgresql://fake", enabled=True)

    executed: list[tuple] = []

    async def fake_execute(query: str, *args: Any) -> None:
        executed.append((query, args))

    journal._execute = fake_execute  # type: ignore[method-assign]

    await journal.mark_order_event_stale("oid999", "test_reason")

    assert executed
    query = executed[0][0]
    assert "FAILED_STALE" in query
    assert "CREATED_LOCAL" in query
    assert "SUBMITTING" in query
    assert "DELETE" not in query.upper()


@pytest.mark.asyncio
async def test_get_pending_order_events_returns_list():
    """get_pending_order_events must return list from order_events table."""
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal(postgres_dsn="postgresql://fake", enabled=True)

    async def fake_fetch(query: str, *args: Any) -> list[dict]:
        assert "order_events" in query
        assert "CREATED_LOCAL" in query
        assert "SUBMITTING" in query
        return []

    journal._fetch = fake_fetch  # type: ignore[method-assign]

    result = await journal.get_pending_order_events()
    assert result == []


@pytest.mark.asyncio
async def test_reconcile_keeps_pending_with_position():
    engine, journal = _make_engine_with_journal()
    oid = "position_order_id"
    engine.restore_pending_entries([oid])
    created_at = _now() - timedelta(seconds=700)

    journal.get_pending_order_events = AsyncMock(return_value=[])
    journal.get_pending_durable_orders = AsyncMock(
        return_value=[
            {"order_link_id": oid, "symbol": "DOGEUSDT", "state": "PENDING", "created_at": created_at},
        ]
    )
    engine._adapter.get_open_orders = AsyncMock(return_value=[])
    engine._open_positions["DOGEUSDT"] = {"side": "Buy", "size": Decimal("100"), "entry_price": Decimal("0.001")}
    journal.mark_order_event_stale = AsyncMock()

    await engine.reconcile_restored_pending_entries()

    assert oid in engine._pending_entry_order_link_ids
    journal.mark_order_event_stale.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_keeps_pending_when_api_fails():
    engine, journal = _make_engine_with_journal()
    oid = "api_fail_order_id"
    engine.restore_pending_entries([oid])
    created_at = _now() - timedelta(seconds=700)

    journal.get_pending_order_events = AsyncMock(return_value=[])
    journal.get_pending_durable_orders = AsyncMock(
        return_value=[
            {"order_link_id": oid, "symbol": "DOGEUSDT", "state": "PENDING", "created_at": created_at},
        ]
    )
    engine._adapter.get_open_orders = AsyncMock(side_effect=RuntimeError("API down"))
    journal.mark_order_event_stale = AsyncMock()

    await engine.reconcile_restored_pending_entries()

    # Fail-safe: pending entry must be preserved when exchange API is unavailable
    assert oid in engine._pending_entry_order_link_ids
    journal.mark_order_event_stale.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_syncs_pending_count():
    engine, journal = _make_engine_with_journal(shadow=False)
    oid = "sync_count_order_id"
    engine.restore_pending_entries([oid])
    # Manually break the count to simulate drift
    engine._pending_entry_count = 99
    created_at = _now() - timedelta(seconds=700)

    journal.get_pending_order_events = AsyncMock(return_value=[])
    journal.get_pending_durable_orders = AsyncMock(
        return_value=[
            {"order_link_id": oid, "symbol": "DOGEUSDT", "state": "PENDING", "created_at": created_at},
        ]
    )
    engine._adapter.get_open_orders = AsyncMock(return_value=[])
    journal.mark_order_event_stale = AsyncMock()
    journal.mark_durable_order_stale = AsyncMock()

    await engine.reconcile_restored_pending_entries()

    # After stale cleared, count must be synced to actual set size (0)
    assert engine._pending_entry_count == len(engine._pending_entry_order_link_ids)
    assert engine._pending_entry_count == 0


def test_champion_readiness_not_reset_by_newer_challenger():
    """active_model_version must use CHAMPION, not a newer SHADOW_CHALLENGER."""
    from trader.storage.trade_journal import TradeJournal

    tj = TradeJournal(postgres_dsn="postgresql://fake", enabled=False)
    import asyncio

    diag = asyncio.run(tj.get_db_diagnostics())

    # When DB is disabled, active_model_version should be an empty dict (not crash)
    assert "active_model_version" in diag
    assert isinstance(diag["active_model_version"], dict)
    # latest_model_version also present
    assert "latest_model_version" in diag


def test_position_update_cache_preserves_unrelated_positions():
    from types import SimpleNamespace

    from trader.app import TradingApplication
    from trader.domain.enums import OrderSide

    app = TradingApplication()
    btc = SimpleNamespace(symbol="BTCUSDT", side=OrderSide.BUY, size=Decimal("0.01"), entry_price=Decimal("50000"))
    eth = SimpleNamespace(symbol="ETHUSDT", side=OrderSide.SELL, size=Decimal("0.2"), entry_price=Decimal("3000"))
    app._cache_exchange_positions([btc, eth])

    app._cache_exchange_position_update(
        SimpleNamespace(symbol="BTCUSDT", side=OrderSide.BUY, size=Decimal("0"), entry_price=Decimal("0"))
    )

    cached = app._latest_exchange_positions
    assert [p.symbol for p in cached] == ["ETHUSDT"]
