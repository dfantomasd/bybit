"""Application entry point.

Lifecycle:
1. Parse config
2. Configure logging
3. Run preflight checks
4. Start health-check HTTP server
5. Start WebSocket connections (public market data)
6. Seed candle store from REST history
7. Start feature pipeline
8. Start strategy ensemble loop → RiskManager → ExecutionEngine
9. Enter shutdown-wait loop
10. On SIGTERM/SIGINT: graceful shutdown

CRITICAL SAFETY RULES:
- System starts in TESTNET or SHADOW mode by default.
- LIVE mode requires explicit LIVE_MODE=true AND TRADING_MODE=LIVE in config.
- The Risk Manager is always the final authority; it cannot be bypassed here.
- In SHADOW mode no orders are ever submitted to the exchange.
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from collections import deque
from datetime import datetime
from decimal import Decimal
from typing import Any, cast

import uvicorn

from trader.domain.enums import SystemStatus
from trader.domain.models import FeatureVector
from trader.modules.diagnostics import DiagnosticsModule
from trader.modules.execution_runtime import ExecutionRuntimeModule
from trader.modules.registry import ModuleRegistry
from trader.modules.signal_policy import SignalPolicyModule
from trader.modules.trading_loop import TradingLoopModule
from trader.monitoring.logging import get_logger
from trader.runtime.constants import (
    _CRITICAL_TASK_NAMES,
    _DIAG_WINDOW,
    _FALLBACK_BALANCE_USD,
    _INTERVAL_MS,
    _JOURNAL_FALLBACK_UUID,
    _SYMBOLS,
    _WS_INTERVAL,
)
from trader.runtime.state_proxy import AppStateProxy, _AppStateProxy

log = get_logger(__name__)


class TradingApplication:
    """Top-level application orchestrator."""

    def __init__(self) -> None:
        self._status: SystemStatus = SystemStatus.STARTING
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._settings: Any | None = None
        self._health_checker: Any | None = None
        self._uvicorn_server: uvicorn.Server | None = None
        self._fastapi_app: Any | None = None
        self._bybit_adapter: Any | None = None
        self._telegram_bot: Any | None = None
        self._db_diagnostics_cache: dict[str, Any] | None = None
        self._db_diagnostics_cache_at: float = 0.0
        self._db_diagnostics_lite_cache: dict[str, Any] | None = None
        self._db_diagnostics_lite_cache_at: float = 0.0
        self._model_performance_cache: list[dict[str, Any]] = []
        self._model_performance_cache_at: datetime | None = None
        self._ws_public: Any | None = None
        self._candle_store: Any | None = None
        self._orderbook_tracker: Any | None = None
        self._flow_tracker: Any | None = None
        self._feature_pipeline: Any | None = None
        # Regime-bucket expectancy stats: {(regime, volatility, hour): (avg_bps, count)}
        self._bucket_stats: dict[tuple[str, str, int], tuple[float, int]] = {}
        # Symbol-side expectancy stats: {(symbol, side): (avg_bps, count)}
        self._symbol_side_stats: dict[tuple[str, str], tuple[float, int]] = {}
        # Shadow probe paper stats (active even in SHADOW mode)
        self._shadow_probe_side_stats: dict[tuple[str, str], tuple[float, int]] = {}
        self._shadow_probe_symbol_stats: dict[str, tuple[float, int]] = {}
        self._shadow_probe_eligible_symbols: set[str] | None = None
        self._shadow_probe_symbol_subscribed_at: dict[str, datetime] = {}
        self._bucket_stats_refreshed_at: datetime | None = None
        # Per-candle training sampler: last sampled candle open_time per symbol
        self._last_candle_sample_at: dict[str, datetime] = {}
        # Candle sampler health counters — reset every log cycle
        self._candle_sampler_total: int = 0
        self._candle_sampler_scored: int = 0
        self._candle_sampler_no_model: int = 0
        self._candle_sampler_gate_pass: int = 0
        self._candle_sampler_gate_block: int = 0
        # Per-symbol signal cooldown: suppress duplicate proposals within one candle period.
        # The strategy loop runs every ~10s but features refresh ~60s (one 1m candle), so
        # without a cooldown the same signal fires 5-6 times and floods training data with
        # correlated duplicates before execution can block them.
        self._last_signal_at: dict[str, datetime] = {}
        self._signal_cooldown_s: float = 60.0
        self._strategy_ensemble: Any | None = None
        self._risk_manager: Any | None = None
        self._execution_engine: Any | None = None
        self._exposure_tracker: Any | None = None
        self._screener: Any | None = None
        self._regime_classifier: Any | None = None
        self._background_tasks: list[asyncio.Task[Any]] = []
        # Cached balance (refreshed periodically)
        self._cached_balance: Decimal = _FALLBACK_BALANCE_USD
        self._balance_refreshed_at: datetime | None = None
        # Operator control state
        self._trading_paused: bool = False
        self._current_risk_profile_str: str = ""
        self._signal_log: deque[Any] = deque(maxlen=20)
        self._kill_switch: Any | None = None
        self._trade_journal: Any | None = None
        self._performance_blocked_symbols: set[str] = set()
        self._closed_pnl_refreshed_at: datetime | None = None
        self._positions_managed_at: datetime | None = None
        self._positions_synced_at: datetime | None = None
        self._latest_exchange_positions: list[Any] = []
        self._latest_exchange_positions_at: datetime | None = None
        self._trailing_stop_keys: set[str] = set()
        self._fee_provider: Any | None = None
        self._last_tx_log_sync_at: datetime | None = None
        self._last_zero_trading_warn_at: datetime | None = None
        self._last_ws_recovery_at: datetime | None = None
        self._shadow_closed_results: deque[tuple[datetime, str, float]] = deque(maxlen=50)
        self._shadow_loss_guard_until: datetime | None = None
        # Set on every confirmed WS kline; drives the canary "fresh confirmed candles" check
        self._last_confirmed_candle_at: datetime | None = None
        # Diagnostics: rolling deque of (timestamp, event_type) for last-hour stats
        self._diag_events: deque[tuple[datetime, str]] = deque(maxlen=10_000)
        self._last_strategy_loop_at: datetime | None = None
        self._training_task: asyncio.Task[Any] | None = None
        self._training_start_lock: asyncio.Lock = asyncio.Lock()
        self._last_training_message: str = "never"
        self._training_failed_at: float | None = None  # monotonic time of last failed training
        # Private WebSocket (order/position/balance real-time events)
        self._ws_private: Any | None = None
        # ML shadow scoring
        self._model_registry: Any | None = None
        self._model_gate_recent_blocks: deque[bool] = deque(maxlen=100)
        self._model_gate_block_counter: int = 0
        self._model_gate_quality: dict[str, Any] = {}
        self._model_gate_quality_checked_at: datetime | None = None
        self._last_strategy_cycle_ms: float = 0.0
        self._drift_status: dict[str, Any] = {"status": "n/a"}
        self._last_retention_run_at: datetime | None = None
        self._startup_retention_done: bool = False
        self._subscribe_watchdog: Any | None = None
        self._online_learning_updates_since_checkpoint: int = 0
        self._modules = ModuleRegistry(self)
        self._trading_loop = TradingLoopModule(self)

    def _candle_store_caps(self) -> dict[str, int]:
        settings = self._settings
        if settings is None:
            return {"1": 250, "5": 250, "15": 200, "60": 120}
        return {
            "1": int(getattr(settings, "CANDLE_STORE_MAX_BARS_1M", 250)),
            "5": int(getattr(settings, "CANDLE_STORE_MAX_BARS_5M", 250)),
            "15": int(getattr(settings, "CANDLE_STORE_MAX_BARS_15M", 200)),
            "60": int(getattr(settings, "CANDLE_STORE_MAX_BARS_1H", 120)),
        }

    def _new_candle_store(self) -> Any:
        from trader.data.candles import CandleStore

        return CandleStore(max_bars=500, max_bars_by_interval=self._candle_store_caps())

    def _active_symbols(self) -> list[str]:
        """Return screener's current active symbols, or fallback list if screener is absent/empty."""
        if self._screener is not None:
            symbols = self._screener.active_symbols
            if symbols:
                return cast(list[str], symbols)
        return list(_SYMBOLS)

    def _market_data_intervals(self) -> list[str]:
        """Configured kline intervals with 1m kept first for strategy compatibility."""
        if self._settings is None or not self._settings.MULTITIMEFRAME_ENABLED:
            return [_WS_INTERVAL]

        intervals: list[str] = []
        for interval in [_WS_INTERVAL, *self._settings.MULTITIMEFRAME_INTERVALS]:
            interval = str(interval).strip()
            if interval and interval not in intervals:
                intervals.append(interval)
        return intervals

    def _should_persist_candle_interval(self, interval: str) -> bool:
        """Whether confirmed candles for this interval should be written to Postgres."""
        if self._settings is None:
            return interval == "1"
        persist_fn = getattr(self._settings, "market_candle_persist_intervals", None)
        if callable(persist_fn):
            return interval in persist_fn()
        raw = getattr(self._settings, "MARKET_CANDLE_PERSIST_INTERVALS", "1")
        return interval in {part.strip() for part in str(raw).split(",") if part.strip()}

    def _ws_topics_for_symbol(self, symbol: str) -> list[str]:
        """Build public WS topic list for one symbol."""
        topics = [f"kline.{interval}.{symbol}" for interval in self._market_data_intervals()]
        topics.append(f"tickers.{symbol}")
        if self._settings is not None and self._settings.ORDERBOOK_FEED_ENABLED:
            topics.append(f"orderbook.50.{symbol}")
        if self._settings is not None and self._settings.TRADE_FLOW_FEED_ENABLED:
            topics.append(f"publicTrade.{symbol}")
        if self._settings is not None and self._settings.LIQUIDATION_FEED_ENABLED:
            topics.append(f"allLiquidation.{symbol}")
        return topics

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def _load_settings(self) -> None:

        await self._modules.lifecycle.load_settings()

    async def _configure_observability(self) -> None:

        await self._modules.lifecycle.configure_observability()

    async def _run_preflight(self) -> None:

        await self._modules.lifecycle.run_preflight()

    async def _start_trade_journal(self) -> None:

        await self._modules.lifecycle.start_trade_journal()

    async def _run_trade_journal_reconnector(self) -> None:
        """Keep trying Postgres after transient Render startup/network failures."""
        await self._modules.ops.run_trade_journal_reconnector()

    async def _restore_execution_pending_entries(self) -> None:

        await self._modules.lifecycle.restore_execution_pending_entries()

    # ------------------------------------------------------------------
    # HTTP state proxy
    # ------------------------------------------------------------------

    def _make_state_proxy(self) -> _AppStateProxy:
        return AppStateProxy(self)

    async def _start_http_server(self) -> asyncio.Task[Any]:

        return await self._modules.lifecycle.start_http_server()

    async def _start_bybit_adapter(self) -> None:

        await self._modules.lifecycle.start_bybit_adapter()

    # ------------------------------------------------------------------
    # Operator control callbacks (wired into TradingController)
    # ------------------------------------------------------------------

    async def _pause_trading(self) -> None:

        await self._modules.operator.pause_trading()

    async def _resume_trading(self) -> None:

        await self._modules.operator.resume_trading()

    async def _set_shadow_mode(self, enabled: bool) -> None:

        await self._modules.operator.set_shadow_mode(enabled)

    def _active_execution_allowed(self) -> bool:
        return self._modules.signal_policy.active_execution_allowed()

    def _initial_shadow_mode(self) -> bool:
        return self._modules.signal_policy.initial_shadow_mode()

    def _is_scalp_profile(self) -> bool:
        return self._modules.signal_policy.is_scalp_profile()

    def _scalp_strict_shadow(self) -> bool:
        return self._modules.signal_policy.scalp_strict_shadow()

    def _expectancy_gates_apply(self) -> bool:
        return self._modules.signal_policy.expectancy_gates_apply()

    async def _change_risk_profile(self, profile: Any) -> None:

        await self._modules.operator.change_risk_profile(profile)

    async def _emergency_stop(self) -> None:

        await self._modules.operator.emergency_stop()

    async def _start_model_training(self, min_samples: int = 500, horizon: int = 15, label_bps: float = 5.0) -> str:

        return await self._modules.operator.start_model_training(min_samples, horizon, label_bps)

    async def _start_model_training_all(self) -> str:

        return await self._modules.operator.start_model_training_all()

    async def _run_model_training_all(self) -> None:
        """Run training sequentially for all horizons using all available labeled data."""
        await self._modules.training.run_model_training_all()

    async def _start_model_promote(self, version: str) -> str:

        return await self._modules.operator.start_model_promote(version)

    async def _run_model_training(self, min_samples: int, horizon: int, label_bps: float) -> None:
        await self._modules.training.run_model_training(min_samples, horizon, label_bps)

    async def _run_auto_model_trainer(self) -> None:
        """Automatically train a shadow challenger when enough new labels accumulate."""
        await self._modules.training.run_auto_model_trainer()

    async def _get_champion_walk_forward_bps(self) -> float:
        """Return current champion's walk-forward expectancy stored in model_versions.metrics."""
        if self._trade_journal is None:
            return 0.0
        try:
            rows = await self._trade_journal._fetch(
                """
                SELECT metrics FROM model_versions
                WHERE status = 'CHAMPION' AND metrics IS NOT NULL
                ORDER BY training_finished_at DESC NULLS LAST
                LIMIT 1
                """
            )
            if not rows:
                return 0.0
            metrics_raw = rows[0]["metrics"] or {}
            metrics = dict(metrics_raw) if not isinstance(metrics_raw, str) else json.loads(metrics_raw)
            return float(
                metrics.get("walk_forward_expectancy_bps")
                or metrics.get("best_threshold_avg_net_return_bps")
                or metrics.get("avg_net_return_predicted_positive_bps")
                or 0.0
            )
        except Exception as exc:
            log.debug("model_auto_promote.champion_metrics_failed", error=str(exc))
            return 0.0

    async def _run_auto_model_promoter(self) -> None:
        """Promote the best eligible challenger and roll back degraded champions."""
        await self._modules.training.run_auto_model_promoter()

    async def _run_model_progress_reporter(self) -> None:
        """Send an hourly Telegram report on model training progress and promotion readiness."""
        await self._modules.training.run_model_progress_reporter()

    def _model_gate_threshold(self, regime_context: Any | None) -> float:
        return self._modules.signal_policy.model_gate_threshold(regime_context)

    def _update_model_gate_quality_from_diag(self, diag: dict[str, Any]) -> None:
        self._modules.signal_policy.update_model_gate_quality_from_diag(diag)

    @staticmethod
    def _dict_or_empty(value: Any) -> dict[str, Any]:
        return DiagnosticsModule.dict_or_empty(value)

    def _model_gate_quality_allows_canary(self) -> tuple[bool, str]:
        return self._modules.signal_policy.model_gate_quality_allows_canary()

    def _model_gate_canary_blocks(self, gate_decision: str, threshold: float, score: float) -> tuple[bool, str]:
        return self._modules.signal_policy.model_gate_canary_blocks(gate_decision, threshold, score)

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        return DiagnosticsModule.float_or_none(value)

    @staticmethod
    def _utc_age_seconds(value: Any) -> float | None:
        return DiagnosticsModule.utc_age_seconds(value)

    def _economic_readiness_report(
        self,
        *,
        db_diag: dict[str, Any],
        runtime_diag: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._modules.diagnostics.economic_readiness_report(db_diag=db_diag, runtime_diag=runtime_diag)

    async def _enforce_economic_readiness_for_active(self) -> None:
        await self._modules.diagnostics.enforce_economic_readiness_for_active()

    @staticmethod
    def _feature_values_for_side(vec: FeatureVector, side: str) -> tuple[list[str], list[float]]:
        return SignalPolicyModule.feature_values_for_side(vec, side)

    def _runtime_settings(self) -> dict[str, Any]:

        return self._modules.operator.runtime_settings()

    async def _set_runtime_setting(self, key: str, value: Any) -> str:

        return await self._modules.operator.set_runtime_setting(key, value)

    def _symbol_candidates(self) -> list[str]:

        return self._modules.operator.symbol_candidates()

    def _selected_symbols(self) -> list[str]:

        return self._modules.operator.selected_symbols()

    async def _toggle_manual_symbol(self, symbol: str) -> str:

        return await self._modules.operator.toggle_manual_symbol(symbol)

    # ------------------------------------------------------------------

    def _resolve_telegram_delivery(self) -> tuple[str, str]:
        return self._modules.telegram.resolve_delivery()

    async def _start_telegram_bot(self) -> None:
        await self._modules.telegram.start()

    # ------------------------------------------------------------------
    # Risk & Execution
    # ------------------------------------------------------------------

    async def _init_risk_manager(self, initial_capital: Decimal) -> None:
        await self._modules.execution.init_risk_manager(initial_capital)

    async def _refresh_balance(self) -> Decimal:
        return await self._modules.execution.refresh_balance()

    async def _init_execution_engine(self) -> None:
        await self._modules.execution.init_execution_engine()

    async def _on_screener_symbols_added(self, symbols: list[str]) -> None:
        """Seed candles and subscribe WebSocket for newly added screener symbols."""
        await self._modules.market_data.on_screener_symbols_added(symbols)

    async def _on_screener_symbols_removed(self, symbols: list[str]) -> None:
        await self._modules.market_data.on_screener_symbols_removed(symbols)

    async def _start_screener(self) -> list[str]:
        """Run the market screener and return initial symbol list."""
        return await self._modules.market_data.start_screener()

    # ------------------------------------------------------------------
    # Market data & features
    # ------------------------------------------------------------------

    async def _seed_candle_store(self, symbols: list[str] | None = None) -> None:
        """Fetch recent historical klines via REST to seed the CandleStore."""
        await self._modules.market_data.seed_candle_store(symbols)

    async def _reconcile_unconfirmed_candles(self) -> None:
        """Backfill candles that have become confirmed since the last write.

        Unconfirmed candles are never persisted (look-ahead bias guard), so a WS
        gap or a restart mid-bar can leave holes. Every 5 minutes this re-fetches
        the most recent klines via REST and upserts only those whose close_time
        has already passed (confirmed by clock, not by stream).
        """
        await self._modules.market_data.reconcile_unconfirmed_candles()

    async def _run_startup_backfill(self) -> None:
        """One-shot historical candle backfill at startup.

        With a fresh/cleared DB the canary checklist needs ~1000 1m candles and
        model training needs labelled history — waiting for WS alone takes many
        hours. This pages back through REST klines for the active symbols and
        persists clock-confirmed candles only, respecting a hard request cap.
        Idempotent: upsert_market_candle deduplicates on (symbol, interval, open_time).

        Behaviour:
        - Waits for the screener to publish its first symbol universe (so the
          backfill targets real trading symbols, not the static fallback list).
        - Waits up to 60s for the DB connection (it may still be bootstrapping).
        - Skips (symbol, interval) pairs whose stored history already covers
          >= 90% of the requested window — restarts cost near-zero REST quota.
        - Never raises: a backfill failure must not take down the supervisor.
        """
        await self._modules.market_data.run_startup_backfill()

    async def _startup_backfill(self) -> None:
        await self._modules.market_data.startup_backfill()

    async def _start_public_ws(self, symbols: list[str]) -> None:
        """Start the public WebSocket and wire events to CandleStore."""
        await self._modules.market_data.start_public_ws(symbols)

    async def _start_private_ws(self) -> None:
        await self._modules.execution.start_private_ws()

    async def _run_load_governor(self) -> None:
        """Adaptive load governor: reduce feature symbols when system is under pressure.

        Monitors event-loop lag and WS queue utilisation every
        LOAD_GOVERNOR_CHECK_SECONDS. When any metric exceeds its threshold,
        the screener's feature universe is narrowed by one symbol (down to the
        configured minimum). When all metrics are healthy the universe is
        gradually restored toward the original maximum.
        """
        await self._modules.market_data.run_load_governor()

    async def _run_symbol_subscribe_watchdog(self) -> None:
        """Retry or reconnect WS when screener symbols never receive 1m klines."""
        await self._modules.market_data.run_symbol_subscribe_watchdog()

    async def _evaluate_feature_drift(self) -> dict[str, Any]:
        return await self._modules.training.evaluate_feature_drift()

    async def _maybe_apply_online_learning(self) -> None:
        await self._modules.training.maybe_apply_online_learning()

    async def _maybe_run_startup_retention(self) -> None:
        """One-shot purge after Postgres connects to trim historical bloat."""
        await self._modules.ops.maybe_run_startup_retention()

    async def _run_data_retention(self) -> None:
        await self._modules.ops.run_data_retention()

    async def _run_outcome_resolver(self) -> None:
        """Resolve prediction outcomes by comparing feature snapshot prices with market_candles."""
        await self._modules.ops.run_outcome_resolver()

    async def _run_risk_monitor(self) -> None:
        await self._modules.execution.run_risk_monitor()

    async def _maybe_recover_stale_ws(self, market_data_age_s: float) -> None:
        await self._modules.execution.maybe_recover_stale_ws(market_data_age_s)

    async def _run_reconciliation(self) -> None:
        """Periodic reconciliation: compare local order state with exchange."""
        await self._modules.ops.run_reconciliation()

    async def _run_transaction_log_sync(self) -> None:
        """Periodically sync Bybit transaction log outside the hot strategy loop."""
        await self._modules.ops.run_transaction_log_sync()

    async def _start_feature_pipeline(self) -> None:

        await self._modules.lifecycle.start_feature_pipeline()

    async def _refresh_closed_pnl_memory(self) -> None:
        await self._modules.execution.refresh_closed_pnl_memory()

    async def _manage_open_positions(self) -> None:
        await self._modules.execution.manage_open_positions()

    async def _sync_transaction_log(self) -> None:
        """Sync Bybit transaction log to database — supports pagination up to 5 pages."""
        await self._modules.ops.sync_transaction_log()

    async def _get_net_results(self) -> dict[str, Any]:
        """Provide daily net PnL for Telegram /net command."""
        if self._trade_journal is None:
            return {}
        return cast(dict[str, Any], await self._trade_journal.get_daily_net_results())

    async def _sync_execution_positions(self) -> None:
        await self._modules.execution.sync_execution_positions()

    def _cache_exchange_positions(self, positions: list[Any]) -> None:
        self._modules.execution.cache_exchange_positions(positions)

    def _cache_exchange_position_update(self, position: Any) -> None:
        self._modules.execution.cache_exchange_position_update(position)

    def _recent_exchange_positions(self) -> list[Any] | None:
        return self._modules.execution.recent_exchange_positions()

    def _effective_performance_blocks(self, active_symbols: list[str]) -> set[str]:
        return self._modules.execution.effective_performance_blocks(active_symbols)

    def _activation_price(self, entry_price: Decimal, side: str) -> Decimal:
        return ExecutionRuntimeModule(self).activation_price(entry_price, side)

    def _breakeven_stop(self, entry_price: Decimal, side: str, fee_rates: Any | None = None) -> Decimal:
        return ExecutionRuntimeModule(self).breakeven_stop(entry_price, side, fee_rates)

    def _round_to_tick(
        self,
        price: Decimal,
        tick_size: Decimal,
        *,
        round_up: bool,
    ) -> Decimal:
        return ExecutionRuntimeModule(self).round_to_tick(price, tick_size, round_up=round_up)

    def _record_diag(self, event: str) -> None:
        self._modules.diagnostics.record(event)

    def _top_blocker_from_diag(self, diag: dict[str, Any], *, default: str) -> tuple[str, dict[str, int]]:
        return self._modules.diagnostics.top_blocker_from_diag(diag, default=default)

    async def _sample_confirmed_candle(self, symbol: str, interval: str, vec: Any) -> None:
        await self._modules.signal_policy.sample_confirmed_candle(symbol, interval, vec)

    def _bucket_blocked(self, regime_ctx: Any) -> bool:
        return self._modules.signal_policy.bucket_blocked(regime_ctx)

    def _symbol_side_blocked(self, symbol: str, side: str) -> bool:
        return self._modules.signal_policy.symbol_side_blocked(symbol, side)

    def _shadow_probe_side_blocked(self, symbol: str, side: str) -> bool:
        return self._modules.signal_policy.shadow_probe_side_blocked(symbol, side)

    def _shadow_probe_quality_allows(self, symbol: str, side: str) -> bool:
        return self._modules.signal_policy.shadow_probe_quality_allows(symbol, side)

    def _shadow_probe_symbol_allowed(self, symbol: str) -> bool:
        return self._modules.signal_policy.shadow_probe_symbol_allowed(symbol)

    def _shadow_probe_symbol_warmed_up(self, symbol: str) -> bool:
        return self._modules.signal_policy.shadow_probe_symbol_warmed_up(symbol)

    def _shadow_probe_regime_allows(self, regime_ctx: Any | None) -> bool:
        return self._modules.signal_policy.shadow_probe_regime_allows(regime_ctx)

    def _record_shadow_probe_symbol_subscribed(self, symbols: list[str]) -> None:
        self._modules.signal_policy.record_shadow_probe_symbol_subscribed(symbols)

    def _record_shadow_close(self, symbol: str, reason: str, pnl_pct: float) -> None:
        self._modules.signal_policy.record_shadow_close(symbol, reason, pnl_pct)

    @staticmethod
    @staticmethod
    def _shadow_exit_hit(position: dict[str, Any], *, high: float, low: float) -> tuple[str, float] | None:
        return SignalPolicyModule.shadow_exit_hit(position, high=high, low=low)

    def _shadow_pnl_pct(self, position: dict[str, Any], exit_price: float) -> float:
        return self._modules.signal_policy.shadow_pnl_pct(position, exit_price)

    def _shadow_gross_pnl_pct(self, position: dict[str, Any], exit_price: float) -> float:
        return self._modules.signal_policy.shadow_gross_pnl_pct(position, exit_price)

    def _shadow_loss_guard_blocks(self) -> bool:
        return self._modules.signal_policy.shadow_loss_guard_blocks()

    def _trend_confirmation_intervals(self) -> list[str]:
        return self._modules.signal_policy.trend_confirmation_intervals()

    def _trend_mtf_confirmed(self, symbol: str, side: str) -> bool:
        return self._modules.signal_policy.trend_mtf_confirmed(symbol, side)

    async def _run_bucket_stats_refresher(self) -> None:
        """Refresh in-memory expectancy gates from Postgres periodically."""
        await self._modules.training.run_bucket_stats_refresher()

    def _check_zero_trading(self) -> None:
        self._modules.diagnostics.check_zero_trading()

    def _runtime_candle_readiness_counts(self) -> dict[str, int]:
        return self._modules.diagnostics.runtime_candle_readiness_counts()

    def _merge_runtime_db_diag_fallbacks(self, diag: dict[str, Any]) -> None:
        self._modules.diagnostics.merge_db_fallbacks(diag)

    def get_diagnostics(self) -> dict[str, Any]:
        return self._modules.diagnostics.get_snapshot()

    async def _run_supervisor(self) -> None:
        """Monitor critical background tasks; on unexpected exit alert + exit(1)."""
        await self._modules.supervisor.run()

    async def _start_strategy_loop(self) -> None:
        """Run strategy ensemble → RiskManager → ExecutionEngine."""
        await self._trading_loop.start()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _main_loop(self) -> None:
        assert self._settings is not None
        self._status = SystemStatus.RUNNING

        if self._health_checker:
            self._health_checker.set_system_status(self._status)

        log.info(
            "trading_system_running",
            trading_mode=self._settings.TRADING_MODE,
            risk_profile=self._settings.RISK_PROFILE,
            live_mode=self._settings.LIVE_MODE,
            shadow_mode=self._settings.SHADOW_MODE,
            symbols=self._active_symbols(),
        )

        # Wait for shutdown
        await self._shutdown_event.wait()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _handle_signal(self, sig: int) -> None:
        log.warning("shutdown_signal_received", signal=signal.Signals(sig).name)
        self._shutdown_event.set()

    async def _graceful_shutdown(self) -> None:

        await self._modules.lifecycle.graceful_shutdown()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_signal, sig)

        try:
            await self._load_settings()
            await self._configure_observability()
            await self._run_preflight()

            await self._start_trade_journal()
            await self._start_http_server()
            await self._start_bybit_adapter()
            await self._start_telegram_bot()
            if self._telegram_bot is not None and hasattr(self._telegram_bot, "refresh_delivery"):

                async def _early_webhook_refresh() -> None:
                    await asyncio.sleep(8.0)
                    try:
                        await self._telegram_bot.refresh_delivery()
                    except Exception as tg_refresh_exc:
                        log.warning("telegram.early_refresh_failed", error=str(tg_refresh_exc))

                self._background_tasks.append(
                    asyncio.create_task(_early_webhook_refresh(), name="telegram-webhook-refresh")
                )

            # Start private WebSocket for real-time order/position/balance events
            await self._start_private_ws()

            # Market data pipeline
            # 1. Screen market to get dynamic symbol list
            active_symbols = await self._start_screener()

            # 2. Seed historical data for all selected symbols
            await self._seed_candle_store(symbols=active_symbols)

            # 3. Start WS with the screened symbol list
            await self._start_public_ws(symbols=active_symbols)

            # Give WS a moment to connect before starting strategies
            await asyncio.sleep(3.0)

            await self._start_feature_pipeline()

            # Give features a moment to compute from seeded data
            await asyncio.sleep(2.0)

            await self._enforce_economic_readiness_for_active()

            await self._trading_loop.start()

            # Supervisor + background loops via pluggable modules
            self._modules.spawn_background_tasks(self._background_tasks)

            # Risk monitor: updates equity/drawdown, checks WS staleness
            risk_monitor_task = asyncio.create_task(self._run_risk_monitor(), name="risk-monitor")
            self._background_tasks.append(risk_monitor_task)

            if self._telegram_bot is not None and hasattr(self._telegram_bot, "refresh_delivery"):
                try:
                    await self._telegram_bot.refresh_delivery()
                except Exception as tg_refresh_exc:
                    log.warning("telegram.refresh_delivery_failed", error=str(tg_refresh_exc))

            try:
                await self._main_loop()
            finally:
                await self._graceful_shutdown()

        except SystemExit:
            raise
        except Exception as exc:
            log.critical(
                "unhandled_exception_in_main",
                error=str(exc),
                error_type=type(exc).__name__,
                exc_info=True,
            )
            self._status = SystemStatus.ERROR
            raise


__all__ = [
    "TradingApplication",
    "AppStateProxy",
    "_AppStateProxy",
    "_CRITICAL_TASK_NAMES",
    "_DIAG_WINDOW",
    "_FALLBACK_BALANCE_USD",
    "_INTERVAL_MS",
    "_JOURNAL_FALLBACK_UUID",
    "_SYMBOLS",
    "_WS_INTERVAL",
    "main",
    "main_sync",
]


async def main() -> None:
    app = TradingApplication()
    await app.run()


def main_sync() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
    except SystemExit as e:
        sys.exit(e.code)


if __name__ == "__main__":
    main_sync()
