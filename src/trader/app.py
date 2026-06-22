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
import html
import json
import os
import signal
import sys
from collections import deque
from datetime import datetime
from decimal import Decimal
from typing import Any, cast

import uvicorn

from trader.domain.enums import SystemStatus, TradingMode
from trader.domain.models import FeatureVector
from trader.modules.diagnostics import DiagnosticsModule
from trader.modules.execution_runtime import ExecutionRuntimeModule
from trader.modules.registry import ModuleRegistry
from trader.modules.signal_policy import SignalPolicyModule
from trader.modules.trading_loop import TradingLoopModule
from trader.monitoring.logging import configure_logging, get_logger
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
        from trader.config import Settings

        self._settings = Settings()

        if self._settings.TRADING_MODE == TradingMode.LIVE and not self._settings.LIVE_MODE:
            log.critical(
                "live_mode_safety_gate_blocked",
                reason="LIVE_MODE env var must be explicitly set to true",
            )
            raise SystemExit(1)

    async def _configure_observability(self) -> None:
        assert self._settings is not None
        configure_logging(
            log_level=self._settings.LOG_LEVEL,
            log_format=self._settings.LOG_FORMAT,
        )
        self._current_risk_profile_str = self._settings.RISK_PROFILE.value
        from trader.monitoring.deploy_info import get_deploy_info
        from trader.training.labels import active_label_schema_version

        deploy = get_deploy_info()
        log.info(
            "settings_loaded",
            trading_mode=self._settings.TRADING_MODE,
            risk_profile=self._settings.RISK_PROFILE,
            bybit_use_testnet=self._settings.BYBIT_USE_TESTNET,
            live_mode=self._settings.LIVE_MODE,
            deploy_id=deploy.get("deploy_id") or None,
            git_commit=deploy.get("git_commit") or None,
            train_strategy_allowlist=self._settings.TRAIN_STRATEGY_ALLOWLIST,
            train_include_candle_baseline=self._settings.TRAIN_INCLUDE_CANDLE_BASELINE,
            model_label_use_tpsl_exit=self._settings.MODEL_LABEL_USE_TPSL_EXIT,
            active_label_schema=active_label_schema_version(
                use_tpsl_exit=bool(self._settings.MODEL_LABEL_USE_TPSL_EXIT)
            ),
        )

    async def _run_preflight(self) -> None:
        from trader.exchange.endpoint_selector import EndpointSelector
        from trader.monitoring.health import HealthChecker

        assert self._settings is not None
        self._status = SystemStatus.PREFLIGHT

        bybit_base = EndpointSelector(
            self._settings.BYBIT_REGION,
            self._settings.BYBIT_USE_TESTNET,
        ).rest_base

        postgres_required = (
            self._settings.PREFLIGHT_POSTGRES_REQUIRED
            if self._settings.PREFLIGHT_POSTGRES_REQUIRED is not None
            else self._settings.TRADING_MODE in (TradingMode.CANARY_LIVE, TradingMode.LIVE)
        )

        self._health_checker = HealthChecker(
            postgres_dsn=self._settings.POSTGRES_DSN.get_secret_value(),
            redis_url=self._settings.REDIS_URL.get_secret_value(),
            redis_required=self._settings.REDIS_REQUIRED,
            bybit_required=self._settings.BYBIT_CONNECTIVITY_REQUIRED,
            bybit_rest_url=bybit_base,
            trading_mode=self._settings.TRADING_MODE,
            system_status=self._status,
            model_enabled=self._settings.MODEL_ENABLED,
            postgres_retry_attempts=self._settings.PREFLIGHT_POSTGRES_RETRY_ATTEMPTS,
            postgres_retry_delay_s=self._settings.PREFLIGHT_POSTGRES_RETRY_DELAY_SECONDS,
            postgres_required=postgres_required,
            postgres_optional_max_attempts=self._settings.PREFLIGHT_POSTGRES_OPTIONAL_MAX_ATTEMPTS,
        )

        result = await self._health_checker.run_preflight()
        checks = result["checks"]
        postgres_required = bool(result.get("postgres_required", True))

        for check_name, passed in checks.items():
            if passed:
                log.info("preflight_check_passed", check=check_name)
            elif check_name == "postgres" and not postgres_required:
                log.warning(
                    "preflight_check_deferred",
                    check=check_name,
                    trading_mode=self._settings.TRADING_MODE.value,
                    hint="continuing startup without postgres",
                )
            else:
                log.error("preflight_check_failed", check=check_name)

        if not result["passed"]:
            log.critical("preflight_failed", checks=checks)
            raise SystemExit(1)

        if not checks.get("postgres") and not postgres_required:
            log.warning(
                "preflight_postgres_optional_continuing",
                trading_mode=self._settings.TRADING_MODE.value,
            )

        log.info("preflight_passed")

    async def _start_trade_journal(self) -> None:
        """Start best-effort Postgres memory for trades and performance."""
        from trader.storage.trade_journal import TradeJournal

        assert self._settings is not None
        self._trade_journal = TradeJournal(
            postgres_dsn=self._settings.POSTGRES_DSN.get_secret_value(),
            enabled=self._settings.TRADE_JOURNAL_ENABLED,
            fetch_timeout_seconds=self._settings.TRADE_JOURNAL_FETCH_TIMEOUT_SECONDS,
            pool_max_size=self._settings.TRADE_JOURNAL_POOL_MAX_SIZE,
            reconnect_max_backoff_seconds=self._settings.TRADE_JOURNAL_RECONNECT_MAX_BACKOFF_SECONDS,
            auth_circuit_breaker_min_backoff_seconds=self._settings.TRADE_JOURNAL_AUTH_CIRCUIT_BREAKER_MIN_BACKOFF_SECONDS,
        )
        await self._trade_journal.connect()
        await self._maybe_run_startup_retention()
        task = asyncio.create_task(self._run_trade_journal_reconnector(), name="trade-journal-reconnector")
        self._background_tasks.append(task)

    async def _run_trade_journal_reconnector(self) -> None:
        """Keep trying Postgres after transient Render startup/network failures."""
        await self._modules.ops.run_trade_journal_reconnector()

    async def _restore_execution_pending_entries(self) -> None:
        """Reload unresolved durable pending entries into ExecutionEngine."""
        if self._initial_shadow_mode():
            return
        if self._trade_journal is None or self._execution_engine is None or not self._trade_journal.is_enabled:
            return
        try:
            pending_records = await self._trade_journal.get_pending_durable_orders()
            unresolved_records = []
            skipped_resolved = []
            for record in pending_records:
                oid = str(record.get("order_link_id") or "")
                if oid and await self._trade_journal.is_order_resolved(oid):
                    skipped_resolved.append(oid)
                    continue
                unresolved_records.append(record)
            if skipped_resolved:
                log.info(
                    "execution_engine.pending_restore_skipped_resolved",
                    ids=skipped_resolved,
                )
            if unresolved_records:
                self._execution_engine.restore_pending_entries_with_symbols(unresolved_records)
                log.info(
                    "execution_engine.pending_restored",
                    count=len(unresolved_records),
                    ids=[r.get("order_link_id") for r in unresolved_records],
                )
        except Exception as exc:
            log.warning("execution_engine.pending_restore_failed", error=str(exc))

    # ------------------------------------------------------------------
    # HTTP state proxy
    # ------------------------------------------------------------------

    def _make_state_proxy(self) -> _AppStateProxy:
        return AppStateProxy(self)

    async def _start_http_server(self) -> asyncio.Task[Any]:
        from trader.api.fastapi_app import create_app

        assert self._settings is not None
        import secrets

        internal_api_key = self._settings.INTERNAL_API_KEY.get_secret_value()
        if not internal_api_key:
            internal_api_key = secrets.token_urlsafe(32)
            log.warning(
                "http_server.generated_internal_api_key",
                reason="INTERNAL_API_KEY is not configured; authenticated endpoints are only usable inside this process",
            )
        port = int(os.getenv("PORT", str(self._settings.FASTAPI_PORT)))
        log.info("http_server_starting", port=port)

        fastapi_app = create_app(
            api_key=internal_api_key,
            health_checker=self._health_checker,
            state_store=_AppStateProxy(self),
            trade_journal=self._trade_journal,
            runtime_settings=self._runtime_settings,
            set_runtime_setting=self._set_runtime_setting,
        )
        self._fastapi_app = fastapi_app

        config = uvicorn.Config(
            app=fastapi_app,
            # Container service must bind internally; external exposure belongs to the platform.
            host="0.0.0.0",  # noqa: S104  # nosec B104
            port=port,
            log_level="warning",
            access_log=False,
        )
        self._uvicorn_server = uvicorn.Server(config=config)
        task = asyncio.create_task(self._uvicorn_server.serve(), name="http-server")
        self._background_tasks.append(task)
        return task

    async def _start_bybit_adapter(self) -> None:
        from trader.exchange.bybit_adapter import BybitAdapter

        assert self._settings is not None
        self._bybit_adapter = BybitAdapter(
            api_key=self._settings.BYBIT_API_KEY.get_secret_value(),
            api_secret=self._settings.BYBIT_API_SECRET.get_secret_value(),
            region_code=self._settings.BYBIT_REGION.value,
            use_testnet=self._settings.BYBIT_USE_TESTNET,
            default_category=self._settings.DEFAULT_MARKET_CATEGORY,
            trade_journal=self._trade_journal,
            trading_mode=self._settings.TRADING_MODE.value,
        )
        log.info("bybit_adapter_created", category=self._settings.DEFAULT_MARKET_CATEGORY)

        from trader.exchange.fee_provider import FeeRateProvider

        self._fee_provider = FeeRateProvider(
            rest=self._bybit_adapter._rest,
            category=self._settings.DEFAULT_MARKET_CATEGORY,
            default_maker=self._settings.DEFAULT_LINEAR_MAKER_FEE_RATE,
            default_taker=self._settings.DEFAULT_LINEAR_TAKER_FEE_RATE,
            shadow_mode=self._initial_shadow_mode(),
        )

        # Run Bybit exchange preflight checks (clock skew, API perms, balance, etc.)
        has_key = bool(self._settings.BYBIT_API_KEY.get_secret_value())
        if has_key:
            try:
                report = await self._bybit_adapter.initialize()
                is_live = self._settings.LIVE_MODE and self._settings.TRADING_MODE in (
                    TradingMode.LIVE,
                    TradingMode.CANARY_LIVE,
                )
                if not report.passed:
                    if is_live:
                        log.critical(
                            "bybit_preflight_failed_blocking_live",
                            errors=report.errors,
                        )
                        raise SystemExit(1)
                    else:
                        log.warning(
                            "bybit_preflight_partial_continuing_shadow",
                            errors=report.errors,
                            warnings=report.warnings,
                        )
                else:
                    log.info("bybit_preflight_passed", warnings=report.warnings)
            except SystemExit:
                raise
            except Exception as exc:
                # P0.7: exception during preflight is fatal for CANARY_LIVE / LIVE
                is_active = self._settings.LIVE_MODE and self._settings.TRADING_MODE in (
                    TradingMode.LIVE,
                    TradingMode.CANARY_LIVE,
                )
                if is_active:
                    log.critical("bybit_preflight_exception_blocking_live", error=str(exc))
                    raise SystemExit(1) from exc
                log.warning("bybit_preflight_exception_continuing_shadow", error=str(exc))
        else:
            log.info("bybit_adapter_skipped_preflight", reason="no_api_key_configured")

    # ------------------------------------------------------------------
    # Operator control callbacks (wired into TradingController)
    # ------------------------------------------------------------------

    async def _pause_trading(self) -> None:
        self._trading_paused = True
        log.info("trading.paused")

    async def _resume_trading(self) -> None:
        self._trading_paused = False
        log.info("trading.resumed")

    async def _set_shadow_mode(self, enabled: bool) -> None:
        assert self._settings is not None
        if not enabled:
            if not self._active_execution_allowed():
                raise RuntimeError(
                    "Active execution requires BYBIT_USE_TESTNET=true, or LIVE_MODE=true with TRADING_MODE=LIVE/CANARY_LIVE."
                )
        if self._execution_engine is not None:
            self._execution_engine._shadow_mode = enabled
        if self._fee_provider is not None:
            self._fee_provider.shadow_mode = enabled
        log.info("shadow_mode.changed", enabled=enabled)

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
        """Hot-swap the risk profile without restarting — preserves all risk state.

        SAFETY: Blocked in LIVE and CANARY_LIVE modes because a profile change
        alters leverage limits, position caps, and daily-loss thresholds while
        real positions are open — an unsafe combination requiring a clean restart.
        """
        assert self._settings is not None
        if self._settings.TRADING_MODE in (TradingMode.LIVE, TradingMode.CANARY_LIVE):
            raise RuntimeError(
                "Risk profile hot-swap is not permitted in LIVE / CANARY_LIVE mode. "
                "Restart the service to apply a new profile."
            )

        old = self._current_risk_profile_str
        capital = await self._refresh_balance()

        # Preserve ALL risk state that spans profile boundaries.
        # Reinitialising would silently reset peak equity → new hard-stop baseline
        # that ignores losses already taken — a critical safety hole.
        old_drawdown = self._risk_manager._drawdown if self._risk_manager is not None else None
        old_daily_pnl = self._risk_manager.daily_pnl if self._risk_manager is not None else Decimal("0")

        if self._settings is not None:
            self._settings.RISK_PROFILE = profile
        await self._init_risk_manager(capital)

        if self._risk_manager is not None:
            if old_drawdown is not None:
                self._risk_manager._drawdown = old_drawdown
            # Restore daily PnL so daily loss limit is not reset mid-day
            if old_daily_pnl != Decimal("0"):
                self._risk_manager._daily_pnl = old_daily_pnl

        # Rewire execution engine to the new risk manager
        if self._execution_engine is not None:
            self._execution_engine._risk_manager = self._risk_manager
        self._current_risk_profile_str = profile.value
        log.info("risk_profile.changed", old=old, new=profile.value)
        if self._telegram_bot is not None:
            await self._telegram_bot.notify_risk_changed(old, profile.value)

    async def _emergency_stop(self) -> None:
        self._trading_paused = True
        if self._kill_switch is not None:
            from trader.domain.enums import KillSwitchMode

            await self._kill_switch.activate(
                KillSwitchMode.FULL_STOP,
                reason="operator emergency stop via Telegram",
                operator="telegram",
            )
        log.critical("emergency_stop.activated", source="telegram")
        if self._telegram_bot is not None:
            await self._telegram_bot.notify(
                "🚨 <b>Emergency stop activated.</b> No new trades. Manual restart required."
            )

    async def _start_model_training(self, min_samples: int = 500, horizon: int = 15, label_bps: float = 5.0) -> str:
        """Start offline model training in a subprocess; trading loop stays isolated."""
        async with self._training_start_lock:
            if self._training_task is not None and not self._training_task.done():
                return "⏳ Обучение уже идет."
            if self._trade_journal is not None and not self._trade_journal.is_enabled:
                await self._trade_journal.reconnect_if_needed(force=True)
            if self._trade_journal is None or not self._trade_journal.is_enabled:
                raise RuntimeError("Trade journal/Postgres is not available.")
            self._training_task = asyncio.create_task(
                self._run_model_training(min_samples, horizon, label_bps),
                name="model-training",
            )
            self._background_tasks.append(self._training_task)
        return (
            "🧠 <b>Обучение запущено</b>\n"
            f"минимум примеров=<code>{min_samples}</code>, горизонт=<code>{horizon}m</code>, "
            f"порог=<code>{label_bps:g} bps</code>\n"
            "Результат придет сюда после завершения."
        )

    async def _start_model_training_all(self) -> str:
        """Start sequential training on all available data for every horizon (5m, 15m, 30m, 60m)."""
        async with self._training_start_lock:
            if self._training_task is not None and not self._training_task.done():
                return "⏳ Обучение уже идет."
            if self._trade_journal is not None and not self._trade_journal.is_enabled:
                await self._trade_journal.reconnect_if_needed(force=True)
            if self._trade_journal is None or not self._trade_journal.is_enabled:
                raise RuntimeError("Trade journal/Postgres is not available.")
            self._training_task = asyncio.create_task(
                self._run_model_training_all(),
                name="model-training-all",
            )
            self._background_tasks.append(self._training_task)
        return (
            "🧠🔁 <b>Обучение ВСЕ запущено</b>\n"
            f"Горизонты: <code>5m, 15m, 30m, 60m</code> | Порог: <code>{self._settings.MODEL_AUTO_TRAIN_LABEL_BPS} bps</code>\n"
            "Используются все доступные примеры (мин. 100).\n"
            "Результаты придут по мере завершения каждого горизонта."
        )

    async def _run_model_training_all(self) -> None:
        """Run training sequentially for all horizons using all available labeled data."""
        await self._modules.training.run_model_training_all()

    async def _start_model_promote(self, version: str) -> str:
        """Promote a model through the same strict engine used by auto-promotion."""
        if self._trade_journal is None or not self._trade_journal.is_enabled:
            raise RuntimeError("Trade journal/Postgres is not available.")

        def code_text(value: str, limit: int = 800) -> str:
            return html.escape(value[-limit:])

        log.info("model_promote.started", version=version)
        try:
            from trader.ml.auto_promotion import AutoPromotionConfig, AutoPromotionEngine

            async def _reload_registry() -> None:
                if self._model_registry is not None:
                    await self._model_registry.load_active_model()

            engine = AutoPromotionEngine(
                trade_journal=self._trade_journal,
                config=AutoPromotionConfig.from_settings(self._settings),
                reload_registry=_reload_registry,
            )
            decision = await asyncio.wait_for(engine.promote(version), timeout=60.0)
            if decision.promote:
                message = f"Model {version} promoted to CHAMPION: {', '.join(decision.reasons)}"
                if self._telegram_bot is not None:
                    await self._telegram_bot.notify(
                        f"🏆 <b>Модель промоутирована</b>\n<code>{code_text(message)}</code>"
                    )
                return f"🏆 <b>Промоут успешен!</b>\n<code>{code_text(message)}</code>"
            out = "; ".join(decision.reasons)
            if self._telegram_bot is not None:
                await self._telegram_bot.notify(f"❌ <b>Промоут не прошёл</b>\n<code>{code_text(out)}</code>")
            return f"❌ <b>Промоут не прошёл:</b>\n<code>{code_text(out)}</code>"
        except TimeoutError:
            return "❌ Промоут завис (timeout 60s)"
        except Exception as exc:
            return f"❌ Ошибка промоута: <code>{html.escape(str(exc))}</code>"

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
        from trader.training.labels import active_label_schema_version

        return {
            "paused": self._trading_paused,
            "shadow": self._execution_engine._shadow_mode if self._execution_engine is not None else True,
            "risk_profile": self._current_risk_profile_str,
            "max_entries_per_minute": (
                self._execution_engine._max_entries_per_minute if self._execution_engine is not None else None
            ),
            "max_concurrent_pending": (
                self._execution_engine._max_concurrent_pending if self._execution_engine is not None else None
            ),
            "max_same_side": self._execution_engine._max_same_side if self._execution_engine is not None else None,
            "max_positions": (
                self._execution_engine._max_open_positions
                if self._execution_engine is not None
                else (self._settings.MAX_POSITIONS if self._settings is not None else None)
            ),
            "screener_max_price_usd": self._settings.SCREENER_MAX_PRICE_USD if self._settings is not None else None,
            "feature_max_symbols": self._screener._feature_max if self._screener is not None else None,
            "execution_candidates": self._screener._exec_candidates if self._screener is not None else None,
            "manual_symbols": self._selected_symbols(),
            "model_gate_canary_enabled": (
                self._settings.MODEL_GATE_CANARY_ENABLED if self._settings is not None else False
            ),
            "model_gate_threshold": self._settings.MODEL_SHADOW_GATE_THRESHOLD if self._settings is not None else None,
            "model_gate_quality": self._model_gate_quality,
            "model_auto_train_min_samples": (
                self._settings.MODEL_AUTO_TRAIN_MIN_SAMPLES if self._settings is not None else 1000
            ),
            "model_auto_train_horizon_minutes": (
                self._settings.MODEL_AUTO_TRAIN_HORIZON_MINUTES if self._settings is not None else 5
            ),
            "model_auto_train_label_bps": (
                self._settings.MODEL_AUTO_TRAIN_LABEL_BPS if self._settings is not None else 2.0
            ),
            "label_schema_version": (
                active_label_schema_version(use_tpsl_exit=bool(self._settings.MODEL_LABEL_USE_TPSL_EXIT))
                if self._settings is not None
                else "directional_net_v1"
            ),
            "strategy_priority_order": (self._settings.STRATEGY_PRIORITY_ORDER if self._settings is not None else ""),
            "scalp_strategy_priority_order": (
                self._settings.SCALP_STRATEGY_PRIORITY_ORDER if self._settings is not None else ""
            ),
        }

    async def _set_runtime_setting(self, key: str, value: Any) -> str:
        assert self._settings is not None
        key = key.lower()
        if key == "entries":
            ivalue = int(value)
            if not 0 < ivalue <= 10:
                raise ValueError("entries must be 1..10")
            self._settings.MAX_NEW_ENTRIES_PER_MINUTE = ivalue
            if self._execution_engine is not None:
                self._execution_engine._max_entries_per_minute = ivalue
            return f"Max entries/min set to {ivalue}"
        if key == "pending":
            ivalue = int(value)
            if not 0 < ivalue <= 10:
                raise ValueError("pending must be 1..10")
            self._settings.MAX_CONCURRENT_PENDING_ENTRIES = ivalue
            if self._execution_engine is not None:
                self._execution_engine._max_concurrent_pending = ivalue
            return f"Max pending entries set to {ivalue}"
        if key == "same_side":
            ivalue = int(value)
            if not 0 < ivalue <= 10:
                raise ValueError("same_side must be 1..10")
            self._settings.MAX_SAME_SIDE_POSITIONS = ivalue
            if self._execution_engine is not None:
                self._execution_engine._max_same_side = ivalue
            return f"Max same-side positions set to {ivalue}"
        if key == "max_positions":
            ivalue = int(value)
            if not 1 <= ivalue <= 10:
                raise ValueError("max_positions must be 1..10")
            self._settings.MAX_POSITIONS = ivalue
            if self._execution_engine is not None:
                self._execution_engine._max_open_positions = ivalue
            return f"Max simultaneous positions set to {ivalue}"
        if key == "price_cap":
            fvalue = float(value)
            if fvalue < 0 or fvalue > 100_000:
                raise ValueError("price_cap must be 0..100000")
            self._settings.SCREENER_MAX_PRICE_USD = fvalue
            if self._screener is not None:
                self._screener._max_price_usd = fvalue
            return f"Screener price cap set to {fvalue:g}"
        if key == "feature_symbols":
            ivalue = int(value)
            if not 1 <= ivalue <= self._settings.SCREENER_WIDE_MAX_SYMBOLS:
                raise ValueError(f"feature_symbols must be 1..{self._settings.SCREENER_WIDE_MAX_SYMBOLS}")
            self._settings.SCREENER_FEATURE_MAX_SYMBOLS = ivalue
            if self._settings.SCREENER_EXECUTION_CANDIDATES > ivalue:
                self._settings.SCREENER_EXECUTION_CANDIDATES = ivalue
            if self._screener is not None:
                self._screener._feature_max = ivalue
                if self._screener._exec_candidates > ivalue:
                    self._screener._exec_candidates = ivalue
            return f"Feature symbols set to {ivalue}"
        if key == "exec_candidates":
            ivalue = int(value)
            if not 1 <= ivalue <= self._settings.SCREENER_FEATURE_MAX_SYMBOLS:
                raise ValueError(f"exec_candidates must be 1..{self._settings.SCREENER_FEATURE_MAX_SYMBOLS}")
            self._settings.SCREENER_EXECUTION_CANDIDATES = ivalue
            if self._screener is not None:
                self._screener._exec_candidates = ivalue
            return f"Execution candidates set to {ivalue}"
        if key == "model_gate":
            sval = str(value).strip().lower()
            if sval not in {"on", "off", "true", "false", "1", "0"}:
                raise ValueError("model_gate must be on/off")
            if sval in {"on", "true", "1"}:
                raise ValueError(
                    "Canary model gate can only be enabled through environment configuration after manual readiness review."
                )
            self._settings.MODEL_GATE_CANARY_ENABLED = False
            return "Model gate canary remains OFF (runtime enable blocked — use env vars)"
        if key == "model_gate_threshold":
            fvalue = float(value)
            if not 0.50 <= fvalue <= 0.80:
                raise ValueError("model_gate_threshold must be 0.50..0.80")
            self._settings.MODEL_SHADOW_GATE_THRESHOLD = fvalue
            return f"Model gate threshold set to {fvalue:.2f}"
        raise ValueError("unknown setting")

    def _symbol_candidates(self) -> list[str]:
        if self._screener is None:
            return list(_SYMBOLS)
        wide = self._screener.wide_universe
        if wide:
            return [str(item.symbol) for item in wide[:100]]
        return cast(list[str], self._screener.active_symbols)

    def _selected_symbols(self) -> list[str]:
        if self._screener is None:
            return []
        return cast(list[str], self._screener.manual_symbols)

    async def _toggle_manual_symbol(self, symbol: str) -> str:
        if self._screener is None:
            raise RuntimeError("Сканер еще не запущен")
        symbol = symbol.upper()
        if symbol not in set(self._symbol_candidates()):
            raise ValueError(f"{symbol} сейчас не проходит фильтры сканера")

        selected = set(self._screener.manual_symbols)
        if symbol in selected:
            selected.remove(symbol)
            self._screener.set_manual_symbols(sorted(selected))
            return f"☐ <code>{symbol}</code> убрана из ручного списка."

        selected.add(symbol)
        self._screener.set_manual_symbols(sorted(selected))
        if symbol not in self._screener.active_symbols:
            await self._on_screener_symbols_added([symbol])
        return f"✅ <code>{symbol}</code> добавлена: бот будет учиться и торговать по ней, пока она проходит фильтры."

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
        """Start event-driven feature pipeline with 60 s staleness watchdog."""
        from trader.features.pipeline import FeaturePipeline
        from trader.features.regime import RegimeClassifier

        assert self._candle_store is not None

        self._feature_pipeline = FeaturePipeline(
            candle_store=self._candle_store,
            health_checker=self._health_checker,
            stale_threshold_s=90.0,
            watchdog_interval_s=60.0,
            orderbook_tracker=self._orderbook_tracker,
            market_stats_source=self._screener,
        )
        self._regime_classifier = RegimeClassifier()

        task = asyncio.create_task(
            self._feature_pipeline.run(
                symbols=self._active_symbols(),  # actual screener universe (fallback if absent)
                intervals=[_WS_INTERVAL],
                symbol_source=self._screener,
            ),
            name="feature-pipeline",
        )
        self._background_tasks.append(task)
        log.info("feature_pipeline.started", mode="event_driven", watchdog_interval_s=60.0)

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

    def _record_shadow_close(self, symbol: str, reason: str, pnl_pct: float) -> None:
        self._modules.signal_policy.record_shadow_close(symbol, reason, pnl_pct)

    @staticmethod
    @staticmethod
    def _shadow_exit_hit(position: dict[str, Any], *, high: float, low: float) -> tuple[str, float] | None:
        return SignalPolicyModule.shadow_exit_hit(position, high=high, low=low)

    def _shadow_pnl_pct(self, position: dict[str, Any], exit_price: float) -> float:
        return self._modules.signal_policy.shadow_pnl_pct(position, exit_price)

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
        log.info("graceful_shutdown_starting")
        self._status = SystemStatus.STOPPING
        self._trading_paused = True  # pause new entries immediately

        if self._health_checker:
            self._health_checker.set_system_status(self._status)

        if self._feature_pipeline:
            self._feature_pipeline.stop()

        # Run reconciliation before stopping to catch any order state mismatches
        _is_shadow = self._settings is None or self._initial_shadow_mode()
        if self._bybit_adapter is not None and not _is_shadow:
            try:
                result = await asyncio.wait_for(self._bybit_adapter.reconcile(), timeout=10.0)
                log.info(
                    "graceful_shutdown.reconciliation",
                    discrepancies=result.discrepancies_found,
                    summary=result.summary,
                )
            except Exception as exc:
                log.warning("graceful_shutdown.reconciliation_failed", error=str(exc))

        # Log final execution state and open positions before shutdown
        if self._execution_engine is not None:
            status = self._execution_engine.get_status()
            log.info(
                "execution_engine.shutdown_status",
                open_positions=len(status["open_positions"]),
                shadow_mode=status["shadow_mode"],
            )
            # Alert via Telegram about shutdown with open positions
            if status["open_positions"] and self._telegram_bot is not None:
                try:
                    pos_list = ", ".join(status["open_positions"].keys())
                    await self._telegram_bot.notify(
                        f"⚠️ <b>Shutdown with open positions</b>: <code>{pos_list}</code>\n"
                        "Open positions remain on exchange. Verify SL manually."
                    )
                except Exception as exc:
                    log.debug("graceful_shutdown.telegram_failed", error=str(exc))

        if self._telegram_bot:
            await self._telegram_bot.stop()

        if self._ws_public:
            await self._ws_public.stop()

        if self._ws_private:
            await self._ws_private.stop()

        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
            await asyncio.sleep(1)

        if self._bybit_adapter:
            await self._bybit_adapter.close()

        if self._trade_journal:
            await self._trade_journal.close()

        self._status = SystemStatus.STOPPED
        log.info("graceful_shutdown_complete")

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

            await self._start_http_server()
            await self._start_trade_journal()
            await self._start_bybit_adapter()
            await self._start_telegram_bot()

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
