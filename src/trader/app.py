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
import hashlib
import html
import json
import os
import signal
import sys
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal
from typing import Any, cast

import uvicorn

from trader.domain.enums import SystemStatus, TradingMode
from trader.domain.models import FeatureVector
from trader.modules.registry import ModuleRegistry
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
        """Return True when orders may be submitted to the configured endpoint."""
        assert self._settings is not None
        if self._settings.TRADING_MODE == TradingMode.SHADOW:
            return False
        if self._settings.BYBIT_USE_TESTNET:
            return True
        return self._settings.LIVE_MODE and self._settings.TRADING_MODE in (
            TradingMode.LIVE,
            TradingMode.CANARY_LIVE,
        )

    def _initial_shadow_mode(self) -> bool:
        """Compute startup execution mode from settings and safety gates."""
        assert self._settings is not None
        if self._settings.SHADOW_MODE:
            return True
        return not self._active_execution_allowed()

    def _is_scalp_profile(self) -> bool:
        from trader.domain.enums import RiskProfile

        assert self._settings is not None
        return self._settings.RISK_PROFILE == RiskProfile.SCALP

    def _scalp_strict_shadow(self) -> bool:
        """SCALP paper-trading should mirror LIVE quality gates."""
        assert self._settings is not None
        return self._is_scalp_profile() and self._settings.SCALP_STRICT_SHADOW and self._initial_shadow_mode()

    def _expectancy_gates_apply(self) -> bool:
        """True when bucket/symbol-side gates should block entries."""
        if self._scalp_strict_shadow():
            return True
        return not self._initial_shadow_mode()

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
        """Return a conservative threshold adjusted by market regime."""
        assert self._settings is not None
        best_threshold = self._model_gate_quality.get("best_threshold")
        threshold = (
            float(best_threshold) if best_threshold is not None else float(self._settings.MODEL_SHADOW_GATE_THRESHOLD)
        )
        if regime_context is None:
            return threshold + 0.02

        regime = getattr(
            getattr(regime_context, "regime", None),
            "value",
            str(getattr(regime_context, "regime", "")),
        )
        volatility = getattr(
            getattr(regime_context, "volatility_level", None),
            "value",
            str(getattr(regime_context, "volatility_level", "")),
        )
        if regime in {"BULL_TREND", "BEAR_TREND"}:
            threshold -= 0.02
        elif regime in {"SIDEWAYS", "UNCERTAIN"}:
            threshold += 0.03
        elif regime in {"HIGH_VOLATILITY", "LOW_LIQUIDITY"}:
            threshold += 0.05
        if volatility in {"HIGH", "EXTREME"}:
            threshold += 0.03
        elif volatility == "LOW":
            threshold += 0.01
        return min(0.80, max(0.50, threshold))

    def _update_model_gate_quality_from_diag(self, diag: dict[str, Any]) -> None:
        latest_model = self._dict_or_empty(diag.get("latest_model_version"))
        metrics = self._dict_or_empty(latest_model.get("metrics"))
        gate = self._dict_or_empty(diag.get("shadow_gate_15m"))
        self._model_gate_quality = {
            "quality": metrics.get("quality"),
            "lift_bps": metrics.get("lift_bps"),
            "best_threshold": metrics.get("best_threshold"),
            "gate_total_count": gate.get("total_count", 0) or 0,
            "gate_lift_vs_all_bps": gate.get("lift_vs_all_bps"),
        }
        self._model_gate_quality_checked_at = datetime.now(tz=UTC)

    @staticmethod
    def _dict_or_empty(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def _model_gate_quality_allows_canary(self) -> tuple[bool, str]:
        assert self._settings is not None
        if not self._model_gate_quality:
            return False, "quality_unknown"
        expected_quality = str(self._settings.MODEL_GATE_CANARY_MIN_QUALITY).upper()
        quality = str(self._model_gate_quality.get("quality") or "").upper()
        if expected_quality and quality != expected_quality:
            return False, f"quality_not_{expected_quality.lower()}:{quality or 'none'}"
        gate_total = int(self._model_gate_quality.get("gate_total_count") or 0)
        if gate_total < int(self._settings.MODEL_GATE_CANARY_MIN_OBSERVATIONS):
            return False, f"insufficient_gate_observations:{gate_total}"
        lift = self._model_gate_quality.get("gate_lift_vs_all_bps")
        if lift is None or float(lift) < float(self._settings.MODEL_GATE_CANARY_MIN_LIFT_BPS):
            return False, f"insufficient_gate_lift:{lift}"
        return True, "quality_ok"

    def _model_gate_canary_blocks(self, gate_decision: str, threshold: float, score: float) -> tuple[bool, str]:
        """Decide whether observational gate may block execution without starving trades."""
        assert self._settings is not None
        if not self._settings.MODEL_GATE_CANARY_ENABLED:
            return False, "canary_disabled"
        if gate_decision != "GATE_BLOCK":
            self._model_gate_recent_blocks.append(False)
            return False, "gate_pass"
        quality_ok, quality_reason = self._model_gate_quality_allows_canary()
        if not quality_ok:
            self._model_gate_recent_blocks.append(False)
            return False, quality_reason

        recent = list(self._model_gate_recent_blocks)
        block_rate = (sum(recent) / len(recent) * 100.0) if recent else 0.0
        if len(recent) >= self._settings.MODEL_GATE_CANARY_MIN_OBSERVATIONS:
            if block_rate >= self._settings.MODEL_GATE_CANARY_MAX_BLOCK_RATE_PCT:
                self._model_gate_recent_blocks.append(False)
                return False, f"max_block_rate_guard:{block_rate:.1f}%"

        self._model_gate_block_counter += 1
        every_n = max(1, int(self._settings.MODEL_GATE_CANARY_ALLOW_EVERY_NTH_BLOCKED))
        if self._model_gate_block_counter % every_n == 0:
            self._model_gate_recent_blocks.append(False)
            return False, f"sample_through_every_{every_n}"

        self._model_gate_recent_blocks.append(True)
        return True, f"score_below_threshold:{score:.3f}<{threshold:.3f}"

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _utc_age_seconds(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        if not isinstance(value, datetime):
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return float(max(0.0, (datetime.now(tz=UTC) - value.astimezone(UTC)).total_seconds()))

    def _economic_readiness_report(
        self,
        *,
        db_diag: dict[str, Any],
        runtime_diag: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return an operator-facing readiness verdict for real-money modes."""
        assert self._settings is not None
        runtime_diag = runtime_diag or {}
        issues: list[str] = []
        metrics: dict[str, Any] = {}

        if not db_diag.get("connected"):
            issues.append("db_not_connected")

        latest_age = self._utc_age_seconds(db_diag.get("latest_candle_1m"))
        metrics["latest_candle_age_s"] = latest_age
        if latest_age is None or latest_age > 600:
            issues.append(f"stale_1m_candle:{latest_age}")

        active_symbols = runtime_diag.get("active_symbols") or []
        metrics["active_symbols"] = len(active_symbols)
        if len(active_symbols) < 3:
            issues.append(f"insufficient_active_symbols:{len(active_symbols)}")

        feature_snapshots = int(db_diag.get("feature_snapshots") or 0)
        metrics["feature_snapshots"] = feature_snapshots
        if feature_snapshots < 1000:
            issues.append(f"insufficient_feature_snapshots:{feature_snapshots}")

        prediction_outcomes = int(db_diag.get("prediction_outcomes") or 0)
        metrics["prediction_outcomes"] = prediction_outcomes
        if prediction_outcomes < 1000:
            issues.append(f"insufficient_prediction_outcomes:{prediction_outcomes}")

        active_model = self._dict_or_empty(db_diag.get("active_model_version"))
        model_metrics = self._dict_or_empty(active_model.get("metrics"))
        model_status = str(active_model.get("status") or "")
        try:
            model_horizon = int(
                db_diag.get("model_gate_horizon_minutes")
                or model_metrics.get("horizon_minutes")
                or model_metrics.get("label_horizon_minutes")
                or getattr(self._settings, "MODEL_AUTO_TRAIN_HORIZON_MINUTES", 15)
                or 15
            )
        except (TypeError, ValueError):
            model_horizon = 15
        metrics["model_horizon_minutes"] = model_horizon
        training_by_horizon = self._dict_or_empty(db_diag.get("training_eligible_by_horizon"))
        raw_trainable = training_by_horizon.get(str(model_horizon))
        if raw_trainable is None:
            raw_trainable = db_diag.get(f"training_eligible_{model_horizon}m")
        if raw_trainable is None:
            raw_trainable = db_diag.get("training_eligible_15m")
        if raw_trainable is None:
            raw_trainable = db_diag.get("labelled_samples_15m")
        trainable = int(raw_trainable or 0)
        metrics["training_eligible_model_horizon"] = trainable
        if trainable < 1000:
            issues.append(f"insufficient_labelled_{model_horizon}m:{trainable}")
        metrics["active_model_version"] = active_model.get("version")
        metrics["active_model_status"] = model_status or None
        if not active_model.get("version"):
            issues.append("missing_active_model")
        if model_status != "CHAMPION":
            issues.append(f"active_model_not_champion:{model_status or 'none'}")

        expected_quality = str(self._settings.MODEL_GATE_CANARY_MIN_QUALITY).upper()
        quality = str(model_metrics.get("quality") or "").upper()
        metrics["model_quality"] = quality or None
        if expected_quality and quality != expected_quality:
            issues.append(f"model_quality_not_{expected_quality.lower()}:{quality or 'none'}")

        walk_forward = self._float_or_none(model_metrics.get("walk_forward_expectancy_bps"))
        metrics["walk_forward_expectancy_bps"] = walk_forward
        if walk_forward is None or walk_forward <= 0:
            issues.append(f"non_positive_walk_forward_bps:{walk_forward}")

        gate_by_horizon = self._dict_or_empty(db_diag.get("shadow_gate_by_horizon"))
        raw_gate = gate_by_horizon.get(str(model_horizon))
        if raw_gate is None:
            raw_gate = db_diag.get(f"shadow_gate_{model_horizon}m")
        if raw_gate is None:
            raw_gate = db_diag.get("shadow_gate_15m")
        gate = self._dict_or_empty(raw_gate)
        gate_total = int(gate.get("total_count") or 0)
        metrics["gate_total_count"] = gate_total
        if gate_total < int(self._settings.MODEL_GATE_CANARY_MIN_OBSERVATIONS):
            issues.append(f"insufficient_gate_observations:{gate_total}")
        gate_lift = self._float_or_none(gate.get("lift_vs_all_bps"))
        metrics["gate_lift_vs_all_bps"] = gate_lift
        if gate_lift is None or gate_lift < float(self._settings.MODEL_GATE_CANARY_MIN_LIFT_BPS):
            issues.append(f"insufficient_gate_lift_bps:{gate_lift}")

        paper_by_horizon = self._dict_or_empty(db_diag.get("paper_pnl_by_horizon"))
        raw_paper = paper_by_horizon.get(str(model_horizon))
        if raw_paper is None:
            raw_paper = db_diag.get(f"paper_pnl_{model_horizon}m")
        if raw_paper is None:
            raw_paper = db_diag.get("paper_pnl_15m")
        paper = self._dict_or_empty(raw_paper)
        paper_gate = self._dict_or_empty(paper.get("model_gate"))
        paper_count = int(paper_gate.get("count") or 0)
        paper_total_bps = self._float_or_none(paper_gate.get("total_bps"))
        metrics["paper_gate_count"] = paper_count
        metrics["paper_gate_total_bps"] = paper_total_bps
        if paper_count < 20:
            issues.append(f"insufficient_paper_gate_trades:{paper_count}")
        if paper_total_bps is None or paper_total_bps <= 0:
            issues.append(f"non_positive_paper_gate_bps:{paper_total_bps}")

        return {
            "ready": not issues,
            "mode": self._settings.TRADING_MODE.value,
            "issues": issues,
            "metrics": metrics,
        }

    async def _enforce_economic_readiness_for_active(self) -> None:
        """Fail closed before real-money modes when paper evidence is not ready."""
        assert self._settings is not None
        if self._settings.TRADING_MODE not in (TradingMode.CANARY_LIVE, TradingMode.LIVE):
            return
        if not self._settings.ECONOMIC_READINESS_REQUIRED_FOR_ACTIVE:
            log.warning(
                "economic_readiness_gate_disabled_for_active_mode",
                trading_mode=self._settings.TRADING_MODE.value,
            )
            return
        if self._trade_journal is None or not self._trade_journal.is_enabled:
            log.critical("economic_readiness_blocked", issues=["trade_journal_unavailable"])
            raise SystemExit(1)

        db_diag = await self._trade_journal.get_db_diagnostics()
        issues = self._economic_readiness_report(
            db_diag=db_diag,
            runtime_diag=self.get_diagnostics(),
        )["issues"]
        if issues:
            log.critical(
                "economic_readiness_blocked",
                trading_mode=self._settings.TRADING_MODE.value,
                issues=issues,
            )
            raise SystemExit(1)
        log.info("economic_readiness_passed", trading_mode=self._settings.TRADING_MODE.value)

    @staticmethod
    def _feature_values_for_side(vec: FeatureVector, side: str) -> tuple[list[str], list[float]]:
        from trader.training.feature_side import feature_values_for_side

        return feature_values_for_side(vec, side)

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
        """Resolve Telegram delivery mode and webhook URL for the current environment."""
        assert self._settings is not None
        mode = self._settings.TELEGRAM_DELIVERY_MODE.strip().lower()
        webhook_url = self._settings.TELEGRAM_WEBHOOK_URL.strip().rstrip("/")
        if not webhook_url:
            render_url = os.getenv("RENDER_EXTERNAL_URL", "").strip().rstrip("/")
            if render_url:
                webhook_url = f"{render_url}/telegram/webhook"
        if mode == "auto":
            mode = "webhook" if webhook_url else "polling"
        if mode == "webhook" and not webhook_url:
            log.warning("telegram_webhook_url_missing_fallback_polling")
            mode = "polling"
        if mode not in {"polling", "webhook"}:
            log.warning("telegram_delivery_mode_unknown_fallback_polling", mode=mode)
            mode = "polling"
        return mode, webhook_url

    async def _start_telegram_bot(self) -> None:
        from trader.telegram_bot import (
            TelegramBotConfig,
            TelegramMonitorBot,
            TradingController,
        )

        assert self._settings is not None
        assert self._health_checker is not None
        token = self._settings.TELEGRAM_BOT_TOKEN.get_secret_value()
        if not token:
            log.info("telegram_bot_skipped", reason="no token configured")
            return

        def _regime_for(symbol: str) -> str | None:
            if self._feature_pipeline is None or self._regime_classifier is None:
                return None
            vec = self._feature_pipeline.latest(symbol, _WS_INTERVAL)
            if vec is None:
                return None
            try:
                ctx = self._regime_classifier.classify(vec)
                return cast(str, ctx.regime.value)
            except Exception:
                return None

        async def _db_diagnostics_provider(*, lite: bool = False) -> dict[str, Any]:
            if self._trade_journal is None:
                return {
                    "connected": False,
                    "configured": False,
                    "error": "trade_journal_not_started",
                    "lite": lite,
                }
            if not self._trade_journal.is_enabled:
                await self._trade_journal.reconnect_if_needed()
            now = time.monotonic()
            cache_ttl = 60.0 if lite else 15.0
            cache = self._db_diagnostics_lite_cache if lite else self._db_diagnostics_cache
            cache_at = self._db_diagnostics_lite_cache_at if lite else self._db_diagnostics_cache_at
            if cache is not None and (now - cache_at) < cache_ttl:
                cached_copy = dict(cache)
                self._merge_runtime_db_diag_fallbacks(cached_copy)
                return cached_copy
            try:
                diag = await asyncio.wait_for(
                    self._trade_journal.get_db_diagnostics(lite=lite),
                    timeout=15.0 if lite else 25.0,
                )
            except TimeoutError:
                log.warning("db_diagnostics_timeout", lite=lite)
                diag = {
                    "connected": self._trade_journal.is_enabled,
                    "configured": self._trade_journal.is_configured,
                    "error": "db_diagnostics_timeout",
                    "schema_degraded": bool(
                        getattr(self._trade_journal, "_last_connect_error", None)
                        and "schema bootstrap degraded"
                        in str(getattr(self._trade_journal, "_last_connect_error", "")).lower()
                    ),
                    "lite": lite,
                }
                self._merge_runtime_db_diag_fallbacks(diag)
                return diag
            except Exception as exc:
                return {"connected": False, "error": str(exc), "lite": lite}
            if not lite:
                self._update_model_gate_quality_from_diag(diag)
            diag["paper_notional_usd"] = (
                float(self._settings.MODEL_PAPER_NOTIONAL_USD) if self._settings is not None else 5.0
            )
            self._merge_runtime_db_diag_fallbacks(diag)
            cached = dict(diag)
            if lite:
                self._db_diagnostics_lite_cache = cached
                self._db_diagnostics_lite_cache_at = now
            else:
                self._db_diagnostics_cache = cached
                self._db_diagnostics_cache_at = now
            return diag

        async def _healthcheck_provider() -> dict[str, Any]:
            diag = self.get_diagnostics()
            top_blocker, blockers = self._top_blocker_from_diag(diag, default="нет блокировок")
            today_avg_net_bps = None
            if self._trade_journal is not None and self._trade_journal.is_enabled:
                try:
                    today_avg_net_bps = await self._trade_journal.get_today_avg_net_bps()
                except Exception as _hc_exc:
                    log.debug("healthcheck.avg_net_failed", error=str(_hc_exc))
            return {
                "hour_signals_emitted": diag.get("hour_signals_emitted", 0),
                "hour_order_placed": diag.get("hour_order_placed", 0),
                "hour_ml_replacement": diag.get("hour_ml_replacement", 0),
                "hour_rule_fallback_signals": diag.get("hour_rule_fallback_signals", 0),
                "top_blocker": top_blocker,
                "blockers": blockers,
                "today_avg_net_bps": today_avg_net_bps,
            }

        async def _recent_trades_provider() -> list[dict[str, Any]]:
            if self._trade_journal is None or not self._trade_journal.is_enabled:
                return []
            return cast(list[dict[str, Any]], await self._trade_journal.get_recent_closed_trades(limit=10))

        async def _bucket_stats_provider() -> dict[str, Any]:
            assert self._settings is not None
            return {
                "buckets": [
                    {
                        "regime": regime,
                        "volatility": volatility,
                        "hour": hour,
                        "avg_bps": avg_bps,
                        "count": count,
                    }
                    for (regime, volatility, hour), (
                        avg_bps,
                        count,
                    ) in self._bucket_stats.items()
                ],
                "refreshed_at": (
                    self._bucket_stats_refreshed_at.strftime("%Y-%m-%d %H:%M UTC")
                    if self._bucket_stats_refreshed_at is not None
                    else None
                ),
                "min_samples": self._settings.BUCKET_MIN_SAMPLES,
                "block_below_bps": self._settings.BUCKET_BLOCK_AVG_BPS,
            }

        async def _pnl_analysis_provider() -> dict[str, Any]:
            if self._trade_journal is None or not self._trade_journal.is_enabled:
                return {"connected": False, "error": "trade_journal_unavailable"}
            from trader.training.labels import active_label_schema_version

            horizon = int(self._settings.MODEL_AUTO_TRAIN_HORIZON_MINUTES)
            label_schema = active_label_schema_version(use_tpsl_exit=bool(self._settings.MODEL_LABEL_USE_TPSL_EXIT))
            return cast(
                dict[str, Any],
                await self._trade_journal.get_strategy_pnl_analysis(
                    horizon_minutes=horizon,
                    label_schema_version=label_schema,
                ),
            )

        async def _compare_provider() -> dict[str, Any]:
            if self._trade_journal is None or not self._trade_journal.is_enabled:
                return {"connected": False, "error": "trade_journal_unavailable"}
            from trader.training.labels import active_label_schema_version

            horizon = int(self._settings.MODEL_AUTO_TRAIN_HORIZON_MINUTES)
            label_schema = active_label_schema_version(use_tpsl_exit=bool(self._settings.MODEL_LABEL_USE_TPSL_EXIT))
            return cast(
                dict[str, Any],
                await self._trade_journal.get_model_compare_analysis(
                    horizon_minutes=horizon,
                    label_schema_version=label_schema,
                ),
            )

        async def _worst_trades_provider(limit: int) -> list[dict[str, Any]]:
            if self._trade_journal is None or not self._trade_journal.is_enabled:
                return []
            return cast(list[dict[str, Any]], await self._trade_journal.get_worst_prediction_outcomes(limit=limit))

        async def _costs_detailed_provider() -> dict[str, Any]:
            if self._trade_journal is None or not self._trade_journal.is_enabled:
                return {"connected": False, "error": "trade_journal_unavailable"}
            from trader.training.labels import active_label_schema_version

            horizon = int(self._settings.MODEL_AUTO_TRAIN_HORIZON_MINUTES)
            label_schema = active_label_schema_version(use_tpsl_exit=bool(self._settings.MODEL_LABEL_USE_TPSL_EXIT))
            return cast(
                dict[str, Any],
                await self._trade_journal.get_detailed_costs(
                    horizon_minutes=horizon,
                    label_schema_version=label_schema,
                ),
            )

        async def _model_performance_provider() -> list[dict[str, Any]]:
            if self._trade_journal is None or not self._trade_journal.is_enabled:
                return list(self._model_performance_cache)
            try:
                rows = cast(
                    list[dict[str, Any]],
                    await asyncio.wait_for(
                        self._trade_journal.get_model_performance_history(),
                        timeout=15.0,
                    ),
                )
            except TimeoutError:
                log.warning("model_performance_history.timeout")
                rows = []
            except Exception as exc:
                log.warning("model_performance_history.failed", error=str(exc))
                rows = []
            if rows:
                self._model_performance_cache = rows
                self._model_performance_cache_at = datetime.now(tz=UTC)
                return rows
            return list(self._model_performance_cache)

        async def _champion_health_provider() -> dict[str, Any]:
            if self._trade_journal is None or not self._trade_journal.is_enabled:
                return {"connected": False, "error": "trade_journal_unavailable"}
            return cast(dict[str, Any], await self._trade_journal.get_champion_health())

        async def _add_subscription(chat_id: int) -> None:
            if self._trade_journal is not None:
                await self._trade_journal.add_telegram_subscription(chat_id)

        async def _remove_subscription(chat_id: int) -> None:
            if self._trade_journal is not None:
                await self._trade_journal.remove_telegram_subscription(chat_id)

        async def _load_subscriptions() -> list[int]:
            if self._trade_journal is None or not self._trade_journal.is_enabled:
                return []
            return cast(list[int], await self._trade_journal.get_telegram_subscriptions())

        async def _attribution_provider(days: int = 7) -> list[dict[str, Any]]:
            if self._trade_journal is None or not self._trade_journal.is_enabled:
                return []
            return await self._trade_journal.get_pnl_attribution(days=days)

        async def _best_challenger_provider() -> str | None:
            if self._trade_journal is None or not self._trade_journal.is_enabled:
                return None
            try:
                from trader.ml.auto_promotion import AutoPromotionConfig, AutoPromotionEngine

                engine = AutoPromotionEngine(
                    trade_journal=self._trade_journal,
                    config=AutoPromotionConfig.from_settings(self._settings),
                )
                return await engine.best_challenger()
            except Exception as exc:
                log.debug("best_challenger.lookup_failed", error=str(exc))
                return None

        controller = TradingController(
            pause=self._pause_trading,
            resume=self._resume_trading,
            set_shadow=self._set_shadow_mode,
            set_risk_profile=self._change_risk_profile,
            emergency_stop=self._emergency_stop,
            start_training=self._start_model_training,
            start_training_all=self._start_model_training_all,
            promote_model=self._start_model_promote,
            runtime_settings=self._runtime_settings,
            set_runtime_setting=self._set_runtime_setting,
            symbol_candidates=self._symbol_candidates,
            selected_symbols=self._selected_symbols,
            toggle_symbol=self._toggle_manual_symbol,
            is_paused=lambda: self._trading_paused,
            is_shadow=lambda: self._execution_engine._shadow_mode if self._execution_engine is not None else True,
            current_profile=lambda: self._current_risk_profile_str,
            active_symbols=lambda: self._screener.active_symbols if self._screener is not None else list(_SYMBOLS),
            regime_for=_regime_for,
            signal_log=self._signal_log,
            diagnostics_provider=self.get_diagnostics,
            db_diagnostics_provider=_db_diagnostics_provider,
            allow_risk_increase=self._settings.TELEGRAM_ALLOW_RISK_INCREASE,
            healthcheck_provider=_healthcheck_provider,
            recent_trades_provider=_recent_trades_provider,
            bucket_stats_provider=_bucket_stats_provider,
            pnl_analysis_provider=_pnl_analysis_provider,
            compare_provider=_compare_provider,
            worst_trades_provider=_worst_trades_provider,
            costs_detailed_provider=_costs_detailed_provider,
            model_performance_provider=_model_performance_provider,
            champion_health_provider=_champion_health_provider,
            attribution_provider=_attribution_provider,
            best_challenger_provider=_best_challenger_provider,
            enrich_db_diag_fallbacks=self._merge_runtime_db_diag_fallbacks,
            add_subscription=_add_subscription,
            remove_subscription=_remove_subscription,
            load_subscriptions=_load_subscriptions,
        )

        allowed_chat_ids = set(self._settings.TELEGRAM_ALLOWED_CHAT_IDS)
        delivery_mode, webhook_url = self._resolve_telegram_delivery()
        self._telegram_bot = TelegramMonitorBot(
            config=TelegramBotConfig(
                token=token,
                allowed_chat_ids=allowed_chat_ids,
                trading_mode=self._settings.TRADING_MODE.value,
                risk_profile=self._settings.RISK_PROFILE.value,
                bybit_use_testnet=self._settings.BYBIT_USE_TESTNET,
                default_category=self._settings.DEFAULT_MARKET_CATEGORY,
                redis_url=self._settings.REDIS_URL.get_secret_value(),
                delivery_mode=delivery_mode,
                webhook_url=webhook_url,
                webhook_secret=self._settings.TELEGRAM_WEBHOOK_SECRET.get_secret_value(),
                polling_conflict_recovery_wait_s=self._settings.TELEGRAM_POLLING_CONFLICT_RECOVERY_WAIT_SECONDS,
                polling_watchdog_interval_s=self._settings.TELEGRAM_POLLING_WATCHDOG_INTERVAL_SECONDS,
                polling_zombie_silence_s=self._settings.TELEGRAM_POLLING_ZOMBIE_SILENCE_SECONDS,
            ),
            health_provider=self._health_checker.overall_health,
            adapter_factory=lambda: self._bybit_adapter,
            controller=controller,
            net_results_provider=self._get_net_results,
        )
        started = await self._telegram_bot.start(http_app=self._fastapi_app)
        if started:
            from trader.monitoring.deploy_info import deploy_label

            deploy_id = deploy_label()
            log.info(
                "telegram_bot_started",
                delivery_mode=delivery_mode,
                webhook_url=webhook_url or None,
                deploy_id=deploy_id,
            )
            try:
                await self._telegram_bot.notify(f"🚀 <b>Бот запущен</b>\nDeploy: <code>{html.escape(deploy_id)}</code>")
            except Exception as notify_exc:
                log.debug("telegram.startup_notify_failed", error=str(notify_exc))
        else:
            log.warning("telegram_bot_not_started", health=self._telegram_bot.health_snapshot())

    # ------------------------------------------------------------------
    # Risk & Execution
    # ------------------------------------------------------------------

    async def _init_risk_manager(self, initial_capital: Decimal) -> None:
        """Initialise RiskManager and all its dependencies."""
        from trader.risk.circuit_breakers import CircuitBreakerManager
        from trader.risk.drawdown import DrawdownTracker
        from trader.risk.exposure import ExposureTracker
        from trader.risk.kill_switch import KillSwitch
        from trader.risk.manager import RiskManager
        from trader.risk.profiles import get_risk_limits

        assert self._settings is not None
        profile = self._settings.RISK_PROFILE
        limits = get_risk_limits(profile)

        drawdown = DrawdownTracker(initial_equity=initial_capital)
        self._exposure_tracker = ExposureTracker(
            total_capital=initial_capital,
            risk_limits=limits,
        )
        breakers = CircuitBreakerManager(risk_limits=limits)
        kill_switch = KillSwitch()

        self._risk_manager = RiskManager(
            risk_profile=profile,
            drawdown_tracker=drawdown,
            exposure_tracker=self._exposure_tracker,
            circuit_breaker_manager=breakers,
            kill_switch=kill_switch,
            require_liquidity_for_sizing=(
                self._settings.LIVE_REQUIRE_LIQUIDITY_FOR_SIZING
                and self._settings.TRADING_MODE in (TradingMode.LIVE, TradingMode.CANARY_LIVE)
            ),
            max_correlated_positions=int(self._settings.MAX_CORRELATED_POSITIONS),
        )
        self._kill_switch = kill_switch
        log.info(
            "risk_manager.initialized",
            profile=profile.value,
            initial_capital=str(initial_capital),
        )

    async def _refresh_balance(self) -> Decimal:
        """Fetch current available balance from exchange; fall back to cached value.

        Also updates ExposureTracker capital when balance changes.
        """
        assert self._settings is not None
        has_key = bool(self._settings.BYBIT_API_KEY.get_secret_value())
        if not has_key or self._bybit_adapter is None:
            return self._cached_balance

        try:
            balance = await self._bybit_adapter.get_balance()
            # Use available; if zero (e.g. all collateralised) fall back to wallet
            available = balance.available_balance
            if available <= Decimal("0") and balance.wallet_balance > Decimal("0"):
                available = balance.wallet_balance
            if available > Decimal("0"):
                if self._balance_refreshed_at is not None and balance.updated_at < self._balance_refreshed_at:
                    log.debug(
                        "balance.refresh_ignored_stale",
                        available_usd=str(available),
                        updated_at=balance.updated_at.isoformat(),
                        current_updated_at=self._balance_refreshed_at.isoformat(),
                    )
                    return self._cached_balance
                old_capital = self._cached_balance
                self._cached_balance = available
                self._balance_refreshed_at = balance.updated_at
                log.info(
                    "balance.refreshed",
                    available_usd=str(available),
                    wallet_usd=str(balance.wallet_balance),
                    updated_at=self._balance_refreshed_at.isoformat(),
                )
                # P1: Update ExposureTracker capital so exposure_pct is always current
                if self._exposure_tracker is not None and available != old_capital:
                    self._exposure_tracker.update_capital(available, updated_at=self._balance_refreshed_at)
                    log.debug(
                        "exposure.capital_updated",
                        old_capital=old_capital,
                        new_capital=available,
                        total_exposure_pct=str(self._exposure_tracker.total_exposure_pct),
                    )
            return self._cached_balance
        except Exception as exc:
            log.warning("balance.refresh_failed", error=str(exc))
            return self._cached_balance

    async def _init_execution_engine(self) -> None:
        """Initialise ExecutionEngine after RiskManager is ready."""
        from trader.execution.engine import ExecutionEngine

        assert self._settings is not None
        assert self._risk_manager is not None
        assert self._exposure_tracker is not None
        assert self._bybit_adapter is not None
        from trader.config import get_risk_profile_config

        profile_cfg = get_risk_profile_config(self._settings.RISK_PROFILE)

        shadow = self._initial_shadow_mode()
        is_canary = self._settings.TRADING_MODE == TradingMode.CANARY_LIVE
        self._execution_engine = ExecutionEngine(
            adapter=self._bybit_adapter,
            risk_manager=self._risk_manager,
            exposure_tracker=self._exposure_tracker,
            shadow_mode=shadow,
            shadow_apply_net_edge_gate=self._scalp_strict_shadow(),
            cooldown_s=profile_cfg.cooldown_seconds,
            category=self._settings.DEFAULT_MARKET_CATEGORY,
            trade_journal=self._trade_journal,
            min_notional_safety_buffer_pct=self._settings.MIN_NOTIONAL_SAFETY_BUFFER_PCT,
            micro_account_balance_usd=self._settings.MICRO_ACCOUNT_BALANCE_USD,
            micro_account_min_notional_buffer_pct=self._settings.MICRO_ACCOUNT_MIN_NOTIONAL_BUFFER_PCT,
            max_new_entries_per_minute=self._settings.MAX_NEW_ENTRIES_PER_MINUTE,
            max_concurrent_pending_entries=self._settings.MAX_CONCURRENT_PENDING_ENTRIES,
            max_queue_utilization_pct=float(self._settings.MAX_QUEUE_UTILIZATION_PCT),
            max_same_side_positions=self._settings.MAX_SAME_SIDE_POSITIONS,
            max_open_positions=self._settings.MAX_POSITIONS,
            startup_warmup_seconds=self._settings.STARTUP_WARMUP_SECONDS,
            is_canary=is_canary,
            fee_provider=self._fee_provider,
            max_spread_bps=self._settings.SCREENER_MAX_SPREAD_BPS,
            expected_slippage_pct=self._settings.EXPECTED_SLIPPAGE_PCT,
            funding_buffer_pct=self._settings.FUNDING_BUFFER_PCT,
            min_net_edge_pct=self._settings.MIN_EXPECTED_NET_EDGE_PCT,
            net_edge_safety_margin_pct=self._settings.NET_EDGE_SAFETY_MARGIN_PCT,
            entry_order_mode=self._settings.ENTRY_ORDER_MODE,
            maker_timeout_s=self._settings.MAKER_TIMEOUT_SECONDS,
            maker_ttl_s=self._settings.MAKER_TTL_SECONDS,
            maker_allow_escalation=self._settings.MAKER_ALLOW_ESCALATION,
            # Late-bound: the tracker is created when the public WS starts
            imbalance_provider=lambda s: (
                self._orderbook_tracker.latest_imbalance(s) if self._orderbook_tracker is not None else None
            ),
            live_armed=self._settings.LIVE_ARMED,
            shadow_min_atr_multiple=(
                self._settings.SHADOW_MIN_ATR_MULTIPLE if shadow and not self._scalp_strict_shadow() else None
            ),
        )

        # P0.2: Restore unresolved pending entries from durable storage before any new entries.
        await self._restore_execution_pending_entries()

        # Sync open positions from exchange so we don't double-enter on restart
        await self._execution_engine.sync_positions()

        # Reconcile restored pending entries against live exchange state
        try:
            await self._execution_engine.reconcile_restored_pending_entries()
        except Exception as exc:
            log.warning("execution_engine.reconcile_failed", error=str(exc))

        log.info("execution_engine.initialized", shadow_mode=shadow, is_canary=is_canary)

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
        """Start Bybit private WebSocket for real-time order/position/balance events."""
        from trader.exchange.bybit_ws_private import BybitPrivateWebSocket
        from trader.exchange.endpoint_selector import EndpointSelector

        assert self._settings is not None
        api_key = self._settings.BYBIT_API_KEY.get_secret_value()
        api_secret = self._settings.BYBIT_API_SECRET.get_secret_value()

        if not api_key or not api_secret:
            log.info("private_ws.skipped", reason="no_api_credentials_configured")
            return

        selector = EndpointSelector(
            self._settings.BYBIT_REGION,
            self._settings.BYBIT_USE_TESTNET,
        )

        private_event_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1000)

        self._ws_private = BybitPrivateWebSocket(
            endpoint=selector.ws_private_base,
            api_key=api_key,
            api_secret=api_secret,
            event_queue=private_event_queue,
        )

        async def consume_private_events() -> None:
            from trader.domain.enums import OrderStatus
            from trader.domain.events import (
                BalanceUpdateEvent,
                ExecutionUpdateEvent,
                OrderUpdateEvent,
                PositionUpdateEvent,
            )

            _terminal_order_states = {
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
                OrderStatus.EXPIRED,
            }

            seen_exec_ids: set[str] = set()
            # In-process cache of released order_link_ids: avoids a DB roundtrip per
            # duplicate terminal event. The authoritative record is order_pending_state
            # (resolved_at) which survives restarts.
            _released_cache: set[str] = set()

            async def _release_pending(order_link_id: str, symbol: str) -> None:
                """Release a pending entry slot exactly once and persist the resolution."""
                if order_link_id in _released_cache:
                    return
                if (
                    self._trade_journal is not None
                    and self._trade_journal.is_enabled
                    and await self._trade_journal.is_order_resolved(order_link_id)
                ):
                    _released_cache.add(order_link_id)
                    return
                if self._execution_engine is not None:
                    self._execution_engine.mark_entry_resolved(order_link_id)
                _released_cache.add(order_link_id)
                if self._trade_journal is not None and self._trade_journal.is_enabled:
                    try:
                        await self._trade_journal.mark_order_resolved(order_link_id, symbol)
                    except Exception as _res_exc:
                        log.debug(
                            "private_ws.mark_order_resolved_failed",
                            order_link_id=order_link_id,
                            error=str(_res_exc),
                        )

            while not self._shutdown_event.is_set():
                try:
                    event = await asyncio.wait_for(private_event_queue.get(), timeout=1.0)
                    if isinstance(event, BalanceUpdateEvent) and event.available_balance > Decimal("0"):
                        if self._balance_refreshed_at is not None and event.timestamp < self._balance_refreshed_at:
                            log.debug(
                                "private_ws.balance_update_ignored_stale",
                                available=str(event.available_balance),
                                updated_at=event.timestamp.isoformat(),
                                current_updated_at=self._balance_refreshed_at.isoformat(),
                            )
                            continue
                        old_capital = self._cached_balance
                        self._cached_balance = event.available_balance
                        self._balance_refreshed_at = event.timestamp
                        # P1: Update ExposureTracker capital from WS balance push
                        if self._exposure_tracker is not None and event.available_balance != old_capital:
                            self._exposure_tracker.update_capital(
                                event.available_balance,
                                updated_at=self._balance_refreshed_at,
                            )
                            log.debug(
                                "exposure.capital_updated_ws",
                                old_capital=old_capital,
                                new_capital=event.available_balance,
                                total_exposure_pct=str(self._exposure_tracker.total_exposure_pct),
                            )
                        log.debug(
                            "private_ws.balance_update",
                            available=str(event.available_balance),
                            updated_at=self._balance_refreshed_at.isoformat(),
                        )
                    elif isinstance(event, OrderUpdateEvent):
                        # Wire OrderUpdateEvent → both idempotency AND durable state via adapter
                        # P0: Never use exchange orderId directly as pending-ID.
                        # Use order_link_id if present, otherwise reverse-lookup via exchange_order_id.
                        order_link_id = event.order_link_id
                        exchange_order_id = event.order_id
                        if order_link_id is None and exchange_order_id:
                            if self._trade_journal is not None:
                                order_link_id = await self._trade_journal.find_order_link_id_by_exchange_order_id(
                                    exchange_order_id
                                )
                            # If lookup fails, we still process the event but can't tie it to a pending slot
                        if order_link_id is None:
                            # Generate a fallback ID for logging only — never used for pending slot
                            order_link_id = f"unknown:{exchange_order_id or 'no_exchange_id'}"

                        order_status = event.status  # OrderUpdateEvent.status is the correct field
                        log.info(
                            "private_ws.order_update",
                            order_link_id=order_link_id,
                            exchange_order_id=exchange_order_id,
                            symbol=event.symbol,
                            status=order_status.value if order_status else "unknown",
                            side=event.side.value if event.side else "unknown",
                        )
                        # Update both idempotency and durable state atomically via adapter
                        if self._bybit_adapter is not None:
                            try:
                                is_terminal = await self._bybit_adapter.handle_order_update(event)
                            except Exception as _h_exc:
                                log.debug(
                                    "private_ws.handle_order_update_failed",
                                    error=str(_h_exc),
                                )
                                is_terminal = order_status in _terminal_order_states
                        else:
                            # Fallback: write directly to journal when adapter unavailable
                            if self._trade_journal is not None:
                                try:
                                    await self._trade_journal.record_order_update_event(
                                        order_link_id=order_link_id,
                                        exchange_order_id=exchange_order_id,
                                        symbol=event.symbol,
                                        side=event.side.value if event.side else "unknown",
                                        qty=event.qty if hasattr(event, "qty") and event.qty else Decimal("0"),
                                        state=order_status.value if order_status else "UNKNOWN",
                                    )
                                except Exception as _j_exc:
                                    log.debug(
                                        "private_ws.order_update_journal_failed",
                                        error=str(_j_exc),
                                    )
                            is_terminal = order_status in _terminal_order_states

                        # Release pending entry slot on terminal — exactly once per order.
                        # Resolution is persisted to order_pending_state so a restart
                        # never re-blocks the slot. Skip "unknown:" prefix IDs — they
                        # are fallback logging IDs, not real pending slots.
                        if is_terminal and order_link_id:
                            if not order_link_id.startswith("unknown:"):
                                await _release_pending(order_link_id, event.symbol)
                            else:
                                _released_cache.add(order_link_id)
                        # Trigger position sync on fill
                        if order_status == OrderStatus.FILLED and self._execution_engine is not None:
                            try:
                                await self._execution_engine.sync_positions()
                            except Exception as _sync_exc:
                                log.debug(
                                    "private_ws.order_fill_sync_failed",
                                    error=str(_sync_exc),
                                )
                    elif isinstance(event, ExecutionUpdateEvent):
                        if event.exec_id in seen_exec_ids:
                            continue
                        # Bound the dedup set: duplicates after a reset are harmless
                        # (downstream journal writes are idempotent on exec_id).
                        if len(seen_exec_ids) >= 10_000:
                            seen_exec_ids.clear()
                        seen_exec_ids.add(event.exec_id)
                        # P0: Reverse lookup order_link_id if not present
                        order_link_id = event.order_link_id
                        exchange_order_id = event.order_id
                        if order_link_id is None and exchange_order_id:
                            if self._trade_journal is not None:
                                order_link_id = await self._trade_journal.find_order_link_id_by_exchange_order_id(
                                    exchange_order_id
                                )

                        log.info(
                            "private_ws.execution_fill",
                            exec_id=event.exec_id,
                            symbol=event.symbol,
                            exec_price=str(event.exec_price),
                            exec_qty=str(event.exec_qty),
                            side=event.side.value,
                            order_link_id=order_link_id,
                            exchange_order_id=exchange_order_id,
                        )
                        if self._trade_journal is not None:
                            try:
                                # P0.5: persist to execution_events (nullable proposal/decision)
                                await self._trade_journal.record_execution_event(
                                    exec_id=event.exec_id,
                                    order_link_id=order_link_id
                                    if order_link_id and not order_link_id.startswith("unknown:")
                                    else None,
                                    exchange_order_id=exchange_order_id,
                                    symbol=event.symbol,
                                    side=event.side.value,
                                    exec_price=event.exec_price,
                                    exec_qty=event.exec_qty,
                                    exec_fee=event.exec_fee if event.exec_fee else None,
                                    exec_value=event.exec_value if event.exec_value else None,
                                    is_maker=event.is_maker if hasattr(event, "is_maker") else None,
                                    closed_size=event.closed_size if event.closed_size else None,
                                )
                                await self._trade_journal.record_order_event(
                                    order_link_id=order_link_id
                                    if order_link_id and not order_link_id.startswith("unknown:")
                                    else event.exec_id,
                                    proposal_id=_JOURNAL_FALLBACK_UUID,
                                    decision_id=_JOURNAL_FALLBACK_UUID,
                                    symbol=event.symbol,
                                    side=event.side.value,
                                    qty=event.exec_qty,
                                    status="FILLED",
                                    exchange_order_id=exchange_order_id,
                                )
                            except Exception as _journal_exc:
                                log.warning(
                                    "private_ws.execution_journal_failed",
                                    exec_id=event.exec_id,
                                    error=str(_journal_exc),
                                )

                        # P0.3: Release pending entry slot for this order_link_id only.
                        # Use the resolved order_link_id (after reverse lookup), skip "unknown:" prefixes.
                        # Resolution is persisted to order_pending_state for restart safety.
                        if order_link_id and not order_link_id.startswith("unknown:"):
                            await _release_pending(order_link_id, event.symbol)

                        if self._execution_engine is not None:
                            try:
                                await self._execution_engine.sync_positions()
                            except Exception as _sync_exc:
                                log.warning(
                                    "private_ws.execution_sync_failed",
                                    exec_id=event.exec_id,
                                    error=str(_sync_exc),
                                )

                        if self._bybit_adapter is not None and not self._initial_shadow_mode():
                            try:
                                await self._bybit_adapter.reconcile()
                            except Exception as _rec_exc:
                                log.debug(
                                    "private_ws.execution_reconcile_failed",
                                    exec_id=event.exec_id,
                                    error=str(_rec_exc),
                                )
                    elif isinstance(event, PositionUpdateEvent):
                        if self._execution_engine is not None:
                            try:
                                await self._execution_engine.apply_position_update(event)
                                self._cache_exchange_position_update(event)
                            except Exception as _pos_exc:
                                log.warning(
                                    "private_ws.position_update_failed",
                                    symbol=event.symbol,
                                    error=str(_pos_exc),
                                )
                except TimeoutError:
                    pass
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    log.warning("private_ws_consumer.error", error=str(exc))

        ws_task = asyncio.create_task(self._ws_private.start(), name="ws-private")
        consumer_task = asyncio.create_task(consume_private_events(), name="ws-private-consumer")
        self._background_tasks.extend([ws_task, consumer_task])
        log.info("private_ws.started", endpoint=selector.ws_private_base)

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
        """Periodic risk monitor: update equity, check WS freshness, feed circuit breakers."""
        assert self._settings is not None
        interval = 15.0

        while not self._shutdown_event.is_set():
            try:
                # Refresh balance and update DrawdownTracker with current equity
                if (
                    self._bybit_adapter is not None
                    and self._risk_manager is not None
                    and bool(self._settings.BYBIT_API_KEY.get_secret_value())
                ):
                    try:
                        balance = await self._bybit_adapter.get_balance()
                        wallet = balance.wallet_balance
                        if wallet > Decimal("0"):
                            await self._risk_manager._drawdown.update(wallet)
                    except Exception as exc:
                        log.debug("risk_monitor.balance_update_failed", error=str(exc))

                # P1: Fetch daily realized PnL and feed to RiskManager for daily loss limit tracking
                if self._trade_journal is not None and self._risk_manager is not None:
                    try:
                        net_results = await self._trade_journal.get_daily_net_results()
                        net_pnl = Decimal(str(net_results.get("net_pnl_usd", 0)))
                        # Replace daily_pnl entirely (get_daily_net_results returns today's total)
                        # RiskManager.update_daily_pnl is additive, so we track the delta
                        old_daily_pnl = self._risk_manager.daily_pnl
                        delta = net_pnl - old_daily_pnl
                        if delta != Decimal("0"):
                            await self._risk_manager.update_daily_pnl(delta)
                            log.debug(
                                "risk_monitor.daily_pnl_synced",
                                old=old_daily_pnl,
                                new=net_pnl,
                                delta=delta,
                            )
                    except Exception as exc:
                        log.debug("risk_monitor.daily_pnl_sync_failed", error=str(exc))

                # P1: Evaluate circuit breakers
                if self._risk_manager is not None and self._risk_manager._breakers is not None:
                    breakers = self._risk_manager._breakers
                    # Daily loss limit
                    await breakers.check_daily_loss(
                        self._risk_manager.daily_pnl,
                        self._cached_balance,
                    )
                    # Max drawdown
                    await breakers.check_drawdown(self._risk_manager._drawdown.drawdown_pct)
                    # WebSocket staleness
                    if self._health_checker is not None and self._health_checker._last_ws_message_at is not None:
                        age = (datetime.now(tz=UTC) - self._health_checker._last_ws_message_at).total_seconds()
                        await breakers.check_websocket_staleness(age)
                    # REST error rate (track from adapter if available)
                    if self._bybit_adapter is not None and hasattr(self._bybit_adapter, "_rest_errors_last_minute"):
                        await breakers.check_rest_error_rate(self._bybit_adapter._rest_errors_last_minute)
                    # Feature quality
                    if self._feature_pipeline is not None and hasattr(self._feature_pipeline, "quality_score"):
                        await breakers.check_feature_quality(self._feature_pipeline.quality_score)
                    # NTP drift
                    if self._bybit_adapter is not None and hasattr(self._bybit_adapter, "ntp_drift_seconds"):
                        await breakers.check_ntp_drift(self._bybit_adapter.ntp_drift_seconds)
                    # Auto-reset eligible breakers
                    await breakers.reset_all_auto()

                # Check WS freshness and alert if stale (legacy logging)
                if self._health_checker is not None and self._health_checker._last_ws_message_at is not None:
                    age = (datetime.now(tz=UTC) - self._health_checker._last_ws_message_at).total_seconds()
                    if age > 60.0:
                        log.warning("risk_monitor.ws_stale", age_s=age)
                        self._record_diag("ws_stale")
                    await self._maybe_recover_stale_ws(age)

            except Exception as exc:
                log.warning("risk_monitor.error", error=str(exc))

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=interval,
                )
            except TimeoutError:
                pass

    async def _maybe_recover_stale_ws(self, market_data_age_s: float) -> None:
        """Nudge the public WS to reconnect when market data stops flowing."""
        assert self._settings is not None
        threshold = float(self._settings.WS_MARKET_DATA_STALE_RECONNECT_SECONDS)
        if market_data_age_s < threshold or self._ws_public is None:
            return
        now = datetime.now(tz=UTC)
        if self._last_ws_recovery_at is not None:
            if (now - self._last_ws_recovery_at).total_seconds() < threshold:
                return
        self._last_ws_recovery_at = now
        log.warning(
            "ws_public.recovery_requested",
            market_data_age_s=round(market_data_age_s, 1),
            threshold_s=threshold,
        )
        try:
            await self._ws_public.force_reconnect()
        except Exception as exc:
            log.warning("ws_public.recovery_failed", error=str(exc))

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
        """Import recent Bybit closed PnL and update performance symbol blocks."""
        assert self._settings is not None
        if (
            self._trade_journal is None
            or not self._settings.PERFORMANCE_FILTER_ENABLED
            or self._bybit_adapter is None
            or not self._trade_journal.is_enabled
        ):
            return

        now = datetime.now(tz=UTC)
        if self._closed_pnl_refreshed_at is not None:
            elapsed = (now - self._closed_pnl_refreshed_at).total_seconds()
            if elapsed < self._settings.CLOSED_PNL_REFRESH_INTERVAL_SECONDS:
                return

        try:
            resp = await self._bybit_adapter._rest.get_closed_pnl(
                category=self._settings.DEFAULT_MARKET_CATEGORY,
                limit=100,
            )
            records = resp.get("result", {}).get("list", [])
            await self._trade_journal.record_closed_pnl_records(records)
            blocked = await self._trade_journal.get_blocked_symbols(
                min_closed_trades=self._settings.PERFORMANCE_MIN_CLOSED_TRADES,
                max_loss_usd=Decimal(str(self._settings.PERFORMANCE_MAX_SYMBOL_LOSS_USD)),
                lookback_days=self._settings.PERFORMANCE_LOOKBACK_DAYS,
            )
            if blocked != self._performance_blocked_symbols:
                log.info(
                    "performance_filter.updated",
                    blocked_symbols=sorted(blocked),
                    min_closed_trades=self._settings.PERFORMANCE_MIN_CLOSED_TRADES,
                    max_loss_usd=self._settings.PERFORMANCE_MAX_SYMBOL_LOSS_USD,
                    lookback_days=self._settings.PERFORMANCE_LOOKBACK_DAYS,
                )
            self._performance_blocked_symbols = blocked
            self._closed_pnl_refreshed_at = now
        except Exception as exc:
            log.debug("performance_filter.refresh_failed", error=str(exc))

    async def _manage_open_positions(self) -> None:
        """Move profitable positions to breakeven and enable exchange trailing stop."""
        assert self._settings is not None
        if (
            not self._settings.PROFIT_MANAGER_ENABLED
            or not self._settings.TRAILING_STOP_ENABLED
            or self._bybit_adapter is None
        ):
            return

        now = datetime.now(tz=UTC)
        if self._positions_managed_at is not None:
            elapsed = (now - self._positions_managed_at).total_seconds()
            if elapsed < self._settings.POSITION_MANAGEMENT_INTERVAL_SECONDS:
                return
        self._positions_managed_at = now

        positions = self._recent_exchange_positions()
        if positions is None:
            try:
                positions = await self._bybit_adapter.get_positions(self._settings.DEFAULT_MARKET_CATEGORY)
                self._cache_exchange_positions(positions)
            except Exception as exc:
                log.debug("profit_manager.positions_fetch_failed", error=str(exc))
                return

        # Prune stale trailing-stop keys for positions that are no longer open
        active_keys = {
            f"{p.symbol}:{p.side.value}:{p.size}:{p.entry_price}"
            for p in positions
            if p.size > Decimal("0") and p.entry_price > Decimal("0")
        }
        self._trailing_stop_keys &= active_keys

        for pos in positions:
            if pos.size <= Decimal("0") or pos.entry_price <= Decimal("0"):
                continue
            mark_price = pos.mark_price or pos.entry_price
            if mark_price <= Decimal("0"):
                continue

            pnl_pct = (
                (mark_price - pos.entry_price) / pos.entry_price * Decimal("100")
                if pos.side.value == "Buy"
                else (pos.entry_price - mark_price) / pos.entry_price * Decimal("100")
            )
            if pnl_pct < Decimal(str(self._settings.TRAILING_ACTIVATION_PCT)):
                continue

            position_key = f"{pos.symbol}:{pos.side.value}:{pos.size}:{pos.entry_price}"
            if position_key in self._trailing_stop_keys:
                continue

            try:
                info = (
                    await self._execution_engine.get_instrument_info(pos.symbol)
                    if self._execution_engine is not None
                    else await self._bybit_adapter.get_instrument_info(
                        self._settings.DEFAULT_MARKET_CATEGORY,
                        pos.symbol,
                    )
                )
                active_price = self._round_to_tick(
                    self._activation_price(pos.entry_price, pos.side.value),
                    info.tick_size,
                    round_up=pos.side.value == "Buy",
                )
                trailing_distance = self._round_to_tick(
                    mark_price * Decimal(str(self._settings.TRAILING_DISTANCE_PCT)) / Decimal("100"),
                    info.tick_size,
                    round_up=True,
                )
                fee_rates = None
                if self._fee_provider is not None:
                    try:
                        fee_rates = await self._fee_provider.get(pos.symbol)
                    except Exception as _fee_exc:
                        log.debug(
                            "profit_manager.fee_rate_failed",
                            symbol=pos.symbol,
                            error=str(_fee_exc),
                        )
                breakeven_stop = self._round_to_tick(
                    self._breakeven_stop(pos.entry_price, pos.side.value, fee_rates=fee_rates),
                    info.tick_size,
                    round_up=pos.side.value == "Sell",
                )
                if trailing_distance < info.tick_size:
                    trailing_distance = info.tick_size

                await self._bybit_adapter.set_trading_stop(
                    category=self._settings.DEFAULT_MARKET_CATEGORY,
                    symbol=pos.symbol,
                    stop_loss=str(breakeven_stop),
                    trailing_stop=str(trailing_distance),
                    active_price=str(active_price),
                    position_idx=0,
                    tpsl_mode="Full",
                )
                self._trailing_stop_keys.add(position_key)
                log.info(
                    "profit_manager.trailing_stop_set",
                    symbol=pos.symbol,
                    side=pos.side.value,
                    pnl_pct=float(round(pnl_pct, 4)),
                    stop_loss=str(breakeven_stop),
                    trailing_stop=str(trailing_distance),
                    active_price=str(active_price),
                )
            except Exception as exc:
                log.debug(
                    "profit_manager.trailing_stop_failed",
                    symbol=pos.symbol,
                    error=str(exc),
                )

    async def _sync_transaction_log(self) -> None:
        """Sync Bybit transaction log to database — supports pagination up to 5 pages."""
        await self._modules.ops.sync_transaction_log()

    async def _get_net_results(self) -> dict[str, Any]:
        """Provide daily net PnL for Telegram /net command."""
        if self._trade_journal is None:
            return {}
        return cast(dict[str, Any], await self._trade_journal.get_daily_net_results())

    async def _sync_execution_positions(self) -> None:
        """Keep local execution/risk state aligned with Bybit TP/SL closures."""
        assert self._settings is not None
        if self._execution_engine is None or self._bybit_adapter is None:
            return
        if self._execution_engine._shadow_mode:
            return

        now = datetime.now(tz=UTC)
        if self._positions_synced_at is not None:
            elapsed = (now - self._positions_synced_at).total_seconds()
            if elapsed < self._settings.POSITION_SYNC_INTERVAL_SECONDS:
                return
        positions = await self._execution_engine.sync_positions()
        if positions is not None:
            self._positions_synced_at = now
            self._cache_exchange_positions(positions)

    def _cache_exchange_positions(self, positions: list[Any]) -> None:
        self._latest_exchange_positions = positions
        self._latest_exchange_positions_at = datetime.now(tz=UTC)

    def _cache_exchange_position_update(self, position: Any) -> None:
        positions = list(self._latest_exchange_positions or [])
        positions = [p for p in positions if getattr(p, "symbol", None) != position.symbol]
        if position.size > Decimal("0"):
            positions.append(position)
        self._cache_exchange_positions(positions)

    def _recent_exchange_positions(self) -> list[Any] | None:
        assert self._settings is not None
        if self._latest_exchange_positions_at is None:
            return None
        age = (datetime.now(tz=UTC) - self._latest_exchange_positions_at).total_seconds()
        if age <= max(
            self._settings.POSITION_SYNC_INTERVAL_SECONDS,
            self._settings.POSITION_MANAGEMENT_INTERVAL_SECONDS,
        ):
            return self._latest_exchange_positions
        return None

    def _effective_performance_blocks(self, active_symbols: list[str]) -> set[str]:
        assert self._settings is not None
        blocked = {symbol for symbol in self._performance_blocked_symbols if symbol in active_symbols}
        tradable_count = len(active_symbols) - len(blocked)
        min_tradable = max(0, self._settings.PERFORMANCE_MIN_TRADABLE_SYMBOLS)
        if blocked and tradable_count < min_tradable:
            log.warning(
                "performance_filter.relaxed",
                reason="too_few_tradable_symbols",
                blocked_symbols=sorted(blocked),
                active_symbols=active_symbols,
                min_tradable=min_tradable,
            )
            return set()
        return blocked

    def _activation_price(self, entry_price: Decimal, side: str) -> Decimal:
        assert self._settings is not None
        delta = entry_price * Decimal(str(self._settings.TRAILING_ACTIVATION_PCT)) / Decimal("100")
        return entry_price + delta if side == "Buy" else entry_price - delta

    def _breakeven_stop(self, entry_price: Decimal, side: str, fee_rates: Any | None = None) -> Decimal:
        """Compute a breakeven stop that covers round-trip taker fees + spread + slippage + buffer."""
        assert self._settings is not None
        # Default to config taker rate if no live fee data
        if fee_rates is not None:
            taker = Decimal(str(fee_rates.taker_fee_rate))
        else:
            taker = Decimal(str(self._settings.DEFAULT_LINEAR_TAKER_FEE_RATE))
        entry_fee_pct = taker * Decimal("100")
        exit_fee_pct = taker * Decimal("100")
        spread_pct = Decimal(str(self._settings.SCREENER_MAX_SPREAD_BPS)) / Decimal("100")
        slippage_pct = Decimal(str(self._settings.EXPECTED_SLIPPAGE_PCT)) * Decimal("2")
        buffer_pct = Decimal(str(self._settings.MIN_NET_PROFIT_BUFFER_PCT))
        total_offset_pct = entry_fee_pct + exit_fee_pct + spread_pct + slippage_pct + buffer_pct
        # Also respect the legacy static offset as a minimum floor
        static_pct = Decimal(str(self._settings.BREAKEVEN_STOP_OFFSET_PCT))
        offset_pct = max(total_offset_pct, static_pct)
        offset = entry_price * offset_pct / Decimal("100")
        return entry_price + offset if side == "Buy" else entry_price - offset

    def _round_to_tick(
        self,
        price: Decimal,
        tick_size: Decimal,
        *,
        round_up: bool,
    ) -> Decimal:
        if tick_size <= Decimal("0"):
            return price
        rounding = ROUND_CEILING if round_up else ROUND_DOWN
        ticks = (price / tick_size).to_integral_value(rounding=rounding)
        return ticks * tick_size

    def _record_diag(self, event: str) -> None:
        """Record a diagnostics event with the current timestamp."""
        self._diag_events.append((datetime.now(tz=UTC), event))

    def _top_blocker_from_diag(self, diag: dict[str, Any], *, default: str) -> tuple[str, dict[str, int]]:
        """Return the most useful blocker label for operator diagnostics."""
        blockers = {
            "risk_rejected": int(diag.get("hour_risk_rejected") or 0),
            "risk_rejected:sizer_rejected": int(diag.get("hour_risk_sizer_rejected") or 0),
            "risk_rejected:min_notional": int(diag.get("hour_min_notional_rejected") or 0),
            "risk_rejected:exposure": int(diag.get("hour_risk_exposure_rejected") or 0),
            "risk_rejected:balance": int(diag.get("hour_risk_balance_rejected") or 0),
            "risk_rejected:spread_or_atr": int(diag.get("hour_risk_market_filter_rejected") or 0),
            "startup_warmup": int(diag.get("hour_skipped_startup_warmup") or 0),
            "rate_limit": int(diag.get("hour_skipped_rate_limit") or 0),
            "model_gate_blocked": int(diag.get("hour_model_gate_canary_blocked") or 0),
            "net_edge_rejected": int(diag.get("hour_net_edge_rejected") or 0),
            "post_signal_size_rejected": int(diag.get("hour_signal_qty_adjustment_rejected") or 0),
            "spread_rejected": int(diag.get("hour_spread_rejected") or 0),
            "scalp_net_edge_rejected": int(diag.get("hour_scalp_net_edge_rejected") or 0),
            "imbalance_rejected": int(diag.get("hour_imbalance_rejected") or 0),
            "bucket_blocked": int(diag.get("hour_bucket_blocked") or 0),
            "symbol_side_blocked": int(diag.get("hour_symbol_side_blocked") or 0),
            "trend_confirmation_blocked": int(diag.get("hour_trend_confirmation_blocked") or 0),
            "shadow_loss_guard_blocked": int(diag.get("hour_shadow_loss_guard_blocked") or 0),
        }
        top_blocker = (
            max(blockers, key=lambda k: (blockers[k], 1 if ":" in k else 0)) if any(blockers.values()) else default
        )
        return top_blocker, blockers

    async def _sample_confirmed_candle(self, symbol: str, interval: str, vec: Any) -> None:
        """Record a training sample on every confirmed 1m candle.

        Writes a feature snapshot plus a RULE_BASELINE_V1 prediction event whose
        direction is the rule trend (EMA9 vs EMA21) and decision=SHADOW_CANDLE.
        The outcome resolver labels these like any other event, multiplying
        training-sample accumulation ~100x versus signal-only sampling.
        SHADOW_CANDLE events are excluded from signal statistics.
        """
        assert self._settings is not None
        if (
            not self._settings.CANDLE_SAMPLING_ENABLED
            or interval != _WS_INTERVAL
            or self._trade_journal is None
            or not self._trade_journal.is_enabled
        ):
            return
        try:
            f = dict(zip(vec.feature_names, vec.values, strict=True))
            ema9 = f.get("ema_9")
            ema21 = f.get("ema_21")
            if ema9 is None or ema21 is None:
                return
            # ema_* features are normalised distances to close; their ordering
            # matches the raw EMA ordering, so this is the rule trend direction.
            side = "Buy" if ema9 > ema21 else "Sell"
            model_feature_names, model_feature_values = self._feature_values_for_side(vec, side)

            candles = self._candle_store.confirmed(symbol, interval) if self._candle_store else []
            if not candles:
                return
            candle_open_time = candles[-1].open_time
            # One sample per candle per symbol (Bybit can re-send confirms)
            if self._last_candle_sample_at.get(symbol) == candle_open_time:
                return
            self._last_candle_sample_at[symbol] = candle_open_time

            schema_hash = hashlib.sha256(json.dumps(model_feature_names).encode()).hexdigest()[:16]
            snapshot_id = await self._trade_journal.record_feature_snapshot(
                symbol=symbol,
                interval=interval,
                candle_open_time=candle_open_time,
                feature_schema_hash=schema_hash,
                feature_names=model_feature_names,
                feature_values=model_feature_values,
            )
            if not snapshot_id:
                return
            await self._trade_journal.record_prediction_event(
                symbol=symbol,
                interval=interval,
                model_version="RULE_BASELINE_V1",
                score=0.5,
                strategy_signal=side,
                decision="SHADOW_CANDLE",
                feature_snapshot_id=snapshot_id,
                metadata={"source": "candle_sampler", "strategy_id": "candle_sampler_v1"},
            )

            self._candle_sampler_total += 1

            # Challenger shadow gate on every sampled candle. Signal-only shadow
            # scoring accumulates GATE_PASS/GATE_BLOCK observations slower than
            # the auto-trainer rotates model versions, so per-version gate stats
            # (lift, paper gate) would otherwise stay at zero forever.
            if self._settings.MODEL_SHADOW_SCORING_ENABLED and self._model_registry is not None:
                shadow_prediction = self._model_registry.score_shadow(model_feature_values, model_feature_names)
                if shadow_prediction is not None:
                    self._candle_sampler_scored += 1
                    threshold = self._model_gate_threshold(None)
                    gate_decision = None
                    gate_reason = "shadow_gate_disabled"
                    if self._settings.MODEL_SHADOW_GATE_ENABLED:
                        gate_decision = "GATE_PASS" if shadow_prediction.score >= threshold else "GATE_BLOCK"
                        gate_reason = (
                            "score_meets_threshold" if gate_decision == "GATE_PASS" else "score_below_threshold"
                        )
                        if gate_decision == "GATE_PASS":
                            self._candle_sampler_gate_pass += 1
                        else:
                            self._candle_sampler_gate_block += 1
                    await self._trade_journal.record_prediction_event(
                        symbol=symbol,
                        interval=interval,
                        model_version=shadow_prediction.model_version,
                        score=shadow_prediction.score,
                        strategy_signal=side,
                        decision=gate_decision,
                        feature_snapshot_id=snapshot_id,
                        metadata={
                            "source": "candle_sampler_shadow",
                            "confidence": shadow_prediction.confidence,
                            "gate_reason": gate_reason,
                            "threshold": threshold,
                        },
                    )
                else:
                    self._candle_sampler_no_model += 1
                    # Only warn once per 50 misses to avoid log spam
                    if self._candle_sampler_no_model % 50 == 1:
                        challenger = self._model_registry.challenger if self._model_registry is not None else None
                        log.warning(
                            "candle_sampler.shadow_score_unavailable",
                            symbol=symbol,
                            feature_count=len(vec.feature_names),
                            challenger_version=(challenger.version if challenger is not None else None),
                            challenger_feature_count=(
                                len(challenger.feature_names) if challenger is not None else None
                            ),
                            no_model_count=self._candle_sampler_no_model,
                        )

            # Periodic health summary — emitted every 200 candles (~30 min at 7 symbols)
            if self._candle_sampler_total % 200 == 0:
                score_rate = (
                    round(self._candle_sampler_scored / self._candle_sampler_total, 3)
                    if self._candle_sampler_total
                    else 0.0
                )
                log.info(
                    "candle_sampler.health",
                    total=self._candle_sampler_total,
                    scored=self._candle_sampler_scored,
                    no_model=self._candle_sampler_no_model,
                    gate_pass=self._candle_sampler_gate_pass,
                    gate_block=self._candle_sampler_gate_block,
                    score_rate=score_rate,
                )

        except Exception as exc:
            log.warning("candle_sampler.failed", symbol=symbol, error=str(exc))

    def _bucket_blocked(self, regime_ctx: Any) -> bool:
        """True when the current (regime, volatility, UTC hour) bucket is toxic.

        A bucket blocks only with >= BUCKET_MIN_SAMPLES resolved outcomes and an
        average net return below BUCKET_BLOCK_AVG_BPS — small samples never block.
        In shadow mode the gate is skipped so virtual orders can accumulate training data.
        SCALP strict shadow keeps the gate enabled to avoid paper-trading toxic pairs.
        """
        assert self._settings is not None
        if not self._expectancy_gates_apply():
            return False
        if not self._settings.BUCKET_BLOCK_ENABLED or not self._bucket_stats:
            return False
        regime = (
            regime_ctx.regime.value
            if regime_ctx is not None and getattr(regime_ctx, "regime", None) is not None
            else "UNKNOWN"
        )
        volatility = (
            regime_ctx.volatility_level.value
            if regime_ctx is not None and getattr(regime_ctx, "volatility_level", None) is not None
            else "UNKNOWN"
        )
        hour = datetime.now(tz=UTC).hour
        stats = self._bucket_stats.get((regime, volatility, hour))
        if stats is None:
            return False
        avg_bps, count = stats
        return bool(count >= self._settings.BUCKET_MIN_SAMPLES and avg_bps < self._settings.BUCKET_BLOCK_AVG_BPS)

    def _symbol_side_blocked(self, symbol: str, side: str) -> bool:
        """True when a symbol+side pair has proven negative expectancy.

        In shadow mode the gate is skipped so virtual orders can accumulate training data.
        SCALP strict shadow keeps the gate enabled to avoid paper-trading toxic pairs.
        """

        assert self._settings is not None
        if not self._expectancy_gates_apply():
            return False
        if not self._settings.SYMBOL_SIDE_BLOCK_ENABLED or not self._symbol_side_stats:
            return False
        stats = self._symbol_side_stats.get((symbol, side))
        if stats is None:
            return False
        avg_bps, count = stats
        return bool(
            count >= self._settings.SYMBOL_SIDE_MIN_SAMPLES and avg_bps < self._settings.SYMBOL_SIDE_BLOCK_AVG_BPS
        )

    def _record_shadow_close(self, symbol: str, reason: str, pnl_pct: float) -> None:
        """Track shadow TP/SL results and arm a cooldown after poor recent outcomes."""

        assert self._settings is not None
        if not self._settings.SHADOW_LOSS_GUARD_ENABLED:
            return
        now = datetime.now(tz=UTC)
        self._shadow_closed_results.append((now, reason, float(pnl_pct)))
        window_size = max(1, int(self._settings.SHADOW_LOSS_GUARD_WINDOW))
        recent = list(self._shadow_closed_results)[-window_size:]
        min_closed = max(1, int(self._settings.SHADOW_LOSS_GUARD_MIN_CLOSED))
        if len(recent) < min_closed:
            return
        losses = [value for _, _, value in recent if value < 0]
        loss_rate = len(losses) / len(recent)
        avg_pnl = sum(value for _, _, value in recent) / len(recent)
        if loss_rate >= float(self._settings.SHADOW_LOSS_GUARD_MAX_LOSS_RATE) and avg_pnl <= float(
            self._settings.SHADOW_LOSS_GUARD_MIN_AVG_PNL_PCT
        ):
            cooldown_s = max(0, int(self._settings.SHADOW_LOSS_GUARD_COOLDOWN_SECONDS))
            self._shadow_loss_guard_until = now + timedelta(seconds=cooldown_s)
            log.warning(
                "shadow_loss_guard.activated",
                symbol=symbol,
                reason=reason,
                recent_count=len(recent),
                loss_rate=round(loss_rate, 3),
                avg_pnl_pct=round(avg_pnl, 4),
                cooldown_seconds=cooldown_s,
            )

    @staticmethod
    def _shadow_exit_hit(position: dict[str, Any], *, high: float, low: float) -> tuple[str, float] | None:
        """Return the first conservative TP/SL hit for one shadow candle."""

        side = str(position.get("side") or "")
        tp = float(position["tp"])
        sl = float(position["sl"])
        if side == "Buy":
            tp_hit = high >= tp
            sl_hit = low <= sl
            if tp_hit and sl_hit:
                return "SL", sl
            if tp_hit:
                return "TP", tp
            if sl_hit:
                return "SL", sl
            return None
        if side == "Sell":
            tp_hit = low <= tp
            sl_hit = high >= sl
            if tp_hit and sl_hit:
                return "SL", sl
            if tp_hit:
                return "TP", tp
            if sl_hit:
                return "SL", sl
            return None
        return None

    def _shadow_pnl_pct(self, position: dict[str, Any], exit_price: float) -> float:
        """Return direction-aware net shadow PnL percent after estimated costs."""

        entry = float(position["entry"])
        if entry <= 0:
            raise ValueError("shadow entry must be positive")
        if str(position.get("side") or "") == "Sell":
            gross = (entry - exit_price) / entry * 100.0
        else:
            gross = (exit_price - entry) / entry * 100.0
        if self._settings is None:
            return gross
        taker_fee_pct = float(self._settings.DEFAULT_LINEAR_TAKER_FEE_RATE) * 100.0
        round_trip_fee_pct = taker_fee_pct * 2.0
        spread_pct = float(self._settings.SCREENER_MAX_SPREAD_BPS) / 100.0
        slippage_pct = float(self._settings.EXPECTED_SLIPPAGE_PCT) * 2.0
        return gross - round_trip_fee_pct - spread_pct - slippage_pct

    def _shadow_loss_guard_blocks(self) -> bool:
        """Return true while recent shadow losses should suppress new entries."""

        assert self._settings is not None
        if not self._settings.SHADOW_LOSS_GUARD_ENABLED or self._shadow_loss_guard_until is None:
            return False
        now = datetime.now(tz=UTC)
        if now >= self._shadow_loss_guard_until:
            self._shadow_loss_guard_until = None
            return False
        return True

    def _trend_confirmation_intervals(self) -> list[str]:
        assert self._settings is not None
        raw = str(getattr(self._settings, "TREND_CONFIRMATION_INTERVALS", "") or "")
        return [part.strip() for part in raw.split(",") if part.strip() and part.strip() != _WS_INTERVAL]

    def _trend_mtf_confirmed(self, symbol: str, side: str) -> bool:
        """Confirm a 1m trend signal with higher-timeframe features."""

        assert self._settings is not None
        if not self._settings.TREND_MTF_CONFIRMATION_ENABLED:
            return True
        if self._feature_pipeline is None:
            return False
        intervals = self._trend_confirmation_intervals()
        if not intervals:
            return True
        confirmations = 0
        for interval in intervals:
            vec = self._feature_pipeline.latest(symbol, interval)
            if vec is None:
                continue
            f = dict(zip(vec.feature_names, vec.values, strict=True))
            ema9 = f.get("ema_9")
            ema21 = f.get("ema_21")
            slope9 = f.get("ema_slope_9")
            macd_hist = f.get("macd_hist")
            if any(value is None for value in (ema9, ema21, slope9, macd_hist)):
                continue
            assert ema9 is not None
            assert ema21 is not None
            assert slope9 is not None
            assert macd_hist is not None
            if side == "Buy":
                confirmed = ema9 > ema21 and slope9 > 0 and macd_hist > 0
            else:
                confirmed = ema9 < ema21 and slope9 < 0 and macd_hist < 0
            if confirmed:
                confirmations += 1
        return confirmations > 0

    async def _run_bucket_stats_refresher(self) -> None:
        """Refresh in-memory expectancy gates from Postgres periodically."""
        await self._modules.training.run_bucket_stats_refresher()

    def _check_zero_trading(self) -> None:
        """Warn (never block) when signals flow but nothing executes for an hour.

        Helps catch over-tight filters: model gate, net edge, spread, risk.
        Throttled to one warning per 10 minutes.
        """
        assert self._settings is not None
        now = datetime.now(tz=UTC)
        if self._last_zero_trading_warn_at is not None:
            if (now - self._last_zero_trading_warn_at).total_seconds() < 600:
                return
        diag = self.get_diagnostics()
        signals = int(diag.get("hour_signals_emitted") or 0)
        placed = int(diag.get("hour_order_placed") or 0)
        shadow_would_place = int(diag.get("hour_shadow_order_would_be_placed") or 0)
        if signals >= max(1, self._settings.MIN_SIGNALS_PER_HOUR) and placed == 0 and shadow_would_place == 0:
            if self._execution_engine is not None and self._execution_engine.is_in_warmup():
                log.info(
                    "zero_trading.suppressed_warmup",
                    hour_signals=signals,
                    warmup_seconds_remaining=round(self._execution_engine.warmup_seconds_remaining(), 1),
                )
                return

            self._last_zero_trading_warn_at = now
            top_blocker, blockers = self._top_blocker_from_diag(diag, default="unknown")
            log.warning(
                "zero_trading.detected",
                hour_signals=signals,
                hour_orders_placed=placed,
                top_blocker=top_blocker,
                blockers=blockers,
                auto_soften_enabled=self._settings.AUTO_SOFTEN_FILTERS_ENABLED,
            )

    def _runtime_candle_readiness_counts(self) -> dict[str, int]:
        """In-memory candle counts when Postgres diagnostics are slow or unavailable."""
        if self._candle_store is None:
            return {}
        symbols = self._screener.active_symbols if self._screener is not None else []
        if not symbols:
            return {}
        targets = {"1": 1000, "5": 200, "15": 200, "60": 100}
        counts: dict[str, int] = {}
        for interval, target in targets.items():
            total = sum(self._candle_store.count(symbol, interval, confirmed_only=True) for symbol in symbols)
            counts[interval] = min(target, total)
        return counts

    def _merge_runtime_db_diag_fallbacks(self, diag: dict[str, Any]) -> None:
        """Fill gaps in DB diagnostics from live runtime state (WS candle store, ML registry)."""
        runtime_candles = self._runtime_candle_readiness_counts()
        if runtime_candles:
            diag["runtime_candles_by_interval"] = runtime_candles
            db_candles = dict(diag.get("candles_by_interval") or {})
            merged = dict(db_candles)
            for interval, runtime_count in runtime_candles.items():
                if int(runtime_count) > int(merged.get(interval) or 0):
                    merged[interval] = int(runtime_count)
            if merged:
                diag["candles_by_interval"] = merged
                if db_candles and merged != db_candles:
                    diag["candles_source"] = "db_with_runtime_fallback"
                elif not db_candles:
                    diag["candles_source"] = "runtime_fallback"
        if self._last_confirmed_candle_at is not None and not diag.get("latest_candle_1m"):
            diag["latest_candle_1m"] = self._last_confirmed_candle_at
            diag["last_confirmed_candle_age_s"] = max(
                0.0,
                (datetime.now(tz=UTC) - self._last_confirmed_candle_at).total_seconds(),
            )
        if self._model_registry is not None:
            challenger = self._model_registry.challenger
            champion = self._model_registry.champion
            latest = diag.get("latest_model_version") or {}
            if challenger is not None and not latest.get("version"):
                diag["latest_model_version"] = {
                    "version": challenger.version,
                    "status": "SHADOW_CHALLENGER",
                    "training_samples": challenger.training_samples,
                    "metrics": getattr(challenger, "metrics", {}) or {},
                }
            if champion is not None:
                diag["active_model_version"] = {
                    "version": champion.version,
                    "status": "CHAMPION",
                    "training_samples": champion.training_samples,
                    "metrics": getattr(champion, "metrics", {}) or {},
                }
            elif challenger is not None and not (diag.get("active_model_version") or {}).get("version"):
                diag["active_model_version"] = diag.get("latest_model_version", {})

    def get_diagnostics(self) -> dict[str, Any]:
        """Return a diagnostics snapshot for the /diagnostics Telegram command."""
        now = datetime.now(tz=UTC)
        cutoff = now - _DIAG_WINDOW

        # Count events in the last hour
        hour_counts: dict[str, int] = {}
        for ts, event in self._diag_events:
            if ts >= cutoff:
                hour_counts[event] = hour_counts.get(event, 0) + 1

        ws_age: float | None = None
        if self._health_checker is not None and self._health_checker._last_ws_message_at is not None:
            ws_age = (now - self._health_checker._last_ws_message_at).total_seconds()
        confirmed_age: float | None = None
        if self._last_confirmed_candle_at is not None:
            confirmed_age = (now - self._last_confirmed_candle_at).total_seconds()
        telegram_health: dict[str, Any] = {}
        if self._telegram_bot is not None and hasattr(self._telegram_bot, "health_snapshot"):
            try:
                telegram_health = self._telegram_bot.health_snapshot()
            except Exception as exc:
                telegram_health = {"enabled": True, "error": str(exc)}

        from trader.monitoring.deploy_info import get_deploy_info

        return {
            "deploy": get_deploy_info(),
            "subscribe_watchdog": (self._subscribe_watchdog.to_dict() if self._subscribe_watchdog is not None else {}),
            "last_strategy_loop_at": self._last_strategy_loop_at.isoformat() if self._last_strategy_loop_at else None,
            "last_ws_message_age_s": ws_age,
            "last_confirmed_candle_age_s": confirmed_age,
            "runtime_candles_by_interval": self._runtime_candle_readiness_counts(),
            "telegram": telegram_health,
            "active_symbols": (self._screener.active_symbols if self._screener is not None else list(_SYMBOLS)),
            "open_positions": (
                list(self._execution_engine._open_positions.keys()) if self._execution_engine is not None else []
            ),
            "portfolio_heat_pct": (
                float(self._exposure_tracker.total_exposure_pct) if self._exposure_tracker is not None else None
            ),
            "hour_signals_emitted": hour_counts.get("signals_emitted", 0),
            "hour_risk_rejected": hour_counts.get("risk_rejected", 0),
            "hour_risk_sizer_rejected": hour_counts.get("risk_sizer_rejected", 0),
            "hour_risk_exposure_rejected": hour_counts.get("risk_exposure_rejected", 0),
            "hour_risk_balance_rejected": hour_counts.get("risk_balance_rejected", 0),
            "hour_risk_market_filter_rejected": hour_counts.get("risk_market_filter_rejected", 0),
            "hour_api_rejected": hour_counts.get("api_rejected", 0),
            "hour_min_notional_rejected": hour_counts.get("post_multiplier_min_notional_rejected", 0),
            "hour_skipped_open_position": hour_counts.get("skipped_open_position", 0),
            "hour_skipped_entry_cooldown": hour_counts.get("skipped_entry_cooldown", 0),
            "hour_skipped_failure_cooldown": hour_counts.get("skipped_failure_cooldown", 0),
            "hour_model_gate_canary_blocked": hour_counts.get("model_gate_canary_blocked", 0),
            "hour_ml_replacement": hour_counts.get("ml_replacement", 0),
            "hour_rule_fallback_signals": hour_counts.get("rule_fallback_signal", 0),
            "hour_spread_rejected": hour_counts.get("spread_rejected", 0),
            "hour_scalp_net_edge_rejected": hour_counts.get("scalp_net_edge_rejected", 0),
            "hour_imbalance_rejected": hour_counts.get("imbalance_rejected", 0),
            "hour_bucket_blocked": hour_counts.get("bucket_blocked", 0),
            "hour_symbol_side_blocked": hour_counts.get("symbol_side_blocked", 0),
            "hour_trend_confirmation_blocked": hour_counts.get("trend_confirmation_blocked", 0),
            "drift_status": self._drift_status,
            "strategy_cycle_ms": round(self._last_strategy_cycle_ms, 1),
            "last_retention_run_at": (
                self._last_retention_run_at.isoformat() if self._last_retention_run_at is not None else None
            ),
            "hour_shadow_loss_guard_blocked": hour_counts.get("shadow_loss_guard_blocked", 0),
            # Engine-level counters (cumulative since startup, read from execution engine)
            "hour_skipped_pending_entries": (
                self._execution_engine.get_diag_counts().get("skipped_pending_entries", 0)
                if self._execution_engine is not None
                else 0
            ),
            "hour_skipped_startup_warmup": (
                self._execution_engine.get_diag_counts().get("skipped_startup_warmup", 0)
                if self._execution_engine is not None
                else 0
            ),
            "hour_skipped_rate_limit": (
                self._execution_engine.get_diag_counts().get("skipped_rate_limit", 0)
                if self._execution_engine is not None
                else 0
            ),
            "hour_signal_qty_adjustment_rejected": (
                self._execution_engine.get_diag_counts().get("signal_qty_adjustment_rejected", 0)
                if self._execution_engine is not None
                else 0
            ),
            "hour_order_placed": (
                self._execution_engine.get_diag_counts().get("order_placed", 0)
                if self._execution_engine is not None
                else 0
            ),
            "hour_shadow_order_would_be_placed": (
                self._execution_engine.get_diag_counts().get("shadow_order_would_be_placed", 0)
                if self._execution_engine is not None
                else 0
            ),
            "hour_order_failed": (
                self._execution_engine.get_diag_counts().get("order_failed", 0)
                if self._execution_engine is not None
                else 0
            ),
            "hour_net_edge_rejected": (
                self._execution_engine.get_diag_counts().get("net_edge_rejected", 0)
                if self._execution_engine is not None
                else 0
            ),
            "hour_no_take_profit_rejected": (
                self._execution_engine.get_diag_counts().get("no_tp_rejected", 0)
                if self._execution_engine is not None
                else 0
            ),
            "hour_fee_rate_unavailable_rejected": (
                self._execution_engine.get_diag_counts().get("fee_unavailable_rejected", 0)
                if self._execution_engine is not None
                else 0
            ),
            # Pending entry details for /diagnostics "why no trades" display
            **(
                self._execution_engine.pending_entry_diagnostics()
                if self._execution_engine is not None
                else {
                    "pending_entry_count": 0,
                    "pending_entry_ids": [],
                    "pending_entry_symbols": [],
                    "oldest_pending_age_s": None,
                }
            ),
            "model": {
                "last_training": self._last_training_message,
                "training_samples": (
                    self._model_registry.champion.training_samples
                    if self._model_registry is not None and self._model_registry.champion is not None
                    else (
                        self._model_registry.challenger.training_samples
                        if self._model_registry is not None and self._model_registry.challenger is not None
                        else 0
                    )
                ),
                "champion_version": (
                    self._model_registry.champion.version
                    if self._model_registry is not None and self._model_registry.champion is not None
                    else "none"
                ),
                "challenger_version": (
                    self._model_registry.challenger.version
                    if self._model_registry is not None and self._model_registry.challenger is not None
                    else "none"
                ),
                "quality": (
                    str(
                        (
                            getattr(self._model_registry.champion, "metrics", {})
                            if self._model_registry is not None and self._model_registry.champion is not None
                            else (
                                getattr(self._model_registry.challenger, "metrics", {})
                                if self._model_registry is not None and self._model_registry.challenger is not None
                                else {}
                            )
                        ).get("quality")
                        or self._model_gate_quality.get("quality")
                        or "n/a"
                    )
                ),
                "lift_bps": (
                    (
                        getattr(self._model_registry.champion, "metrics", {})
                        if self._model_registry is not None and self._model_registry.champion is not None
                        else (
                            getattr(self._model_registry.challenger, "metrics", {})
                            if self._model_registry is not None and self._model_registry.challenger is not None
                            else {}
                        )
                    ).get("lift_bps")
                ),
                "walk_forward_expectancy": (
                    (
                        getattr(self._model_registry.champion, "metrics", {})
                        if self._model_registry is not None and self._model_registry.champion is not None
                        else (
                            getattr(self._model_registry.challenger, "metrics", {})
                            if self._model_registry is not None and self._model_registry.challenger is not None
                            else {}
                        )
                    ).get("walk_forward_expectancy_bps", "n/a")
                ),
                "drift_status": self._drift_status.get("status", "n/a"),
                "drift_psi": self._drift_status.get("psi"),
                "gate_quality": self._model_gate_quality.get("quality"),
            },
        }

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
