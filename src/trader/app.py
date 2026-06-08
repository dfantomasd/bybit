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
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal
from typing import Any

import uvicorn

from trader.domain.enums import SystemStatus, TradingMode
from trader.monitoring.logging import configure_logging, get_logger

log = get_logger(__name__)

# Fallback symbols (used only if screener fails); prefer cheap coins for small balance
_SYMBOLS = ["DOGEUSDT", "XRPUSDT", "ADAUSDT", "WLDUSDT", "NEARUSDT"]
_WS_INTERVAL = "1"  # 1-minute klines over WS
_MIN_SEED_BARS = 60  # bars to fetch from REST at startup
_STRATEGY_LOOP_INTERVAL = 10.0  # seconds between strategy evaluations
_FEATURE_INTERVAL = 5.0  # seconds between feature recomputation
_TRAINING_HEARTBEAT_SECONDS = 30.0
_TRAINING_TIMEOUT_SECONDS = 900.0
_TRADE_JOURNAL_RECONNECT_INTERVAL = 30.0
_BALANCE_REFRESH_INTERVAL = 60.0  # seconds between balance refreshes
_FALLBACK_BALANCE_USD = Decimal("1000")  # used when API key not configured
_SUPERVISOR_CHECK_INTERVAL = 5.0  # seconds between supervisor task health checks
_SUPERVISOR_HEARTBEAT_INTERVAL = 60.0  # seconds between heartbeat log lines
_DIAG_WINDOW = timedelta(hours=1)  # sliding window for per-hour diagnostics
_INTERVAL_MS = {
    "1": 60_000,
    "3": 180_000,
    "5": 300_000,
    "15": 900_000,
    "30": 1_800_000,
    "60": 3_600_000,
}
_CRITICAL_TASK_NAMES = frozenset(
    {
        "screener",
        "ws-public",
        "ws-consumer",
        "ws-private",
        "ws-private-consumer",
        "feature-pipeline",
        "strategy-loop",
        "risk-monitor",
        "reconciliation",
        "outcome-resolver",
        "load-governor",
    }
)


class TradingApplication:
    """Top-level application orchestrator."""

    def __init__(self) -> None:
        self._status: SystemStatus = SystemStatus.STARTING
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._settings: Any | None = None
        self._health_checker: Any | None = None
        self._uvicorn_server: uvicorn.Server | None = None
        self._bybit_adapter: Any | None = None
        self._telegram_bot: Any | None = None
        self._ws_public: Any | None = None
        self._candle_store: Any | None = None
        self._feature_pipeline: Any | None = None
        self._strategy_ensemble: Any | None = None
        self._risk_manager: Any | None = None
        self._execution_engine: Any | None = None
        self._exposure_tracker: Any | None = None
        self._screener: Any | None = None
        self._regime_classifier: Any | None = None
        self._background_tasks: list[asyncio.Task] = []
        # Cached balance (refreshed periodically)
        self._cached_balance: Decimal = _FALLBACK_BALANCE_USD
        self._balance_refreshed_at: datetime | None = None
        # Operator control state
        self._trading_paused: bool = False
        self._current_risk_profile_str: str = ""
        self._signal_log: deque = deque(maxlen=20)
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
        # Diagnostics: rolling deque of (timestamp, event_type) for last-hour stats
        self._diag_events: deque[tuple[datetime, str]] = deque(maxlen=10_000)
        self._last_strategy_loop_at: datetime | None = None
        self._training_task: asyncio.Task | None = None
        self._training_start_lock: asyncio.Lock = asyncio.Lock()
        self._last_training_message: str = "never"
        # Private WebSocket (order/position/balance real-time events)
        self._ws_private: Any | None = None
        # ML shadow scoring
        self._model_registry: Any | None = None
        self._model_gate_recent_blocks: deque[bool] = deque(maxlen=100)
        self._model_gate_block_counter: int = 0
        self._model_gate_quality: dict[str, Any] = {}
        self._model_gate_quality_checked_at: datetime | None = None

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
        log.info(
            "settings_loaded",
            trading_mode=self._settings.TRADING_MODE,
            risk_profile=self._settings.RISK_PROFILE,
            bybit_use_testnet=self._settings.BYBIT_USE_TESTNET,
            live_mode=self._settings.LIVE_MODE,
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

        self._health_checker = HealthChecker(
            postgres_dsn=self._settings.POSTGRES_DSN.get_secret_value(),
            redis_url=self._settings.REDIS_URL.get_secret_value(),
            redis_required=self._settings.REDIS_REQUIRED,
            bybit_required=self._settings.BYBIT_CONNECTIVITY_REQUIRED,
            bybit_rest_url=bybit_base,
            trading_mode=self._settings.TRADING_MODE,
            system_status=self._status,
            model_enabled=self._settings.MODEL_ENABLED,
        )

        result = await self._health_checker.run_preflight()
        checks = result["checks"]

        for check_name, passed in checks.items():
            if passed:
                log.info("preflight_check_passed", check=check_name)
            else:
                log.error("preflight_check_failed", check=check_name)

        if not result["passed"]:
            log.critical("preflight_failed", checks=checks)
            raise SystemExit(1)

        log.info("preflight_passed")

    async def _start_trade_journal(self) -> None:
        """Start best-effort Postgres memory for trades and performance."""
        from trader.storage.trade_journal import TradeJournal

        assert self._settings is not None
        self._trade_journal = TradeJournal(
            postgres_dsn=self._settings.POSTGRES_DSN.get_secret_value(),
            enabled=self._settings.TRADE_JOURNAL_ENABLED,
        )
        await self._trade_journal.connect()
        task = asyncio.create_task(self._run_trade_journal_reconnector(), name="trade-journal-reconnector")
        self._background_tasks.append(task)

    async def _run_trade_journal_reconnector(self) -> None:
        """Keep trying Postgres after transient Render startup/network failures."""
        if self._trade_journal is None:
            return
        while not self._shutdown_event.is_set():
            if not self._trade_journal.is_enabled:
                try:
                    connected = await self._trade_journal.reconnect_if_needed(
                        min_interval=_TRADE_JOURNAL_RECONNECT_INTERVAL
                    )
                    if connected:
                        log.info("trade_journal.reconnected")
                except Exception as exc:
                    log.debug("trade_journal.reconnect_failed", error=str(exc))
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=_TRADE_JOURNAL_RECONNECT_INTERVAL)
            except TimeoutError:
                continue

    async def _start_http_server(self) -> asyncio.Task:
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
        )

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
        log.info("shadow_mode.changed", enabled=enabled)

    def _active_execution_allowed(self) -> bool:
        """Return True when orders may be submitted to the configured endpoint."""
        assert self._settings is not None
        if self._settings.TRADING_MODE == TradingMode.SHADOW:
            return False
        if self._settings.BYBIT_USE_TESTNET:
            return True
        return self._settings.LIVE_MODE and self._settings.TRADING_MODE in (TradingMode.LIVE, TradingMode.CANARY_LIVE)

    def _initial_shadow_mode(self) -> bool:
        """Compute startup execution mode from settings and safety gates."""
        assert self._settings is not None
        if self._settings.SHADOW_MODE:
            return True
        return not self._active_execution_allowed()

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
        """Start sequential training on all available data for every horizon (5m, 15m, 30m)."""
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
            "Горизонты: <code>5m, 15m, 30m</code> | Порог: <code>5 bps</code>\n"
            "Используются все доступные примеры (мин. 100).\n"
            "Результаты придут по мере завершения каждого горизонта."
        )

    async def _run_model_training_all(self) -> None:
        """Run training sequentially for all horizons using all available labeled data."""
        horizons = [5, 15, 30]
        label_bps = 5.0
        min_samples = 100
        results: list[str] = []
        for horizon in horizons:
            if self._telegram_bot is not None:
                await self._telegram_bot.notify(f"⏳ <b>Training ALL</b>: запускаю горизонт <code>{horizon}m</code>…")
            await self._run_model_training(min_samples, horizon, label_bps)
            results.append(f"h{horizon}m: готово")
        if self._telegram_bot is not None:
            summary = " | ".join(results)
            await self._telegram_bot.notify(f"✅ <b>Training ALL завершено</b>\n{summary}")

    async def _start_model_promote(self, version: str) -> str:
        """Promote a SHADOW_CHALLENGER model to CHAMPION via subprocess."""
        if self._trade_journal is None or not self._trade_journal.is_enabled:
            raise RuntimeError("Trade journal/Postgres is not available.")

        def code_text(value: str, limit: int = 800) -> str:
            return html.escape(value[-limit:])

        cmd = [sys.executable, "-m", "trader.training.promote", "--version", version, "--confirm"]
        log.info("model_promote.started", version=version)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=60.0)
            stdout = stdout_b.decode(errors="replace").strip()
            stderr = stderr_b.decode(errors="replace").strip()
            if proc.returncode == 0 and "promoted to CHAMPION" in stdout:
                if (
                    self._model_registry is not None
                    and self._trade_journal is not None
                    and self._trade_journal.is_enabled
                ):
                    await self._model_registry.load_active_model()
                if self._telegram_bot is not None:
                    await self._telegram_bot.notify(
                        f"🏆 <b>Модель промоутирована</b>\n<code>{code_text(stdout)}</code>"
                    )
                return f"🏆 <b>Промоут успешен!</b>\n<code>{code_text(stdout)}</code>"
            else:
                out = stderr or stdout or f"exit {proc.returncode}"
                if self._telegram_bot is not None:
                    await self._telegram_bot.notify(f"❌ <b>Промоут не прошёл</b>\n<code>{code_text(out)}</code>")
                return f"❌ <b>Промоут не прошёл:</b>\n<code>{code_text(out)}</code>"
        except TimeoutError:
            return "❌ Промоут завис (timeout 60s)"
        except Exception as exc:
            return f"❌ Ошибка промоута: <code>{html.escape(str(exc))}</code>"

    async def _run_model_training(self, min_samples: int, horizon: int, label_bps: float) -> None:
        cmd = [
            sys.executable,
            "-m",
            "trader.training.train",
            "--min-samples",
            str(min_samples),
            "--horizon",
            str(horizon),
            "--label-bps",
            str(label_bps),
        ]
        log.info("model_training.started", min_samples=min_samples, horizon=horizon, label_bps=label_bps)
        started_at = datetime.now(tz=UTC)

        def code_text(value: str, limit: int = 1500) -> str:
            return html.escape(value[-limit:])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            communicate_task = asyncio.create_task(proc.communicate(), name="model-training-communicate")
            timed_out = False
            while True:
                try:
                    stdout_b, stderr_b = await asyncio.wait_for(
                        asyncio.shield(communicate_task),
                        timeout=_TRAINING_HEARTBEAT_SECONDS,
                    )
                    break
                except TimeoutError:
                    elapsed = (datetime.now(tz=UTC) - started_at).total_seconds()
                    if elapsed >= _TRAINING_TIMEOUT_SECONDS:
                        timed_out = True
                        if proc.returncode is None:
                            proc.kill()
                        try:
                            stdout_b, stderr_b = await asyncio.wait_for(communicate_task, timeout=10.0)
                        except TimeoutError:
                            communicate_task.cancel()
                            stdout_b = b""
                            stderr_b = f"training timeout after {elapsed:.0f}s".encode()
                        break
                    if self._telegram_bot is not None:
                        await self._telegram_bot.notify(
                            "⏳ <b>Training still running</b>\n"
                            f"elapsed=<code>{int(elapsed)}s</code>, "
                            f"min_samples=<code>{min_samples}</code>, horizon=<code>{horizon}m</code>"
                        )
            stdout = stdout_b.decode(errors="replace").strip()
            stderr = stderr_b.decode(errors="replace").strip()
            if timed_out:
                self._last_training_message = stderr or stdout or "training timeout"
                text = "❌ <b>Training timed out</b>\n" + f"<code>{code_text(self._last_training_message)}</code>"
            elif proc.returncode == 0 and "Checkpoint saved" in stdout:
                self._last_training_message = stdout.splitlines()[-2] if len(stdout.splitlines()) >= 2 else stdout
                if (
                    self._model_registry is not None
                    and self._trade_journal is not None
                    and self._trade_journal.is_enabled
                ):
                    await self._model_registry.load_active_model()
                text = "✅ <b>Training completed</b>\n" + f"<code>{code_text(self._last_training_message)}</code>"
            elif proc.returncode == 0:
                self._last_training_message = stdout or stderr or "training finished without checkpoint"
                text = (
                    "⚠️ <b>Training finished without checkpoint</b>\n"
                    + f"<code>{code_text(self._last_training_message)}</code>"
                )
            else:
                self._last_training_message = stderr or stdout or f"exit code {proc.returncode}"
                text = "❌ <b>Training failed</b>\n" + f"<code>{code_text(self._last_training_message)}</code>"
            log.info("model_training.finished", returncode=proc.returncode, message=self._last_training_message)
        except Exception as exc:
            self._last_training_message = str(exc)
            text = f"❌ <b>Training crashed</b>\n<code>{code_text(str(exc))}</code>"
            log.warning("model_training.crashed", error=str(exc))
        if self._trade_journal is not None and self._trade_journal.is_enabled:
            try:
                self._update_model_gate_quality_from_diag(await self._trade_journal.get_db_diagnostics())
            except Exception as diag_exc:
                log.debug("model_gate.quality_refresh_failed", error=str(diag_exc))
        if self._telegram_bot is not None:
            await self._telegram_bot.notify(text)

    async def _run_auto_model_trainer(self) -> None:
        """Automatically train a shadow challenger when enough new labels accumulate."""
        assert self._settings is not None
        if not self._settings.MODEL_AUTO_TRAIN_ENABLED:
            log.info("model_auto_training.disabled")
            return

        check_seconds = max(60, int(self._settings.MODEL_AUTO_TRAIN_CHECK_SECONDS))
        min_samples = max(50, int(self._settings.MODEL_AUTO_TRAIN_MIN_SAMPLES))
        increment_samples = max(1, int(self._settings.MODEL_AUTO_TRAIN_INCREMENT_SAMPLES))
        horizon = int(self._settings.MODEL_AUTO_TRAIN_HORIZON_MINUTES)
        label_bps = float(self._settings.MODEL_AUTO_TRAIN_LABEL_BPS)

        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=check_seconds)
                break
            except TimeoutError:
                pass

            if self._training_task is not None and not self._training_task.done():
                continue

            if self._trade_journal is None:
                log.info("model_auto_training.waiting", reason="trade_journal_not_started")
                continue
            if not self._trade_journal.is_enabled:
                await self._trade_journal.reconnect_if_needed()
                if not self._trade_journal.is_enabled:
                    log.info("model_auto_training.waiting", reason="trade_journal_unavailable")
                    continue

            try:
                diag = await self._trade_journal.get_db_diagnostics()
                self._update_model_gate_quality_from_diag(diag)
                trainable = int(diag.get("labelled_samples_15m", 0) or 0)
                latest_model = diag.get("latest_model_version", {}) or {}
                latest_samples = int(latest_model.get("training_samples", 0) or 0)
                enough_initial = latest_samples == 0 and trainable >= min_samples
                enough_increment = latest_samples > 0 and (trainable - latest_samples) >= increment_samples
                if not (enough_initial or enough_increment):
                    continue

                msg = await self._start_model_training(min_samples, horizon, label_bps)
                log.info(
                    "model_auto_training.started",
                    trainable=trainable,
                    latest_samples=latest_samples,
                    min_samples=min_samples,
                    increment_samples=increment_samples,
                )
                if self._telegram_bot is not None:
                    await self._telegram_bot.notify(
                        "🤖 <b>Auto-training triggered</b>\n"
                        f"trainable_15m=<code>{trainable}</code>, "
                        f"latest_model_samples=<code>{latest_samples}</code>\n"
                        f"{msg}"
                    )
            except Exception as exc:
                log.warning("model_auto_training.failed", error=str(exc))

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
        """Promote challenger to champion automatically when it consistently beats the champion.

        Conservative criteria (all must pass):
          1. Challenger status is SHADOW_CHALLENGER (not already champion).
          2. >= MODEL_AUTO_PROMOTE_MIN_SIGNALS resolved gate decisions.
          3. Live lift_vs_all_bps >= MODEL_AUTO_PROMOTE_MIN_LIFT_BPS.
          4. Live lift_vs_all_bps > current champion's walk-forward expectancy (conservative).
        """
        assert self._settings is not None
        if not self._settings.MODEL_AUTO_PROMOTE_ENABLED:
            log.info("model_auto_promote.disabled")
            return

        check_seconds = max(120, int(self._settings.MODEL_AUTO_PROMOTE_CHECK_SECONDS))
        min_signals = max(10, int(self._settings.MODEL_AUTO_PROMOTE_MIN_SIGNALS))
        min_lift_bps = float(self._settings.MODEL_AUTO_PROMOTE_MIN_LIFT_BPS)

        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=check_seconds)
                break
            except TimeoutError:
                pass

            if self._trade_journal is None or not self._trade_journal.is_enabled:
                continue

            try:
                diag = await self._trade_journal.get_db_diagnostics()
                gate = diag.get("shadow_gate_15m", {}) or {}
                latest_model = diag.get("latest_model_version", {}) or {}

                challenger_version = str(latest_model.get("version", "") or "")
                challenger_status = str(latest_model.get("status", "") or "")
                total_count = int(gate.get("total_count", 0) or 0)
                lift_bps = float(gate.get("lift_vs_all_bps") or 0.0)

                if challenger_status != "SHADOW_CHALLENGER" or not challenger_version:
                    continue

                if total_count < min_signals:
                    log.debug(
                        "model_auto_promote.waiting",
                        reason="insufficient_signals",
                        total_count=total_count,
                        min_signals=min_signals,
                    )
                    continue

                if lift_bps < min_lift_bps:
                    log.debug(
                        "model_auto_promote.waiting",
                        reason="insufficient_lift",
                        lift_bps=lift_bps,
                        min_lift_bps=min_lift_bps,
                    )
                    continue

                # Conservative: challenger must beat champion's own walk-forward expectancy
                champion_wf_bps = await self._get_champion_walk_forward_bps()
                if lift_bps <= champion_wf_bps:
                    log.debug(
                        "model_auto_promote.waiting",
                        reason="not_better_than_champion",
                        lift_bps=lift_bps,
                        champion_wf_bps=champion_wf_bps,
                    )
                    continue

                log.info(
                    "model_auto_promote.triggered",
                    version=challenger_version,
                    total_count=total_count,
                    lift_bps=lift_bps,
                    champion_wf_bps=champion_wf_bps,
                )
                result = await self._start_model_promote(challenger_version)

                # Auto-enable canary gate if model quality now allows it
                canary_msg = ""
                if self._settings is not None and not self._settings.MODEL_GATE_CANARY_ENABLED:
                    self._update_model_gate_quality_from_diag(diag)
                    quality_ok, _ = self._model_gate_quality_allows_canary()
                    if quality_ok:
                        self._settings.MODEL_GATE_CANARY_ENABLED = True
                        log.info("model_auto_promote.canary_enabled", version=challenger_version)
                        canary_msg = (
                            "\n🚦 <b>Canary-фильтр включён</b> — модель начинает фильтровать слабые сигналы.\n"
                            "Чтобы сохранить после перезапуска: задайте "
                            "<code>MODEL_GATE_CANARY_ENABLED=true</code> в Render env vars."
                        )

                if self._telegram_bot is not None:
                    await self._telegram_bot.notify(
                        f"🤖 <b>Авто-промоут</b>\n"
                        f"Версия: <code>{challenger_version}</code>\n"
                        f"Сигналов: <code>{total_count}</code> | "
                        f"Lift: <code>{lift_bps:+.2f} bps</code> vs чемпион <code>{champion_wf_bps:+.2f} bps</code>\n"
                        f"{result}{canary_msg}"
                    )
            except Exception as exc:
                log.warning("model_auto_promote.failed", error=str(exc))

    async def _run_model_progress_reporter(self) -> None:
        """Send an hourly Telegram report on model training progress and promotion readiness."""
        assert self._settings is not None
        if self._telegram_bot is None:
            return

        report_interval = 3600  # 1 hour

        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=report_interval)
                break
            except TimeoutError:
                pass

            if self._trade_journal is None or not self._trade_journal.is_enabled:
                continue

            try:
                diag = await self._trade_journal.get_db_diagnostics()
                self._update_model_gate_quality_from_diag(diag)

                gate = diag.get("shadow_gate_15m", {}) or {}
                latest_model = diag.get("latest_model_version", {}) or {}
                champion_wf_bps = await self._get_champion_walk_forward_bps()

                version = str(latest_model.get("version", "—") or "—")
                status = str(latest_model.get("status", "—") or "—")
                training_samples = int(latest_model.get("training_samples", 0) or 0)
                total_count = int(gate.get("total_count", 0) or 0)
                lift_bps = gate.get("lift_vs_all_bps")
                pass_precision = gate.get("pass_precision")
                labelled = int(diag.get("labelled_samples_15m", 0) or 0)

                min_signals = max(10, int(self._settings.MODEL_AUTO_PROMOTE_MIN_SIGNALS))
                min_lift = float(self._settings.MODEL_AUTO_PROMOTE_MIN_LIFT_BPS)

                # Build promotion checklist
                def check(ok: bool, label: str) -> str:
                    return f"{'✅' if ok else '❌'} {label}"

                has_signals = total_count >= min_signals
                has_lift = lift_bps is not None and float(lift_bps) >= min_lift
                beats_champion = lift_bps is not None and float(lift_bps) > champion_wf_bps
                is_challenger = status == "SHADOW_CHALLENGER"

                lift_str = f"{float(lift_bps):+.2f} bps" if lift_bps is not None else "н/д"
                precision_str = f"{float(pass_precision) * 100:.1f}%" if pass_precision is not None else "н/д"

                lines = [
                    "📊 <b>Прогресс модели</b>",
                    f"Версия: <code>{version}</code> [{status}]",
                    f"Обучено на: <code>{training_samples}</code> примерах | Доступно: <code>{labelled}</code>",
                    "",
                    "<b>Условия для авто-промоута:</b>",
                    check(is_challenger, f"Статус SHADOW_CHALLENGER → {status}"),
                    check(has_signals, f"Сигналов ≥ {min_signals} → сейчас {total_count}"),
                    check(has_lift, f"Lift ≥ {min_lift:+.1f} bps → сейчас {lift_str}"),
                    check(beats_champion, f"Лучше чемпиона ({champion_wf_bps:+.2f} bps) → {lift_str}"),
                    "",
                    f"Точность GATE_PASS: <code>{precision_str}</code>",
                    f"Canary: <code>{'включён' if self._settings.MODEL_GATE_CANARY_ENABLED else 'выключен'}</code>",
                ]

                if all([is_challenger, has_signals, has_lift, beats_champion]):
                    lines.append("\n🟢 <b>Все условия выполнены — промоут скоро!</b>")
                elif not is_challenger and status == "CHAMPION":
                    lines.append("\n🏆 Модель уже чемпион — ждём нового challenger после следующего обучения.")
                else:
                    missing = []
                    if not has_signals:
                        missing.append(f"ещё {min_signals - total_count} сигналов")
                    if not has_lift:
                        missing.append("lift > 0")
                    if not beats_champion and has_lift:
                        missing.append(f"обогнать чемпиона на {champion_wf_bps - float(lift_bps or 0):+.2f} bps")
                    lines.append(f"\n⏳ Не хватает: {', '.join(missing)}")

                await self._telegram_bot.notify("\n".join(lines))

            except Exception as exc:
                log.debug("model_progress_reporter.failed", error=str(exc))

    def _model_gate_threshold(self, regime_context: Any | None) -> float:
        """Return a conservative threshold adjusted by market regime."""
        assert self._settings is not None
        best_threshold = self._model_gate_quality.get("best_threshold")
        threshold = (
            float(best_threshold) if best_threshold is not None else float(self._settings.MODEL_SHADOW_GATE_THRESHOLD)
        )
        if regime_context is None:
            return threshold + 0.02

        regime = getattr(getattr(regime_context, "regime", None), "value", str(getattr(regime_context, "regime", "")))
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

    def _runtime_settings(self) -> dict[str, Any]:
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
            "screener_max_price_usd": self._settings.SCREENER_MAX_PRICE_USD if self._settings is not None else None,
            "feature_max_symbols": self._screener._feature_max if self._screener is not None else None,
            "execution_candidates": self._screener._exec_candidates if self._screener is not None else None,
            "manual_symbols": self._selected_symbols(),
            "model_gate_canary_enabled": (
                self._settings.MODEL_GATE_CANARY_ENABLED if self._settings is not None else False
            ),
            "model_gate_threshold": self._settings.MODEL_SHADOW_GATE_THRESHOLD if self._settings is not None else None,
            "model_gate_quality": self._model_gate_quality,
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
            if self._screener is not None:
                self._screener._feature_max = ivalue
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
            self._settings.MODEL_GATE_CANARY_ENABLED = sval in {"on", "true", "1"}
            return f"Model gate canary set to {'ON' if self._settings.MODEL_GATE_CANARY_ENABLED else 'OFF'}"
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
            return [item.symbol for item in wide[:100]]
        return self._screener.active_symbols

    def _selected_symbols(self) -> list[str]:
        if self._screener is None:
            return []
        return self._screener.manual_symbols

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

    async def _start_telegram_bot(self) -> None:
        from trader.telegram_bot import TelegramBotConfig, TelegramMonitorBot, TradingController

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
                return ctx.regime.value
            except Exception:
                return None

        async def _db_diagnostics_provider() -> dict:
            if self._trade_journal is None:
                return {"connected": False, "configured": False, "error": "trade_journal_not_started"}
            if not self._trade_journal.is_enabled:
                await self._trade_journal.reconnect_if_needed(force=True)
            diag = await self._trade_journal.get_db_diagnostics()
            self._update_model_gate_quality_from_diag(diag)
            diag["paper_notional_usd"] = (
                float(self._settings.MODEL_PAPER_NOTIONAL_USD) if self._settings is not None else 5.0
            )
            return diag

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
            signal_log=self._signal_log,  # type: ignore[arg-type]
            diagnostics_provider=self.get_diagnostics,
            db_diagnostics_provider=_db_diagnostics_provider,
            allow_risk_increase=self._settings.TELEGRAM_ALLOW_RISK_INCREASE,
        )

        allowed_chat_ids = set(self._settings.TELEGRAM_ALLOWED_CHAT_IDS)
        self._telegram_bot = TelegramMonitorBot(
            config=TelegramBotConfig(
                token=token,
                allowed_chat_ids=allowed_chat_ids,
                trading_mode=self._settings.TRADING_MODE.value,
                risk_profile=self._settings.RISK_PROFILE.value,
                bybit_use_testnet=self._settings.BYBIT_USE_TESTNET,
                default_category=self._settings.DEFAULT_MARKET_CATEGORY,
            ),
            health_provider=self._health_checker.overall_health,
            adapter_factory=lambda: self._bybit_adapter,
            controller=controller,
        )
        await self._telegram_bot.start()
        log.info("telegram_bot_started")

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
        )
        self._kill_switch = kill_switch
        log.info(
            "risk_manager.initialized",
            profile=profile.value,
            initial_capital=str(initial_capital),
        )

    async def _refresh_balance(self) -> Decimal:
        """Fetch current available balance from exchange; fall back to cached value."""
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
                self._cached_balance = available
                self._balance_refreshed_at = datetime.now(tz=UTC)
                log.info(
                    "balance.refreshed",
                    available_usd=str(available),
                    wallet_usd=str(balance.wallet_balance),
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
            cooldown_s=profile_cfg.cooldown_seconds,
            category=self._settings.DEFAULT_MARKET_CATEGORY,
            trade_journal=self._trade_journal,
            min_notional_safety_buffer_pct=self._settings.MIN_NOTIONAL_SAFETY_BUFFER_PCT,
            max_new_entries_per_minute=self._settings.MAX_NEW_ENTRIES_PER_MINUTE,
            max_concurrent_pending_entries=self._settings.MAX_CONCURRENT_PENDING_ENTRIES,
            max_same_side_positions=self._settings.MAX_SAME_SIDE_POSITIONS,
            startup_warmup_seconds=self._settings.STARTUP_WARMUP_SECONDS,
            is_canary=is_canary,
            fee_provider=self._fee_provider,
            max_spread_bps=self._settings.SCREENER_MAX_SPREAD_BPS,
            expected_slippage_pct=self._settings.EXPECTED_SLIPPAGE_PCT,
            funding_buffer_pct=self._settings.FUNDING_BUFFER_PCT,
            min_net_edge_pct=self._settings.MIN_EXPECTED_NET_EDGE_PCT,
            entry_order_mode=self._settings.ENTRY_ORDER_MODE,
        )

        # P0.2: Restore pending entry IDs from durable storage before any new entries
        if self._trade_journal is not None:
            try:
                pending_ids = await self._trade_journal.load_pending_from_db()
                if pending_ids:
                    self._execution_engine.restore_pending_entries(pending_ids)
                    log.info("execution_engine.pending_restored", count=len(pending_ids))
            except Exception as exc:
                log.warning("execution_engine.pending_restore_failed", error=str(exc))

        # Sync open positions from exchange so we don't double-enter on restart
        await self._execution_engine.sync_positions()
        log.info("execution_engine.initialized", shadow_mode=shadow, is_canary=is_canary)

    async def _on_screener_symbols_added(self, symbols: list[str]) -> None:
        """Seed candles and subscribe WebSocket for newly added screener symbols."""
        for symbol in symbols:
            # Seed historical candles
            await self._seed_candle_store(symbols=[symbol])
            # Subscribe WebSocket to the new symbol's topics
            if self._ws_public is not None:
                topics = [f"kline.{interval}.{symbol}" for interval in self._market_data_intervals()]
                topics.append(f"tickers.{symbol}")
                await self._ws_public.subscribe(topics)
                log.info("screener.symbol_subscribed", symbol=symbol, topics=topics)

    async def _on_screener_symbols_removed(self, symbols: list[str]) -> None:
        log.info("screener.symbols_removed", symbols=symbols)

    async def _start_screener(self) -> list[str]:
        """Run the market screener and return initial symbol list."""
        from trader.features.screener import MarketScreener

        assert self._bybit_adapter is not None

        assert self._settings is not None
        self._screener = MarketScreener(
            rest_client=self._bybit_adapter._rest,
            wide_max_symbols=self._settings.SCREENER_WIDE_MAX_SYMBOLS,
            feature_max_symbols=self._settings.SCREENER_FEATURE_MAX_SYMBOLS,
            execution_candidates=self._settings.SCREENER_EXECUTION_CANDIDATES,
            min_volume_usd=self._settings.SCREENER_MIN_VOLUME_USD,
            max_spread_bps=self._settings.SCREENER_MAX_SPREAD_BPS,
            min_top_book_depth_usd=self._settings.SCREENER_MIN_TOP_BOOK_DEPTH_USD,
            min_price_usd=self._settings.SCREENER_MIN_PRICE_USD,
            max_price_usd=self._settings.SCREENER_MAX_PRICE_USD,
            interval_s=self._settings.SCREENER_REFRESH_SECONDS,
            denylist=list(self._settings.SCREENER_DENYLIST),
            on_symbols_added=self._on_screener_symbols_added,
            on_symbols_removed=self._on_screener_symbols_removed,
            has_open_position=lambda symbol: (
                self._execution_engine is not None and self._execution_engine.has_open_position(symbol)
            ),
        )

        # Run first screen synchronously so we have symbols before WS starts
        try:
            task = asyncio.create_task(self._screener.run(), name="screener")
            self._background_tasks.append(task)
            await self._screener.wait_ready()
            symbols = self._screener.active_symbols
            log.info("screener.initial_symbols", symbols=symbols)
            return symbols
        except Exception as exc:
            log.warning(
                "screener.startup_failed",
                error=str(exc),
                fallback=_SYMBOLS,
            )
            return list(_SYMBOLS)

    # ------------------------------------------------------------------
    # Market data & features
    # ------------------------------------------------------------------

    async def _seed_candle_store(self, symbols: list[str] | None = None) -> None:
        """Fetch recent historical klines via REST to seed the CandleStore."""
        from trader.data.candles import Candle, CandleStore

        assert self._settings is not None
        assert self._bybit_adapter is not None

        if self._candle_store is None:
            self._candle_store = CandleStore(max_bars=500)

        has_api_key = bool(self._settings.BYBIT_API_KEY.get_secret_value())
        seed_symbols = symbols or _SYMBOLS

        for symbol in seed_symbols:
            for interval in self._market_data_intervals():
                try:
                    resp = await self._bybit_adapter._rest.get_kline(
                        category="linear",
                        symbol=symbol,
                        interval=interval,
                        limit=_MIN_SEED_BARS,
                    )
                    items = resp.get("result", {}).get("list", [])
                    # Bybit returns newest-first; reverse to oldest-first
                    items = list(reversed(items))
                    count = 0
                    for row in items:
                        # row: [startTime, open, high, low, close, volume, turnover]
                        try:
                            ts_ms = int(row[0])
                            open_time = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
                            candle = Candle(
                                open_time=open_time,
                                open=float(row[1]),
                                high=float(row[2]),
                                low=float(row[3]),
                                close=float(row[4]),
                                volume=float(row[5]),
                                confirm=True,  # historical bars are confirmed
                            )
                            self._candle_store.add(symbol, interval, candle)
                            if self._trade_journal is not None and self._trade_journal.is_enabled:
                                bar_ms = _INTERVAL_MS.get(interval, 60_000)
                                close_time = datetime.fromtimestamp((ts_ms + bar_ms - 1) / 1000, tz=UTC)
                                await self._trade_journal.upsert_market_candle(
                                    symbol=symbol,
                                    interval=interval,
                                    open_time=open_time,
                                    close_time=close_time,
                                    open=Decimal(str(row[1])),
                                    high=Decimal(str(row[2])),
                                    low=Decimal(str(row[3])),
                                    close=Decimal(str(row[4])),
                                    volume=Decimal(str(row[5])),
                                    turnover=Decimal(str(row[6])),
                                    confirmed=True,
                                    source="rest_seed",
                                )
                            count += 1
                        except (IndexError, ValueError):
                            continue
                    log.info(
                        "candle_store.seeded",
                        symbol=symbol,
                        interval=interval,
                        bars=count,
                    )
                except Exception as exc:
                    log.warning(
                        "candle_store.seed_failed",
                        symbol=symbol,
                        interval=interval,
                        error=str(exc),
                        has_api_key=has_api_key,
                    )

    async def _start_public_ws(self, symbols: list[str]) -> None:
        """Start the public WebSocket and wire events to CandleStore."""
        from trader.data.candles import CandleStore
        from trader.exchange.bybit_ws_public import BybitPublicWebSocket
        from trader.exchange.endpoint_selector import EndpointSelector

        assert self._settings is not None
        assert self._health_checker is not None

        if self._candle_store is None:
            self._candle_store = CandleStore(max_bars=500)

        selector = EndpointSelector(
            self._settings.BYBIT_REGION,
            self._settings.BYBIT_USE_TESTNET,
        )

        # Build subscription list from screened symbols
        category = self._settings.DEFAULT_MARKET_CATEGORY
        subs: list[str] = []
        for symbol in symbols:
            for interval in self._market_data_intervals():
                subs.append(f"kline.{interval}.{symbol}")
            subs.append(f"tickers.{symbol}")

        event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

        self._ws_public = BybitPublicWebSocket(
            endpoint=f"{selector.ws_public_base}/{category}",
            subscriptions=subs,
            event_queue=event_queue,
        )

        # Event consumer: feeds CandleStore, triggers features, writes candle journal
        async def consume_events() -> None:

            from trader.data.candles import candle_from_kline_event
            from trader.domain.events import KlineEvent

            while not self._shutdown_event.is_set():
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                    if isinstance(event, KlineEvent):
                        candle = candle_from_kline_event(event)
                        self._candle_store.add(event.symbol, event.interval, candle)

                        if event.confirm:
                            # Event-driven feature recompute for this (symbol, interval)
                            if self._feature_pipeline is not None:
                                await self._feature_pipeline.on_confirmed_candle(event.symbol, event.interval)

                            # Persist confirmed candle to PostgreSQL (best-effort)
                            if self._trade_journal is not None and self._trade_journal.is_enabled:
                                bar_ms = _INTERVAL_MS.get(event.interval, 60_000)
                                close_time = datetime.fromtimestamp(
                                    (event.open_time.timestamp() * 1000 + bar_ms - 1) / 1000,
                                    tz=UTC,
                                )
                                try:
                                    await self._trade_journal.upsert_market_candle(
                                        symbol=event.symbol,
                                        interval=event.interval,
                                        open_time=event.open_time,
                                        close_time=close_time,
                                        open=event.open,
                                        high=event.high,
                                        low=event.low,
                                        close=event.close,
                                        volume=event.volume,
                                        turnover=event.turnover,
                                        confirmed=True,
                                        source="ws",
                                    )
                                except Exception as _candle_exc:
                                    log.debug(
                                        "ws_consumer.candle_journal_failed",
                                        symbol=event.symbol,
                                        error=str(_candle_exc),
                                    )

                    # Update WS health on any message
                    if self._health_checker:
                        self._health_checker.set_ws_status(
                            connected=True,
                            last_message_at=datetime.now(tz=UTC),
                        )
                except TimeoutError:
                    # Check if WS is still connected
                    if self._ws_public and not self._ws_public.is_connected:
                        if self._health_checker:
                            self._health_checker.set_ws_status(connected=False)
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    log.warning("ws_consumer.error", error=str(exc))

        ws_task = asyncio.create_task(self._ws_public.start(), name="ws-public")
        consumer_task = asyncio.create_task(consume_events(), name="ws-consumer")
        self._background_tasks.extend([ws_task, consumer_task])
        log.info(
            "public_ws.started",
            endpoint=selector.ws_public_base,
            subscriptions=subs,
        )

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
            from trader.domain.events import BalanceUpdateEvent, ExecutionUpdateEvent, OrderUpdateEvent

            _terminal_order_states = {
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
                OrderStatus.EXPIRED,
            }

            seen_exec_ids: set[str] = set()
            # Guard: track order_link_ids whose pending count has already been released
            # to prevent double-release if multiple terminal events arrive for the same order.
            _pending_released: set[str] = set()

            while not self._shutdown_event.is_set():
                try:
                    event = await asyncio.wait_for(private_event_queue.get(), timeout=1.0)
                    if isinstance(event, BalanceUpdateEvent) and event.available_balance > Decimal("0"):
                        self._cached_balance = event.available_balance
                        self._balance_refreshed_at = datetime.now(tz=UTC)
                        log.debug(
                            "private_ws.balance_update",
                            available=str(event.available_balance),
                        )
                    elif isinstance(event, OrderUpdateEvent):
                        # Wire OrderUpdateEvent → both idempotency AND durable state via adapter
                        order_link_id = event.order_link_id or event.order_id
                        order_status = event.status  # OrderUpdateEvent.status is the correct field
                        log.info(
                            "private_ws.order_update",
                            order_link_id=order_link_id,
                            symbol=event.symbol,
                            status=order_status.value if order_status else "unknown",
                            side=event.side.value if event.side else "unknown",
                        )
                        # Update both idempotency and durable state atomically via adapter
                        if self._bybit_adapter is not None:
                            try:
                                is_terminal = await self._bybit_adapter.handle_order_update(event)
                            except Exception as _h_exc:
                                log.debug("private_ws.handle_order_update_failed", error=str(_h_exc))
                                is_terminal = order_status in _terminal_order_states
                        else:
                            # Fallback: write directly to journal when adapter unavailable
                            if self._trade_journal is not None:
                                try:
                                    await self._trade_journal.record_order_update_event(
                                        order_link_id=order_link_id,
                                        exchange_order_id=event.order_id,
                                        symbol=event.symbol,
                                        side=event.side.value if event.side else "unknown",
                                        qty=event.qty if hasattr(event, "qty") and event.qty else Decimal("0"),
                                        state=order_status.value if order_status else "UNKNOWN",
                                    )
                                except Exception as _j_exc:
                                    log.debug("private_ws.order_update_journal_failed", error=str(_j_exc))
                            is_terminal = order_status in _terminal_order_states
                        # Release pending entry count on terminal — exactly once per order
                        if is_terminal and order_link_id not in _pending_released:
                            if self._execution_engine is not None:
                                self._execution_engine.mark_entry_resolved()
                            _pending_released.add(order_link_id)
                        # Trigger position sync on fill
                        if order_status == OrderStatus.FILLED and self._execution_engine is not None:
                            try:
                                await self._execution_engine.sync_positions()
                            except Exception as _sync_exc:
                                log.debug("private_ws.order_fill_sync_failed", error=str(_sync_exc))
                    elif isinstance(event, ExecutionUpdateEvent):
                        if event.exec_id in seen_exec_ids:
                            continue
                        seen_exec_ids.add(event.exec_id)
                        log.info(
                            "private_ws.execution_fill",
                            exec_id=event.exec_id,
                            symbol=event.symbol,
                            exec_price=str(event.exec_price),
                            exec_qty=str(event.exec_qty),
                            side=event.side.value,
                        )
                        if self._trade_journal is not None:
                            try:
                                # P0.5: persist to execution_events (nullable proposal/decision)
                                await self._trade_journal.record_execution_event(
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
                                await self._trade_journal.record_order_event(
                                    order_link_id=event.order_link_id or event.exec_id,
                                    proposal_id=None,
                                    decision_id=None,
                                    symbol=event.symbol,
                                    side=event.side.value,
                                    qty=event.exec_qty,
                                    status="FILLED",
                                    exchange_order_id=event.order_id,
                                )
                            except Exception as _journal_exc:
                                log.warning(
                                    "private_ws.execution_journal_failed",
                                    exec_id=event.exec_id,
                                    error=str(_journal_exc),
                                )
                        # P0.3: Release pending entry slot for this order_link_id only
                        if self._execution_engine is not None and event.order_link_id:
                            self._execution_engine.mark_entry_resolved(event.order_link_id)
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
        assert self._settings is not None
        if not self._settings.ADAPTIVE_LOAD_GOVERNOR_ENABLED:
            return

        check_interval = float(self._settings.LOAD_GOVERNOR_CHECK_SECONDS)
        max_lag_ms = float(self._settings.MAX_EVENT_LOOP_LAG_MS)
        min_symbols = int(self._settings.LOAD_GOVERNOR_MIN_FEATURE_SYMBOLS)

        # Original feature_max from screener (set at startup)
        original_max: int | None = None

        while not self._shutdown_event.is_set():
            await asyncio.sleep(check_interval)
            if self._screener is None:
                continue

            if original_max is None:
                original_max = self._screener._feature_max

            # --- Measure event-loop lag ---
            t0 = asyncio.get_event_loop().time()
            await asyncio.sleep(0)  # yield and immediately return
            lag_ms = (asyncio.get_event_loop().time() - t0) * 1000

            # --- Measure WS queue utilisation (if accessible) ---
            # The event queue is local to _start_public_ws, so we track pressure
            # by checking if health checker reports recent WS staleness
            ws_stale = False
            if self._health_checker is not None and self._health_checker._last_ws_message_at is not None:
                ws_age = (datetime.now(tz=UTC) - self._health_checker._last_ws_message_at).total_seconds()
                ws_stale = ws_age > 30.0

            overloaded = lag_ms > max_lag_ms or ws_stale
            current = self._screener._feature_max

            if overloaded and current > min_symbols:
                new_max = max(min_symbols, current - 1)
                self._screener._feature_max = new_max
                log.warning(
                    "load_governor.reducing_symbols",
                    lag_ms=round(lag_ms, 1),
                    ws_stale=ws_stale,
                    from_max=current,
                    to_max=new_max,
                    min_symbols=min_symbols,
                )
            elif not overloaded and current < original_max:
                # Restore one symbol at a time
                new_max = min(original_max, current + 1)
                self._screener._feature_max = new_max
                log.info(
                    "load_governor.restoring_symbols",
                    lag_ms=round(lag_ms, 1),
                    from_max=current,
                    to_max=new_max,
                )

    async def _run_outcome_resolver(self) -> None:
        """Resolve prediction outcomes by comparing feature snapshot prices with market_candles."""
        interval = 300.0  # every 5 minutes
        horizons = [5, 15, 30]

        while not self._shutdown_event.is_set():
            if self._trade_journal is not None and self._trade_journal.is_enabled:
                for horizon in horizons:
                    try:
                        resolved = await self._trade_journal.resolve_outcomes_from_candles(
                            horizon_minutes=horizon,
                            label_bps_threshold=5.0,
                        )
                        if resolved > 0:
                            log.info(
                                "outcome_resolver.resolved",
                                horizon_minutes=horizon,
                                count=resolved,
                            )
                    except Exception as exc:
                        log.debug("outcome_resolver.error", horizon=horizon, error=str(exc))

            try:
                await asyncio.wait_for(
                    asyncio.shield(self._shutdown_event.wait()),
                    timeout=interval,
                )
            except TimeoutError:
                pass

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

                # Check WS freshness and alert if stale
                if self._health_checker is not None and self._health_checker._last_ws_message_at is not None:
                    age = (datetime.now(tz=UTC) - self._health_checker._last_ws_message_at).total_seconds()
                    if age > 60.0:
                        log.warning("risk_monitor.ws_stale", age_s=age)
                        self._record_diag("ws_stale")

            except Exception as exc:
                log.warning("risk_monitor.error", error=str(exc))

            try:
                await asyncio.wait_for(
                    asyncio.shield(self._shutdown_event.wait()),
                    timeout=interval,
                )
            except TimeoutError:
                pass

    async def _run_reconciliation(self) -> None:
        """Periodic reconciliation: compare local order state with exchange."""
        assert self._settings is not None
        interval = float(self._settings.RECONCILIATION_INTERVAL_SECONDS)

        while not self._shutdown_event.is_set():
            try:
                if self._bybit_adapter is not None and not self._initial_shadow_mode():
                    result = await self._bybit_adapter.reconcile()
                    if result.discrepancies_found > 0:
                        log.warning(
                            "reconciliation.discrepancies_found",
                            discrepancies=result.discrepancies_found,
                            mismatched=result.mismatched_order_ids[:10],
                            summary=result.summary,
                        )
                    else:
                        log.debug("reconciliation.clean", summary=result.summary)
            except Exception as exc:
                log.warning("reconciliation.error", error=str(exc))

            try:
                await asyncio.wait_for(
                    asyncio.shield(self._shutdown_event.wait()),
                    timeout=interval,
                )
            except TimeoutError:
                pass

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
        )
        self._regime_classifier = RegimeClassifier()

        task = asyncio.create_task(
            self._feature_pipeline.run(
                symbols=_SYMBOLS,  # fallback; screener overrides via symbol_source
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
                        log.debug("profit_manager.fee_rate_failed", symbol=pos.symbol, error=str(_fee_exc))
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
        """Sync Bybit transaction log to database periodically."""
        assert self._settings is not None
        if self._trade_journal is None or self._bybit_adapter is None:
            return
        try:
            resp = await self._bybit_adapter._rest.get_transaction_log(
                account_type="UNIFIED",
                category=self._settings.DEFAULT_MARKET_CATEGORY,
                currency="USDT",
                limit=50,
            )
            entries = (resp.get("result") or {}).get("list", [])
            if entries:
                inserted = await self._trade_journal.record_transaction_log_entries(entries)
                if inserted:
                    log.info("transaction_log.synced", inserted=inserted)
        except Exception as exc:
            log.debug("transaction_log.sync_failed", error=str(exc))

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
        slippage_pct = Decimal(str(self._settings.EXPECTED_SLIPPAGE_PCT))
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

        return {
            "last_strategy_loop_at": self._last_strategy_loop_at.isoformat() if self._last_strategy_loop_at else None,
            "last_ws_message_age_s": ws_age,
            "active_symbols": (self._screener.active_symbols if self._screener is not None else list(_SYMBOLS)),
            "open_positions": (
                list(self._execution_engine._open_positions.keys()) if self._execution_engine is not None else []
            ),
            "portfolio_heat_pct": (
                float(self._exposure_tracker.total_exposure_pct) if self._exposure_tracker is not None else None
            ),
            "hour_signals_emitted": hour_counts.get("signals_emitted", 0),
            "hour_risk_rejected": hour_counts.get("risk_rejected", 0),
            "hour_api_rejected": hour_counts.get("api_rejected", 0),
            "hour_min_notional_rejected": hour_counts.get("post_multiplier_min_notional_rejected", 0),
            "hour_skipped_open_position": hour_counts.get("skipped_open_position", 0),
            "hour_skipped_entry_cooldown": hour_counts.get("skipped_entry_cooldown", 0),
            "hour_skipped_failure_cooldown": hour_counts.get("skipped_failure_cooldown", 0),
            "hour_model_gate_canary_blocked": hour_counts.get("model_gate_canary_blocked", 0),
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
                "walk_forward_expectancy": "n/a",
                "drift_status": "n/a",
            },
        }

    async def _run_supervisor(self) -> None:
        """Monitor critical background tasks; on unexpected exit alert + exit(1)."""
        last_heartbeat = datetime.now(tz=UTC)
        while not self._shutdown_event.is_set():
            now = datetime.now(tz=UTC)

            if (now - last_heartbeat).total_seconds() >= _SUPERVISOR_HEARTBEAT_INTERVAL:
                alive = [t.get_name() for t in self._background_tasks if not t.done()]
                log.info("runtime_supervisor.heartbeat", alive_tasks=alive)
                last_heartbeat = now

            for task in list(self._background_tasks):
                if not task.done():
                    continue
                name = task.get_name()
                if name not in _CRITICAL_TASK_NAMES:
                    continue
                if self._shutdown_event.is_set():
                    return

                exc = task.exception() if not task.cancelled() else None
                log.critical(
                    "runtime_supervisor.critical_task_died",
                    task=name,
                    error=str(exc),
                )
                if self._telegram_bot is not None:
                    try:
                        await self._telegram_bot.notify(
                            f"🚨 <b>Critical task died</b>: <code>{name}</code>\n"
                            f"Error: <code>{exc}</code>\n"
                            "Container will restart automatically."
                        )
                    except Exception as notify_exc:  # noqa: BLE001
                        log.warning("supervisor.telegram_notify_failed", error=str(notify_exc))
                sys.exit(1)

            try:
                await asyncio.wait_for(
                    asyncio.shield(self._shutdown_event.wait()),
                    timeout=_SUPERVISOR_CHECK_INTERVAL,
                )
            except TimeoutError:
                pass

    async def _start_strategy_loop(self) -> None:
        """Run strategy ensemble → RiskManager → ExecutionEngine."""
        from trader.strategies.ensemble import StrategyEnsemble
        from trader.strategies.trend import EMAcrossoverStrategy

        assert self._settings is not None

        # Fetch initial balance to seed RiskManager
        from trader.config import get_risk_profile_config

        profile_cfg = get_risk_profile_config(self._settings.RISK_PROFILE)
        initial_capital = await self._refresh_balance()
        if initial_capital <= Decimal("0"):
            initial_capital = _FALLBACK_BALANCE_USD
            log.warning(
                "strategy_loop.using_fallback_capital",
                capital=str(initial_capital),
            )

        # Build risk + execution stack
        await self._init_risk_manager(initial_capital)
        await self._init_execution_engine()

        # One symbol-agnostic strategy instance handles ALL screener symbols
        strategies = [
            EMAcrossoverStrategy(
                symbol=None,  # None = evaluate any symbol passed in
                allow_short=True,
                min_qty_usd=5.0,  # Bybit minimum notional is $5
                max_risk_pct=0.01,  # 1% of balance per trade
            )
        ]

        self._strategy_ensemble = StrategyEnsemble(
            strategies=strategies,
            health_checker=self._health_checker,
            min_confidence=profile_cfg.min_confidence,
        )
        await self._refresh_closed_pnl_memory()

        # Initialise ML shadow scoring registry (shadow only; never influences decisions)
        if self._settings.MODEL_SHADOW_SCORING_ENABLED:
            try:
                from trader.ml.challenger import ModelRegistry

                self._model_registry = ModelRegistry(trade_journal=self._trade_journal)
                if self._trade_journal is not None and self._trade_journal.is_enabled:
                    await self._model_registry.load_active_model()
                log.info("model_registry.initialized")
            except Exception as _mr_exc:
                log.warning("model_registry.init_failed", error=str(_mr_exc))

        _balance_tick: int = 0
        _effective_blocked_symbols: set[str] = set()
        # Shadow TP/SL tracker: symbol → {entry, tp, sl, side, opened_at}
        _shadow_positions: dict[str, dict[str, Any]] = {}

        async def _check_shadow_exits(symbol: str, current_price: float) -> None:
            """Close shadow positions that hit TP or SL."""
            pos = _shadow_positions.get(symbol)
            if pos is None:
                return
            side = pos["side"]
            tp = pos["tp"]
            sl = pos["sl"]
            hit = None
            if side == "Buy" and current_price >= tp:
                hit = "TP"
            elif side == "Buy" and current_price <= sl:
                hit = "SL"
            elif side == "Sell" and current_price <= tp:
                hit = "TP"
            elif side == "Sell" and current_price >= sl:
                hit = "SL"
            if hit:
                pnl_pct = (
                    (current_price - pos["entry"]) / pos["entry"] * 100
                    if side == "Buy"
                    else (pos["entry"] - current_price) / pos["entry"] * 100
                )
                log.info(
                    "shadow.position_closed",
                    symbol=symbol,
                    reason=hit,
                    entry=pos["entry"],
                    exit=current_price,
                    pnl_pct=round(pnl_pct, 3),
                )
                del _shadow_positions[symbol]
                if self._execution_engine is not None:
                    await self._execution_engine.record_position_closed(symbol)
                if self._telegram_bot is not None:
                    try:
                        label = "✅ TP" if hit == "TP" else "🛑 SL"
                        pnl_sign = "+" if pnl_pct >= 0 else ""
                        await self._telegram_bot.notify(
                            f"{label} {symbol} {side} closed\n"
                            f"Entry: {pos['entry']:.4f} → Exit: {current_price:.4f}\n"
                            f"PnL: {pnl_sign}{pnl_pct:.2f}% [SHADOW]"
                        )
                    except Exception as exc:
                        log.debug("telegram.shadow_exit_notify_failed", error=str(exc))

        async def process_symbol(symbol: str, balance: Decimal, capital: Decimal) -> None:
            """Evaluate one symbol: features → regime → ensemble → execution."""
            if symbol in _effective_blocked_symbols:
                log.debug("performance_filter.symbol_blocked", symbol=symbol)
                return

            if self._feature_pipeline is None:
                return

            vec = self._feature_pipeline.latest(symbol, _WS_INTERVAL)
            if vec is None:
                return

            closes = self._candle_store.closes(symbol, _WS_INTERVAL, 1) if self._candle_store else []
            if not closes:
                return
            current_price = closes[-1]

            # Check shadow TP/SL exits first
            await _check_shadow_exits(symbol, current_price)

            # Classify regime
            regime_ctx = None
            if self._regime_classifier is not None:
                try:
                    regime_ctx = self._regime_classifier.classify(vec)
                except Exception as exc:
                    log.warning("strategy_loop.regime_error", symbol=symbol, error=str(exc))

            # Strategy ensemble
            try:
                proposal = self._strategy_ensemble.evaluate_all(
                    feature_vector=vec,
                    current_price=current_price,
                    available_balance_usd=float(balance),
                )
            except Exception as exc:
                log.warning("strategy_loop.ensemble_error", symbol=symbol, error=str(exc))
                return

            if proposal is None:
                return

            self._record_diag("signals_emitted")

            if self._trade_journal is not None:
                await self._trade_journal.record_signal(
                    proposal=proposal,
                    feature_vector=vec,
                    regime_context=regime_ctx,
                )

            # Record feature snapshot for ML training (no lookahead — uses candle open_time)
            snapshot_id = ""
            if self._trade_journal is not None and self._trade_journal.is_enabled and vec.feature_names:
                try:
                    import hashlib
                    import json as _json

                    _schema_hash = hashlib.sha256(_json.dumps(sorted(vec.feature_names)).encode()).hexdigest()[:16]
                    _candles = self._candle_store.confirmed(proposal.symbol, _WS_INTERVAL) if self._candle_store else []
                    _candle_open_time = _candles[-1].open_time if _candles else vec.timestamp
                    snapshot_id = await self._trade_journal.record_feature_snapshot(
                        symbol=proposal.symbol,
                        interval=_WS_INTERVAL,
                        candle_open_time=_candle_open_time,
                        feature_schema_hash=_schema_hash,
                        feature_names=vec.feature_names,
                        feature_values=vec.values,
                    )
                except Exception as _snap_exc:
                    log.debug("strategy_loop.feature_snapshot_failed", error=str(_snap_exc))

            # ML shadow scoring — only records metadata, never influences trade decisions
            if self._trade_journal is not None and self._trade_journal.is_enabled and snapshot_id:
                try:
                    await self._trade_journal.record_prediction_event(
                        symbol=proposal.symbol,
                        interval=_WS_INTERVAL,
                        model_version="RULE_BASELINE_V1",
                        score=proposal.confidence,
                        strategy_signal=proposal.side.value,
                        decision="SHADOW_BASELINE",
                        feature_snapshot_id=snapshot_id,
                    )
                except Exception as _baseline_exc:
                    log.debug(
                        "strategy_loop.baseline_prediction_failed",
                        symbol=proposal.symbol,
                        error=str(_baseline_exc),
                    )

            if self._settings.MODEL_SHADOW_SCORING_ENABLED and self._model_registry is not None and snapshot_id:
                try:
                    prediction = self._model_registry.score(vec.values)
                    if prediction is not None:
                        threshold = self._model_gate_threshold(regime_ctx)
                        gate_decision = None
                        gate_reason = "gate_disabled"
                        canary_blocked = False
                        canary_reason = "canary_disabled"
                        regime_name = (
                            regime_ctx.regime.value
                            if regime_ctx is not None and getattr(regime_ctx, "regime", None) is not None
                            else "UNKNOWN"
                        )
                        volatility_name = (
                            regime_ctx.volatility_level.value
                            if regime_ctx is not None and getattr(regime_ctx, "volatility_level", None) is not None
                            else "UNKNOWN"
                        )
                        if self._settings.MODEL_SHADOW_GATE_ENABLED:
                            gate_decision = "GATE_PASS" if prediction.score >= threshold else "GATE_BLOCK"
                            gate_reason = (
                                "score_meets_threshold"
                                if gate_decision == "GATE_PASS"
                                else "score_below_regime_threshold"
                            )
                            canary_blocked, canary_reason = self._model_gate_canary_blocks(
                                gate_decision,
                                threshold,
                                prediction.score,
                            )
                        if self._trade_journal is not None and self._trade_journal.is_enabled:
                            await self._trade_journal.record_prediction_event(
                                symbol=proposal.symbol,
                                interval=_WS_INTERVAL,
                                model_version=prediction.model_version,
                                score=prediction.score,
                                strategy_signal=proposal.side.value,
                                decision=gate_decision,
                                feature_snapshot_id=snapshot_id,
                                metadata={
                                    "canary_blocked": canary_blocked,
                                    "canary_reason": canary_reason,
                                    "confidence": prediction.confidence,
                                    "gate_reason": gate_reason,
                                    "regime": regime_name,
                                    "score": prediction.score,
                                    "threshold": threshold,
                                    "volatility": volatility_name,
                                },
                            )
                        if canary_blocked:
                            self._record_diag("model_gate_canary_blocked")
                            log.info(
                                "model_gate.canary_blocked",
                                symbol=proposal.symbol,
                                model_version=prediction.model_version,
                                score=prediction.score,
                                threshold=threshold,
                                reason=canary_reason,
                            )
                            return
                    else:
                        log.debug("ml_shadow.no_challenger", symbol=proposal.symbol)
                except Exception as _ml_exc:
                    log.debug("ml_shadow.scoring_failed", symbol=proposal.symbol, error=str(_ml_exc))

            # Skip execution if operator paused trading
            if self._trading_paused:
                log.debug("strategy_loop.paused", symbol=symbol)
                return

            # DB availability guard for CANARY_LIVE / LIVE
            if self._settings.TRADING_MODE in (TradingMode.CANARY_LIVE, TradingMode.LIVE):
                if self._settings.TRADE_JOURNAL_REQUIRED_FOR_ACTIVE:
                    if self._trade_journal is None or not self._trade_journal.is_enabled:
                        log.warning(
                            "strategy_loop.blocked_no_journal",
                            symbol=symbol,
                            mode=self._settings.TRADING_MODE,
                        )
                        return
                if self._settings.DURABLE_ORDER_STATE_REQUIRED_FOR_ACTIVE:
                    if self._trade_journal is None or not self._trade_journal.durable_state_healthy:
                        log.warning(
                            "strategy_loop.blocked_durable_store_unhealthy",
                            symbol=symbol,
                            mode=self._settings.TRADING_MODE,
                            write_health=(
                                self._trade_journal.write_health() if self._trade_journal is not None else {}
                            ),
                        )
                        return

            # ExecutionEngine: dedup/cooldown/risk → order (or shadow log)
            # Notification fires only when execution engine actually approves
            if self._execution_engine is None:
                return

            try:
                decision = await self._execution_engine.submit(
                    proposal=proposal,
                    capital=capital,
                    available_balance=balance,
                    feature_vector=vec,
                    regime_context=regime_ctx,
                )
            except Exception as exc:
                log.warning("strategy_loop.execution_error", symbol=symbol, error=str(exc))
                return

            from trader.domain.enums import RiskDecisionStatus

            if decision is None:
                return
            if decision.status == RiskDecisionStatus.REJECTED:
                self._record_diag("risk_rejected")
                # Track specific rejection reasons
                for rule in decision.triggered_rules or []:
                    if rule == "post_multiplier_min_notional_rejected":
                        self._record_diag("post_multiplier_min_notional_rejected")
            if decision.status not in (
                RiskDecisionStatus.APPROVED,
                RiskDecisionStatus.RESIZED,
            ):
                return

            # Trade approved — notify Telegram once and log to signal deque
            is_shadow = self._execution_engine._shadow_mode
            regime_str = regime_ctx.regime.value if regime_ctx is not None else "UNKNOWN"
            from trader.telegram_bot import SignalEntry

            entry = SignalEntry(
                timestamp=datetime.now(tz=UTC),
                symbol=proposal.symbol,
                side=proposal.side.value,
                confidence=proposal.confidence,
                regime=regime_str,
                rationale=proposal.rationale or "",
                shadow=is_shadow,
            )
            self._signal_log.append(entry)
            if self._telegram_bot is not None:
                try:
                    await self._telegram_bot.notify_signal(entry)
                except Exception as exc:
                    log.warning("telegram.notify_signal_failed", error=str(exc))

            # Track shadow position for TP/SL simulation
            if is_shadow and proposal.stop_loss and proposal.take_profit:
                _shadow_positions[symbol] = {
                    "side": proposal.side.value,
                    "entry": float(proposal.entry_price or current_price),
                    "tp": float(proposal.take_profit),
                    "sl": float(proposal.stop_loss),
                    "opened_at": datetime.now(tz=UTC),
                }

        async def strategy_loop() -> None:
            nonlocal _balance_tick, _effective_blocked_symbols

            while not self._shutdown_event.is_set():
                self._last_strategy_loop_at = datetime.now(tz=UTC)
                # Refresh balance every N iterations
                _balance_tick += 1
                refresh_every = max(1, int(_BALANCE_REFRESH_INTERVAL / _STRATEGY_LOOP_INTERVAL))
                if _balance_tick % refresh_every == 0:
                    await self._refresh_balance()
                    await self._refresh_closed_pnl_memory()
                await self._sync_execution_positions()
                await self._manage_open_positions()

                # Sync transaction log periodically
                now = datetime.now(tz=UTC)
                tx_interval = self._settings.TRANSACTION_LOG_SYNC_INTERVAL_SECONDS
                if (
                    self._last_tx_log_sync_at is None
                    or (now - self._last_tx_log_sync_at).total_seconds() >= tx_interval
                ):
                    self._last_tx_log_sync_at = now
                    try:
                        await self._sync_transaction_log()
                    except Exception as _tx_exc:
                        log.debug("strategy_loop.tx_log_sync_failed", error=str(_tx_exc))

                balance = self._cached_balance
                capital = balance

                # Feature pipeline runs on full active_symbols universe (set at startup)
                active_symbols = self._screener.active_symbols if self._screener is not None else list(_SYMBOLS)
                _effective_blocked_symbols = self._effective_performance_blocks(active_symbols)

                # Strategy evaluation uses execution_candidates only (Starter-optimized subset)
                exec_symbols = self._screener.execution_candidates if self._screener is not None else list(_SYMBOLS)

                results = await asyncio.gather(
                    *[process_symbol(symbol, balance, capital) for symbol in exec_symbols],
                    return_exceptions=True,
                )
                for symbol, result in zip(exec_symbols, results, strict=False):
                    if isinstance(result, Exception):
                        log.warning(
                            "strategy_loop.symbol_task_failed",
                            symbol=symbol,
                            error=str(result),
                            error_type=type(result).__name__,
                        )

                try:
                    await asyncio.wait_for(
                        asyncio.shield(self._shutdown_event.wait()),
                        timeout=_STRATEGY_LOOP_INTERVAL,
                    )
                except TimeoutError:
                    pass

        task = asyncio.create_task(strategy_loop(), name="strategy-loop")
        self._background_tasks.append(task)
        shadow = self._initial_shadow_mode()
        log.info(
            "strategy_loop.started",
            shadow_mode=shadow,
            initial_capital=str(initial_capital),
        )

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
            symbols=_SYMBOLS,
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

            await self._start_strategy_loop()

            # Supervisor monitors critical tasks and exits on unexpected failure
            supervisor_task = asyncio.create_task(self._run_supervisor(), name="supervisor")
            self._background_tasks.append(supervisor_task)

            # Periodic order/position reconciliation (non-critical, shadow skipped)
            reconciliation_task = asyncio.create_task(self._run_reconciliation(), name="reconciliation")
            self._background_tasks.append(reconciliation_task)

            # Risk monitor: updates equity/drawdown, checks WS staleness
            risk_monitor_task = asyncio.create_task(self._run_risk_monitor(), name="risk-monitor")
            self._background_tasks.append(risk_monitor_task)

            # Outcome resolver: labels prediction events with horizon returns (every 5 min)
            outcome_resolver_task = asyncio.create_task(self._run_outcome_resolver(), name="outcome-resolver")
            self._background_tasks.append(outcome_resolver_task)

            # Auto-training: creates a new shadow challenger when enough fresh labels accumulate
            auto_trainer_task = asyncio.create_task(self._run_auto_model_trainer(), name="auto-model-trainer")
            self._background_tasks.append(auto_trainer_task)

            # Auto-promotion: promotes challenger to champion when it consistently beats the champion
            auto_promoter_task = asyncio.create_task(self._run_auto_model_promoter(), name="auto-model-promoter")
            self._background_tasks.append(auto_promoter_task)

            # Hourly model progress report via Telegram
            model_reporter_task = asyncio.create_task(self._run_model_progress_reporter(), name="model-progress-reporter")
            self._background_tasks.append(model_reporter_task)

            # Adaptive load governor: narrows feature universe under memory/lag pressure
            load_governor_task = asyncio.create_task(self._run_load_governor(), name="load-governor")
            self._background_tasks.append(load_governor_task)

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
