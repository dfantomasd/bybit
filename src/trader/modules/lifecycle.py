"""Application lifecycle: settings, preflight, HTTP, adapters, shutdown."""

from __future__ import annotations

import asyncio
import os
import secrets
from typing import Any

import uvicorn

from trader.domain.enums import SystemStatus, TradingMode
from trader.modules.base import AppBoundModule
from trader.monitoring.logging import configure_logging, get_logger
from trader.runtime.constants import _WS_INTERVAL

log = get_logger(__name__)


class LifecycleModule(AppBoundModule):
    name = "lifecycle"

    def _initial_shadow_mode(self) -> bool:
        return self._app._modules.signal_policy.initial_shadow_mode()

    async def load_settings(self) -> None:
        from trader.config import Settings

        self._app._settings = Settings()

        if self._app._settings.TRADING_MODE == TradingMode.LIVE and not self._app._settings.LIVE_MODE:
            log.critical(
                "live_mode_safety_gate_blocked",
                reason="LIVE_MODE env var must be explicitly set to true",
            )
            raise SystemExit(1)

    async def configure_observability(self) -> None:
        assert self._app._settings is not None
        configure_logging(
            log_level=self._app._settings.LOG_LEVEL,
            log_format=self._app._settings.LOG_FORMAT,
        )
        self._app._current_risk_profile_str = self._app._settings.RISK_PROFILE.value
        from trader.monitoring.deploy_info import get_deploy_info
        from trader.training.labels import active_label_schema_version

        deploy = get_deploy_info()
        log.info(
            "settings_loaded",
            trading_mode=self._app._settings.TRADING_MODE,
            risk_profile=self._app._settings.RISK_PROFILE,
            bybit_use_testnet=self._app._settings.BYBIT_USE_TESTNET,
            live_mode=self._app._settings.LIVE_MODE,
            deploy_id=deploy.get("deploy_id") or None,
            git_commit=deploy.get("git_commit") or None,
            train_strategy_allowlist=self._app._settings.TRAIN_STRATEGY_ALLOWLIST,
            train_include_candle_baseline=self._app._settings.TRAIN_INCLUDE_CANDLE_BASELINE,
            model_label_use_tpsl_exit=self._app._settings.MODEL_LABEL_USE_TPSL_EXIT,
            active_label_schema=active_label_schema_version(
                use_tpsl_exit=bool(self._app._settings.MODEL_LABEL_USE_TPSL_EXIT)
            ),
        )

    async def run_preflight(self) -> None:
        from trader.exchange.endpoint_selector import EndpointSelector
        from trader.monitoring.health import HealthChecker

        assert self._app._settings is not None
        self._app._status = SystemStatus.PREFLIGHT

        bybit_base = EndpointSelector(
            self._app._settings.BYBIT_REGION,
            self._app._settings.BYBIT_USE_TESTNET,
        ).rest_base

        postgres_required = (
            self._app._settings.PREFLIGHT_POSTGRES_REQUIRED
            if self._app._settings.PREFLIGHT_POSTGRES_REQUIRED is not None
            else self._app._settings.TRADING_MODE in (TradingMode.CANARY_LIVE, TradingMode.LIVE)
        )

        self._app._health_checker = HealthChecker(
            postgres_dsn=self._app._settings.POSTGRES_DSN.get_secret_value(),
            redis_url=self._app._settings.REDIS_URL.get_secret_value(),
            redis_required=self._app._settings.REDIS_REQUIRED,
            bybit_required=self._app._settings.BYBIT_CONNECTIVITY_REQUIRED,
            bybit_rest_url=bybit_base,
            trading_mode=self._app._settings.TRADING_MODE,
            system_status=self._app._status,
            model_enabled=self._app._settings.MODEL_ENABLED,
            postgres_retry_attempts=self._app._settings.PREFLIGHT_POSTGRES_RETRY_ATTEMPTS,
            postgres_retry_delay_s=self._app._settings.PREFLIGHT_POSTGRES_RETRY_DELAY_SECONDS,
            postgres_required=postgres_required,
            postgres_optional_max_attempts=self._app._settings.PREFLIGHT_POSTGRES_OPTIONAL_MAX_ATTEMPTS,
        )

        result = await self._app._health_checker.run_preflight()
        checks = result["checks"]
        postgres_required = bool(result.get("postgres_required", True))

        for check_name, passed in checks.items():
            if passed:
                log.info("preflight_check_passed", check=check_name)
            elif check_name == "postgres" and not postgres_required:
                log.warning(
                    "preflight_check_deferred",
                    check=check_name,
                    trading_mode=self._app._settings.TRADING_MODE.value,
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
                trading_mode=self._app._settings.TRADING_MODE.value,
            )

        log.info("preflight_passed")

    async def start_trade_journal(self) -> None:
        """Start best-effort Postgres memory for trades and performance."""
        from trader.storage.trade_journal import TradeJournal

        assert self._app._settings is not None
        self._app._trade_journal = TradeJournal(
            postgres_dsn=self._app._settings.POSTGRES_DSN.get_secret_value(),
            enabled=self._app._settings.TRADE_JOURNAL_ENABLED,
            fetch_timeout_seconds=self._app._settings.TRADE_JOURNAL_FETCH_TIMEOUT_SECONDS,
            pool_max_size=self._app._settings.TRADE_JOURNAL_POOL_MAX_SIZE,
            reconnect_max_backoff_seconds=self._app._settings.TRADE_JOURNAL_RECONNECT_MAX_BACKOFF_SECONDS,
            auth_circuit_breaker_min_backoff_seconds=self._app._settings.TRADE_JOURNAL_AUTH_CIRCUIT_BREAKER_MIN_BACKOFF_SECONDS,
        )
        await self._app._trade_journal.connect()
        await self._app._modules.ops.maybe_run_startup_retention()
        task = asyncio.create_task(
            self._app._modules.ops.run_trade_journal_reconnector(),
            name="trade-journal-reconnector",
        )
        self._app._background_tasks.append(task)

    async def restore_execution_pending_entries(self) -> None:
        """Reload unresolved durable pending entries into ExecutionEngine."""
        if self._initial_shadow_mode():
            return
        if (
            self._app._trade_journal is None
            or self._app._execution_engine is None
            or not self._app._trade_journal.is_enabled
        ):
            return
        try:
            pending_records = await self._app._trade_journal.get_pending_durable_orders()
            unresolved_records = []
            skipped_resolved = []
            for record in pending_records:
                oid = str(record.get("order_link_id") or "")
                if oid and await self._app._trade_journal.is_order_resolved(oid):
                    skipped_resolved.append(oid)
                    continue
                unresolved_records.append(record)
            if skipped_resolved:
                log.info(
                    "execution_engine.pending_restore_skipped_resolved",
                    ids=skipped_resolved,
                )
            if unresolved_records:
                self._app._execution_engine.restore_pending_entries_with_symbols(unresolved_records)
                log.info(
                    "execution_engine.pending_restored",
                    count=len(unresolved_records),
                    ids=[r.get("order_link_id") for r in unresolved_records],
                )
        except Exception as exc:
            log.warning("execution_engine.pending_restore_failed", error=str(exc))

    async def start_http_server(self) -> asyncio.Task[Any]:
        from trader.api.fastapi_app import create_app
        from trader.runtime.state_proxy import AppStateProxy

        assert self._app._settings is not None

        internal_api_key = self._app._settings.INTERNAL_API_KEY.get_secret_value()
        if not internal_api_key:
            internal_api_key = secrets.token_urlsafe(32)
            log.warning(
                "http_server.generated_internal_api_key",
                reason="INTERNAL_API_KEY is not configured; authenticated endpoints are only usable inside this process",
            )
        port = int(os.getenv("PORT", str(self._app._settings.FASTAPI_PORT)))
        log.info("http_server_starting", port=port)

        operator = self._app._modules.operator
        fastapi_app = create_app(
            api_key=internal_api_key,
            health_checker=self._app._health_checker,
            state_store=AppStateProxy(self._app),
            trade_journal=self._app._trade_journal,
            runtime_settings=operator.runtime_settings,
            set_runtime_setting=operator.set_runtime_setting,
        )
        self._app._fastapi_app = fastapi_app

        config = uvicorn.Config(
            app=fastapi_app,
            # Container service must bind internally; external exposure belongs to the platform.
            host="0.0.0.0",  # noqa: S104  # nosec B104
            port=port,
            log_level="warning",
            access_log=False,
        )
        self._app._uvicorn_server = uvicorn.Server(config=config)
        task = asyncio.create_task(self._app._uvicorn_server.serve(), name="http-server")
        self._app._background_tasks.append(task)
        return task

    async def start_bybit_adapter(self) -> None:
        from trader.exchange.bybit_adapter import BybitAdapter

        assert self._app._settings is not None
        self._app._bybit_adapter = BybitAdapter(
            api_key=self._app._settings.BYBIT_API_KEY.get_secret_value(),
            api_secret=self._app._settings.BYBIT_API_SECRET.get_secret_value(),
            region_code=self._app._settings.BYBIT_REGION.value,
            use_testnet=self._app._settings.BYBIT_USE_TESTNET,
            default_category=self._app._settings.DEFAULT_MARKET_CATEGORY,
            trade_journal=self._app._trade_journal,
            trading_mode=self._app._settings.TRADING_MODE.value,
        )
        log.info("bybit_adapter_created", category=self._app._settings.DEFAULT_MARKET_CATEGORY)

        from trader.exchange.fee_provider import FeeRateProvider

        self._app._fee_provider = FeeRateProvider(
            rest=self._app._bybit_adapter._rest,
            category=self._app._settings.DEFAULT_MARKET_CATEGORY,
            default_maker=self._app._settings.DEFAULT_LINEAR_MAKER_FEE_RATE,
            default_taker=self._app._settings.DEFAULT_LINEAR_TAKER_FEE_RATE,
            shadow_mode=self._initial_shadow_mode(),
        )

        # Run Bybit exchange preflight checks (clock skew, API perms, balance, etc.)
        has_key = bool(self._app._settings.BYBIT_API_KEY.get_secret_value())
        if has_key:
            try:
                report = await self._app._bybit_adapter.initialize()
                is_live = self._app._settings.LIVE_MODE and self._app._settings.TRADING_MODE in (
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
                is_active = self._app._settings.LIVE_MODE and self._app._settings.TRADING_MODE in (
                    TradingMode.LIVE,
                    TradingMode.CANARY_LIVE,
                )
                if is_active:
                    log.critical("bybit_preflight_exception_blocking_live", error=str(exc))
                    raise SystemExit(1) from exc
                log.warning("bybit_preflight_exception_continuing_shadow", error=str(exc))
        else:
            log.info("bybit_adapter_skipped_preflight", reason="no_api_key_configured")

    async def start_feature_pipeline(self) -> None:
        """Start event-driven feature pipeline with 60 s staleness watchdog."""
        from trader.features.pipeline import FeaturePipeline
        from trader.features.regime import RegimeClassifier

        assert self._app._candle_store is not None

        self._app._feature_pipeline = FeaturePipeline(
            candle_store=self._app._candle_store,
            health_checker=self._app._health_checker,
            stale_threshold_s=90.0,
            watchdog_interval_s=60.0,
            orderbook_tracker=self._app._orderbook_tracker,
            market_stats_source=self._app._screener,
        )
        self._app._regime_classifier = RegimeClassifier()

        task = asyncio.create_task(
            self._app._feature_pipeline.run(
                symbols=self._active_symbols(),  # actual screener universe (fallback if absent)
                intervals=[_WS_INTERVAL, "5", "15"],
                symbol_source=self._app._screener,
            ),
            name="feature-pipeline",
        )
        self._app._background_tasks.append(task)

        # Warm HTF feature vectors from REST-seeded candles so LIVE MTF gates
        # do not wait up to 15 minutes for the first WS close.
        for symbol in self._active_symbols():
            for interval in (_WS_INTERVAL, "5", "15"):
                try:
                    await self._app._feature_pipeline.on_confirmed_candle(symbol, interval)
                except Exception as exc:
                    log.debug(
                        "feature_pipeline.startup_warmup_failed",
                        symbol=symbol,
                        interval=interval,
                        error=str(exc),
                    )

        log.info("feature_pipeline.started", mode="event_driven", watchdog_interval_s=60.0)

    async def graceful_shutdown(self) -> None:
        log.info("graceful_shutdown_starting")
        self._app._status = SystemStatus.STOPPING
        self._app._trading_paused = True  # pause new entries immediately

        if self._app._health_checker:
            self._app._health_checker.set_system_status(self._app._status)

        if self._app._feature_pipeline:
            self._app._feature_pipeline.stop()

        # Run reconciliation before stopping to catch any order state mismatches
        _is_shadow = self._app._settings is None or self._initial_shadow_mode()
        if self._app._bybit_adapter is not None and not _is_shadow:
            try:
                result = await asyncio.wait_for(self._app._bybit_adapter.reconcile(), timeout=10.0)
                log.info(
                    "graceful_shutdown.reconciliation",
                    discrepancies=result.discrepancies_found,
                    summary=result.summary,
                )
            except Exception as exc:
                log.warning("graceful_shutdown.reconciliation_failed", error=str(exc))

        # Log final execution state and open positions before shutdown
        if self._app._execution_engine is not None:
            status = self._app._execution_engine.get_status()
            log.info(
                "execution_engine.shutdown_status",
                open_positions=len(status["open_positions"]),
                shadow_mode=status["shadow_mode"],
            )
            # Alert via Telegram about shutdown with open positions
            if status["open_positions"] and self._app._telegram_bot is not None:
                try:
                    pos_list = ", ".join(status["open_positions"].keys())
                    await self._app._telegram_bot.notify(
                        f"⚠️ <b>Shutdown with open positions</b>: <code>{pos_list}</code>\n"
                        "Open positions remain on exchange. Verify SL manually."
                    )
                except Exception as exc:
                    log.debug("graceful_shutdown.telegram_failed", error=str(exc))

        if self._app._telegram_bot:
            await self._app._telegram_bot.stop()

        if self._app._ws_public:
            await self._app._ws_public.stop()

        if self._app._ws_private:
            await self._app._ws_private.stop()

        # Cancel all background tasks
        for task in self._app._background_tasks:
            if not task.done():
                task.cancel()
        if self._app._background_tasks:
            await asyncio.gather(*self._app._background_tasks, return_exceptions=True)

        if self._app._uvicorn_server:
            self._app._uvicorn_server.should_exit = True
            await asyncio.sleep(1)

        if self._app._bybit_adapter:
            await self._app._bybit_adapter.close()

        if self._app._trade_journal:
            await self._app._trade_journal.close()

        self._app._status = SystemStatus.STOPPED
        log.info("graceful_shutdown_complete")
