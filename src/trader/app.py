"""Application entry point.

Lifecycle:
1. Parse config
2. Configure logging
3. Run preflight checks
4. Start health-check HTTP server
5. Start WebSocket connections (public market data)
6. Seed candle store from REST history
7. Start feature pipeline
8. Start strategy ensemble loop (SHADOW: log proposals, no execution)
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
import os
import signal
import sys
from datetime import UTC, datetime
from typing import Any

import uvicorn

from trader.domain.enums import SystemStatus, TradingMode
from trader.monitoring.logging import configure_logging, get_logger

log = get_logger(__name__)

# Symbols and intervals to track
_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
_WS_INTERVAL = "1"   # 1-minute klines over WS
_MIN_SEED_BARS = 60  # bars to fetch from REST at startup
_STRATEGY_LOOP_INTERVAL = 10.0  # seconds between strategy evaluations
_FEATURE_INTERVAL = 5.0         # seconds between feature recomputation


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
        self._background_tasks: list[asyncio.Task] = []

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

    async def _start_http_server(self) -> asyncio.Task:
        from trader.api.fastapi_app import create_app

        assert self._settings is not None
        import secrets

        internal_api_key = secrets.token_urlsafe(32)
        port = int(os.getenv("PORT", str(self._settings.FASTAPI_PORT)))
        log.info("http_server_starting", port=port)

        fastapi_app = create_app(
            api_key=internal_api_key,
            health_checker=self._health_checker,
        )

        config = uvicorn.Config(
            app=fastapi_app,
            host="0.0.0.0",  # noqa: S104
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
        )
        log.info("bybit_adapter_created", category=self._settings.DEFAULT_MARKET_CATEGORY)

    async def _start_telegram_bot(self) -> None:
        from trader.telegram_bot import TelegramBotConfig, TelegramMonitorBot

        assert self._settings is not None
        assert self._health_checker is not None
        token = self._settings.TELEGRAM_BOT_TOKEN.get_secret_value()
        if not token:
            log.info("telegram_bot_skipped", reason="no token configured")
            return

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
        )
        await self._telegram_bot.start()
        log.info("telegram_bot_started")

    # ------------------------------------------------------------------
    # Market data & features
    # ------------------------------------------------------------------

    async def _seed_candle_store(self) -> None:
        """Fetch recent historical klines via REST to seed the CandleStore."""
        from trader.data.candles import Candle, CandleStore

        assert self._settings is not None
        assert self._bybit_adapter is not None

        if self._candle_store is None:
            self._candle_store = CandleStore(max_bars=500)

        has_api_key = bool(self._settings.BYBIT_API_KEY.get_secret_value())

        for symbol in _SYMBOLS:
            for interval in [_WS_INTERVAL]:
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

    async def _start_public_ws(self) -> None:
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

        # Build subscription list
        category = self._settings.DEFAULT_MARKET_CATEGORY
        subs: list[str] = []
        for symbol in _SYMBOLS:
            subs.append(f"kline.{_WS_INTERVAL}.{symbol}")
            subs.append(f"tickers.{symbol}")

        event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

        self._ws_public = BybitPublicWebSocket(
            endpoint=f"{selector.ws_public_base}/{category}",
            subscriptions=subs,
            event_queue=event_queue,
        )

        # Event consumer: feeds CandleStore and updates health
        async def consume_events() -> None:
            from trader.data.candles import candle_from_kline_event
            from trader.domain.events import KlineEvent

            while not self._shutdown_event.is_set():
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                    if isinstance(event, KlineEvent):
                        candle = candle_from_kline_event(event)
                        self._candle_store.add(event.symbol, event.interval, candle)
                    # Update WS health on any message
                    if self._health_checker:
                        self._health_checker.set_ws_status(
                            connected=True,
                            last_message_at=datetime.now(tz=UTC),
                        )
                except asyncio.TimeoutError:
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

    async def _start_feature_pipeline(self) -> None:
        """Start feature computation in background."""
        from trader.features.pipeline import FeaturePipeline

        assert self._candle_store is not None

        self._feature_pipeline = FeaturePipeline(
            candle_store=self._candle_store,
            health_checker=self._health_checker,
            interval_s=_FEATURE_INTERVAL,
        )

        task = asyncio.create_task(
            self._feature_pipeline.run(
                symbols=_SYMBOLS,
                intervals=[_WS_INTERVAL],
            ),
            name="feature-pipeline",
        )
        self._background_tasks.append(task)
        log.info("feature_pipeline.started")

    async def _start_strategy_loop(self) -> None:
        """Run strategy ensemble in SHADOW mode (proposals logged, not executed)."""
        from trader.strategies.ensemble import StrategyEnsemble
        from trader.strategies.trend import EMAcrossoverStrategy

        assert self._settings is not None

        strategies = [
            EMAcrossoverStrategy(
                symbol=symbol,
                allow_short=False,  # CONSERVATIVE: no shorts initially
                min_qty_usd=10.0,
                max_risk_pct=0.003,
            )
            for symbol in _SYMBOLS
        ]

        self._strategy_ensemble = StrategyEnsemble(
            strategies=strategies,
            health_checker=self._health_checker,
            min_confidence=0.55,
        )

        async def strategy_loop() -> None:
            while not self._shutdown_event.is_set():
                if self._feature_pipeline is not None:
                    for symbol in _SYMBOLS:
                        vec = self._feature_pipeline.latest(symbol, _WS_INTERVAL)
                        if vec is None:
                            continue
                        # Current price from last close
                        closes = self._candle_store.closes(symbol, _WS_INTERVAL, 1) if self._candle_store else []
                        if not closes:
                            continue
                        current_price = closes[-1]
                        try:
                            proposal = self._strategy_ensemble.evaluate_all(
                                feature_vector=vec,
                                current_price=current_price,
                                available_balance_usd=1000.0,  # placeholder
                            )
                            if proposal is not None:
                                log.info(
                                    "shadow.proposal",
                                    symbol=proposal.symbol,
                                    side=proposal.side.value,
                                    confidence=round(proposal.confidence, 3),
                                    qty=str(proposal.requested_qty),
                                    rationale=proposal.rationale,
                                    mode="SHADOW_NO_EXECUTION",
                                )
                        except Exception as exc:
                            log.warning(
                                "strategy_loop.error",
                                symbol=symbol,
                                error=str(exc),
                            )

                try:
                    await asyncio.wait_for(
                        asyncio.shield(self._shutdown_event.wait()),
                        timeout=_STRATEGY_LOOP_INTERVAL,
                    )
                except asyncio.TimeoutError:
                    pass

        task = asyncio.create_task(strategy_loop(), name="strategy-loop")
        self._background_tasks.append(task)
        log.info("strategy_loop.started", mode="SHADOW")

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

        if self._health_checker:
            self._health_checker.set_system_status(self._status)

        if self._feature_pipeline:
            self._feature_pipeline.stop()

        if self._telegram_bot:
            await self._telegram_bot.stop()

        if self._ws_public:
            await self._ws_public.stop()

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
            self._bybit_adapter.close()

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
            await self._start_bybit_adapter()
            await self._start_telegram_bot()

            # Market data pipeline
            await self._seed_candle_store()
            await self._start_public_ws()

            # Give WS a moment to connect before starting strategies
            await asyncio.sleep(3.0)

            await self._start_feature_pipeline()

            # Give features a moment to compute from seeded data
            await asyncio.sleep(2.0)

            await self._start_strategy_loop()

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
