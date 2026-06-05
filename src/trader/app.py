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
import os
import signal
import sys
from datetime import UTC, datetime
from decimal import Decimal
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
_BALANCE_REFRESH_INTERVAL = 60.0   # seconds between balance refreshes
_FALLBACK_BALANCE_USD = Decimal("1000")  # used when API key not configured


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
            available = balance.available_balance
            if available > Decimal("0"):
                self._cached_balance = available
                self._balance_refreshed_at = datetime.now(tz=UTC)
                log.debug(
                    "balance.refreshed",
                    available_usd=str(available),
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

        shadow = self._settings.SHADOW_MODE or (
            self._settings.TRADING_MODE != TradingMode.LIVE
        )
        self._execution_engine = ExecutionEngine(
            adapter=self._bybit_adapter,
            risk_manager=self._risk_manager,
            exposure_tracker=self._exposure_tracker,
            shadow_mode=shadow,
            cooldown_s=300,
            category=self._settings.DEFAULT_MARKET_CATEGORY,
        )

        # Sync open positions from exchange so we don't double-enter on restart
        await self._execution_engine.sync_positions()
        log.info("execution_engine.initialized", shadow_mode=shadow)

    async def _start_screener(self) -> list[str]:
        """Run the market screener and return initial symbol list."""
        from trader.features.screener import MarketScreener

        assert self._bybit_adapter is not None

        self._screener = MarketScreener(
            rest_client=self._bybit_adapter._rest,
            max_symbols=10,
            min_volume_usd=20_000_000,
            interval_s=900,
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
        """Start feature computation in background (parallel across symbols)."""
        from trader.features.pipeline import FeaturePipeline
        from trader.features.regime import RegimeClassifier

        assert self._candle_store is not None

        self._feature_pipeline = FeaturePipeline(
            candle_store=self._candle_store,
            health_checker=self._health_checker,
            interval_s=_FEATURE_INTERVAL,
        )
        self._regime_classifier = RegimeClassifier()

        task = asyncio.create_task(
            self._feature_pipeline.run(
                symbols=_SYMBOLS,  # fallback; screener overrides at runtime
                intervals=[_WS_INTERVAL],
                symbol_source=self._screener,  # dynamic symbol list
            ),
            name="feature-pipeline",
        )
        self._background_tasks.append(task)
        log.info("feature_pipeline.started", parallel=True)

    async def _start_strategy_loop(self) -> None:
        """Run strategy ensemble → RiskManager → ExecutionEngine."""
        from trader.strategies.ensemble import StrategyEnsemble
        from trader.strategies.trend import EMAcrossoverStrategy

        assert self._settings is not None

        # Fetch initial balance to seed RiskManager
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

        strategies = [
            EMAcrossoverStrategy(
                symbol=symbol,
                allow_short=False,
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

        _balance_tick: int = 0

        async def process_symbol(symbol: str, balance: Decimal, capital: Decimal) -> None:
            """Evaluate one symbol: features → regime → ensemble → execution."""
            if self._feature_pipeline is None:
                return

            vec = self._feature_pipeline.latest(symbol, _WS_INTERVAL)
            if vec is None:
                return

            closes = (
                self._candle_store.closes(symbol, _WS_INTERVAL, 1)
                if self._candle_store
                else []
            )
            if not closes:
                return
            current_price = closes[-1]

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

            # ExecutionEngine: RiskManager → order (or shadow log)
            if self._execution_engine is not None:
                try:
                    await self._execution_engine.submit(
                        proposal=proposal,
                        capital=capital,
                        available_balance=balance,
                        feature_vector=vec,
                        regime_context=regime_ctx,
                    )
                except Exception as exc:
                    log.warning("strategy_loop.execution_error", symbol=symbol, error=str(exc))

        async def strategy_loop() -> None:
            nonlocal _balance_tick

            while not self._shutdown_event.is_set():
                # Refresh balance every N iterations
                _balance_tick += 1
                refresh_every = max(1, int(_BALANCE_REFRESH_INTERVAL / _STRATEGY_LOOP_INTERVAL))
                if _balance_tick % refresh_every == 0:
                    await self._refresh_balance()

                balance = self._cached_balance
                capital = balance

                # Get current active symbols from screener (dynamic)
                active_symbols = (
                    self._screener.active_symbols
                    if self._screener is not None
                    else _SYMBOLS
                )

                # Analyse ALL symbols in parallel
                await asyncio.gather(
                    *[
                        process_symbol(symbol, balance, capital)
                        for symbol in active_symbols
                    ],
                    return_exceptions=True,
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
        shadow = self._settings.SHADOW_MODE or (
            self._settings.TRADING_MODE != TradingMode.LIVE
        )
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

        if self._health_checker:
            self._health_checker.set_system_status(self._status)

        if self._feature_pipeline:
            self._feature_pipeline.stop()

        # Log final execution state before shutdown
        if self._execution_engine is not None:
            status = self._execution_engine.get_status()
            log.info(
                "execution_engine.shutdown_status",
                open_positions=len(status["open_positions"]),
                shadow_mode=status["shadow_mode"],
            )

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
