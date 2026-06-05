"""Application entry point.

Lifecycle:
1. Parse config
2. Configure logging
3. Run preflight checks
4. Start health-check HTTP server
5. Enter main event loop
6. On SIGTERM/SIGINT: graceful shutdown

CRITICAL SAFETY RULES:
- System starts in TESTNET or SHADOW mode by default.
- LIVE mode requires explicit LIVE_MODE=true AND TRADING_MODE=LIVE in config.
- The Risk Manager is always the final authority; it cannot be bypassed here.
"""
from __future__ import annotations

import asyncio
import signal
import sys
from typing import Any

import uvicorn

from trader.domain.enums import SystemStatus, TradingMode
from trader.monitoring.logging import configure_logging, get_logger

log = get_logger(__name__)


class TradingApplication:
    """Top-level application orchestrator."""

    def __init__(self) -> None:
        self._status: SystemStatus = SystemStatus.STARTING
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._settings: Any | None = None
        self._health_checker: Any | None = None
        self._uvicorn_server: uvicorn.Server | None = None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def _load_settings(self) -> None:
        """Load and validate configuration. Raises ConfigurationError on failure."""
        from trader.config import Settings

        self._settings = Settings()
        log.info(
            "settings_loaded",
            trading_mode=self._settings.TRADING_MODE,
            risk_profile=self._settings.RISK_PROFILE,
            bybit_use_testnet=self._settings.BYBIT_USE_TESTNET,
            live_mode=self._settings.LIVE_MODE,
        )

        # Enforce safety gate
        if self._settings.TRADING_MODE == TradingMode.LIVE and not self._settings.LIVE_MODE:
            log.critical(
                "live_mode_safety_gate_blocked",
                reason="LIVE_MODE env var must be explicitly set to true",
            )
            raise SystemExit(1)

    async def _configure_observability(self) -> None:
        """Set up structured logging and Prometheus metrics."""
        assert self._settings is not None
        configure_logging(
            log_level=self._settings.LOG_LEVEL,
            log_format=self._settings.LOG_FORMAT,
        )
        log.info("observability_configured")

    async def _run_preflight(self) -> None:
        """Run preflight checks. Abort startup if critical checks fail."""
        from trader.monitoring.health import HealthChecker

        assert self._settings is not None
        self._status = SystemStatus.PREFLIGHT

        bybit_base = (
            "https://api-testnet.bybit.com"
            if self._settings.BYBIT_USE_TESTNET
            else "https://api.bybit.com"
        )

        self._health_checker = HealthChecker(
            postgres_dsn=self._settings.POSTGRES_DSN.get_secret_value(),
            redis_url=self._settings.REDIS_URL.get_secret_value(),
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

    async def _start_http_server(self) -> asyncio.Task[None]:
        """Start the FastAPI observability server as a background task."""
        from trader.api.fastapi_app import create_app

        assert self._settings is not None

        # Generate a random internal API key at startup (not user-facing)
        import secrets

        internal_api_key = secrets.token_urlsafe(32)
        log.info(
            "http_server_starting",
            port=self._settings.FASTAPI_PORT,
        )

        fastapi_app = create_app(
            api_key=internal_api_key,
            health_checker=self._health_checker,
        )

        config = uvicorn.Config(
            app=fastapi_app,
            host="0.0.0.0",
            port=self._settings.FASTAPI_PORT,
            log_level="warning",
            access_log=False,
        )
        self._uvicorn_server = uvicorn.Server(config=config)

        return asyncio.create_task(
            self._uvicorn_server.serve(),
            name="http-server",
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _main_loop(self) -> None:
        """Main trading loop — placeholder for Phase 2 components."""
        assert self._settings is not None
        self._status = SystemStatus.RUNNING

        if self._health_checker:
            self._health_checker.set_system_status(self._status)

        log.info(
            "trading_system_running",
            trading_mode=self._settings.TRADING_MODE,
            risk_profile=self._settings.RISK_PROFILE,
            live_mode=self._settings.LIVE_MODE,
        )

        # Phase 2 will start strategy workers, data feeds, etc. here.
        # For now, just keep the process alive until shutdown is requested.
        while not self._shutdown_event.is_set():
            await asyncio.sleep(1)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _handle_signal(self, sig: int) -> None:
        """Signal handler: request graceful shutdown."""
        log.warning("shutdown_signal_received", signal=signal.Signals(sig).name)
        self._shutdown_event.set()

    async def _graceful_shutdown(self) -> None:
        """Perform ordered shutdown of all components."""
        log.info("graceful_shutdown_starting")
        self._status = SystemStatus.STOPPING

        if self._health_checker:
            self._health_checker.set_system_status(self._status)

        # Phase 2: cancel open orders, flush queues, close WS, etc.
        # For now, just stop the HTTP server.
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
            # Give it a moment to finish in-flight requests
            await asyncio.sleep(1)

        self._status = SystemStatus.STOPPED
        log.info("graceful_shutdown_complete")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Full application lifecycle."""
        loop = asyncio.get_running_loop()

        # Register signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_signal, sig)

        try:
            await self._load_settings()
            await self._configure_observability()
            await self._run_preflight()

            http_task = await self._start_http_server()

            try:
                await self._main_loop()
            finally:
                await self._graceful_shutdown()
                http_task.cancel()
                try:
                    await http_task
                except asyncio.CancelledError:
                    pass

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
    """Async entry point."""
    app = TradingApplication()
    await app.run()


def main_sync() -> None:
    """Synchronous entry point for the ``trader`` CLI command."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
    except SystemExit as e:
        sys.exit(e.code)


# Allow running as ``python -m trader.app``
if __name__ == "__main__":
    main_sync()
