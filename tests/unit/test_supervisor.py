"""Tests for the runtime supervisor in TradingApplication."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.app import _CRITICAL_TASK_NAMES, TradingApplication


class TestRuntimeSupervisor:
    def test_critical_task_names_are_defined(self):
        """All expected critical task names are present."""
        expected = {"screener", "ws-public", "ws-consumer", "feature-pipeline", "strategy-loop"}
        assert expected <= set(_CRITICAL_TASK_NAMES)

    @pytest.mark.asyncio
    async def test_supervisor_exits_when_critical_task_dies(self):
        """_run_supervisor sets shutdown_event when a critical task finishes unexpectedly."""
        app = TradingApplication()
        app._shutdown_event = asyncio.Event()
        app._telegram_bot = None

        # Create a task that's already done with an exception
        async def _fail() -> None:
            raise RuntimeError("strategy loop crashed")

        dying_task = asyncio.create_task(_fail(), name="strategy-loop")
        await asyncio.sleep(0)  # let it run to completion

        app._background_tasks = [dying_task]

        await app._run_supervisor()

        assert app._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_supervisor_does_not_exit_for_non_critical_task(self):
        """Supervisor ignores tasks whose names are not in _CRITICAL_TASK_NAMES."""
        app = TradingApplication()
        app._shutdown_event = asyncio.Event()
        app._telegram_bot = None

        async def _http_done() -> None:
            pass

        done_task = asyncio.create_task(_http_done(), name="http-server")
        await asyncio.sleep(0)

        app._background_tasks = [done_task]

        # Supervisor should NOT exit; set shutdown immediately so loop terminates
        async def _stop_soon() -> None:
            await asyncio.sleep(0.05)
            app._shutdown_event.set()

        stopper = asyncio.create_task(_stop_soon())
        # Should complete without SystemExit
        await app._run_supervisor()
        await stopper

    @pytest.mark.asyncio
    async def test_supervisor_sends_telegram_alert_before_exit(self):
        """Telegram alert is sent and shutdown_event is set when a critical task dies."""
        app = TradingApplication()
        app._shutdown_event = asyncio.Event()

        notify_mock = AsyncMock()
        app._telegram_bot = MagicMock()
        app._telegram_bot.notify = notify_mock

        async def _fail() -> None:
            raise RuntimeError("ws-public died")

        dying_task = asyncio.create_task(_fail(), name="ws-public")
        await asyncio.sleep(0)
        app._background_tasks = [dying_task]

        await app._run_supervisor()

        assert app._shutdown_event.is_set()
        notify_mock.assert_awaited_once()
        call_text: str = notify_mock.await_args.args[0]
        assert "ws-public" in call_text

    @pytest.mark.asyncio
    async def test_supervisor_cancelled_task_does_not_exit(self):
        """A cancelled task (normal shutdown) is not treated as an unexpected death."""
        app = TradingApplication()
        app._shutdown_event = asyncio.Event()
        app._telegram_bot = None

        async def _forever() -> None:
            await asyncio.sleep(9999)

        running_task = asyncio.create_task(_forever(), name="strategy-loop")
        running_task.cancel()
        try:
            await running_task
        except asyncio.CancelledError:
            pass

        app._background_tasks = [running_task]
        # Signal shutdown so supervisor exits cleanly
        app._shutdown_event.set()

        # Should not raise SystemExit — cancelled tasks are from clean shutdown
        await app._run_supervisor()

    def test_diagnostics_include_telegram_health(self):
        """Telegram polling health is visible in app diagnostics."""
        app = TradingApplication()
        app._telegram_bot = MagicMock()
        app._telegram_bot.health_snapshot.return_value = {
            "enabled": True,
            "app_running": True,
            "polling_running": True,
            "polling_conflict_count": 0,
        }

        diag = app.get_diagnostics()

        assert diag["telegram"]["polling_running"] is True
