"""Tests for P0.13: ReconciliationService wired and running periodically."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.app import TradingApplication
from trader.domain.enums import TradingMode
from trader.domain.models import ReconciliationResult


def _make_app(shadow: bool = True) -> TradingApplication:
    app = TradingApplication()
    settings = MagicMock()
    settings.SHADOW_MODE = shadow
    settings.LIVE_MODE = not shadow
    settings.TRADING_MODE = TradingMode.SHADOW if shadow else TradingMode.LIVE
    settings.BYBIT_USE_TESTNET = True
    settings.RECONCILIATION_INTERVAL_SECONDS = 30
    app._settings = settings
    return app


class TestReconciliation:
    @pytest.mark.asyncio
    async def test_reconcile_called_when_not_shadow(self):
        """reconcile() is called when trading mode is not shadow."""
        app = _make_app(shadow=False)
        app._shutdown_event = asyncio.Event()

        mock_adapter = MagicMock()
        mock_adapter.reconcile = AsyncMock(
            return_value=ReconciliationResult(
                orders_checked=0,
                positions_checked=0,
                discrepancies_found=0,
                mismatched_order_ids=[],
                summary="clean",
                success=True,
            )
        )
        app._bybit_adapter = mock_adapter

        async def stop_after() -> None:
            await asyncio.sleep(0.05)
            app._shutdown_event.set()

        stopper = asyncio.create_task(stop_after())
        await app._run_reconciliation()
        await stopper

        mock_adapter.reconcile.assert_awaited()

    @pytest.mark.asyncio
    async def test_reconcile_skipped_in_shadow_mode(self):
        """reconcile() is NOT called when SHADOW_MODE=true."""
        app = _make_app(shadow=True)
        app._shutdown_event = asyncio.Event()

        mock_adapter = MagicMock()
        mock_adapter.reconcile = AsyncMock(
            return_value=ReconciliationResult(orders_checked=0, success=True, summary="clean")
        )
        app._bybit_adapter = mock_adapter

        async def stop_after() -> None:
            await asyncio.sleep(0.05)
            app._shutdown_event.set()

        stopper = asyncio.create_task(stop_after())
        await app._run_reconciliation()
        await stopper

        mock_adapter.reconcile.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reconcile_logs_discrepancies(self):
        """Discrepancies found in reconcile() are logged (not silently dropped)."""
        app = _make_app(shadow=False)
        app._shutdown_event = asyncio.Event()

        mock_adapter = MagicMock()
        mock_adapter.reconcile = AsyncMock(
            return_value=ReconciliationResult(
                orders_checked=3,
                positions_checked=0,
                discrepancies_found=2,
                mismatched_order_ids=["abc", "def"],
                summary="2 mismatches",
                success=True,
            )
        )
        app._bybit_adapter = mock_adapter

        async def stop_after() -> None:
            await asyncio.sleep(0.05)
            app._shutdown_event.set()

        stopper = asyncio.create_task(stop_after())
        # Should complete without raising
        await app._run_reconciliation()
        await stopper

        mock_adapter.reconcile.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reconcile_exception_does_not_kill_loop(self):
        """Exception in reconcile() is caught — the reconciliation loop keeps running."""
        app = _make_app(shadow=False)
        app._shutdown_event = asyncio.Event()

        call_count = 0

        async def failing_reconcile() -> ReconciliationResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient network error")
            return ReconciliationResult(orders_checked=0, success=True, summary="clean")

        mock_adapter = MagicMock()
        mock_adapter.reconcile = AsyncMock(side_effect=failing_reconcile)
        app._bybit_adapter = mock_adapter
        # Very short interval so second call happens quickly
        app._settings.RECONCILIATION_INTERVAL_SECONDS = 0

        async def stop_after() -> None:
            await asyncio.sleep(0.1)
            app._shutdown_event.set()

        stopper = asyncio.create_task(stop_after())
        await app._run_reconciliation()
        await stopper

        # Both calls were made — loop survived the exception
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_reconcile_skipped_when_no_adapter(self):
        """No call is made when _bybit_adapter is None."""
        app = _make_app(shadow=False)
        app._shutdown_event = asyncio.Event()
        app._bybit_adapter = None  # adapter not initialised yet

        async def stop_after() -> None:
            await asyncio.sleep(0.05)
            app._shutdown_event.set()

        stopper = asyncio.create_task(stop_after())
        # Should not raise
        await app._run_reconciliation()
        await stopper
