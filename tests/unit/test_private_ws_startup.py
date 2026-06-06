"""Tests for P0.11: Private WebSocket startup and event routing."""
from __future__ import annotations

import asyncio
from decimal import Decimal
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trader.app import TradingApplication
from trader.domain.enums import TradingMode


def _make_app(api_key: str = "key123", api_secret: str = "secret456") -> TradingApplication:
    from trader.domain.enums import BybitRegion
    app = TradingApplication()
    settings = MagicMock()
    settings.BYBIT_API_KEY.get_secret_value.return_value = api_key
    settings.BYBIT_API_SECRET.get_secret_value.return_value = api_secret
    settings.BYBIT_REGION = BybitRegion.GLOBAL
    settings.BYBIT_USE_TESTNET = True
    settings.TRADING_MODE = TradingMode.SHADOW
    app._settings = settings
    return app


class TestPrivateWSStartup:
    @pytest.mark.asyncio
    async def test_private_ws_skipped_when_no_api_credentials(self):
        """_start_private_ws returns without creating WS when credentials absent."""
        app = _make_app(api_key="", api_secret="")

        await app._start_private_ws()

        assert app._ws_private is None
        # No background tasks added
        assert not any(
            t.get_name().startswith("ws-private") for t in app._background_tasks
        )

    @pytest.mark.asyncio
    async def test_private_ws_created_when_credentials_present(self):
        """_start_private_ws creates WS instance and background tasks when credentials set."""
        app = _make_app()
        app._shutdown_event = asyncio.Event()
        app._shutdown_event.set()  # stop immediately

        mock_ws = MagicMock()
        mock_ws.start = AsyncMock()

        with patch("trader.exchange.bybit_ws_private.BybitPrivateWebSocket", return_value=mock_ws):
            await app._start_private_ws()

        assert app._ws_private is mock_ws
        # Two tasks: ws-private + ws-private-consumer
        names = {t.get_name() for t in app._background_tasks}
        assert "ws-private" in names
        assert "ws-private-consumer" in names

    @pytest.mark.asyncio
    async def test_private_ws_balance_update_updates_cached_balance(self):
        """BalanceUpdateEvent with positive balance must update app._cached_balance."""
        from trader.domain.events import BalanceUpdateEvent

        app = _make_app()
        app._cached_balance = Decimal("10")
        shutdown = asyncio.Event()
        queue: asyncio.Queue = asyncio.Queue()

        balance_event = BalanceUpdateEvent(
            account_type="UNIFIED",
            currency="USDT",
            wallet_balance=Decimal("50"),
            available_balance=Decimal("45"),
            unrealised_pnl=Decimal("0"),
        )

        async def fake_consumer() -> None:
            from trader.domain.events import BalanceUpdateEvent as BAE
            while not shutdown.is_set():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.1)
                    if isinstance(event, BAE) and event.available_balance > Decimal("0"):
                        app._cached_balance = event.available_balance
                except TimeoutError:
                    pass
                except asyncio.CancelledError:
                    break

        consumer_task = asyncio.create_task(fake_consumer())
        await queue.put(balance_event)
        await asyncio.sleep(0.05)
        shutdown.set()
        await consumer_task

        assert app._cached_balance == Decimal("45")

    @pytest.mark.asyncio
    async def test_private_ws_balance_zero_does_not_overwrite_cached(self):
        """Zero available_balance from WS must NOT overwrite a valid cached balance."""
        from trader.domain.events import BalanceUpdateEvent

        app = _make_app()
        app._cached_balance = Decimal("100")  # valid cached balance
        shutdown = asyncio.Event()
        queue: asyncio.Queue = asyncio.Queue()

        async def fake_consumer() -> None:
            from trader.domain.events import BalanceUpdateEvent as BAE
            while not shutdown.is_set():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.1)
                    if isinstance(event, BAE) and event.available_balance > Decimal("0"):
                        app._cached_balance = event.available_balance
                except TimeoutError:
                    pass
                except asyncio.CancelledError:
                    break

        zero_event = BalanceUpdateEvent(
            account_type="UNIFIED",
            currency="USDT",
            wallet_balance=Decimal("0"),
            available_balance=Decimal("0"),
            unrealised_pnl=Decimal("0"),
        )
        consumer_task = asyncio.create_task(fake_consumer())
        await queue.put(zero_event)
        await asyncio.sleep(0.05)
        shutdown.set()
        await consumer_task

        # Must remain unchanged
        assert app._cached_balance == Decimal("100")
