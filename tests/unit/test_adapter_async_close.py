"""Tests for P1.9: bybit_adapter.close() and bybit_rest.close() are now async."""
from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAdapterAsyncClose:
    def test_bybit_rest_close_is_coroutine(self):
        """BybitRestClient.close() must be a coroutine function (async def)."""
        from trader.exchange.bybit_rest import BybitRestClient
        assert asyncio.iscoroutinefunction(BybitRestClient.close), (
            "BybitRestClient.close must be async def"
        )

    def test_bybit_adapter_close_is_coroutine(self):
        """BybitAdapter.close() must be a coroutine function (async def)."""
        from trader.exchange.bybit_adapter import BybitAdapter
        assert asyncio.iscoroutinefunction(BybitAdapter.close), (
            "BybitAdapter.close must be async def"
        )

    @pytest.mark.asyncio
    async def test_rest_close_awaits_session(self):
        """BybitRestClient.close() awaits aiohttp session.close()."""
        from trader.exchange.bybit_rest import BybitRestClient
        from trader.exchange.endpoint_selector import EndpointSelector
        from trader.exchange.rate_limiter import RateLimiter
        from trader.domain.enums import BybitRegion

        client = BybitRestClient(
            api_key="",
            api_secret="",
            endpoint_selector=EndpointSelector(BybitRegion.GLOBAL, True),
            rate_limiter=RateLimiter(),
            use_testnet=True,
        )

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        client._session = mock_session

        await client.close()

        mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rest_close_noop_when_session_already_closed(self):
        """BybitRestClient.close() does nothing when session is already closed."""
        from trader.exchange.bybit_rest import BybitRestClient
        from trader.exchange.endpoint_selector import EndpointSelector
        from trader.exchange.rate_limiter import RateLimiter
        from trader.domain.enums import BybitRegion

        client = BybitRestClient(
            api_key="",
            api_secret="",
            endpoint_selector=EndpointSelector(BybitRegion.GLOBAL, True),
            rate_limiter=RateLimiter(),
            use_testnet=True,
        )

        mock_session = MagicMock()
        mock_session.closed = True
        mock_session.close = AsyncMock()
        client._session = mock_session

        await client.close()

        mock_session.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_adapter_close_delegates_to_rest(self):
        """BybitAdapter.close() awaits BybitRestClient.close()."""
        from trader.exchange.bybit_adapter import BybitAdapter

        mock_rest = MagicMock()
        mock_rest.close = AsyncMock()

        adapter = BybitAdapter.__new__(BybitAdapter)
        adapter._rest = mock_rest

        await adapter.close()

        mock_rest.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_awaits_adapter_close(self):
        """TradingApplication._graceful_shutdown() awaits adapter.close()."""
        from trader.app import TradingApplication

        app = TradingApplication()
        mock_adapter = MagicMock()
        mock_adapter.close = AsyncMock()
        app._bybit_adapter = mock_adapter

        # Provide the minimal stubs needed by _graceful_shutdown
        app._health_checker = None
        app._feature_pipeline = None
        app._execution_engine = None
        app._telegram_bot = None
        app._ws_public = None
        app._ws_private = None
        app._uvicorn_server = None
        app._trade_journal = None

        await app._graceful_shutdown()

        mock_adapter.close.assert_awaited_once()
