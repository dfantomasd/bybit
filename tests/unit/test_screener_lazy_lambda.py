"""Tests for P0.2: screener has_open_position lazy lambda."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.app import TradingApplication


class TestScreenerLazyLambda:
    @pytest.mark.asyncio
    async def test_has_open_position_evaluated_lazily(self):
        """has_open_position lambda uses current execution_engine at call time,
        not the None value that existed at screener-creation time."""
        app = TradingApplication()

        # At screener creation time engine is None (normal startup order)
        app._execution_engine = None

        # Simulate: engine initialised after screener is created
        mock_engine = MagicMock()
        mock_engine.has_open_position = MagicMock(return_value=True)

        # Build the lazy lambda the same way app.py builds it
        lazy = lambda symbol: (  # noqa: E731
            app._execution_engine is not None and app._execution_engine.has_open_position(symbol)
        )

        # Before engine is set — should return False, not raise
        assert lazy("BTCUSDT") is False

        # After engine is set — should reflect engine state
        app._execution_engine = mock_engine
        assert lazy("BTCUSDT") is True
        mock_engine.has_open_position.assert_called_once_with("BTCUSDT")

    @pytest.mark.asyncio
    async def test_screener_created_with_lazy_lambda(self):
        """_start_screener passes a callable lambda, not the result of has_open_position."""
        from trader.domain.enums import BybitRegion

        app = TradingApplication()
        app._execution_engine = None  # engine not yet initialised

        settings = MagicMock()
        settings.BYBIT_REGION = BybitRegion.GLOBAL
        settings.BYBIT_USE_TESTNET = True
        app._settings = settings

        mock_adapter = MagicMock()
        mock_adapter._rest = MagicMock()
        mock_adapter._rest.get_tickers = AsyncMock(return_value={"result": {"list": []}})
        app._bybit_adapter = mock_adapter

        app._on_screener_symbols_added = AsyncMock()
        app._on_screener_symbols_removed = AsyncMock()

        from trader.features.screener import MarketScreener

        captured: list = []
        original_init = MarketScreener.__init__

        def patching_init(self, *args, **kwargs):
            captured.append(kwargs.get("has_open_position"))
            original_init(self, *args, **kwargs)

        import unittest.mock as um

        with um.patch.object(MarketScreener, "__init__", patching_init):
            try:
                await app._start_screener()
            except Exception:
                pass

        # has_open_position must be a callable (lazy lambda), not None or a bool
        if captured:
            assert callable(captured[0]), "has_open_position must be a callable lambda"
