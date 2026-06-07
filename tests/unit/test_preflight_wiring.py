"""Tests for P0.1: bybit_adapter.initialize() wired in _start_bybit_adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trader.app import TradingApplication
from trader.domain.enums import TradingMode
from trader.domain.models import PreflightReport


def _make_app(trading_mode: TradingMode = TradingMode.SHADOW, live_mode: bool = False) -> TradingApplication:
    app = TradingApplication()
    settings = MagicMock()
    settings.BYBIT_API_KEY.get_secret_value.return_value = "test-key-abc"
    settings.BYBIT_API_SECRET.get_secret_value.return_value = "test-secret"
    settings.BYBIT_REGION.value = "GLOBAL"
    settings.BYBIT_USE_TESTNET = trading_mode not in (TradingMode.LIVE, TradingMode.CANARY_LIVE)
    settings.DEFAULT_MARKET_CATEGORY = "linear"
    settings.TRADING_MODE = trading_mode
    settings.LIVE_MODE = live_mode
    settings.LIVE_ARMED = live_mode
    settings.SHADOW_MODE = not live_mode
    app._settings = settings
    return app


def _report(passed: bool, errors: list[str] | None = None) -> PreflightReport:
    return PreflightReport(
        passed=passed,
        checks={},
        errors=errors or [],
        warnings=[],
    )


class TestPreflightWiring:
    @pytest.mark.asyncio
    async def test_initialize_called_when_api_key_present(self):
        """adapter.initialize() is awaited when API key is configured."""
        app = _make_app()

        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock(return_value=_report(passed=True))

        with patch("trader.app.TradingApplication._start_bybit_adapter", new=AsyncMock()):
            # Build real _start_bybit_adapter logic by calling the method directly
            pass

        # Test the actual method by injecting the mock adapter
        with patch("trader.exchange.bybit_adapter.BybitAdapter", return_value=mock_adapter):
            await app._start_bybit_adapter()

        mock_adapter.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_preflight_failure_in_shadow_mode_continues(self):
        """Preflight failure in shadow/testnet mode logs warning but does not raise."""
        app = _make_app(trading_mode=TradingMode.SHADOW, live_mode=False)

        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock(return_value=_report(passed=False, errors=["API key invalid"]))

        with patch("trader.exchange.bybit_adapter.BybitAdapter", return_value=mock_adapter):
            # Should NOT raise SystemExit in shadow mode
            await app._start_bybit_adapter()

        assert app._bybit_adapter is not None

    @pytest.mark.asyncio
    async def test_preflight_failure_in_live_mode_raises_system_exit(self):
        """Preflight failure in LIVE mode must raise SystemExit(1)."""
        app = _make_app(trading_mode=TradingMode.LIVE, live_mode=True)

        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock(return_value=_report(passed=False, errors=["balance too low"]))

        with patch("trader.exchange.bybit_adapter.BybitAdapter", return_value=mock_adapter):
            with pytest.raises(SystemExit) as exc_info:
                await app._start_bybit_adapter()

        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_preflight_skipped_when_no_api_key(self):
        """initialize() is NOT called when no API key is configured."""
        app = _make_app()
        app._settings.BYBIT_API_KEY.get_secret_value.return_value = ""

        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock(return_value=_report(passed=True))

        with patch("trader.exchange.bybit_adapter.BybitAdapter", return_value=mock_adapter):
            await app._start_bybit_adapter()

        mock_adapter.initialize.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_preflight_exception_logged_not_raised_in_shadow(self):
        """Exception during initialize() is caught and logged, not propagated in shadow mode."""
        app = _make_app(trading_mode=TradingMode.SHADOW)

        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock(side_effect=RuntimeError("network error"))

        with patch("trader.exchange.bybit_adapter.BybitAdapter", return_value=mock_adapter):
            # Should not raise
            await app._start_bybit_adapter()
