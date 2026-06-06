"""Tests for P0.5: DrawdownTracker state preserved on risk profile hot-swap."""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.app import TradingApplication
from trader.domain.enums import RiskProfile
from trader.risk.drawdown import DrawdownTracker


def _make_app_with_risk_manager() -> TradingApplication:
    from trader.risk.circuit_breakers import CircuitBreakerManager
    from trader.risk.exposure import ExposureTracker
    from trader.risk.kill_switch import KillSwitch
    from trader.risk.manager import RiskManager
    from trader.risk.profiles import get_risk_limits

    app = TradingApplication()

    settings = MagicMock()
    settings.BYBIT_API_KEY.get_secret_value.return_value = ""
    settings.BYBIT_USE_TESTNET = True
    settings.SHADOW_MODE = True
    settings.LIVE_MODE = False
    settings.TRADING_MODE = MagicMock()
    settings.RISK_PROFILE = RiskProfile.CONSERVATIVE
    app._settings = settings

    profile = RiskProfile.CONSERVATIVE
    limits = get_risk_limits(profile)
    capital = Decimal("100")

    drawdown = DrawdownTracker(initial_equity=capital)
    exposure = ExposureTracker(total_capital=capital, risk_limits=limits)
    breakers = CircuitBreakerManager(risk_limits=limits)
    kill_switch = KillSwitch()

    app._risk_manager = RiskManager(
        risk_profile=profile,
        drawdown_tracker=drawdown,
        exposure_tracker=exposure,
        circuit_breaker_manager=breakers,
        kill_switch=kill_switch,
    )
    app._exposure_tracker = exposure
    app._kill_switch = kill_switch
    app._execution_engine = None
    app._telegram_bot = None
    return app


class TestDrawdownPreserved:
    @pytest.mark.asyncio
    async def test_drawdown_tracker_instance_preserved_after_profile_swap(self):
        """The exact DrawdownTracker instance must survive a profile change."""
        app = _make_app_with_risk_manager()
        original_tracker = app._risk_manager._drawdown

        # Simulate some accumulated drawdown
        await original_tracker.update(Decimal("90"))  # 10% drawdown from $100 peak

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                app, "_init_risk_manager", AsyncMock()
            )
            mp.setattr(app, "_refresh_balance", AsyncMock(return_value=Decimal("100")))

            await app._change_risk_profile(RiskProfile.MODERATE)

        assert app._risk_manager._drawdown is original_tracker, (
            "DrawdownTracker instance must be the same object after profile swap"
        )

    @pytest.mark.asyncio
    async def test_drawdown_pct_preserved_after_profile_swap(self):
        """Accumulated drawdown % must not be reset on risk profile change."""
        app = _make_app_with_risk_manager()
        tracker = app._risk_manager._drawdown

        await tracker.update(Decimal("80"))  # 20% drawdown
        drawdown_before = tracker.drawdown_pct
        assert drawdown_before == Decimal("20")

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(app, "_init_risk_manager", AsyncMock())
            mp.setattr(app, "_refresh_balance", AsyncMock(return_value=Decimal("100")))

            await app._change_risk_profile(RiskProfile.MODERATE)

        assert app._risk_manager._drawdown.drawdown_pct == drawdown_before, (
            "Drawdown % must remain the same after profile swap"
        )

    @pytest.mark.asyncio
    async def test_profile_name_updated_after_swap(self):
        """Current profile string is updated to the new profile after swap."""
        app = _make_app_with_risk_manager()
        app._current_risk_profile_str = "CONSERVATIVE"

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(app, "_init_risk_manager", AsyncMock())
            mp.setattr(app, "_refresh_balance", AsyncMock(return_value=Decimal("100")))

            await app._change_risk_profile(RiskProfile.MODERATE)

        assert app._current_risk_profile_str == "MODERATE"

    @pytest.mark.asyncio
    async def test_zero_drawdown_case_preserved(self):
        """When drawdown is 0, profile swap must not introduce fake drawdown."""
        app = _make_app_with_risk_manager()
        tracker = app._risk_manager._drawdown

        # No drawdown: peak == current
        assert tracker.drawdown_pct == Decimal("0")

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(app, "_init_risk_manager", AsyncMock())
            mp.setattr(app, "_refresh_balance", AsyncMock(return_value=Decimal("100")))

            await app._change_risk_profile(RiskProfile.AGGRESSIVE)

        assert app._risk_manager._drawdown.drawdown_pct == Decimal("0")
