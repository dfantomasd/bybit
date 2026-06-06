"""Tests for P0.5 complete + P0.4: profile hot-swap safety and state migration."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.domain.enums import RiskProfile, TradingMode


def _make_app_with_rm(trading_mode: TradingMode = TradingMode.SHADOW) -> object:
    from trader.app import TradingApplication
    from trader.risk.circuit_breakers import CircuitBreakerManager
    from trader.risk.drawdown import DrawdownTracker
    from trader.risk.exposure import ExposureTracker
    from trader.risk.kill_switch import KillSwitch
    from trader.risk.manager import RiskManager
    from trader.risk.profiles import get_risk_limits

    app = TradingApplication()
    settings = MagicMock()
    settings.TRADING_MODE = trading_mode
    settings.LIVE_MODE = trading_mode == TradingMode.LIVE
    settings.RISK_PROFILE = RiskProfile.CONSERVATIVE
    settings.BYBIT_API_KEY.get_secret_value.return_value = ""
    settings.SHADOW_MODE = True
    settings.BYBIT_USE_TESTNET = True
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


class TestProfileHotswapSafety:
    @pytest.mark.asyncio
    async def test_hotswap_blocked_in_live_mode(self):
        """Profile hot-swap raises RuntimeError in LIVE mode."""
        app = _make_app_with_rm(trading_mode=TradingMode.LIVE)
        with pytest.raises(RuntimeError, match="LIVE"):
            await app._change_risk_profile(RiskProfile.MODERATE)

    @pytest.mark.asyncio
    async def test_hotswap_blocked_in_canary_live_mode(self):
        """Profile hot-swap raises RuntimeError in CANARY_LIVE mode."""
        app = _make_app_with_rm(trading_mode=TradingMode.CANARY_LIVE)
        with pytest.raises(RuntimeError):
            await app._change_risk_profile(RiskProfile.MODERATE)

    @pytest.mark.asyncio
    async def test_daily_pnl_preserved_on_profile_swap(self):
        """Daily PnL is not reset when the profile changes."""
        app = _make_app_with_rm()
        app._risk_manager._daily_pnl = Decimal("-5.50")

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(app, "_init_risk_manager", AsyncMock())
            mp.setattr(app, "_refresh_balance", AsyncMock(return_value=Decimal("100")))
            await app._change_risk_profile(RiskProfile.MODERATE)

        assert app._risk_manager._daily_pnl == Decimal("-5.50")

    @pytest.mark.asyncio
    async def test_drawdown_and_daily_pnl_both_preserved(self):
        """Both drawdown AND daily PnL survive a profile swap."""
        app = _make_app_with_rm()
        tracker = app._risk_manager._drawdown
        await tracker.update(Decimal("85"))  # 15% drawdown from $100
        app._risk_manager._daily_pnl = Decimal("-3.00")

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(app, "_init_risk_manager", AsyncMock())
            mp.setattr(app, "_refresh_balance", AsyncMock(return_value=Decimal("100")))
            await app._change_risk_profile(RiskProfile.MODERATE)

        assert app._risk_manager._drawdown is tracker
        assert app._risk_manager._daily_pnl == Decimal("-3.00")


class TestRiskProfileConsistency:
    def test_all_profiles_have_both_config_and_limits(self):
        """Every RiskProfile enum value has both a RiskProfileConfig and RiskLimits entry."""
        from trader.config import get_risk_profile_config
        from trader.domain.enums import RiskProfile
        from trader.risk.profiles import get_risk_limits

        for profile in RiskProfile:
            cfg = get_risk_profile_config(profile)
            limits = get_risk_limits(profile)
            assert cfg is not None, f"Missing RiskProfileConfig for {profile}"
            assert limits is not None, f"Missing RiskLimits for {profile}"

    def test_max_positions_consistent(self):
        """RiskProfileConfig.max_positions and RiskLimits.max_simultaneous_positions
        should be aligned (same or RiskLimits >= RiskProfileConfig)."""
        from trader.config import get_risk_profile_config
        from trader.domain.enums import RiskProfile
        from trader.risk.profiles import get_risk_limits

        for profile in RiskProfile:
            cfg = get_risk_profile_config(profile)
            limits = get_risk_limits(profile)
            assert limits.max_simultaneous_positions >= cfg.max_positions, (
                f"{profile}: RiskLimits.max_positions ({limits.max_simultaneous_positions}) "
                f"< RiskProfileConfig.max_positions ({cfg.max_positions})"
            )
