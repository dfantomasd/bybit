"""Tests for security and trading bug fixes (2026-06-21)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trader.config import Settings, TradingMode
from trader.domain.enums import KillSwitchMode, MarketRegime, MarketType, OrderSide, RiskDecisionStatus, RiskProfile
from trader.domain.models import InstrumentInfo, TradeProposal
from trader.execution.engine import ExecutionEngine
from trader.risk.exposure import ExposureTracker
from trader.risk.manager import RiskManager
from trader.risk.profiles import RISK_PROFILES, get_risk_limits
from trader.telegram_bot import TelegramBotConfig, TelegramMonitorBot


@pytest.mark.asyncio
async def test_telegram_subscriptions_do_not_expand_allowed_chat_ids() -> None:
    controller = MagicMock()
    controller.load_subscriptions = AsyncMock(return_value=[99999])

    config = TelegramBotConfig(
        token="dummy-token",
        allowed_chat_ids={12345},
        trading_mode="SHADOW",
        risk_profile="CONSERVATIVE",
        bybit_use_testnet=True,
    )
    bot = TelegramMonitorBot(
        config=config,
        health_provider=AsyncMock(),
        adapter_factory=MagicMock(),
        controller=controller,
    )

    with patch("trader.telegram_bot.Application") as app_cls:
        app = MagicMock()
        app.initialize = AsyncMock()
        app.start = AsyncMock()
        app.stop = AsyncMock()
        app.shutdown = AsyncMock()
        app.updater = MagicMock()
        app.updater.start_polling = AsyncMock()
        app_cls.builder.return_value.token.return_value.build.return_value = app

        await bot.start()

    assert config.allowed_chat_ids == {12345}
    assert 99999 not in config.allowed_chat_ids
    assert 99999 not in bot._subscribed


def test_pending_gate_is_per_symbol_not_global() -> None:
    adapter = MagicMock()
    risk_manager = MagicMock()
    exposure = MagicMock()
    engine = ExecutionEngine(
        adapter=adapter,
        risk_manager=risk_manager,
        exposure_tracker=exposure,
        shadow_mode=True,
    )
    engine.mark_entry_submitted("oid-btc", symbol="BTCUSDT")

    assert engine.has_pending_order_for_symbol("BTCUSDT") is True
    assert engine.has_pending_order_for_symbol("ETHUSDT") is False
    assert engine.has_pending_entries() is True


@pytest.mark.asyncio
async def test_cancel_all_open_orders_cancels_and_clears_pending() -> None:
    adapter = MagicMock()
    adapter.get_open_orders = AsyncMock(
        return_value=[
            {"symbol": "BTCUSDT", "orderLinkId": "link-1"},
            {"symbol": "ETHUSDT", "orderLinkId": "link-2"},
        ]
    )
    adapter.cancel_order = AsyncMock(return_value={})

    engine = ExecutionEngine(
        adapter=adapter,
        risk_manager=MagicMock(),
        exposure_tracker=MagicMock(),
        shadow_mode=False,
    )
    engine.mark_entry_submitted("link-1", symbol="BTCUSDT")
    engine.resolve_pending_durable = AsyncMock()

    cancelled = await engine.cancel_all_open_orders()

    assert cancelled == 2
    assert adapter.cancel_order.await_count == 2
    assert engine.resolve_pending_durable.await_count == 2


@pytest.mark.asyncio
async def test_emergency_stop_cancels_open_orders() -> None:
    from trader.app import TradingApplication

    app = TradingApplication()
    app._telegram_bot = None
    app._kill_switch = MagicMock()
    app._kill_switch.activate = AsyncMock()
    app._execution_engine = MagicMock()
    app._execution_engine.cancel_all_open_orders = AsyncMock(return_value=3)

    await app._emergency_stop()

    app._execution_engine.cancel_all_open_orders.assert_awaited_once()
    assert app._trading_paused is True


def test_execution_max_open_positions_uses_profile_when_legacy_default() -> None:
    from trader.app import TradingApplication

    app = TradingApplication()
    app._settings = MagicMock()
    app._settings.MAX_POSITIONS = 2
    app._settings.RISK_PROFILE = RiskProfile.SCALP

    assert app._execution_max_open_positions() == RISK_PROFILES[RiskProfile.SCALP].max_simultaneous_positions


def test_execution_max_open_positions_respects_explicit_lower_cap() -> None:
    from trader.app import TradingApplication

    app = TradingApplication()
    app._settings = MagicMock()
    app._settings.MAX_POSITIONS = 4
    app._settings.RISK_PROFILE = RiskProfile.SCALP

    assert app._execution_max_open_positions() == 4


@pytest.mark.asyncio
async def test_risk_monitor_resets_daily_stats_at_utc_midnight() -> None:
    from trader.app import TradingApplication

    app = TradingApplication()
    app._shutdown_event.clear()
    app._settings = MagicMock()
    app._settings.BYBIT_API_KEY.get_secret_value.return_value = ""
    app._kill_switch = MagicMock()
    app._kill_switch.check_file_flag = AsyncMock()
    app._kill_switch.is_active = False
    app._risk_manager = MagicMock()
    app._risk_manager.reset_daily_stats = AsyncMock()
    app._risk_manager.daily_pnl = Decimal("0")
    app._last_daily_reset_date = date(2026, 6, 20)

    async def stop_after_first_wait(*args, **kwargs):
        app._shutdown_event.set()
        raise TimeoutError

    with (
        patch("asyncio.wait_for", side_effect=stop_after_first_wait),
        patch("trader.app.datetime") as dt_mock,
    ):
        dt_mock.now.return_value = datetime(2026, 6, 21, 0, 5, tzinfo=UTC)
        await app._run_risk_monitor()

    app._risk_manager.reset_daily_stats.assert_awaited_once()
    assert app._last_daily_reset_date == date(2026, 6, 21)


def test_live_mode_requires_model_encrypt_key_when_ml_live_decisions_enabled() -> None:
    with pytest.raises(ValueError, match="MODEL_ENCRYPT_KEY"):
        Settings(
            _env_file=None,
            TRADING_MODE=TradingMode.LIVE,
            LIVE_MODE=True,
            LIVE_ARMED=True,
            BYBIT_USE_TESTNET=False,
            MODEL_ALLOW_LIVE_DECISIONS=True,
            MODEL_ENCRYPT_KEY="",
        )


@pytest.mark.asyncio
async def test_correlation_adjustment_reduces_approved_qty() -> None:
    limits = get_risk_limits(RiskProfile.CONSERVATIVE)
    exposure = ExposureTracker(total_capital=Decimal("10000"), risk_limits=limits)
    await exposure.update_position("BTCUSDT", "Buy", Decimal("1000"), leverage=Decimal("1"))

    drawdown = MagicMock()
    drawdown.drawdown_pct = Decimal("0")
    drawdown.is_at_hard_stop = MagicMock(return_value=False)

    breakers = MagicMock()
    breakers.should_emergency = MagicMock(return_value=False)
    breakers.should_block_entries = MagicMock(return_value=False)
    breakers.should_safe_mode = MagicMock(return_value=False)

    kill_switch = MagicMock()
    kill_switch.is_active = False

    manager = RiskManager(
        risk_profile=RiskProfile.CONSERVATIVE,
        drawdown_tracker=drawdown,
        exposure_tracker=exposure,
        circuit_breaker_manager=breakers,
        kill_switch=kill_switch,
    )
    instrument = InstrumentInfo(
        symbol="WBTCUSDT",
        market_type=MarketType.LINEAR,
        base_coin="WBTC",
        quote_coin="USDT",
        min_order_qty=Decimal("0.001"),
        max_order_qty=Decimal("100"),
        qty_step=Decimal("0.001"),
        tick_size=Decimal("0.5"),
        min_notional=Decimal("5"),
    )
    proposal = TradeProposal(
        strategy_id="test",
        symbol="WBTCUSDT",
        market_type=MarketType.LINEAR,
        side=OrderSide.BUY,
        requested_qty=Decimal("0.02"),
        entry_price=Decimal("50000"),
        stop_loss=Decimal("49000"),
        take_profit=Decimal("52000"),
        confidence=0.8,
        regime=MarketRegime.BULL_TREND,
    )

    decision = await manager.evaluate(
        proposal,
        instrument_info=instrument,
        available_balance=Decimal("10000"),
        capital=Decimal("10000"),
    )

    assert decision.status == RiskDecisionStatus.APPROVED
    assert decision.approved_qty is not None
    assert "correlation_reduced" in decision.triggered_rules


@pytest.mark.asyncio
async def test_kill_switch_file_flag_sets_full_stop(tmp_path) -> None:
    from trader.risk.kill_switch import KillSwitch

    flag = tmp_path / "kill.flag"
    flag.write_text("manual stop", encoding="utf-8")
    ks = KillSwitch(flag_file=flag)
    await ks.check_file_flag()
    assert ks.is_active
    assert ks.current_mode == KillSwitchMode.FULL_STOP
