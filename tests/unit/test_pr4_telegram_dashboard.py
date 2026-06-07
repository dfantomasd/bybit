"""PR 4 Telegram dashboard tests.

Covers:
- test_shadow_off_blocked (Telegram cannot disable shadow)
- test_live_activation_blocked (mode:active blocked)
- test_control_menu_has_no_dangerous_controls
- test_main_menu_structure (expected buttons present)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.telegram_bot import TelegramBotConfig, TelegramMonitorBot, TradingController


def _make_bot() -> TelegramMonitorBot:
    controller = TradingController(
        pause=AsyncMock(),
        resume=AsyncMock(),
        set_shadow=AsyncMock(),
        set_risk_profile=AsyncMock(),
        emergency_stop=AsyncMock(),
        is_paused=lambda: False,
        is_shadow=lambda: True,
        current_profile=lambda: "CONSERVATIVE",
        active_symbols=lambda: [],
        regime_for=lambda s: None,
        diagnostics_provider=lambda: {},
        db_diagnostics_provider=None,
    )

    config = TelegramBotConfig(
        token="fake:TOKEN",
        allowed_chat_ids={12345},
        trading_mode="SHADOW",
        risk_profile="CONSERVATIVE",
        bybit_use_testnet=True,
    )

    return TelegramMonitorBot(
        config=config,
        health_provider=AsyncMock(return_value=MagicMock(ok=True)),
        adapter_factory=lambda: None,
        controller=controller,
    )


def _fake_update(chat_id: int = 12345) -> MagicMock:
    u = MagicMock()
    u.effective_chat = MagicMock()
    u.effective_chat.id = chat_id
    u.callback_query = None
    # Use effective_message (what _reply() uses) with AsyncMock
    fake_msg = MagicMock()
    fake_msg.reply_text = AsyncMock()
    u.effective_message = fake_msg
    u.message = fake_msg
    return u


def _fake_context() -> MagicMock:
    return type("_Ctx", (), {"args": ["off"]})()  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_shadow_off_blocked() -> None:
    """Telegram cannot turn shadow mode OFF."""
    bot = _make_bot()
    update = _fake_update()
    ctx = _fake_context()

    await bot._cmd_shadow(update, ctx)  # type: ignore[arg-type]

    # set_shadow should NOT be called with False
    for call in bot._controller.set_shadow.call_args_list:  # type: ignore[union-attr]
        assert call.args[0] is not False, "Shadow OFF was allowed via Telegram"

    # Reply should contain "blocked" text
    assert update.effective_message.reply_text.called
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "blocked" in reply_text.lower() or "CANARY" in reply_text


@pytest.mark.asyncio
async def test_live_activation_blocked_via_mode_button() -> None:
    """mode:active button should return blocked message."""
    bot = _make_bot()
    update = _fake_update()

    await bot._handle_mode_button(update, "active")

    assert update.effective_message.reply_text.called
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "blocked" in reply_text.lower() or "CANARY" in reply_text


@pytest.mark.asyncio
async def test_live_activation_blocked_via_cmd_mode() -> None:
    """/mode active command should return blocked message."""
    bot = _make_bot()
    update = _fake_update()
    ctx = type("_Ctx", (), {"args": ["active"]})()

    await bot._cmd_mode(update, ctx)  # type: ignore[arg-type]

    assert update.effective_message.reply_text.called
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "blocked" in reply_text.lower() or "CANARY" in reply_text


def test_main_menu_has_db_model_button() -> None:
    """Main menu should include 🗄 База и модель button."""
    bot = _make_bot()
    markup = bot._main_menu()
    all_texts = [btn.text for row in markup.inline_keyboard for btn in row]
    assert any("База" in t or "модель" in t or "🗄" in t for t in all_texts), (
        f"No DB/model button found in menu: {all_texts}"
    )


def test_control_menu_no_active_button() -> None:
    """Control menu should NOT have a dangerous Active enable button."""
    bot = _make_bot()
    markup = bot._control_menu()
    active_buttons = [
        btn for row in markup.inline_keyboard for btn in row if "active" in (btn.callback_data or "").lower()
    ]
    assert len(active_buttons) == 1
    assert "заблокирован" in active_buttons[0].text.lower() or "blocked" in active_buttons[0].text.lower()


def test_control_menu_no_risk_escalation() -> None:
    """Control menu should NOT have risk escalation buttons."""
    bot = _make_bot()
    markup = bot._control_menu()
    all_callbacks = [btn.callback_data for row in markup.inline_keyboard for btn in row]
    risk_buttons = [c for c in all_callbacks if c and c.startswith("risk:")]
    assert not risk_buttons, f"Risk escalation buttons found in control menu: {risk_buttons}"


def test_control_menu_has_training_and_limits() -> None:
    """Control menu should expose safe training and limit controls."""
    bot = _make_bot()
    markup = bot._control_menu()
    all_callbacks = [btn.callback_data for row in markup.inline_keyboard for btn in row]
    assert "train:500:15:5" in all_callbacks
    assert "train:1000:15:5" in all_callbacks
    assert "control:limits" in all_callbacks
    assert "view:db_model" in all_callbacks
    assert "view:canary" in all_callbacks


@pytest.mark.asyncio
async def test_canary_readiness_reports_ready_with_good_inputs() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.runtime_settings = MagicMock(
        return_value={
            "model_gate_canary_enabled": False,
            "model_gate_quality": {"quality": "GOOD", "sample_count": 150},
        }
    )
    bot._controller.diagnostics_provider = MagicMock(
        return_value={
            "active_symbols": ["ETHUSDT", "XRPUSDT", "DOGEUSDT"],
            "last_ws_message_age_s": 10,
            "last_strategy_loop_at": "2026-06-07T10:00:00Z",
            "hour_api_rejected": 0,
            "hour_min_notional_rejected": 0,
        }
    )
    bot._controller.db_diagnostics_provider = AsyncMock(
        return_value={
            "connected": True,
            "latest_candle_1m": datetime.now(UTC) - timedelta(seconds=30),
            "candles_by_interval": {"1": 5000, "5": 1000, "15": 500, "60": 200},
            "feature_snapshots": 3000,
            "prediction_outcomes": 3000,
            "labelled_samples_15m": 2500,
            "latest_training_run": {"status": "COMPLETED"},
            "latest_model_version": {"version": "v1"},
            "shadow_gate_15m": {"total_count": 120, "lift_vs_all_bps": 1.5},
            "paper_pnl_15m": {
                "baseline": {"count": 50, "total_bps": -2.0},
                "model_gate": {"count": 40, "total_bps": 10.0},
            },
        }
    )
    update = _fake_update()
    ctx = type("_Ctx", (), {"args": []})()

    await bot._cmd_canary_ready(update, ctx)  # type: ignore[arg-type]

    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "CANARY readiness" in reply_text
    assert "READY" in reply_text
    assert "NOT READY" not in reply_text


@pytest.mark.asyncio
async def test_canary_readiness_reports_not_ready_without_data() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.runtime_settings = MagicMock(return_value={})
    bot._controller.diagnostics_provider = MagicMock(return_value={"active_symbols": []})
    bot._controller.db_diagnostics_provider = AsyncMock(return_value={"connected": False})
    update = _fake_update()
    ctx = type("_Ctx", (), {"args": []})()

    await bot._cmd_canary_ready(update, ctx)  # type: ignore[arg-type]

    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "CANARY readiness" in reply_text
    assert "NOT READY" in reply_text


def test_limits_menu_has_common_presets() -> None:
    """Limits submenu should expose common small-account presets."""
    bot = _make_bot()
    markup = bot._limits_menu()
    all_callbacks = [btn.callback_data for row in markup.inline_keyboard for btn in row]
    for expected in [
        "limit:entries:1",
        "limit:pending:1",
        "limit:same_side:1",
        "limit:price_cap:25",
        "limit:feature_symbols:20",
        "limit:exec_candidates:10",
    ]:
        assert expected in all_callbacks


@pytest.mark.asyncio
async def test_train_command_starts_shadow_training() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.start_training = AsyncMock(return_value="training started")
    update = _fake_update()
    ctx = type("_Ctx", (), {"args": ["500", "15", "5"]})()

    await bot._cmd_train(update, ctx)  # type: ignore[arg-type]

    bot._controller.start_training.assert_awaited_once_with(500, 15, 5.0)
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "training started" in reply_text


@pytest.mark.asyncio
async def test_limits_command_updates_safe_runtime_setting() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.runtime_settings = MagicMock(
        return_value={
            "paused": False,
            "shadow": True,
            "risk_profile": "CONSERVATIVE",
            "max_entries_per_minute": 1,
            "max_concurrent_pending": 1,
            "max_same_side": 1,
            "screener_max_price_usd": 25,
            "feature_max_symbols": 20,
            "execution_candidates": 10,
        }
    )
    bot._controller.set_runtime_setting = AsyncMock(return_value="Max entries/min set to 2")
    update = _fake_update()
    ctx = type("_Ctx", (), {"args": ["entries", "2"]})()

    await bot._cmd_limits(update, ctx)  # type: ignore[arg-type]

    bot._controller.set_runtime_setting.assert_awaited_once_with("entries", 2)
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "Max entries/min set to 2" in reply_text


@pytest.mark.asyncio
async def test_limit_button_updates_runtime_setting() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.runtime_settings = MagicMock(return_value={})
    bot._controller.set_runtime_setting = AsyncMock(return_value="Screener price cap set to 25")
    update = _fake_update()

    await bot._handle_limit_button(update, "price_cap:25")

    bot._controller.set_runtime_setting.assert_awaited_once_with("price_cap", 25.0)
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "Screener price cap set to 25" in reply_text


@pytest.mark.asyncio
async def test_train_button_starts_shadow_training() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.start_training = AsyncMock(return_value="training started")
    update = _fake_update()

    await bot._handle_train_button(update, "1000:15:5")

    bot._controller.start_training.assert_awaited_once_with(1000, 15, 5.0)
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "training started" in reply_text
