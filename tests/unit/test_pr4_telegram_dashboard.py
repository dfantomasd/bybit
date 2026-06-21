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
        symbol_candidates=lambda: ["BTCUSDT", "ETHUSDT", "DOGEUSDT"],
        selected_symbols=lambda: ["ETHUSDT"],
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


def _fake_callback_update(chat_id: int = 12345) -> MagicMock:
    update = _fake_update(chat_id=chat_id)
    query = MagicMock()
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    query.message = MagicMock()
    query.message.reply_text = AsyncMock()
    query.message.message_id = 1
    query.message.chat_id = chat_id
    update.callback_query = query
    return update


def _fake_text_update(text: str, chat_id: int = 12345) -> MagicMock:
    update = _fake_update(chat_id=chat_id)
    update.effective_message.text = text
    update.message.text = text
    return update


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
async def test_render_pnl_uses_provider_net_without_double_subtracting_fees() -> None:
    bot = _make_bot()
    bot._net_results_provider = AsyncMock(
        return_value={
            "gross_closed_pnl_usd": -1.0,
            "total_fees_usd": -0.25,
            "net_pnl_usd": -1.0,
        }
    )

    text, _ = await bot._render_pnl()

    assert "Net PnL: <code>-1.00 USD</code>" in text
    assert "-1.25 USD" not in text


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
    """Main menu should include the model section."""
    bot = _make_bot()
    markup = bot._main_menu()
    all_texts = [btn.text for row in markup.inline_keyboard for btn in row]
    assert any("Модель" in t or "🧠" in t for t in all_texts), f"No DB/model button found in menu: {all_texts}"


def test_main_menu_has_symbol_selection_button() -> None:
    bot = _make_bot()
    markup = bot._main_menu()
    all_texts = [btn.text for row in markup.inline_keyboard for btn in row]
    assert any("Настройки" in text for text in all_texts)


def test_main_menu_is_grouped_and_has_mode_indicator_text() -> None:
    bot = _make_bot()
    markup = bot._main_menu()
    all_texts = [btn.text for row in markup.inline_keyboard for btn in row]
    all_callbacks = [btn.callback_data for row in markup.inline_keyboard for btn in row]

    assert "🏠 Статус и режим" in all_texts
    assert "💰 Баланс и позиции" in all_texts
    assert "📈 Торговля и сигналы" in all_texts
    assert "⚙️ Настройки" in all_texts
    assert "🧠 Модель и обучение" in all_texts
    assert "🩺 Диагностика" in all_texts
    assert "❓ Помощь" in all_texts
    assert "view:menu" in all_callbacks
    assert "SHADOW" in bot._menu_text()


def test_symbol_select_menu_marks_selected_symbols() -> None:
    bot = _make_bot()
    markup = bot._symbol_select_menu()
    all_texts = [btn.text for row in markup.inline_keyboard for btn in row]
    assert "☐ BTCUSDT" in all_texts
    assert "✅ ETHUSDT" in all_texts


@pytest.mark.asyncio
async def test_symbol_toggle_button_calls_controller() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.toggle_symbol = AsyncMock(return_value="✅ <code>DOGEUSDT</code> добавлена")
    update = _fake_update()

    await bot._handle_symbol_button(update, "toggle:DOGEUSDT:0")

    bot._controller.toggle_symbol.assert_awaited_once_with("DOGEUSDT")
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "DOGEUSDT" in reply_text


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
    assert "view:model_help" in all_callbacks


@pytest.mark.asyncio
async def test_control_train_button_requires_confirm() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.start_training = AsyncMock(return_value="started")
    update = _fake_update()

    await bot._handle_control_button(update, "train")

    bot._controller.start_training.assert_not_awaited()
    reply_markup = update.effective_message.reply_text.call_args.kwargs["reply_markup"]
    callbacks = [btn.callback_data for row in reply_markup.inline_keyboard for btn in row]
    _confirm_callback(callbacks, "train:500:15:5")


def _confirm_callback(callbacks: list[str], action: str) -> str:
    """Return the confirm:<nonce>:<action>:yes callback for *action*."""
    suffix = f":{action}:yes"
    matches = [c for c in callbacks if c.startswith("confirm:") and c.endswith(suffix)]
    assert matches, f"no confirm button for {action!r} in {callbacks}"
    return matches[0]


@pytest.mark.asyncio
async def test_confirm_train_button_starts_training() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.start_training = AsyncMock(return_value="✅ started")
    update = _fake_update()

    markup = bot._confirm_menu("train:500:15:5")
    payload = markup.inline_keyboard[0][0].callback_data.removeprefix("confirm:")
    await bot._handle_confirm_button(update, payload)

    bot._controller.start_training.assert_awaited_once_with(500, 15, 5.0)
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "started" in reply_text


def test_submenus_have_home_button() -> None:
    bot = _make_bot()
    for markup in [bot._control_menu(), bot._canary_menu(), bot._limits_menu(), bot._symbol_select_menu()]:
        all_callbacks = [btn.callback_data for row in markup.inline_keyboard for btn in row]
        assert "view:menu" in all_callbacks


def test_model_help_is_russian_operator_text() -> None:
    bot = _make_bot()
    text = bot._model_help_text()
    assert "Что означают метрики" in text
    assert "Walk-forward" in text
    assert "CANARY" in text
    assert "Paper" in text


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
            "last_confirmed_candle_age_s": 30,
            "last_strategy_loop_at": "2026-06-07T10:00:00Z",
            "hour_api_rejected": 0,
            "hour_min_notional_rejected": 0,
            "model": {"champion_version": "v1"},
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
            "latest_model_version": {
                "version": "v1",
                "status": "CHAMPION",
                "metrics": {"quality": "GOOD", "walk_forward_expectancy_bps": 2.5},
            },
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
    assert "Готовность к реальным деньгам" in reply_text
    assert "ГОТОВО" in reply_text
    assert "НЕ ГОТОВО" not in reply_text


@pytest.mark.asyncio
async def test_canary_readiness_blocks_weak_unprofitable_model() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.runtime_settings = MagicMock(return_value={"model_gate_canary_enabled": False})
    bot._controller.diagnostics_provider = MagicMock(
        return_value={
            "active_symbols": ["ETHUSDT", "XRPUSDT", "DOGEUSDT"],
            "last_ws_message_age_s": 0,
            "last_confirmed_candle_age_s": 4,
            "last_strategy_loop_at": "2026-06-12T10:00:00Z",
            "hour_api_rejected": 0,
            "hour_min_notional_rejected": 0,
            "model": {"champion_version": "none"},
        }
    )
    bot._controller.db_diagnostics_provider = AsyncMock(
        return_value={
            "connected": True,
            "latest_candle_1m": datetime.now(UTC) - timedelta(seconds=65),
            "candles_by_interval": {"1": 40936, "5": 12974, "15": 11533, "60": 1584},
            "feature_snapshots": 4978,
            "prediction_outcomes": 37517,
            "training_eligible_15m": 3378,
            "latest_training_run": {"status": "COMPLETED"},
            "latest_model_version": {
                "version": "v20260612_0934_h15m_dnv1",
                "status": "SHADOW_CHALLENGER",
                "metrics": {"quality": "WEAK", "walk_forward_expectancy_bps": -46.91},
            },
            "shadow_gate_15m": {"total_count": 111, "lift_vs_all_bps": 6.98},
            "paper_pnl_15m": {
                "baseline": {"count": 1000, "total_bps": -47434.6},
                "model_gate": {"count": 6, "total_bps": -81.2},
            },
        }
    )
    update = _fake_update()
    ctx = type("_Ctx", (), {"args": []})()

    await bot._cmd_canary_ready(update, ctx)  # type: ignore[arg-type]

    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "НЕ ГОТОВО" in reply_text
    assert "ПОЧТИ" not in reply_text
    assert "Качество модели GOOD" in reply_text
    assert "Walk-forward модели > 0 bps" in reply_text
    assert "Paper model-gate: 20+ сделок и PnL > 0" in reply_text


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

    reply_text = "\n".join(call.args[0] for call in update.effective_message.reply_text.await_args_list)
    assert "Готовность к реальным деньгам" in reply_text
    assert "НЕ ГОТОВО" in reply_text


def test_limits_menu_has_common_presets() -> None:
    """Limits submenu should expose common small-account presets."""
    bot = _make_bot()
    markup = bot._limits_menu()
    all_callbacks = [btn.callback_data for row in markup.inline_keyboard for btn in row]
    for expected in [
        "limit:entries:1",
        "limit:pending:1",
        "limit:same_side:1",
        "limit:custom:max_positions",
        "limit:price_cap:25",
        "limit:feature_symbols:20",
        "limit:custom:feature_symbols",
        "limit:exec_candidates:10",
    ]:
        assert expected in all_callbacks


@pytest.mark.asyncio
async def test_train_command_requires_confirm() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.start_training = AsyncMock(return_value="training started")
    update = _fake_update()
    ctx = type("_Ctx", (), {"args": ["500", "15", "5"]})()

    await bot._cmd_train(update, ctx)  # type: ignore[arg-type]

    bot._controller.start_training.assert_not_awaited()
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "Запустить обучение" in reply_text
    reply_markup = update.effective_message.reply_text.call_args.kwargs["reply_markup"]
    callbacks = [btn.callback_data for row in reply_markup.inline_keyboard for btn in row]
    _confirm_callback(callbacks, "train:500:15:5")


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
async def test_settings_plus_button_updates_runtime_setting() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.runtime_settings = MagicMock(
        return_value={
            "max_entries_per_minute": 1,
            "max_positions": 2,
            "max_same_side": 1,
            "screener_max_price_usd": 25,
            "feature_max_symbols": 20,
        }
    )
    bot._controller.set_runtime_setting = AsyncMock(return_value="Max entries/min set to 2")
    update = _fake_callback_update()

    await bot._handle_limit_button(update, "entries_per_min_limit:inc")

    bot._controller.set_runtime_setting.assert_awaited_once_with("entries", 2)
    update.callback_query.edit_message_text.assert_awaited()


@pytest.mark.asyncio
async def test_long_reply_is_split_below_telegram_limit() -> None:
    bot = _make_bot()
    update = _fake_update()

    await bot._reply(update, "x" * 8000)

    assert update.effective_message.reply_text.await_count >= 2
    for call in update.effective_message.reply_text.await_args_list:
        assert len(call.args[0]) <= 4000


@pytest.mark.asyncio
async def test_custom_feature_symbols_button_accepts_plain_number() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.runtime_settings = MagicMock(return_value={})
    bot._controller.set_runtime_setting = AsyncMock(return_value="Feature symbols set to 12")
    update = _fake_update()

    await bot._handle_limit_button(update, "custom:feature_symbols")
    await bot._on_text(_fake_text_update("12"), MagicMock())

    bot._controller.set_runtime_setting.assert_awaited_once_with("feature_symbols", 12)


@pytest.mark.asyncio
async def test_custom_max_positions_button_accepts_plain_number() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.runtime_settings = MagicMock(return_value={})
    bot._controller.set_runtime_setting = AsyncMock(return_value="Max simultaneous positions set to 3")
    update = _fake_update()

    await bot._handle_limit_button(update, "custom:max_positions")
    await bot._on_text(_fake_text_update("3"), MagicMock())

    bot._controller.set_runtime_setting.assert_awaited_once_with("max_positions", 3)


@pytest.mark.asyncio
async def test_train_button_requires_confirm() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.start_training = AsyncMock(return_value="training started")
    update = _fake_update()

    await bot._handle_train_button(update, "1000:15:5")

    bot._controller.start_training.assert_not_awaited()
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "Запустить обучение" in reply_text
    reply_markup = update.effective_message.reply_text.call_args.kwargs["reply_markup"]
    callbacks = [btn.callback_data for row in reply_markup.inline_keyboard for btn in row]
    _confirm_callback(callbacks, "train:1000:15:5")


@pytest.mark.asyncio
async def test_stop_confirm_button_runs_emergency_stop() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    update = _fake_update()

    markup = bot._confirm_menu("stop")
    payload = markup.inline_keyboard[0][0].callback_data.removeprefix("confirm:")
    await bot._handle_confirm_button(update, payload)

    bot._controller.emergency_stop.assert_awaited_once()
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "Аварийная остановка выполнена" in reply_text


@pytest.mark.asyncio
async def test_menu_command_shows_main_menu() -> None:
    bot = _make_bot()
    update = _fake_update()
    ctx = type("_Ctx", (), {"args": []})()

    await bot._cmd_menu(update, ctx)  # type: ignore[arg-type]

    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "Bybit AI Trader" in reply_text
    assert "SHADOW" in reply_text


@pytest.mark.asyncio
async def test_callback_answer_failure_still_handles_button() -> None:
    bot = _make_bot()
    update = _fake_callback_update()
    update.callback_query.data = "view:menu"
    update.callback_query.answer = AsyncMock(side_effect=RuntimeError("query is too old"))

    await bot._on_button(update, MagicMock())  # type: ignore[arg-type]

    update.callback_query.edit_message_text.assert_awaited()
    edited_text = update.callback_query.edit_message_text.await_args.args[0]
    assert "Bybit AI Trader" in edited_text


@pytest.mark.asyncio
async def test_dashboard_action_pause_button_is_routed() -> None:
    bot = _make_bot()
    update = _fake_callback_update()
    update.callback_query.data = "action:pause"

    await bot._on_button(update, MagicMock())  # type: ignore[arg-type]

    assert bot._controller is not None
    bot._controller.pause.assert_awaited_once()
    update.callback_query.edit_message_text.assert_awaited()
    edited_text = update.callback_query.edit_message_text.await_args.args[0]
    assert "Bybit AI Trader" in edited_text


@pytest.mark.asyncio
async def test_dashboard_action_canary_button_is_routed() -> None:
    bot = _make_bot()
    bot._db_diagnostics_provider = AsyncMock(return_value={"connected": True})
    bot._diagnostics_provider = MagicMock(return_value={"active_symbols": ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]})
    update = _fake_callback_update()
    update.callback_query.data = "action:canary"

    await bot._on_button(update, MagicMock())  # type: ignore[arg-type]

    update.callback_query.edit_message_text.assert_awaited()
    edited_text = update.callback_query.edit_message_text.await_args.args[0]
    assert "Готовность к реальным деньгам" in edited_text


@pytest.mark.asyncio
async def test_callback_exception_returns_visible_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = _make_bot()
    update = _fake_callback_update()
    update.callback_query.data = "view:model"

    async def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("db timeout")

    monkeypatch.setattr(bot, "_handle_view_button", _boom)

    await bot._on_button(update, MagicMock())  # type: ignore[arg-type]

    update.callback_query.edit_message_text.assert_awaited()
    edited_text = update.callback_query.edit_message_text.await_args.args[0]
    assert "Кнопка не выполнилась" in edited_text


@pytest.mark.asyncio
async def test_button_reply_falls_back_to_new_message_when_edit_unavailable() -> None:
    bot = _make_bot()
    update = _fake_callback_update()
    update.callback_query.edit_message_text = AsyncMock(side_effect=RuntimeError("edit timeout"))

    await bot._button_reply(update, "✅ Готово", reply_markup=bot._main_menu())

    update.callback_query.message.reply_text.assert_awaited()
    reply_text = update.callback_query.message.reply_text.await_args.args[0]
    assert "Готово" in reply_text


@pytest.mark.asyncio
async def test_button_reply_uses_direct_send_when_callback_message_unavailable() -> None:
    bot = _make_bot()
    update = _fake_callback_update()
    update.callback_query.edit_message_text = AsyncMock(side_effect=RuntimeError("edit timeout"))
    update.callback_query.message.reply_text = AsyncMock(side_effect=RuntimeError("reply timeout"))
    sent = MagicMock(message_id=99, chat_id=12345)
    bot._app = MagicMock()
    bot._app.bot.send_message = AsyncMock(return_value=sent)

    await bot._button_reply(update, "✅ Готово", reply_markup=bot._main_menu())

    bot._app.bot.send_message.assert_awaited()
    kwargs = bot._app.bot.send_message.await_args.kwargs
    assert kwargs["chat_id"] == 12345
    assert "Готово" in kwargs["text"]


@pytest.mark.asyncio
async def test_callback_auth_uses_message_chat_id_when_effective_chat_missing() -> None:
    bot = _make_bot()
    update = _fake_callback_update()
    update.effective_chat = None
    update.callback_query.data = "view:menu"

    await bot._on_button(update, MagicMock())  # type: ignore[arg-type]

    update.callback_query.edit_message_text.assert_awaited()
    edited_text = update.callback_query.edit_message_text.await_args.args[0]
    assert "Bybit AI Trader" in edited_text


@pytest.mark.asyncio
async def test_confirm_button_replay_is_rejected() -> None:
    """A confirm button fires once; the second press must be ignored."""
    bot = _make_bot()
    assert bot._controller is not None
    update = _fake_update()

    markup = bot._confirm_menu("stop")
    payload = markup.inline_keyboard[0][0].callback_data.removeprefix("confirm:")
    await bot._handle_confirm_button(update, payload)
    bot._controller.emergency_stop.assert_awaited_once()

    replay = _fake_update()
    await bot._handle_confirm_button(replay, payload)
    bot._controller.emergency_stop.assert_awaited_once()  # still once
    reply_text = replay.effective_message.reply_text.call_args[0][0]
    assert "устарела" in reply_text


@pytest.mark.asyncio
async def test_legacy_confirm_payload_without_nonce_is_rejected() -> None:
    """Pre-nonce buttons from old chat history must not execute anything."""
    bot = _make_bot()
    assert bot._controller is not None
    update = _fake_update()

    await bot._handle_confirm_button(update, "stop:yes")

    bot._controller.emergency_stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_expired_confirm_nonce_is_rejected() -> None:
    from datetime import timedelta

    bot = _make_bot()
    assert bot._controller is not None
    update = _fake_update()

    markup = bot._confirm_menu("stop")
    payload = markup.inline_keyboard[0][0].callback_data.removeprefix("confirm:")
    nonce = payload.split(":", 1)[0]
    action, created_at = bot._confirm_nonces[nonce]
    bot._confirm_nonces[nonce] = (action, created_at - timedelta(seconds=301))

    await bot._handle_confirm_button(update, payload)

    bot._controller.emergency_stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirm_no_invalidates_nonce() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    update = _fake_update()

    markup = bot._confirm_menu("stop")
    no_payload = markup.inline_keyboard[0][1].callback_data.removeprefix("confirm:")
    yes_payload = markup.inline_keyboard[0][0].callback_data.removeprefix("confirm:")
    await bot._handle_confirm_button(update, no_payload)

    await bot._handle_confirm_button(_fake_update(), yes_payload)
    bot._controller.emergency_stop.assert_not_awaited()


def test_split_message_keeps_chunks_below_telegram_limit() -> None:
    text = "<b>header</b>\n" + ("x" * 9100)
    chunks = TelegramMonitorBot._split_message(text)

    assert len(chunks) >= 3
    assert all(len(chunk) <= 4000 for chunk in chunks)


def test_plain_text_fallback_strips_html_tags() -> None:
    text = "<b>Модель</b> &amp; <code>v1</code>"

    assert TelegramMonitorBot._plain_text(text) == "Модель & v1"


def test_telegram_health_snapshot_records_polling_conflicts() -> None:
    from telegram.error import Conflict

    bot = _make_bot()

    bot._polling_error_callback(Conflict("terminated by other getUpdates request"))

    health = bot.health_snapshot()
    assert health["enabled"] is True
    assert health["polling_conflict_count"] == 1
    assert "getUpdates" in health["last_polling_error"]


@pytest.mark.asyncio
async def test_db_model_screen_uses_model_gate_horizon() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.diagnostics_provider = MagicMock(
        return_value={
            "model": {
                "champion_version": "none",
                "challenger_version": "v5",
                "last_training": "never",
                "training_samples": 1200,
            }
        }
    )
    bot._controller.db_diagnostics_provider = AsyncMock(
        return_value={
            "connected": True,
            "configured": True,
            "candles_by_interval": {"1": 5000, "5": 1000, "15": 0, "60": 100},
            "prediction_outcomes_by_horizon": {"5": 1500, "15": 0},
            "training_eligible_by_horizon": {"5": 1234, "15": 0},
            "latest_training_run": {"status": "COMPLETED", "sample_count": 1234, "model_version": "v5"},
            "latest_model_version": {
                "version": "v5",
                "status": "SHADOW_CHALLENGER",
                "training_samples": 1234,
                "metrics": {"quality": "GOOD", "horizon_minutes": 5, "lift_bps": 1.2},
            },
            "model_gate_horizon_minutes": 5,
            "shadow_gate_by_horizon": {
                "5": {
                    "horizon_minutes": 5,
                    "total_count": 44,
                    "pass_count": 12,
                    "block_count": 32,
                    "lift_vs_all_bps": 2.5,
                }
            },
            "paper_pnl_by_horizon": {
                "5": {
                    "baseline": {"count": 20, "total_bps": -1.0},
                    "model_gate": {"count": 12, "total_bps": 8.0},
                }
            },
        }
    )
    update = _fake_update()
    ctx = type("_Ctx", (), {"args": []})()

    await bot._cmd_db_model(update, ctx)  # type: ignore[arg-type]

    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "Готово для обучения (5m)" in reply_text
    assert "Фильтр модели 5m" in reply_text
    assert "+2.50 bps" in reply_text
    assert "Фильтр модели 15m" not in reply_text
