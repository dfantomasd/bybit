"""PR 4 Telegram dashboard tests.

Covers:
- test_shadow_off_blocked (Telegram cannot disable shadow)
- test_live_activation_blocked (mode:active blocked)
- test_control_menu_has_no_dangerous_controls
- test_main_menu_structure (expected buttons present)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

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
        runtime_settings=lambda: {
            "model_auto_train_min_samples": 1000,
            "model_auto_train_horizon_minutes": 5,
            "model_auto_train_label_bps": 2.0,
            "label_schema_version": "directional_net_v2",
        },
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


@pytest.mark.asyncio
async def test_persisted_subscriptions_do_not_expand_allowed_chat_ids() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.load_subscriptions = AsyncMock(return_value=[12345, 99999])
    bot._config.allowed_chat_ids = {12345}

    app = MagicMock()
    app.initialize = AsyncMock()
    app.start = AsyncMock()
    app.stop = AsyncMock()
    app.shutdown = AsyncMock()
    app.updater = MagicMock()
    app.updater.start_polling = AsyncMock()

    with patch("trader.telegram_bot.Application") as app_cls:
        app_cls.builder.return_value.token.return_value.build.return_value = app
        await bot.start()

    assert bot._config.allowed_chat_ids == {12345}
    assert 12345 in bot._subscribed
    assert 99999 not in bot._subscribed


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
async def test_pnl_analysis_renders_canonical_regime_and_weekday_returns() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.pnl_analysis_provider = AsyncMock(
        return_value={
            "horizon_minutes": 5,
            "label_schema_version": "directional_net_v2",
            "symbols_best": [],
            "symbols_worst": [],
            "hours": [],
            "regimes": [
                {
                    "regime": "BULL_TREND",
                    "count": 69,
                    "avg_net_return_bps": -12.34,
                    "total_net_return_bps": -851.46,
                }
            ],
            "weekdays": [
                {
                    "weekday": 2,
                    "count": 418,
                    "avg_net_return_bps": -26.02,
                    "total_net_return_bps": -10876.36,
                }
            ],
            "strategies": [
                {
                    "strategy_id": "scalp_micro_v1",
                    "count": 25,
                    "avg_gross_return_bps": 1.0,
                    "avg_cost_bps": 27.0,
                    "avg_net_return_bps": -26.0,
                    "total_net_return_bps": -650.0,
                }
            ],
        }
    )
    update = _fake_update()

    await bot._cmd_pnl_analysis(update, _fake_context())  # type: ignore[arg-type]

    text = update.effective_message.reply_text.call_args.args[0]
    assert "BULL_TREND: <code>69</code>, avg <code>-12.34</code>, Σ <code>-851.5</code>" in text
    assert "2: <code>418</code>, avg <code>-26.02</code>, Σ <code>-10876.4</code>" in text
    assert "⛔ <code>scalp_micro_v1</code>: <code>25</code>" in text
    assert "gross <code>+1.00</code>, costs <code>27.00</code>, net <code>-26.00 bps</code>" in text


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
    assert bot._train_callback(500) in all_callbacks
    assert bot._train_callback() in all_callbacks
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
    _confirm_callback(callbacks, bot._train_callback(500))


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

    markup = bot._confirm_menu(bot._train_callback(500))
    payload = markup.inline_keyboard[0][0].callback_data.removeprefix("confirm:")
    await bot._handle_confirm_button(update, payload)

    bot._controller.start_training.assert_awaited_once_with(500, 5, 2.0)
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
                "version": "v_candidate",
                "status": "SHADOW_CHALLENGER",
                "metrics": {"quality": "WEAK", "walk_forward_expectancy_bps": -2.5},
            },
            "active_model_version": {
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
                "metrics": {
                    "quality": "WEAK",
                    "walk_forward_expectancy_bps": -46.91,
                    "wf_positive_folds": 1,
                    "wf_folds": 5,
                    "wf_min_bps": -60.0,
                    "wf_std_bps": 31.0,
                },
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
    assert "folds 1/5" in reply_text
    assert "std +31.00 bps" in reply_text
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
    ctx = type("_Ctx", (), {"args": ["500", "5", "2"]})()

    await bot._cmd_train(update, ctx)  # type: ignore[arg-type]

    bot._controller.start_training.assert_not_awaited()
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "Запустить обучение" in reply_text
    reply_markup = update.effective_message.reply_text.call_args.kwargs["reply_markup"]
    callbacks = [btn.callback_data for row in reply_markup.inline_keyboard for btn in row]
    _confirm_callback(callbacks, "train:500:5:2")


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

    await bot._handle_train_button(update, "1000:5:2")

    bot._controller.start_training.assert_not_awaited()
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "Запустить обучение" in reply_text
    reply_markup = update.effective_message.reply_text.call_args.kwargs["reply_markup"]
    callbacks = [btn.callback_data for row in reply_markup.inline_keyboard for btn in row]
    _confirm_callback(callbacks, "train:1000:5:2")


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
    """The dashboard pause button must go through a confirm dialog, not
    mutate trading state on the first tap — same as /pause and the
    control:pause submenu button."""
    bot = _make_bot()
    update = _fake_callback_update()
    update.callback_query.data = "action:pause"

    await bot._on_button(update, MagicMock())  # type: ignore[arg-type]

    assert bot._controller is not None
    bot._controller.pause.assert_not_awaited()
    update.callback_query.edit_message_text.assert_awaited()
    edited_text = update.callback_query.edit_message_text.await_args.args[0]
    assert "пауз" in edited_text.lower()


@pytest.mark.asyncio
async def test_render_model_uses_lite_db_diag(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.db_diagnostics_provider = AsyncMock(return_value={"connected": True})
    load_diag = AsyncMock(
        return_value={
            "lite": True,
            "latest_model_version": {
                "version": "v-lite",
                "status": "SHADOW_CHALLENGER",
                "training_samples": 500,
                "metrics": {
                    "quality": "GOOD",
                    "lift_bps": 1.5,
                    "precision": 0.55,
                    "wf_positive_folds": 3,
                    "wf_folds": 5,
                    "wf_min_bps": 1.2,
                    "wf_std_bps": 4.5,
                },
            },
            "training_eligible_15m": 500,
        }
    )
    monkeypatch.setattr(bot, "_load_db_diag", load_diag)

    text, _markup = await bot._render_model()

    load_diag.assert_awaited_once_with(lite=True)
    assert "v-lite" in text
    assert "folds 3/5" in text
    assert "std +4.50 bps" in text


@pytest.mark.asyncio
async def test_canary_button_uses_respond_not_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = _make_bot()
    bot._controller.db_diagnostics_provider = AsyncMock(return_value={"connected": True, "lite": True})
    bot._controller.diagnostics_provider = MagicMock(return_value={"active_symbols": ["BTCUSDT"]})
    respond = AsyncMock()
    monkeypatch.setattr(bot, "_respond", respond)
    update = _fake_callback_update()
    update.callback_query.data = "action:canary"

    await bot._on_button(update, MagicMock())  # type: ignore[arg-type]

    respond.assert_awaited()


@pytest.mark.asyncio
async def test_load_db_diag_timeout_returns_error() -> None:
    bot = _make_bot()

    async def _slow(*, lite: bool = False) -> dict[str, object]:
        del lite
        await asyncio.sleep(0.2)
        return {"connected": True}

    assert bot._controller is not None
    bot._controller.db_diagnostics_provider = _slow
    bot._DB_DIAG_TIMEOUT_LITE_S = 0.05

    diag = await bot._load_db_diag(lite=True)

    assert diag["error"] == "db_diagnostics_timeout"


@pytest.mark.asyncio
async def test_load_db_diag_full_timeout_uses_lite_fallback() -> None:
    bot = _make_bot()

    async def _provider(*, lite: bool = False) -> dict[str, object]:
        if lite:
            return {
                "connected": True,
                "configured": True,
                "lite": True,
                "latest_model_version": {"version": "v-lite"},
                "candles_by_interval": {"1": 1000},
            }
        await asyncio.sleep(0.2)
        return {"connected": True, "lite": False}

    assert bot._controller is not None
    bot._controller.db_diagnostics_provider = _provider
    bot._DB_DIAG_TIMEOUT_FULL_S = 0.05
    bot._DB_DIAG_TIMEOUT_LITE_S = 0.1

    diag = await bot._load_db_diag(lite=False)

    assert diag["connected"] is True
    assert diag["error"] == "db_diagnostics_timeout"
    assert diag["full_diagnostics_timeout"] is True
    assert diag["lite"] is True
    assert diag["latest_model_version"] == {"version": "v-lite"}


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
async def test_runtime_settings_none_does_not_break_dashboard_buttons() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.runtime_settings = MagicMock(return_value=None)

    home_text, _ = await bot._render_home()
    settings_text, _ = await bot._render_settings()
    limits_text = bot._limits_text()

    assert "Bybit AI Trader" in home_text
    assert "Настройки" in settings_text
    assert "Лимиты" in limits_text


@pytest.mark.asyncio
async def test_deep_report_tolerates_none_provider_payloads() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.runtime_settings = MagicMock(return_value=None)
    bot._controller.diagnostics_provider = MagicMock(return_value=None)
    bot._controller.db_diagnostics_provider = AsyncMock(return_value=None)
    bot._controller.healthcheck_provider = AsyncMock(return_value=None)
    bot._controller.compare_provider = AsyncMock(return_value=None)
    bot._controller.pnl_analysis_provider = AsyncMock(return_value=None)
    bot._controller.costs_detailed_provider = AsyncMock(return_value=None)
    bot._controller.model_performance_provider = AsyncMock(return_value=None)
    bot._controller.champion_health_provider = AsyncMock(return_value=None)

    text = await bot._render_deep_report_text()

    assert "ПОЛНАЯ СВОДКА ДЛЯ АНАЛИЗА" in text
    assert "Runtime settings JSON" in text
    assert "Compare / PnL / Costs JSON" in text


@pytest.mark.asyncio
async def test_deep_report_compact_db_includes_connect_error_and_target() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.db_diagnostics_provider = AsyncMock(
        return_value={
            "connected": False,
            "last_connect_error": "Failed to connect to database: {:error, :econnrefused}",
            "connection_target": {
                "scheme": "postgresql",
                "host": "db.internal",
                "port": 5432,
                "database": "trades",
            },
            "schema_health": {
                "ok": False,
                "missing_columns": ["prediction_outcomes.label_threshold_bps"],
            },
            "prediction_events": 12,
            "prediction_event_decision_counts": {
                "total_count": 12,
                "resolved_count": 7,
                "pending_count": 5,
                "by_decision": {"SHADOW_BASELINE": {"total_count": 12}},
            },
        }
    )

    text = await bot._render_deep_report_text()

    assert "econnrefused" in text
    assert "db.internal" in text
    assert "connection_target" in text
    assert "prediction_outcomes.label_threshold_bps" in text
    assert "prediction_events" in text
    assert "SHADOW_BASELINE" in text


@pytest.mark.asyncio
async def test_deep_report_shows_model_gate_breakdown() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.diagnostics_provider = MagicMock(
        return_value={
            "model": {"challenger_version": "v5"},
            "hour_signals_emitted": 3,
        }
    )
    bot._controller.db_diagnostics_provider = AsyncMock(
        return_value={
            "connected": True,
            "latest_model_version": {
                "version": "v5",
                "status": "SHADOW_CHALLENGER",
                "metrics": {"quality": "GOOD", "horizon_minutes": 5, "walk_forward_expectancy_bps": 2.0},
            },
            "model_gate_horizon_minutes": 5,
            "prediction_event_decision_counts": {
                "by_decision": {
                    "SHADOW_BASELINE": {"total_count": 40, "resolved_count": 30, "pending_count": 10},
                    "GATE_PASS": {"total_count": 15, "resolved_count": 12, "pending_count": 3},
                    "GATE_BLOCK": {"total_count": 10, "resolved_count": 8, "pending_count": 2},
                }
            },
            "shadow_gate_by_horizon": {
                "5": {
                    "total_count": 33,
                    "event_total_count": 40,
                    "event_pending_count": 7,
                    "pass_count": 12,
                    "lift_vs_all_bps": 2.5,
                    "side_filtered_count": 21,
                    "score_block_count": 7,
                    "score_block_avg_net_return_bps": -1.25,
                    "top_block_reasons": {
                        "side_not_selected_by_model": 21,
                        "score_below_regime_threshold": 7,
                    },
                }
            },
            "paper_pnl_by_horizon": {"5": {"baseline": {"count": 10}, "model_gate": {"count": 4}}},
        }
    )

    text = await bot._render_deep_report_text()

    assert "Gate breakdown" in text
    assert "side-filter=<code>21</code>" in text
    assert "score-block=<code>7</code>" in text
    assert "score avg=<code>-1.25</code>" in text
    assert "&quot;gate_breakdown&quot;" in text


@pytest.mark.asyncio
async def test_deep_report_shows_recent_signal_block_reasons() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.diagnostics_provider = MagicMock(
        return_value={
            "hour_signals_emitted": 5,
            "hour_shadow_order_would_be_placed": 0,
            "shadow_mode": True,
        }
    )
    bot._controller.db_diagnostics_provider = AsyncMock(
        return_value={
            "connected": True,
            "latest_model_version": {
                "version": "v5",
                "status": "SHADOW_CHALLENGER",
                "metrics": {"quality": "WEAK", "horizon_minutes": 5, "walk_forward_expectancy_bps": -1.0},
            },
            "model_gate_horizon_minutes": 5,
            "recent_signal_block_reasons": [
                {
                    "reason": "net_edge_rejected",
                    "count": 4,
                    "latest_at": "2026-06-29T10:00:00+00:00",
                },
                {
                    "reason": "min_notional_rejected",
                    "count": 1,
                    "latest_at": "2026-06-29T10:01:00+00:00",
                },
            ],
        }
    )

    text = await bot._render_deep_report_text()

    assert "Почему входы не доходят до paper/live" in text
    assert "net_edge_rejected" in text
    assert "ожидаемая net-edge ниже комиссий" in text
    assert "min_notional_rejected" in text
    assert "&quot;recent_signal_block_reasons_24h&quot;" in text


@pytest.mark.asyncio
async def test_deep_report_shows_strategy_gate_limits() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.runtime_settings = MagicMock(
        return_value={
            "strategy_side_blocked": ["scalp_micro_v1:Buy"],
            "strategy_side_confidence_limited": ["mean_reversion_v1:Sell:8:+1.0"],
            "strategy_regime_blocked": ["scalp_micro_v1:SIDEWAYS"],
            "strategy_regime_confidence_limited": ["mean_reversion_v1:BULL_TREND:8:+1.0"],
        }
    )
    bot._controller.diagnostics_provider = MagicMock(return_value={"shadow_mode": True})
    bot._controller.db_diagnostics_provider = AsyncMock(return_value={"connected": True})

    text = await bot._render_deep_report_text()

    assert "Strategy gates: что сейчас режет" in text
    assert "scalp_micro_v1:Buy" in text
    assert "mean_reversion_v1:Sell:8:+1.0" in text
    assert "scalp_micro_v1:SIDEWAYS" in text
    assert "mean_reversion_v1:BULL_TREND:8:+1.0" in text


def test_diagnostics_menu_has_db_probe_button() -> None:
    bot = _make_bot()

    callback_data = [
        button.callback_data
        for row in bot._diagnostics_menu().inline_keyboard
        for button in row
        if button.callback_data is not None
    ]

    assert "view:db_probe" in callback_data


@pytest.mark.asyncio
async def test_db_probe_success_redacts_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = _make_bot()
    monkeypatch.setenv(
        "POSTGRES_DSN",
        "postgresql+asyncpg://postgres.projectref:secret@aws-0-eu-west-1.pooler.supabase.com:6543/postgres?sslmode=require",
    )

    fake_conn = MagicMock()
    fake_conn.fetchrow = AsyncMock(return_value={"db": "postgres", "usr": "postgres.projectref", "addr": "10.0.0.1"})
    fake_conn.close = AsyncMock()

    import asyncpg

    monkeypatch.setattr(asyncpg, "connect", AsyncMock(return_value=fake_conn))

    text = await bot._render_db_probe_text()

    assert "asyncpg.connect OK" in text
    assert "SSL arg: <code>SSLContext(no_verify)</code>" in text
    assert "username_has_project_ref" in text
    assert "secret" not in text


@pytest.mark.asyncio
async def test_db_probe_failure_reports_sanitized_error(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = _make_bot()
    monkeypatch.setenv(
        "POSTGRES_DSN",
        "postgresql+asyncpg://postgres.projectref:secret@aws-0-eu-west-1.pooler.supabase.com:6543/postgres?sslmode=require",
    )

    import asyncpg

    monkeypatch.setattr(asyncpg, "connect", AsyncMock(side_effect=RuntimeError("EAUTHQUERY failed")))

    text = await bot._render_db_probe_text()

    assert "asyncpg.connect failed" in text
    assert "EAUTHQUERY failed" in text
    assert "SSL arg: <code>SSLContext(no_verify)</code>" in text
    assert "secret" not in text


def test_db_connection_fix_hint_for_supabase_pooler_refused() -> None:
    hint = TelegramMonitorBot._db_connection_fix_hint(
        {
            "last_connect_error": "Failed to connect to database: {:error, :econnrefused}",
            "connection_target": {
                "host": "aws-0-eu-west-1.pooler.supabase.com",
                "port": 5432,
                "database": "postgres",
            },
        }
    )

    assert "6543" in hint
    assert "sslmode=require" in hint
    assert "Supabase pooler" in hint


def test_db_connection_fix_hint_for_supabase_schema_bootstrap_degraded() -> None:
    hint = TelegramMonitorBot._db_connection_fix_hint(
        {
            "last_connect_error": "schema bootstrap degraded: connection was closed in the middle of operation",
            "connection_target": {
                "host": "aws-0-eu-west-1.pooler.supabase.com",
                "port": 6543,
                "database": "postgres",
            },
        }
    )

    assert "split-bootstrap" in hint
    assert "schema bootstrap" in hint
    assert "6543" not in hint


def test_db_connection_fix_hint_for_supabase_eauthquery() -> None:
    hint = TelegramMonitorBot._db_connection_fix_hint(
        {
            "last_connect_error": (
                "schema bootstrap degraded: (EAUTHQUERY) authentication query failed: "
                "connection to database not available"
            ),
            "connection_target": {
                "host": "aws-0-eu-west-1.pooler.supabase.com",
                "port": 6543,
                "database": "postgres",
                "username_prefix": "postgres",
                "username_has_project_ref": False,
            },
        }
    )

    assert "postgres.<project-ref>" in hint
    assert "DB password" in hint
    assert "not paused" in hint or "не paused" in hint


@pytest.mark.asyncio
async def test_button_reply_falls_back_to_new_message_when_edit_unavailable() -> None:
    bot = _make_bot()
    update = _fake_callback_update()
    update.callback_query.edit_message_text = AsyncMock(side_effect=RuntimeError("edit timeout"))
    sent = MagicMock(message_id=99, chat_id=12345)
    bot._app = MagicMock()
    bot._app.bot.send_message = AsyncMock(return_value=sent)

    await bot._button_reply(update, "✅ Готово", reply_markup=bot._main_menu())

    bot._app.bot.send_message.assert_awaited()
    kwargs = bot._app.bot.send_message.await_args.kwargs
    assert "Готово" in kwargs.get("text", "")


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
async def test_telegram_polling_conflicts_schedule_recovery_after_threshold() -> None:
    from telegram.error import Conflict

    bot = _make_bot()

    for _ in range(3):
        bot._polling_error_callback(Conflict("terminated by other getUpdates request"))

    await asyncio.sleep(0)
    health = bot.health_snapshot()
    assert health["polling_conflict_count"] == 3
    assert health["polling_disabled_reason"] == "polling_conflict_recovery_pending"
    assert bot._polling_recovery_task is not None
    assert not bot._polling_recovery_task.done()


@pytest.mark.asyncio
async def test_telegram_early_deploy_conflict_schedules_recovery() -> None:
    from telegram.error import Conflict

    bot = _make_bot()
    bot._started_at = datetime.now(tz=UTC)

    bot._polling_error_callback(Conflict("terminated by other getUpdates request"))

    await asyncio.sleep(0)
    assert bot._polling_recovery_task is not None
    assert bot.health_snapshot()["polling_disabled_reason"] == "polling_conflict_recovery_pending"


def test_telegram_conflict_recovery_deferred_without_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    from telegram.error import Conflict

    bot = _make_bot()

    def _no_loop() -> asyncio.AbstractEventLoop:
        raise RuntimeError("no running event loop")

    monkeypatch.setattr(asyncio, "get_running_loop", _no_loop)
    for _ in range(3):
        bot._polling_error_callback(Conflict("terminated by other getUpdates request"))

    assert bot._polling_recovery_pending is True
    assert bot._polling_recovery_task is None


@pytest.mark.asyncio
async def test_telegram_zombie_polling_triggers_full_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = _make_bot()
    bot._started_at = datetime.now(tz=UTC) - timedelta(seconds=300)
    bot._polling_conflict_count = 2
    bot._config.polling_zombie_silence_s = 60

    fake_updater = MagicMock()
    fake_updater.running = True
    fake_app = MagicMock()
    fake_app.updater = fake_updater
    bot._app = fake_app

    teardown = AsyncMock()
    start = AsyncMock(return_value=True)
    monkeypatch.setattr(bot, "_teardown_app", teardown)
    monkeypatch.setattr(bot, "start", start)

    await bot.ensure_polling_running()

    teardown.assert_awaited_once()
    start.assert_awaited_once()


@pytest.mark.asyncio
async def test_telegram_polling_lock_acquire_and_release(monkeypatch: pytest.MonkeyPatch) -> None:
    import redis.asyncio as aioredis

    class FakeRedis:
        def __init__(self) -> None:
            self.closed = False
            self.eval_calls = 0

        async def set(self, *_args: object, **_kwargs: object) -> bool:
            return True

        async def eval(self, *_args: object) -> int:
            self.eval_calls += 1
            return 1

        async def aclose(self) -> None:
            self.closed = True

    fake = FakeRedis()
    monkeypatch.setattr(aioredis, "from_url", lambda *_args, **_kwargs: fake)
    bot = _make_bot()
    bot._config.redis_url = "redis://localhost:6379/0"

    assert await bot._acquire_polling_lock() is True
    assert bot.health_snapshot()["polling_lock_owner"] is True

    await bot._release_polling_lock()

    assert fake.closed is True
    assert bot.health_snapshot()["polling_lock_owner"] is False


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
            "training_filtered_total_by_horizon": {"5": 1400, "15": 0},
            "newest_training_schema_by_horizon": {
                "5": {
                    "sample_count": 110,
                    "best_schema_count": 1234,
                    "best_schema_hash": "best_schema_hash",
                    "trainable_schema_count": 1234,
                    "trainable_schema_hash": "best_schema_hash",
                }
            },
            "latest_training_run": {"status": "COMPLETED", "sample_count": 1234, "model_version": "v5"},
            "latest_model_version": {
                "version": "v5",
                "status": "SHADOW_CHALLENGER",
                "training_samples": 1234,
                "metrics": {
                    "quality": "GOOD",
                    "horizon_minutes": 5,
                    "lift_bps": 1.2,
                    "walk_forward_expectancy_bps": 4.0,
                    "raw_wf_mean_bps": -8.0,
                    "selected_sides": ["Sell"],
                    "side_filter": {"reason": "positive_out_of_sample_side_expectancy"},
                },
            },
            "model_gate_horizon_minutes": 5,
            "prediction_event_decision_counts": {
                "by_decision": {
                    "SHADOW_BASELINE": {"total_count": 40, "resolved_count": 30, "pending_count": 10},
                    "GATE_PASS": {"total_count": 15, "resolved_count": 12, "pending_count": 3},
                    "GATE_BLOCK": {"total_count": 10, "resolved_count": 8, "pending_count": 2},
                }
            },
            "shadow_gate_by_horizon": {
                "5": {
                    "horizon_minutes": 5,
                    "total_count": 44,
                    "pass_count": 12,
                    "block_count": 32,
                    "lift_vs_all_bps": 2.5,
                    "side_filtered_count": 24,
                    "score_block_count": 8,
                    "score_block_avg_net_return_bps": -1.75,
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
    assert "Схема обучения (5m)" in reply_text
    assert "filtered=<code>1400</code>" in reply_text
    assert "best=<code>1234</code>" in reply_text
    assert "trainable=<code>best_sch</code>" in reply_text
    assert "best schema 1234" in reply_text
    assert "Фильтр модели 5m" in reply_text
    assert "Side-filter: <code>Sell</code>" in reply_text
    assert "WF до side-filter: <code>-8.00 bps</code>" in reply_text
    assert "positive_out_of_sample_side_expectancy" in reply_text
    assert "Блоки side-filter/score: <code>24/8</code>" in reply_text
    assert "score avg=<code>-1.75 bps</code>" in reply_text
    assert "SHADOW_BASELINE: 40/30/10" in reply_text
    assert "GATE_PASS: 15/12/3" in reply_text
    assert "+2.50 bps" in reply_text
    assert "Фильтр модели 15m" not in reply_text


@pytest.mark.asyncio
async def test_db_model_warns_when_shadow_closes_exist_but_db_paper_gate_empty() -> None:
    bot = _make_bot()
    assert bot._controller is not None
    bot._controller.diagnostics_provider = MagicMock(
        return_value={
            "model": {"champion_version": "none", "challenger_version": "v5"},
            "hour_shadow_closed": 5,
            "hour_shadow_closed_avg_pnl_pct": -0.1688,
        }
    )
    bot._controller.db_diagnostics_provider = AsyncMock(
        return_value={
            "connected": True,
            "configured": True,
            "latest_candle_1m": datetime.now(tz=UTC),
            "candles_by_interval": {"1": 5000, "5": 1000, "15": 0, "60": 100},
            "prediction_outcomes_by_horizon": {"5": 1500},
            "training_eligible_by_horizon": {"5": 1234},
            "latest_model_version": {
                "version": "v5",
                "status": "SHADOW_CHALLENGER",
                "training_samples": 1234,
                "metrics": {
                    "quality": "GOOD",
                    "horizon_minutes": 5,
                    "walk_forward_expectancy_bps": 4.0,
                },
            },
            "model_gate_horizon_minutes": 5,
            "shadow_gate_by_horizon": {"5": {"total_count": 0, "event_pending_count": 5}},
            "paper_pnl_by_horizon": {"5": {"baseline": {"count": 0}, "model_gate": {"count": 0}}},
        }
    )
    update = _fake_update()
    ctx = type("_Ctx", (), {"args": []})()

    await bot._cmd_db_model(update, ctx)  # type: ignore[arg-type]

    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "Runtime shadow closes есть, но DB paper gate пуст" in reply_text
    assert "5 закрытий, avg -0.1688%" in reply_text
    assert "Проверьте запись prediction outcomes/DB" in reply_text
