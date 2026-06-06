"""Tests for P0.14: Telegram risk escalation gate."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.telegram_bot import _RISK_LEVEL, TelegramBotConfig, TelegramMonitorBot, TradingController


class TestTelegramRiskGate:
    def test_risk_level_ordering(self):
        """CONSERVATIVE < MODERATE < AGGRESSIVE < SCALP by risk level."""
        assert _RISK_LEVEL["CONSERVATIVE"] < _RISK_LEVEL["MODERATE"]
        assert _RISK_LEVEL["MODERATE"] < _RISK_LEVEL["AGGRESSIVE"]
        assert _RISK_LEVEL["AGGRESSIVE"] < _RISK_LEVEL["SCALP"]

    def test_allow_risk_increase_defaults_false(self):
        """TradingController.allow_risk_increase defaults to False."""
        ctrl = TradingController(
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
        )
        assert ctrl.allow_risk_increase is False

    @pytest.mark.asyncio
    async def test_risk_escalation_blocked_when_flag_false(self):
        """Telegram /risk command blocks escalation when allow_risk_increase=False."""
        from unittest.mock import AsyncMock

        from telegram import Update

        ctrl = TradingController(
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
            allow_risk_increase=False,  # BLOCKED
        )

        bot = TelegramMonitorBot(
            config=TelegramBotConfig(
                token="fake",
                allowed_chat_ids={12345},
                trading_mode="SHADOW",
                risk_profile="CONSERVATIVE",
                bybit_use_testnet=True,
            ),
            health_provider=AsyncMock(),
            adapter_factory=lambda: None,
            controller=ctrl,
        )

        # Simulate an authorised update requesting AGGRESSIVE profile
        update = MagicMock(spec=Update)
        update.effective_chat = MagicMock()
        update.effective_chat.id = 12345
        update.effective_message = MagicMock()
        update.effective_message.reply_text = AsyncMock()

        context = MagicMock()
        context.args = ["aggressive"]

        await bot._cmd_risk(update, context)

        # Risk profile change must NOT have been queued
        assert 12345 not in bot._pending
        # set_risk_profile must NOT have been called
        ctrl.set_risk_profile.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_risk_escalation_allowed_when_flag_true(self):
        """Telegram /risk escalation queues /confirm action when flag is True."""
        from unittest.mock import AsyncMock

        from telegram import Update

        ctrl = TradingController(
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
            allow_risk_increase=True,  # ALLOWED
        )

        bot = TelegramMonitorBot(
            config=TelegramBotConfig(
                token="fake",
                allowed_chat_ids={12345},
                trading_mode="SHADOW",
                risk_profile="CONSERVATIVE",
                bybit_use_testnet=True,
            ),
            health_provider=AsyncMock(),
            adapter_factory=lambda: None,
            controller=ctrl,
        )

        update = MagicMock(spec=Update)
        update.effective_chat = MagicMock()
        update.effective_chat.id = 12345
        update.effective_message = MagicMock()
        update.effective_message.reply_text = AsyncMock()

        context = MagicMock()
        context.args = ["aggressive"]

        await bot._cmd_risk(update, context)

        # Should be queued for /confirm
        assert 12345 in bot._pending

    @pytest.mark.asyncio
    async def test_risk_reduction_always_allowed(self):
        """Reducing risk profile (AGGRESSIVE → CONSERVATIVE) is always allowed."""
        from unittest.mock import AsyncMock

        from telegram import Update

        ctrl = TradingController(
            pause=AsyncMock(),
            resume=AsyncMock(),
            set_shadow=AsyncMock(),
            set_risk_profile=AsyncMock(),
            emergency_stop=AsyncMock(),
            is_paused=lambda: False,
            is_shadow=lambda: True,
            current_profile=lambda: "AGGRESSIVE",  # currently aggressive
            active_symbols=lambda: [],
            regime_for=lambda s: None,
            allow_risk_increase=False,  # escalation blocked, but reduction should work
        )

        bot = TelegramMonitorBot(
            config=TelegramBotConfig(
                token="fake",
                allowed_chat_ids={12345},
                trading_mode="SHADOW",
                risk_profile="AGGRESSIVE",
                bybit_use_testnet=True,
            ),
            health_provider=AsyncMock(),
            adapter_factory=lambda: None,
            controller=ctrl,
        )

        update = MagicMock(spec=Update)
        update.effective_chat = MagicMock()
        update.effective_chat.id = 12345
        update.effective_message = MagicMock()
        update.effective_message.reply_text = AsyncMock()

        context = MagicMock()
        context.args = ["conservative"]  # reduction

        await bot._cmd_risk(update, context)

        # Should be queued for /confirm (reduction is allowed)
        assert 12345 in bot._pending
