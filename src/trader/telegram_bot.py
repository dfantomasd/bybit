"""Read-only Telegram monitoring bot.

The bot intentionally exposes observability commands only. It must never submit,
cancel, amend, or close orders.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from trader.domain.models import Balance, HealthStatus, Position
from trader.monitoring.logging import get_logger

log = get_logger(__name__)


AdapterFactory = Callable[[], Any | None]


@dataclass(frozen=True)
class TelegramBotConfig:
    """Runtime configuration for the read-only Telegram bot."""

    token: str
    allowed_chat_ids: set[int]
    trading_mode: str
    risk_profile: str
    bybit_use_testnet: bool
    default_category: str = "linear"


class TelegramMonitorBot:
    """Small read-only Telegram bot for operator visibility."""

    def __init__(
        self,
        config: TelegramBotConfig,
        health_provider: Callable[[], Awaitable[HealthStatus]],
        adapter_factory: AdapterFactory,
    ) -> None:
        self._config = config
        self._health_provider = health_provider
        self._adapter_factory = adapter_factory
        self._app: Application[Any, Any, Any, Any, Any, Any] | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._config.token and self._config.allowed_chat_ids)

    async def start(self) -> None:
        """Start Telegram polling in the current event loop."""
        if not self.enabled:
            log.info("telegram_bot_disabled")
            return

        app = Application.builder().token(self._config.token).build()
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("help", self._cmd_help))
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("balance", self._cmd_balance))
        app.add_handler(CommandHandler("positions", self._cmd_positions))

        await app.initialize()
        await app.start()
        if app.updater is None:
            raise RuntimeError("Telegram updater was not created")
        await app.updater.start_polling(drop_pending_updates=True)

        self._app = app
        log.info("telegram_bot_started", allowed_chats=len(self._config.allowed_chat_ids))

    async def stop(self) -> None:
        """Stop Telegram polling and release resources."""
        if self._app is None:
            return

        app = self._app
        self._app = None
        if app.updater is not None:
            await app.updater.stop()
        await app.stop()
        await app.shutdown()
        log.info("telegram_bot_stopped")

    async def _authorised(self, update: Update) -> bool:
        chat = update.effective_chat
        if chat is None or chat.id not in self._config.allowed_chat_ids:
            if update.effective_message is not None:
                suffix = f" Chat ID: {chat.id}" if chat is not None else ""
                await update.effective_message.reply_text(f"Access denied.{suffix}")
            log.warning("telegram_unauthorised_chat", chat_id=chat.id if chat else None)
            return False
        return True

    async def _reply(self, update: Update, text: str) -> None:
        if update.effective_message is not None:
            await update.effective_message.reply_text(
                text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        await self._reply(update, self._help_text())

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        await self._reply(update, self._help_text())

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return

        try:
            health = await self._health_provider()
        except Exception as exc:
            log.warning("telegram_status_failed", error=str(exc))
            await self._reply(update, f"<b>Status</b>\nHealth check failed: <code>{exc}</code>")
            return

        lines = [
            "<b>Status</b>",
            f"Overall: <code>{health.overall}</code>",
            f"System: <code>{health.system_status.value}</code>",
            f"Mode: <code>{health.trading_mode.value}</code>",
            f"Risk: <code>{self._config.risk_profile}</code>",
            f"Testnet: <code>{str(self._config.bybit_use_testnet).lower()}</code>",
            "",
            self._component_line("Postgres", health.postgres, health.postgres_latency_ms),
            self._component_line("Redis", health.redis, health.redis_latency_ms),
            self._component_line("Bybit REST", health.bybit_rest, health.bybit_rest_latency_ms),
            self._component_line("Bybit WS", health.bybit_ws, None),
        ]
        if health.messages:
            lines.append("")
            lines.append("<b>Messages</b>")
            lines.extend(f"- {message}" for message in health.messages[:6])
        await self._reply(update, "\n".join(lines))

    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return

        adapter = self._adapter_factory()
        if adapter is None:
            await self._reply(update, "Bybit adapter is not available yet.")
            return

        try:
            balance: Balance = await adapter.get_balance()
        except Exception as exc:
            log.warning("telegram_balance_failed", error=str(exc))
            await self._reply(update, f"<b>Balance</b>\nBybit request failed: <code>{exc}</code>")
            return

        lines = [
            "<b>Balance</b>",
            f"Currency: <code>{balance.currency}</code>",
            f"Wallet: <code>{balance.wallet_balance}</code>",
            f"Available: <code>{balance.available_balance}</code>",
        ]
        if balance.margin_balance is not None:
            lines.append(f"Margin: <code>{balance.margin_balance}</code>")
        if balance.unrealised_pnl:
            lines.append(f"Unrealised PnL: <code>{balance.unrealised_pnl}</code>")
        await self._reply(update, "\n".join(lines))

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return

        adapter = self._adapter_factory()
        if adapter is None:
            await self._reply(update, "Bybit adapter is not available yet.")
            return

        try:
            positions: list[Position] = await adapter.get_positions(self._config.default_category)
        except Exception as exc:
            log.warning("telegram_positions_failed", error=str(exc))
            await self._reply(update, f"<b>Positions</b>\nBybit request failed: <code>{exc}</code>")
            return

        open_positions = [position for position in positions if position.size > 0]
        if not open_positions:
            await self._reply(update, "<b>Positions</b>\nNo open positions.")
            return

        lines = ["<b>Positions</b>"]
        for position in open_positions[:10]:
            lines.extend(
                [
                    "",
                    f"<b>{position.symbol}</b> {position.side.value}",
                    f"Size: <code>{position.size}</code>",
                    f"Entry: <code>{position.entry_price}</code>",
                    f"Mark: <code>{position.mark_price}</code>",
                    f"PnL: <code>{position.unrealised_pnl}</code>",
                    f"Leverage: <code>{position.leverage}</code>",
                ]
            )
        if len(open_positions) > 10:
            lines.append(f"\nShowing 10 of {len(open_positions)} positions.")
        await self._reply(update, "\n".join(lines))

    def _help_text(self) -> str:
        return "\n".join(
            [
                "<b>Bybit AI Trader</b>",
                "Read-only monitoring bot.",
                "",
                "/status - system health",
                "/balance - Bybit wallet balance",
                "/positions - open positions",
                "/help - command list",
            ]
        )

    def _component_line(self, name: str, ok: bool, latency_ms: float | None) -> str:
        state = "ok" if ok else "fail"
        if latency_ms is None:
            return f"{name}: <code>{state}</code>"
        return f"{name}: <code>{state}</code> ({latency_ms:.0f} ms)"
