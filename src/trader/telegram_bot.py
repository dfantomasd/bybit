"""Telegram bot — monitoring + operator control.

Observability commands (anyone in allowed_chat_ids):
  /status    — system health
  /balance   — wallet balance
  /positions — open positions + unrealised PnL
  /signals   — last 10 strategy signals
  /regime    — current market regime per symbol
  /symbols   — active symbols from screener
  /pnl       — recent closed PnL
  /canary    — readiness check for a tiny CANARY_LIVE test

Control commands (require /confirm for dangerous ones):
  /pause              — pause new entries (keep existing positions)
  /resume             — resume after pause
  /train [min] [horizon] [label_bps] — train shadow challenger
  /limits             — show runtime safety limits
  /limits entries|pending|same_side|price_cap|feature_symbols|exec_candidates N
  /shadow on|off      — toggle shadow mode
  /risk conservative|moderate|aggressive|scalp — change risk profile
  /mode shadow|active — switch execution mode
  /stop               — emergency full stop (requires /confirm)

Push notifications (sent to subscribed chats):
  • Signal generated
  • Position opened / closed
  • Circuit breaker triggered
  • Risk profile changed

Safety:
  - Control commands only affect new-entry logic and shadow mode.
  - The bot can NEVER submit, cancel, amend, or close orders directly.
  - All financial operations still go through RiskManager + ExecutionEngine.
"""

from __future__ import annotations

import html
import json
import os
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import structlog
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

from trader.domain.enums import RiskProfile
from trader.domain.models import Balance, HealthStatus, Position

log = structlog.get_logger(__name__)

AdapterFactory = Callable[[], Any | None]
HealthProvider = Callable[[], Awaitable[HealthStatus]]


# ---------------------------------------------------------------------------
# Signal log entry (lightweight — no circular import on TradeProposal)
# ---------------------------------------------------------------------------


@dataclass
class SignalEntry:
    timestamp: datetime
    symbol: str
    side: str  # "BUY" or "SELL"
    confidence: float
    regime: str  # MarketRegime.value
    rationale: str
    shadow: bool  # True = not executed


# ---------------------------------------------------------------------------
# Control interface (callbacks wired from app.py)
# ---------------------------------------------------------------------------

_RISK_LEVEL: dict[str, int] = {
    "CONSERVATIVE": 0,
    "MODERATE": 1,
    "AGGRESSIVE": 2,
    "SCALP": 3,
}

_STATUS_RU: dict[str, str] = {
    "GOOD": "ХОРОШО",
    "WEAK": "СЛАБО",
    "INSUFFICIENT_VALIDATION": "МАЛО ПРОВЕРКИ",
    "SHADOW_CHALLENGER": "кандидат в тени",
    "VALIDATED": "проверена",
    "CHAMPION": "основная модель",
    "REJECTED": "отклонена",
    "ROLLED_BACK": "откачена",
    "COMPLETED": "завершено",
    "RUNNING": "идет",
    "FAILED": "ошибка",
    "none": "нет",
    "never": "никогда",
    "n/a": "нет данных",
}


@dataclass
class TradingController:
    """Callbacks that the bot can call to control the trading system."""

    pause: Callable[[], Awaitable[None]]
    resume: Callable[[], Awaitable[None]]
    set_shadow: Callable[[bool], Awaitable[None]]
    set_risk_profile: Callable[[RiskProfile], Awaitable[None]]
    emergency_stop: Callable[[], Awaitable[None]]

    # Read-only state
    is_paused: Callable[[], bool]
    is_shadow: Callable[[], bool]
    current_profile: Callable[[], str]
    active_symbols: Callable[[], list[str]]
    regime_for: Callable[[str], str | None]  # symbol → regime string
    symbol_candidates: Callable[[], list[str]] | None = None
    selected_symbols: Callable[[], list[str]] | None = None
    toggle_symbol: Callable[[str], Awaitable[str]] | None = None
    signal_log: deque[SignalEntry] = field(default_factory=lambda: deque(maxlen=20))
    # Optional diagnostics provider (returns dict from TradingApplication.get_diagnostics)
    diagnostics_provider: Callable[[], dict[str, Any]] | None = None
    # Optional async DB diagnostics provider
    db_diagnostics_provider: Callable[[], Awaitable[dict[str, Any]]] | None = None
    start_training: Callable[[int, int, float], Awaitable[str]] | None = None
    start_training_all: Callable[[], Awaitable[str]] | None = None
    promote_model: Callable[[str], Awaitable[str]] | None = None
    runtime_settings: Callable[[], dict[str, Any]] | None = None
    set_runtime_setting: Callable[[str, Any], Awaitable[str]] | None = None
    # Safety gate: when False, Telegram cannot escalate to a riskier profile
    allow_risk_increase: bool = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TelegramBotConfig:
    """Runtime configuration for the Telegram bot."""

    token: str
    allowed_chat_ids: set[int]
    trading_mode: str
    risk_profile: str
    bybit_use_testnet: bool
    default_category: str = "linear"


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------


class TelegramMonitorBot:
    """Telegram bot with monitoring and operator control."""

    def __init__(
        self,
        config: TelegramBotConfig,
        health_provider: HealthProvider,
        adapter_factory: AdapterFactory,
        controller: TradingController | None = None,
    ) -> None:
        self._config = config
        self._health_provider = health_provider
        self._adapter_factory = adapter_factory
        self._controller = controller
        self._app: Application[Any, Any, Any, Any, Any, Any] | None = None

        # Pre-populate subscribed set so allowed chats receive push notifications
        # immediately after restart without requiring /start.
        self._subscribed: set[int] = set(config.allowed_chat_ids)

        # Pending confirmations: chat_id → (action_name, coroutine_factory)
        self._pending: dict[int, tuple[str, Callable[[], Awaitable[None]]]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self._config.token and self._config.allowed_chat_ids)

    async def start(self) -> None:
        if not self.enabled:
            log.info("telegram_bot_disabled")
            return

        app = Application.builder().token(self._config.token).build()

        # Observability
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("help", self._cmd_help))
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("balance", self._cmd_balance))
        app.add_handler(CommandHandler("positions", self._cmd_positions))
        app.add_handler(CommandHandler("signals", self._cmd_signals))
        app.add_handler(CommandHandler("regime", self._cmd_regime))
        app.add_handler(CommandHandler("symbols", self._cmd_symbols))
        app.add_handler(CommandHandler("pnl", self._cmd_pnl))
        app.add_handler(CommandHandler("net", self._cmd_net_results))
        app.add_handler(CommandHandler("diagnostics", self._cmd_diagnostics))
        app.add_handler(CommandHandler("canary", self._cmd_canary_ready))
        app.add_handler(CommandHandler("model_help", self._cmd_model_help))

        # Control
        app.add_handler(CommandHandler("pause", self._cmd_pause))
        app.add_handler(CommandHandler("resume", self._cmd_resume))
        app.add_handler(CommandHandler("shadow", self._cmd_shadow))
        app.add_handler(CommandHandler("mode", self._cmd_mode))
        app.add_handler(CommandHandler("risk", self._cmd_risk))
        app.add_handler(CommandHandler("train", self._cmd_train))
        app.add_handler(CommandHandler("limits", self._cmd_limits))
        app.add_handler(CommandHandler("stop", self._cmd_stop))
        app.add_handler(CommandHandler("confirm", self._cmd_confirm))
        app.add_handler(CallbackQueryHandler(self._on_button))

        await app.initialize()
        await app.start()
        if app.updater is None:
            raise RuntimeError("Telegram updater was not created")
        # allowed_updates=[] means "all types"; error_callback suppresses the
        # one-time Conflict error that fires during rolling redeploys on Render
        # (old instance is still alive for a few seconds while new one starts).
        await app.updater.start_polling(
            drop_pending_updates=True,
            error_callback=self._polling_error_callback,
        )
        self._app = app
        log.info("telegram_bot_started", allowed_chats=len(self._config.allowed_chat_ids))

    async def stop(self) -> None:
        if self._app is None:
            return
        app = self._app
        self._app = None
        if app.updater is not None:
            await app.updater.stop()
        await app.stop()
        await app.shutdown()
        log.info("telegram_bot_stopped")

    def _polling_error_callback(self, error: Exception) -> None:
        """Suppress Conflict errors during rolling redeploys; log the rest."""
        from telegram.error import Conflict, NetworkError

        if isinstance(error, Conflict):
            # Expected during Render rolling deploys — old instance displaced
            log.debug("telegram_polling_conflict_suppressed")
            return
        if isinstance(error, NetworkError):
            log.warning("telegram_polling_network_error", error=str(error))
            return
        log.error("telegram_polling_error", error=str(error))

    # ------------------------------------------------------------------
    # Push notifications (called by app.py)
    # ------------------------------------------------------------------

    async def notify(self, text: str) -> None:
        """Send a push message to all subscribed chats."""
        if self._app is None:
            return
        for chat_id in list(self._subscribed):
            try:
                await self._app.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
            except Exception as exc:
                log.warning("telegram_notify_failed", chat_id=chat_id, error=str(exc))

    async def notify_signal(self, entry: SignalEntry) -> None:
        icon = "🟢" if entry.side == "BUY" else "🔴"
        mode = "ТЕНЬ" if entry.shadow else "LIVE"
        text = (
            f"{icon} <b>Сигнал [{mode}]</b>\n"
            f"{entry.symbol} {entry.side} | уверенность: <code>{entry.confidence:.2f}</code>\n"
            f"Режим рынка: <code>{entry.regime}</code>\n"
            f"{entry.rationale}"
        )
        await self.notify(text)

    async def notify_position_opened(self, symbol: str, side: str, qty: Decimal, price: Decimal) -> None:
        icon = "🟢" if side == "BUY" else "🔴"
        await self.notify(f"{icon} <b>Позиция открыта</b>\n{symbol} {side} {qty} @ {price}")

    async def notify_position_closed(self, symbol: str, realized_pnl: Decimal) -> None:
        icon = "✅" if realized_pnl >= 0 else "❌"
        await self.notify(f"{icon} <b>Позиция закрыта</b>\n{symbol} PnL: <code>{realized_pnl:+.4f} USDT</code>")

    async def notify_circuit_breaker(self, breaker_type: str, reason: str) -> None:
        await self.notify(f"⚠️ <b>Защитный стоп</b>\nТип: <code>{breaker_type}</code>\nПричина: {reason}")

    async def notify_risk_changed(self, old_profile: str, new_profile: str) -> None:
        await self.notify(f"⚙️ <b>Риск-профиль изменен</b>\n{old_profile} → <code>{new_profile}</code>")

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    async def _authorised(self, update: Update) -> bool:
        chat = update.effective_chat
        if chat is None or chat.id not in self._config.allowed_chat_ids:
            if update.effective_message is not None:
                suffix = f" Chat ID: {chat.id}" if chat else ""
                await update.effective_message.reply_text(f"Доступ запрещен.{suffix}")
            log.warning("telegram_unauthorised_chat", chat_id=chat.id if chat else None)
            return False
        return True

    async def _reply(
        self,
        update: Update,
        text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> None:
        if update.effective_message is not None:
            await update.effective_message.reply_text(
                text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_markup=reply_markup,
            )

    def _chat_id(self, update: Update) -> int | None:
        return update.effective_chat.id if update.effective_chat else None

    def _main_menu(self) -> InlineKeyboardMarkup:
        rows = [
            [
                InlineKeyboardButton("📂 Позиции", callback_data="view:positions"),
                InlineKeyboardButton("🔎 Сканер", callback_data="view:symbols"),
            ],
            [InlineKeyboardButton("✅ Выбрать пары", callback_data="view:symbol_select")],
            [
                InlineKeyboardButton("📈 Результаты", callback_data="view:pnl"),
                InlineKeyboardButton("🧠 Почему нет сделок", callback_data="view:signals"),
            ],
            [
                InlineKeyboardButton("🗄 База и модель", callback_data="view:db_model"),
                InlineKeyboardButton("🖥 Нагрузка", callback_data="view:diagnostics"),
            ],
            [
                InlineKeyboardButton("⚙️ Управление", callback_data="view:control"),
                InlineKeyboardButton("🔄 Обновить", callback_data="view:status"),
            ],
        ]
        return InlineKeyboardMarkup(rows)

    def _control_menu(self) -> InlineKeyboardMarkup:
        """Control submenu — safe operations only (no risk escalation, no LIVE activation)."""
        rows = [
            [
                InlineKeyboardButton("⏸ Пауза", callback_data="control:pause"),
                InlineKeyboardButton("▶️ Возобновить", callback_data="control:resume"),
            ],
            [InlineKeyboardButton("🚫 LIVE заблокирован (только env vars)", callback_data="mode:active")],
            [
                InlineKeyboardButton("🧠 Обучить 500", callback_data="train:500:15:5"),
                InlineKeyboardButton("🧠 Обучить 1000", callback_data="train:1000:15:5"),
            ],
            [InlineKeyboardButton("🏆 Промоутировать кандидата → CHAMPION", callback_data="control:promote")],
            [
                InlineKeyboardButton("🎚 Лимиты", callback_data="control:limits"),
                InlineKeyboardButton("🗄 База и модель", callback_data="view:db_model"),
            ],
            [InlineKeyboardButton("🚦 Готовность CANARY", callback_data="view:canary")],
            [InlineKeyboardButton("❓ Как читать модель + путь к реальным деньгам", callback_data="view:model_help")],
            [InlineKeyboardButton("🚨 Аварийная остановка", callback_data="control:stop")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="view:status")],
        ]
        return InlineKeyboardMarkup(rows)

    def _canary_menu(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("🔄 Обновить готовность", callback_data="view:canary")],
                [
                    InlineKeyboardButton("🗄 База и модель", callback_data="view:db_model"),
                    InlineKeyboardButton("⬅️ Назад", callback_data="view:control"),
                ],
            ]
        )

    def _limits_menu(self) -> InlineKeyboardMarkup:
        rows = [
            [
                InlineKeyboardButton("Entries 1", callback_data="limit:entries:1"),
                InlineKeyboardButton("Entries 2", callback_data="limit:entries:2"),
            ],
            [
                InlineKeyboardButton("Pending 1", callback_data="limit:pending:1"),
                InlineKeyboardButton("Pending 2", callback_data="limit:pending:2"),
            ],
            [
                InlineKeyboardButton("Same-side 1", callback_data="limit:same_side:1"),
                InlineKeyboardButton("Same-side 2", callback_data="limit:same_side:2"),
            ],
            [
                InlineKeyboardButton("Price ≤10", callback_data="limit:price_cap:10"),
                InlineKeyboardButton("Price ≤25", callback_data="limit:price_cap:25"),
            ],
            [
                InlineKeyboardButton("Feature 10", callback_data="limit:feature_symbols:10"),
                InlineKeyboardButton("Feature 20", callback_data="limit:feature_symbols:20"),
            ],
            [
                InlineKeyboardButton("Exec 5", callback_data="limit:exec_candidates:5"),
                InlineKeyboardButton("Exec 10", callback_data="limit:exec_candidates:10"),
            ],
            [InlineKeyboardButton("⬅️ Управление", callback_data="view:control")],
        ]
        return InlineKeyboardMarkup(rows)

    def _symbol_select_menu(self, *, page: int = 0, page_size: int = 10) -> InlineKeyboardMarkup:
        candidates = self._symbol_candidates()
        selected = set(self._selected_symbols())
        total_pages = max(1, (len(candidates) + page_size - 1) // page_size)
        page = max(0, min(page, total_pages - 1))
        start = page * page_size
        rows: list[list[InlineKeyboardButton]] = []
        for symbol in candidates[start : start + page_size]:
            mark = "✅" if symbol in selected else "☐"
            rows.append([InlineKeyboardButton(f"{mark} {symbol}", callback_data=f"sym:toggle:{symbol}:{page}")])
        rows.append(
            [
                InlineKeyboardButton("◀️", callback_data=f"sym:page:{max(0, page - 1)}"),
                InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data=f"sym:page:{page}"),
                InlineKeyboardButton("▶️", callback_data=f"sym:page:{min(total_pages - 1, page + 1)}"),
            ]
        )
        rows.append(
            [
                InlineKeyboardButton("🔄 Обновить", callback_data=f"sym:page:{page}"),
                InlineKeyboardButton("⬅️ Меню", callback_data="view:status"),
            ]
        )
        return InlineKeyboardMarkup(rows)

    async def _button_reply(
        self,
        update: Update,
        text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> None:
        from telegram.error import BadRequest

        query = update.callback_query
        if query is not None and query.message is not None:
            try:
                await query.edit_message_text(
                    text,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                    reply_markup=reply_markup,
                )
                return
            except BadRequest as exc:
                log.debug("telegram.edit_message_failed_fallback_to_reply", error=str(exc))
            await query.message.reply_text(
                text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_markup=reply_markup,
            )
            return
        await self._reply(update, text, reply_markup=reply_markup)

    # ------------------------------------------------------------------
    # Observability commands
    # ------------------------------------------------------------------

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        cid = self._chat_id(update)
        if cid:
            self._subscribed.add(cid)
        await self._reply(update, self._help_text(), reply_markup=self._main_menu())

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        await self._reply(update, self._help_text(), reply_markup=self._main_menu())

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        try:
            health = await self._health_provider()
        except Exception as exc:
            await self._reply(update, f"<b>Статус</b>\nПроверка не прошла: <code>{exc}</code>")
            return

        ctrl = self._controller
        lines = [
            "<b>Статус системы</b>",
            f"Общее состояние: <code>{self._ru(health.overall)}</code>",
            f"Процесс: <code>{self._ru(health.system_status.value)}</code>",
            f"Режим торговли: <code>{health.trading_mode.value}</code>",
            f"Testnet: <code>{'да' if self._config.bybit_use_testnet else 'нет'}</code>",
        ]
        if ctrl:
            paused = " ⏸ пауза" if ctrl.is_paused() else ""
            shadow = " (тень, без ордеров)" if ctrl.is_shadow() else ""
            lines.append(f"Риск-профиль: <code>{ctrl.current_profile()}</code>{shadow}{paused}")
        else:
            lines.append(f"Риск-профиль: <code>{self._config.risk_profile}</code>")

        lines += [
            "",
            self._component_line("Postgres", health.postgres, health.postgres_latency_ms, required=True),
            self._component_line("Redis", health.redis, health.redis_latency_ms, required=False),
            self._component_line("Bybit REST", health.bybit_rest, health.bybit_rest_latency_ms, required=False),
            self._component_line("Bybit WS", health.bybit_ws, None, required=True),
            self._component_line("Признаки модели", health.features_fresh, None, required=True),
        ]
        if health.messages:
            lines += ["", "<b>Предупреждения</b>"]
            lines.extend(f"• {m}" for m in health.messages[:6])
        await self._reply(update, "\n".join(lines))

    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        adapter = self._adapter_factory()
        if adapter is None:
            await self._reply(update, "Подключение к Bybit еще не готово.")
            return
        try:
            bal: Balance = await adapter.get_balance()
        except Exception as exc:
            await self._reply(update, f"<b>Баланс</b>\nЗапрос к Bybit не прошел: <code>{exc}</code>")
            return
        lines = [
            "<b>Баланс Bybit UNIFIED</b>",
            f"Валюта: <code>{bal.currency}</code>",
            f"В кошельке: <code>{bal.wallet_balance}</code>",
            f"Доступно: <code>{bal.available_balance}</code>",
        ]
        if bal.margin_balance is not None:
            lines.append(f"Маржа: <code>{bal.margin_balance}</code>")
        if bal.unrealised_pnl:
            pnl_icon = "📈" if bal.unrealised_pnl >= 0 else "📉"
            lines.append(f"Плавающий PnL: {pnl_icon} <code>{bal.unrealised_pnl:+}</code>")
        await self._reply(update, "\n".join(lines))

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        adapter = self._adapter_factory()
        if adapter is None:
            await self._reply(update, "Подключение к Bybit еще не готово.")
            return
        try:
            positions: list[Position] = await adapter.get_positions(self._config.default_category)
        except Exception as exc:
            await self._reply(update, f"<b>Позиции</b>\nЗапрос к Bybit не прошел: <code>{exc}</code>")
            return
        open_pos = [p for p in positions if p.size > 0]
        if not open_pos:
            await self._reply(update, "<b>Позиции</b>\nОткрытых позиций нет.")
            return
        lines = [f"<b>Открытые позиции ({len(open_pos)})</b>"]
        for pos in open_pos[:10]:
            pnl_icon = "📈" if pos.unrealised_pnl >= 0 else "📉"
            side_icon = "🟢" if pos.side.value == "BUY" else "🔴"
            lines += [
                "",
                f"{side_icon} <b>{pos.symbol}</b> {pos.side.value}",
                f"  Размер: <code>{pos.size}</code>",
                f"  Вход: <code>{pos.entry_price}</code>",
                f"  Марк: <code>{pos.mark_price}</code>",
                f"  PnL:   {pnl_icon} <code>{pos.unrealised_pnl:+}</code>",
                f"  Плечо: <code>{pos.leverage}x</code>",
            ]
            if pos.liquidation_price:
                lines.append(f"  Ликвидация: <code>{pos.liquidation_price}</code>")
        if len(open_pos) > 10:
            lines.append(f"\n… еще {len(open_pos) - 10}")
        await self._reply(update, "\n".join(lines))

    async def _cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        if self._controller is None or not self._controller.signal_log:
            await self._reply(update, "<b>Сигналы</b>\nСигналов пока нет.")
            return
        lines = ["<b>Последние сигналы</b>"]
        for s in list(self._controller.signal_log)[-10:]:
            icon = "🟢" if s.side == "BUY" else "🔴"
            mode = "тень" if s.shadow else "live"
            ts = s.timestamp.strftime("%H:%M:%S")
            lines += [
                "",
                f"{icon} <b>{s.symbol}</b> {s.side} [{mode}] {ts}",
                f"  Уверенность: <code>{s.confidence:.2f}</code>  Режим рынка: <code>{s.regime}</code>",
                f"  {s.rationale[:80]}",
            ]
        await self._reply(update, "\n".join(lines))

    async def _cmd_regime(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        if self._controller is None:
            await self._reply(update, "Управляющий модуль недоступен.")
            return
        symbols = self._controller.active_symbols()
        if not symbols:
            await self._reply(update, "<b>Режим рынка</b>\nАктивных монет пока нет.")
            return
        regime_icons = {
            "BULL_TREND": "+",
            "BEAR_TREND": "-",
            "SIDEWAYS": "=",
            "HIGH_VOLATILITY": "!",
            "LOW_LIQUIDITY": ".",
            "EVENT_RISK": "!",
            "UNCERTAIN": "?",
        }
        lines = ["<b>Режим рынка по монетам</b>"]
        for sym in symbols:
            regime = self._controller.regime_for(sym) or "UNKNOWN"
            icon = regime_icons.get(regime, "*")
            lines.append(f"{icon} <code>{sym}</code>: {regime}")
        await self._reply(update, "\n".join(lines))

    async def _cmd_symbols(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        if self._controller is None:
            await self._reply(update, "Управляющий модуль недоступен.")
            return
        symbols = self._controller.active_symbols()
        if not symbols:
            await self._reply(update, "<b>Активные монеты</b>\nПока нет.")
            return
        lines = [f"<b>Активные монеты ({len(symbols)})</b>"]
        lines.extend(f"• <code>{s}</code>" for s in symbols)
        await self._reply(update, "\n".join(lines))

    async def _cmd_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        adapter = self._adapter_factory()
        if adapter is None:
            await self._reply(update, "Подключение к Bybit еще не готово.")
            return
        try:
            resp = await adapter._rest.get_closed_pnl(category=self._config.default_category, limit=20)
            records = resp.get("result", {}).get("list", [])
        except Exception as exc:
            await self._reply(update, f"<b>PnL</b>\nЗапрос к Bybit не прошел: <code>{exc}</code>")
            return
        if not records:
            await self._reply(update, "<b>Закрытый PnL</b>\nЗакрытых сделок пока нет.")
            return
        total = Decimal("0")
        shown = records[:20]
        lines = [f"<b>Закрытый PnL — последние {len(shown)} сделок</b>", ""]
        for r in shown:
            sym = r.get("symbol", "?")
            pnl = Decimal(str(r.get("closedPnl", "0")))
            total += pnl
            side_raw = str(r.get("side", "")).upper()
            side = "LONG" if side_raw == "BUY" else ("SHORT" if side_raw == "SELL" else side_raw or "?")
            qty = r.get("qty", "")
            entry = r.get("avgEntryPrice", "")
            exit_p = r.get("avgExitPrice", "")
            icon = "✅" if pnl >= 0 else "❌"
            price_part = f" @ {entry}→{exit_p}" if entry and exit_p else ""
            qty_part = f" ×{qty}" if qty else ""
            lines.append(f"{icon} <code>{sym}</code> {side}{qty_part}{price_part}  <b>{pnl:+.4f} USDT</b>")
        total_icon = "📈" if total >= 0 else "📉"
        lines.append(f"\n{total_icon} <b>Итого за {len(shown)} сделок:</b> <code>{total:+.4f} USDT</code>")
        lines.append(
            "\n💡 <i>LONG = куплено и закрыто; SHORT = продано и закрыто.\n"
            "Сумма — реализованный PnL по Bybit (без учёта незакрытых позиций).</i>"
        )
        await self._reply(update, "\n".join(lines))

    async def _cmd_net_results(self, update: Update, context: Any) -> None:
        """Show daily net P&L breakdown including fees and funding."""
        del context
        if not await self._authorised(update):
            return
        # Delegate to health provider for stats
        try:
            stats = await self._health_provider() if callable(self._health_provider) else {}
            net_stats = stats.get("net_results", {}) if stats else {}
        except Exception as exc:
            log.warning("telegram.net_results_failed", error=str(exc))
            net_stats = {}

        gross = net_stats.get("gross_pnl_usd", 0.0)
        fees = net_stats.get("total_fees_usd", 0.0)
        funding = net_stats.get("total_funding_usd", 0.0)
        slippage_est = net_stats.get("estimated_slippage_usd", 0.0)
        net = net_stats.get("net_pnl_usd", gross - fees - funding)
        maker_pct = net_stats.get("maker_fill_pct", 0.0)
        taker_pct = net_stats.get("taker_fill_pct", 100.0)
        fee_drag = abs(fees) + abs(funding) + abs(slippage_est)

        text = (
            "📈 <b>Чистый результат за сегодня</b>\n\n"
            f"Валовый PnL:      <code>{gross:+.4f} USDT</code>\n"
            f"Комиссии:         <code>{fees:+.4f} USDT</code>\n"
            f"Фандинг:          <code>{funding:+.4f} USDT</code>\n"
            f"Оценка проскальз.: <code>{slippage_est:+.4f} USDT</code>\n"
            f"─────────────────────────────\n"
            f"Чистый PnL:       <code>{net:+.4f} USDT</code>\n"
            f"Съели издержки:   <code>{fee_drag:.4f} USDT</code>\n\n"
            f"Maker исполнения: <code>{maker_pct:.1f}%</code>\n"
            f"Taker исполнения: <code>{taker_pct:.1f}%</code>"
        )
        await self._reply(update, text)

    async def _cmd_diagnostics(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        if self._controller is None or self._controller.diagnostics_provider is None:
            await self._reply(update, "<b>Диагностика</b>\nПока недоступна.")
            return
        try:
            diag = self._controller.diagnostics_provider()
        except Exception as exc:
            await self._reply(update, f"<b>Диагностика</b>\nОшибка: <code>{exc}</code>")
            return

        loop_at = diag.get("last_strategy_loop_at") or "никогда"
        ws_age = diag.get("last_ws_message_age_s")
        ws_str = self._age_label_ru(ws_age)
        symbols = diag.get("active_symbols") or []
        positions = diag.get("open_positions") or []
        heat = diag.get("portfolio_heat_pct")
        heat_str = f"{heat:.1f}%" if heat is not None else "n/a"

        lines = [
            "<b>Диагностика за последний час</b>",
            f"Цикл стратегии: <code>{loop_at}</code>",
            f"Последнее WS-сообщение: <code>{ws_str}</code>",
            f"Активные монеты: <code>{len(symbols)}</code>  {' '.join(symbols[:5])}{'…' if len(symbols) > 5 else ''}",
            f"Открытые позиции: <code>{len(positions)}</code>  {' '.join(positions[:5])}{'…' if len(positions) > 5 else ''}",
            f"Риск портфеля: <code>{heat_str}</code>",
            "",
            f"Сигналов создано:       <code>{diag.get('hour_signals_emitted', 0)}</code>",
            f"Отклонено риск-менедж.: <code>{diag.get('hour_risk_rejected', 0)}</code>",
            f"Отклонено Bybit API:    <code>{diag.get('hour_api_rejected', 0)}</code>",
            f"Малый размер заявки:    <code>{diag.get('hour_min_notional_rejected', 0)}</code>",
            f"Пропущено из-за позиции:<code>{diag.get('hour_skipped_open_position', 0)}</code>",
            f"Пропущено cooldown:     <code>{diag.get('hour_skipped_entry_cooldown', 0)}</code>",
            f"Пропущено после ошибки: <code>{diag.get('hour_skipped_failure_cooldown', 0)}</code>",
            f"Блоков фильтра модели:  <code>{diag.get('hour_model_gate_canary_blocked', 0)}</code>",
        ]
        await self._reply(update, "\n".join(lines))

    async def _cmd_canary_ready(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show whether the system is ready for a tiny CANARY_LIVE test."""
        del context
        if not await self._authorised(update):
            return

        db_diag: dict[str, Any] = {}
        if self._controller is not None and self._controller.db_diagnostics_provider is not None:
            try:
                db_diag = await self._controller.db_diagnostics_provider()
            except Exception as exc:
                db_diag = {"connected": False, "error": str(exc)}

        diag: dict[str, Any] = {}
        if self._controller is not None and self._controller.diagnostics_provider is not None:
            try:
                diag = self._controller.diagnostics_provider()
            except Exception as exc:
                diag = {"error": str(exc)}

        await self._reply(
            update,
            self._canary_readiness_text(db_diag=db_diag, diag=diag),
            reply_markup=self._canary_menu(),
        )

    def _canary_readiness_text(self, *, db_diag: dict[str, Any], diag: dict[str, Any]) -> str:
        checks: list[tuple[str, bool, str, str, str]] = []
        warnings: list[tuple[str, str]] = []
        estimates: list[str] = []

        def require(label: str, ok: bool, detail: str, fix: str, eta: str = "") -> None:
            checks.append((label, ok, detail, fix, eta))

        def warn_if(condition: bool, detail: str, fix: str) -> None:
            if condition:
                warnings.append((detail, fix))

        candles = db_diag.get("candles_by_interval", {}) or {}
        latest_1m = db_diag.get("latest_candle_1m")
        latest_age_s = self._utc_age_seconds(latest_1m)
        active_symbols = diag.get("active_symbols") or []
        runtime = self._controller.runtime_settings() if self._controller and self._controller.runtime_settings else {}
        latest_run = db_diag.get("latest_training_run", {}) or {}
        latest_model = db_diag.get("latest_model_version", {}) or {}
        gate = db_diag.get("shadow_gate_15m", {}) or {}
        paper = db_diag.get("paper_pnl_15m", {}) or {}
        paper_baseline = paper.get("baseline", {}) or {}
        paper_gate = paper.get("model_gate", {}) or {}

        trainable_15m = int(db_diag.get("labelled_samples_15m") or 0)
        prediction_outcomes = int(db_diag.get("prediction_outcomes") or 0)
        feature_snapshots = int(db_diag.get("feature_snapshots") or 0)
        gate_total = int(gate.get("total_count") or 0)
        gate_lift = gate.get("lift_vs_all_bps")
        paper_gate_count = int(paper_gate.get("count") or 0)
        paper_gate_bps = float(paper_gate.get("total_bps") or 0.0)
        paper_base_count = int(paper_baseline.get("count") or 0)
        ws_age = diag.get("last_ws_message_age_s")
        loop_at = diag.get("last_strategy_loop_at")
        hour_api_rejected = int(diag.get("hour_api_rejected") or 0)
        hour_min_notional = int(diag.get("hour_min_notional_rejected") or 0)
        is_shadow = self._controller.is_shadow() if self._controller is not None else True
        is_paused = self._controller.is_paused() if self._controller is not None else False

        active_count = max(1, len(active_symbols))
        one_m = int(candles.get("1") or 0)
        five_m = int(candles.get("5") or 0)
        fifteen_m = int(candles.get("15") or 0)
        one_h = int(candles.get("60") or 0)

        require(
            "База данных подключена",
            bool(db_diag.get("connected")),
            "хранилище доступно" if db_diag.get("connected") else "бот не видит Postgres",
            "На Render проверьте POSTGRES_DSN и включите внешний PostgreSQL/Supabase.",
        )
        require(
            "Свежая свеча 1m",
            latest_age_s is not None and latest_age_s <= 600,
            self._age_label_ru(latest_age_s),
            "Запустите сервис и проверьте Bybit WS/REST; данные должны обновиться за 1-2 минуты.",
            "1-2 минуты после восстановления WS",
        )
        require(
            "История 1m",
            one_m >= 1000,
            f"{one_m}/1000 свечей",
            "Запустите backfill: python -m trader.training.backfill --symbol BTCUSDT --interval 1 --days 2",
            self._eta_for_samples(1000 - one_m, active_count * 60),
        )
        require(
            "История 5m",
            five_m >= 200,
            f"{five_m}/200 свечей",
            "Запустите backfill с --interval 5 --days 2.",
            self._eta_for_samples(200 - five_m, active_count * 12),
        )
        require(
            "История 15m",
            fifteen_m >= 200,
            f"{fifteen_m}/200 свечей",
            "Запустите backfill с --interval 15 --days 4.",
            self._eta_for_samples(200 - fifteen_m, active_count * 4),
        )
        require(
            "История 1h",
            one_h >= 100,
            f"{one_h}/100 свечей",
            "Запустите backfill с --interval 60 --days 7.",
            self._eta_for_samples(100 - one_h, active_count),
        )
        require(
            "Активные монеты",
            len(active_symbols) >= 3,
            f"{len(active_symbols)}/3 монет",
            "Дайте сканеру обновиться или ослабьте фильтры ликвидности/цены в /limits.",
            "до 15 минут после запуска сканера",
        )
        require(
            "Снимки признаков",
            feature_snapshots >= 1000,
            f"{feature_snapshots}/1000 снимков",
            "Оставьте SHADOW включенным; бот будет сохранять признаки на каждом цикле стратегии.",
            self._eta_for_samples(1000 - feature_snapshots, active_count * 240),
        )
        require(
            "Размеченные примеры 15m",
            trainable_15m >= 1000,
            f"{trainable_15m}/1000 примеров",
            "Дождитесь закрытия 15m исходов или загрузите историю свечей backfill, затем запустите обучение.",
            self._eta_for_samples(1000 - trainable_15m, active_count * 4),
        )
        require(
            "Результаты прогнозов",
            prediction_outcomes >= 1000,
            f"{prediction_outcomes}/1000 исходов",
            "Проверьте задачу outcome-resolver: она должна сопоставлять сигналы с результатом через 15 минут.",
            self._eta_for_samples(1000 - prediction_outcomes, active_count * 4),
        )
        require(
            "Модель обучена",
            bool(latest_model.get("version")) or latest_run.get("status") == "COMPLETED",
            f"запуск={self._ru(latest_run.get('status', 'none'))}, модель={latest_model.get('version') or 'нет'}",
            "Нажмите 'Обучить 1000' или выполните python -m trader.training.train --min-samples 1000.",
            "10-20 минут после накопления 1000 примеров",
        )
        in_canary_live = self._config.trading_mode == "CANARY_LIVE"
        if in_canary_live:
            require(
                "Режим CANARY_LIVE активен",
                True,
                "бот торгует реальными деньгами через модель-фильтр",
                "",
            )
        else:
            require(
                "SHADOW включен",
                bool(is_shadow),
                "ордера пока не отправляются" if is_shadow else "бот уже не в тени",
                "Перед проверкой держите SHADOW включенным; CANARY включается только env vars на Render.",
            )
        require(
            "Пауза снята",
            not bool(is_paused),
            "сканер работает" if not is_paused else "новые входы остановлены",
            "Нажмите 'Возобновить', чтобы бот продолжил собирать сигналы и исходы.",
        )
        require(
            "WebSocket живой",
            ws_age is None or float(ws_age) <= 180,
            self._age_label_ru(ws_age),
            "Перезапустите сервис или проверьте доступ к Bybit WebSocket.",
            "1-3 минуты после восстановления соединения",
        )

        warn_if(
            trainable_15m < 2000,
            f"Размеченных 15m примеров {trainable_15m}; для более уверенного CANARY лучше 2000+.",
            "Можно начать с 1000 для первого кандидата, но перед реальными деньгами лучше добрать данные.",
        )
        warn_if(
            gate_total < 50,
            f"Фильтр модели проверен только на {gate_total} теневых решениях; желательно 50+.",
            "Оставьте SHADOW работать после обучения, чтобы модель набрала статистику pass/block.",
        )
        if gate_lift is not None:
            warn_if(
                float(gate_lift) <= 0,
                f"Lift фильтра модели {float(gate_lift):+.2f} bps; фильтр пока не улучшает отбор.",
                "Не включайте MODEL_GATE_CANARY_ENABLED, пока lift не станет положительным.",
            )
        else:
            warnings.append(("Lift фильтра модели еще не измерен.", "Дождитесь исходов после обучения модели."))
        warn_if(
            paper_gate_count < 20,
            f"Paper model-gate сделок {paper_gate_count}; желательно 20+.",
            "Дайте модели поработать в тени.",
        )
        warn_if(
            paper_gate_count >= 20 and paper_gate_bps <= 0,
            f"Paper model-gate PnL {paper_gate_bps:+.1f} bps.",
            "Не промоутируйте модель, пока бумажный результат фильтра отрицательный.",
        )
        warn_if(
            paper_base_count == 0,
            "У baseline пока нет завершенных бумажных сделок.",
            "Дайте стратегии накопить исходы.",
        )
        warn_if(
            hour_api_rejected > 0,
            f"Bybit отклонил {hour_api_rejected} действий за последний час.",
            "Проверьте min notional, плечо и права API ключа перед CANARY.",
        )
        warn_if(
            hour_min_notional > 0,
            f"{hour_min_notional} заявок отклонены из-за минимального размера.",
            "Увеличьте минимальный notional или ограничьте монеты с маленькой ценой.",
        )
        warn_if(
            bool(runtime.get("model_gate_canary_enabled")),
            "Model gate canary уже включен.",
            "Для первой проверки реальных денег лучше держать модель в режиме наблюдения.",
        )
        diag_error = diag.get("error")
        db_error = db_diag.get("error") or db_diag.get("last_connect_error") or db_diag.get("last_read_error")
        warn_if(
            bool(diag_error),
            f"Ошибка runtime диагностики: {html.escape(str(diag_error))}",
            "Посмотрите логи сервиса на Render.",
        )
        warn_if(
            bool(db_error),
            f"Ошибка DB диагностики: {html.escape(str(db_error))}",
            "Проверьте POSTGRES_DSN и pgbouncer режим.",
        )
        if loop_at in (None, "never"):
            warnings.append(
                ("Нет времени последнего цикла стратегии.", "Проверьте, что strategy-loop запущен и не падает.")
            )

        failed = [item for item in checks if not item[1]]
        passed_count = len(checks) - len(failed)
        if failed:
            if in_canary_live:
                status = f"⚠️ CANARY_LIVE активен, но {len(failed)} условий нарушено"
            else:
                status = f"❌ НЕ ГОТОВО — {passed_count} из {len(checks)} условий выполнено"
        elif warnings:
            if in_canary_live:
                status = f"✅ CANARY_LIVE работает — {len(warnings)} предупреждений (некритично)"
            else:
                status = f"⚠️ ПОЧТИ — мешают только предупреждения: {len(warnings)}"
        else:
            if in_canary_live:
                status = "✅ CANARY_LIVE работает штатно"
            else:
                status = "✅ ГОТОВО — можно включать CANARY_LIVE"

        for _label, _ok, _detail, _fix, eta in failed:
            if eta:
                estimates.append(f"• {_label}: {eta}")
        if not estimates and warnings:
            estimates.append("• Основные условия закрыты; нужно добрать только качество/статистику модели.")
        if not estimates:
            estimates.append("• Технически готово сейчас.")

        required_env = {
            "POSTGRES_DSN": "хранилище сделок и модели",
            "BYBIT_API_KEY": "доступ к Bybit",
            "BYBIT_API_SECRET": "подпись заявок Bybit",
            "TELEGRAM_BOT_TOKEN": "бот в Telegram",
            "TELEGRAM_ALLOWED_CHAT_IDS": "доступ только вашему чату",
            "TRADING_MODE": "поставить CANARY_LIVE при запуске реальных денег",
            "LIVE_MODE": "поставить true при запуске реальных денег",
            "LIVE_ARMED": "поставить true как вторую защиту live",
            "BYBIT_USE_TESTNET": "поставить false для реального Bybit",
        }
        missing_env = [name for name in required_env if not os.getenv(name)]

        lines = [
            "<b>🚦 Готовность к реальным деньгам</b>",
            f"<b>{status}</b>",
            "",
            "<b>Обязательные условия</b>",
        ]
        for label, ok, detail, fix, _eta in checks:
            icon = "✅" if ok else "❌"
            lines.append(f"{icon} {label}: <code>{html.escape(str(detail))}</code>")
            if not ok:
                lines.append(f"   Как исправить: {html.escape(fix)}")
        lines.extend(["", "<b>Предупреждения</b>"])
        if warnings:
            for warning, fix in warnings[:8]:
                lines.append(f"⚠️ {warning}")
                lines.append(f"   Как исправить: {html.escape(fix)}")
        else:
            lines.append("✅ Предупреждений нет.")
        lines.extend(
            [
                "",
                "<b>⏱ Примерно до готовности</b>",
                *estimates[:6],
                "",
                "<b>🔑 Что нужно сделать вручную на Render</b>",
            ]
        )
        if missing_env:
            lines.extend(
                f"• <code>{name}</code> — {desc}" for name, desc in required_env.items() if name in missing_env
            )
        else:
            lines.append("✅ Все ключевые env vars уже заданы в текущем окружении.")
        lines.extend(
            [
                "",
                "<b>Следующий шаг</b>",
                "CANARY держим маленьким: 1-2 позиции, минимальный notional, модель сначала только наблюдает.",
                "Telegram не включает live: реальные деньги включаются только через env vars на Render.",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _utc_age_seconds(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        if not isinstance(value, datetime):
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return max(0.0, (datetime.now(UTC) - value.astimezone(UTC)).total_seconds())

    @staticmethod
    def _age_label(age_s: Any) -> str:
        if age_s is None:
            return "unknown"
        try:
            seconds = float(age_s)
        except (TypeError, ValueError):
            return "unknown"
        if seconds < 120:
            return f"{seconds:.0f}s ago"
        return str(timedelta(seconds=int(seconds)))

    @staticmethod
    def _age_label_ru(age_s: Any) -> str:
        if age_s is None:
            return "нет данных"
        try:
            seconds = float(age_s)
        except (TypeError, ValueError):
            return "нет данных"
        if seconds < 120:
            return f"{seconds:.0f} сек назад"
        if seconds < 7200:
            return f"{seconds / 60:.1f} мин назад"
        return f"{seconds / 3600:.1f} ч назад"

    @staticmethod
    def _eta_for_samples(missing: int, per_hour: int | float) -> str:
        if missing <= 0:
            return "готово"
        if per_hour <= 0:
            return "темп неизвестен"
        hours = missing / float(per_hour)
        if hours < 1:
            return "меньше часа при текущем темпе"
        if hours < 48:
            return f"примерно {hours:.1f} ч при текущем темпе"
        return f"примерно {hours / 24:.1f} дн при текущем темпе"

    @staticmethod
    def _ru(value: Any) -> str:
        text = str(value or "none")
        return _STATUS_RU.get(text, text)

    async def _cmd_model_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Explain model/training screens in operator-friendly Russian."""
        del context
        if not await self._authorised(update):
            return
        await self._reply(update, self._model_help_text(), reply_markup=self._control_menu())

    def _model_help_text(self) -> str:
        return (
            "<b>❓ Что означают метрики и как добраться до реальных денег</b>\n\n"
            "<b>Метрики модели (в «База и модель»)</b>\n\n"
            "• <b>Precision (точность)</b> — сколько % сигналов, пропущенных моделью, оказались прибыльными.\n"
            "  Хорошо: ≥ 55%. Ваш baseline ~40-45%.\n\n"
            "• <b>Lift против baseline</b> — на сколько bps прибыльнее сигналы модели по сравнению со средним.\n"
            "  Нужно > 0 bps. Хорошо: > 3 bps.\n\n"
            "• <b>Walk-forward ожидание</b> — ожидаемая прибыль на сигнал (уже с учётом комиссий и порога).\n"
            "  Нужно > 0 bps для промоута. Хорошо: > 3 bps.\n\n"
            "• <b>Lift фильтра (Gate lift)</b> — улучшает ли модель отбор сигналов В РЕАЛЬНОМ ТЕНЕВОМ РЕЖИМЕ.\n"
            "  Показывает n/a или 0/0, пока новая модель не оценила ~50 живых сигналов.\n"
            "  ⏳ Это нормально — накопится автоматически за 1-2 часа работы бота.\n\n"
            "• <b>Paper baseline / Paper gate</b> — бумажный счёт: что было бы если бы бот торговал.\n"
            "  Baseline = все сигналы, Gate = только пропущенные моделью.\n\n"
            "<b>🛣 Путь к реальным деньгам (CANARY)</b>\n\n"
            "<b>Шаг 1 — Накопить данные</b>\n"
            "✓ Примеры 15m ≥ 1000 (сейчас уже есть)\n\n"
            "<b>Шаг 2 — Обучить модель</b>\n"
            "✓ Нажать «Обучить ВСЕ» или «Обучить 1000»\n"
            "✓ Качество = ХОРОШО, Walk-forward > 0 bps\n\n"
            "<b>Шаг 3 — Промоутировать кандидата</b>\n"
            "✓ Нажать «🏆 Промоутировать кандидата → CHAMPION»\n"
            "✓ После этого модель начнёт оценивать живые сигналы\n\n"
            "<b>Шаг 4 — Дать модели поработать в тени (~2-4 часа)</b>\n"
            "✓ Gate lift > 0 bps (на ≥ 50 сигналах)\n"
            "✓ Paper gate: ≥ 20 бумажных сделок с положительным PnL\n\n"
            "<b>Шаг 5 — Включить CANARY на Render</b>\n"
            "Переменные окружения:\n"
            "<code>TRADING_MODE=CANARY_LIVE\n"
            "LIVE_ARMED=true</code>\n"
            "CANARY торгует минимальным размером (5-10 USDT) чтобы проверить механику.\n\n"
            "<b>Важно:</b> Кнопка «Shadow ON» в боте — это не настоящий Shadow. "
            "Реальное включение только через Render env vars."
        )

    async def _cmd_db_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """🗄 БАЗА И МОДЕЛЬ screen."""
        del context
        if not await self._authorised(update):
            return

        db_diag: dict[str, Any] = {}
        if self._controller is not None and self._controller.db_diagnostics_provider is not None:
            try:
                db_diag = await self._controller.db_diagnostics_provider()
            except Exception as exc:
                db_diag = {"error": str(exc)}

        diag: dict[str, Any] = {}
        if self._controller is not None and self._controller.diagnostics_provider is not None:
            try:
                diag = self._controller.diagnostics_provider()
            except Exception as _diag_exc:
                log.debug("telegram.diagnostics_provider_failed", error=str(_diag_exc))

        connected = db_diag.get("connected", False)
        db_icon = "🟢" if connected else "🔴"
        configured = db_diag.get("configured", True)
        db_status = "подключена" if connected else ("настроена, переподключается" if configured else "не настроена")
        db_error = db_diag.get("error") or db_diag.get("last_connect_error") or db_diag.get("last_read_error")
        db_error_str = html.escape(str(db_error)) if db_error else ""
        if len(db_error_str) > 180:
            db_error_str = f"{db_error_str[:177]}..."
        candles = db_diag.get("candles_by_interval", {})
        latest_1m = db_diag.get("latest_candle_1m")
        latest_str = latest_1m.strftime("%H:%M:%S UTC") if latest_1m else "нет"
        outcomes_by_horizon = db_diag.get("prediction_outcomes_by_horizon", {}) or {}
        labelled_15m = db_diag.get("labelled_samples_15m", 0)
        outcome_parts = [
            f"{horizon}m={count}"
            for horizon, count in sorted(outcomes_by_horizon.items(), key=lambda item: int(item[0]))
        ]
        outcome_breakdown = ", ".join(outcome_parts) if outcome_parts else "нет"

        model_info = diag.get("model", {}) or {}
        db_model = db_diag.get("latest_model_version", {}) or {}
        latest_run = db_diag.get("latest_training_run", {}) or {}
        model_metrics = db_model.get("metrics") or latest_run.get("metrics") or {}
        if isinstance(model_metrics, str):
            try:
                model_metrics = json.loads(model_metrics)
            except json.JSONDecodeError:
                model_metrics = {}
        champion_ver = model_info.get("champion_version", "none")
        challenger_ver = model_info.get("challenger_version", "none")
        last_training = model_info.get("last_training", "never")
        samples = model_info.get("training_samples", 0)
        if not samples and db_model:
            samples = db_model.get("training_samples", 0)
        db_model_version = db_model.get("version")
        db_model_status = db_model.get("status")
        if champion_ver == "none" and db_model_status == "CHAMPION":
            champion_ver = db_model_version or "none"
        if challenger_ver == "none" and db_model_status and db_model_status != "CHAMPION":
            challenger_ver = db_model_version or "none"
        if last_training == "never" and latest_run.get("finished_at"):
            last_training = latest_run["finished_at"].strftime("%Y-%m-%d %H:%M UTC")
        wf_exp = model_info.get("walk_forward_expectancy", "n/a")
        drift = model_info.get("drift_status", "n/a")
        latest_run_status = latest_run.get("status", "none")
        latest_run_samples = latest_run.get("sample_count", 0) or 0
        latest_run_model = latest_run.get("model_version") or "none"
        latest_run_error = str(latest_run.get("error") or "")
        if len(latest_run_error) > 120:
            latest_run_error = f"{latest_run_error[:117]}..."
        latest_run_error = html.escape(latest_run_error)
        model_quality = model_metrics.get("quality", "n/a")
        validation_samples = model_metrics.get("validation_samples", "n/a")
        precision = model_metrics.get("precision")
        lift_bps = model_metrics.get("lift_bps")
        expectancy_bps = model_metrics.get("walk_forward_expectancy_bps")
        best_threshold = model_metrics.get("best_threshold")
        best_threshold_avg = model_metrics.get("best_threshold_avg_net_return_bps")
        precision_str = f"{float(precision):.1%}" if precision is not None else "n/a"
        lift_str = f"{float(lift_bps):+.2f} bps" if lift_bps is not None else "n/a"
        expectancy_str = f"{float(expectancy_bps):+.2f} bps" if expectancy_bps is not None else "n/a"
        best_threshold_str = f"{float(best_threshold):.2f}" if best_threshold is not None else "n/a"
        best_threshold_avg_str = f"{float(best_threshold_avg):+.2f} bps" if best_threshold_avg is not None else "n/a"
        gate = db_diag.get("shadow_gate_15m", {}) or {}
        gate_total = gate.get("total_count", 0) or 0
        gate_pass = gate.get("pass_count", 0) or 0
        gate_block = gate.get("block_count", 0) or 0
        gate_pass_avg = gate.get("pass_avg_net_return_bps")
        gate_block_avg = gate.get("block_avg_net_return_bps")
        gate_lift = gate.get("lift_vs_all_bps")
        gate_reasons = gate.get("top_block_reasons", {}) or {}
        gate_pass_avg_str = f"{float(gate_pass_avg):+.2f} bps" if gate_pass_avg is not None else "n/a"
        gate_block_avg_str = f"{float(gate_block_avg):+.2f} bps" if gate_block_avg is not None else "n/a"
        gate_lift_str = f"{float(gate_lift):+.2f} bps" if gate_lift is not None else "n/a"
        gate_reasons_str = (
            ", ".join(f"{html.escape(str(reason))}:{count}" for reason, count in gate_reasons.items()) or "n/a"
        )
        paper = db_diag.get("paper_pnl_15m", {}) or {}
        paper_notional = float(db_diag.get("paper_notional_usd") or 5.0)
        paper_baseline = paper.get("baseline", {}) or {}
        paper_gate = paper.get("model_gate", {}) or {}

        def _paper_line(stats: dict[str, Any]) -> str:
            total_bps = float(stats.get("total_bps") or 0.0)
            drawdown_bps = float(stats.get("max_drawdown_bps") or 0.0)
            usd = paper_notional * total_bps / 10000.0
            dd_usd = paper_notional * drawdown_bps / 10000.0
            return (
                f"{int(stats.get('count') or 0)} сделок, {total_bps:+.1f} bps "
                f"(примерно {usd:+.3f} USDT), просадка {dd_usd:+.3f}"
            )

        data_note = "данные собираются" if connected and labelled_15m >= 1000 else "мало данных для уверенного обучения"
        training_note = "модель-кандидат обучена" if db_model_version else "модель еще не обучена"
        if gate_lift is None:
            gate_note = "фильтр модели еще не оценен"
        elif float(gate_lift) > 0:
            gate_note = "фильтр модели улучшает отбор сигналов"
        else:
            gate_note = "фильтр модели пока НЕ улучшает отбор сигналов"

        lines = [
            "<b>🗄 БАЗА И МОДЕЛЬ</b>",
            "",
            f"БД: {db_icon} {db_status}",
            f"Ошибка БД: <code>{db_error_str or 'нет'}</code>",
            f"Последняя свеча 1m: <code>{latest_str}</code>",
            f"Свечей 1m:  <code>{candles.get('1', 0)}</code>",
            f"Свечей 5m:  <code>{candles.get('5', 0)}</code>",
            f"Свечей 15m: <code>{candles.get('15', 0)}</code>",
            f"Свечей 1h:  <code>{candles.get('60', 0)}</code>",
            f"Снимки признаков: <code>{db_diag.get('feature_snapshots', 0)}</code>",
            f"Размеченные исходы: <code>{db_diag.get('prediction_outcomes', 0)}</code>",
            f"Горизонты разметки: <code>{outcome_breakdown}</code>",
            f"Готово для обучения 15m: <code>{labelled_15m}</code>",
            "",
            "<b>Простыми словами</b>",
            f"Данные: <code>{data_note}</code>",
            f"Обучение: <code>{training_note}</code>",
            f"Оценка модели: <code>{gate_note}</code>",
            "Реальные сделки: <code>модель не управляет ордерами</code>",
            "",
            "<b>Модель</b>",
            f"Последнее обучение: <code>{last_training}</code>",
            f"Обучающих примеров: <code>{samples}</code>",
            f"Основная модель: <code>{champion_ver}</code>",
            f"Кандидат: <code>{challenger_ver}</code>",
            f"Последняя модель в БД: <code>{db_model_version or 'нет'}</code> <code>{self._ru(db_model_status)}</code>",
            f"Последний запуск: <code>{self._ru(latest_run_status)}</code>, примеров=<code>{latest_run_samples}</code>",
            f"Версия из запуска: <code>{latest_run_model}</code>",
            f"Качество: <code>{self._ru(model_quality)}</code>",
            f"Проверочных примеров: <code>{validation_samples}</code>",
            f"Точность прибыльных сигналов: <code>{precision_str}</code>",
            f"Улучшение против baseline: <code>{lift_str}</code>",
            f"Лучший порог модели: <code>{best_threshold_str}</code>, среднее=<code>{best_threshold_avg_str}</code>",
            f"Ожидание walk-forward: <code>{expectancy_str if expectancy_bps is not None else wf_exp}</code>",
            f"Фильтр модели 15m: <code>{gate_pass}/{gate_total} пропущено</code>, блок=<code>{gate_block}</code>",
            f"Среднее пропущенных: <code>{gate_pass_avg_str}</code>",
            f"Среднее заблокированных: <code>{gate_block_avg_str}</code>",
            "Lift фильтра: <code>"
            + ("⏳ ждём ~50 живых сигналов" if gate_total == 0 and db_model_version else gate_lift_str)
            + "</code>",
            f"Причины блоков: <code>{gate_reasons_str}</code>",
            f"Paper baseline: <code>{_paper_line(paper_baseline)}</code>",
            f"Paper model gate: <code>{_paper_line(paper_gate)}</code>",
            f"Дрифт данных: <code>{self._ru(drift)}</code>",
            "Решения модели в live: <b>выключены</b>",
            "",
            "<i>bps = 0.01%. Precision = % прибыльных среди пропущенных моделью. "
            "Lift = насколько пропущенные лучше среднего по всем сигналам.</i>",
            "",
        ]

        # ── Roadmap к реальным деньгам ──────────────────────────────────
        lines.append("<b>📋 Путь к реальным сделкам (CANARY)</b>")
        # 1. Данные
        lbl_ok = int(labelled_15m or 0) >= 1000
        lines.append(f"{'✅' if lbl_ok else '❌'} Данных 15m ≥ 1000 → сейчас: <code>{labelled_15m}</code>")
        # 2. Модель обучена, quality GOOD
        trained_ok = bool(db_model_version) and model_quality in ("GOOD", "ХОРОШО")
        lines.append(
            f"{'✅' if trained_ok else '❌'} Качество модели = ХОРОШО → "
            f"<code>{self._ru(model_quality) if db_model_version else 'не обучена'}</code>"
        )
        # 3. Walk-forward > 0
        wfe_val = float(expectancy_bps) if expectancy_bps is not None else None
        wfe_ok = wfe_val is not None and wfe_val > 0
        lines.append(
            f"{'✅' if wfe_ok else '❌'} Walk-forward > 0 bps → "
            f"<code>{expectancy_str if expectancy_bps is not None else 'n/a'}</code>"
        )
        # 4. Champion
        champ_ok = champion_ver not in ("none", "", None)
        lines.append(
            f"{'✅' if champ_ok else '❌'} Основная модель = CHAMPION → "
            f"<code>{'есть: ' + str(champion_ver) if champ_ok else 'нет → нажмите «Промоутировать»'}</code>"
        )
        # 5. Gate lift
        if gate_total == 0 and db_model_version:
            gate_road_icon = "⏳"
            gate_road_val = f"ждём ~50 сигналов (сейчас {gate_total})"
        elif gate_lift is not None and float(gate_lift) > 0:
            gate_road_icon = "✅"
            gate_road_val = gate_lift_str
        else:
            gate_road_icon = "❌" if gate_total > 0 else "⏳"
            gate_road_val = gate_lift_str if gate_lift is not None else "n/a"
        lines.append(f"{gate_road_icon} Lift фильтра > 0 bps (≥50 сигналов) → <code>{gate_road_val}</code>")
        # 6. Paper gate
        paper_gate_count = int(paper_gate.get("count") or 0)
        paper_gate_bps_val = float(paper_gate.get("total_bps") or 0.0)
        if paper_gate_count < 20:
            paper_road_icon = "⏳"
            paper_road_val = f"ждём 20 бумажных сделок (сейчас {paper_gate_count})"
        elif paper_gate_bps_val > 0:
            paper_road_icon = "✅"
            paper_road_val = f"{paper_gate_count} сделок, {paper_gate_bps_val:+.1f} bps"
        else:
            paper_road_icon = "❌"
            paper_road_val = f"{paper_gate_count} сделок, {paper_gate_bps_val:+.1f} bps (нужен > 0)"
        lines.append(f"{paper_road_icon} Paper gate ≥ 20 сделок > 0 bps → <code>{paper_road_val}</code>")

        all_done = all([lbl_ok, trained_ok, wfe_ok, champ_ok])
        if (
            all_done
            and gate_total >= 50
            and gate_lift is not None
            and float(gate_lift) > 0
            and paper_gate_count >= 20
            and paper_gate_bps_val > 0
        ):
            lines.append("\n🚀 <b>Все условия выполнены!</b> Можно включать CANARY на Render.")
        else:
            lines.append("\n💡 <i>Нажмите «❓ Как читать модель» для пошагового руководства.</i>")

        if latest_run_error:
            lines.append(f"Ошибка последнего обучения: <code>{latest_run_error}</code>")
        if db_diag.get("error"):
            lines.append(f"\n<i>Ошибка: {html.escape(str(db_diag['error']))}</i>")

        await self._reply(update, "\n".join(lines), reply_markup=self._main_menu())

    # ------------------------------------------------------------------
    # Control commands
    # ------------------------------------------------------------------

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        if self._controller is None:
            await self._reply(update, "Управление сейчас недоступно.")
            return
        await self._controller.pause()
        await self._reply(update, "⏸ Бот на <b>паузе</b>: новые входы не открываются.\nКоманда /resume снимет паузу.")

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        if self._controller is None:
            await self._reply(update, "Управление сейчас недоступно.")
            return
        if not self._controller.is_paused():
            await self._reply(update, "Бот не на паузе.")
            return
        await self._controller.resume()
        await self._reply(update, "▶️ Бот <b>возобновлен</b>: сбор данных и новые входы снова разрешены.")

    async def _cmd_shadow(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorised(update):
            return
        if self._controller is None:
            await self._reply(update, "Управление сейчас недоступно.")
            return
        args = context.args or []
        if not args or args[0].lower() not in ("on", "off"):
            current = "on" if self._controller.is_shadow() else "off"
            await self._reply(
                update, f"Shadow mode is currently <code>{current}</code>.\nUse: /shadow on  or  /shadow off"
            )
            return
        enable = args[0].lower() == "on"
        if not enable:
            # Disabling shadow mode via Telegram is blocked.
            # CANARY_LIVE activation requires env-var change (TRADING_MODE, LIVE_MODE, LIVE_ARMED).
            await self._reply(
                update,
                "<b>Выключение SHADOW заблокировано.</b>\n"
                "Реальные деньги включаются только через переменные окружения "
                "(TRADING_MODE=CANARY_LIVE, LIVE_MODE=true, LIVE_ARMED=true). "
                "Telegram не может включить live-торговлю.",
            )
            return
        await self._controller.set_shadow(True)
        await self._reply(update, "🔦 Теневой режим: <code>включен, ордера считаются, но не отправляются</code>")

    async def _cmd_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorised(update):
            return
        if self._controller is None:
            await self._reply(update, "Управление сейчас недоступно.")
            return
        args = context.args or []
        if not args or args[0].lower() not in ("shadow", "active"):
            current = "тень" if self._controller.is_shadow() else "активный"
            venue = "testnet" if self._config.bybit_use_testnet else "реальный Bybit"
            await self._reply(
                update,
                f"Текущий режим исполнения: <code>{current}</code> на <code>{venue}</code>.\n"
                "Команды: /mode shadow или /mode active",
            )
            return
        if args[0].lower() == "shadow":
            await self._controller.set_shadow(True)
            await self._reply(update, "Режим: <b>SHADOW</b>. Ордера не отправляются.")
            return
        # "active" is blocked — requires env-var change
        await self._reply(
            update,
            "<b>Включение CANARY_LIVE заблокировано из Telegram.</b>\n"
            "Live-режим требует переменных окружения: "
            "TRADING_MODE=CANARY_LIVE, LIVE_MODE=true, LIVE_ARMED=true.\n"
            "Telegram не включает реальные деньги.",
        )

    async def _cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorised(update):
            return
        if self._controller is None:
            await self._reply(update, "Управление сейчас недоступно.")
            return
        args = context.args or []
        valid = [p.value.lower() for p in RiskProfile]
        if not args or args[0].lower() not in valid:
            current = self._controller.current_profile()
            await self._reply(
                update, f"Текущий риск-профиль: <code>{current}</code>\nФормат: /risk {' | '.join(valid)}"
            )
            return
        new_profile_str = args[0].upper()
        new_profile = RiskProfile(new_profile_str)
        old_profile = self._controller.current_profile()
        if new_profile_str == old_profile:
            await self._reply(update, f"Риск-профиль уже <code>{old_profile}</code>.")
            return

        # Block escalation to a riskier profile unless explicitly allowed.
        old_level = _RISK_LEVEL.get(old_profile, 0)
        new_level = _RISK_LEVEL.get(new_profile_str, 0)
        if new_level > old_level and not self._controller.allow_risk_increase:
            await self._reply(
                update,
                f"🚫 <b>Повышение риска заблокировано</b>: <code>{old_profile}</code> → <code>{new_profile_str}</code>\n"
                "Чтобы разрешить, задайте <code>TELEGRAM_ALLOW_RISK_INCREASE=true</code> на Render.\n"
                "После изменения нужен перезапуск сервиса.",
            )
            log.warning(
                "telegram.risk_escalation_blocked",
                old=old_profile,
                new=new_profile_str,
            )
            return

        cid = self._chat_id(update)
        if cid:
            self._pending[cid] = (
                f"сменить риск-профиль с {old_profile} на {new_profile_str}",
                lambda: self._controller.set_risk_profile(new_profile),  # type: ignore[union-attr]
            )
        await self._reply(
            update,
            f"⚠️ Сменить риск-профиль: <code>{old_profile}</code> → <code>{new_profile_str}</code>\n"
            "Отправьте /confirm для применения или ничего не делайте для отмены.",
        )

    async def _cmd_train(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorised(update):
            return
        if self._controller is None or self._controller.start_training is None:
            await self._reply(update, "Запуск обучения сейчас недоступен.")
            return
        args = context.args or []
        try:
            min_samples = int(args[0]) if len(args) >= 1 else 500
            horizon = int(args[1]) if len(args) >= 2 else 15
            label_bps = float(args[2]) if len(args) >= 3 else 5.0
        except ValueError:
            await self._reply(update, "Формат: /train [примеров] [горизонт_минут] [порог_bps]")
            return
        if min_samples < 50 or horizon <= 0 or label_bps < 0:
            await self._reply(update, "Параметры обучения отклонены: примеров>=50, горизонт>0, bps>=0.")
            return
        try:
            msg = await self._controller.start_training(min_samples, horizon, label_bps)
        except Exception as exc:
            await self._reply(update, f"❌ Обучение не стартовало: <code>{exc}</code>")
            return
        await self._reply(update, msg, reply_markup=self._main_menu())

    async def _cmd_limits(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorised(update):
            return
        if self._controller is None or self._controller.runtime_settings is None:
            await self._reply(update, "Runtime-настройки сейчас недоступны.")
            return
        args = context.args or []
        if not args:
            await self._reply(update, self._limits_text(), reply_markup=self._control_menu())
            return
        if len(args) != 2 or self._controller.set_runtime_setting is None:
            await self._reply(
                update,
                "Формат: /limits entries|pending|same_side|price_cap|feature_symbols|exec_candidates N\n"
                "Фильтр модели: /limits model_gate on|off, /limits model_gate_threshold 0.60",
            )
            return
        key = args[0].lower()
        raw_value = args[1]
        try:
            if key in {"price_cap", "model_gate_threshold"}:
                value: Any = float(raw_value)
            elif key == "model_gate":
                value = raw_value
            else:
                value = int(raw_value)
            msg = await self._controller.set_runtime_setting(key, value)
        except Exception as exc:
            await self._reply(update, f"❌ Изменение лимита отклонено: <code>{exc}</code>")
            return
        await self._reply(update, f"✅ {msg}\n\n{self._limits_text()}", reply_markup=self._control_menu())

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        if self._controller is None:
            await self._reply(update, "Управление сейчас недоступно.")
            return
        cid = self._chat_id(update)
        if cid:
            self._pending[cid] = (
                "АВАРИЙНАЯ ОСТАНОВКА (нужен ручной перезапуск)",
                self._controller.emergency_stop,
            )
        await self._reply(
            update,
            "🚨 Запрошена <b>аварийная остановка</b>.\n"
            "Новые входы будут полностью остановлены, затем нужен ручной перезапуск.\n"
            "Отправьте /confirm для выполнения или ничего не делайте для отмены.",
        )

    async def _cmd_confirm(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        cid = self._chat_id(update)
        if cid is None or cid not in self._pending:
            await self._reply(update, "Нет действия, которое ждет подтверждения.")
            return
        action_name, action_fn = self._pending.pop(cid)
        try:
            await action_fn()
            await self._reply(update, f"✅ Готово: <i>{action_name}</i>")
            log.info("telegram_control_confirmed", action=action_name, chat_id=cid)
        except Exception as exc:
            await self._reply(update, f"❌ Не получилось: <code>{exc}</code>")
            log.error("telegram_control_failed", action=action_name, error=str(exc))

    async def _on_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        query = update.callback_query
        if query is None:
            return
        await query.answer()
        if not await self._authorised(update):
            return

        data = query.data or ""
        if data.startswith("view:"):
            await self._handle_view_button(update, data.removeprefix("view:"))
            return
        if data.startswith("control:"):
            await self._handle_control_button(update, data.removeprefix("control:"))
            return
        if data.startswith("train:"):
            await self._handle_train_button(update, data.removeprefix("train:"))
            return
        if data.startswith("limit:"):
            await self._handle_limit_button(update, data.removeprefix("limit:"))
            return
        if data.startswith("mode:"):
            await self._handle_mode_button(update, data.removeprefix("mode:"))
            return
        if data.startswith("risk:"):
            await self._queue_risk_change(update, data.removeprefix("risk:").upper())
            return
        if data.startswith("sym:"):
            await self._handle_symbol_button(update, data.removeprefix("sym:"))
            return
        await self._button_reply(update, "Неизвестная кнопка.", reply_markup=self._main_menu())

    async def _handle_view_button(self, update: Update, action: str) -> None:
        fake_context = type("_Context", (), {"args": []})()
        handlers = {
            "status": self._cmd_status,
            "balance": self._cmd_balance,
            "positions": self._cmd_positions,
            "signals": self._cmd_signals,
            "symbols": self._cmd_symbols,
            "pnl": self._cmd_pnl,
            "diagnostics": self._cmd_diagnostics,
        }
        if action == "db_model":
            await self._cmd_db_model(update, fake_context)  # type: ignore[arg-type]
            return
        if action == "canary":
            await self._cmd_canary_ready(update, fake_context)  # type: ignore[arg-type]
            return
        if action == "model_help":
            await self._cmd_model_help(update, fake_context)  # type: ignore[arg-type]
            return
        if action == "symbol_select":
            await self._show_symbol_select(update, page=0)
            return
        if action == "control":
            await self._button_reply(
                update,
                "<b>Управление системой</b>\n\nВыберите действие:",
                reply_markup=self._control_menu(),
            )
            return
        handler = handlers.get(action)
        if handler is None:
            await self._button_reply(update, "Неизвестный экран.", reply_markup=self._main_menu())
            return
        await handler(update, fake_context)  # type: ignore[arg-type]

    async def _show_symbol_select(self, update: Update, *, page: int) -> None:
        candidates = self._symbol_candidates()
        selected = self._selected_symbols()
        if not candidates:
            await self._button_reply(
                update,
                "<b>✅ Выбор торговых пар</b>\n\n"
                "Сканер пока не вернул подходящие пары. Подождите обновления сканера или проверьте подключение к Bybit.",
                reply_markup=self._main_menu(),
            )
            return
        await self._button_reply(
            update,
            "<b>✅ Выбор торговых пар</b>\n\n"
            f"Показано до 100 пар, которые прошли фильтры ликвидности, спреда и цены. "
            f"Выбрано вручную: <code>{len(selected)}</code>.\n\n"
            "Нажмите на пару, чтобы добавить или убрать галочку. Выбранные пары будут попадать в обучение, "
            "расчет признаков и торговый отбор, если остаются подходящими по рынку.",
            reply_markup=self._symbol_select_menu(page=page),
        )

    async def _handle_symbol_button(self, update: Update, payload: str) -> None:
        parts = payload.split(":")
        if not parts:
            await self._button_reply(update, "Неизвестное действие выбора пар.", reply_markup=self._main_menu())
            return
        if parts[0] == "page":
            page = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            await self._show_symbol_select(update, page=page)
            return
        if parts[0] == "toggle" and len(parts) >= 2:
            symbol = parts[1].upper()
            page = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            if self._controller is None or self._controller.toggle_symbol is None:
                await self._button_reply(update, "Выбор пар сейчас недоступен.", reply_markup=self._main_menu())
                return
            try:
                msg = await self._controller.toggle_symbol(symbol)
            except Exception as exc:
                msg = f"❌ Не удалось изменить выбор: <code>{html.escape(str(exc))}</code>"
            await self._button_reply(
                update,
                f"<b>✅ Выбор торговых пар</b>\n\n{msg}\n\n"
                "Изменение применится к сканеру; новые пары получат историю свечей и WebSocket-подписки.",
                reply_markup=self._symbol_select_menu(page=page),
            )
            return
        await self._button_reply(update, "Неизвестное действие выбора пар.", reply_markup=self._main_menu())

    async def _handle_control_button(self, update: Update, action: str) -> None:
        if self._controller is None:
            await self._button_reply(update, "Управление сейчас недоступно.", reply_markup=self._main_menu())
            return
        if action == "pause":
            await self._controller.pause()
            await self._button_reply(
                update,
                "Бот на <b>паузе</b>: новые входы не открываются.",
                reply_markup=self._main_menu(),
            )
            return
        if action == "resume":
            await self._controller.resume()
            await self._button_reply(update, "Бот <b>возобновлен</b>.", reply_markup=self._main_menu())
            return
        if action == "promote":
            if self._controller.promote_model is None:
                await self._button_reply(update, "Промоут сейчас недоступен.", reply_markup=self._main_menu())
                return
            db_diag: dict[str, Any] = {}
            if self._controller.db_diagnostics_provider is not None:
                try:
                    db_diag = await self._controller.db_diagnostics_provider()
                except Exception as exc:
                    log.warning("telegram.promote.db_diag_failed", error=str(exc))
            latest_model = db_diag.get("latest_model_version", {}) or {}
            version = latest_model.get("version") or ""
            if not version:
                await self._button_reply(update, "Нет модели-кандидата для промоута.", reply_markup=self._main_menu())
                return
            try:
                msg = await self._controller.promote_model(version)
            except Exception as exc:
                msg = f"❌ Промоут не удался: <code>{html.escape(str(exc))}</code>"
            await self._button_reply(update, msg, reply_markup=self._main_menu())
            return
        if action == "train":
            if self._controller.start_training is None:
                await self._button_reply(update, "Запуск обучения сейчас недоступен.", reply_markup=self._main_menu())
                return
            try:
                msg = await self._controller.start_training(500, 15, 5.0)
            except Exception as exc:
                msg = f"❌ Обучение не стартовало: <code>{exc}</code>"
            await self._button_reply(update, msg, reply_markup=self._main_menu())
            return
        if action == "limits":
            await self._button_reply(update, self._limits_text(), reply_markup=self._limits_menu())
            return
        if action == "stop":
            cid = self._chat_id(update)
            if cid:
                self._pending[cid] = (
                    "АВАРИЙНАЯ ОСТАНОВКА (нужен ручной перезапуск)",
                    self._controller.emergency_stop,
                )
            await self._button_reply(
                update,
                "<b>Аварийная остановка</b> запрошена.\nОтправьте /confirm для выполнения или ничего не делайте для отмены.",
                reply_markup=self._main_menu(),
            )
            return
        await self._button_reply(update, "Неизвестное действие управления.", reply_markup=self._main_menu())

    async def _handle_train_button(self, update: Update, payload: str) -> None:
        if self._controller is None:
            await self._button_reply(update, "Запуск обучения сейчас недоступен.", reply_markup=self._main_menu())
            return
        if payload == "all":
            if self._controller.start_training_all is None:
                await self._button_reply(update, "Обучение ВСЕ сейчас недоступно.", reply_markup=self._main_menu())
                return
            try:
                msg = await self._controller.start_training_all()
            except Exception as exc:
                msg = f"❌ Обучение не стартовало: <code>{exc}</code>"
            await self._button_reply(update, msg, reply_markup=self._main_menu())
            return
        if self._controller.start_training is None:
            await self._button_reply(update, "Запуск обучения сейчас недоступен.", reply_markup=self._main_menu())
            return
        try:
            min_s_raw, horizon_raw, label_raw = payload.split(":", maxsplit=2)
            min_samples = int(min_s_raw)
            horizon = int(horizon_raw)
            label_bps = float(label_raw)
            msg = await self._controller.start_training(min_samples, horizon, label_bps)
        except Exception as exc:
            msg = f"❌ Обучение не стартовало: <code>{exc}</code>"
        await self._button_reply(update, msg, reply_markup=self._main_menu())

    async def _handle_limit_button(self, update: Update, payload: str) -> None:
        if self._controller is None or self._controller.set_runtime_setting is None:
            await self._button_reply(update, "Runtime-настройки сейчас недоступны.", reply_markup=self._main_menu())
            return
        try:
            key, raw_value = payload.split(":", maxsplit=1)
            value: Any = float(raw_value) if key == "price_cap" else int(raw_value)
            msg = await self._controller.set_runtime_setting(key, value)
        except Exception as exc:
            await self._button_reply(
                update,
                f"❌ Изменение лимита отклонено: <code>{exc}</code>",
                reply_markup=self._limits_menu(),
            )
            return
        await self._button_reply(update, f"✅ {msg}\n\n{self._limits_text()}", reply_markup=self._limits_menu())

    async def _handle_mode_button(self, update: Update, action: str) -> None:
        if self._controller is None:
            await self._button_reply(update, "Управление сейчас недоступно.", reply_markup=self._main_menu())
            return
        if action == "shadow":
            await self._controller.set_shadow(True)
            await self._button_reply(
                update,
                "Режим: <b>SHADOW</b>\nСигналы оцениваются, ордера не отправляются.",
                reply_markup=self._main_menu(),
            )
            return
        # "active" / "shadow off" are BLOCKED — CANARY_LIVE activation requires env-var change only
        if action == "active":
            await self._button_reply(
                update,
                "<b>Включение CANARY_LIVE заблокировано из Telegram.</b>\n"
                "Live-режим включается только через переменные окружения "
                "(TRADING_MODE=CANARY_LIVE, LIVE_MODE=true, LIVE_ARMED=true).\n"
                "Telegram не включает реальные деньги.",
                reply_markup=self._main_menu(),
            )
            return
        await self._button_reply(update, "Неизвестный режим.", reply_markup=self._main_menu())

    async def _queue_risk_change(self, update: Update, new_profile_str: str) -> None:
        if self._controller is None:
            await self._button_reply(update, "Управление сейчас недоступно.", reply_markup=self._main_menu())
            return
        try:
            new_profile = RiskProfile(new_profile_str)
        except ValueError:
            await self._button_reply(update, "Неизвестный риск-профиль.", reply_markup=self._main_menu())
            return
        old_profile = self._controller.current_profile()
        if new_profile_str == old_profile:
            await self._button_reply(
                update,
                f"Риск-профиль уже <code>{old_profile}</code>.",
                reply_markup=self._main_menu(),
            )
            return
        cid = self._chat_id(update)
        if cid:
            self._pending[cid] = (
                f"сменить риск-профиль с {old_profile} на {new_profile_str}",
                lambda: self._controller.set_risk_profile(new_profile),
            )
        await self._button_reply(
            update,
            f"Сменить риск-профиль: <code>{old_profile}</code> → <code>{new_profile_str}</code>\n"
            "Отправьте /confirm для применения или ничего не делайте для отмены.",
            reply_markup=self._main_menu(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _help_text(self) -> str:
        ctrl_section = ""
        if self._controller is not None:
            ctrl_section = (
                "\n<b>Управление</b>\n"
                "/pause   — остановить новые входы\n"
                "/resume  — снять паузу\n"
                "/train [500] [15] [5] — обучить модель-кандидат\n"
                "/limits — показать или изменить безопасные лимиты\n"
                "/canary — готовность к маленькому CANARY_LIVE\n"
                "/model_help — объяснение модели и обучения\n"
                "/mode shadow|active — режим исполнения\n"
                "/shadow on|off — теневой режим\n"
                "/risk conservative|moderate|aggressive|scalp\n"
                "/stop    — аварийная остановка (нужен /confirm)\n"
                "/confirm — подтвердить ожидающее действие\n"
                "\n/start покажет меню кнопок.\n"
            )
        return (
            "<b>Bybit AI Trader</b>\n\n"
            "<b>Наблюдение</b>\n"
            "/status      — здоровье системы\n"
            "/balance     — баланс кошелька\n"
            "/positions   — открытые позиции\n"
            "/signals     — последние сигналы стратегии\n"
            "/regime      — режим рынка по монетам\n"
            "/symbols     — активные монеты\n"
            "/start       — меню, включая выбор торговых пар\n"
            "/pnl         — история закрытого PnL\n"
            "/net         — чистый PnL с комиссиями и фандингом\n"
            "/diagnostics — счетчики и задержки циклов\n"
            "/help        — это сообщение\n" + ctrl_section
        )

    def _symbol_candidates(self) -> list[str]:
        if self._controller is None or self._controller.symbol_candidates is None:
            return []
        return self._controller.symbol_candidates()[:100]

    def _selected_symbols(self) -> list[str]:
        if self._controller is None or self._controller.selected_symbols is None:
            return []
        return self._controller.selected_symbols()

    def _limits_text(self) -> str:
        if self._controller is None or self._controller.runtime_settings is None:
            return "<b>Лимиты</b>\nПока недоступны."
        s = self._controller.runtime_settings()
        gate_quality = s.get("model_gate_quality", {}) or {}
        return (
            "<b>Лимиты риска и нагрузки</b>\n"
            f"Пауза: <code>{'да' if s.get('paused', False) else 'нет'}</code>\n"
            f"SHADOW: <code>{'да' if s.get('shadow', True) else 'нет'}</code>\n"
            f"Риск-профиль: <code>{s.get('risk_profile', 'n/a')}</code>\n"
            f"Новых входов в минуту: <code>{s.get('max_entries_per_minute', 'n/a')}</code> — ограничивает скорость риска\n"
            f"Одновременно pending: <code>{s.get('max_concurrent_pending', 'n/a')}</code> — сколько заявок может висеть\n"
            f"Позиций в одну сторону: <code>{s.get('max_same_side', 'n/a')}</code> — защита от перекоса Long/Short\n"
            f"Потолок цены монеты: <code>{s.get('screener_max_price_usd', 'n/a')}</code> — отсеивает дорогие монеты\n"
            f"Монет для признаков: <code>{s.get('feature_max_symbols', 'n/a')}</code> — нагрузка на расчет модели\n"
            f"Кандидатов на вход: <code>{s.get('execution_candidates', 'n/a')}</code> — сколько монет смотрит стратегия\n"
            f"Фильтр модели в CANARY: <code>{'да' if s.get('model_gate_canary_enabled', False) else 'нет'}</code>\n"
            f"Порог фильтра модели: <code>{s.get('model_gate_threshold', 'n/a')}</code>\n\n"
            f"Качество фильтра модели: <code>{self._ru(gate_quality.get('quality', 'n/a'))}</code>, "
            f"лучший порог=<code>{gate_quality.get('best_threshold', 'n/a')}</code>, "
            f"наблюдений=<code>{gate_quality.get('gate_total_count', 0)}</code>, "
            f"lift=<code>{gate_quality.get('gate_lift_vs_all_bps', 'n/a')}</code>\n\n"
            "Изменить: <code>/limits entries 1</code>, <code>/limits pending 1</code>, "
            "<code>/limits same_side 1</code>, <code>/limits price_cap 25</code>\n"
            "Фильтр модели: <code>/limits model_gate on</code>, <code>/limits model_gate_threshold 0.60</code>"
        )

    def _component_line(self, name: str, ok: bool, latency_ms: float | None, required: bool = True) -> str:
        if ok:
            icon = "✅"
        elif required:
            icon = "❌"
        else:
            icon = "⚠️"
        s = f"{icon} {name}"
        if latency_ms is not None:
            s += f" <code>{latency_ms:.0f}ms</code>"
        return s
