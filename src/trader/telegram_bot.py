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
    signal_log: deque[SignalEntry] = field(default_factory=lambda: deque(maxlen=20))
    # Optional diagnostics provider (returns dict from TradingApplication.get_diagnostics)
    diagnostics_provider: Callable[[], dict[str, Any]] | None = None
    # Optional async DB diagnostics provider
    db_diagnostics_provider: Callable[[], Awaitable[dict[str, Any]]] | None = None
    start_training: Callable[[int, int, float], Awaitable[str]] | None = None
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
        mode = "SHADOW" if entry.shadow else "LIVE"
        text = (
            f"{icon} <b>Signal [{mode}]</b>\n"
            f"{entry.symbol} {entry.side} | conf: <code>{entry.confidence:.2f}</code>\n"
            f"Regime: <code>{entry.regime}</code>\n"
            f"{entry.rationale}"
        )
        await self.notify(text)

    async def notify_position_opened(self, symbol: str, side: str, qty: Decimal, price: Decimal) -> None:
        icon = "🟢" if side == "BUY" else "🔴"
        await self.notify(f"{icon} <b>Position opened</b>\n{symbol} {side} {qty} @ {price}")

    async def notify_position_closed(self, symbol: str, realized_pnl: Decimal) -> None:
        icon = "✅" if realized_pnl >= 0 else "❌"
        await self.notify(f"{icon} <b>Position closed</b>\n{symbol} PnL: <code>{realized_pnl:+.4f} USDT</code>")

    async def notify_circuit_breaker(self, breaker_type: str, reason: str) -> None:
        await self.notify(f"⚠️ <b>Circuit breaker</b>\nType: <code>{breaker_type}</code>\nReason: {reason}")

    async def notify_risk_changed(self, old_profile: str, new_profile: str) -> None:
        await self.notify(f"⚙️ <b>Risk profile changed</b>\n{old_profile} → <code>{new_profile}</code>")

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    async def _authorised(self, update: Update) -> bool:
        chat = update.effective_chat
        if chat is None or chat.id not in self._config.allowed_chat_ids:
            if update.effective_message is not None:
                suffix = f" Chat ID: {chat.id}" if chat else ""
                await update.effective_message.reply_text(f"Access denied.{suffix}")
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
            [
                InlineKeyboardButton("🔦 Shadow ON", callback_data="mode:shadow"),
                InlineKeyboardButton("🚫 LIVE заблокирован", callback_data="mode:active"),
            ],
            [
                InlineKeyboardButton("🧠 Обучить 500", callback_data="train:500:15:5"),
                InlineKeyboardButton("🧠 Обучить 1000", callback_data="train:1000:15:5"),
            ],
            [
                InlineKeyboardButton("🎚 Лимиты", callback_data="control:limits"),
                InlineKeyboardButton("🗄 База и модель", callback_data="view:db_model"),
            ],
            [InlineKeyboardButton("🚦 Готовность CANARY", callback_data="view:canary")],
            [InlineKeyboardButton("❓ Как читать модель", callback_data="view:model_help")],
            [InlineKeyboardButton("🚨 Emergency stop", callback_data="control:stop")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="view:status")],
        ]
        return InlineKeyboardMarkup(rows)

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
            except BadRequest:
                pass
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
            await self._reply(update, f"<b>Status</b>\nCheck failed: <code>{exc}</code>")
            return

        ctrl = self._controller
        lines = [
            "<b>System Status</b>",
            f"Overall: <code>{health.overall}</code>",
            f"System: <code>{health.system_status.value}</code>",
            f"Mode: <code>{health.trading_mode.value}</code>",
            f"Testnet: <code>{str(self._config.bybit_use_testnet).lower()}</code>",
        ]
        if ctrl:
            paused = " ⏸ PAUSED" if ctrl.is_paused() else ""
            shadow = " (SHADOW)" if ctrl.is_shadow() else ""
            lines.append(f"Risk: <code>{ctrl.current_profile()}</code>{shadow}{paused}")
        else:
            lines.append(f"Risk: <code>{self._config.risk_profile}</code>")

        lines += [
            "",
            self._component_line("Postgres", health.postgres, health.postgres_latency_ms, required=True),
            self._component_line("Redis", health.redis, health.redis_latency_ms, required=False),
            self._component_line("Bybit REST", health.bybit_rest, health.bybit_rest_latency_ms, required=False),
            self._component_line("Bybit WS", health.bybit_ws, None, required=True),
            self._component_line("Features", health.features_fresh, None, required=True),
        ]
        if health.messages:
            lines += ["", "<b>Alerts</b>"]
            lines.extend(f"• {m}" for m in health.messages[:6])
        await self._reply(update, "\n".join(lines))

    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        adapter = self._adapter_factory()
        if adapter is None:
            await self._reply(update, "Adapter not ready.")
            return
        try:
            bal: Balance = await adapter.get_balance()
        except Exception as exc:
            await self._reply(update, f"<b>Balance</b>\nFailed: <code>{exc}</code>")
            return
        lines = [
            "<b>Balance (UNIFIED)</b>",
            f"Currency: <code>{bal.currency}</code>",
            f"Wallet:   <code>{bal.wallet_balance}</code>",
            f"Available:<code>{bal.available_balance}</code>",
        ]
        if bal.margin_balance is not None:
            lines.append(f"Margin:   <code>{bal.margin_balance}</code>")
        if bal.unrealised_pnl:
            pnl_icon = "📈" if bal.unrealised_pnl >= 0 else "📉"
            lines.append(f"Unreal PnL: {pnl_icon} <code>{bal.unrealised_pnl:+}</code>")
        await self._reply(update, "\n".join(lines))

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        adapter = self._adapter_factory()
        if adapter is None:
            await self._reply(update, "Adapter not ready.")
            return
        try:
            positions: list[Position] = await adapter.get_positions(self._config.default_category)
        except Exception as exc:
            await self._reply(update, f"<b>Positions</b>\nFailed: <code>{exc}</code>")
            return
        open_pos = [p for p in positions if p.size > 0]
        if not open_pos:
            await self._reply(update, "<b>Positions</b>\nNo open positions.")
            return
        lines = [f"<b>Open Positions ({len(open_pos)})</b>"]
        for pos in open_pos[:10]:
            pnl_icon = "📈" if pos.unrealised_pnl >= 0 else "📉"
            side_icon = "🟢" if pos.side.value == "BUY" else "🔴"
            lines += [
                "",
                f"{side_icon} <b>{pos.symbol}</b> {pos.side.value}",
                f"  Size:  <code>{pos.size}</code>",
                f"  Entry: <code>{pos.entry_price}</code>",
                f"  Mark:  <code>{pos.mark_price}</code>",
                f"  PnL:   {pnl_icon} <code>{pos.unrealised_pnl:+}</code>",
                f"  Lev:   <code>{pos.leverage}x</code>",
            ]
            if pos.liquidation_price:
                lines.append(f"  Liq:   <code>{pos.liquidation_price}</code>")
        if len(open_pos) > 10:
            lines.append(f"\n… +{len(open_pos) - 10} more")
        await self._reply(update, "\n".join(lines))

    async def _cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        if self._controller is None or not self._controller.signal_log:
            await self._reply(update, "<b>Signals</b>\nNo signals yet.")
            return
        lines = ["<b>Recent Signals</b>"]
        for s in list(self._controller.signal_log)[-10:]:
            icon = "🟢" if s.side == "BUY" else "🔴"
            mode = "shadow" if s.shadow else "live"
            ts = s.timestamp.strftime("%H:%M:%S")
            lines += [
                "",
                f"{icon} <b>{s.symbol}</b> {s.side} [{mode}] {ts}",
                f"  Conf: <code>{s.confidence:.2f}</code>  Regime: <code>{s.regime}</code>",
                f"  {s.rationale[:80]}",
            ]
        await self._reply(update, "\n".join(lines))

    async def _cmd_regime(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        if self._controller is None:
            await self._reply(update, "Controller not available.")
            return
        symbols = self._controller.active_symbols()
        if not symbols:
            await self._reply(update, "<b>Regime</b>\nNo active symbols.")
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
        lines = ["<b>Market Regime</b>"]
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
            await self._reply(update, "Controller not available.")
            return
        symbols = self._controller.active_symbols()
        if not symbols:
            await self._reply(update, "<b>Active Symbols</b>\nNone.")
            return
        lines = [f"<b>Active Symbols ({len(symbols)})</b>"]
        lines.extend(f"• <code>{s}</code>" for s in symbols)
        await self._reply(update, "\n".join(lines))

    async def _cmd_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        adapter = self._adapter_factory()
        if adapter is None:
            await self._reply(update, "Adapter not ready.")
            return
        try:
            resp = await adapter._rest.get_closed_pnl(category=self._config.default_category, limit=20)
            records = resp.get("result", {}).get("list", [])
        except Exception as exc:
            await self._reply(update, f"<b>PnL</b>\nFailed: <code>{exc}</code>")
            return
        if not records:
            await self._reply(update, "<b>Closed PnL</b>\nNo closed trades.")
            return
        total = Decimal("0")
        lines = ["<b>Closed PnL (last 20)</b>"]
        for r in records[:10]:
            sym = r.get("symbol", "?")
            pnl = Decimal(str(r.get("closedPnl", "0")))
            total += pnl
            icon = "✅" if pnl >= 0 else "❌"
            lines.append(f"{icon} <code>{sym}</code>: <code>{pnl:+.4f}</code>")
        total_icon = "📈" if total >= 0 else "📉"
        lines.append(f"\n{total_icon} <b>Total (shown):</b> <code>{total:+.4f} USDT</code>")
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
        except Exception:
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
            "📈 <b>NET RESULTS (today)</b>\n\n"
            f"Gross PnL:        <code>{gross:+.4f} USDT</code>\n"
            f"Trading fees:     <code>{fees:+.4f} USDT</code>\n"
            f"Funding:          <code>{funding:+.4f} USDT</code>\n"
            f"Est. slippage:    <code>{slippage_est:+.4f} USDT</code>\n"
            f"─────────────────────────────\n"
            f"Net PnL:          <code>{net:+.4f} USDT</code>\n"
            f"Fee drag:         <code>{fee_drag:.4f} USDT</code>\n\n"
            f"Maker fills:      <code>{maker_pct:.1f}%</code>\n"
            f"Taker fills:      <code>{taker_pct:.1f}%</code>"
        )
        await self._reply(update, text)

    async def _cmd_diagnostics(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        if self._controller is None or self._controller.diagnostics_provider is None:
            await self._reply(update, "<b>Diagnostics</b>\nNot available.")
            return
        try:
            diag = self._controller.diagnostics_provider()
        except Exception as exc:
            await self._reply(update, f"<b>Diagnostics</b>\nFailed: <code>{exc}</code>")
            return

        loop_at = diag.get("last_strategy_loop_at") or "never"
        ws_age = diag.get("last_ws_message_age_s")
        ws_str = f"{ws_age:.0f}s ago" if ws_age is not None else "unknown"
        symbols = diag.get("active_symbols") or []
        positions = diag.get("open_positions") or []
        heat = diag.get("portfolio_heat_pct")
        heat_str = f"{heat:.1f}%" if heat is not None else "n/a"

        lines = [
            "<b>Diagnostics (last hour)</b>",
            f"Strategy loop: <code>{loop_at}</code>",
            f"Last WS msg:   <code>{ws_str}</code>",
            f"Active symbols: <code>{len(symbols)}</code>  {' '.join(symbols[:5])}{'…' if len(symbols) > 5 else ''}",
            f"Open positions: <code>{len(positions)}</code>  {' '.join(positions[:5])}{'…' if len(positions) > 5 else ''}",
            f"Portfolio heat: <code>{heat_str}</code>",
            "",
            f"Signals emitted:    <code>{diag.get('hour_signals_emitted', 0)}</code>",
            f"Risk rejected:      <code>{diag.get('hour_risk_rejected', 0)}</code>",
            f"API rejected:       <code>{diag.get('hour_api_rejected', 0)}</code>",
            f"Min-notional rej:   <code>{diag.get('hour_min_notional_rejected', 0)}</code>",
            f"Skipped open pos:   <code>{diag.get('hour_skipped_open_position', 0)}</code>",
            f"Skipped entry cd:   <code>{diag.get('hour_skipped_entry_cooldown', 0)}</code>",
            f"Skipped fail cd:    <code>{diag.get('hour_skipped_failure_cooldown', 0)}</code>",
            f"Model gate blocks:  <code>{diag.get('hour_model_gate_canary_blocked', 0)}</code>",
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
            reply_markup=self._control_menu(),
        )

    def _canary_readiness_text(self, *, db_diag: dict[str, Any], diag: dict[str, Any]) -> str:
        checks: list[tuple[str, bool, str]] = []
        warnings: list[str] = []

        def require(label: str, ok: bool, detail: str) -> None:
            checks.append((label, ok, detail))

        def warn_if(condition: bool, detail: str) -> None:
            if condition:
                warnings.append(detail)

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

        require("DB connected", bool(db_diag.get("connected")), "storage is reachable")
        require("Fresh 1m candle", latest_age_s is not None and latest_age_s <= 600, self._age_label(latest_age_s))
        require("1m history", int(candles.get("1") or 0) >= 1000, f"{int(candles.get('1') or 0)} candles")
        require("5m history", int(candles.get("5") or 0) >= 200, f"{int(candles.get('5') or 0)} candles")
        require("15m history", int(candles.get("15") or 0) >= 200, f"{int(candles.get('15') or 0)} candles")
        require("1h history", int(candles.get("60") or 0) >= 100, f"{int(candles.get('60') or 0)} candles")
        require("Active symbols", len(active_symbols) >= 3, f"{len(active_symbols)} symbols")
        require("Feature snapshots", feature_snapshots >= 1000, f"{feature_snapshots} snapshots")
        require("Labelled 15m samples", trainable_15m >= 1000, f"{trainable_15m} samples")
        require("Prediction outcomes", prediction_outcomes >= 1000, f"{prediction_outcomes} outcomes")
        require(
            "Training completed",
            bool(latest_model.get("version")) or latest_run.get("status") == "COMPLETED",
            f"run={latest_run.get('status', 'none')} model={latest_model.get('version') or 'none'}",
        )
        require("Shadow mode", bool(is_shadow), "orders are not live yet")
        require("Not paused", not bool(is_paused), "scanner can keep collecting/evaluating")
        require("Websocket alive", ws_age is None or float(ws_age) <= 180, self._age_label(ws_age))

        warn_if(trainable_15m < 2000, f"Trainable 15m is {trainable_15m}; preferred before real CANARY is 2000+.")
        warn_if(gate_total < 50, f"Model gate has only {gate_total} labelled shadow decisions; preferred is 50+.")
        if gate_lift is not None:
            warn_if(
                float(gate_lift) <= 0,
                f"Model gate lift is {float(gate_lift):+.2f} bps; better to wait for positive lift.",
            )
        else:
            warnings.append("Model gate lift is not measured yet.")
        warn_if(paper_gate_count < 20, f"Paper model-gate sample is {paper_gate_count}; preferred is 20+.")
        warn_if(paper_gate_count >= 20 and paper_gate_bps <= 0, f"Paper model-gate PnL is {paper_gate_bps:+.1f} bps.")
        warn_if(paper_base_count == 0, "Paper baseline has no completed trades yet.")
        warn_if(hour_api_rejected > 0, f"API rejected {hour_api_rejected} actions in the last hour.")
        warn_if(hour_min_notional > 0, f"Min-notional rejected {hour_min_notional} actions in the last hour.")
        warn_if(bool(runtime.get("model_gate_canary_enabled")), "Model gate canary is already enabled.")
        diag_error = diag.get("error")
        db_error = db_diag.get("error") or db_diag.get("last_connect_error") or db_diag.get("last_read_error")
        warn_if(bool(diag_error), f"Runtime diagnostics error: {html.escape(str(diag_error))}")
        warn_if(bool(db_error), f"DB diagnostics error: {html.escape(str(db_error))}")
        if loop_at in (None, "never"):
            warnings.append("Strategy loop timestamp is not available.")

        failed = [item for item in checks if not item[1]]
        if failed:
            status = "❌ NOT READY"
        elif warnings:
            status = "⚠️ ALMOST READY"
        else:
            status = "✅ READY"

        lines = [
            "<b>🚦 CANARY readiness</b>",
            f"Status: <b>{status}</b>",
            "",
            "<b>Required</b>",
        ]
        for label, ok, detail in checks:
            icon = "✅" if ok else "❌"
            lines.append(f"{icon} {label}: <code>{html.escape(str(detail))}</code>")
        lines.extend(["", "<b>Warnings</b>"])
        if warnings:
            lines.extend(f"⚠️ {warning}" for warning in warnings[:8])
        else:
            lines.append("✅ No warnings.")
        lines.extend(
            [
                "",
                "<b>Next</b>",
                "Keep CANARY tiny: 1 position, low notional, model live decisions still disabled.",
                "Telegram cannot enable LIVE; env vars must be changed deliberately on Render.",
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

    async def _cmd_model_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Explain model/training screens in operator-friendly Russian."""
        del context
        if not await self._authorised(update):
            return
        await self._reply(update, self._model_help_text(), reply_markup=self._control_menu())

    def _model_help_text(self) -> str:
        return (
            "<b>❓ Как пользоваться моделью</b>\n\n"
            "<b>1. База</b>\n"
            "Если БД зеленая, бот сохраняет свечи, сигналы и результаты. "
            "Свечи 1m/5m/15m/1h — это история рынка. Чем больше истории, тем надежнее обучение.\n\n"
            "<b>2. Trainable 15m</b>\n"
            "Это количество готовых размеченных примеров для обучения на горизонте 15 минут. "
            "Для первого обучения достаточно примерно 1000. Для более уверенного CANARY лучше 2000+.\n\n"
            "<b>3. Challenger</b>\n"
            "Это новая обученная модель-кандидат. Она пока наблюдает в тени и не открывает сделки сама.\n\n"
            "<b>4. Gate pass/block</b>\n"
            "Модель смотрит на сигналы стратегии и решает: пропустить сигнал или заблокировать. "
            "Если Gate lift положительный — фильтр модели помогает. Если отрицательный — модель пока не улучшает отбор.\n\n"
            "<b>5. Model live decisions</b>\n"
            "Сейчас <b>disabled</b>: модель не управляет реальными сделками. Это правильно для этапа проверки.\n\n"
            "<b>Что делать пользователю</b>\n"
            "Нажимать <b>База и модель</b> и <b>Готовность CANARY</b>. "
            "Обучение можно запускать вручную кнопками <b>Обучить 500/1000</b>, "
            "а автообучение само сработает, когда накопятся новые размеченные примеры."
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
        db_error = db_diag.get("error") or db_diag.get("last_connect_error") or db_diag.get("last_read_error")
        db_error_str = html.escape(str(db_error)) if db_error else ""
        if len(db_error_str) > 180:
            db_error_str = f"{db_error_str[:177]}..."
        candles = db_diag.get("candles_by_interval", {})
        latest_1m = db_diag.get("latest_candle_1m")
        latest_str = latest_1m.strftime("%H:%M:%S UTC") if latest_1m else "none"
        outcomes_by_horizon = db_diag.get("prediction_outcomes_by_horizon", {}) or {}
        labelled_15m = db_diag.get("labelled_samples_15m", 0)
        outcome_parts = [
            f"{horizon}m={count}"
            for horizon, count in sorted(outcomes_by_horizon.items(), key=lambda item: int(item[0]))
        ]
        outcome_breakdown = ", ".join(outcome_parts) if outcome_parts else "none"

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
        latest_run_samples = latest_run.get("sample_count", 0) or 0
        latest_run_error = str(latest_run.get("error") or "")
        if len(latest_run_error) > 120:
            latest_run_error = f"{latest_run_error[:117]}..."
        latest_run_error = html.escape(latest_run_error)
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
            return f"{int(stats.get('count') or 0)} trades, {total_bps:+.1f} bps / {usd:+.3f} USDT, DD {dd_usd:+.3f}"

        # --- прогресс по шагам к реальным сделкам ---
        step1_ok = labelled_15m >= 2000
        step1_partial = 1000 <= labelled_15m < 2000
        step2_ok = bool(db_model_version)
        step3_ok = gate_lift is not None and float(gate_lift) > 0
        step3_partial = gate_lift is not None and not step3_ok
        step4_ok = champion_ver != "none"

        def _sicon(ok: bool, partial: bool = False) -> str:
            return "✅" if ok else ("⚠️" if partial else "❌")

        steps_done = sum([step1_ok, step2_ok, step3_ok, step4_ok])
        bar_f = steps_done * 2
        progress_bar = "█" * bar_f + "░" * (10 - bar_f)

        # прогресс-бар накопления данных
        data_pct = min(100, int(labelled_15m / 2000 * 100))
        data_bar_f = min(10, int(labelled_15m / 2000 * 10))
        data_bar = "█" * data_bar_f + "░" * (10 - data_bar_f)

        # когда следующее автообучение
        _trained_on = int(samples) if samples else int(latest_run_samples)
        _next_train_at = _trained_on + 1000
        _samples_to_next = max(0, _next_train_at - labelled_15m)

        # статус модели
        _status_map = {
            "CHAMPION": "🏆 Чемпион (управляет решениями)",
            "SHADOW_CHALLENGER": "🔵 Кандидат в тени (наблюдает, не торгует)",
            "VALIDATED": "🔵 Проверен (ещё не чемпион)",
        }
        model_status_label = _status_map.get(db_model_status or "", f"⚪ {db_model_status or 'нет'}")

        # объяснение фильтра
        pass_pct = int(gate_pass / gate_total * 100) if gate_total > 0 else 0
        gate_blocks_good = (
            gate_lift is not None
            and gate_pass_avg is not None
            and gate_block_avg is not None
            and float(gate_block_avg) > float(gate_pass_avg)
        )

        # текущий блокер и следующий шаг
        if not step2_ok:
            blocker = "нет обученной модели"
            next_action = "Нажмите «Обучить 1000» или подождите автообучения"
        elif not step3_ok:
            if gate_total < 50:
                blocker = f"фильтр ещё накапливает наблюдения ({gate_total}/50)"
                next_action = "Ждём — наблюдения накапливаются автоматически"
            else:
                blocker = f"фильтр пока снижает доходность (lift {gate_lift_str})"
                next_action = f"Ждём следующего автообучения (ещё ~{_samples_to_next} примеров)"
        elif not step4_ok:
            blocker = "кандидат ещё не повышен до чемпиона"
            next_action = "Запустить promote через консоль Render"
        else:
            blocker = "ждём включения реальных сделок"
            next_action = "Вручную: TRADING_MODE=CANARY_LIVE + LIVE_MODE=true + LIVE_ARMED=true"

        lines = [
            "<b>🗄 БАЗА И МОДЕЛЬ</b>",
            "",
            "━━ 🎯 ПУТЬ К РЕАЛЬНЫМ СДЕЛКАМ ━━",
            f"{_sicon(step1_ok, step1_partial)} Шаг 1: Данных достаточно"
            + (f" (<code>{labelled_15m}</code> примеров)" if step1_ok else f" (<code>{labelled_15m}/2000</code>)"),
            f"{_sicon(step2_ok)} Шаг 2: Модель обучена"
            + (f" (<code>{db_model_version}</code>)" if step2_ok else " — ещё нет"),
            f"{_sicon(step3_ok, step3_partial)} Шаг 3: Фильтр улучшает отбор"
            + (f" (lift <code>{gate_lift_str}</code>)" if gate_lift is not None else " — ещё не измерен"),
            f"{_sicon(step4_ok)} Шаг 4: Повысить до чемпиона"
            + (f" (<code>{champion_ver}</code>)" if step4_ok else " — пока не выполнено"),
            "❌ Шаг 5: Включить CANARY_LIVE на Render — финальное действие вручную",
            "",
            f"Прогресс: <code>[{progress_bar}] {steps_done}/4 шагов до повышения</code>",
            "",
            "━━ 📊 ДАННЫЕ ━━",
            f"БД: {db_icon} {'подключена' if connected else 'недоступна'}"
            + (f"  •  ошибка: <code>{db_error_str}</code>" if db_error_str else ""),
            f"Последняя свеча: <code>{latest_str}</code>",
            f"История: 1m <code>{candles.get('1', 0)}</code>"
            f" | 5m <code>{candles.get('5', 0)}</code>"
            f" | 15m <code>{candles.get('15', 0)}</code>"
            f" | 1h <code>{candles.get('60', 0)}</code>",
            f"Примеры для обучения (15m): <code>{labelled_15m}/2000</code>"
            f" <code>[{data_bar}]</code> {data_pct}%",
            f"Записано исходов: <code>{db_diag.get('prediction_outcomes', 0)}</code>"
            f"  (по горизонтам: <code>{outcome_breakdown}</code>)",
            "",
            "━━ 🧠 МОДЕЛЬ ━━",
            f"Обучена: <code>{last_training}</code>  (<code>{samples}</code> примеров)",
            f"Статус: {model_status_label}",
            f"Чемпион: <code>{'нет' if champion_ver == 'none' else champion_ver}</code>"
            + ("  ← нужно повысить" if not step4_ok and step3_ok else ""),
            f"Следующее автообучение: примерно через <code>{_samples_to_next}</code> примеров",
            "",
            "━━ 🔍 ОЦЕНКА ФИЛЬТРА (15m) ━━",
        ]

        if gate_total == 0:
            lines.append("Наблюдений ещё нет — фильтр только начинает работу")
        else:
            lines += [
                f"Проверено сигналов: <code>{gate_total}</code>",
                f"  ✔ Пропустил   : <code>{gate_pass}</code> ({pass_pct}%) → <code>{gate_pass_avg_str}</code> в среднем",
                f"  ✘ Заблокировал: <code>{gate_block}</code> ({100 - pass_pct}%) → <code>{gate_block_avg_str}</code> в среднем",
                f"Польза от фильтра: <code>{gate_lift_str}</code>"
                + (" ✅ фильтр работает!" if step3_ok else " ⚠️ нужно > 0"),
            ]
            if gate_blocks_good:
                lines.append("⚠️ <i>Проблема: заблокированные сделки выгоднее пропущенных — модель учится</i>")
            if gate_reasons_str and gate_reasons_str != "n/a":
                lines.append(f"Причина блоков: <code>{gate_reasons_str}</code>")

        paper_base_count = int(paper_baseline.get("count") or 0)
        paper_gate_count = int(paper_gate.get("count") or 0)
        if paper_base_count > 0 or paper_gate_count > 0:
            lines += [
                "",
                "━━ 📈 БУМАЖНЫЕ СДЕЛКИ (симуляция) ━━",
                f"Правила (без модели): <code>{_paper_line(paper_baseline)}</code>",
                f"С фильтром модели  : <code>{_paper_line(paper_gate)}</code>",
            ]

        lines += [
            "",
            "━━ 📋 ТЕКУЩИЙ БЛОКЕР ━━",
            f"Ждём: {blocker}",
            f"Действие: {next_action}",
        ]
        if latest_run_error:
            lines += ["", f"⚠️ Ошибка обучения: <code>{latest_run_error}</code>"]
        if db_diag.get("error"):
            lines.append(f"\n<i>Ошибка БД: {html.escape(str(db_diag['error']))}</i>")

        await self._reply(update, "\n".join(lines), reply_markup=self._main_menu())

    # ------------------------------------------------------------------
    # Control commands
    # ------------------------------------------------------------------

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        if self._controller is None:
            await self._reply(update, "Control not available.")
            return
        await self._controller.pause()
        await self._reply(update, "⏸ Trading <b>paused</b>. No new entries will be opened.\nUse /resume to restart.")

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        if self._controller is None:
            await self._reply(update, "Control not available.")
            return
        if not self._controller.is_paused():
            await self._reply(update, "Trading is not paused.")
            return
        await self._controller.resume()
        await self._reply(update, "▶️ Trading <b>resumed</b>.")

    async def _cmd_shadow(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorised(update):
            return
        if self._controller is None:
            await self._reply(update, "Control not available.")
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
                "<b>Shadow OFF blocked.</b>\n"
                "Live trading activation requires environment variable changes "
                "(TRADING_MODE=CANARY_LIVE, LIVE_MODE=true, LIVE_ARMED=true). "
                "Telegram cannot activate live trading.",
            )
            return
        await self._controller.set_shadow(True)
        await self._reply(update, "🔦 Shadow mode: <code>ON (orders logged, not sent)</code>")

    async def _cmd_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorised(update):
            return
        if self._controller is None:
            await self._reply(update, "Control not available.")
            return
        args = context.args or []
        if not args or args[0].lower() not in ("shadow", "active"):
            current = "shadow" if self._controller.is_shadow() else "active"
            venue = "testnet" if self._config.bybit_use_testnet else "configured live endpoint"
            await self._reply(
                update,
                f"Current execution mode: <code>{current}</code> on <code>{venue}</code>.\n"
                "Use: /mode shadow  or  /mode active",
            )
            return
        if args[0].lower() == "shadow":
            await self._controller.set_shadow(True)
            await self._reply(update, "Mode: <b>SHADOW</b>. Orders are not sent.")
            return
        # "active" is blocked — requires env-var change
        await self._reply(
            update,
            "<b>CANARY_LIVE activation blocked.</b>\n"
            "Live mode requires environment variable changes: "
            "TRADING_MODE=CANARY_LIVE, LIVE_MODE=true, LIVE_ARMED=true.\n"
            "Telegram cannot activate live trading.",
        )

    async def _cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorised(update):
            return
        if self._controller is None:
            await self._reply(update, "Control not available.")
            return
        args = context.args or []
        valid = [p.value.lower() for p in RiskProfile]
        if not args or args[0].lower() not in valid:
            current = self._controller.current_profile()
            await self._reply(update, f"Current risk profile: <code>{current}</code>\nUsage: /risk {' | '.join(valid)}")
            return
        new_profile_str = args[0].upper()
        new_profile = RiskProfile(new_profile_str)
        old_profile = self._controller.current_profile()
        if new_profile_str == old_profile:
            await self._reply(update, f"Risk profile is already <code>{old_profile}</code>.")
            return

        # Block escalation to a riskier profile unless explicitly allowed.
        old_level = _RISK_LEVEL.get(old_profile, 0)
        new_level = _RISK_LEVEL.get(new_profile_str, 0)
        if new_level > old_level and not self._controller.allow_risk_increase:
            await self._reply(
                update,
                f"🚫 <b>Risk escalation blocked</b>: <code>{old_profile}</code> → <code>{new_profile_str}</code>\n"
                "Set <code>TELEGRAM_ALLOW_RISK_INCREASE=true</code> in environment to enable.\n"
                "A service restart is required after changing this env var.",
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
                f"change risk profile from {old_profile} to {new_profile_str}",
                lambda: self._controller.set_risk_profile(new_profile),  # type: ignore[union-attr]
            )
        await self._reply(
            update,
            f"⚠️ Change risk profile: <code>{old_profile}</code> → <code>{new_profile_str}</code>\n"
            "Send /confirm to apply, or ignore to cancel.",
        )

    async def _cmd_train(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorised(update):
            return
        if self._controller is None or self._controller.start_training is None:
            await self._reply(update, "Training control not available.")
            return
        args = context.args or []
        try:
            min_samples = int(args[0]) if len(args) >= 1 else 500
            horizon = int(args[1]) if len(args) >= 2 else 15
            label_bps = float(args[2]) if len(args) >= 3 else 5.0
        except ValueError:
            await self._reply(update, "Usage: /train [min_samples] [horizon_minutes] [label_bps]")
            return
        if min_samples < 50 or horizon <= 0 or label_bps < 0:
            await self._reply(update, "Training parameters rejected: min_samples>=50, horizon>0, label_bps>=0.")
            return
        try:
            msg = await self._controller.start_training(min_samples, horizon, label_bps)
        except Exception as exc:
            await self._reply(update, f"❌ Training failed to start: <code>{exc}</code>")
            return
        await self._reply(update, msg, reply_markup=self._main_menu())

    async def _cmd_limits(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorised(update):
            return
        if self._controller is None or self._controller.runtime_settings is None:
            await self._reply(update, "Runtime settings not available.")
            return
        args = context.args or []
        if not args:
            await self._reply(update, self._limits_text(), reply_markup=self._control_menu())
            return
        if len(args) != 2 or self._controller.set_runtime_setting is None:
            await self._reply(
                update,
                "Usage: /limits entries|pending|same_side|price_cap|feature_symbols|exec_candidates N\n"
                "Model gate: /limits model_gate on|off, /limits model_gate_threshold 0.60",
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
            await self._reply(update, f"❌ Limit change rejected: <code>{exc}</code>")
            return
        await self._reply(update, f"✅ {msg}\n\n{self._limits_text()}", reply_markup=self._control_menu())

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        if self._controller is None:
            await self._reply(update, "Control not available.")
            return
        cid = self._chat_id(update)
        if cid:
            self._pending[cid] = (
                "EMERGENCY STOP (requires manual restart)",
                self._controller.emergency_stop,
            )
        await self._reply(
            update,
            "🚨 <b>Emergency stop</b> requested.\n"
            "This will halt ALL new entries and require a manual restart.\n"
            "Send /confirm to execute, or ignore to cancel.",
        )

    async def _cmd_confirm(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        cid = self._chat_id(update)
        if cid is None or cid not in self._pending:
            await self._reply(update, "No pending action to confirm.")
            return
        action_name, action_fn = self._pending.pop(cid)
        try:
            await action_fn()
            await self._reply(update, f"✅ Done: <i>{action_name}</i>")
            log.info("telegram_control_confirmed", action=action_name, chat_id=cid)
        except Exception as exc:
            await self._reply(update, f"❌ Failed: <code>{exc}</code>")
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
        await self._button_reply(update, "Unknown button.", reply_markup=self._main_menu())

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
        if action == "control":
            await self._button_reply(
                update,
                "<b>Управление системой</b>\n\nВыберите действие:",
                reply_markup=self._control_menu(),
            )
            return
        handler = handlers.get(action)
        if handler is None:
            await self._button_reply(update, "Unknown view.", reply_markup=self._main_menu())
            return
        await handler(update, fake_context)  # type: ignore[arg-type]

    async def _handle_control_button(self, update: Update, action: str) -> None:
        if self._controller is None:
            await self._button_reply(update, "Control not available.", reply_markup=self._main_menu())
            return
        if action == "pause":
            await self._controller.pause()
            await self._button_reply(
                update,
                "Trading <b>paused</b>. No new entries will be opened.",
                reply_markup=self._main_menu(),
            )
            return
        if action == "resume":
            await self._controller.resume()
            await self._button_reply(update, "Trading <b>resumed</b>.", reply_markup=self._main_menu())
            return
        if action == "train":
            if self._controller.start_training is None:
                await self._button_reply(update, "Training control not available.", reply_markup=self._main_menu())
                return
            try:
                msg = await self._controller.start_training(500, 15, 5.0)
            except Exception as exc:
                msg = f"❌ Training failed to start: <code>{exc}</code>"
            await self._button_reply(update, msg, reply_markup=self._main_menu())
            return
        if action == "limits":
            await self._button_reply(update, self._limits_text(), reply_markup=self._limits_menu())
            return
        if action == "stop":
            cid = self._chat_id(update)
            if cid:
                self._pending[cid] = (
                    "EMERGENCY STOP (requires manual restart)",
                    self._controller.emergency_stop,
                )
            await self._button_reply(
                update,
                "<b>Emergency stop</b> requested.\nSend /confirm to execute, or ignore to cancel.",
                reply_markup=self._main_menu(),
            )
            return
        await self._button_reply(update, "Unknown control.", reply_markup=self._main_menu())

    async def _handle_train_button(self, update: Update, payload: str) -> None:
        if self._controller is None or self._controller.start_training is None:
            await self._button_reply(update, "Training control not available.", reply_markup=self._main_menu())
            return
        try:
            min_s_raw, horizon_raw, label_raw = payload.split(":", maxsplit=2)
            min_samples = int(min_s_raw)
            horizon = int(horizon_raw)
            label_bps = float(label_raw)
            msg = await self._controller.start_training(min_samples, horizon, label_bps)
        except Exception as exc:
            msg = f"❌ Training failed to start: <code>{exc}</code>"
        await self._button_reply(update, msg, reply_markup=self._main_menu())

    async def _handle_limit_button(self, update: Update, payload: str) -> None:
        if self._controller is None or self._controller.set_runtime_setting is None:
            await self._button_reply(update, "Runtime settings not available.", reply_markup=self._main_menu())
            return
        try:
            key, raw_value = payload.split(":", maxsplit=1)
            value: Any = float(raw_value) if key == "price_cap" else int(raw_value)
            msg = await self._controller.set_runtime_setting(key, value)
        except Exception as exc:
            await self._button_reply(
                update,
                f"❌ Limit change rejected: <code>{exc}</code>",
                reply_markup=self._limits_menu(),
            )
            return
        await self._button_reply(update, f"✅ {msg}\n\n{self._limits_text()}", reply_markup=self._limits_menu())

    async def _handle_mode_button(self, update: Update, action: str) -> None:
        if self._controller is None:
            await self._button_reply(update, "Control not available.", reply_markup=self._main_menu())
            return
        if action == "shadow":
            await self._controller.set_shadow(True)
            await self._button_reply(
                update,
                "Mode: <b>SHADOW</b>\nSignals are evaluated, orders are not sent.",
                reply_markup=self._main_menu(),
            )
            return
        # "active" / "shadow off" are BLOCKED — CANARY_LIVE activation requires env-var change only
        if action == "active":
            await self._button_reply(
                update,
                "<b>CANARY_LIVE activation blocked.</b>\n"
                "Live mode can only be enabled via environment variable change "
                "(TRADING_MODE=CANARY_LIVE, LIVE_MODE=true, LIVE_ARMED=true).\n"
                "Telegram cannot activate live trading.",
                reply_markup=self._main_menu(),
            )
            return
        await self._button_reply(update, "Unknown mode.", reply_markup=self._main_menu())

    async def _queue_risk_change(self, update: Update, new_profile_str: str) -> None:
        if self._controller is None:
            await self._button_reply(update, "Control not available.", reply_markup=self._main_menu())
            return
        try:
            new_profile = RiskProfile(new_profile_str)
        except ValueError:
            await self._button_reply(update, "Unknown risk profile.", reply_markup=self._main_menu())
            return
        old_profile = self._controller.current_profile()
        if new_profile_str == old_profile:
            await self._button_reply(
                update,
                f"Risk profile is already <code>{old_profile}</code>.",
                reply_markup=self._main_menu(),
            )
            return
        cid = self._chat_id(update)
        if cid:
            self._pending[cid] = (
                f"change risk profile from {old_profile} to {new_profile_str}",
                lambda: self._controller.set_risk_profile(new_profile),
            )
        await self._button_reply(
            update,
            f"Change risk profile: <code>{old_profile}</code> → <code>{new_profile_str}</code>\n"
            "Send /confirm to apply, or ignore to cancel.",
            reply_markup=self._main_menu(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _help_text(self) -> str:
        ctrl_section = ""
        if self._controller is not None:
            ctrl_section = (
                "\n<b>Control</b>\n"
                "/pause   — stop new entries\n"
                "/resume  — restart after pause\n"
                "/train [500] [15] [5] — train shadow model\n"
                "/limits — show/change safe runtime limits\n"
                "/canary — readiness check for tiny CANARY_LIVE test\n"
                "/model_help — explain model/training in Russian\n"
                "/mode shadow|active — switch execution mode\n"
                "/shadow on|off — toggle shadow mode\n"
                "/risk conservative|moderate|aggressive|scalp\n"
                "/stop    — emergency stop (requires /confirm)\n"
                "/confirm — confirm a pending action\n"
                "\nUse /start to show the button menu.\n"
            )
        return (
            "<b>Bybit AI Trader</b>\n\n"
            "<b>Monitoring</b>\n"
            "/status      — system health\n"
            "/balance     — wallet balance\n"
            "/positions   — open positions\n"
            "/signals     — recent strategy signals\n"
            "/regime      — market regime per symbol\n"
            "/symbols     — active symbols\n"
            "/pnl         — closed PnL history\n"
            "/net         — net P&L breakdown (fees, funding, slippage)\n"
            "/diagnostics — counters & loop timing\n"
            "/help        — this message\n" + ctrl_section
        )

    def _limits_text(self) -> str:
        if self._controller is None or self._controller.runtime_settings is None:
            return "<b>Runtime limits</b>\nNot available."
        s = self._controller.runtime_settings()
        gate_quality = s.get("model_gate_quality", {}) or {}
        return (
            "<b>Runtime limits</b>\n"
            f"Paused: <code>{str(s.get('paused', False)).lower()}</code>\n"
            f"Shadow: <code>{str(s.get('shadow', True)).lower()}</code>\n"
            f"Risk: <code>{s.get('risk_profile', 'n/a')}</code>\n"
            f"Max entries/min: <code>{s.get('max_entries_per_minute', 'n/a')}</code>\n"
            f"Max pending: <code>{s.get('max_concurrent_pending', 'n/a')}</code>\n"
            f"Max same-side: <code>{s.get('max_same_side', 'n/a')}</code>\n"
            f"Price cap: <code>{s.get('screener_max_price_usd', 'n/a')}</code>\n"
            f"Feature symbols: <code>{s.get('feature_max_symbols', 'n/a')}</code>\n"
            f"Exec candidates: <code>{s.get('execution_candidates', 'n/a')}</code>\n"
            f"Model gate canary: <code>{str(s.get('model_gate_canary_enabled', False)).lower()}</code>\n"
            f"Model gate threshold: <code>{s.get('model_gate_threshold', 'n/a')}</code>\n\n"
            f"Model gate quality: <code>{gate_quality.get('quality', 'n/a')}</code>, "
            f"best=<code>{gate_quality.get('best_threshold', 'n/a')}</code>, "
            f"obs=<code>{gate_quality.get('gate_total_count', 0)}</code>, "
            f"lift=<code>{gate_quality.get('gate_lift_vs_all_bps', 'n/a')}</code>\n\n"
            "Change: <code>/limits entries 1</code>, <code>/limits pending 1</code>, "
            "<code>/limits same_side 1</code>, <code>/limits price_cap 25</code>\n"
            "Model gate: <code>/limits model_gate on</code>, <code>/limits model_gate_threshold 0.60</code>"
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
