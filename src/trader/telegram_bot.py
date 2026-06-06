"""Telegram bot — monitoring + operator control.

Observability commands (anyone in allowed_chat_ids):
  /status    — system health
  /balance   — wallet balance
  /positions — open positions + unrealised PnL
  /signals   — last 10 strategy signals
  /regime    — current market regime per symbol
  /symbols   — active symbols from screener
  /pnl       — recent closed PnL

Control commands (require /confirm for dangerous ones):
  /pause              — pause new entries (keep existing positions)
  /resume             — resume after pause
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

from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
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
        app.add_handler(CommandHandler("diagnostics", self._cmd_diagnostics))

        # Control
        app.add_handler(CommandHandler("pause", self._cmd_pause))
        app.add_handler(CommandHandler("resume", self._cmd_resume))
        app.add_handler(CommandHandler("shadow", self._cmd_shadow))
        app.add_handler(CommandHandler("mode", self._cmd_mode))
        app.add_handler(CommandHandler("risk", self._cmd_risk))
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
                InlineKeyboardButton("Pause", callback_data="control:pause"),
                InlineKeyboardButton("Resume", callback_data="control:resume"),
            ],
            [
                InlineKeyboardButton("Shadow ON", callback_data="mode:shadow"),
            ],
            [InlineKeyboardButton("Emergency stop", callback_data="control:stop")],
            [InlineKeyboardButton("Back", callback_data="view:status")],
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
        ]
        await self._reply(update, "\n".join(lines))

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
        candles = db_diag.get("candles_by_interval", {})
        latest_1m = db_diag.get("latest_candle_1m")
        latest_str = latest_1m.strftime("%H:%M:%S UTC") if latest_1m else "none"

        model_info = diag.get("model", {}) or {}
        champion_ver = model_info.get("champion_version", "none")
        challenger_ver = model_info.get("challenger_version", "none")
        last_training = model_info.get("last_training", "never")
        samples = model_info.get("training_samples", 0)
        wf_exp = model_info.get("walk_forward_expectancy", "n/a")
        drift = model_info.get("drift_status", "n/a")

        lines = [
            "<b>🗄 БАЗА И МОДЕЛЬ</b>",
            "",
            f"БД: {db_icon} {'connected' if connected else 'unavailable'}",
            f"Последняя свеча 1m: <code>{latest_str}</code>",
            f"Свечей 1m:  <code>{candles.get('1', 0)}</code>",
            f"Свечей 5m:  <code>{candles.get('5', 0)}</code>",
            f"Свечей 15m: <code>{candles.get('15', 0)}</code>",
            f"Свечей 1h:  <code>{candles.get('60', 0)}</code>",
            f"Feature snapshots:   <code>{db_diag.get('feature_snapshots', 0)}</code>",
            f"Prediction outcomes: <code>{db_diag.get('prediction_outcomes', 0)}</code>",
            "",
            "<b>Модель</b>",
            f"Последнее обучение: <code>{last_training}</code>",
            f"Training samples:   <code>{samples}</code>",
            f"Champion version:   <code>{champion_ver}</code>",
            f"Challenger version: <code>{challenger_ver}</code>",
            f"Walk-forward exp:   <code>{wf_exp}</code>",
            f"Drift:              <code>{drift}</code>",
            "Model live decisions: <b>disabled</b>",
        ]
        if db_diag.get("error"):
            lines.append(f"\n<i>Error: {db_diag['error']}</i>")

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
            "/diagnostics — counters & loop timing\n"
            "/help        — this message\n" + ctrl_section
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
