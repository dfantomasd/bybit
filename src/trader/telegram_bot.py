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

import asyncio
import hashlib
import html
import inspect
import io
import json
import os
import re
import secrets
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, cast

import structlog
from starlette.requests import Request
from starlette.responses import Response
from telegram import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from trader.domain.enums import RiskProfile
from trader.domain.models import Balance, HealthStatus, Position
from trader.operator_priorities import canary_readiness_priority_text, full_priority_overview

log = structlog.get_logger(__name__)

# Inline confirm buttons and pending /confirm actions expire after this long.
_CONFIRM_TTL_SECONDS = 300.0
_CONFIRM_MAX_PENDING = 1000
_TELEGRAM_MESSAGE_LIMIT = 4000
_POLLING_LOCK_KEY_PREFIX = "bybit-trader:telegram-polling"

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
    db_diagnostics_provider: Callable[..., Awaitable[dict[str, Any]]] | None = None
    start_training: Callable[[int, int, float], Awaitable[str]] | None = None
    start_training_all: Callable[[], Awaitable[str]] | None = None
    promote_model: Callable[[str], Awaitable[str]] | None = None
    runtime_settings: Callable[[], dict[str, Any]] | None = None
    set_runtime_setting: Callable[[str, Any], Awaitable[str]] | None = None
    # Safety gate: when False, Telegram cannot escalate to a riskier profile
    allow_risk_increase: bool = False
    # /healthcheck data: signals/fills/blockers/avg net edge
    healthcheck_provider: Callable[[], Awaitable[dict[str, Any]]] | None = None
    # /trades data: recent closed trades
    recent_trades_provider: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None
    # /buckets data: regime-bucket expectancy stats
    bucket_stats_provider: Callable[[], Awaitable[dict[str, Any]]] | None = None
    # Strategy loss diagnostics
    pnl_analysis_provider: Callable[[], Awaitable[dict[str, Any]]] | None = None
    compare_provider: Callable[[], Awaitable[dict[str, Any]]] | None = None
    worst_trades_provider: Callable[[int], Awaitable[list[dict[str, Any]]]] | None = None
    costs_detailed_provider: Callable[[], Awaitable[dict[str, Any]]] | None = None
    model_performance_provider: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None
    champion_health_provider: Callable[[], Awaitable[dict[str, Any]]] | None = None
    attribution_provider: Callable[[int], Awaitable[list[dict[str, Any]]]] | None = None
    best_challenger_provider: Callable[[], Awaitable[str | None]] | None = None
    enrich_db_diag_fallbacks: Callable[[dict[str, Any]], None] | None = None
    # Persistent Telegram subscriptions (survive restarts)
    add_subscription: Callable[[int], Awaitable[None]] | None = None
    remove_subscription: Callable[[int], Awaitable[None]] | None = None
    load_subscriptions: Callable[[], Awaitable[list[int]]] | None = None


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
    redis_url: str = ""
    delivery_mode: str = "polling"
    webhook_url: str = ""
    webhook_secret: str = ""
    polling_lock_ttl_s: int = 45
    polling_lock_wait_s: int = 90
    polling_conflict_recovery_wait_s: int = 10
    polling_watchdog_interval_s: float = 30.0
    polling_zombie_silence_s: float = 180.0


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
        net_results_provider: Callable[[], Awaitable[dict[str, Any]]] | None = None,
    ) -> None:
        self._config = config
        self._health_provider = health_provider
        self._adapter_factory = adapter_factory
        self._controller = controller
        self._net_results_provider = net_results_provider
        self._app: Application[Any, Any, Any, Any, Any, Any] | None = None

        # Pre-populate subscribed set so allowed chats receive push notifications
        # immediately after restart without requiring /start.
        self._subscribed: set[int] = set(config.allowed_chat_ids)

        # Pending confirmations: chat_id → (action_name, coroutine_factory, created_at)
        self._pending: dict[int, tuple[str, Callable[[], Awaitable[None]], datetime]] = {}
        # One-shot nonces for inline confirm buttons: nonce → (action, created_at)
        self._confirm_nonces: dict[str, tuple[str, datetime]] = {}
        # Chats currently in the "enter custom limit value" flow.
        # None means the generic "key value" form; otherwise store the key that
        # should receive the next numeric message from this chat.
        self._awaiting_custom_limit: dict[int, str | None] = {}

        # Single dynamic message state
        self._dashboard_message_id: int | None = None
        self._dashboard_chat_id: int | None = None
        self._started_at: datetime | None = None
        self._last_callback_at: datetime | None = None
        self._last_handler_at: datetime | None = None
        self._last_polling_error_at: datetime | None = None
        self._last_polling_error: str | None = None
        self._polling_conflict_count: int = 0
        self._polling_network_error_count: int = 0
        self._polling_disabled_reason: str | None = None
        self._polling_lock_key: str | None = None
        self._polling_lock_token: str | None = None
        self._polling_lock_refresh_task: asyncio.Task[None] | None = None
        self._polling_conflict_stop_task: asyncio.Task[None] | None = None
        self._polling_recovery_task: asyncio.Task[None] | None = None
        self._polling_recovery_pending: bool = False
        self._polling_watchdog_task: asyncio.Task[None] | None = None
        self._webhook_route_mounted: bool = False
        self._redis_client: Any | None = None
        self._model_performance_cache: list[dict[str, Any]] = []

    def _uses_webhook(self) -> bool:
        return self._config.delivery_mode == "webhook"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self._config.token and self._config.allowed_chat_ids)

    async def start(self, http_app: Any | None = None) -> bool:
        if not self.enabled:
            log.info("telegram_bot_disabled")
            return False
        if not self._uses_webhook():
            if not await self._acquire_polling_lock():
                return False

        if self._app is not None:
            await self._teardown_app()

        app = Application.builder().token(self._config.token).build()

        # Observability
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("menu", self._cmd_menu))
        app.add_handler(CommandHandler("help", self._cmd_help))
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("balance", self._cmd_balance))
        app.add_handler(CommandHandler("positions", self._cmd_positions))
        app.add_handler(CommandHandler("signals", self._cmd_signals))
        app.add_handler(CommandHandler("regime", self._cmd_regime))
        app.add_handler(CommandHandler("symbols", self._cmd_symbols))
        app.add_handler(CommandHandler("pnl", self._cmd_pnl))
        app.add_handler(CommandHandler("net", self._cmd_net_results))
        app.add_handler(CommandHandler("costs", self._cmd_costs))
        app.add_handler(CommandHandler("costs_detailed", self._cmd_costs_detailed))
        app.add_handler(CommandHandler("pnl_analysis", self._cmd_pnl_analysis))
        app.add_handler(CommandHandler("compare", self._cmd_compare))
        app.add_handler(CommandHandler("worst", self._cmd_worst))
        app.add_handler(CommandHandler("strategy_report", self._cmd_strategy_report))
        app.add_handler(CommandHandler("report", self._cmd_strategy_report))
        app.add_handler(CommandHandler("model_performance", self._cmd_model_performance))
        app.add_handler(CommandHandler("champion_health", self._cmd_champion_health))
        app.add_handler(CommandHandler("diagnostics", self._cmd_diagnostics))
        app.add_handler(CommandHandler("deep_report", self._cmd_deep_report))
        app.add_handler(CommandHandler("deep_report_text", self._cmd_deep_report_text))
        app.add_handler(CommandHandler("attribution", self._cmd_attribution))
        app.add_handler(CommandHandler("canary", self._cmd_canary_ready))
        app.add_handler(CommandHandler("priorities", self._cmd_priorities))
        app.add_handler(CommandHandler("model_help", self._cmd_model_help))
        app.add_handler(CommandHandler("db", self._cmd_db_model))
        app.add_handler(CommandHandler("model", self._cmd_db_model))
        app.add_handler(CommandHandler("trades", self._cmd_trades))
        app.add_handler(CommandHandler("healthcheck", self._cmd_healthcheck))
        app.add_handler(CommandHandler("buckets", self._cmd_buckets))
        app.add_handler(CommandHandler("subscribe", self._cmd_subscribe))
        app.add_handler(CommandHandler("unsubscribe", self._cmd_unsubscribe))

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
        # Free-text handler for the "custom limit value" flow
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text))
        app.add_error_handler(self._on_error)

        # Load persisted subscriptions so push notifications survive restarts
        if self._controller is not None and self._controller.load_subscriptions is not None:
            try:
                for chat_id in await self._controller.load_subscriptions():
                    self._subscribed.add(chat_id)
                    self._config.allowed_chat_ids.add(chat_id)
                log.info("telegram_subscriptions_loaded", count=len(self._subscribed))
            except Exception as exc:
                log.warning("telegram_subscriptions_load_failed", error=str(exc))

        try:
            await app.initialize()
            await app.start()
            if self._uses_webhook():
                if http_app is None:
                    self._polling_disabled_reason = "webhook_http_app_missing"
                    log.error("telegram_webhook_requires_http_app")
                    await self._teardown_app()
                    return False
                self._mount_webhook_route(http_app)
                await self._activate_webhook(app)
                # Same watchdog loop re-registers webhook after rolling-deploy overlap.
                self._start_polling_watchdog()
            else:
                await self._start_polling(app)
                self._start_polling_watchdog()
            self._app = app
            self._started_at = datetime.now(tz=UTC)
            self._polling_conflict_count = 0
            self._polling_disabled_reason = None
            self._polling_recovery_pending = False
            log.info(
                "telegram_bot_started",
                allowed_chats=len(self._config.allowed_chat_ids),
                delivery_mode=self._config.delivery_mode,
            )
            return True
        except Exception:
            try:
                if getattr(app, "running", False):
                    await app.stop()
                await app.shutdown()
            except Exception as exc:
                log.debug("telegram_bot_start_cleanup_failed", error=str(exc))
            if not self._uses_webhook():
                await self._release_polling_lock()
            raise

    async def stop(self) -> None:
        watchdog_task = self._polling_watchdog_task
        self._polling_watchdog_task = None
        if watchdog_task is not None:
            watchdog_task.cancel()
            await asyncio.gather(watchdog_task, return_exceptions=True)
        recovery_task = self._polling_recovery_task
        self._polling_recovery_task = None
        if recovery_task is not None:
            recovery_task.cancel()
            await asyncio.gather(recovery_task, return_exceptions=True)
        conflict_stop_task = self._polling_conflict_stop_task
        self._polling_conflict_stop_task = None
        if conflict_stop_task is not None:
            conflict_stop_task.cancel()
            await asyncio.gather(conflict_stop_task, return_exceptions=True)
        await self._teardown_app()
        await self._release_polling_lock()
        log.info("telegram_bot_stopped")

    async def _teardown_app(self) -> None:
        """Stop updater/application without releasing the optional Redis lock."""
        app = self._app
        self._app = None
        if app is None:
            return
        try:
            if self._uses_webhook():
                # Keep webhook registered on shutdown. During Render rolling deploys the
                # replacement instance already called set_webhook; deleting here leaves
                # Telegram deaf until the new process finishes long startup work.
                log.info("telegram_webhook_preserved_on_shutdown")
            elif app.updater is not None and getattr(app.updater, "running", False):
                await app.updater.stop()
            if getattr(app, "running", False):
                await app.stop()
            await app.shutdown()
        except Exception as exc:
            log.warning("telegram_bot_teardown_failed", error=str(exc))

    def _mount_webhook_route(self, http_app: Any) -> None:
        if self._webhook_route_mounted:
            return
        from fastapi import HTTPException

        bot_ref = self

        @http_app.get("/telegram/livez")
        async def telegram_livez() -> dict[str, Any]:
            return bot_ref.health_snapshot()

        @http_app.post("/telegram/webhook")
        async def telegram_webhook(incoming: Request) -> Response:
            secret = bot_ref._config.webhook_secret.strip()
            if secret:
                provided = incoming.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
                if provided != secret:
                    raise HTTPException(status_code=403, detail="invalid webhook secret")
            tg_app = bot_ref._app
            if tg_app is None:
                raise HTTPException(status_code=503, detail="telegram not ready")
            try:
                data = await incoming.json()
                update = Update.de_json(data, tg_app.bot)
                if update is not None:
                    kind = "message" if update.message else "callback_query" if update.callback_query else "other"
                    log.info(
                        "telegram_webhook_update_received",
                        update_id=update.update_id,
                        kind=kind,
                    )

                    async def _process() -> None:
                        try:
                            await tg_app.process_update(update)
                            log.info("telegram_webhook_update_processed", update_id=update.update_id)
                        except Exception as proc_exc:
                            log.warning(
                                "telegram_webhook_background_process_failed",
                                update_id=update.update_id,
                                error=str(proc_exc),
                            )

                    asyncio.create_task(_process(), name=f"telegram-update-{update.update_id}")
            except HTTPException:
                raise
            except Exception as exc:
                log.warning("telegram_webhook_process_failed", error=str(exc))
                raise HTTPException(status_code=500, detail="telegram process failed") from exc
            return Response(status_code=200)

        self._webhook_route_mounted = True
        log.info("telegram_webhook_route_mounted", path="/telegram/webhook")

    async def _activate_webhook(self, app: Application[Any, Any, Any, Any, Any, Any]) -> None:
        url = self._config.webhook_url.strip()
        if not url:
            raise RuntimeError("telegram webhook URL is empty")
        secret = self._config.webhook_secret.strip() or None
        await app.bot.set_webhook(
            url=url,
            secret_token=secret,
            drop_pending_updates=True,
        )
        log.info("telegram_webhook_registered", url=url)

    async def _acquire_polling_lock(self) -> bool:
        """Acquire a distributed Telegram polling lease when Redis is configured."""
        redis_url = self._config.redis_url.strip().strip("\"'")
        if not redis_url:
            log.info("telegram_polling_lock_disabled", reason="redis_url_empty")
            return True

        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(redis_url, socket_connect_timeout=2.0, socket_timeout=2.0)
            token_hash = hashlib.sha256(self._config.token.encode("utf-8")).hexdigest()[:16]
            key = f"{_POLLING_LOCK_KEY_PREFIX}:{token_hash}"
            owner = secrets.token_hex(16)
            ttl = max(10, int(self._config.polling_lock_ttl_s))
            wait_s = max(0, int(self._config.polling_lock_wait_s))
            deadline = asyncio.get_running_loop().time() + wait_s
            while True:
                acquired = await client.set(key, owner, nx=True, ex=ttl)
                if acquired:
                    self._redis_client = client
                    self._polling_lock_key = key
                    self._polling_lock_token = owner
                    self._polling_lock_refresh_task = asyncio.create_task(
                        self._refresh_polling_lock(ttl),
                        name="telegram-polling-lock-refresh",
                    )
                    log.info("telegram_polling_lock_acquired", ttl_s=ttl)
                    return True
                if asyncio.get_running_loop().time() >= deadline:
                    self._polling_disabled_reason = "polling_lock_timeout"
                    log.warning("telegram_polling_lock_timeout", wait_s=wait_s, ttl_s=ttl)
                    await self._close_redis_client(client)
                    return False
                await asyncio.sleep(min(2.0, max(0.25, ttl / 10)))
        except Exception as exc:
            # Telegram remains usable even if optional Redis is unavailable; the
            # conflict counter will still expose any duplicate pollers.
            self._polling_disabled_reason = f"polling_lock_error:{exc}"
            log.warning("telegram_polling_lock_failed_open", error=str(exc))
            return True

    async def _refresh_polling_lock(self, ttl_s: int) -> None:
        assert self._redis_client is not None
        assert self._polling_lock_key is not None
        assert self._polling_lock_token is not None
        script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('expire', KEYS[1], ARGV[2])
        end
        return 0
        """
        interval = max(3.0, ttl_s / 3)
        while True:
            await asyncio.sleep(interval)
            try:
                refreshed = await self._redis_client.eval(
                    script,
                    1,
                    self._polling_lock_key,
                    self._polling_lock_token,
                    int(ttl_s),
                )
                if not refreshed:
                    self._polling_disabled_reason = "polling_lock_lost"
                    log.warning("telegram_polling_lock_lost")
                    self._polling_lock_key = None
                    self._polling_lock_token = None
                    await self._stop_app_after_lock_loss()
                    return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._polling_disabled_reason = f"polling_lock_refresh_failed:{exc}"
                log.warning("telegram_polling_lock_refresh_failed", error=str(exc))
                self._polling_lock_key = None
                self._polling_lock_token = None
                await self._stop_app_after_lock_loss()
                return

    async def _stop_app_after_lock_loss(self) -> None:
        """Stop Telegram polling when the distributed lock is lost."""
        if self._app is None:
            return
        await self._teardown_app()
        log.warning("telegram_bot_stopped_lock_lost")

    async def _release_polling_lock(self) -> None:
        task = self._polling_lock_refresh_task
        self._polling_lock_refresh_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        client = self._redis_client
        key = self._polling_lock_key
        owner = self._polling_lock_token
        self._redis_client = None
        self._polling_lock_key = None
        self._polling_lock_token = None
        if client is None:
            return
        try:
            if key and owner:
                script = """
                if redis.call('get', KEYS[1]) == ARGV[1] then
                    return redis.call('del', KEYS[1])
                end
                return 0
                """
                await client.eval(script, 1, key, owner)
        except Exception as exc:
            log.debug("telegram_polling_lock_release_failed", error=str(exc))
        finally:
            await self._close_redis_client(client)

    @staticmethod
    async def _close_redis_client(client: Any) -> None:
        close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if not callable(close):
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    async def _start_polling(self, app: Application[Any, Any, Any, Any, Any, Any]) -> None:
        if app.updater is None:
            raise RuntimeError("Telegram updater was not created")
        try:
            await app.bot.delete_webhook(drop_pending_updates=True)
        except Exception as exc:
            log.debug("telegram_delete_webhook_failed", error=str(exc))
        await app.updater.start_polling(
            drop_pending_updates=True,
            error_callback=self._polling_error_callback,
        )

    async def _stop_polling_only(self) -> None:
        app = self._app
        if app is None or app.updater is None:
            return
        if getattr(app.updater, "running", False):
            await app.updater.stop()

    async def _restart_polling(self) -> bool:
        app = self._app
        if app is None or app.updater is None:
            return False
        updater = app.updater
        if getattr(updater, "running", False):
            await updater.stop()
        await self._start_polling(app)
        self._polling_conflict_count = 0
        self._polling_disabled_reason = None
        return True

    async def _recover_polling_after_conflicts(self) -> None:
        """Wait for rolling-deploy overlap to end, then resume getUpdates."""
        wait_s = max(5.0, float(self._config.polling_conflict_recovery_wait_s))
        await self._stop_polling_only()
        log.warning("telegram_polling_recovery_scheduled", wait_seconds=wait_s)
        try:
            await asyncio.sleep(wait_s)
        except asyncio.CancelledError:
            raise
        if self._app is None:
            started = await self.start()
            if started:
                log.info("telegram_polling_recovered_via_full_restart")
            else:
                log.warning("telegram_polling_recovery_full_restart_failed")
            return
        try:
            await self._restart_polling()
            log.info("telegram_polling_recovered_after_conflicts")
        except Exception as exc:
            self._polling_disabled_reason = f"polling_recovery_failed:{exc}"
            log.warning("telegram_polling_recovery_failed", error=str(exc))
            started = await self.start()
            if started:
                log.info("telegram_polling_recovered_via_full_restart_after_failure")

    def _polling_looks_zombie(self) -> bool:
        """Detect updater.running=True while no user interaction is flowing."""
        if self._app is None or self._started_at is None:
            return False
        uptime_s = (datetime.now(tz=UTC) - self._started_at).total_seconds()
        silence_s = max(60.0, float(self._config.polling_zombie_silence_s))
        if uptime_s < silence_s:
            return False
        updater = getattr(self._app, "updater", None)
        if updater is None or not getattr(updater, "running", False):
            return False
        if self._polling_conflict_count > 0 and self._last_handler_at is None:
            return True
        if self._last_handler_at is None:
            return False
        return (datetime.now(tz=UTC) - self._last_handler_at).total_seconds() >= silence_s

    async def ensure_polling_running(self) -> None:
        """Restart polling/webhook if a deploy conflict or crash left the bot deaf."""
        if not self.enabled:
            return
        if self._uses_webhook():
            await self._ensure_webhook_active()
            return
        if self._polling_recovery_pending and (
            self._polling_recovery_task is None or self._polling_recovery_task.done()
        ):
            self._polling_recovery_pending = False
            self._schedule_polling_recovery_after_conflicts()
        snap = self.health_snapshot()
        zombie = self._polling_looks_zombie()
        if snap.get("polling_running") and not zombie:
            return
        if self._polling_recovery_task is not None and not self._polling_recovery_task.done():
            return
        reason = snap.get("polling_disabled_reason")
        if zombie:
            reason = "polling_zombie"
            log.warning("telegram_polling_zombie_detected", last_handler_at=self._last_handler_at)
        else:
            log.warning("telegram_polling_not_running", reason=reason)
        if self._app is not None and not zombie:
            try:
                if await self._restart_polling():
                    log.info("telegram_polling_restarted_by_watchdog")
                    return
            except Exception as exc:
                log.warning("telegram_polling_watchdog_restart_failed", error=str(exc))
        elif self._app is not None and zombie:
            await self._teardown_app()
        started = await self.start()
        if started:
            log.info("telegram_polling_restarted_by_watchdog_full_start")
        else:
            log.warning("telegram_polling_watchdog_full_start_failed", health=self.health_snapshot())

    async def refresh_delivery(self) -> None:
        """Re-register webhook after long startup or deploy."""
        if not self.enabled or self._app is None:
            return
        if self._uses_webhook():
            await self._activate_webhook(self._app)
            log.info("telegram_webhook_refreshed")
            return
        await self.ensure_polling_running()

    async def _ensure_webhook_active(self) -> None:
        app = self._app
        if app is None:
            return
        expected = self._config.webhook_url.strip()
        if not expected:
            return
        try:
            info = await app.bot.get_webhook_info()
            if info.url != expected:
                log.warning("telegram_webhook_mismatch", current=info.url, expected=expected)
                await self._activate_webhook(app)
        except Exception as exc:
            log.warning("telegram_webhook_health_check_failed", error=str(exc))

    def _start_polling_watchdog(self) -> None:
        task = self._polling_watchdog_task
        if task is not None and not task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._polling_watchdog_task = loop.create_task(
            self._polling_watchdog_loop(),
            name="telegram-polling-watchdog",
        )

    async def _polling_watchdog_loop(self) -> None:
        interval = max(10.0, float(self._config.polling_watchdog_interval_s))
        while True:
            try:
                await asyncio.sleep(interval)
                await self.ensure_polling_running()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.debug("telegram_polling_watchdog_tick_failed", error=str(exc))

    def _touch_handler_activity(self) -> None:
        self._last_handler_at = datetime.now(tz=UTC)

    def _schedule_polling_recovery_after_conflicts(self) -> None:
        """Rolling deploys briefly run two pollers; recover instead of staying deaf."""
        task = self._polling_recovery_task
        if task is not None and not task.done():
            return
        self._polling_disabled_reason = "polling_conflict_recovery_pending"
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._polling_recovery_pending = True
            log.warning("telegram_polling_conflict_recovery_deferred")
            return
        self._polling_recovery_task = loop.create_task(
            self._recover_polling_after_conflicts(),
            name="telegram-polling-conflict-recovery",
        )

    def _polling_error_callback(self, error: Exception) -> None:
        """Suppress Conflict errors during rolling redeploys; log the rest."""
        from telegram.error import Conflict, NetworkError

        now = datetime.now(tz=UTC)
        self._last_polling_error_at = now
        self._last_polling_error = str(error)
        if isinstance(error, Conflict):
            # Expected during Render rolling deploys — old instance displaced
            self._polling_conflict_count += 1
            log_method = log.warning if self._polling_conflict_count >= 3 else log.debug
            log_method(
                "telegram_polling_conflict_suppressed",
                conflicts=self._polling_conflict_count,
                error=str(error),
            )
            if self._polling_conflict_count >= 3:
                self._schedule_polling_recovery_after_conflicts()
            elif self._started_at is not None:
                uptime_s = (now - self._started_at).total_seconds()
                if uptime_s <= 180.0 and self._polling_conflict_count >= 1:
                    self._schedule_polling_recovery_after_conflicts()
            return
        if isinstance(error, NetworkError):
            self._polling_network_error_count += 1
            log.warning(
                "telegram_polling_network_error",
                count=self._polling_network_error_count,
                error=str(error),
            )
            return
        log.error("telegram_polling_error", error=str(error))

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Log handler crashes and give callback users a visible fallback."""
        error = getattr(context, "error", None)
        log.warning("telegram_handler_error", error=str(error), update_type=type(update).__name__)
        if not isinstance(update, Update) or update.callback_query is None:
            return
        try:
            await self._button_reply(
                update,
                "⚠️ Кнопка не выполнилась из-за внутренней ошибки. Попробуйте ещё раз.",
                reply_markup=self._main_menu(),
            )
        except Exception as exc:
            log.debug("telegram_handler_error_reply_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Push notifications (called by app.py)
    # ------------------------------------------------------------------

    async def notify(self, text: str) -> None:
        """Send a push message to all subscribed chats."""
        if self._app is None:
            return
        for chat_id in list(self._subscribed):
            try:
                chunks = self._split_message(text)
                for chunk in chunks:
                    await self._app.bot.send_message(
                        chat_id=chat_id,
                        text=chunk,
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
            f"{html.escape(str(entry.rationale))}"
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
        chat_id = self._chat_id(update)
        if chat_id is None or chat_id not in self._config.allowed_chat_ids:
            if update.effective_message is not None:
                suffix = f" Chat ID: {chat_id}" if chat_id is not None else ""
                await update.effective_message.reply_text(f"Доступ запрещен.{suffix}")
            elif chat_id is not None:
                await self._send_direct_message(chat_id, f"Доступ запрещен. Chat ID: {chat_id}")
            log.warning("telegram_unauthorised_chat", chat_id=chat_id)
            return False
        self._touch_handler_activity()
        return True

    async def _reply(
        self,
        update: Update,
        text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> Any | None:
        if update.effective_message is not None:
            return await self._reply_chunks(update.effective_message, text, reply_markup=reply_markup)
        return None

    @staticmethod
    def _split_message(text: str, limit: int = _TELEGRAM_MESSAGE_LIMIT) -> list[str]:
        """Split Telegram HTML text conservatively below the API message limit."""

        if len(text) <= limit:
            return [text]
        chunks: list[str] = []
        current = ""
        for line in text.splitlines(keepends=True):
            if len(current) + len(line) <= limit:
                current += line
                continue
            if current:
                chunks.append(current.rstrip())
                current = ""
            while len(line) > limit:
                chunks.append(line[:limit].rstrip())
                line = line[limit:]
            current = line
        if current:
            chunks.append(current.rstrip())
        return chunks or [text[:limit]]

    @staticmethod
    def _plain_text(text: str) -> str:
        """Convert Telegram HTML-ish text into readable plaintext fallback."""

        return html.unescape(re.sub(r"<[^>]+>", "", text))

    async def _safe_reply_text(
        self,
        message: Any,
        text: str,
        *,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> Any:
        from telegram.error import BadRequest

        try:
            return await message.reply_text(
                text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_markup=reply_markup,
            )
        except BadRequest as exc:
            log.debug("telegram.reply_html_failed_plaintext_fallback", error=str(exc))
            return await message.reply_text(
                self._plain_text(text),
                disable_web_page_preview=True,
                reply_markup=reply_markup,
            )

    async def _send_direct_message(
        self,
        chat_id: int | None,
        text: str,
        *,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> Any | None:
        """Send via bot API when callback message edit/reply is unavailable."""
        if chat_id is None or self._app is None:
            return None
        from telegram.error import BadRequest

        chunks = self._split_message(text)
        sent = None
        for index, chunk in enumerate(chunks):
            markup = reply_markup if index == len(chunks) - 1 else None
            try:
                sent = await self._app.bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                    reply_markup=markup,
                )
            except BadRequest as exc:
                log.debug("telegram.direct_send_html_failed_plaintext_fallback", error=str(exc))
                sent = await self._app.bot.send_message(
                    chat_id=chat_id,
                    text=self._plain_text(chunk),
                    disable_web_page_preview=True,
                    reply_markup=markup,
                )
        return sent

    async def _reply_chunks(
        self,
        message: Any,
        text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> Any | None:
        chunks = self._split_message(text)
        sent = None
        for index, chunk in enumerate(chunks):
            sent = await self._safe_reply_text(
                message,
                chunk,
                reply_markup=reply_markup if index == len(chunks) - 1 else None,
            )
        return sent

    def _chat_id(self, update: Update) -> int | None:
        chat = getattr(update, "effective_chat", None)
        chat_id = getattr(chat, "id", None)
        if chat_id is not None:
            return int(chat_id)
        query = getattr(update, "callback_query", None)
        return self._message_chat_id(getattr(query, "message", None))

    @staticmethod
    def _message_chat_id(message: Any) -> int | None:
        chat_id = getattr(message, "chat_id", None)
        if chat_id is None:
            msg_chat = getattr(message, "chat", None)
            chat_id = getattr(msg_chat, "id", None)
        return int(chat_id) if chat_id is not None else None

    def health_snapshot(self) -> dict[str, Any]:
        """Return lightweight Telegram polling/callback health for app diagnostics."""
        updater = getattr(self._app, "updater", None)
        running = bool(getattr(self._app, "running", False))
        polling = bool(getattr(updater, "running", False)) if updater is not None else False
        webhook_mode = self._uses_webhook()
        return {
            "enabled": self.enabled,
            "delivery_mode": self._config.delivery_mode,
            "webhook_url": self._config.webhook_url if webhook_mode else None,
            "app_running": running,
            "polling_running": polling if not webhook_mode else None,
            "webhook_active": webhook_mode and running,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "last_handler_at": self._last_handler_at.isoformat() if self._last_handler_at else None,
            "last_callback_at": self._last_callback_at.isoformat() if self._last_callback_at else None,
            "last_polling_error_at": (self._last_polling_error_at.isoformat() if self._last_polling_error_at else None),
            "last_polling_error": self._last_polling_error,
            "polling_conflict_count": self._polling_conflict_count,
            "polling_network_error_count": self._polling_network_error_count,
            "polling_disabled_reason": self._polling_disabled_reason,
            "polling_lock_owner": bool(self._polling_lock_token),
            "subscribed_chats": len(self._subscribed),
        }

    @staticmethod
    def _model_horizon_and_gate(
        db_diag: dict[str, Any],
        metrics: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        metrics = metrics or {}
        training_config = db_diag.get("training_config", {}) or {}
        raw_horizon = (
            db_diag.get("model_gate_horizon_minutes")
            or metrics.get("horizon_minutes")
            or metrics.get("label_horizon_minutes")
            or training_config.get("auto_train_horizon_minutes")
            or 5
        )
        try:
            horizon = int(raw_horizon)
        except (TypeError, ValueError):
            horizon = 15
        by_horizon = db_diag.get("shadow_gate_by_horizon", {}) or {}
        gate = (
            by_horizon.get(str(horizon))
            or db_diag.get(f"shadow_gate_{horizon}m")
            or db_diag.get("shadow_gate_15m")
            or {}
        )
        try:
            horizon = int(gate.get("horizon_minutes") or horizon)
        except (TypeError, ValueError, AttributeError):
            pass
        return horizon, gate if isinstance(gate, dict) else {}

    def _train_defaults(self) -> tuple[int, int, float]:
        """Return (min_samples, horizon_minutes, label_bps) aligned with auto-trainer config."""
        runtime = self._controller.runtime_settings() if self._controller and self._controller.runtime_settings else {}
        min_samples = int(runtime.get("model_auto_train_min_samples") or 1000)
        horizon = int(runtime.get("model_auto_train_horizon_minutes") or 5)
        label_bps = float(runtime.get("model_auto_train_label_bps") or 2.0)
        return min_samples, horizon, label_bps

    def _train_callback(self, min_samples: int | None = None) -> str:
        default_min, horizon, label_bps = self._train_defaults()
        samples = min_samples if min_samples is not None else default_min
        return f"train:{samples}:{horizon}:{label_bps:g}"

    def _mode_indicator(self) -> str:
        """Compact current-mode line for menu headers."""
        mode = (self._config.trading_mode or "SHADOW").upper()
        is_shadow = self._controller.is_shadow() if self._controller is not None else True
        is_paused = self._controller.is_paused() if self._controller is not None else False
        if mode == "LIVE":
            badge = "🔴 LIVE — реальные деньги"
        elif mode == "CANARY_LIVE":
            badge = "🟠 CANARY — реальные деньги, малый размер"
        elif is_shadow:
            badge = "🟢 SHADOW — ордера не отправляются"
        else:
            badge = f"⚪ {mode}"
        pause_str = " | ⏸ пауза" if is_paused else ""
        venue = "testnet" if self._config.bybit_use_testnet else "Bybit mainnet"
        return f"{badge} | {venue}{pause_str}"

    def _menu_text(self) -> str:
        return f"<b>🏠 Bybit AI Trader</b>\n{self._mode_indicator()}\n\nВыберите раздел:"

    def _main_menu(self) -> InlineKeyboardMarkup:
        rows = [
            [
                InlineKeyboardButton("🏠 Статус и режим", callback_data="view:status_mode"),
                InlineKeyboardButton("💰 Баланс и позиции", callback_data="view:balance_positions"),
            ],
            [
                InlineKeyboardButton("📈 Торговля и сигналы", callback_data="view:trading"),
                InlineKeyboardButton("⚙️ Настройки", callback_data="view:settings"),
            ],
            [
                InlineKeyboardButton("🧠 Модель и обучение", callback_data="view:model"),
                InlineKeyboardButton("🩺 Диагностика", callback_data="view:diagnostics"),
            ],
            [InlineKeyboardButton("❓ Помощь", callback_data="view:help")],
            [InlineKeyboardButton("🔄 Обновить", callback_data="view:menu")],
        ]
        return InlineKeyboardMarkup(rows)

    def _home_row(self) -> list[InlineKeyboardButton]:
        return [InlineKeyboardButton("🏠 Главное меню", callback_data="view:menu")]

    def _status_mode_menu(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("📊 Статус", callback_data="view:status")],
                [
                    InlineKeyboardButton("⏸ Пауза", callback_data="control:pause"),
                    InlineKeyboardButton("▶️ Возобновить", callback_data="control:resume"),
                ],
                [InlineKeyboardButton("🎚 Изменить риск-профиль", callback_data="view:risk_profiles")],
                self._home_row(),
            ]
        )

    def _balance_positions_menu(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("💰 Баланс", callback_data="view:balance"),
                    InlineKeyboardButton("📂 Позиции", callback_data="view:positions"),
                ],
                [InlineKeyboardButton("📈 Результаты", callback_data="view:pnl")],
                self._home_row(),
            ]
        )

    def _trading_menu(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("🧠 Последние сигналы", callback_data="view:signals"),
                    InlineKeyboardButton("📜 Закрытые сделки", callback_data="view:trades"),
                ],
                [
                    InlineKeyboardButton("📈 Чистый PnL", callback_data="view:pnl"),
                    InlineKeyboardButton("🚦 Готовность CANARY", callback_data="view:canary"),
                ],
                [InlineKeyboardButton("📑 Отчет стратегии", callback_data="view:strategy_report")],
                self._home_row(),
            ]
        )

    def _settings_menu(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("🎚 Лимиты", callback_data="control:limits"),
                    InlineKeyboardButton("✅ Выбрать пары", callback_data="view:symbol_select"),
                ],
                [
                    InlineKeyboardButton("⏸ Пауза", callback_data="control:pause"),
                    InlineKeyboardButton("▶️ Возобновить", callback_data="control:resume"),
                ],
                [InlineKeyboardButton("🔎 Сканер", callback_data="view:symbols")],
                self._home_row(),
            ]
        )

    def _model_menu(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("🗄 База и модель", callback_data="view:db_model"),
                    InlineKeyboardButton("📉 История моделей", callback_data="view:model_performance"),
                ],
                [InlineKeyboardButton("🏆 Champion health", callback_data="view:champion_health")],
                [
                    InlineKeyboardButton(
                        f"🧠 Обучить {self._train_defaults()[0]}",
                        callback_data=self._train_callback(),
                    ),
                    InlineKeyboardButton("🏆 Промоутить", callback_data="control:promote"),
                ],
                [
                    InlineKeyboardButton("⚖️ Compare", callback_data="view:compare"),
                    InlineKeyboardButton("❓ Как читать модель", callback_data="view:model_help"),
                ],
                self._home_row(),
            ]
        )

    def _diagnostics_menu(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("🩺 Healthcheck", callback_data="view:healthcheck"),
                    InlineKeyboardButton("🖥 Нагрузка", callback_data="view:load_diagnostics"),
                ],
                [
                    InlineKeyboardButton("🧾 Издержки детально", callback_data="view:costs_detailed"),
                    InlineKeyboardButton("💸 Издержки", callback_data="view:costs"),
                ],
                [
                    InlineKeyboardButton("🔬 PnL-анализ", callback_data="view:pnl_analysis"),
                    InlineKeyboardButton("⚠️ Худшие сделки", callback_data="view:worst"),
                ],
                [InlineKeyboardButton("🧾 Полная сводка для анализа", callback_data="view:deep_report")],
                [InlineKeyboardButton("📌 Приоритеты", callback_data="view:priorities")],
                self._home_row(),
            ]
        )

    def _risk_menu(self) -> InlineKeyboardMarkup:
        rows = [
            [
                InlineKeyboardButton(
                    "CONSERVATIVE",
                    callback_data=f"risk:{RiskProfile.CONSERVATIVE.value}",
                ),
                InlineKeyboardButton("MODERATE", callback_data=f"risk:{RiskProfile.MODERATE.value}"),
            ],
            [
                InlineKeyboardButton("AGGRESSIVE", callback_data=f"risk:{RiskProfile.AGGRESSIVE.value}"),
                InlineKeyboardButton("SCALP", callback_data=f"risk:{RiskProfile.SCALP.value}"),
            ],
            self._home_row(),
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
                InlineKeyboardButton(
                    "🚫 LIVE заблокирован (только env vars)",
                    callback_data="mode:active",
                )
            ],
            [
                InlineKeyboardButton("🧠 Обучить 500", callback_data=self._train_callback(500)),
                InlineKeyboardButton(
                    f"🧠 Обучить {self._train_defaults()[0]}",
                    callback_data=self._train_callback(),
                ),
            ],
            [
                InlineKeyboardButton(
                    "🏆 Промоутировать кандидата → CHAMPION",
                    callback_data="control:promote",
                )
            ],
            [
                InlineKeyboardButton("🎚 Лимиты", callback_data="control:limits"),
                InlineKeyboardButton("🗄 База и модель", callback_data="view:db_model"),
            ],
            [InlineKeyboardButton("🚦 Готовность CANARY", callback_data="view:canary")],
            [
                InlineKeyboardButton(
                    "❓ Как читать модель + путь к реальным деньгам",
                    callback_data="view:model_help",
                )
            ],
            [InlineKeyboardButton("🚨 Аварийная остановка", callback_data="control:stop")],
            self._home_row(),
        ]
        return InlineKeyboardMarkup(rows)

    def _canary_menu(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("🔄 Обновить готовность", callback_data="view:canary")],
                [InlineKeyboardButton("📌 Приоритеты", callback_data="view:priorities")],
                [InlineKeyboardButton("📊 Метрики модели", callback_data="view:canary_model")],
                [InlineKeyboardButton("🧾 Полная сводка для анализа", callback_data="view:deep_report")],
                [
                    InlineKeyboardButton("🗄 База и модель", callback_data="view:db_model"),
                    InlineKeyboardButton("⬅️ Назад", callback_data="view:control"),
                ],
                self._home_row(),
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
                InlineKeyboardButton("Изучать 10 монет", callback_data="limit:feature_symbols:10"),
                InlineKeyboardButton("Изучать 20 монет", callback_data="limit:feature_symbols:20"),
            ],
            [
                InlineKeyboardButton("Кандидатов 5", callback_data="limit:exec_candidates:5"),
                InlineKeyboardButton("Кандидатов 10", callback_data="limit:exec_candidates:10"),
            ],
            [
                InlineKeyboardButton(
                    "✏️ Изучать монет числом",
                    callback_data="limit:custom:feature_symbols",
                ),
                InlineKeyboardButton(
                    "✏️ Сделок одновременно числом",
                    callback_data="limit:custom:max_positions",
                ),
            ],
            [InlineKeyboardButton("✏️ Своё значение", callback_data="limit:custom")],
            [InlineKeyboardButton("⬅️ Управление", callback_data="view:control")],
            self._home_row(),
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
                InlineKeyboardButton("🏠 Главное меню", callback_data="view:menu"),
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
        chat_id = self._chat_id(update)
        if query is not None and query.message is not None:
            message = cast(Any, query.message)
            chunks = self._split_message(text)
            edited = False
            try:
                await query.edit_message_text(
                    chunks[0],
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                    reply_markup=reply_markup if len(chunks) == 1 else None,
                )
                edited = True
                if len(chunks) == 1:
                    return
            except BadRequest as exc:
                err = str(exc).lower()
                if reply_markup is not None and "message is not modified" in err:
                    try:
                        await query.edit_message_reply_markup(reply_markup=reply_markup)
                        return
                    except Exception as markup_exc:
                        log.debug("telegram.edit_markup_failed", error=str(markup_exc))
                log.info("telegram.edit_message_failed_sending_new", error=str(exc))
            except Exception as exc:
                log.info("telegram.edit_message_unavailable_sending_new", error=str(exc))
            if not edited and chat_id is not None:
                await self._send_direct_message(chat_id, text, reply_markup=reply_markup)
                return
            start = 1 if edited and len(chunks) > 1 else 0
            try:
                for index, chunk in enumerate(chunks[start:], start=start):
                    await self._safe_reply_text(
                        message,
                        chunk,
                        reply_markup=reply_markup if index == len(chunks) - 1 else None,
                    )
            except Exception as exc:
                log.warning("telegram.button_reply_failed", error=str(exc))
                if chat_id is not None:
                    await self._send_direct_message(chat_id, text, reply_markup=reply_markup)
            return
        await self._reply(update, text, reply_markup=reply_markup)

    # Keep the Telegram-side guard slightly above the provider-side timeout.
    # Otherwise the outer wait_for may fire first and lose the provider's
    # richer fallback (journal connected/configured + runtime candle counts).
    _DB_DIAG_TIMEOUT_LITE_S = 20.0
    _DB_DIAG_TIMEOUT_FULL_S = 35.0

    def _apply_db_diag_fallbacks(self, diag: dict[str, Any]) -> None:
        if self._controller is not None and self._controller.enrich_db_diag_fallbacks is not None:
            try:
                self._controller.enrich_db_diag_fallbacks(diag)
            except Exception as exc:
                log.debug("telegram.db_diag_fallback_failed", error=str(exc))

    async def _load_db_diag(self, *, lite: bool = True) -> dict[str, Any]:
        """Load DB diagnostics with timeout; lite mode avoids heavy Postgres scans."""
        if self._controller is None or self._controller.db_diagnostics_provider is None:
            return {"connected": False, "error": "db_diagnostics_unavailable", "lite": lite}
        provider = self._controller.db_diagnostics_provider
        timeout = self._DB_DIAG_TIMEOUT_LITE_S if lite else self._DB_DIAG_TIMEOUT_FULL_S
        try:
            try:
                coro = provider(lite=lite)
            except TypeError:
                coro = provider()
            diag = await asyncio.wait_for(coro, timeout=timeout)
            self._apply_db_diag_fallbacks(diag)
            return diag
        except TimeoutError:
            log.warning("telegram.db_diagnostics_timeout", lite=lite, timeout_s=timeout)
            if not lite:
                try:
                    try:
                        quick_coro = provider(lite=True)
                    except TypeError:
                        quick_coro = provider()
                    quick = await asyncio.wait_for(quick_coro, timeout=self._DB_DIAG_TIMEOUT_LITE_S)
                    quick["error"] = "db_diagnostics_timeout"
                    quick["full_diagnostics_timeout"] = True
                    quick["lite"] = True
                    self._apply_db_diag_fallbacks(quick)
                    return quick
                except Exception as exc:
                    log.debug("telegram.db_diagnostics_lite_fallback_failed", error=str(exc))
            diag = {
                "connected": False,
                "error": "db_diagnostics_timeout",
                "lite": lite,
            }
            self._apply_db_diag_fallbacks(diag)
            return diag
        except Exception as exc:
            log.warning("telegram.db_diagnostics_failed", lite=lite, error=str(exc))
            diag = {"connected": False, "error": str(exc), "lite": lite}
            self._apply_db_diag_fallbacks(diag)
            return diag

    async def _respond(
        self,
        update: Update,
        text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> None:
        """Reply via edit for callback buttons, otherwise send a new message."""
        if update.callback_query is not None:
            await self._button_reply(update, text, reply_markup=reply_markup)
            return
        await self._reply(update, text, reply_markup=reply_markup)

    def _db_diag_banner(self, db_diag: dict[str, Any]) -> str | None:
        if db_diag.get("error") == "db_diagnostics_timeout":
            return "⚠️ Postgres отвечает медленно — показаны быстрые данные без gate/paper."
        if db_diag.get("schema_degraded"):
            return "⚠️ Схема БД инициализируется в фоне — часть метрик может быть неполной."
        if db_diag.get("lite"):
            return None
        return None

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
        try:
            text, markup = await self._render_home()
            msg = await self._reply(update, text, reply_markup=markup)
            if msg is not None:
                self._dashboard_message_id = msg.message_id
                self._dashboard_chat_id = msg.chat_id
        except Exception as exc:
            log.warning("telegram.start_failed", error=str(exc))
            await self._reply(
                update,
                f"⚠️ Меню не загрузилось: <code>{html.escape(str(exc))[:300]}</code>",
                reply_markup=self._main_menu(),
            )

    async def _cmd_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        try:
            text, markup = await self._render_home()
            await self._reply(update, text, reply_markup=markup)
        except Exception as exc:
            log.warning("telegram.menu_failed", error=str(exc))
            await self._reply(
                update,
                f"⚠️ Меню не загрузилось: <code>{html.escape(str(exc))[:300]}</code>",
                reply_markup=self._main_menu(),
            )

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

        from trader.monitoring.deploy_info import get_deploy_info

        deploy = get_deploy_info()
        ctrl = self._controller
        lines = [
            "<b>Статус системы</b>",
        ]
        if deploy.get("deploy_id"):
            lines.append(f"Deploy: <code>{html.escape(str(deploy['deploy_id']))}</code>")
        elif deploy.get("git_commit"):
            lines.append(f"Commit: <code>{html.escape(str(deploy['git_commit']))}</code>")
        lines += [
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
            self._component_line(
                "Bybit REST",
                health.bybit_rest,
                health.bybit_rest_latency_ms,
                required=False,
            ),
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
        today_total = Decimal("0")
        wins = 0
        day_start_ms = int(datetime.now(tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
        shown = records[:20]
        lines = [f"<b>Закрытый PnL — последние {len(shown)} сделок</b>", ""]
        for r in shown:
            sym = r.get("symbol", "?")
            pnl = Decimal(str(r.get("closedPnl", "0")))
            total += pnl
            if pnl >= 0:
                wins += 1
            try:
                if int(r.get("updatedTime") or r.get("createdTime") or 0) >= day_start_ms:
                    today_total += pnl
            except (TypeError, ValueError):
                pass
            # Bybit closed-pnl `side` is the side of the CLOSING order:
            # a long position is closed by Sell, a short by Buy.
            side_raw = str(r.get("side", "")).upper()
            side = "LONG" if side_raw == "SELL" else ("SHORT" if side_raw == "BUY" else side_raw or "?")
            qty = r.get("qty", "")
            entry = r.get("avgEntryPrice", "")
            exit_p = r.get("avgExitPrice", "")
            icon = "✅" if pnl >= 0 else "❌"
            price_part = f" @ {entry}→{exit_p}" if entry and exit_p else ""
            qty_part = f" ×{qty}" if qty else ""
            lines.append(f"{icon} <code>{sym}</code> {side}{qty_part}{price_part}  <b>{pnl:+.4f} USDT</b>")
        total_icon = "📈" if total >= 0 else "📉"
        winrate = wins / len(shown) * 100 if shown else 0.0
        lines.append(f"\n{total_icon} <b>Итого за {len(shown)} сделок:</b> <code>{total:+.4f} USDT</code>")
        lines.append(f"Win-rate: <code>{wins}/{len(shown)} ({winrate:.0f}%)</code>")
        lines.append(f"Сегодня (UTC): <code>{today_total:+.4f} USDT</code>")
        lines.append(
            "\n💡 <i>Направление — сторона позиции (лонг закрывается продажей).\n"
            "Сумма — реализованный PnL по Bybit (без учёта незакрытых позиций).\n"
            "Чистый результат с комиссиями и фандингом — /net.</i>"
        )
        await self._reply(update, "\n".join(lines))

    async def _cmd_net_results(self, update: Update, context: Any) -> None:
        """Show daily net P&L breakdown including fees and funding."""
        del context
        if not await self._authorised(update):
            return
        net_stats: dict[str, Any] = {}
        try:
            if self._net_results_provider is not None:
                net_stats = await self._net_results_provider()
            else:
                # Fallback: try health_provider for backwards compat
                stats = await self._health_provider() if callable(self._health_provider) else None
                if stats is not None and isinstance(stats, dict):
                    net_stats = stats.get("net_results", {}) or {}
        except Exception as exc:
            log.warning("telegram.net_results_failed", error=str(exc))
            net_stats = {}

        gross = float(net_stats.get("gross_closed_pnl_usd") or net_stats.get("gross_pnl_usd") or 0.0)
        fees = float(net_stats.get("total_fees_usd") or 0.0)
        funding = float(net_stats.get("total_funding_usd") or 0.0)
        slippage_est = float(net_stats.get("estimated_slippage_usd") or 0.0)
        # Bybit closedPnl already includes fees and funding — it IS the net.
        net = float(net_stats.get("net_pnl_usd") or gross)
        maker_pct = float(net_stats.get("maker_fill_pct") or 0.0)
        taker_pct = float(net_stats.get("taker_fill_pct") or 100.0)
        fee_drag = abs(fees) + abs(funding) + abs(slippage_est)
        trade_count = int(net_stats.get("closed_trade_count") or 0)
        tx_count = int(net_stats.get("transaction_event_count") or 0)
        latest_tx = net_stats.get("latest_transaction_at")
        latest_tx_str = self._fmt_timestamp(latest_tx) if latest_tx else "нет"

        text = (
            "📈 <b>Чистый результат за сегодня UTC</b>\n\n"
            f"Закрытых сделок:   <code>{trade_count}</code>\n"
            f"Реализованный PnL: <code>{gross:+.4f} USDT</code>\n"
            f"  вкл. комиссии:   <code>{fees:+.4f} USDT</code>\n"
            f"  вкл. фандинг:    <code>{funding:+.4f} USDT</code>\n"
            f"Оценка проскальз.: <code>{slippage_est:+.4f} USDT</code>\n"
            "────────────────────────────\n"
            f"Чистый PnL:        <code>{net:+.4f} USDT</code>\n"
            f"Съели издержки:    <code>{fee_drag:.4f} USDT</code>\n\n"
            f"Transaction events: <code>{tx_count}</code>\n"
            f"Последняя транзакция: <code>{latest_tx_str}</code>\n\n"
            f"Maker исполнения: <code>{maker_pct:.1f}%</code>\n"
            f"Taker исполнения: <code>{taker_pct:.1f}%</code>"
        )

        if trade_count > 0 and tx_count == 0:
            text += (
                "\n\n⚠️ <b>Комиссии не загружены из Bybit.</b>\n"
                "Чистый PnL может быть завышен.\n"
                "Проверьте <code>transaction_log.sync_failed</code> в Render."
            )

        await self._reply(update, text)

    async def _cmd_costs(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show execution cost breakdown for today."""
        del context
        if not await self._authorised(update):
            return
        net_stats: dict[str, Any] = {}
        try:
            if self._net_results_provider is not None:
                net_stats = await self._net_results_provider()
        except Exception as exc:
            log.warning("telegram.costs_failed", error=str(exc))

        trade_count = int(net_stats.get("closed_trade_count") or 0)
        gross = float(net_stats.get("gross_closed_pnl_usd") or 0.0)
        fees = float(net_stats.get("total_fees_usd") or 0.0)
        funding = float(net_stats.get("total_funding_usd") or 0.0)
        net = float(net_stats.get("net_pnl_usd") or 0.0)
        maker_pct = float(net_stats.get("maker_fill_pct") or 0.0)
        taker_pct = float(net_stats.get("taker_fill_pct") or 100.0)

        diag: dict[str, Any] = {}
        if self._controller is not None and self._controller.diagnostics_provider is not None:
            try:
                diag = self._controller.diagnostics_provider()
            except Exception as _diag_exc:
                log.debug("telegram.costs_diag_failed", error=str(_diag_exc))

        net_edge_rejected = diag.get("hour_net_edge_rejected", 0)
        no_tp_rejected = diag.get("hour_no_take_profit_rejected", 0)
        fee_unavail_rejected = diag.get("hour_fee_rate_unavailable_rejected", 0)

        text = (
            "💸 <b>Экономика исполнения за сегодня UTC</b>\n\n"
            f"Закрытых сделок:   <code>{trade_count}</code>\n"
            f"Реализованный PnL: <code>{gross:+.4f} USDT</code>\n"
            f"  вкл. комиссии:   <code>{fees:+.4f} USDT</code>\n"
            f"  вкл. фандинг:    <code>{funding:+.4f} USDT</code>\n"
            f"Чистый PnL:        <code>{net:+.4f} USDT</code>\n\n"
            f"Maker / Taker:     <code>{maker_pct:.1f}% / {taker_pct:.1f}%</code>\n\n"
            f"Отклонено (edge мал):     <code>{net_edge_rejected}</code>\n"
            f"Отклонено (нет TP):       <code>{no_tp_rejected}</code>\n"
            f"Отклонено (нет fee rate): <code>{fee_unavail_rejected}</code>"
        )

        if taker_pct > 80:
            text += "\n\n⚠️ <b>Большинство исполнений taker.</b>\nДля скальпинга комиссии критичны."

        await self._respond(update, text, reply_markup=self._main_menu())

    async def _cmd_costs_detailed(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show paper-strategy gross/net edge and maker share diagnostics."""
        del context
        if not await self._authorised(update):
            return
        if self._controller is None or self._controller.costs_detailed_provider is None:
            await self._reply(
                update,
                "<b>Издержки детально</b>\nПока недоступно.",
                reply_markup=self._main_menu(),
            )
            return
        try:
            data = await self._controller.costs_detailed_provider()
        except Exception as exc:
            log.warning("telegram.costs_detailed_failed", error=str(exc))
            await self._reply(
                update,
                f"<b>Издержки детально</b>\nОшибка: <code>{exc}</code>",
                reply_markup=self._main_menu(),
            )
            return

        def _period_line(title: str, row: dict[str, Any]) -> str:
            gross = row.get("avg_gross_bps") or row.get("avg_gross_return_bps")
            net = row.get("avg_net_bps") or row.get("avg_net_return_bps")
            cost = row.get("avg_cost_bps")
            return (
                f"<b>{title}</b>\n"
                f"Сделок: <code>{int(row.get('count') or 0)}</code>\n"
                f"Gross avg: <code>{float(gross or 0.0):+.2f} bps</code>\n"
                f"Net avg:   <code>{float(net or 0.0):+.2f} bps</code>\n"
                f"Cost avg:  <code>{float(cost or 0.0):+.2f} bps</code>"
            )

        today = data.get("today") or {}
        total = data.get("all_time") or data.get("all") or {}
        horizon = int(data.get("horizon_minutes") or self._train_defaults()[1])
        label_schema = str(data.get("label_schema_version") or "directional_net_v2")
        lines = [
            "🧾 <b>Издержки детально</b>",
            f"База: <code>RULE_BASELINE_V1</code>, "
            f"<code>{html.escape(label_schema)}</code>, horizon <code>{horizon}m</code>",
            "",
            _period_line("Сегодня UTC", today),
            "",
            _period_line("Всего", total),
        ]
        maker_share = data.get("maker_share_pct")
        maker_count = int(data.get("maker_count") or 0)
        exec_count = int(data.get("execution_count") or 0)
        if maker_share is not None and exec_count > 0:
            lines.append(f"\nMaker fills: <code>{float(maker_share):.1f}%</code> ({maker_count}/{exec_count})")
        lines.append("\nПроскальзывание пока не считается: нет надежной связки signal → fill/close для всех режимов.")
        await self._reply(update, "\n".join(lines), reply_markup=self._main_menu())

    async def _cmd_pnl_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show baseline PnL breakdowns by symbol, hour, regime and weekday."""
        del context
        if not await self._authorised(update):
            return
        if self._controller is None or self._controller.pnl_analysis_provider is None:
            await self._reply(
                update,
                "<b>PnL-анализ</b>\nПока недоступен.",
                reply_markup=self._main_menu(),
            )
            return
        try:
            data = await self._controller.pnl_analysis_provider()
        except Exception as exc:
            log.warning("telegram.pnl_analysis_failed", error=str(exc))
            await self._reply(
                update,
                f"<b>PnL-анализ</b>\nОшибка: <code>{exc}</code>",
                reply_markup=self._main_menu(),
            )
            return

        def _row(label: str, count: Any, avg: Any, total: Any | None = None) -> str:
            total_part = f", Σ <code>{float(total or 0.0):+.1f}</code>" if total is not None else ""
            return f"{label}: <code>{int(count or 0)}</code>, avg <code>{float(avg or 0.0):+.2f}</code>{total_part}"

        def _net_value(row: dict[str, Any], short_key: str, canonical_key: str) -> Any:
            """Read PnL without treating a legitimate zero as a missing value."""
            value = row.get(short_key)
            return row.get(canonical_key) if value is None else value

        symbol_best = data.get("symbols_best") or data.get("top_symbols") or []
        symbol_worst = data.get("symbols_worst") or data.get("worst_symbols") or []
        regimes = data.get("regimes") or []
        weekdays = data.get("weekdays") or []
        strategies = data.get("strategies") or []
        hours = {int(row.get("hour") or row.get("hour_utc") or 0): row for row in data.get("hours") or []}
        horizon = int(data.get("horizon_minutes") or self._train_defaults()[1])
        label_schema = str(data.get("label_schema_version") or "directional_net_v2")
        lines = [
            "🔬 <b>PnL-анализ baseline</b>",
            f"Фильтр: <code>{html.escape(label_schema)}</code>, "
            f"<code>RULE_BASELINE_V1</code>, horizon <code>{horizon}m</code>",
            "",
            "<b>Топ-5 прибыльных символов</b>",
        ]
        lines.extend(
            _row(
                f"<code>{row.get('symbol', '?')}</code>",
                row.get("count"),
                _net_value(row, "avg_net_bps", "avg_net_return_bps"),
                _net_value(row, "total_net_bps", "total_net_return_bps"),
            )
            for row in symbol_best[:5]
        )
        lines.append("\n<b>Топ-5 убыточных символов</b>")
        lines.extend(
            _row(
                f"<code>{row.get('symbol', '?')}</code>",
                row.get("count"),
                _net_value(row, "avg_net_bps", "avg_net_return_bps"),
                _net_value(row, "total_net_bps", "total_net_return_bps"),
            )
            for row in symbol_worst[:5]
        )
        lines.append("\n<b>По часам UTC</b>")
        hour_chunks: list[str] = []
        for hour in range(24):
            row = hours.get(hour, {})
            label = f"{hour:02d}-{(hour + 1) % 24:02d}"
            avg = _net_value(row, "avg_net_bps", "avg_net_return_bps")
            hour_chunks.append(f"{label}:{float(avg or 0.0):+.1f}/{int(row.get('count') or 0)}")
        lines.extend("<code>" + "  ".join(hour_chunks[i : i + 4]) + "</code>" for i in range(0, 24, 4))
        lines.append("\n<b>По режимам</b>")
        lines.extend(
            _row(
                str(row.get("regime") or "unknown"),
                row.get("count"),
                _net_value(row, "avg_net_bps", "avg_net_return_bps"),
                _net_value(row, "total_net_bps", "total_net_return_bps"),
            )
            for row in regimes[:10]
        )
        lines.append("\n<b>По дням недели</b>")
        lines.extend(
            _row(
                str(row.get("weekday") or "?"),
                row.get("count"),
                _net_value(row, "avg_net_bps", "avg_net_return_bps"),
                _net_value(row, "total_net_bps", "total_net_return_bps"),
            )
            for row in weekdays
        )
        lines.append("\n<b>По стратегиям</b>")
        if strategies:
            for row in strategies:
                strategy_id = html.escape(str(row.get("strategy_id") or "UNKNOWN"))
                count = int(row.get("count") or 0)
                gross = float(row.get("avg_gross_return_bps") or 0.0)
                costs = float(row.get("avg_cost_bps") or 0.0)
                net = float(_net_value(row, "avg_net_bps", "avg_net_return_bps") or 0.0)
                status = "✅" if net >= 0 else "⛔" if count >= 20 else "🧪"
                lines.append(
                    f"{status} <code>{strategy_id}</code>: <code>{count}</code>, "
                    f"gross <code>{gross:+.2f}</code>, costs <code>{costs:.2f}</code>, "
                    f"net <code>{net:+.2f} bps</code>"
                )
        else:
            lines.append("Нет размеченных strategy_id.")
        await self._reply(update, "\n".join(lines), reply_markup=self._main_menu())

    async def _cmd_compare(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Compare baseline, model gate-pass and equal-size random baseline sample."""
        del context
        if not await self._authorised(update):
            return
        if self._controller is None or self._controller.compare_provider is None:
            await self._reply(
                update,
                "<b>Compare</b>\nПока недоступно.",
                reply_markup=self._main_menu(),
            )
            return
        try:
            data = await self._controller.compare_provider()
        except Exception as exc:
            log.warning("telegram.compare_failed", error=str(exc))
            await self._reply(
                update,
                f"<b>Compare</b>\nОшибка: <code>{exc}</code>",
                reply_markup=self._main_menu(),
            )
            return

        def _sample_line(title: str, row: dict[str, Any]) -> str:
            return (
                f"{title:<10} "
                f"n=<code>{int(row.get('count') or 0):4d}</code>  "
                f"sum=<code>{float(row.get('sum_net_bps') or 0.0):+9.1f}</code>  "
                f"avg=<code>{float(row.get('avg_net_bps') or 0.0):+7.2f}</code>"
            )

        baseline = data.get("baseline") or {}
        gate = data.get("gate_pass") or {}
        random_sample = data.get("random_sample") or {}
        model_version = data.get("model_version") or "none"
        p_value = data.get("p_value")
        lines = [
            "⚖️ <b>Compare</b>",
            f"Модель: <code>{model_version}</code>",
            "<code>sample     n     sum_bps     avg_bps</code>",
            _sample_line("baseline", baseline),
            _sample_line("gate", gate),
            _sample_line("random", random_sample),
        ]
        if p_value is None:
            lines.append("\np-value: <code>n/a</code> — мало данных для проверки.")
        else:
            lines.append(f"\np-value gate vs baseline: <code>{float(p_value):.4f}</code>")
        if int(gate.get("count") or 0) < 20:
            lines.append("⚠️ Gate-pass сделок меньше 20: статистика ещё очень шумная.")
        await self._reply(update, "\n".join(lines), reply_markup=self._main_menu())

    async def _cmd_worst(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show worst labelled baseline outcomes with key features and model decision."""
        if not await self._authorised(update):
            return
        limit = 10
        if context.args:
            try:
                limit = max(1, min(20, int(context.args[0])))
            except (TypeError, ValueError):
                limit = 10
        if self._controller is None or self._controller.worst_trades_provider is None:
            await self._reply(update, "<b>Worst</b>\nПока недоступно.", reply_markup=self._main_menu())
            return
        try:
            rows = await self._controller.worst_trades_provider(limit)
        except Exception as exc:
            log.warning("telegram.worst_failed", error=str(exc))
            await self._reply(
                update,
                f"<b>Worst</b>\nОшибка: <code>{exc}</code>",
                reply_markup=self._main_menu(),
            )
            return
        if not rows:
            await self._reply(
                update,
                "<b>Worst</b>\nУбыточных размеченных исходов пока нет.",
                reply_markup=self._main_menu(),
            )
            return

        lines = [f"📉 <b>{len(rows)} худших исходов baseline</b>"]
        for row in rows:
            features = row.get("features") or {}
            feat_text = (
                f"rsi={features.get('rsi_14', 'n/a')}, "
                f"atr%={features.get('atr_14_pct', 'n/a')}, "
                f"ob={features.get('ob_imbalance_l5', 'n/a')}, "
                f"micro={features.get('microprice_deviation_bps', 'n/a')}"
            )
            score = row.get("model_score")
            score_text = f"{float(score):.3f}" if score is not None else "n/a"
            ts = self._fmt_timestamp(row.get("created_at")) if row.get("created_at") else "n/a"
            lines.append(
                "\n"
                f"<code>{row.get('symbol', '?')}</code> {row.get('side', '?')} {ts}\n"
                f"net: <code>{float(row.get('net_return_bps') or 0.0):+.2f} bps</code>, "
                f"model: <code>{row.get('model_decision') or 'n/a'}</code> score=<code>{score_text}</code>\n"
                f"<code>{feat_text}</code>"
            )
        await self._reply(update, "\n".join(lines), reply_markup=self._main_menu())

    async def _cmd_model_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show model metric history from model_versions.metrics."""
        del context
        if not await self._authorised(update):
            return
        if self._controller is None or self._controller.model_performance_provider is None:
            await self._respond(
                update,
                "<b>История моделей</b>\nПока недоступна.",
                reply_markup=self._main_menu(),
            )
            return
        stale_note = ""
        try:
            rows = await asyncio.wait_for(self._controller.model_performance_provider(), timeout=15.0)
        except TimeoutError:
            log.warning("telegram.model_performance_timeout")
            rows = list(self._model_performance_cache)
            stale_note = "⚠️ Postgres медленный — показан последний успешный снимок.\n\n"
        except Exception as exc:
            log.warning("telegram.model_performance_failed", error=str(exc))
            rows = list(self._model_performance_cache)
            if rows:
                stale_note = "⚠️ Ошибка чтения — показан последний успешный снимок.\n\n"
            else:
                await self._respond(
                    update,
                    f"<b>История моделей</b>\nОшибка: <code>{exc}</code>",
                    reply_markup=self._main_menu(),
                )
                return
        if rows:
            self._model_performance_cache = rows
        if not rows:
            await self._respond(
                update,
                "<b>История моделей</b>\nВерсий модели пока нет.",
                reply_markup=self._main_menu(),
            )
            return
        lines = [
            stale_note + "📉 <b>История моделей</b>",
            "<code>date UTC          q        score    n   lift   wf</code>",
        ]
        for row in rows:
            ts = self._fmt_timestamp(row.get("created_at")) if row.get("created_at") else "n/a"
            quality = str(row.get("quality") or "n/a")[:8]
            score = row.get("model_score")
            paper_count = int(row.get("paper_gate_count") or 0)
            lift = row.get("lift_bps")
            walk_forward = row.get("walk_forward_bps")
            score_text = "   n/a" if score is None else f"{float(score):+6.1f}"
            lift_text = " n/a" if lift is None else f"{float(lift):+5.1f}"
            wf_text = " n/a" if walk_forward is None else f"{float(walk_forward):+5.1f}"
            reason = str(row.get("selection_reason") or "")[:34]
            lines.append(
                f"<code>{ts[:16]:16} {quality:8} {score_text} {paper_count:4d} {lift_text} {wf_text}</code>"
                + (f"\n<code>  {reason}</code>" if reason else "")
            )
        await self._respond(update, "\n".join(lines), reply_markup=self._main_menu())

    async def _cmd_champion_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        text, markup = await self._render_champion_health()
        await self._reply(update, text, reply_markup=markup)

    async def _cmd_strategy_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Build one compact strategy diagnostics report for canary decisions."""
        del context
        if not await self._authorised(update):
            return
        if self._controller is None:
            await self._reply(
                update,
                "<b>Отчет стратегии</b>\nПока недоступен.",
                reply_markup=self._main_menu(),
            )
            return

        async def _safe(name: str, call: Callable[[], Awaitable[Any]] | None, default: Any) -> Any:
            if call is None:
                return default
            try:
                return await call()
            except Exception as exc:
                log.warning(
                    "telegram.strategy_report_section_failed",
                    section=name,
                    error=str(exc),
                )
                return default

        pnl = await _safe("pnl", self._controller.pnl_analysis_provider, {})
        compare = await _safe("compare", self._controller.compare_provider, {})
        costs = await _safe("costs", self._controller.costs_detailed_provider, {})
        models = await _safe("models", self._controller.model_performance_provider, [])
        worst: list[dict[str, Any]] = []
        if self._controller.worst_trades_provider is not None:
            try:
                worst = await self._controller.worst_trades_provider(3)
            except Exception as exc:
                log.warning("telegram.strategy_report_worst_failed", error=str(exc))

        baseline = compare.get("baseline") or {}
        gate = compare.get("gate_pass") or {}
        random_sample = compare.get("random_sample") or {}
        p_value = compare.get("p_value")
        best_symbols = pnl.get("symbols_best") or []
        worst_symbols = pnl.get("symbols_worst") or []
        regimes = pnl.get("regimes") or []
        total_costs = costs.get("all_time") or {}
        today_costs = costs.get("today") or {}
        latest_model = models[0] if models else {}

        baseline_avg = float(baseline.get("avg_net_bps") or 0.0)
        gate_avg = float(gate.get("avg_net_bps") or 0.0)
        gate_count = int(gate.get("count") or 0)
        gate_sum = float(gate.get("sum_net_bps") or 0.0)
        baseline_sum = float(baseline.get("sum_net_bps") or 0.0)
        random_avg = float(random_sample.get("avg_net_bps") or 0.0)
        quality = str(latest_model.get("quality") or "n/a")
        walk_forward = latest_model.get("walk_forward_bps")
        issues: list[str] = []
        if baseline_avg <= 0:
            issues.append(f"baseline avg отрицательный: {baseline_avg:+.2f} bps")
        if gate_count < 20:
            issues.append(f"gate-pass мало сделок: {gate_count}/20")
        if gate_count >= 20 and gate_avg <= 0:
            issues.append(f"gate-pass avg отрицательный: {gate_avg:+.2f} bps")
        if p_value is None or float(p_value) >= 0.05:
            issues.append("преимущество gate статистически не доказано")
        if quality not in ("GOOD", "ХОРОШО"):
            issues.append(f"качество модели: {quality}")
        if walk_forward is not None and float(walk_forward) <= 0:
            issues.append(f"walk-forward: {float(walk_forward):+.2f} bps")

        def _symbol_line(row: dict[str, Any]) -> str:
            return (
                f"<code>{row.get('symbol', '?')}</code> "
                f"avg=<code>{float(row.get('avg_net_bps') or 0.0):+.1f}</code> "
                f"n=<code>{int(row.get('count') or 0)}</code>"
            )

        def _regime_line(row: dict[str, Any]) -> str:
            return (
                f"{row.get('regime') or 'unknown'}: "
                f"<code>{float(row.get('avg_net_bps') or 0.0):+.1f}</code> "
                f"n=<code>{int(row.get('count') or 0)}</code>"
            )

        verdict = "НЕ ГОТОВ к CANARY" if issues else "можно рассматривать маленький CANARY"
        lines = [
            "📑 <b>Отчет стратегии</b>",
            f"Вердикт: <b>{verdict}</b>",
            "",
            "<b>PnL выборки</b>",
            f"Baseline: n=<code>{int(baseline.get('count') or 0)}</code>, "
            f"Σ=<code>{baseline_sum:+.1f}</code>, avg=<code>{baseline_avg:+.2f}</code>",
            f"Gate:     n=<code>{gate_count}</code>, Σ=<code>{gate_sum:+.1f}</code>, avg=<code>{gate_avg:+.2f}</code>",
            f"Random:   n=<code>{int(random_sample.get('count') or 0)}</code>, avg=<code>{random_avg:+.2f}</code>",
            f"p-value: <code>{'n/a' if p_value is None else f'{float(p_value):.4f}'}</code>",
            "",
            "<b>Что мешает</b>",
        ]
        lines.extend(f"• {issue}" for issue in issues[:6])
        if not issues:
            lines.append("• явных блокеров в диагностике не найдено; размер CANARY всё равно держать минимальным")

        lines.append("\n<b>Лучшие символы</b>")
        lines.extend(_symbol_line(row) for row in best_symbols[:3])
        lines.append("\n<b>Худшие символы</b>")
        lines.extend(_symbol_line(row) for row in worst_symbols[:3])
        lines.append("\n<b>Режимы рынка</b>")
        lines.extend(_regime_line(row) for row in regimes[:4])

        lines.append("\n<b>Издержки baseline</b>")
        lines.append(
            f"Сегодня net avg=<code>{float(today_costs.get('avg_net_bps') or 0.0):+.2f}</code>, "
            f"cost avg=<code>{float(today_costs.get('avg_cost_bps') or 0.0):+.2f}</code>"
        )
        lines.append(
            f"Всего net avg=<code>{float(total_costs.get('avg_net_bps') or 0.0):+.2f}</code>, "
            f"cost avg=<code>{float(total_costs.get('avg_cost_bps') or 0.0):+.2f}</code>"
        )

        if latest_model:
            precision = latest_model.get("precision")
            lift = latest_model.get("lift_bps")
            precision_text = "n/a" if precision is None else f"{float(precision) * 100:.1f}%"
            lift_text = "n/a" if lift is None else f"{float(lift):+.1f} bps"
            wf_text = "n/a" if walk_forward is None else f"{float(walk_forward):+.1f} bps"
            lines.append(
                f"\n<b>Последняя модель</b>\nquality=<code>{quality}</code>, precision=<code>{precision_text}</code>, "
                f"lift=<code>{lift_text}</code>, wf=<code>{wf_text}</code>"
            )

        if worst:
            lines.append("\n<b>3 худших исхода</b>")
            for row in worst:
                ts = self._fmt_timestamp(row.get("created_at")) if row.get("created_at") else "n/a"
                lines.append(
                    f"<code>{row.get('symbol', '?')}</code> {row.get('side', '?')} "
                    f"<code>{float(row.get('net_return_bps') or 0.0):+.1f} bps</code> {ts[:16]}"
                )

        lines.append("\nDrill-down: /pnl_analysis /compare /worst 10 /costs_detailed /model_performance")
        await self._reply(update, "\n".join(lines), reply_markup=self._main_menu())

    async def _cmd_attribution(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        if self._controller is None or self._controller.attribution_provider is None:
            await self._reply(update, "<b>Attribution</b>\nПока недоступно.")
            return
        try:
            rows = await self._controller.attribution_provider(7)
        except Exception as exc:
            await self._reply(update, f"<b>Attribution</b>\nОшибка: <code>{exc}</code>")
            return
        if not rows:
            await self._reply(
                update,
                "<b>Attribution (7 дней)</b>\nНет данных по символам.",
                reply_markup=self._main_menu(),
            )
            return
        source = str(rows[0].get("source") or "live")
        unit = "USDT" if source == "live" else "avg bps"
        lines = [f"<b>Attribution за 7 дней</b> ({source})", ""]
        for row in rows[:12]:
            sym = row.get("symbol", "?")
            wins = int(row.get("wins") or 0)
            losses = int(row.get("losses") or 0)
            total = float(row.get("total_pnl") or 0)
            lines.append(f"<code>{sym}</code>: {total:+.4f} {unit} | W/L {wins}/{losses}")
        await self._reply(update, "\n".join(lines), reply_markup=self._main_menu())

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
        ]
        drift = diag.get("drift_status")
        if isinstance(drift, dict) and drift.get("status") not in {None, "n/a"}:
            lines.append(f"Drift: <code>{drift.get('status')}</code> PSI=<code>{drift.get('psi', 'n/a')}</code>")
        if diag.get("strategy_cycle_ms") is not None:
            lines.append(f"Цикл стратегии: <code>{diag.get('strategy_cycle_ms'):.0f} ms</code>")
        if diag.get("last_retention_run_at"):
            lines.append(f"Retention: <code>{diag.get('last_retention_run_at')}</code>")
        telegram_health = diag.get("telegram") or {}
        if telegram_health:
            tg_ok = bool(telegram_health.get("app_running")) and (
                bool(telegram_health.get("polling_running")) or bool(telegram_health.get("webhook_active"))
            )
            tg_mode = "webhook" if telegram_health.get("webhook_active") else "polling"
            lines += [
                f"Telegram: <code>{'ok' if tg_ok else 'problem'}</code> mode=<code>{tg_mode}</code>",
                f"Telegram conflicts: <code>{telegram_health.get('polling_conflict_count', 0)}</code>, "
                f"net errors=<code>{telegram_health.get('polling_network_error_count', 0)}</code>",
            ]
            if telegram_health.get("last_polling_error_at"):
                lines.append(f"Последняя ошибка Telegram: <code>{telegram_health.get('last_polling_error_at')}</code>")

        lines += [
            "",
            f"Сигналов создано:       <code>{diag.get('hour_signals_emitted', 0)}</code>",
            f"Отклонено риск-менедж.: <code>{diag.get('hour_risk_rejected', 0)}</code>",
            f"Отклонено Bybit API:    <code>{diag.get('hour_api_rejected', 0)}</code>",
            f"Малый размер заявки:    <code>{diag.get('hour_min_notional_rejected', 0)}</code>",
            f"Пропущено из-за позиции:<code>{diag.get('hour_skipped_open_position', 0)}</code>",
            f"Пропущено cooldown:     <code>{diag.get('hour_skipped_entry_cooldown', 0)}</code>",
            f"Пропущено после ошибки: <code>{diag.get('hour_skipped_failure_cooldown', 0)}</code>",
            f"Блоков фильтра модели:  <code>{diag.get('hour_model_gate_canary_blocked', 0)}</code>",
            f"Отклонено spread (scalp): <code>{diag.get('hour_spread_rejected', 0)}</code>",
            f"Отклонено imbalance:     <code>{diag.get('hour_imbalance_rejected', 0)}</code>",
            f"Отклонено net-edge scalp: <code>{diag.get('hour_scalp_net_edge_rejected', 0)}</code>",
            f"Блок bucket/regime:      <code>{diag.get('hour_bucket_blocked', 0)}</code>",
            f"Пропущено pending-заявка:<code>{diag.get('hour_skipped_pending_entries', 0)}</code>",
            f"Ордеров размещено:      <code>{diag.get('hour_order_placed', 0)}</code>",
            f"Ордеров неудачно:       <code>{diag.get('hour_order_failed', 0)}</code>",
        ]

        # Pending entry blocking warning
        pending_ids = diag.get("pending_entry_ids") or []
        pending_symbols = diag.get("pending_entry_symbols") or []
        pending_count = diag.get("pending_entry_count") or 0
        if pending_count > 0:
            ids_str = ", ".join(f"<code>{pid[:16]}…</code>" for pid in pending_ids[:3])
            sym_str = ", ".join(pending_symbols[:3]) if pending_symbols else "неизвестно"
            lines.append(
                f"\n⚠️ Новые входы <b>заблокированы</b> pending-заявкой:\n"
                f"ID: {ids_str or 'нет'}, символ: {sym_str}\n"
                f"Проверка stale pending выполняется автоматически при старте."
            )

        signals = int(diag.get("hour_signals_emitted") or 0)
        rejections = sum(
            int(diag.get(key) or 0)
            for key in (
                "hour_risk_rejected",
                "hour_api_rejected",
                "hour_model_gate_canary_blocked",
                "hour_spread_rejected",
                "hour_imbalance_rejected",
                "hour_scalp_net_edge_rejected",
                "hour_bucket_blocked",
                "hour_net_edge_rejected",
            )
        )
        if signals == 0 and rejections == 0:
            lines.append(
                "\nℹ️ Сигналов нет — scalp ждёт свежий EMA-cross + объём + spread + imbalance.\n"
                "Это нормально в тихом рынке; отклонения появятся только после кандидата на вход."
            )

        await self._respond(update, "\n".join(lines))

    async def _cmd_deep_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send all operator diagnostics as one downloadable text file."""
        del context
        if not await self._authorised(update):
            return
        text = await self._render_deep_report_text()
        await self._send_deep_report_document(update, text)

    async def _cmd_deep_report_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Legacy text mode for the deep report (may split into many messages)."""
        del context
        if not await self._authorised(update):
            return
        text = await self._render_deep_report_text()
        await self._respond(update, text, reply_markup=self._diagnostics_menu())

    async def _send_deep_report_document(self, update: Update, html_text: str) -> None:
        """Send the deep report as a single .txt attachment for easy copying."""
        generated_at = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"bybit_deep_report_{generated_at}.txt"
        plain_text = self._plain_text(html_text)
        payload = io.BytesIO(plain_text.encode("utf-8"))
        payload.name = filename
        caption = (
            "🧾 <b>Полная сводка готова файлом</b>\n"
            "Скачайте .txt и пришлите/скопируйте его целиком для анализа.\n"
            "Если нужен старый режим сообщениями: <code>/deep_report_text</code>"
        )
        message = update.effective_message
        if message is not None:
            await message.reply_document(
                document=payload,
                filename=filename,
                caption=caption,
                parse_mode=ParseMode.HTML,
                reply_markup=self._diagnostics_menu(),
            )
            if update.callback_query is not None:
                try:
                    await update.callback_query.answer("Файл со сводкой отправлен")
                except Exception as exc:
                    log.debug("telegram.deep_report_callback_answer_failed", error=str(exc))
            return
        chat_id = self._chat_id(update)
        if chat_id is not None and self._app is not None:
            await self._app.bot.send_document(
                chat_id=chat_id,
                document=payload,
                filename=filename,
                caption=caption,
                parse_mode=ParseMode.HTML,
                reply_markup=self._diagnostics_menu(),
            )
            return
        await self._respond(
            update, "Не удалось определить чат для отправки файла.", reply_markup=self._diagnostics_menu()
        )

    @staticmethod
    def _redact_for_report(value: Any) -> Any:
        """Redact obvious secret-bearing fields before dumping raw diagnostics."""
        sensitive = (
            "api_key",
            "api_secret",
            "secret",
            "token",
            "password",
            "postgres_dsn",
            "database_url",
            "redis_url",
            "encrypt_key",
        )
        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for key, item in value.items():
                key_s = str(key)
                if any(marker in key_s.lower() for marker in sensitive):
                    redacted[key_s] = "***REDACTED***"
                else:
                    redacted[key_s] = TelegramMonitorBot._redact_for_report(item)
            return redacted
        if isinstance(value, list | tuple):
            return [TelegramMonitorBot._redact_for_report(item) for item in value]
        if isinstance(value, str):
            redacted = value
            redacted = re.sub(
                r"(?i)(api[_-]?key|api[_-]?secret|token|password|postgres[_-]?dsn|database[_-]?url|redis[_-]?url)(\s*[=:]\s*)([^\s,;]+)",
                r"\1\2***REDACTED***",
                redacted,
            )
            redacted = re.sub(r"(?i)(bearer\s+)[a-z0-9._~+/\-=]+", r"\1***REDACTED***", redacted)
            return redacted
        return value

    @classmethod
    def _json_for_report(cls, value: Any) -> str:
        return html.escape(
            json.dumps(
                cls._redact_for_report(value),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                default=str,
            )
        )

    @staticmethod
    def _detail_rows(items: Any, *, max_rows: int = 8) -> list[str]:
        if not isinstance(items, list):
            return []
        rows: list[str] = []
        for item in items[:max_rows]:
            if not isinstance(item, dict):
                continue
            symbol = html.escape(str(item.get("symbol") or "?"))
            side = html.escape(str(item.get("side") or ""))
            count = int(item.get("count") or 0)
            suffix = f" {side}" if side else ""
            rows.append(f"  - <code>{symbol}{suffix}</code>: <code>{count}</code>")
        return rows

    @staticmethod
    def _strategy_detail_rows(items: Any, *, max_rows: int = 10) -> list[str]:
        if not isinstance(items, list):
            return []
        rows: list[str] = []
        for item in items[:max_rows]:
            if not isinstance(item, dict):
                continue
            strategy_id = html.escape(str(item.get("symbol") or item.get("reason") or "?"))
            count = int(item.get("count") or 0)
            rows.append(f"  - <code>{strategy_id}</code>: <code>{count}</code>")
        return rows

    @classmethod
    def _render_runtime_explainers(cls, runtime_diag: dict[str, Any]) -> list[str]:
        if not isinstance(runtime_diag, dict):
            return []
        rejection_details = runtime_diag.get("hour_rejection_details") or {}
        strategy_details = runtime_diag.get("hour_strategy_details") or {}
        lines: list[str] = ["<b>Почему нет/мало сделок за последний час</b>"]
        top_blocker = runtime_diag.get("top_blocker")
        if top_blocker:
            lines.append(f"Самый частый блокер: <code>{html.escape(str(top_blocker))}</code>")

        sections = [
            ("Отказы по imbalance/stakan", rejection_details.get("imbalance_rejected")),
            ("Отказы по net-edge scalp", rejection_details.get("scalp_net_edge_rejected")),
            ("Отказы по spread", rejection_details.get("spread_rejected")),
            ("Нет/устарел стакан", rejection_details.get("imbalance_missing")),
        ]
        any_rejection_rows = False
        for title, items in sections:
            rows = cls._detail_rows(items)
            if rows:
                any_rejection_rows = True
                lines.append(f"{html.escape(title)}:")
                lines.extend(rows)
        if not any_rejection_rows:
            lines.append("Детальных отказов по symbol/side пока нет.")

        strategy_sections = [
            ("Стратегии молчали", strategy_details.get("no_signal")),
            ("Стратегии дали кандидата", strategy_details.get("proposed")),
            ("Ансамбль выпустил сигнал", strategy_details.get("emitted")),
            ("Блок confirmation", strategy_details.get("confirmation_blocked")),
            ("Ниже min confidence", strategy_details.get("below_min_confidence")),
            ("Shadow probe причины", strategy_details.get("shadow_probe")),
            ("Shadow probe по символам", strategy_details.get("shadow_probe_symbols")),
        ]
        any_strategy_rows = False
        for title, items in strategy_sections:
            rows = cls._strategy_detail_rows(items)
            if rows:
                any_strategy_rows = True
                lines.append(f"{html.escape(title)}:")
                lines.extend(rows)
        if not any_strategy_rows:
            lines.append("Деталей по стратегиям ещё нет — нужен деплой с ensemble diagnostics и 10–20 минут работы.")

        conflict_count = int(runtime_diag.get("hour_ensemble_conflict_blocked") or 0)
        if conflict_count:
            lines.append(f"Конфликт BUY/SELL равного приоритета: <code>{conflict_count}</code>")

        lines.append(
            "Следующий рычаг: если большинство строк — <code>strategy_no_signal</code>, расширяем/настраиваем стратегии; "
            "если <code>imbalance_rejected</code>, не ослабляем стакан вслепую, а смотрим symbol/side и качество этих отказов."
        )
        return lines

    @classmethod
    def _render_log_tail_for_report(cls) -> str | None:
        raw = os.getenv("RENDER_LOG_TAIL") or os.getenv("APP_LOG_TAIL")
        if not raw:
            return None
        lines = raw.splitlines()[-120:]
        redacted = cls._redact_for_report("\n".join(lines))
        return html.escape(str(redacted))

    async def _render_deep_report_text(self) -> str:
        ctrl = self._controller

        async def _safe_async(label: str, provider: Callable[[], Awaitable[Any]] | None, default: Any) -> Any:
            if provider is None:
                return default
            try:
                return await asyncio.wait_for(provider(), timeout=12.0)
            except Exception as exc:
                return {"error": f"{label}: {type(exc).__name__}: {exc}"}

        def _safe_sync(label: str, provider: Callable[[], Any] | None, default: Any) -> Any:
            if provider is None:
                return default
            try:
                return provider()
            except Exception as exc:
                return {"error": f"{label}: {type(exc).__name__}: {exc}"}

        from trader.monitoring.deploy_info import get_deploy_info

        generated_at = datetime.now(tz=UTC).isoformat()
        deploy = get_deploy_info()
        runtime_diag = _safe_sync(
            "runtime_diagnostics",
            ctrl.diagnostics_provider if ctrl is not None else None,
            {},
        )
        runtime_settings = _safe_sync(
            "runtime_settings",
            ctrl.runtime_settings if ctrl is not None else None,
            {},
        )
        db_diag = await self._load_db_diag(lite=False)
        healthcheck = await _safe_async(
            "healthcheck",
            ctrl.healthcheck_provider if ctrl is not None else None,
            {},
        )
        compare = await _safe_async(
            "compare",
            ctrl.compare_provider if ctrl is not None else None,
            {},
        )
        pnl_analysis = await _safe_async(
            "pnl_analysis",
            ctrl.pnl_analysis_provider if ctrl is not None else None,
            {},
        )
        costs = await _safe_async(
            "costs_detailed",
            ctrl.costs_detailed_provider if ctrl is not None else None,
            {},
        )
        log_tail = self._render_log_tail_for_report()
        model_rows = await _safe_async(
            "model_performance",
            ctrl.model_performance_provider if ctrl is not None else None,
            [],
        )
        champion = await _safe_async(
            "champion_health",
            ctrl.champion_health_provider if ctrl is not None else None,
            {},
        )
        canary_text = self._canary_readiness_text(
            db_diag=db_diag, diag=runtime_diag if isinstance(runtime_diag, dict) else {}
        )

        model_info = runtime_diag.get("model") if isinstance(runtime_diag, dict) else {}
        latest_model = db_diag.get("latest_model_version") or db_diag.get("active_model_version") or {}
        latest_metrics = latest_model.get("metrics") if isinstance(latest_model, dict) else {}
        if isinstance(latest_metrics, str):
            try:
                latest_metrics = json.loads(latest_metrics) or {}
            except Exception:
                latest_metrics = {}
        gate_quality = (
            runtime_settings.get("model_gate_quality") if isinstance(runtime_settings, dict) else None
        ) or {}
        if not gate_quality and isinstance(model_info, dict):
            gate_quality = {"quality": model_info.get("quality")}

        shadow = ctrl.is_shadow() if ctrl is not None else bool(runtime_diag.get("shadow_mode", True))
        paused = ctrl.is_paused() if ctrl is not None else bool(runtime_diag.get("paused", False))
        hour_signals = int(runtime_diag.get("hour_signals_emitted") or healthcheck.get("signals_last_hour") or 0)
        hour_orders = int(runtime_diag.get("hour_order_placed") or healthcheck.get("fills_last_hour") or 0)
        pending_count = int(runtime_diag.get("pending_entry_count") or 0)
        model_quality = str(gate_quality.get("quality") or latest_metrics.get("quality") or "n/a")
        wf_bps = latest_metrics.get("walk_forward_expectancy_bps")
        champion_row = champion.get("champion") if isinstance(champion, dict) else {}
        champion_version = champion_row.get("version") if isinstance(champion_row, dict) else None
        model_horizon, shadow_gate = self._model_horizon_and_gate(
            db_diag,
            latest_metrics if isinstance(latest_metrics, dict) else {},
        )
        paper_by_horizon = db_diag.get("paper_pnl_by_horizon", {}) or {}
        paper_horizon = (
            paper_by_horizon.get(str(model_horizon))
            or db_diag.get(f"paper_pnl_{model_horizon}m")
            or db_diag.get("paper_pnl_15m")
            or {}
        )
        paper_baseline = paper_horizon.get("baseline", {}) if isinstance(paper_horizon, dict) else {}
        paper_model_gate = paper_horizon.get("model_gate", {}) if isinstance(paper_horizon, dict) else {}
        candle_sampler = runtime_diag.get("candle_sampler", {}) if isinstance(runtime_diag, dict) else {}
        shadow_closes = {
            "total": runtime_diag.get("hour_shadow_closed", 0) if isinstance(runtime_diag, dict) else 0,
            "tp": runtime_diag.get("hour_shadow_closed_tp", 0) if isinstance(runtime_diag, dict) else 0,
            "sl": runtime_diag.get("hour_shadow_closed_sl", 0) if isinstance(runtime_diag, dict) else 0,
            "time": runtime_diag.get("hour_shadow_closed_time", 0) if isinstance(runtime_diag, dict) else 0,
            "avg_pnl_pct": runtime_diag.get("hour_shadow_closed_avg_pnl_pct")
            if isinstance(runtime_diag, dict)
            else None,
        }

        blockers: list[str] = []
        if shadow:
            blockers.append("SHADOW включен: реальные ордера намеренно не отправляются.")
        if paused:
            blockers.append("Бот на паузе.")
        if model_quality.upper() != "GOOD":
            blockers.append(f"Качество модели не GOOD: {self._ru(model_quality)}.")
        try:
            if wf_bps is not None and float(wf_bps) <= 0:
                blockers.append(f"Walk-forward <= 0 bps: {float(wf_bps):+.2f} bps.")
        except (TypeError, ValueError):
            pass
        if not champion_version:
            blockers.append("CHAMPION-модель отсутствует; есть только shadow challenger.")
        if hour_signals == 0:
            blockers.append("За последний час нет сигналов стратегии.")
        if pending_count > 0:
            blockers.append(f"Есть pending-заявки: {pending_count}.")

        signal_rows: list[dict[str, Any]] = []
        if ctrl is not None:
            for entry in list(ctrl.signal_log)[-20:]:
                signal_rows.append(
                    {
                        "timestamp": entry.timestamp.isoformat(),
                        "symbol": entry.symbol,
                        "side": entry.side,
                        "confidence": entry.confidence,
                        "regime": entry.regime,
                        "shadow": entry.shadow,
                        "rationale": entry.rationale,
                    }
                )

        compact = {
            "generated_at": generated_at,
            "deploy": deploy,
            "runtime": {
                "status": runtime_diag.get("status") if isinstance(runtime_diag, dict) else None,
                "trading_mode": runtime_diag.get("trading_mode") if isinstance(runtime_diag, dict) else None,
                "shadow_mode": runtime_diag.get("shadow_mode") if isinstance(runtime_diag, dict) else shadow,
                "paused": runtime_diag.get("paused") if isinstance(runtime_diag, dict) else paused,
                "active_symbols": runtime_diag.get("active_symbols") if isinstance(runtime_diag, dict) else [],
                "execution_candidates": runtime_diag.get("execution_candidates")
                if isinstance(runtime_diag, dict)
                else None,
                "open_positions": runtime_diag.get("open_positions") if isinstance(runtime_diag, dict) else [],
                "pending_entry_count": pending_count,
                "last_ws_message_age_s": runtime_diag.get("last_ws_message_age_s")
                if isinstance(runtime_diag, dict)
                else None,
                "last_feature_age_s": runtime_diag.get("last_feature_age_s")
                if isinstance(runtime_diag, dict)
                else None,
                "model_version": runtime_diag.get("model_version") if isinstance(runtime_diag, dict) else None,
                "model_gate_quality": runtime_diag.get("model_gate_quality")
                if isinstance(runtime_diag, dict)
                else None,
            },
            "model": {
                "quality": model_quality,
                "walk_forward_expectancy_bps": wf_bps,
                "champion_version": champion_version,
                "latest_model_version": latest_model.get("version") if isinstance(latest_model, dict) else None,
                "latest_status": latest_model.get("status") if isinstance(latest_model, dict) else None,
                "shadow_gate_quality": gate_quality,
            },
            "virtual_orders": {
                "explanation": (
                    "strategy_signals are actual rule proposals; candle_sampler checks score every confirmed "
                    "candle so model pass/block and +/- outcomes can accumulate even when strategy signals are rare"
                ),
                "model_horizon_minutes": model_horizon,
                "strategy_paper_baseline": paper_baseline,
                "strategy_paper_model_gate": paper_model_gate,
                "shadow_gate": shadow_gate,
                "candle_sampler_runtime": candle_sampler,
                "shadow_closes_runtime": shadow_closes,
            },
            "hour": {
                "signals": hour_signals,
                "orders": hour_orders,
                "risk_rejected": runtime_diag.get("hour_risk_rejected") if isinstance(runtime_diag, dict) else None,
                "api_rejected": runtime_diag.get("hour_api_rejected") if isinstance(runtime_diag, dict) else None,
                "spread_rejected": runtime_diag.get("hour_spread_rejected") if isinstance(runtime_diag, dict) else None,
                "imbalance_rejected": runtime_diag.get("hour_imbalance_rejected")
                if isinstance(runtime_diag, dict)
                else None,
                "model_gate_blocked": runtime_diag.get("hour_model_gate_canary_blocked")
                if isinstance(runtime_diag, dict)
                else None,
            },
            "db": {
                "connected": db_diag.get("connected"),
                "error": db_diag.get("error"),
                "candles_by_interval": db_diag.get("candles_by_interval"),
                "feature_snapshots": db_diag.get("feature_snapshots"),
                "prediction_outcomes": db_diag.get("prediction_outcomes"),
                "training_eligible_by_horizon": db_diag.get("training_eligible_by_horizon"),
                "latest_training_run": db_diag.get("latest_training_run"),
                "training_config": db_diag.get("training_config"),
            },
        }

        lines = [
            "🧾 <b>ПОЛНАЯ СВОДКА ДЛЯ АНАЛИЗА</b>",
            f"Сгенерировано UTC: <code>{html.escape(generated_at)}</code>",
            "Скопируйте все части этого сообщения в Cursor/ChatGPT для разбора.",
            "",
            "<b>Короткий вывод: когда будут ордера</b>",
            f"Реальные ордера сейчас: <code>{'нет' if blockers else 'условия близки к готовности'}</code>",
            f"SHADOW: <code>{'да' if shadow else 'нет'}</code> | Пауза: <code>{'да' if paused else 'нет'}</code>",
            f"Сигналов за час: <code>{hour_signals}</code> | Ордеров за час: <code>{hour_orders}</code>",
            f"Модель: <code>{html.escape(str(compact['runtime']['model_version'] or 'n/a'))}</code>",
            f"Качество: <code>{html.escape(self._ru(model_quality))}</code> | Champion: <code>{html.escape(str(champion_version or 'нет'))}</code>",
            "",
            "<b>Условные сделки / virtual orders</b>",
            "• <b>Strategy paper</b> — реальные сигналы стратегии: "
            f"baseline=<code>{int((paper_baseline or {}).get('count') or 0)}</code>, "
            f"model_gate=<code>{int((paper_model_gate or {}).get('count') or 0)}</code>",
            "• <b>Candle-sampler checks</b> — модель оценивает каждую подтверждённую свечу: "
            f"scored=<code>{int((candle_sampler or {}).get('scored') or 0)}</code>, "
            f"pass=<code>{int((candle_sampler or {}).get('gate_pass') or 0)}</code>, "
            f"block=<code>{int((candle_sampler or {}).get('gate_block') or 0)}</code>, "
            f"thr=<code>{html.escape(str((candle_sampler or {}).get('threshold', 'n/a')))}</code>/"
            f"<code>{html.escape(str((candle_sampler or {}).get('threshold_source', 'n/a')))}</code>, "
            f"pass-rate=<code>{html.escape(str((candle_sampler or {}).get('pass_rate_pct', 'n/a')))}%</code>",
            "• <b>Shadow closes</b>: "
            f"total=<code>{int(shadow_closes.get('total') or 0)}</code>, "
            f"TP=<code>{int(shadow_closes.get('tp') or 0)}</code>, "
            f"SL=<code>{int(shadow_closes.get('sl') or 0)}</code>, "
            f"TIME=<code>{int(shadow_closes.get('time') or 0)}</code>, "
            f"avg=<code>{html.escape(str(shadow_closes.get('avg_pnl_pct') if shadow_closes.get('avg_pnl_pct') is not None else 'n/a'))}%</code>",
            f"• Shadow gate ({model_horizon}m): resolved=<code>{int((shadow_gate or {}).get('total_count') or 0)}</code>, "
            f"observed=<code>{int((shadow_gate or {}).get('event_total_count') or (shadow_gate or {}).get('total_count') or 0)}</code>, "
            f"pending=<code>{int((shadow_gate or {}).get('event_pending_count') or 0)}</code>, "
            f"pass=<code>{int((shadow_gate or {}).get('pass_count') or 0)}</code>, "
            f"lift=<code>{html.escape(str((shadow_gate or {}).get('lift_vs_all_bps') or 'n/a'))}</code>",
            "",
            "<b>Strategy signal pipeline</b>",
            f"• Emitted за час: <code>{hour_signals}</code>; "
            f"risk rejects: <code>{int(runtime_diag.get('hour_risk_rejected') or 0) if isinstance(runtime_diag, dict) else 0}</code>; "
            f"in-memory last signals: <code>{len(signal_rows)}</code>/20",
            "• Если emitted &gt; 0, а Strategy paper/Shadow closes ещё 0 — это обычно значит: "
            f"ждём TP/SL/TIME или {model_horizon}m outcome; следующий отчёт должен показать resolved/pending.",
        ]
        runtime_explainers = self._render_runtime_explainers(runtime_diag if isinstance(runtime_diag, dict) else {})
        if runtime_explainers:
            lines.extend(["", *runtime_explainers])
        if blockers:
            lines.append("\n<b>Блокеры реальных ордеров / CANARY</b>")
            lines.extend(f"❌ {html.escape(item)}" for item in blockers)
        else:
            lines.append("\n✅ Критичных блокеров в краткой сводке нет; проверьте readiness ниже.")

        lines += [
            "",
            "<b>CANARY readiness</b>",
            canary_text,
            "",
            "<b>Compact JSON</b>",
            f"<pre>{self._json_for_report(compact)}</pre>",
            "",
            "<b>Last 20 in-memory signals</b>",
            f"<pre>{self._json_for_report(signal_rows)}</pre>",
            "",
            "<b>Healthcheck JSON</b>",
            f"<pre>{self._json_for_report(healthcheck)}</pre>",
            "",
            "<b>Runtime settings JSON</b>",
            f"<pre>{self._json_for_report(runtime_settings)}</pre>",
            "",
            "<b>Runtime diagnostics JSON</b>",
            f"<pre>{self._json_for_report(runtime_diag)}</pre>",
            "",
            "<b>DB diagnostics JSON</b>",
            f"<pre>{self._json_for_report(db_diag)}</pre>",
            "",
            "<b>Model performance JSON</b>",
            f"<pre>{self._json_for_report(model_rows)}</pre>",
            "",
            "<b>Champion health JSON</b>",
            f"<pre>{self._json_for_report(champion)}</pre>",
            "",
            "<b>Compare / PnL / Costs JSON</b>",
            f"<pre>{self._json_for_report({'compare': compare, 'pnl_analysis': pnl_analysis, 'costs': costs})}</pre>",
        ]
        if log_tail:
            lines.extend(
                [
                    "",
                    "<b>Render / app log tail</b>",
                    "<i>Берётся из env RENDER_LOG_TAIL или APP_LOG_TAIL, если окружение его передало.</i>",
                    f"<pre>{log_tail}</pre>",
                ]
            )
        return "\n".join(lines)

    async def _cmd_canary_ready(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show whether the system is ready for a tiny CANARY_LIVE test."""
        del context
        if not await self._authorised(update):
            return

        text = await self._render_canary_readiness_text(lite=True)
        await self._respond(
            update,
            text,
            reply_markup=self._canary_menu(),
        )

    async def _cmd_priorities(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Explain what matters more than what: safety, filters, CANARY, strategies."""
        del context
        if not await self._authorised(update):
            return
        runtime: dict[str, Any] = {}
        if self._controller is not None and self._controller.runtime_settings is not None:
            try:
                runtime = self._controller.runtime_settings()
            except Exception:
                runtime = {}
        text = full_priority_overview(runtime_settings=runtime)
        await self._respond(update, text, reply_markup=self._canary_menu())

    async def _render_canary_readiness_text(self, *, lite: bool = True) -> str:
        db_diag = await self._load_db_diag(lite=lite)

        diag: dict[str, Any] = {}
        if self._controller is not None and self._controller.diagnostics_provider is not None:
            try:
                diag = self._controller.diagnostics_provider()
            except Exception as exc:
                diag = {"error": str(exc)}

        text = self._canary_readiness_text(db_diag=db_diag, diag=diag)
        banner = self._db_diag_banner(db_diag)
        if banner:
            text = f"{banner}\n\n{text}"
        return text

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
        runtime_candles = diag.get("runtime_candles_by_interval") or db_diag.get("runtime_candles_by_interval") or {}
        if runtime_candles:
            merged = dict(candles)
            for interval, runtime_count in runtime_candles.items():
                if int(runtime_count) > int(merged.get(interval) or 0):
                    merged[interval] = int(runtime_count)
            candles = merged
        elif not any(int(candles.get(key) or 0) for key in ("1", "5", "15", "60")):
            candles = runtime_candles or candles
        latest_1m = db_diag.get("latest_candle_1m")
        latest_age_s = self._utc_age_seconds(latest_1m)
        if latest_age_s is None:
            runtime_age = diag.get("last_confirmed_candle_age_s")
            if runtime_age is not None:
                latest_age_s = float(runtime_age)
        active_symbols = diag.get("active_symbols") or []
        runtime = self._controller.runtime_settings() if self._controller and self._controller.runtime_settings else {}
        latest_run = db_diag.get("latest_training_run", {}) or {}
        latest_model = db_diag.get("latest_model_version", {}) or {}
        active_model = db_diag.get("active_model_version", {}) or latest_model
        readiness_model = active_model or latest_model
        model_info = diag.get("model", {}) or {}
        model_metrics = readiness_model.get("metrics") or latest_run.get("metrics") or {}
        if isinstance(model_metrics, str):
            try:
                model_metrics = json.loads(model_metrics)
            except json.JSONDecodeError:
                model_metrics = {}
        try:
            model_horizon = int(db_diag.get("model_gate_horizon_minutes") or model_metrics.get("horizon_minutes") or 15)
        except (TypeError, ValueError):
            model_horizon = 15
        gate_by_horizon = db_diag.get("shadow_gate_by_horizon", {}) or {}
        paper_by_horizon = db_diag.get("paper_pnl_by_horizon", {}) or {}
        gate = (
            gate_by_horizon.get(str(model_horizon))
            or db_diag.get(f"shadow_gate_{model_horizon}m")
            or db_diag.get("shadow_gate_15m", {})
            or {}
        )
        paper = (
            paper_by_horizon.get(str(model_horizon))
            or db_diag.get(f"paper_pnl_{model_horizon}m")
            or db_diag.get("paper_pnl_15m", {})
            or {}
        )
        paper_baseline = paper.get("baseline", {}) or {}
        paper_gate = paper.get("model_gate", {}) or {}

        def _as_float(value: Any) -> float | None:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        training_by_horizon = db_diag.get("training_eligible_by_horizon", {}) or {}
        db_diag_timed_out = bool(
            db_diag.get("full_diagnostics_timeout") or db_diag.get("error") == "db_diagnostics_timeout"
        )

        def _optional_int(value: Any) -> int | None:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        trainable_model_horizon = _optional_int(
            training_by_horizon.get(
                str(model_horizon),
                db_diag.get("training_eligible_15m", db_diag.get("labelled_samples_15m")),
            )
        )
        prediction_outcomes = _optional_int(db_diag.get("prediction_outcomes"))
        feature_snapshots = _optional_int(db_diag.get("feature_snapshots"))
        gate_total = int(gate.get("total_count") or 0)
        gate_lift = _as_float(gate.get("lift_vs_all_bps"))
        paper_gate_count = int(paper_gate.get("count") or 0)
        paper_gate_bps = _as_float(paper_gate.get("total_bps")) or 0.0
        paper_base_count = int(paper_baseline.get("count") or 0)
        model_version = readiness_model.get("version")
        model_quality = str(model_metrics.get("quality") or "n/a")
        walk_forward_bps = _as_float(model_metrics.get("walk_forward_expectancy_bps"))
        champion_ver = model_info.get("champion_version", "none")
        if champion_ver == "none" and readiness_model.get("status") == "CHAMPION":
            champion_ver = model_version or "none"
        model_quality_ok = bool(model_version) and model_quality in ("GOOD", "ХОРОШО")
        walk_forward_ok = walk_forward_bps is not None and walk_forward_bps > 0
        champion_ok = champion_ver not in ("none", "", None)
        gate_ready = gate_total >= 50 and gate_lift is not None and gate_lift > 0
        paper_gate_ready = paper_gate_count >= 20 and paper_gate_bps > 0
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
            feature_snapshots is not None and feature_snapshots >= 1000,
            f"{feature_snapshots}/1000 снимков"
            if feature_snapshots is not None
            else "н/д — DB diagnostics timeout, не считаю как 0",
            "Оставьте SHADOW включенным; бот будет сохранять признаки на каждом цикле стратегии."
            if feature_snapshots is not None
            else "Повторите отчёт позже или проверьте pgbouncer/POSTGRES_DSN: быстрый отчёт не успел прочитать счётчик.",
            self._eta_for_samples(1000 - feature_snapshots, active_count * 240)
            if feature_snapshots is not None
            else "",
        )
        require(
            f"Размеченные примеры {model_horizon}m",
            trainable_model_horizon is not None and trainable_model_horizon >= 1000,
            f"{trainable_model_horizon}/1000 примеров"
            if trainable_model_horizon is not None
            else "н/д — DB diagnostics timeout, не считаю как 0",
            f"Дождитесь закрытия {model_horizon}m исходов или загрузите историю свечей backfill, затем запустите обучение."
            if trainable_model_horizon is not None
            else "Повторите отчёт позже: полный DB-блок не успел посчитать training eligibility.",
            self._eta_for_samples(1000 - trainable_model_horizon, active_count * 4)
            if trainable_model_horizon is not None
            else "",
        )
        require(
            "Результаты прогнозов",
            prediction_outcomes is not None and prediction_outcomes >= 1000,
            f"{prediction_outcomes}/1000 исходов"
            if prediction_outcomes is not None
            else "н/д — DB diagnostics timeout, не считаю как 0",
            f"Проверьте задачу outcome-resolver: она должна сопоставлять сигналы с результатом через {model_horizon} минут."
            if prediction_outcomes is not None
            else "Повторите отчёт позже: полный DB-блок не успел посчитать prediction outcomes.",
            self._eta_for_samples(1000 - prediction_outcomes, active_count * 4)
            if prediction_outcomes is not None
            else "",
        )
        require(
            "Модель обучена",
            bool(model_version),
            f"запуск={self._ru(latest_run.get('status', 'none'))}, модель={model_version or 'нет'}",
            "Нажмите 'Обучить 1000' или выполните python -m trader.training.train --min-samples 1000.",
            "10-20 минут после накопления 1000 примеров",
        )
        require(
            "Качество модели GOOD",
            model_quality_ok,
            self._ru(model_quality) if model_version else "модель еще не обучена",
            "Не включайте CANARY_LIVE: дождитесь новой модели с quality=GOOD или меняйте стратегию/признаки.",
        )
        require(
            "Walk-forward модели > 0 bps",
            walk_forward_ok,
            f"{walk_forward_bps:+.2f} bps" if walk_forward_bps is not None else "n/a",
            "Не промоутируйте модель и не включайте model-gate, пока walk-forward отрицательный.",
        )
        require(
            "Основная модель CHAMPION",
            champion_ok,
            f"есть: {champion_ver}" if champion_ok else "нет",
            "Промоутируйте только кандидата, который прошел quality, walk-forward, gate lift и paper-gate PnL.",
        )
        require(
            "Lift фильтра > 0 bps на 50+ сигналах",
            gate_ready,
            f"{gate_lift:+.2f} bps на {gate_total} сигналах"
            if gate_lift is not None
            else f"n/a на {gate_total} сигналах",
            "Дайте модели поработать в тени; если lift остается <= 0, меняйте стратегию/признаки.",
        )
        require(
            "Paper model-gate: 20+ сделок и PnL > 0",
            paper_gate_ready,
            f"{paper_gate_count} сделок, {paper_gate_bps:+.1f} bps",
            "Дождитесь 20+ бумажных сделок; если PnL <= 0, CANARY_LIVE запускать нельзя.",
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
            ws_age is None or float(ws_age) <= 30,
            self._age_label_ru(ws_age),
            "Перезапустите сервис или проверьте доступ к Bybit WebSocket.",
            "1-3 минуты после восстановления соединения",
        )
        confirmed_age = diag.get("last_confirmed_candle_age_s")
        require(
            "Свежие подтверждённые свечи",
            confirmed_age is not None and confirmed_age <= 120,
            self._age_label_ru(confirmed_age),
            "Проверьте что WS доставляет confirmed=True события в CandleStore.",
            "1-2 минуты после восстановления WS",
        )

        warn_if(
            trainable_model_horizon is not None and trainable_model_horizon < 2000,
            f"Размеченных {model_horizon}m примеров {trainable_model_horizon}; для более уверенного CANARY лучше 2000+.",
            "Можно начать с 1000 для первого кандидата, но перед реальными деньгами лучше добрать данные.",
        )
        warn_if(
            db_diag_timed_out,
            "Полная DB-диагностика не успела ответить; часть счётчиков показана как н/д, а не как 0.",
            "Это не торговый стоппер само по себе, но для анализа лучше повторить отчёт или проверить медленные SQL/pgbouncer.",
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
                (
                    "Нет времени последнего цикла стратегии.",
                    "Проверьте, что strategy-loop запущен и не падает.",
                )
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
        if not estimates and failed:
            estimates.append(
                "• Данные уже могут быть собраны, но CANARY нельзя включать до положительного качества модели и paper-gate PnL."
            )
        if not estimates and warnings:
            estimates.append("• Обязательные условия закрыты; остались только некритичные предупреждения.")
        if not estimates:
            estimates.append("• Технически и модельно готово сейчас.")

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
                "Если есть ❌ по модели или paper-gate, CANARY_LIVE не включаем: модель продолжает наблюдать в SHADOW.",
                "Если все обязательные условия ✅, CANARY держим маленьким: 1-2 позиции, минимальный notional.",
                "Telegram не включает live: реальные деньги включаются только через env vars на Render.",
                "",
                canary_readiness_priority_text(),
                "",
                "Подробный разбор безопасности и стратегий: /priorities",
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
        return float(max(0.0, (datetime.now(UTC) - value.astimezone(UTC)).total_seconds()))

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

    @staticmethod
    def _fmt_timestamp(ts: Any) -> str:
        """Format a datetime or ISO string; returns 'нет' if absent."""
        if ts is None:
            return "нет"
        from datetime import datetime as _dt

        if isinstance(ts, _dt):
            return ts.strftime("%H:%M:%S UTC")
        try:
            parsed = _dt.fromisoformat(str(ts))
            return parsed.strftime("%H:%M:%S UTC")
        except Exception:
            return str(ts)[:19]

    async def _cmd_model_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Explain model/training screens in operator-friendly Russian."""
        del context
        if not await self._authorised(update):
            return
        await self._reply(update, self._model_help_text(), reply_markup=self._control_menu())

    def _model_help_text(self) -> str:
        min_samples, horizon, label_bps = self._train_defaults()
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
            f"✓ Scalp v2 примеры {horizon}m ≥ {min_samples} (с pool <code>scalp_micro_v1</code>)\n\n"
            "<b>Шаг 2 — Обучить модель</b>\n"
            f"✓ «Обучить {min_samples}» или авто-train ({horizon}m / {label_bps:g} bps)\n"
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
        try:
            db_diag = await self._load_db_diag(lite=update.callback_query is not None)
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
            if db_diag.get("candles_source") == "runtime_fallback":
                candles_note = " (из памяти WS, Postgres медленный)"
            else:
                candles_note = ""
            latest_1m = db_diag.get("latest_candle_1m")
            latest_str = self._fmt_timestamp(latest_1m)
            outcomes_by_horizon = db_diag.get("prediction_outcomes_by_horizon", {}) or {}
            labelled_15m = db_diag.get("labelled_samples_15m", 0)
            training_by_horizon = db_diag.get("training_eligible_by_horizon", {}) or {}
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
            best_threshold_avg_str = (
                f"{float(best_threshold_avg):+.2f} bps" if best_threshold_avg is not None else "n/a"
            )
            model_horizon, gate = self._model_horizon_and_gate(db_diag, model_metrics)
            training_config = db_diag.get("training_config", {}) or {}
            pool_breakdown = db_diag.get("training_pool_breakdown", {}) or {}
            model_horizon_key = str(model_horizon)
            trainable_model_horizon = int(training_by_horizon.get(model_horizon_key, labelled_15m) or 0)
            filtered_total_by_horizon = db_diag.get("training_filtered_total_by_horizon", {}) or {}
            schema_by_horizon = db_diag.get("newest_training_schema_by_horizon", {}) or {}
            horizon_schema = schema_by_horizon.get(model_horizon_key, {}) or {}
            filtered_total_horizon = int(filtered_total_by_horizon.get(model_horizon_key, 0) or 0)
            if not filtered_total_horizon:
                filtered_total_horizon = int(pool_breakdown.get(f"filtered_total_{model_horizon}m", 0) or 0)
            filtered_total_5m = int(pool_breakdown.get("filtered_total_5m", 0) or 0)
            newest_schema_count = int(horizon_schema.get("sample_count", 0) or 0)
            best_schema_count = int(horizon_schema.get("best_schema_count", 0) or 0)
            trainable_schema_hash = str(horizon_schema.get("trainable_schema_hash", "") or "")
            best_schema_hash = str(horizon_schema.get("best_schema_hash", "") or "")
            newest_schema_hash = str(horizon_schema.get("feature_schema_hash", "") or "")
            loose_v2_5m = int((db_diag.get("prediction_outcomes_by_horizon", {}) or {}).get(str(model_horizon), 0) or 0)
            train_allowlist = training_config.get("strategy_allowlist") or []
            train_include_candle = training_config.get("include_candle_baseline")
            train_label_schema = training_config.get("label_schema_version") or db_diag.get(
                "label_schema_version", "n/a"
            )
            scalp_active = int(pool_breakdown.get("scalp_micro_v1_active_schema", 0) or 0)
            probe_hv_active = int(pool_breakdown.get("shadow_probe_hv_v2_active_schema", 0) or 0)
            legacy_candle_v1 = int(pool_breakdown.get("legacy_v1_candle_baseline", 0) or 0)
            other_active = int(pool_breakdown.get("other_active_schema", 0) or 0)
            candle_sampler_active = int(pool_breakdown.get("candle_sampler_v1_active_schema", 0) or 0)
            schema_compatible = bool((db_diag.get("latest_model_version") or {}).get("schema_compatible", False))
            allowlist_display = ",".join(train_allowlist) if train_allowlist else "ALL"
            include_candle_display = (
                "true" if train_include_candle is True else "false" if train_include_candle is False else "n/a"
            )
            gate_total = gate.get("total_count", 0) or 0
            gate_observed = int(gate.get("event_total_count", gate_total) or 0)
            gate_pending = int(gate.get("event_pending_count", 0) or 0)
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
            paper_by_horizon = db_diag.get("paper_pnl_by_horizon", {}) or {}
            paper = (
                paper_by_horizon.get(str(model_horizon))
                or db_diag.get(f"paper_pnl_{model_horizon}m")
                or db_diag.get("paper_pnl_15m")
                or {}
            )
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

            data_note = (
                "данные собираются"
                if connected and trainable_model_horizon >= 1000
                else "мало данных для уверенного обучения"
            )
            training_note = "модель-кандидат обучена" if db_model_version else "модель еще не обучена"
            champion_display = str(champion_ver or "none")
            if champion_display == "none" and challenger_ver not in ("none", "", None):
                champion_display = "none (CHAMPION нет; кандидат только наблюдает)"
            challenger_display = str(challenger_ver or "none")
            if gate_lift is None:
                gate_note = "фильтр модели еще не оценен"
            elif float(gate_lift) > 0:
                gate_note = "фильтр модели улучшает отбор сигналов"
            else:
                gate_note = "фильтр модели пока НЕ улучшает отбор сигналов"

            lines = [
                "<b>🗄 БАЗА И МОДЕЛЬ</b>",
                "",
            ]
            banner = self._db_diag_banner(db_diag)
            if banner:
                lines.append(banner)
                lines.append("")
            lines += [
                f"БД: {db_icon} {db_status}",
                f"Ошибка БД: <code>{db_error_str or 'нет'}</code>",
                f"Последняя свеча 1m: <code>{latest_str}</code>",
                f"Свечей 1m:  <code>{candles.get('1', 0)}</code>{candles_note}",
                f"Свечей 5m:  <code>{candles.get('5', 0)}</code>{candles_note}",
                f"Свечей 15m: <code>{candles.get('15', 0)}</code>{candles_note}",
                f"Свечей 1h:  <code>{candles.get('60', 0)}</code>{candles_note}",
                f"Снимки признаков: <code>{db_diag.get('feature_snapshots', 0)}</code>",
                f"Размеченные исходы: <code>{db_diag.get('prediction_outcomes', 0)}</code>",
            ]
            storage = db_diag.get("storage_stats") or {}
            if storage.get("database_size_mb") is not None:
                lines.append(f"Размер БД: <code>{storage.get('database_size_mb')} MB</code>")
            invalid_snaps = storage.get("feature_snapshots_invalid")
            if invalid_snaps is not None:
                lines.append(f"Invalid snapshots: <code>{invalid_snaps}</code>")
            lines += [
                f"Горизонты разметки: <code>{outcome_breakdown}</code>",
                f"Готово для обучения ({model_horizon}m): <code>{trainable_model_horizon}</code>",
            ]
            if horizon_schema:
                schema_bits = [
                    f"filtered=<code>{filtered_total_horizon}</code>",
                    f"newest=<code>{newest_schema_count}</code>",
                    f"best=<code>{best_schema_count}</code>",
                ]
                if trainable_schema_hash:
                    schema_bits.append(f"trainable=<code>{html.escape(trainable_schema_hash[:8])}</code>")
                elif best_schema_hash or newest_schema_hash:
                    schema_bits.append(
                        f"ждём schema <code>{html.escape((best_schema_hash or newest_schema_hash)[:8])}</code> до 1000"
                    )
                lines.append(f"Схема обучения ({model_horizon}m): " + " | ".join(schema_bits))
            if loose_v2_5m and trainable_model_horizon < loose_v2_5m:
                lines.append(
                    f"Размечено v2 без фильтра scalp: <code>{loose_v2_5m}</code> "
                    f"(в пуле allowlist всего: <code>{filtered_total_5m}</code>)"
                )
            if scalp_active == 0 and trainable_model_horizon < 1000:
                lines.append(
                    "⚠️ <i>Scalp-пул пуст: обучение ждёт сигналы scalp_micro_v1. "
                    "Сейчас v2-метки в основном от candle_sampler (не входит в allowlist).</i>"
                )
            lines += [
                f"Фильтр обучения: schema=<code>{html.escape(str(train_label_schema))}</code>, "
                f"allowlist=<code>{html.escape(allowlist_display)}</code>, "
                f"candle_baseline=<code>{include_candle_display}</code>",
                f"Пул scalp (активная schema): <code>{scalp_active}</code> | "
                f"HV probe v2: <code>{probe_hv_active}</code> | "
                f"legacy candle v1 (исключён): <code>{legacy_candle_v1}</code>",
            ]
            if candle_sampler_active or other_active:
                lines.append(
                    f"Исключённые/прочие пулы v2: candle_sampler=<code>{candle_sampler_active}</code>, "
                    f"другое=<code>{other_active}</code>"
                )
            if db_model_version and not schema_compatible:
                lines.append(
                    "⚠️ <i>Модель в БД устарела (другая schema разметки). "
                    "Ждём авто-переобучение на v2+scalp — метрики 7451/dnv1 не актуальны.</i>"
                )
            lines += [
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
                f"Основная модель: <code>{html.escape(champion_display)}</code>",
                f"Кандидат: <code>{html.escape(challenger_display)}</code>",
                f"Последняя модель в БД: <code>{db_model_version or 'нет'}</code> <code>{self._ru(db_model_status)}</code>",
                f"Последний запуск: <code>{self._ru(latest_run_status)}</code>, примеров=<code>{latest_run_samples}</code>",
                f"Версия из запуска: <code>{latest_run_model}</code>",
                f"Качество: <code>{self._ru(model_quality)}</code>",
                f"Проверочных примеров: <code>{validation_samples}</code>",
                f"Точность прибыльных сигналов: <code>{precision_str}</code>",
                f"Улучшение против baseline: <code>{lift_str}</code>",
                f"Лучший порог модели: <code>{best_threshold_str}</code>, среднее=<code>{best_threshold_avg_str}</code>",
                f"Ожидание walk-forward: <code>{expectancy_str if expectancy_bps is not None else wf_exp}</code>",
                f"Фильтр модели {model_horizon}m: <code>{gate_pass}/{gate_total} resolved пропущено</code>, "
                f"блок=<code>{gate_block}</code>, observed=<code>{gate_observed}</code>, pending=<code>{gate_pending}</code>",
                f"Среднее пропущенных: <code>{gate_pass_avg_str}</code>",
                f"Среднее заблокированных: <code>{gate_block_avg_str}</code>",
                "Lift фильтра: <code>"
                + (
                    f"⏳ ждём outcome ({gate_pending} pending)"
                    if gate_total == 0 and gate_observed > 0
                    else ("⏳ ждём ~50 живых сигналов" if gate_total == 0 and db_model_version else gate_lift_str)
                )
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
            lbl_ok = int(trainable_model_horizon or 0) >= 1000
            lbl_detail = f"{trainable_model_horizon}"
            if horizon_schema:
                lbl_detail += (
                    f" (filtered {filtered_total_horizon}, best schema {best_schema_count}, "
                    f"newest {newest_schema_count})"
                )
            lines.append(
                f"{'✅' if lbl_ok else '❌'} Совместимых trainable-семплов ({model_horizon}m) ≥ 1000 → сейчас: "
                f"<code>{html.escape(lbl_detail)}</code>"
            )
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
                gate_road_val = (
                    f"ждём outcome: {gate_pending} pending из {gate_observed} observed"
                    if gate_observed > 0
                    else f"ждём ~50 сигналов (сейчас {gate_total})"
                )
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

            await self._respond(update, "\n".join(lines), reply_markup=self._main_menu())
        except Exception as exc:
            log.warning("telegram.db_model_render_failed", error=str(exc), exc_info=True)
            err_text = html.escape(str(exc))[:200]
            try:
                await self._respond(
                    update,
                    f"<b>База и модель</b>\nНе удалось отправить диагностику.\nОшибка: <code>{err_text}</code>",
                    reply_markup=self._main_menu(),
                )
            except Exception as _reply_exc:
                log.debug("telegram.db_model_fallback_reply_failed", error=str(_reply_exc))

    # ------------------------------------------------------------------
    # Control commands
    # ------------------------------------------------------------------

    def _confirm_menu(self, action: str) -> InlineKeyboardMarkup:
        # One-shot nonce: confirm buttons live forever in chat history, so a
        # bare confirm:<action> payload could be replayed months later. The
        # nonce binds the button to THIS dialog and expires after the TTL.
        nonce = secrets.token_hex(8)
        self._confirm_nonces[nonce] = (action, datetime.now(tz=UTC))
        self._prune_confirm_nonces()
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("✅ Да", callback_data=f"confirm:{nonce}:{action}:yes"),
                    InlineKeyboardButton("❌ Нет", callback_data=f"confirm:{nonce}:{action}:no"),
                ]
            ]
        )

    def _prune_confirm_nonces(self) -> None:
        now = datetime.now(tz=UTC)
        expired = [
            nonce
            for nonce, (_, created_at) in self._confirm_nonces.items()
            if (now - created_at).total_seconds() > _CONFIRM_TTL_SECONDS
        ]
        for nonce in expired:
            self._confirm_nonces.pop(nonce, None)
        # Hard cap as a second line of defence against unbounded growth
        while len(self._confirm_nonces) > _CONFIRM_MAX_PENDING:
            self._confirm_nonces.pop(next(iter(self._confirm_nonces)), None)

    def _consume_confirm_nonce(self, nonce: str, action: str) -> bool:
        """Validate and invalidate a confirm nonce (one shot, TTL-bound)."""
        entry = self._confirm_nonces.pop(nonce, None)
        if entry is None:
            return False
        stored_action, created_at = entry
        if stored_action != action:
            return False
        return (datetime.now(tz=UTC) - created_at).total_seconds() <= _CONFIRM_TTL_SECONDS

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        if self._controller is None:
            await self._reply(update, "Управление сейчас недоступно.")
            return
        await self._reply(
            update,
            "⏸ Поставить бота на <b>паузу</b>? Новые входы открываться не будут.",
            reply_markup=self._confirm_menu("pause"),
        )

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
        await self._reply(
            update,
            "▶️ <b>Возобновить</b> работу бота? Сбор данных и новые входы будут снова разрешены.",
            reply_markup=self._confirm_menu("resume"),
        )

    async def _handle_confirm_button(self, update: Update, payload: str) -> None:
        """Handle confirm:<nonce>:<action>:<yes|no> inline buttons."""
        if self._controller is None:
            await self._button_reply(update, "Управление сейчас недоступно.", reply_markup=self._main_menu())
            return
        try:
            nonce, rest = payload.split(":", maxsplit=1)
            action, answer = rest.rsplit(":", maxsplit=1)
        except ValueError:
            await self._button_reply(update, "Неизвестное подтверждение.", reply_markup=self._main_menu())
            return
        if answer != "yes":
            self._confirm_nonces.pop(nonce, None)
            await self._button_reply(update, "Действие отменено.", reply_markup=self._main_menu())
            return
        if not self._consume_confirm_nonce(nonce, action):
            # Covers replayed buttons from chat history, expired dialogs, and
            # legacy pre-nonce buttons (their first segment is not a nonce).
            await self._button_reply(
                update,
                "⏳ Кнопка подтверждения устарела. Запросите действие заново.",
                reply_markup=self._main_menu(),
            )
            return
        if action == "pause":
            await self._controller.pause()
            await self._button_reply(
                update,
                "⏸ Бот на <b>паузе</b>: новые входы не открываются.\nКоманда /resume снимет паузу.",
                reply_markup=self._main_menu(),
            )
            return
        if action == "resume":
            await self._controller.resume()
            await self._button_reply(
                update,
                "▶️ Бот <b>возобновлен</b>: сбор данных и новые входы снова разрешены.",
                reply_markup=self._main_menu(),
            )
            return
        if action == "stop":
            try:
                await self._controller.emergency_stop()
            except Exception as exc:
                await self._button_reply(
                    update,
                    f"❌ Остановка не удалась: <code>{html.escape(str(exc))}</code>",
                    reply_markup=self._main_menu(),
                )
                return
            log.info("telegram_control_confirmed", action="emergency_stop")
            await self._button_reply(
                update,
                "🚨 <b>Аварийная остановка выполнена.</b> Новые входы остановлены; для возобновления перезапустите сервис.",
                reply_markup=self._main_menu(),
            )
            return
        if action.startswith("train:"):
            if self._controller.start_training is None:
                await self._button_reply(
                    update,
                    "Запуск обучения сейчас недоступен.",
                    reply_markup=self._main_menu(),
                )
                return
            try:
                _, min_s_raw, horizon_raw, label_raw = action.split(":", maxsplit=3)
                min_samples = int(min_s_raw)
                horizon = int(horizon_raw)
                label_bps = float(label_raw)
                msg = await self._controller.start_training(min_samples, horizon, label_bps)
            except Exception as exc:
                msg = f"❌ Обучение не стартовало: <code>{html.escape(str(exc))}</code>"
            await self._button_reply(update, msg, reply_markup=self._main_menu())
            return
        if action == "train_all":
            if self._controller.start_training_all is None:
                await self._button_reply(
                    update,
                    "Обучение ВСЕ сейчас недоступно.",
                    reply_markup=self._main_menu(),
                )
                return
            try:
                msg = await self._controller.start_training_all()
            except Exception as exc:
                msg = f"❌ Обучение не стартовало: <code>{html.escape(str(exc))}</code>"
            await self._button_reply(update, msg, reply_markup=self._main_menu())
            return
        if action.startswith("promote:"):
            if self._controller.promote_model is None:
                await self._button_reply(update, "Промоут сейчас недоступен.", reply_markup=self._main_menu())
                return
            version = action.split(":", maxsplit=1)[1]
            try:
                msg = await self._controller.promote_model(version)
            except Exception as exc:
                msg = f"❌ Промоут не удался: <code>{html.escape(str(exc))}</code>"
            await self._button_reply(update, msg, reply_markup=self._main_menu())
            return
        await self._button_reply(update, "Неизвестное подтверждение.", reply_markup=self._main_menu())

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
                update,
                f"Shadow mode is currently <code>{current}</code>.\nUse: /shadow on  or  /shadow off",
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
        await self._reply(
            update,
            "🔦 Теневой режим: <code>включен, ордера считаются, но не отправляются</code>",
        )

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
                update,
                f"Текущий риск-профиль: <code>{current}</code>\nФормат: /risk {' | '.join(valid)}",
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
                datetime.now(tz=UTC),
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
        default_min, default_horizon, default_label_bps = self._train_defaults()
        try:
            min_samples = int(args[0]) if len(args) >= 1 else default_min
            horizon = int(args[1]) if len(args) >= 2 else default_horizon
            label_bps = float(args[2]) if len(args) >= 3 else default_label_bps
        except ValueError:
            await self._reply(update, "Формат: /train [примеров] [горизонт_минут] [порог_bps]")
            return
        if min_samples < 50 or horizon <= 0 or label_bps < 0:
            await self._reply(
                update,
                "Параметры обучения отклонены: примеров>=50, горизонт>0, bps>=0.",
            )
            return
        await self._reply(
            update,
            "🧠 <b>Запустить обучение?</b>\n\n"
            f"Примеры: <code>{min_samples}</code>, горизонт: <code>{horizon}m</code>, "
            f"порог: <code>{label_bps:g} bps</code>.",
            reply_markup=self._confirm_menu(f"train:{min_samples}:{horizon}:{label_bps:g}"),
        )

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
                "Формат: /limits entries|pending|same_side|max_positions|price_cap|feature_symbols|exec_candidates N\n"
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
        await self._reply(
            update,
            f"✅ {msg}\n\n{self._limits_text()}",
            reply_markup=self._control_menu(),
        )

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        if self._controller is None:
            await self._reply(update, "Управление сейчас недоступно.")
            return
        await self._reply(
            update,
            "🚨 <b>Аварийная остановка</b>\n\n"
            "Все новые входы будут полностью остановлены. "
            "Для возобновления потребуется ручной перезапуск сервиса.\n\n"
            "Выполнить?",
            reply_markup=self._confirm_menu("stop"),
        )

    async def _cmd_confirm(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        cid = self._chat_id(update)
        if cid is None or cid not in self._pending:
            await self._reply(update, "Нет действия, которое ждет подтверждения.")
            return
        action_name, action_fn, created_at = self._pending.pop(cid)
        if (datetime.now(tz=UTC) - created_at).total_seconds() > _CONFIRM_TTL_SECONDS:
            await self._reply(update, "⏳ Подтверждение устарело. Запросите действие заново.")
            return
        try:
            await action_fn()
            await self._reply(update, f"✅ Готово: <i>{action_name}</i>")
            log.info("telegram_control_confirmed", action=action_name, chat_id=cid)
        except Exception as exc:
            await self._reply(update, f"❌ Не получилось: <code>{exc}</code>")
            log.error("telegram_control_failed", action=action_name, error=str(exc))

    async def _show_canary_model_metrics(self, update: Update) -> None:
        """Second /canary screen: model quality metrics (lift, precision, gate stats)."""
        db_diag = await self._load_db_diag(lite=True)
        if db_diag.get("error") == "db_diagnostics_timeout":
            await self._button_reply(
                update,
                "⚠️ Postgres отвечает медленно — метрики модели временно недоступны.\n"
                "Попробуйте через минуту или откройте /canary снова.",
                reply_markup=self._canary_menu(),
            )
            return
        if db_diag.get("error"):
            await self._button_reply(
                update,
                f"❌ Не удалось получить метрики: <code>{html.escape(str(db_diag['error']))}</code>",
                reply_markup=self._canary_menu(),
            )
            return
        latest_model = db_diag.get("latest_model_version", {}) or {}
        metrics = latest_model.get("metrics") or {}
        if isinstance(metrics, str):
            try:
                import json as _json

                metrics = _json.loads(metrics)
            except Exception:
                metrics = {}
        model_horizon, gate = self._model_horizon_and_gate(db_diag, metrics)

        def _fmt(value: Any, suffix: str = "") -> str:
            if value is None:
                return "нет данных"
            try:
                return f"{float(value):+.3f}{suffix}" if suffix == " bps" else f"{float(value):.3f}{suffix}"
            except (TypeError, ValueError):
                return html.escape(str(value))

        gate_lift = gate.get("lift_vs_all_bps")
        lines = [
            "<b>📊 Метрики модели</b>",
            "",
            f"Версия: <code>{html.escape(str(latest_model.get('version') or 'нет'))}</code>",
            f"Статус: <code>{self._ru(latest_model.get('status', 'none'))}</code>",
            f"Качество: <code>{self._ru(str(metrics.get('quality') or 'нет данных'))}</code>",
            "",
            f"Val-split expectancy: <code>{_fmt(metrics.get('walk_forward_expectancy_bps') or metrics.get('best_threshold_avg_net_return_bps'), ' bps')}</code>",
            f"Precision (val): <code>{_fmt(metrics.get('val_precision') or metrics.get('precision'))}</code>",
            f"AUC (val): <code>{_fmt(metrics.get('val_auc') or metrics.get('auc'))}</code>",
            f"Bootstrap p-value: <code>{_fmt(metrics.get('bootstrap_p_value'))}</code>",
            "",
            f"<b>Shadow gate ({model_horizon}m):</b>",
            f"Решений: <code>{int(gate.get('total_count') or 0)}</code> | "
            f"Pass: <code>{int(gate.get('pass_count') or 0)}</code>",
            f"Gate lift: <code>{_fmt(gate_lift, ' bps')}</code>",
            f"Pass expectancy: <code>{_fmt(gate.get('pass_avg_net_return_bps'), ' bps')}</code>",
        ]
        await self._button_reply(update, "\n".join(lines), reply_markup=self._canary_menu())

    # ------------------------------------------------------------------
    # Subscriptions / trades / healthcheck
    # ------------------------------------------------------------------

    async def _cmd_subscribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        cid = self._chat_id(update)
        if cid is None:
            return
        self._subscribed.add(cid)
        if self._controller is not None and self._controller.add_subscription is not None:
            try:
                await self._controller.add_subscription(cid)
                await self._reply(
                    update,
                    "🔔 Подписка оформлена и сохранена: уведомления переживут перезапуск.",
                )
                return
            except Exception as exc:
                log.warning("telegram_subscribe_persist_failed", error=str(exc))
        await self._reply(update, "🔔 Подписка оформлена (без БД — до перезапуска).")

    async def _cmd_unsubscribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        cid = self._chat_id(update)
        if cid is None:
            return
        self._subscribed.discard(cid)
        if self._controller is not None and self._controller.remove_subscription is not None:
            try:
                await self._controller.remove_subscription(cid)
            except Exception as exc:
                log.warning("telegram_unsubscribe_persist_failed", error=str(exc))
        await self._reply(
            update,
            "🔕 Подписка отключена: push-уведомления больше не приходят в этот чат.",
        )

    async def _cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        if self._controller is None or self._controller.recent_trades_provider is None:
            await self._reply(update, "История сделок сейчас недоступна.")
            return
        try:
            trades = await self._controller.recent_trades_provider()
        except Exception as exc:
            await self._reply(
                update,
                f"❌ Не удалось получить сделки: <code>{html.escape(str(exc))}</code>",
            )
            return
        if not trades:
            await self._reply(update, "📭 Закрытых сделок пока нет.")
            return
        lines = ["<b>📜 Последние закрытые сделки</b>", ""]
        for t in trades[:10]:
            ts = self._fmt_timestamp(t.get("created_at"))
            pnl = float(t.get("pnl_usdt") or 0)
            pnl_mark = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
            net_bps = t.get("net_bps")
            net_str = f" | {float(net_bps):+.1f} bps" if net_bps is not None else ""
            lines.append(
                f"{pnl_mark} <code>{ts}</code> {html.escape(str(t.get('symbol', '?')))} "
                f"{html.escape(str(t.get('side', '?')))} "
                f"PnL: <code>{pnl:+.4f} USDT</code>{net_str}"
            )
        await self._reply(update, "\n".join(lines), reply_markup=self._main_menu())

    async def _cmd_healthcheck(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        if self._controller is None or self._controller.healthcheck_provider is None:
            await self._reply(update, "Healthcheck сейчас недоступен.")
            return
        try:
            hc = await self._controller.healthcheck_provider()
        except Exception as exc:
            await self._reply(
                update,
                f"❌ Healthcheck не удался: <code>{html.escape(str(exc))}</code>",
            )
            return
        signals = int(hc.get("hour_signals_emitted") or 0)
        placed = int(hc.get("hour_order_placed") or 0)
        top_blocker = str(hc.get("top_blocker") or "нет данных")
        blockers: dict[str, int] = hc.get("blockers") or {}
        avg_net = hc.get("today_avg_net_bps")
        ml_replaced = int(hc.get("hour_ml_replacement") or 0)
        rule_fallback = int(hc.get("hour_rule_fallback_signals") or 0)
        status_icon = "✅" if placed > 0 or signals == 0 else "⚠️"
        lines = [
            "<b>🩺 Healthcheck</b>",
            "",
            f"Сигналов за час: <code>{signals}</code>",
            f"Исполнено сделок за час: <code>{placed}</code> {status_icon}",
            f"ML-замен за час: <code>{ml_replaced}</code> | Rule-fallback: <code>{rule_fallback}</code>",
            "",
            f"Самый частый блокер: <code>{html.escape(top_blocker)}</code>",
        ]
        if blockers:
            lines.append("Блокеры за час:")
            for name, count in sorted(blockers.items(), key=lambda kv: -kv[1]):
                if count > 0:
                    lines.append(f"  • {html.escape(name)}: <code>{count}</code>")
        lines.append("")
        if avg_net is not None:
            lines.append(f"Средний net edge сегодня: <code>{float(avg_net):+.2f} bps</code>")
        else:
            lines.append("Средний net edge сегодня: <code>нет разрешённых исходов</code>")
        if signals > 0 and placed == 0:
            lines.append("")
            lines.append("⚠️ Сигналы есть, сделок нет — посмотрите блокеры выше и /limits.")
        await self._reply(update, "\n".join(lines), reply_markup=self._main_menu())

    async def _cmd_buckets(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._authorised(update):
            return
        if self._controller is None or self._controller.bucket_stats_provider is None:
            await self._reply(update, "Статистика по bucket'ам сейчас недоступна.")
            return
        try:
            data = await self._controller.bucket_stats_provider()
        except Exception as exc:
            await self._reply(
                update,
                f"❌ Не удалось получить bucket-статистику: <code>{html.escape(str(exc))}</code>",
            )
            return
        buckets: list[dict[str, Any]] = data.get("buckets") or []
        refreshed_at = data.get("refreshed_at")
        min_samples = int(data.get("min_samples") or 30)
        block_below = float(data.get("block_below_bps") or -2.0)
        lines = ["<b>📊 Bucket-статистика (режим × волатильность × час UTC)</b>", ""]
        if not buckets:
            lines.append("Пока нет разрешённых исходов — статистика накапливается.")
        else:
            # Worst expectancy first; cap output so the message stays readable
            shown = sorted(buckets, key=lambda b: float(b["avg_bps"]))[:20]
            for b in shown:
                blocked = int(b["count"]) >= min_samples and float(b["avg_bps"]) < block_below
                icon = "🚫" if blocked else "✅"
                lines.append(
                    f"{icon} {html.escape(str(b['regime']))}/{html.escape(str(b['volatility']))} "
                    f"{int(b['hour']):02d}h: <code>{float(b['avg_bps']):+.1f} bps</code> "
                    f"(n={int(b['count'])})"
                )
            if len(buckets) > len(shown):
                lines.append(f"… и ещё {len(buckets) - len(shown)} bucket(ов)")
        lines.append("")
        lines.append(f"Блокировка: n ≥ {min_samples} и avg &lt; {block_below:+.1f} bps")
        if refreshed_at:
            lines.append(f"Обновлено: <code>{html.escape(str(refreshed_at))}</code>")
        await self._reply(update, "\n".join(lines), reply_markup=self._main_menu())

    async def _on_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle free-text input: currently only the custom-limit flow."""
        del context
        if not await self._authorised(update):
            return
        cid = self._chat_id(update)
        if cid is None or cid not in self._awaiting_custom_limit:
            return
        expected_key = self._awaiting_custom_limit.pop(cid)
        if self._controller is None or self._controller.set_runtime_setting is None:
            await self._reply(update, "Runtime-настройки сейчас недоступны.")
            return
        text = (update.effective_message.text or "").strip() if update.effective_message else ""
        if expected_key is None:
            parts = text.split()
            if len(parts) != 2:
                await self._reply(
                    update,
                    "Не понял. Формат: <code>ключ значение</code>, например <code>entries 3</code>.",
                    reply_markup=self._limits_menu(),
                )
                return
            key, raw_value = parts[0].lower(), parts[1]
        else:
            parts = text.split()
            if len(parts) != 1:
                await self._reply(
                    update,
                    "Отправьте только число, например <code>12</code>.",
                    reply_markup=self._limits_menu(),
                )
                return
            key, raw_value = expected_key, parts[0]
        try:
            if key in {"price_cap", "model_gate_threshold"}:
                value: Any = float(raw_value)
            elif key == "model_gate":
                value = raw_value
            else:
                value = int(raw_value)
            msg = await self._controller.set_runtime_setting(key, value)
        except Exception as exc:
            await self._reply(
                update,
                f"❌ Изменение лимита отклонено: <code>{html.escape(str(exc))}</code>",
                reply_markup=self._limits_menu(),
            )
            return
        await self._reply(
            update,
            f"✅ {msg}\n\n{self._limits_text()}",
            reply_markup=self._limits_menu(),
        )

    async def _on_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        query = update.callback_query
        if query is None:
            return
        self._last_callback_at = datetime.now(tz=UTC)
        try:
            await query.answer()
        except Exception as exc:
            # Old buttons from chat history, redeploy races, or Telegram network
            # hiccups should not prevent the actual callback action from running.
            log.debug("telegram.callback_answer_failed_continuing", error=str(exc))
        if not await self._authorised(update):
            return

        data = query.data or ""
        try:
            if data.startswith("confirm:"):
                await self._handle_confirm_button(update, data.removeprefix("confirm:"))
                return
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
            if data == "noop":
                return
            if data.startswith("action:"):
                await self._handle_action_button(update, data.removeprefix("action:"))
                return
            await self._button_reply(update, "Неизвестная кнопка.", reply_markup=self._main_menu())
        except Exception as exc:
            log.warning("telegram.callback_failed", callback_data=data[:80], error=str(exc), exc_info=True)
            await self._button_reply(
                update,
                f"⚠️ Кнопка не выполнилась: <code>{html.escape(str(exc))[:500]}</code>",
                reply_markup=self._main_menu(),
            )

    async def _update_dashboard(
        self,
        query: CallbackQuery,
        text: str,
        reply_markup: InlineKeyboardMarkup,
        parse_mode: str = "HTML",
    ) -> None:
        """Edit the existing dashboard message, or send a new one if editing fails."""
        from telegram.error import BadRequest

        try:
            await query.edit_message_text(
                text=text,
                reply_markup=reply_markup,
                parse_mode=parse_mode,
                disable_web_page_preview=True,
            )
            if isinstance(query.message, Message):
                self._dashboard_message_id = query.message.message_id
                self._dashboard_chat_id = query.message.chat_id
        except (BadRequest, Exception) as exc:
            log.debug("telegram.edit_failed_sending_new", error=str(exc))
            if isinstance(query.message, Message):
                try:
                    msg = await query.message.reply_text(
                        text=text,
                        reply_markup=reply_markup,
                        parse_mode=parse_mode,
                        disable_web_page_preview=True,
                    )
                    self._dashboard_message_id = msg.message_id
                    self._dashboard_chat_id = msg.chat_id
                except Exception as _exc:
                    log.debug("telegram.fallback_send_failed", error=str(_exc))
                    sent_msg = await self._send_direct_message(
                        self._message_chat_id(query.message) or self._dashboard_chat_id,
                        text,
                        reply_markup=reply_markup,
                    )
                    if sent_msg is not None:
                        self._dashboard_message_id = getattr(sent_msg, "message_id", self._dashboard_message_id)
                        self._dashboard_chat_id = getattr(sent_msg, "chat_id", self._dashboard_chat_id)
            else:
                sent_msg = await self._send_direct_message(
                    self._message_chat_id(query.message) or self._dashboard_chat_id,
                    text,
                    reply_markup=reply_markup,
                )
                if sent_msg is not None:
                    self._dashboard_message_id = getattr(sent_msg, "message_id", self._dashboard_message_id)
                    self._dashboard_chat_id = getattr(sent_msg, "chat_id", self._dashboard_chat_id)

    async def _render_home(self) -> tuple[str, InlineKeyboardMarkup]:
        """Render the main dashboard text and keyboard."""
        ctrl = self._controller
        mode = "SHADOW" if (ctrl is not None and ctrl.is_shadow()) else "ACTIVE"
        paused = ctrl.is_paused() if ctrl is not None else False
        risk = ctrl.current_profile() if ctrl is not None else "—"
        positions = 0
        if ctrl and hasattr(ctrl, "exposure") and ctrl.exposure is not None:
            positions = getattr(ctrl.exposure, "position_count", 0)
        symbols_count = len(ctrl.active_symbols()) if ctrl is not None else 0
        entries_per_min: Any = "—"
        max_pos: Any = "—"
        if ctrl and ctrl.runtime_settings:
            try:
                s = ctrl.runtime_settings()
                entries_per_min = s.get("max_entries_per_minute", "—")
                max_pos = s.get("max_positions", "—")
            except Exception as _exc:
                log.debug("telegram.render_home_settings_failed", error=str(_exc))

        text = (
            "🏠 <b>Bybit AI Trader</b>\n"
            f"Режим: <code>{mode}</code> | Риск: <code>{risk}</code> | Пауза: <code>{'да' if paused else 'нет'}</code>\n"
            f"Позиции: <code>{positions}/{max_pos}</code> | Ордеров/мин: <code>{entries_per_min}</code>\n"
            f"Активных монет: <code>{symbols_count}</code>"
        )
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "⏸ Пауза" if not paused else "▶️ Возобновить",
                        callback_data="action:pause" if not paused else "action:resume",
                    ),
                    InlineKeyboardButton("🚦 CANARY", callback_data="action:canary"),
                ],
                [
                    InlineKeyboardButton("📈 Сигналы", callback_data="view:signals"),
                    InlineKeyboardButton("📊 Сделки", callback_data="view:trades"),
                    InlineKeyboardButton("📉 PnL", callback_data="view:pnl"),
                ],
                [
                    InlineKeyboardButton("⚙️ Настройки", callback_data="view:settings"),
                    InlineKeyboardButton("🧠 Модель", callback_data="view:model"),
                    InlineKeyboardButton("❓ Помощь", callback_data="view:help"),
                ],
                [InlineKeyboardButton("📋 Разделы меню", callback_data="view:menu")],
                [InlineKeyboardButton("🔄 Обновить", callback_data="view:home")],
            ]
        )
        return text, keyboard

    async def _render_settings(self) -> tuple[str, InlineKeyboardMarkup]:
        """Render the interactive settings screen."""
        ctrl = self._controller
        s: dict[str, Any] = {}
        if ctrl and ctrl.runtime_settings:
            try:
                s = ctrl.runtime_settings()
            except Exception as _exc:
                log.debug("telegram.render_settings_failed", error=str(_exc))

        entries = s.get("max_entries_per_minute", 4)
        max_pos = s.get("max_positions", 2)
        same_side = s.get("max_same_side", 2)
        price_cap = s.get("screener_max_price_usd", 25)
        feat_sym = s.get("feature_max_symbols", 20)

        text = (
            "⚙️ <b>Настройки</b>\n\n"
            f"📈 Новых ордеров/мин: <code>{entries}</code>\n"
            f"📊 Одновременно позиций: <code>{max_pos}</code>\n"
            f"🔄 Позиций в одну сторону: <code>{same_side}</code>\n"
            f"💰 Потолок цены: <code>{price_cap}</code>\n"
            f"🔍 Изучать монет: <code>{feat_sym}</code>"
        )
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("📈 Ордеров/мин", callback_data="noop"),
                    InlineKeyboardButton("−", callback_data="limit:entries_per_min_limit:dec"),
                    InlineKeyboardButton(str(entries), callback_data="noop"),
                    InlineKeyboardButton("+", callback_data="limit:entries_per_min_limit:inc"),
                ],
                [
                    InlineKeyboardButton("📊 Позиций макс", callback_data="noop"),
                    InlineKeyboardButton("−", callback_data="limit:max_simultaneous_positions:dec"),
                    InlineKeyboardButton(str(max_pos), callback_data="noop"),
                    InlineKeyboardButton("+", callback_data="limit:max_simultaneous_positions:inc"),
                ],
                [
                    InlineKeyboardButton("🔄 Одна сторона", callback_data="noop"),
                    InlineKeyboardButton("−", callback_data="limit:max_same_side:dec"),
                    InlineKeyboardButton(str(same_side), callback_data="noop"),
                    InlineKeyboardButton("+", callback_data="limit:max_same_side:inc"),
                ],
                [
                    InlineKeyboardButton("💰 Потолок цены", callback_data="noop"),
                    InlineKeyboardButton("−", callback_data="limit:screener_max_price_usd:dec"),
                    InlineKeyboardButton(str(price_cap), callback_data="noop"),
                    InlineKeyboardButton("+", callback_data="limit:screener_max_price_usd:inc"),
                ],
                [
                    InlineKeyboardButton("🔍 Монет изучать", callback_data="noop"),
                    InlineKeyboardButton("−", callback_data="limit:feature_max_symbols:dec"),
                    InlineKeyboardButton(str(feat_sym), callback_data="noop"),
                    InlineKeyboardButton("+", callback_data="limit:feature_max_symbols:inc"),
                ],
                [InlineKeyboardButton("✅ Выбрать пары", callback_data="view:symbol_select")],
                [InlineKeyboardButton("🏠 Главная", callback_data="view:home")],
            ]
        )
        return text, keyboard

    async def _render_signals(self) -> tuple[str, InlineKeyboardMarkup]:
        """Render the last signals screen."""
        ctrl = self._controller
        if ctrl is None or not ctrl.signal_log:
            text = "📜 <b>Последние сигналы</b>\n\nНет данных"
        else:
            lines = ["📜 <b>Последние сигналы (10)</b>\n"]
            for sig in list(ctrl.signal_log)[-10:]:
                icon = "🟢" if str(getattr(sig, "side", "")).upper() == "BUY" else "🔴"
                ts_raw = getattr(sig, "timestamp", None)
                ts = ts_raw.strftime("%H:%M:%S") if ts_raw else "—"
                conf_raw = getattr(sig, "confidence", None)
                conf = f"{conf_raw:.2f}" if conf_raw is not None else "—"
                lines.append(f"{icon} {ts} {getattr(sig, 'symbol', '?')} {getattr(sig, 'side', '?')} {conf}")
            text = "\n".join(lines)
        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("🔄 Обновить", callback_data="view:signals")],
                [InlineKeyboardButton("🏠 Главная", callback_data="view:home")],
            ]
        )
        return text, keyboard

    async def _render_trades(self) -> tuple[str, InlineKeyboardMarkup]:
        """Render the last closed trades screen."""
        ctrl = self._controller
        text_parts = ["📊 <b>Последние сделки</b>\n"]
        if ctrl and ctrl.recent_trades_provider:
            try:
                rows = await ctrl.recent_trades_provider()
                if rows:
                    for row in rows[-10:]:
                        sym = row.get("symbol", "?")
                        side = row.get("side", "?")
                        pnl = row.get("closed_pnl", 0)
                        pnl_str = f"{pnl:+.2f}" if pnl is not None else "—"
                        icon = "🟢" if (pnl or 0) >= 0 else "🔴"
                        text_parts.append(f"{icon} {sym} {side} {pnl_str} USD")
                else:
                    text_parts.append("Нет данных")
            except Exception as exc:
                text_parts.append(f"Ошибка: <code>{exc}</code>")
        else:
            text_parts.append("Контроллер недоступен")
        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("🔄 Обновить", callback_data="view:trades")],
                [InlineKeyboardButton("🏠 Главная", callback_data="view:home")],
            ]
        )
        return "\n".join(text_parts), keyboard

    async def _render_pnl(self) -> tuple[str, InlineKeyboardMarkup]:
        """Render the net PnL summary screen."""
        text_parts = ["📉 <b>Чистый PnL</b>\n"]
        if self._net_results_provider:
            try:
                data = await self._net_results_provider()
                gross = data.get("gross_closed_pnl_usd", 0)
                fees = data.get("total_fees_usd", 0)
                net = data.get("net_pnl_usd", gross)
                text_parts.append(f"Gross PnL: <code>{gross:+.2f} USD</code>")
                text_parts.append(f"Комиссии: <code>{fees:.2f} USD</code>")
                if net is not None:
                    text_parts.append(f"Net PnL: <code>{net:+.2f} USD</code>")
            except Exception as exc:
                text_parts.append(f"Ошибка: <code>{exc}</code>")
        else:
            text_parts.append("Провайдер PnL недоступен")
        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("🔄 Обновить", callback_data="view:pnl")],
                [InlineKeyboardButton("🏠 Главная", callback_data="view:home")],
            ]
        )
        return "\n".join(text_parts), keyboard

    async def _render_model(self) -> tuple[str, InlineKeyboardMarkup]:
        """Render the model info screen."""
        ctrl = self._controller
        lines: list[str] = ["🧠 <b>Модель и обучение</b>\n"]
        if ctrl and hasattr(ctrl, "db_diagnostics_provider") and ctrl.db_diagnostics_provider:
            info = await self._load_db_diag(lite=True)
            banner = self._db_diag_banner(info)
            if banner:
                lines.append(banner)
                lines.append("")
            if info.get("error") and info.get("error") != "db_diagnostics_timeout":
                lines.append(f"Ошибка: <code>{html.escape(str(info['error']))}</code>")
            else:
                latest = info.get("latest_model_version") or info.get("active_model_version") or {}
                if not latest.get("version") and self._controller and self._controller.diagnostics_provider:
                    try:
                        runtime = self._controller.diagnostics_provider()
                        model_info = runtime.get("model") or {}
                        challenger_ver = model_info.get("challenger_version")
                        if challenger_ver not in (None, "", "none"):
                            latest = {
                                "version": challenger_ver,
                                "status": "SHADOW_CHALLENGER",
                                "training_samples": model_info.get("training_samples", 0),
                                "metrics": {
                                    "quality": model_info.get("quality"),
                                    "lift_bps": model_info.get("lift_bps"),
                                    "walk_forward_expectancy_bps": model_info.get("walk_forward_expectancy"),
                                },
                            }
                    except Exception as exc:
                        log.debug("telegram.model_runtime_fallback_failed", error=str(exc))

                metrics: dict[str, Any] = latest.get("metrics") or {}
                if isinstance(metrics, str):
                    try:
                        metrics = json.loads(metrics) or {}
                    except Exception:
                        metrics = {}
                model_horizon, gate = self._model_horizon_and_gate(info, metrics)
                training_by_horizon = info.get("training_eligible_by_horizon", {}) or {}
                eligible = int(training_by_horizon.get(str(model_horizon), info.get("training_eligible_15m", 0)) or 0)

                version = html.escape(str(latest.get("version") or "нет"))
                status = html.escape(self._ru(str(latest.get("status") or "—")))
                quality = html.escape(self._ru(str(metrics.get("quality") or "—")))
                lift = metrics.get("lift_bps")
                precision = metrics.get("precision")
                best_thresh = metrics.get("best_threshold")
                samples = int(latest.get("training_samples") or 0)
                actual_samples = int(latest.get("actual_training_samples", samples) or samples)
                compatible_samples = int(latest.get("training_samples_compatible", 0) or 0)
                schema_ok = bool(latest.get("schema_compatible", False))

                lift_str = f"{float(lift):+.1f} bps" if lift is not None else "нет"
                prec_str = f"{float(precision):.1%}" if precision is not None else "нет"
                thresh_str = str(best_thresh) if best_thresh is not None else "нет"

                gate_lift = gate.get("lift_vs_all_bps")
                gate_total = int(gate.get("total_count") or 0)
                gate_pass = int(gate.get("pass_count") or 0)
                gate_lift_str = f"{float(gate_lift):+.1f} bps" if gate_lift is not None else "нет"

                lines += [
                    f"Версия: <code>{version}</code>",
                    f"Статус: <code>{status}</code>  Качество: <code>{quality}</code>",
                    f"Lift (val-split): <code>{lift_str}</code>",
                    f"Precision: <code>{prec_str}</code>  Порог: <code>{thresh_str}</code>",
                    f"Обучено: <code>{samples}</code> образцов"
                    + (
                        f" (актуально для schema: <code>{compatible_samples}</code>)"
                        if not schema_ok and actual_samples
                        else ""
                    ),
                    f"Схема совместима: <code>{'да' if schema_ok else 'нет — ждём переобучение'}</code>",
                ]
                if not schema_ok and actual_samples:
                    lines.append(
                        f"⚠️ Последний запуск на старой schema: <code>{actual_samples}</code> образцов "
                        f"(<code>{version}</code>)"
                    )
                if gate_total or gate_pass or gate_lift is not None:
                    lines += [
                        f"<b>Shadow gate ({model_horizon}m):</b>",
                        f"Всего решений: <code>{gate_total}</code>  Pass: <code>{gate_pass}</code>",
                        f"Gate lift: <code>{gate_lift_str}</code>",
                        "",
                    ]
                elif info.get("lite"):
                    lines.append("Shadow gate: <code>загружается в фоне</code>")
                    lines.append("")
                lines += [
                    (
                        f"Данных для обучения ({model_horizon}m): <code>не менее {eligible}</code> (быстрая оценка)"
                        if info.get("lite")
                        else f"Данных для обучения ({model_horizon}m): <code>{eligible}</code>"
                    ),
                ]
        else:
            lines.append("Данные недоступны")
        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("📉 История моделей", callback_data="view:model_performance")],
                [InlineKeyboardButton("🏆 Champion health", callback_data="view:champion_health")],
                [InlineKeyboardButton("🗄 База и модель", callback_data="view:db_model")],
                [InlineKeyboardButton("🔄 Обновить", callback_data="view:model")],
                [InlineKeyboardButton("🏠 Главная", callback_data="view:home")],
            ]
        )
        return "\n".join(lines), keyboard

    async def _render_champion_health(self) -> tuple[str, InlineKeyboardMarkup]:
        ctrl = self._controller
        if ctrl is None or ctrl.champion_health_provider is None:
            return (
                "🏆 <b>Champion health</b>\n\nДанные недоступны.",
                InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Главная", callback_data="view:home")]]),
            )
        try:
            data = await ctrl.champion_health_provider()
        except Exception as exc:
            return (
                f"🏆 <b>Champion health</b>\n\nОшибка: <code>{html.escape(str(exc))}</code>",
                InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Главная", callback_data="view:home")]]),
            )
        champion = data.get("champion") or {}
        alt = data.get("best_alternative") or {}
        checks = data.get("checks") or []
        logs = data.get("promotion_log") or []

        def _fmt_bps(value: Any) -> str:
            return "n/a" if value is None else f"{float(value):+.2f} bps"

        def _model_lines(title: str, row: dict[str, Any]) -> list[str]:
            if not row:
                return [f"<b>{title}</b>: <code>нет</code>"]
            return [
                f"<b>{title}</b>: <code>{html.escape(str(row.get('version') or 'unknown'))}</code>",
                f"status=<code>{html.escape(str(row.get('status') or 'n/a'))}</code> "
                f"quality=<code>{html.escape(str(row.get('quality') or 'n/a'))}</code>",
                f"score=<code>{float(row.get('model_score') or 0.0):+.2f}</code> "
                f"wf=<code>{_fmt_bps(row.get('walk_forward_bps'))}</code> "
                f"lift=<code>{_fmt_bps(row.get('lift_bps'))}</code>",
                f"paper=<code>{int(row.get('paper_gate_count') or 0)}</code> "
                f"folds=<code>{row.get('wf_positive_folds') or 0}/{row.get('wf_folds') or 0}</code> "
                f"std=<code>{_fmt_bps(row.get('wf_std_bps'))}</code>",
                f"reason=<code>{html.escape(str(row.get('selection_reason') or 'n/a'))}</code>",
            ]

        lines = ["🏆 <b>Champion health</b>", ""]
        lines.extend(_model_lines("Champion", champion))
        lines.append("")
        lines.extend(_model_lines("Best alternative", alt))
        lines.append("\n<b>Checks</b>")
        if checks:
            for check in checks:
                mark = "OK" if check.get("ok") else "FAIL"
                lines.append(
                    f"{mark} <code>{html.escape(str(check.get('name') or 'check'))}</code>: "
                    f"<code>{html.escape(str(check.get('value') or 'n/a'))}</code>"
                )
        else:
            lines.append("Нет проверок: champion не найден.")

        if logs:
            lines.append("\n<b>Last promotion events</b>")
            for row in logs[:3]:
                reasons = row.get("reasons")
                if isinstance(reasons, str):
                    try:
                        reasons = json.loads(reasons)
                    except json.JSONDecodeError:
                        reasons = [reasons]
                reason_text = ", ".join(str(item) for item in (reasons or [])[:2])
                lines.append(
                    f"<code>{html.escape(str(row.get('event_type') or 'event'))}</code> "
                    f"{html.escape(str(row.get('from_version') or 'none'))} -> "
                    f"{html.escape(str(row.get('to_version') or 'none'))} "
                    f"<code>{html.escape(reason_text[:80])}</code>"
                )

        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("🔄 Обновить", callback_data="view:champion_health")],
                [InlineKeyboardButton("📉 История моделей", callback_data="view:model_performance")],
                [InlineKeyboardButton("🏠 Главная", callback_data="view:home")],
            ]
        )
        return "\n".join(lines), keyboard

    async def _render_help(self) -> tuple[str, InlineKeyboardMarkup]:
        """Render the help screen."""
        text = self._help_text()
        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("🏠 Главная", callback_data="view:home")],
            ]
        )
        return text, keyboard

    async def _handle_action_button(self, update: Update, action: str) -> None:
        """Handle action: prefix callbacks from the dashboard."""
        ctrl = self._controller
        if action in ("pause", "resume"):
            if ctrl is None:
                await self._button_reply(update, "Управление недоступно.", reply_markup=self._main_menu())
                return
            if action == "pause":
                await ctrl.pause()
            else:
                await ctrl.resume()
            text, markup = await self._render_home()
            await self._button_reply(update, text, reply_markup=markup)
            return
        if action == "canary":
            fake_context = type("_Context", (), {"args": []})()
            await self._cmd_canary_ready(update, fake_context)
            return
        await self._button_reply(update, f"Неизвестное действие: {action}", reply_markup=self._main_menu())

    async def _handle_limit_adjust(self, query: CallbackQuery, data: str) -> None:
        """Handle limit +/- buttons from the new settings dashboard."""
        try:
            await query.answer()
        except Exception as exc:
            log.debug("telegram.limit_adjust_answer_failed", error=str(exc))
        parts = data.split(":")
        if len(parts) != 2:
            return
        setting_key, direction = parts
        if direction not in {"inc", "dec"}:
            return
        s: dict[str, Any] = {}
        if self._controller and self._controller.runtime_settings:
            try:
                s = self._controller.runtime_settings()
            except Exception as _exc:
                log.debug("telegram.limit_adjust_settings_failed", error=str(_exc))
        # Maps button key → (runtime_settings read key, set_runtime_setting key, min, max)
        limit_map: dict[str, tuple[str, str, float, float]] = {
            "entries_per_min_limit": ("max_entries_per_minute", "entries", 1, 10),
            "max_simultaneous_positions": ("max_positions", "max_positions", 1, 10),
            "max_same_side": ("max_same_side", "same_side", 1, 10),
            "screener_max_price_usd": ("screener_max_price_usd", "price_cap", 0, 1000),
            "feature_max_symbols": ("feature_max_symbols", "feature_symbols", 5, 50),
        }
        if setting_key not in limit_map:
            return
        read_key, write_key, min_val, max_val = limit_map[setting_key]
        is_float = setting_key == "screener_max_price_usd"
        current = float(s.get(read_key, min_val)) if is_float else int(s.get(read_key, min_val))
        step = 5.0 if setting_key == "screener_max_price_usd" else 1
        new_val: Any = current + (step if direction == "inc" else -step)
        new_val = max(min_val, min(max_val, new_val))
        if self._controller and self._controller.set_runtime_setting and new_val != current:
            try:
                await self._controller.set_runtime_setting(write_key, new_val)
            except Exception as exc:
                log.warning("telegram.limit_adjust_failed", error=str(exc))
                if query.message:
                    message = cast(Any, query.message)
                    await message.reply_text(
                        f"❌ Изменение лимита отклонено: <code>{html.escape(str(exc))}</code>",
                        parse_mode=ParseMode.HTML,
                        disable_web_page_preview=True,
                        reply_markup=self._limits_menu(),
                    )
                return
        text, markup = await self._render_settings()
        # Re-use _update_dashboard via the query object directly
        from telegram.error import BadRequest

        try:
            await query.edit_message_text(
                text=text,
                reply_markup=markup,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
        except (BadRequest, Exception) as exc:
            log.debug("telegram.settings_edit_failed", error=str(exc))
            if isinstance(query.message, Message):
                try:
                    await query.message.reply_text(
                        text=text,
                        reply_markup=markup,
                        parse_mode=ParseMode.HTML,
                        disable_web_page_preview=True,
                    )
                except Exception as _exc2:
                    log.debug("telegram.fallback_send_view_failed", error=str(_exc2))

    async def _handle_view_button(self, update: Update, action: str) -> None:
        fake_context = type("_Context", (), {"args": []})()
        handlers = {
            "status": self._cmd_status,
            "balance": self._cmd_balance,
            "positions": self._cmd_positions,
            "signals": self._cmd_signals,
            "symbols": self._cmd_symbols,
            "pnl": self._cmd_pnl,
            "load_diagnostics": self._cmd_diagnostics,
            "costs": self._cmd_costs,
            "costs_detailed": self._cmd_costs_detailed,
            "pnl_analysis": self._cmd_pnl_analysis,
            "compare": self._cmd_compare,
            "strategy_report": self._cmd_strategy_report,
            "model_performance": self._cmd_model_performance,
            "champion_health": self._cmd_champion_health,
            "deep_report": self._cmd_deep_report,
        }
        if action == "home":
            text, markup = await self._render_home()
            await self._button_reply(update, text, reply_markup=markup)
            return
        if action == "menu":
            await self._button_reply(update, self._menu_text(), reply_markup=self._main_menu())
            return
        if action == "status_mode":
            text = f"<b>🏠 Статус и режим</b>\n{self._mode_indicator()}\n\nРежим, пауза и риск-профиль собраны здесь."
            await self._button_reply(update, text, reply_markup=self._status_mode_menu())
            return
        if action == "balance_positions":
            await self._button_reply(
                update,
                "<b>💰 Баланс и позиции</b>\n\nФинансовое состояние аккаунта и открытые позиции.",
                reply_markup=self._balance_positions_menu(),
            )
            return
        if action == "trading":
            await self._button_reply(
                update,
                "<b>📈 Торговля и сигналы</b>\n\nСигналы, сделки, PnL и готовность к CANARY.",
                reply_markup=self._trading_menu(),
            )
            return
        if action == "settings":
            text, markup = await self._render_settings()
            await self._button_reply(update, text, reply_markup=markup)
            return
        if action == "model":
            text, markup = await self._render_model()
            await self._button_reply(update, text, reply_markup=markup)
            return
        if action == "champion_health":
            text, markup = await self._render_champion_health()
            await self._button_reply(update, text, reply_markup=markup)
            return
        if action == "diagnostics":
            await self._button_reply(
                update,
                "<b>🩺 Диагностика</b>\n\nHealthcheck, издержки и разбор убыточности.",
                reply_markup=self._diagnostics_menu(),
            )
            return
        if action == "risk_profiles":
            await self._button_reply(
                update,
                "<b>🎚 Риск-профиль</b>\n\nВыберите профиль. Повышение риска защищено настройкой Render.",
                reply_markup=self._risk_menu(),
            )
            return
        if action == "help":
            text, markup = await self._render_help()
            await self._button_reply(update, text, reply_markup=markup)
            return
        if action == "signals":
            text, markup = await self._render_signals()
            await self._button_reply(update, text, reply_markup=markup)
            return
        if action == "trades":
            text, markup = await self._render_trades()
            await self._button_reply(update, text, reply_markup=markup)
            return
        if action == "healthcheck":
            await self._cmd_healthcheck(update, fake_context)
            return
        if action == "worst":
            fake_context.args = ["10"]
            await self._cmd_worst(update, fake_context)
            return
        if action == "db_model":
            await self._cmd_db_model(update, fake_context)
            return
        if action == "canary":
            await self._cmd_canary_ready(update, fake_context)
            return
        if action == "priorities":
            await self._cmd_priorities(update, fake_context)
            return
        if action == "canary_model":
            await self._show_canary_model_metrics(update)
            return
        if action == "model_help":
            await self._cmd_model_help(update, fake_context)
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
        if action == "pnl":
            text, markup = await self._render_pnl()
            await self._button_reply(update, text, reply_markup=markup)
            return
        handler = handlers.get(action)
        if handler is None:
            await self._button_reply(update, "Неизвестный экран.", reply_markup=self._main_menu())
            return
        await handler(update, fake_context)

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
            await self._button_reply(
                update,
                "Неизвестное действие выбора пар.",
                reply_markup=self._main_menu(),
            )
            return
        if parts[0] == "page":
            page = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            await self._show_symbol_select(update, page=page)
            return
        if parts[0] == "toggle" and len(parts) >= 2:
            symbol = parts[1].upper()
            page = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            if self._controller is None or self._controller.toggle_symbol is None:
                await self._button_reply(
                    update,
                    "Выбор пар сейчас недоступен.",
                    reply_markup=self._main_menu(),
                )
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
            await self._button_reply(
                update,
                "⏸ Поставить бота на <b>паузу</b>? Новые входы открываться не будут.",
                reply_markup=self._confirm_menu("pause"),
            )
            return
        if action == "resume":
            if not self._controller.is_paused():
                await self._button_reply(update, "Бот не на паузе.", reply_markup=self._main_menu())
                return
            await self._button_reply(
                update,
                "▶️ <b>Возобновить</b> работу бота?",
                reply_markup=self._confirm_menu("resume"),
            )
            return
        if action == "promote":
            if self._controller.promote_model is None:
                await self._button_reply(update, "Промоут сейчас недоступен.", reply_markup=self._main_menu())
                return
            version = ""
            if self._controller.best_challenger_provider is not None:
                try:
                    version = str(await self._controller.best_challenger_provider() or "")
                except Exception as exc:
                    log.warning("telegram.promote.best_challenger_failed", error=str(exc))
            if not version:
                db_diag = await self._load_db_diag(lite=True)
                if db_diag.get("error"):
                    log.warning("telegram.promote.db_diag_failed", error=str(db_diag.get("error")))
                latest_model = db_diag.get("latest_model_version", {}) or {}
                version = str(latest_model.get("version") or "")
            if not version:
                await self._button_reply(
                    update,
                    "Нет модели-кандидата для промоута.",
                    reply_markup=self._main_menu(),
                )
                return
            await self._button_reply(
                update,
                "🏆 <b>Промоутировать модель в CHAMPION?</b>\n\n"
                f"Версия: <code>{html.escape(str(version))}</code>\n"
                "Выбрана лучшая eligible challenger-модель (не просто последняя).\n"
                "Backend повторно проверит качество, schema, lift и shadow-gate статистику.",
                reply_markup=self._confirm_menu(f"promote:{version}"),
            )
            return
        if action == "train":
            if self._controller.start_training is None:
                await self._button_reply(
                    update,
                    "Запуск обучения сейчас недоступен.",
                    reply_markup=self._main_menu(),
                )
                return
            await self._button_reply(
                update,
                "🧠 <b>Запустить обучение?</b>\n\n"
                f"Примеры: <code>500</code>, горизонт: <code>{self._train_defaults()[1]}m</code>, "
                f"порог: <code>{self._train_defaults()[2]:g} bps</code>.",
                reply_markup=self._confirm_menu(self._train_callback(500)),
            )
            return
        if action == "limits":
            await self._button_reply(update, self._limits_text(), reply_markup=self._limits_menu())
            return
        if action == "stop":
            await self._button_reply(
                update,
                "🚨 <b>Аварийная остановка</b>\n\n"
                "Все новые входы будут полностью остановлены. "
                "Для возобновления потребуется ручной перезапуск сервиса.\n\n"
                "Выполнить?",
                reply_markup=self._confirm_menu("stop"),
            )
            return
        await self._button_reply(update, "Неизвестное действие управления.", reply_markup=self._main_menu())

    async def _handle_train_button(self, update: Update, payload: str) -> None:
        if self._controller is None:
            await self._button_reply(
                update,
                "Запуск обучения сейчас недоступен.",
                reply_markup=self._main_menu(),
            )
            return
        if payload == "all":
            if self._controller.start_training_all is None:
                await self._button_reply(
                    update,
                    "Обучение ВСЕ сейчас недоступно.",
                    reply_markup=self._main_menu(),
                )
                return
            await self._button_reply(
                update,
                "🧠 <b>Запустить обучение по всем горизонтам?</b>\n\n"
                "Это длительная операция и изменит кандидатов модели.",
                reply_markup=self._confirm_menu("train_all"),
            )
            return
        if self._controller.start_training is None:
            await self._button_reply(
                update,
                "Запуск обучения сейчас недоступен.",
                reply_markup=self._main_menu(),
            )
            return
        try:
            min_s_raw, horizon_raw, label_raw = payload.split(":", maxsplit=2)
            min_samples = int(min_s_raw)
            horizon = int(horizon_raw)
            label_bps = float(label_raw)
            msg = (
                "🧠 <b>Запустить обучение?</b>\n\n"
                f"Примеры: <code>{min_samples}</code>, горизонт: <code>{horizon}m</code>, "
                f"порог: <code>{label_bps:g} bps</code>."
            )
            markup = self._confirm_menu(f"train:{min_samples}:{horizon}:{label_bps:g}")
        except Exception as exc:
            msg = f"❌ Обучение не стартовало: <code>{exc}</code>"
            markup = self._main_menu()
        await self._button_reply(update, msg, reply_markup=markup)

    async def _handle_limit_button(self, update: Update, payload: str) -> None:
        if self._controller is None or self._controller.set_runtime_setting is None:
            await self._button_reply(
                update,
                "Runtime-настройки сейчас недоступны.",
                reply_markup=self._main_menu(),
            )
            return
        if payload == "custom" or payload.startswith("custom:"):
            cid = self._chat_id(update)
            requested_key = payload.split(":", maxsplit=1)[1] if ":" in payload else None
            if cid is not None:
                self._awaiting_custom_limit[cid] = requested_key
            if requested_key == "feature_symbols":
                text = (
                    "✏️ <b>Сколько монет изучать?</b>\n\n"
                    "Отправьте одно число. Бот сам возьмёт топ монет по сканеру.\n"
                    "Например: <code>12</code> или <code>30</code>."
                )
            elif requested_key == "max_positions":
                text = (
                    "✏️ <b>Сколько сделок держать одновременно?</b>\n\n"
                    "Отправьте одно число. Это ограничит максимум открытых позиций.\n"
                    "Например: <code>1</code>, <code>2</code> или <code>4</code>."
                )
            else:
                text = (
                    "✏️ <b>Своё значение лимита</b>\n\n"
                    "Отправьте сообщение в формате: <code>ключ значение</code>\n"
                    "Например: <code>entries 3</code>, <code>price_cap 15.5</code>, "
                    "<code>model_gate_threshold 0.6</code>\n\n"
                    "Доступные ключи: entries, pending, same_side, max_positions, price_cap, "
                    "feature_symbols, exec_candidates, model_gate, model_gate_threshold"
                )
            await self._button_reply(update, text, reply_markup=self._limits_menu())
            return
        # Handle new +/- increment/decrement buttons from _render_settings
        if payload.endswith(":inc") or payload.endswith(":dec"):
            query = update.callback_query
            if query is not None:
                await self._handle_limit_adjust(query, payload)
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
        await self._button_reply(
            update,
            f"✅ {msg}\n\n{self._limits_text()}",
            reply_markup=self._limits_menu(),
        )

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
        controller = self._controller
        if cid and controller is not None:
            self._pending[cid] = (
                f"сменить риск-профиль с {old_profile} на {new_profile_str}",
                lambda: controller.set_risk_profile(new_profile),
                datetime.now(tz=UTC),
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
                "/priorities — что важнее чего: безопасность, фильтры, CANARY, стратегии\n"
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
            "/menu        — главное меню с кнопками\n"
            "/status      — здоровье системы\n"
            "/balance     — баланс кошелька\n"
            "/positions   — открытые позиции\n"
            "/signals     — последние сигналы стратегии\n"
            "/regime      — режим рынка по монетам\n"
            "/symbols     — активные монеты\n"
            "/start       — меню, включая выбор торговых пар\n"
            "/pnl         — история закрытого PnL\n"
            "/net         — чистый PnL с комиссиями и фандингом\n"
            "/strategy_report — единый отчет по качеству стратегии\n"
            "/report      — короткий alias для /strategy_report\n"
            "/pnl_analysis /compare /worst [N] /costs_detailed /model_performance — drill-down диагностика\n"
            "/trades      — последние 10 закрытых сделок\n"
            "/healthcheck — сигналы/сделки за час и главный блокер\n"
            "/buckets     — экспектанси по режимам/часам и блокировки\n"
            "/deep_report — полная сводка одним .txt файлом\n"
            "/deep_report_text — та же сводка сообщениями (может быть длинной)\n"
            "/subscribe   — подписаться на уведомления (хранится в БД)\n"
            "/unsubscribe — отписаться от уведомлений\n"
            "/diagnostics — счетчики и задержки циклов\n"
            "/attribution — PnL по символам за 7 дней\n"
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
            f"Сделок одновременно: <code>{s.get('max_positions', 'n/a')}</code> — максимум открытых позиций\n"
            f"Одновременно pending: <code>{s.get('max_concurrent_pending', 'n/a')}</code> — сколько заявок может висеть\n"
            f"Позиций в одну сторону: <code>{s.get('max_same_side', 'n/a')}</code> — защита от перекоса Long/Short\n"
            f"Потолок цены монеты: <code>{s.get('screener_max_price_usd', 'n/a')}</code> — отсеивает дорогие монеты\n"
            f"Изучаемых монет: <code>{s.get('feature_max_symbols', 'n/a')}</code> — топ монет, где считаются признаки\n"
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
