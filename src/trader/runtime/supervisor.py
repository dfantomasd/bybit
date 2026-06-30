"""Runtime supervisor: critical task health and system heartbeat."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from trader.modules.base import AppBoundModule
from trader.monitoring.logging import get_logger
from trader.runtime.constants import CRITICAL_TASK_NAMES, SUPERVISOR_CHECK_INTERVAL, SUPERVISOR_HEARTBEAT_INTERVAL

log = get_logger(__name__)

_CRITICAL_TASK_NAMES = CRITICAL_TASK_NAMES
_SUPERVISOR_CHECK_INTERVAL = SUPERVISOR_CHECK_INTERVAL
_SUPERVISOR_HEARTBEAT_INTERVAL = SUPERVISOR_HEARTBEAT_INTERVAL


class RuntimeSupervisor(AppBoundModule):
    async def run(self) -> None:
        """Monitor critical background tasks; on unexpected exit alert + exit(1)."""
        last_heartbeat = datetime.now(tz=UTC)
        while not self._app._shutdown_event.is_set():
            now = datetime.now(tz=UTC)

            if (now - last_heartbeat).total_seconds() >= _SUPERVISOR_HEARTBEAT_INTERVAL:
                alive = [t.get_name() for t in self._app._background_tasks if not t.done()]
                log.info("runtime_supervisor.heartbeat", alive_tasks=alive)
                # Structured system heartbeat for observability
                try:
                    pending_diag = (
                        self._app._execution_engine.pending_entry_diagnostics()
                        if self._app._execution_engine is not None
                        else {}
                    )
                    ws_age: float | None = None
                    if (
                        self._app._health_checker is not None
                        and self._app._health_checker._last_ws_message_at is not None
                    ):
                        ws_age = (now - self._app._health_checker._last_ws_message_at).total_seconds()
                    feat_age: float | None = None
                    if self._app._health_checker is not None and hasattr(
                        self._app._health_checker, "_last_feature_computed_at"
                    ):
                        fat = self._app._health_checker._last_feature_computed_at
                        if fat is not None:
                            feat_age = (now - fat).total_seconds()
                    if self._app._telegram_bot is not None and hasattr(
                        self._app._telegram_bot, "ensure_polling_running"
                    ):
                        try:
                            await self._app._telegram_bot.ensure_polling_running()
                        except Exception as tg_watch_exc:
                            log.debug("supervisor.telegram_watchdog_failed", error=str(tg_watch_exc))
                    telegram_health: dict[str, Any] = {}
                    if self._app._telegram_bot is not None and hasattr(self._app._telegram_bot, "health_snapshot"):
                        try:
                            telegram_health = self._app._telegram_bot.health_snapshot()
                        except Exception as tg_exc:
                            telegram_health = {"error": str(tg_exc)}
                    from trader.monitoring.deploy_info import get_deploy_info

                    deploy = get_deploy_info()
                    journal_connected = self._app._trade_journal is not None and self._app._trade_journal.is_enabled
                    log.info(
                        "system.heartbeat",
                        deploy_id=deploy.get("deploy_id") or None,
                        git_commit=deploy.get("git_commit") or None,
                        status=(
                            self._app._status.value if hasattr(self._app._status, "value") else str(self._app._status)
                        ),
                        trading_mode=(
                            self._app._settings.TRADING_MODE.value
                            if self._app._settings is not None and hasattr(self._app._settings.TRADING_MODE, "value")
                            else "unknown"
                        ),
                        shadow_mode=(
                            self._app._execution_engine._shadow_mode
                            if self._app._execution_engine is not None
                            else True
                        ),
                        last_strategy_loop_at=(
                            self._app._last_strategy_loop_at.isoformat()
                            if self._app._last_strategy_loop_at is not None
                            else None
                        ),
                        last_ws_message_age_s=round(ws_age, 1) if ws_age is not None else None,
                        last_feature_age_s=round(feat_age, 1) if feat_age is not None else None,
                        active_symbols=self._app._active_symbols()[:10],
                        pending_entry_count=pending_diag.get("pending_entry_count", 0),
                        pending_entry_ids=pending_diag.get("pending_entry_ids", []),
                        open_positions=(
                            list(self._app._execution_engine._open_positions.keys())
                            if self._app._execution_engine is not None
                            else []
                        ),
                        model_version=(
                            self._app._model_registry.champion.version
                            if self._app._model_registry is not None and self._app._model_registry.champion is not None
                            else (
                                f"challenger:{self._app._model_registry.challenger.version}"
                                if self._app._model_registry is not None
                                and self._app._model_registry.challenger is not None
                                else "none"
                            )
                        ),
                        paused=self._app._trading_paused,
                        execution_candidates=(
                            len(self._app._screener.execution_candidates) if self._app._screener is not None else None
                        ),
                        last_inference_age_s=(
                            round(
                                (now - self._app._model_gate_quality_checked_at).total_seconds(),
                                1,
                            )
                            if self._app._model_gate_quality_checked_at is not None
                            else None
                        ),
                        model_gate_quality=(
                            self._app._model_gate_quality.get("quality") if self._app._model_gate_quality else None
                        ),
                        telegram_polling=telegram_health.get("polling_running"),
                        telegram_webhook=telegram_health.get("webhook_active"),
                        telegram_delivery_mode=telegram_health.get("delivery_mode"),
                        telegram_conflicts=telegram_health.get("polling_conflict_count"),
                        trade_journal_connected=journal_connected,
                    )
                except Exception as _hb_exc:
                    log.debug("supervisor.heartbeat_failed", error=str(_hb_exc))
                last_heartbeat = now

            for task in list(self._app._background_tasks):
                if not task.done():
                    continue
                name = task.get_name()
                if name not in _CRITICAL_TASK_NAMES:
                    continue
                if self._app._shutdown_event.is_set():
                    return

                exc = task.exception() if not task.cancelled() else None
                log.critical(
                    "runtime_supervisor.critical_task_died",
                    task=name,
                    error=str(exc),
                )
                if self._app._telegram_bot is not None:
                    try:
                        await self._app._telegram_bot.notify(
                            f"🚨 <b>Critical task died</b>: <code>{name}</code>\n"
                            f"Error: <code>{exc}</code>\n"
                            "Container will restart automatically."
                        )
                    except Exception as notify_exc:  # noqa: BLE001
                        log.warning("supervisor.telegram_notify_failed", error=str(notify_exc))
                # Signal the main loop to exit via graceful_shutdown rather than
                # calling sys.exit() directly, which would bypass journal flushing
                # and adapter teardown.
                self._app._shutdown_event.set()
                return

            try:
                await asyncio.wait_for(
                    self._app._shutdown_event.wait(),
                    timeout=_SUPERVISOR_CHECK_INTERVAL,
                )
            except TimeoutError:
                pass
