"""Telegram operator bridge: bot startup and provider wiring."""

from __future__ import annotations

import asyncio
import html
import os
import time
from datetime import UTC, datetime
from typing import Any, cast

from trader.modules.base import AppBoundModule
from trader.monitoring.logging import get_logger
from trader.runtime.constants import _WS_INTERVAL

log = get_logger(__name__)


class TelegramBridgeModule(AppBoundModule):
    name = "telegram"

    def resolve_delivery(self) -> tuple[str, str]:
        """Resolve Telegram delivery mode and webhook URL for the current environment."""
        assert self._app._settings is not None
        mode = self._app._settings.TELEGRAM_DELIVERY_MODE.strip().lower()
        webhook_url = self._app._settings.TELEGRAM_WEBHOOK_URL.strip().rstrip("/")
        if not webhook_url:
            render_url = os.getenv("RENDER_EXTERNAL_URL", "").strip().rstrip("/")
            if render_url:
                webhook_url = f"{render_url}/telegram/webhook"
        if mode == "auto":
            mode = "webhook" if webhook_url else "polling"
        if mode == "webhook" and not webhook_url:
            log.warning("telegram_webhook_url_missing_fallback_polling")
            mode = "polling"
        if mode not in {"polling", "webhook"}:
            log.warning("telegram_delivery_mode_unknown_fallback_polling", mode=mode)
            mode = "polling"
        return mode, webhook_url

    async def start(self) -> None:
        from trader.telegram_bot import (
            TelegramBotConfig,
            TelegramMonitorBot,
            TradingController,
        )

        assert self._app._settings is not None
        assert self._app._health_checker is not None
        token = self._app._settings.TELEGRAM_BOT_TOKEN.get_secret_value()
        if not token:
            log.info("telegram_bot_skipped", reason="no token configured")
            return

        def _regime_for(symbol: str) -> str | None:
            if self._app._feature_pipeline is None or self._app._regime_classifier is None:
                return None
            vec = self._app._feature_pipeline.latest(symbol, _WS_INTERVAL)
            if vec is None:
                return None
            try:
                ctx = self._app._regime_classifier.classify(vec)
                return cast(str, ctx.regime.value)
            except Exception:
                return None

        async def _db_diagnostics_provider(*, lite: bool = False) -> dict[str, Any]:
            if self._app._trade_journal is None:
                return {
                    "connected": False,
                    "configured": False,
                    "error": "trade_journal_not_started",
                    "lite": lite,
                }
            if not self._app._trade_journal.is_enabled:
                await self._app._trade_journal.reconnect_if_needed()
            now = time.monotonic()
            cache_ttl = 60.0 if lite else 15.0
            cache = self._app._db_diagnostics_lite_cache if lite else self._app._db_diagnostics_cache
            cache_at = self._app._db_diagnostics_lite_cache_at if lite else self._app._db_diagnostics_cache_at
            if cache is not None and (now - cache_at) < cache_ttl:
                cached_copy = dict(cache)
                self._app._modules.diagnostics.merge_db_fallbacks(cached_copy)
                return cached_copy

            async def _timeout_fallback() -> dict[str, Any]:
                diag: dict[str, Any] = {
                    "connected": self._app._trade_journal.is_enabled,
                    "configured": self._app._trade_journal.is_configured,
                    "error": "db_diagnostics_timeout",
                    "schema_degraded": bool(
                        getattr(self._app._trade_journal, "_last_connect_error", None)
                        and "schema bootstrap degraded"
                        in str(getattr(self._app._trade_journal, "_last_connect_error", "")).lower()
                    ),
                    "lite": lite,
                }
                if not lite:
                    try:
                        quick = await asyncio.wait_for(
                            self._app._trade_journal.get_db_diagnostics(lite=True),
                            timeout=10.0,
                        )
                        diag.update(quick)
                        diag["error"] = "db_diagnostics_timeout"
                        diag["full_diagnostics_timeout"] = True
                    except Exception as exc:
                        log.debug("db_diagnostics_quick_fallback_failed", error=str(exc))
                    active_model = diag.get("active_model_version") or diag.get("latest_model_version") or {}
                    active_version = str(active_model.get("version") or "") if isinstance(active_model, dict) else ""
                    if not active_version:
                        latest_model = diag.get("latest_model_version") or {}
                        active_version = (
                            str(latest_model.get("version") or "") if isinstance(latest_model, dict) else ""
                        )
                    if active_version:
                        gate: dict[str, Any] = {"model_version": active_version}
                        metrics = active_model.get("metrics") if isinstance(active_model, dict) else {}
                        if isinstance(metrics, str):
                            import json as _json

                            metrics = _json.loads(metrics)
                        horizon = int(
                            diag.get("model_gate_horizon_minutes")
                            or (metrics or {}).get("horizon_minutes")
                            or getattr(self._app._settings, "MODEL_AUTO_TRAIN_HORIZON_MINUTES", 5)
                            or 5
                        )
                        label_schema = str(
                            diag.get("label_schema_version")
                            or getattr(self._app._settings, "LABEL_SCHEMA_VERSION", "directional_net_v2")
                        )
                        active_schema_hash = (
                            str(active_model.get("feature_schema_hash") or "") if isinstance(active_model, dict) else ""
                        )

                        gate_event_counter = getattr(self._app._trade_journal, "get_shadow_gate_event_counts", None)
                        if gate_event_counter is not None:
                            try:
                                gate_events = await asyncio.wait_for(
                                    gate_event_counter(
                                        active_version,
                                        horizon,
                                        label_schema,
                                    ),
                                    timeout=5.0,
                                )
                                if gate_events:
                                    gate["event_total_count"] = int(gate_events.get("total_count", 0) or 0)
                                    gate["event_resolved_count"] = int(gate_events.get("resolved_count", 0) or 0)
                                    gate["event_pending_count"] = int(gate_events.get("pending_count", 0) or 0)
                                    gate["event_pass_count"] = int(gate_events.get("pass_count", 0) or 0)
                                    gate["event_block_count"] = int(gate_events.get("block_count", 0) or 0)
                            except Exception as exc:
                                gate["event_error"] = str(exc) or exc.__class__.__name__

                        if hasattr(self._app._trade_journal, "get_shadow_gate_stats"):
                            try:
                                resolved_gate = await asyncio.wait_for(
                                    self._app._trade_journal.get_shadow_gate_stats(
                                        active_version,
                                        horizon,
                                        label_schema,
                                    ),
                                    timeout=5.0,
                                )
                                if isinstance(resolved_gate, dict):
                                    gate.update(resolved_gate)
                            except Exception as exc:
                                gate["resolved_error"] = str(exc) or exc.__class__.__name__

                        if len(gate) > 1:
                            gate["horizon_minutes"] = horizon
                            diag["shadow_gate_by_horizon"] = {str(horizon): gate}
                            diag[f"shadow_gate_{horizon}m"] = gate
                            diag["shadow_gate_15m"] = gate

                        if hasattr(self._app._trade_journal, "_paper_pnl_for_model"):
                            try:
                                paper = await asyncio.wait_for(
                                    self._app._trade_journal._paper_pnl_for_model(
                                        active_version,
                                        horizon,
                                        active_schema_hash,
                                    ),
                                    timeout=5.0,
                                )
                                diag["paper_pnl_by_horizon"] = {str(horizon): paper}
                                diag[f"paper_pnl_{horizon}m"] = paper
                                diag["paper_pnl_15m"] = paper
                            except Exception as exc:
                                diag["paper_pnl_error"] = str(exc) or exc.__class__.__name__
                self._app._modules.diagnostics.merge_db_fallbacks(diag)
                return diag

            try:
                diag = await asyncio.wait_for(
                    self._app._trade_journal.get_db_diagnostics(lite=lite),
                    timeout=15.0 if lite else 25.0,
                )
            except TimeoutError:
                log.warning("db_diagnostics_timeout", lite=lite)
                return await _timeout_fallback()
            except Exception as exc:
                return {"connected": False, "error": str(exc), "lite": lite}
            if not lite:
                self._app._update_model_gate_quality_from_diag(diag)
            diag["paper_notional_usd"] = (
                float(self._app._settings.MODEL_PAPER_NOTIONAL_USD) if self._app._settings is not None else 5.0
            )
            self._app._modules.diagnostics.merge_db_fallbacks(diag)
            cached = dict(diag)
            if lite:
                self._app._db_diagnostics_lite_cache = cached
                self._app._db_diagnostics_lite_cache_at = now
            else:
                self._app._db_diagnostics_cache = cached
                self._app._db_diagnostics_cache_at = now
            return diag

        async def _healthcheck_provider() -> dict[str, Any]:
            diag = self._app._modules.diagnostics.get_snapshot()
            top_blocker, blockers = self._app._modules.diagnostics.top_blocker_from_diag(diag, default="нет блокировок")
            today_avg_net_bps = None
            if self._app._trade_journal is not None and self._app._trade_journal.is_enabled:
                try:
                    today_avg_net_bps = await self._app._trade_journal.get_today_avg_net_bps()
                except Exception as _hc_exc:
                    log.debug("healthcheck.avg_net_failed", error=str(_hc_exc))
            return {
                "hour_signals_emitted": diag.get("hour_signals_emitted", 0),
                "hour_order_placed": diag.get("hour_order_placed", 0),
                "hour_ml_replacement": diag.get("hour_ml_replacement", 0),
                "hour_rule_fallback_signals": diag.get("hour_rule_fallback_signals", 0),
                "top_blocker": top_blocker,
                "blockers": blockers,
                "today_avg_net_bps": today_avg_net_bps,
            }

        async def _recent_trades_provider() -> list[dict[str, Any]]:
            if self._app._trade_journal is None or not self._app._trade_journal.is_enabled:
                return []
            return cast(list[dict[str, Any]], await self._app._trade_journal.get_recent_closed_trades(limit=10))

        async def _bucket_stats_provider() -> dict[str, Any]:
            assert self._app._settings is not None
            return {
                "buckets": [
                    {
                        "regime": regime,
                        "volatility": volatility,
                        "hour": hour,
                        "avg_bps": avg_bps,
                        "count": count,
                    }
                    for (regime, volatility, hour), (
                        avg_bps,
                        count,
                    ) in self._app._bucket_stats.items()
                ],
                "refreshed_at": (
                    self._app._bucket_stats_refreshed_at.strftime("%Y-%m-%d %H:%M UTC")
                    if self._app._bucket_stats_refreshed_at is not None
                    else None
                ),
                "min_samples": self._app._settings.BUCKET_MIN_SAMPLES,
                "block_below_bps": self._app._settings.BUCKET_BLOCK_AVG_BPS,
            }

        async def _pnl_analysis_provider() -> dict[str, Any]:
            if self._app._trade_journal is None or not self._app._trade_journal.is_enabled:
                return {"connected": False, "error": "trade_journal_unavailable"}
            from trader.training.labels import active_label_schema_version

            horizon = int(self._app._settings.MODEL_AUTO_TRAIN_HORIZON_MINUTES)
            label_schema = active_label_schema_version(
                use_tpsl_exit=bool(self._app._settings.MODEL_LABEL_USE_TPSL_EXIT)
            )
            return cast(
                dict[str, Any],
                await self._app._trade_journal.get_strategy_pnl_analysis(
                    horizon_minutes=horizon,
                    label_schema_version=label_schema,
                ),
            )

        async def _compare_provider() -> dict[str, Any]:
            if self._app._trade_journal is None or not self._app._trade_journal.is_enabled:
                return {"connected": False, "error": "trade_journal_unavailable"}
            from trader.training.labels import active_label_schema_version

            horizon = int(self._app._settings.MODEL_AUTO_TRAIN_HORIZON_MINUTES)
            label_schema = active_label_schema_version(
                use_tpsl_exit=bool(self._app._settings.MODEL_LABEL_USE_TPSL_EXIT)
            )
            return cast(
                dict[str, Any],
                await self._app._trade_journal.get_model_compare_analysis(
                    horizon_minutes=horizon,
                    label_schema_version=label_schema,
                ),
            )

        async def _worst_trades_provider(limit: int) -> list[dict[str, Any]]:
            if self._app._trade_journal is None or not self._app._trade_journal.is_enabled:
                return []
            return cast(list[dict[str, Any]], await self._app._trade_journal.get_worst_prediction_outcomes(limit=limit))

        async def _costs_detailed_provider() -> dict[str, Any]:
            if self._app._trade_journal is None or not self._app._trade_journal.is_enabled:
                return {"connected": False, "error": "trade_journal_unavailable"}
            from trader.training.labels import active_label_schema_version

            horizon = int(self._app._settings.MODEL_AUTO_TRAIN_HORIZON_MINUTES)
            label_schema = active_label_schema_version(
                use_tpsl_exit=bool(self._app._settings.MODEL_LABEL_USE_TPSL_EXIT)
            )
            return cast(
                dict[str, Any],
                await self._app._trade_journal.get_detailed_costs(
                    horizon_minutes=horizon,
                    label_schema_version=label_schema,
                ),
            )

        async def _model_performance_provider() -> list[dict[str, Any]]:
            if self._app._trade_journal is None or not self._app._trade_journal.is_enabled:
                return list(self._app._model_performance_cache)
            try:
                rows = cast(
                    list[dict[str, Any]],
                    await asyncio.wait_for(
                        self._app._trade_journal.get_model_performance_history(),
                        timeout=15.0,
                    ),
                )
            except TimeoutError:
                log.warning("model_performance_history.timeout")
                rows = []
            except Exception as exc:
                log.warning("model_performance_history.failed", error=str(exc))
                rows = []
            if rows:
                self._app._model_performance_cache = rows
                self._app._model_performance_cache_at = datetime.now(tz=UTC)
                return rows
            return list(self._app._model_performance_cache)

        async def _champion_health_provider() -> dict[str, Any]:
            if self._app._trade_journal is None or not self._app._trade_journal.is_enabled:
                return {"connected": False, "error": "trade_journal_unavailable"}
            return cast(dict[str, Any], await self._app._trade_journal.get_champion_health())

        async def _add_subscription(chat_id: int) -> None:
            if self._app._trade_journal is not None:
                await self._app._trade_journal.add_telegram_subscription(chat_id)

        async def _remove_subscription(chat_id: int) -> None:
            if self._app._trade_journal is not None:
                await self._app._trade_journal.remove_telegram_subscription(chat_id)

        async def _load_subscriptions() -> list[int]:
            if self._app._trade_journal is None or not self._app._trade_journal.is_enabled:
                return []
            return cast(list[int], await self._app._trade_journal.get_telegram_subscriptions())

        async def _attribution_provider(days: int = 7) -> list[dict[str, Any]]:
            if self._app._trade_journal is None or not self._app._trade_journal.is_enabled:
                return []
            return await self._app._trade_journal.get_pnl_attribution(days=days)

        async def _best_challenger_provider() -> str | None:
            if self._app._trade_journal is None or not self._app._trade_journal.is_enabled:
                return None
            try:
                from trader.ml.auto_promotion import AutoPromotionConfig, AutoPromotionEngine

                engine = AutoPromotionEngine(
                    trade_journal=self._app._trade_journal,
                    config=AutoPromotionConfig.from_settings(self._app._settings),
                )
                return await engine.best_challenger()
            except Exception as exc:
                log.debug("best_challenger.lookup_failed", error=str(exc))
                return None

        controller = TradingController(
            pause=self._app._pause_trading,
            resume=self._app._resume_trading,
            set_shadow=self._app._set_shadow_mode,
            set_risk_profile=self._app._change_risk_profile,
            emergency_stop=self._app._emergency_stop,
            start_training=self._app._start_model_training,
            start_training_all=self._app._start_model_training_all,
            promote_model=self._app._start_model_promote,
            runtime_settings=self._app._runtime_settings,
            set_runtime_setting=self._app._set_runtime_setting,
            symbol_candidates=self._app._symbol_candidates,
            selected_symbols=self._app._selected_symbols,
            toggle_symbol=self._app._toggle_manual_symbol,
            is_paused=lambda: self._app._trading_paused,
            is_shadow=lambda: (
                self._app._execution_engine._shadow_mode if self._app._execution_engine is not None else True
            ),
            current_profile=lambda: self._app._current_risk_profile_str,
            active_symbols=self._app._modules.diagnostics.runtime_active_symbols,
            regime_for=_regime_for,
            signal_log=self._app._signal_log,
            diagnostics_provider=self._app._modules.diagnostics.get_snapshot,
            db_diagnostics_provider=_db_diagnostics_provider,
            allow_risk_increase=self._app._settings.TELEGRAM_ALLOW_RISK_INCREASE,
            healthcheck_provider=_healthcheck_provider,
            recent_trades_provider=_recent_trades_provider,
            bucket_stats_provider=_bucket_stats_provider,
            pnl_analysis_provider=_pnl_analysis_provider,
            compare_provider=_compare_provider,
            worst_trades_provider=_worst_trades_provider,
            costs_detailed_provider=_costs_detailed_provider,
            model_performance_provider=_model_performance_provider,
            champion_health_provider=_champion_health_provider,
            attribution_provider=_attribution_provider,
            best_challenger_provider=_best_challenger_provider,
            enrich_db_diag_fallbacks=self._app._modules.diagnostics.merge_db_fallbacks,
            add_subscription=_add_subscription,
            remove_subscription=_remove_subscription,
            load_subscriptions=_load_subscriptions,
        )

        allowed_chat_ids = set(self._app._settings.TELEGRAM_ALLOWED_CHAT_IDS)
        delivery_mode, webhook_url = self.resolve_delivery()
        self._app._telegram_bot = TelegramMonitorBot(
            config=TelegramBotConfig(
                token=token,
                allowed_chat_ids=allowed_chat_ids,
                trading_mode=self._app._settings.TRADING_MODE.value,
                risk_profile=self._app._settings.RISK_PROFILE.value,
                bybit_use_testnet=self._app._settings.BYBIT_USE_TESTNET,
                default_category=self._app._settings.DEFAULT_MARKET_CATEGORY,
                redis_url=self._app._settings.REDIS_URL.get_secret_value(),
                delivery_mode=delivery_mode,
                webhook_url=webhook_url,
                webhook_secret=self._app._settings.TELEGRAM_WEBHOOK_SECRET.get_secret_value(),
                polling_conflict_recovery_wait_s=self._app._settings.TELEGRAM_POLLING_CONFLICT_RECOVERY_WAIT_SECONDS,
                polling_watchdog_interval_s=self._app._settings.TELEGRAM_POLLING_WATCHDOG_INTERVAL_SECONDS,
                polling_zombie_silence_s=self._app._settings.TELEGRAM_POLLING_ZOMBIE_SILENCE_SECONDS,
            ),
            health_provider=self._app._health_checker.overall_health,
            adapter_factory=lambda: self._app._bybit_adapter,
            controller=controller,
            net_results_provider=self._app._get_net_results,
        )
        try:
            started = await self._app._telegram_bot.start(http_app=self._app._fastapi_app)
        except Exception as exc:
            log.warning(
                "telegram_bot_not_started",
                error=f"{type(exc).__name__}: {exc}",
                health=self._app._telegram_bot.health_snapshot(),
            )
            return
        if started:
            from trader.monitoring.deploy_info import deploy_label

            deploy_id = deploy_label()
            log.info(
                "telegram_bot_started",
                delivery_mode=delivery_mode,
                webhook_url=webhook_url or None,
                deploy_id=deploy_id,
            )
            try:
                await self._app._telegram_bot.notify(
                    f"🚀 <b>Бот запущен</b>\nDeploy: <code>{html.escape(deploy_id)}</code>"
                )
            except Exception as notify_exc:
                log.debug("telegram.startup_notify_failed", error=str(notify_exc))
        else:
            log.warning("telegram_bot_not_started", health=self._app._telegram_bot.health_snapshot())
