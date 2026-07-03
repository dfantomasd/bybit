"""Operator controls: pause/shadow/risk, runtime limits, symbol selection, train/promote."""

from __future__ import annotations

import asyncio
import html
import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast

from trader.domain.enums import TradingMode
from trader.modules.base import AppBoundModule
from trader.monitoring.logging import get_logger
from trader.runtime.constants import _SYMBOLS

log = get_logger(__name__)


class OperatorControlsModule(AppBoundModule):
    name = "operator"

    async def _refresh_balance(self) -> Decimal:
        return await self._app._modules.execution.refresh_balance()

    async def _init_risk_manager(self, initial_capital: Decimal) -> None:
        await self._app._modules.execution.init_risk_manager(initial_capital)

    async def _run_model_training(self, min_samples: int, horizon: int, label_bps: float) -> None:
        await self._app._modules.training.run_model_training(min_samples, horizon, label_bps)

    async def _run_model_training_all(self) -> None:
        await self._app._modules.training.run_model_training_all()

    @staticmethod
    def _strategy_lab_summary(path: str | None) -> dict[str, Any] | None:
        if not path:
            return None
        try:
            from trader.strategies.discovered_rule import writable_discovered_rules_path

            rule_path = writable_discovered_rules_path(path)
            if not rule_path.exists():
                return {"exists": False, "path": str(rule_path)}
            payload = json.loads(rule_path.read_text(encoding="utf-8"))
            rules = list(payload.get("rules") or []) if isinstance(payload, dict) else []
            error = payload.get("error") if isinstance(payload, dict) else None
            stage = payload.get("stage") if isinstance(payload, dict) else None
            hint = None
            error_text = str(error or "").lower()
            if "timeout" in error_text:
                hint = "Generator timed out; increase timeout or reduce min samples/rule search size."
            elif "eauthquery" in error_text or "authentication query failed" in error_text:
                hint = "Database auth/pooler failed; fix POSTGRES_DSN/Supabase availability before rule generation."
            elif "read-only" in error_text:
                hint = "Training/discovery is connected to a read-only DB/session; use primary writable Postgres."
            elif error:
                hint = "Open logs around discovered_rule.auto_generate_failed for the full traceback."
            return {
                "exists": True,
                "path": str(rule_path),
                "status": payload.get("status") if isinstance(payload, dict) else None,
                "sample_count": payload.get("sample_count") if isinstance(payload, dict) else None,
                "rule_count": len(rules),
                "stage": stage,
                "error": error,
                "hint": hint,
                "top_rule": rules[0].get("rule_id") if rules and isinstance(rules[0], dict) else None,
                "top_validation_avg_net_bps": (
                    rules[0].get("validation_avg_net_bps") if rules and isinstance(rules[0], dict) else None
                ),
            }
        except Exception as exc:
            return {"exists": None, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}

    def _active_execution_allowed(self) -> bool:
        return self._app._modules.signal_policy.active_execution_allowed()

    async def _on_screener_symbols_added(self, symbols: list[str]) -> None:
        await self._app._modules.market_data.on_screener_symbols_added(symbols)

    async def pause_trading(self) -> None:
        self._app._trading_paused = True
        log.info("trading.paused")

    async def resume_trading(self) -> None:
        if self._app._kill_switch is not None and self._app._kill_switch.is_active:
            from trader.domain.enums import KillSwitchMode

            if self._app._kill_switch.current_mode != KillSwitchMode.PAUSE_NEW_ENTRIES:
                log.error(
                    "resume_trading.blocked_by_kill_switch",
                    mode=self._app._kill_switch.current_mode,
                )
                return
        self._app._trading_paused = False
        log.info("trading.resumed")

    async def set_shadow_mode(self, enabled: bool) -> None:
        assert self._app._settings is not None
        if not enabled:
            if not self._active_execution_allowed():
                raise RuntimeError(
                    "Active execution requires BYBIT_USE_TESTNET=true, or LIVE_MODE=true with TRADING_MODE=LIVE/CANARY_LIVE."
                )
        self._app._settings.SHADOW_MODE = enabled
        if self._app._execution_engine is not None:
            self._app._execution_engine._shadow_mode = enabled
        if self._app._fee_provider is not None:
            self._app._fee_provider.shadow_mode = enabled
        log.info("shadow_mode.changed", enabled=enabled)

    async def change_risk_profile(self, profile: Any) -> None:
        """Hot-swap the risk profile without restarting — preserves all risk state.

        SAFETY: Blocked in LIVE and CANARY_LIVE modes because a profile change
        alters leverage limits, position caps, and daily-loss thresholds while
        real positions are open — an unsafe combination requiring a clean restart.
        """
        assert self._app._settings is not None
        if self._app._settings.TRADING_MODE in (TradingMode.LIVE, TradingMode.CANARY_LIVE):
            raise RuntimeError(
                "Risk profile hot-swap is not permitted in LIVE / CANARY_LIVE mode. "
                "Restart the service to apply a new profile."
            )

        old = self._app._current_risk_profile_str
        capital = await self._refresh_balance()

        # Preserve ALL risk state that spans profile boundaries.
        # Reinitialising would silently reset peak equity → new hard-stop baseline
        # that ignores losses already taken — a critical safety hole.
        old_drawdown = self._app._risk_manager._drawdown if self._app._risk_manager is not None else None
        old_daily_pnl = self._app._risk_manager.daily_pnl if self._app._risk_manager is not None else Decimal("0")
        # Preserve the kill switch too — init_risk_manager() always builds a
        # fresh, inactive KillSwitch, which would silently clear an active
        # FULL_STOP/emergency-stop condition on every profile hot-swap.
        old_kill_switch = self._app._kill_switch
        # Stop the outgoing risk manager's daily-reset background task before
        # it's discarded; init_risk_manager() doesn't track/cancel it, so
        # skipping this leaks one orphaned task per profile change.
        if self._app._risk_manager is not None:
            self._app._risk_manager.stop_daily_reset_scheduler()

        if self._app._settings is not None:
            self._app._settings.RISK_PROFILE = profile
        await self._init_risk_manager(capital)

        if self._app._risk_manager is not None:
            if old_drawdown is not None:
                self._app._risk_manager._drawdown = old_drawdown
            # Restore daily PnL so daily loss limit is not reset mid-day
            if old_daily_pnl != Decimal("0"):
                self._app._risk_manager._daily_pnl = old_daily_pnl
            if old_kill_switch is not None and old_kill_switch.is_active:
                self._app._kill_switch = old_kill_switch
                self._app._risk_manager._kill_switch = old_kill_switch

        # Rewire execution engine to the new risk manager and exposure
        # tracker — init_risk_manager() always builds a fresh ExposureTracker
        # (self._app._exposure_tracker), so without this the execution
        # engine keeps recording fills into the discarded old tracker while
        # risk checks read from the new (falsely empty) one.
        if self._app._execution_engine is not None:
            self._app._execution_engine._risk_manager = self._app._risk_manager
            self._app._execution_engine._exposure = self._app._exposure_tracker
        self._app._current_risk_profile_str = profile.value
        log.info("risk_profile.changed", old=old, new=profile.value)
        if self._app._telegram_bot is not None:
            await self._app._telegram_bot.notify_risk_changed(old, profile.value)

    async def emergency_stop(self) -> None:
        self._app._trading_paused = True
        if self._app._kill_switch is not None:
            from trader.domain.enums import KillSwitchMode

            await self._app._kill_switch.activate(
                KillSwitchMode.FULL_STOP,
                reason="operator emergency stop via Telegram",
                operator="telegram",
            )
        cancelled = 0
        if self._app._execution_engine is not None:
            try:
                cancelled = await self._app._execution_engine.cancel_all_open_orders()
            except Exception as exc:
                log.error("emergency_stop.cancel_orders_failed", error=str(exc))
        log.critical("emergency_stop.activated", source="telegram", orders_cancelled=cancelled)
        if self._app._telegram_bot is not None:
            cancel_note = f" Cancelled {cancelled} open order(s)." if cancelled else ""
            await self._app._telegram_bot.notify(
                f"🚨 <b>Emergency stop activated.</b> No new trades.{cancel_note} Manual restart required."
            )

    async def start_model_training(self, min_samples: int = 500, horizon: int = 15, label_bps: float = 5.0) -> str:
        """Start offline model training in a subprocess; trading loop stays isolated."""
        async with self._app._training_start_lock:
            if self._app._training_task is not None and not self._app._training_task.done():
                return "⏳ Обучение уже идет."
            if self._app._trade_journal is not None and not self._app._trade_journal.is_enabled:
                await self._app._trade_journal.reconnect_if_needed(force=True)
            if self._app._trade_journal is None or not self._app._trade_journal.is_enabled:
                raise RuntimeError("Trade journal/Postgres is not available.")
            self._app._training_task = asyncio.create_task(
                self._run_model_training(min_samples, horizon, label_bps),
                name="model-training",
            )
            self._app._background_tasks.append(self._app._training_task)
        return (
            "🧠 <b>Обучение запущено</b>\n"
            f"минимум примеров=<code>{min_samples}</code>, горизонт=<code>{horizon}m</code>, "
            f"порог=<code>{label_bps:g} bps</code>\n"
            "Результат придет сюда после завершения."
        )

    async def start_model_training_all(self) -> str:
        """Start sequential training on all available data for every horizon (5m, 15m, 30m, 60m)."""
        async with self._app._training_start_lock:
            if self._app._training_task is not None and not self._app._training_task.done():
                return "⏳ Обучение уже идет."
            if self._app._trade_journal is not None and not self._app._trade_journal.is_enabled:
                await self._app._trade_journal.reconnect_if_needed(force=True)
            if self._app._trade_journal is None or not self._app._trade_journal.is_enabled:
                raise RuntimeError("Trade journal/Postgres is not available.")
            self._app._training_task = asyncio.create_task(
                self._run_model_training_all(),
                name="model-training-all",
            )
            self._app._background_tasks.append(self._app._training_task)
        return (
            "🧠🔁 <b>Обучение ВСЕ запущено</b>\n"
            f"Горизонты: <code>5m, 15m, 30m, 60m</code> | Порог: <code>{self._app._settings.MODEL_AUTO_TRAIN_LABEL_BPS} bps</code>\n"
            "Используются все доступные примеры (мин. 100).\n"
            "Результаты придут по мере завершения каждого горизонта."
        )

    async def start_model_promote(self, version: str) -> str:
        """Promote a model through the same strict engine used by auto-promotion."""
        if self._app._trade_journal is None or not self._app._trade_journal.is_enabled:
            raise RuntimeError("Trade journal/Postgres is not available.")

        def code_text(value: str, limit: int = 800) -> str:
            return html.escape(value[-limit:])

        log.info("model_promote.started", version=version)
        try:
            from trader.ml.auto_promotion import AutoPromotionConfig, AutoPromotionEngine

            async def _reload_registry() -> None:
                if self._app._model_registry is not None:
                    await self._app._model_registry.load_active_model()

            engine = AutoPromotionEngine(
                trade_journal=self._app._trade_journal,
                config=AutoPromotionConfig.from_settings(self._app._settings),
                reload_registry=_reload_registry,
            )
            decision = await asyncio.wait_for(engine.promote(version), timeout=60.0)
            if decision.promote:
                message = f"Model {version} promoted to CHAMPION: {', '.join(decision.reasons)}"
                if self._app._telegram_bot is not None:
                    await self._app._telegram_bot.notify(
                        f"🏆 <b>Модель промоутирована</b>\n<code>{code_text(message)}</code>"
                    )
                return f"🏆 <b>Промоут успешен!</b>\n<code>{code_text(message)}</code>"
            out = "; ".join(decision.reasons)
            if self._app._telegram_bot is not None:
                await self._app._telegram_bot.notify(f"❌ <b>Промоут не прошёл</b>\n<code>{code_text(out)}</code>")
            return f"❌ <b>Промоут не прошёл:</b>\n<code>{code_text(out)}</code>"
        except TimeoutError:
            return "❌ Промоут завис (timeout 60s)"
        except Exception as exc:
            log.exception("model_promote.failed", version=version)
            return f"❌ Ошибка промоута: <code>{html.escape(str(exc))}</code>"

    def runtime_settings(self) -> dict[str, Any]:
        from trader.training.labels import active_label_schema_version

        bucket_stats_age_s = None
        bucket_stats_refreshed_at = getattr(self._app, "_bucket_stats_refreshed_at", None)
        if isinstance(bucket_stats_refreshed_at, datetime):
            bucket_stats_age_s = max(
                0.0,
                (
                    datetime.now(tz=UTC)
                    - bucket_stats_refreshed_at.astimezone(UTC)
                ).total_seconds(),
            )
        shadow_probe_side_stats = getattr(self._app, "_shadow_probe_side_stats", None) or {}
        shadow_probe_symbol_stats = getattr(self._app, "_shadow_probe_symbol_stats", None) or {}
        shadow_probe_symbol_cooldowns = getattr(self._app, "_shadow_probe_symbol_cooldowns", None) or {}
        now = datetime.now(tz=UTC)
        active_shadow_probe_symbol_cooldowns = {
            symbol: max(0.0, (until.astimezone(UTC) - now).total_seconds())
            for symbol, until in shadow_probe_symbol_cooldowns.items()
            if isinstance(until, datetime) and until > now
        }
        blocked_probe_symbols = []
        if self._app._settings is not None:
            blocked_probe_symbols = [
                symbol
                for symbol, (avg_bps, count) in shadow_probe_symbol_stats.items()
                if count >= getattr(self._app._settings, "SHADOW_PROBE_SYMBOL_MIN_SAMPLES", 0)
                and avg_bps < getattr(self._app._settings, "SHADOW_PROBE_SYMBOL_MIN_AVG_BPS", 0.0)
            ][:20]

        return {
            "paused": self._app._trading_paused,
            "shadow": self._app._execution_engine._shadow_mode if self._app._execution_engine is not None else True,
            "risk_profile": self._app._current_risk_profile_str,
            "max_entries_per_minute": (
                self._app._execution_engine._max_entries_per_minute if self._app._execution_engine is not None else None
            ),
            "max_concurrent_pending": (
                self._app._execution_engine._max_concurrent_pending if self._app._execution_engine is not None else None
            ),
            "max_same_side": self._app._execution_engine._max_same_side
            if self._app._execution_engine is not None
            else None,
            "max_positions": (
                self._app._execution_engine._max_open_positions
                if self._app._execution_engine is not None
                else (self._app._settings.MAX_POSITIONS if self._app._settings is not None else None)
            ),
            "screener_max_price_usd": self._app._settings.SCREENER_MAX_PRICE_USD
            if self._app._settings is not None
            else None,
            "feature_max_symbols": self._app._screener._feature_max if self._app._screener is not None else None,
            "execution_candidates": self._app._screener._exec_candidates if self._app._screener is not None else None,
            "manual_symbols": self.selected_symbols(),
            "model_gate_canary_enabled": (
                self._app._settings.MODEL_GATE_CANARY_ENABLED if self._app._settings is not None else False
            ),
            "model_gate_threshold": self._app._settings.MODEL_SHADOW_GATE_THRESHOLD
            if self._app._settings is not None
            else None,
            "model_gate_quality": self._app._model_gate_quality,
            "scalp_strict_shadow": self._app._scalp_strict_shadow(),
            "shadow_apply_net_edge_gate": (
                self._app._execution_engine._shadow_apply_net_edge_gate
                if self._app._execution_engine is not None
                else None
            ),
            "min_expected_net_edge_pct": (
                self._app._settings.MIN_EXPECTED_NET_EDGE_PCT if self._app._settings is not None else None
            ),
            "net_edge_safety_margin_pct": (
                self._app._settings.NET_EDGE_SAFETY_MARGIN_PCT if self._app._settings is not None else None
            ),
            "shadow_probe_enabled": self._app._settings.SHADOW_PROBE_ENABLED
            if self._app._settings is not None
            else None,
            "shadow_probe_paper_collection_mode": (
                self._app._settings.SHADOW_PROBE_PAPER_COLLECTION_MODE if self._app._settings is not None else None
            ),
            "shadow_probe_paper_regimes": (
                self._app._settings.SHADOW_PROBE_PAPER_REGIMES if self._app._settings is not None else None
            ),
            "shadow_probe_bypasses_live_edge_gate": True,
            "shadow_probe_min_net_return_pct": (
                self._app._settings.SHADOW_PROBE_MIN_NET_RETURN_PCT if self._app._settings is not None else None
            ),
            "shadow_probe_effective_min_net_return_pct": (
                self._app._settings.SHADOW_PROBE_MIN_NET_RETURN_PCT if self._app._settings is not None else None
            ),
            "shadow_probe_min_abs_imbalance": (
                getattr(self._app._settings, "SHADOW_PROBE_MIN_ABS_IMBALANCE", None)
                if self._app._settings is not None
                else None
            ),
            "shadow_probe_min_tp_pct": (
                getattr(self._app._settings, "SHADOW_PROBE_MIN_TP_PCT", None)
                if self._app._settings is not None
                else None
            ),
            "shadow_probe_max_tp_pct": (
                getattr(self._app._settings, "SHADOW_PROBE_MAX_TP_PCT", None)
                if self._app._settings is not None
                else None
            ),
            "shadow_probe_min_net_reward_risk": (
                getattr(self._app._settings, "SHADOW_PROBE_MIN_NET_REWARD_RISK", None)
                if self._app._settings is not None
                else None
            ),
            "shadow_probe_symbol_top_n": (
                self._app._settings.SHADOW_PROBE_SYMBOL_TOP_N if self._app._settings is not None else None
            ),
            "shadow_probe_symbol_warmup_seconds": (
                self._app._settings.SHADOW_PROBE_SYMBOL_WARMUP_SECONDS if self._app._settings is not None else None
            ),
            "shadow_probe_sell_enabled": (
                self._app._settings.SHADOW_PROBE_SELL_ENABLED if self._app._settings is not None else None
            ),
            "shadow_probe_side_block_enabled": (
                self._app._settings.SHADOW_PROBE_SIDE_BLOCK_ENABLED if self._app._settings is not None else None
            ),
            "bucket_stats_refresh_seconds": (
                getattr(self._app._settings, "BUCKET_STATS_REFRESH_SECONDS", None)
                if self._app._settings is not None
                else None
            ),
            "bucket_stats_age_s": bucket_stats_age_s,
            "shadow_probe_side_stats_count": len(shadow_probe_side_stats),
            "shadow_probe_symbol_stats_count": len(shadow_probe_symbol_stats),
            "shadow_probe_blocked_symbols": blocked_probe_symbols,
            "shadow_probe_symbol_loss_cooldown_enabled": (
                getattr(self._app._settings, "SHADOW_PROBE_SYMBOL_LOSS_COOLDOWN_ENABLED", None)
                if self._app._settings is not None
                else None
            ),
            "shadow_probe_symbol_cooldowns": {
                symbol: round(seconds_left, 1)
                for symbol, seconds_left in sorted(active_shadow_probe_symbol_cooldowns.items())
            },
            "shadow_probe_eligible_symbols": sorted(self._app._shadow_probe_eligible_symbols or []),
            "shadow_probe_blocked_sides": [
                f"{symbol}:{side}"
                for (symbol, side), (avg_bps, count) in shadow_probe_side_stats.items()
                if self._app._settings is not None
                and count >= self._app._settings.SHADOW_PROBE_SIDE_MIN_SAMPLES
                and avg_bps < self._app._settings.SHADOW_PROBE_SIDE_BLOCK_AVG_BPS
            ][:20],
            "discovered_rule_strategy_enabled": (
                getattr(self._app._settings, "DISCOVERED_RULE_STRATEGY_ENABLED", None)
                if self._app._settings is not None
                else None
            ),
            "discovered_rules_path": (
                getattr(self._app._settings, "DISCOVERED_RULES_PATH", None) if self._app._settings is not None else None
            ),
            "discovered_rules_file": (
                self._strategy_lab_summary(getattr(self._app._settings, "DISCOVERED_RULES_PATH", None))
                if self._app._settings is not None
                else None
            ),
            "discovered_rule_auto_generate": (
                getattr(self._app._settings, "DISCOVERED_RULE_AUTO_GENERATE", None)
                if self._app._settings is not None
                else None
            ),
            "discovered_rule_auto_generate_min_samples": (
                getattr(self._app._settings, "DISCOVERED_RULE_AUTO_GENERATE_MIN_SAMPLES", None)
                if self._app._settings is not None
                else None
            ),
            "discovered_rule_auto_generate_timeout_seconds": (
                getattr(self._app._settings, "DISCOVERED_RULE_AUTO_GENERATE_TIMEOUT_SECONDS", None)
                if self._app._settings is not None
                else None
            ),
            "discovered_rule_min_validation_count": (
                getattr(self._app._settings, "DISCOVERED_RULE_MIN_VALIDATION_COUNT", None)
                if self._app._settings is not None
                else None
            ),
            "discovered_rule_min_validation_net_bps": (
                getattr(self._app._settings, "DISCOVERED_RULE_MIN_VALIDATION_NET_BPS", None)
                if self._app._settings is not None
                else None
            ),
            "model_auto_train_min_samples": (
                self._app._settings.MODEL_AUTO_TRAIN_MIN_SAMPLES if self._app._settings is not None else 1000
            ),
            "model_auto_train_horizon_minutes": (
                self._app._settings.MODEL_AUTO_TRAIN_HORIZON_MINUTES if self._app._settings is not None else 5
            ),
            "model_auto_train_label_bps": (
                self._app._settings.MODEL_AUTO_TRAIN_LABEL_BPS if self._app._settings is not None else 2.0
            ),
            "label_schema_version": (
                active_label_schema_version(use_tpsl_exit=bool(self._app._settings.MODEL_LABEL_USE_TPSL_EXIT))
                if self._app._settings is not None
                else "directional_net_v1"
            ),
            "strategy_priority_order": (
                self._app._settings.STRATEGY_PRIORITY_ORDER if self._app._settings is not None else ""
            ),
            "scalp_strategy_priority_order": (
                self._app._settings.SCALP_STRATEGY_PRIORITY_ORDER if self._app._settings is not None else ""
            ),
        }

    async def set_runtime_setting(self, key: str, value: Any) -> str:
        assert self._app._settings is not None
        key = key.lower()
        if key == "entries":
            ivalue = int(value)
            if not 0 < ivalue <= 10:
                raise ValueError("entries must be 1..10")
            self._app._settings.MAX_NEW_ENTRIES_PER_MINUTE = ivalue
            if self._app._execution_engine is not None:
                self._app._execution_engine._max_entries_per_minute = ivalue
            return f"Max entries/min set to {ivalue}"
        if key == "pending":
            ivalue = int(value)
            if not 0 < ivalue <= 10:
                raise ValueError("pending must be 1..10")
            self._app._settings.MAX_CONCURRENT_PENDING_ENTRIES = ivalue
            if self._app._execution_engine is not None:
                self._app._execution_engine._max_concurrent_pending = ivalue
            return f"Max pending entries set to {ivalue}"
        if key == "same_side":
            ivalue = int(value)
            if not 0 < ivalue <= 10:
                raise ValueError("same_side must be 1..10")
            self._app._settings.MAX_SAME_SIDE_POSITIONS = ivalue
            if self._app._execution_engine is not None:
                self._app._execution_engine._max_same_side = ivalue
            return f"Max same-side positions set to {ivalue}"
        if key == "max_positions":
            ivalue = int(value)
            if not 1 <= ivalue <= 10:
                raise ValueError("max_positions must be 1..10")
            self._app._settings.MAX_POSITIONS = ivalue
            if self._app._execution_engine is not None:
                self._app._execution_engine._max_open_positions = ivalue
            return f"Max simultaneous positions set to {ivalue}"
        if key == "price_cap":
            fvalue = float(value)
            if fvalue < 0 or fvalue > 100_000:
                raise ValueError("price_cap must be 0..100000")
            self._app._settings.SCREENER_MAX_PRICE_USD = fvalue
            if self._app._screener is not None:
                self._app._screener._max_price_usd = fvalue
            return f"Screener price cap set to {fvalue:g}"
        if key == "feature_symbols":
            ivalue = int(value)
            if not 1 <= ivalue <= self._app._settings.SCREENER_WIDE_MAX_SYMBOLS:
                raise ValueError(f"feature_symbols must be 1..{self._app._settings.SCREENER_WIDE_MAX_SYMBOLS}")
            self._app._settings.SCREENER_FEATURE_MAX_SYMBOLS = ivalue
            if self._app._settings.SCREENER_EXECUTION_CANDIDATES > ivalue:
                self._app._settings.SCREENER_EXECUTION_CANDIDATES = ivalue
            if self._app._screener is not None:
                self._app._screener._feature_max = ivalue
                if self._app._screener._exec_candidates > ivalue:
                    self._app._screener._exec_candidates = ivalue
            return f"Feature symbols set to {ivalue}"
        if key == "exec_candidates":
            ivalue = int(value)
            if not 1 <= ivalue <= self._app._settings.SCREENER_FEATURE_MAX_SYMBOLS:
                raise ValueError(f"exec_candidates must be 1..{self._app._settings.SCREENER_FEATURE_MAX_SYMBOLS}")
            self._app._settings.SCREENER_EXECUTION_CANDIDATES = ivalue
            if self._app._screener is not None:
                self._app._screener._exec_candidates = ivalue
            return f"Execution candidates set to {ivalue}"
        if key == "model_gate":
            sval = str(value).strip().lower()
            if sval not in {"on", "off", "true", "false", "1", "0"}:
                raise ValueError("model_gate must be on/off")
            if sval in {"on", "true", "1"}:
                raise ValueError(
                    "Canary model gate can only be enabled through environment configuration after manual readiness review."
                )
            self._app._settings.MODEL_GATE_CANARY_ENABLED = False
            return "Model gate canary remains OFF (runtime enable blocked — use env vars)"
        if key == "model_gate_threshold":
            fvalue = float(value)
            if not 0.50 <= fvalue <= 0.80:
                raise ValueError("model_gate_threshold must be 0.50..0.80")
            self._app._settings.MODEL_SHADOW_GATE_THRESHOLD = fvalue
            return f"Model gate threshold set to {fvalue:.2f}"
        raise ValueError("unknown setting")

    def symbol_candidates(self) -> list[str]:
        if self._app._screener is None:
            return list(_SYMBOLS)
        wide = self._app._screener.wide_universe
        if wide:
            return [str(item.symbol) for item in wide[:100]]
        return cast(list[str], self._app._screener.active_symbols)

    def selected_symbols(self) -> list[str]:
        if self._app._screener is None:
            return []
        return cast(list[str], self._app._screener.manual_symbols)

    async def toggle_manual_symbol(self, symbol: str) -> str:
        if self._app._screener is None:
            raise RuntimeError("Сканер еще не запущен")
        symbol = symbol.upper()
        if symbol not in set(self.symbol_candidates()):
            raise ValueError(f"{symbol} сейчас не проходит фильтры сканера")

        selected = set(self._app._screener.manual_symbols)
        if symbol in selected:
            selected.remove(symbol)
            self._app._screener.set_manual_symbols(sorted(selected))
            return f"☐ <code>{symbol}</code> убрана из ручного списка."

        selected.add(symbol)
        self._app._screener.set_manual_symbols(sorted(selected))
        if symbol not in self._app._screener.active_symbols:
            await self._on_screener_symbols_added([symbol])
        return f"✅ <code>{symbol}</code> добавлена: бот будет учиться и торговать по ней, пока она проходит фильтры."
