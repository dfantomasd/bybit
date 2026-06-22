"""Operator diagnostics and readiness reporting."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from trader.domain.enums import TradingMode
from trader.modules.base import AppBoundModule
from trader.monitoring.logging import get_logger
from trader.runtime.constants import _DIAG_WINDOW, _SYMBOLS

log = get_logger(__name__)


class DiagnosticsModule(AppBoundModule):
    name = "diagnostics"

    @staticmethod
    def dict_or_empty(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def utc_age_seconds(value: Any) -> float | None:
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
        return float(max(0.0, (datetime.now(tz=UTC) - value.astimezone(UTC)).total_seconds()))

    def economic_readiness_report(
        self,
        *,
        db_diag: dict[str, Any],
        runtime_diag: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return an operator-facing readiness verdict for real-money modes."""
        assert self._app._settings is not None
        runtime_diag = runtime_diag or {}
        issues: list[str] = []
        metrics: dict[str, Any] = {}

        if not db_diag.get("connected"):
            issues.append("db_not_connected")

        latest_age = DiagnosticsModule.utc_age_seconds(db_diag.get("latest_candle_1m"))
        metrics["latest_candle_age_s"] = latest_age
        if latest_age is None or latest_age > 600:
            issues.append(f"stale_1m_candle:{latest_age}")

        active_symbols = runtime_diag.get("active_symbols") or []
        metrics["active_symbols"] = len(active_symbols)
        if len(active_symbols) < 3:
            issues.append(f"insufficient_active_symbols:{len(active_symbols)}")

        feature_snapshots = int(db_diag.get("feature_snapshots") or 0)
        metrics["feature_snapshots"] = feature_snapshots
        if feature_snapshots < 1000:
            issues.append(f"insufficient_feature_snapshots:{feature_snapshots}")

        prediction_outcomes = int(db_diag.get("prediction_outcomes") or 0)
        metrics["prediction_outcomes"] = prediction_outcomes
        if prediction_outcomes < 1000:
            issues.append(f"insufficient_prediction_outcomes:{prediction_outcomes}")

        active_model = DiagnosticsModule.dict_or_empty(db_diag.get("active_model_version"))
        model_metrics = DiagnosticsModule.dict_or_empty(active_model.get("metrics"))
        model_status = str(active_model.get("status") or "")
        try:
            model_horizon = int(
                db_diag.get("model_gate_horizon_minutes")
                or model_metrics.get("horizon_minutes")
                or model_metrics.get("label_horizon_minutes")
                or getattr(self._app._settings, "MODEL_AUTO_TRAIN_HORIZON_MINUTES", 15)
                or 15
            )
        except (TypeError, ValueError):
            model_horizon = 15
        metrics["model_horizon_minutes"] = model_horizon
        training_by_horizon = DiagnosticsModule.dict_or_empty(db_diag.get("training_eligible_by_horizon"))
        raw_trainable = training_by_horizon.get(str(model_horizon))
        if raw_trainable is None:
            raw_trainable = db_diag.get(f"training_eligible_{model_horizon}m")
        if raw_trainable is None:
            raw_trainable = db_diag.get("training_eligible_15m")
        if raw_trainable is None:
            raw_trainable = db_diag.get("labelled_samples_15m")
        trainable = int(raw_trainable or 0)
        metrics["training_eligible_model_horizon"] = trainable
        if trainable < 1000:
            issues.append(f"insufficient_labelled_{model_horizon}m:{trainable}")
        metrics["active_model_version"] = active_model.get("version")
        metrics["active_model_status"] = model_status or None
        if not active_model.get("version"):
            issues.append("missing_active_model")
        if model_status != "CHAMPION":
            issues.append(f"active_model_not_champion:{model_status or 'none'}")

        expected_quality = str(self._app._settings.MODEL_GATE_CANARY_MIN_QUALITY).upper()
        quality = str(model_metrics.get("quality") or "").upper()
        metrics["model_quality"] = quality or None
        if expected_quality and quality != expected_quality:
            issues.append(f"model_quality_not_{expected_quality.lower()}:{quality or 'none'}")

        walk_forward = DiagnosticsModule.float_or_none(model_metrics.get("walk_forward_expectancy_bps"))
        metrics["walk_forward_expectancy_bps"] = walk_forward
        if walk_forward is None or walk_forward <= 0:
            issues.append(f"non_positive_walk_forward_bps:{walk_forward}")

        gate_by_horizon = DiagnosticsModule.dict_or_empty(db_diag.get("shadow_gate_by_horizon"))
        raw_gate = gate_by_horizon.get(str(model_horizon))
        if raw_gate is None:
            raw_gate = db_diag.get(f"shadow_gate_{model_horizon}m")
        if raw_gate is None:
            raw_gate = db_diag.get("shadow_gate_15m")
        gate = DiagnosticsModule.dict_or_empty(raw_gate)
        gate_total = int(gate.get("total_count") or 0)
        metrics["gate_total_count"] = gate_total
        if gate_total < int(self._app._settings.MODEL_GATE_CANARY_MIN_OBSERVATIONS):
            issues.append(f"insufficient_gate_observations:{gate_total}")
        gate_lift = DiagnosticsModule.float_or_none(gate.get("lift_vs_all_bps"))
        metrics["gate_lift_vs_all_bps"] = gate_lift
        if gate_lift is None or gate_lift < float(self._app._settings.MODEL_GATE_CANARY_MIN_LIFT_BPS):
            issues.append(f"insufficient_gate_lift_bps:{gate_lift}")

        paper_by_horizon = DiagnosticsModule.dict_or_empty(db_diag.get("paper_pnl_by_horizon"))
        raw_paper = paper_by_horizon.get(str(model_horizon))
        if raw_paper is None:
            raw_paper = db_diag.get(f"paper_pnl_{model_horizon}m")
        if raw_paper is None:
            raw_paper = db_diag.get("paper_pnl_15m")
        paper = DiagnosticsModule.dict_or_empty(raw_paper)
        paper_gate = DiagnosticsModule.dict_or_empty(paper.get("model_gate"))
        paper_count = int(paper_gate.get("count") or 0)
        paper_total_bps = DiagnosticsModule.float_or_none(paper_gate.get("total_bps"))
        metrics["paper_gate_count"] = paper_count
        metrics["paper_gate_total_bps"] = paper_total_bps
        if paper_count < 20:
            issues.append(f"insufficient_paper_gate_trades:{paper_count}")
        if paper_total_bps is None or paper_total_bps <= 0:
            issues.append(f"non_positive_paper_gate_bps:{paper_total_bps}")

        return {
            "ready": not issues,
            "mode": self._app._settings.TRADING_MODE.value,
            "issues": issues,
            "metrics": metrics,
        }

    async def enforce_economic_readiness_for_active(self) -> None:
        """Fail closed before real-money modes when paper evidence is not ready."""
        assert self._app._settings is not None
        if self._app._settings.TRADING_MODE not in (TradingMode.CANARY_LIVE, TradingMode.LIVE):
            return
        if not self._app._settings.ECONOMIC_READINESS_REQUIRED_FOR_ACTIVE:
            log.warning(
                "economic_readiness_gate_disabled_for_active_mode",
                trading_mode=self._app._settings.TRADING_MODE.value,
            )
            return
        if self._app._trade_journal is None or not self._app._trade_journal.is_enabled:
            log.critical("economic_readiness_blocked", issues=["trade_journal_unavailable"])
            raise SystemExit(1)

        db_diag = await self._app._trade_journal.get_db_diagnostics()
        issues = self.economic_readiness_report(
            db_diag=db_diag,
            runtime_diag=self.get_snapshot(),
        )["issues"]
        if issues:
            log.critical(
                "economic_readiness_blocked",
                trading_mode=self._app._settings.TRADING_MODE.value,
                issues=issues,
            )
            raise SystemExit(1)
        log.info("economic_readiness_passed", trading_mode=self._app._settings.TRADING_MODE.value)

    def record(self, event: str) -> None:
        """Record a diagnostics event with the current timestamp."""
        self._app._diag_events.append((datetime.now(tz=UTC), event))

    def top_blocker_from_diag(self, diag: dict[str, Any], *, default: str) -> tuple[str, dict[str, int]]:
        """Return the most useful blocker label for operator diagnostics."""
        blockers = {
            "risk_rejected": int(diag.get("hour_risk_rejected") or 0),
            "risk_rejected:sizer_rejected": int(diag.get("hour_risk_sizer_rejected") or 0),
            "risk_rejected:min_notional": int(diag.get("hour_min_notional_rejected") or 0),
            "risk_rejected:exposure": int(diag.get("hour_risk_exposure_rejected") or 0),
            "risk_rejected:balance": int(diag.get("hour_risk_balance_rejected") or 0),
            "risk_rejected:spread_or_atr": int(diag.get("hour_risk_market_filter_rejected") or 0),
            "startup_warmup": int(diag.get("hour_skipped_startup_warmup") or 0),
            "rate_limit": int(diag.get("hour_skipped_rate_limit") or 0),
            "model_gate_blocked": int(diag.get("hour_model_gate_canary_blocked") or 0),
            "net_edge_rejected": int(diag.get("hour_net_edge_rejected") or 0),
            "post_signal_size_rejected": int(diag.get("hour_signal_qty_adjustment_rejected") or 0),
            "spread_rejected": int(diag.get("hour_spread_rejected") or 0),
            "scalp_net_edge_rejected": int(diag.get("hour_scalp_net_edge_rejected") or 0),
            "imbalance_rejected": int(diag.get("hour_imbalance_rejected") or 0),
            "bucket_blocked": int(diag.get("hour_bucket_blocked") or 0),
            "symbol_side_blocked": int(diag.get("hour_symbol_side_blocked") or 0),
            "trend_confirmation_blocked": int(diag.get("hour_trend_confirmation_blocked") or 0),
            "shadow_loss_guard_blocked": int(diag.get("hour_shadow_loss_guard_blocked") or 0),
        }
        top_blocker = (
            max(blockers, key=lambda k: (blockers[k], 1 if ":" in k else 0)) if any(blockers.values()) else default
        )
        return top_blocker, blockers

    def check_zero_trading(self) -> None:
        """Warn (never block) when signals flow but nothing executes for an hour.

        Helps catch over-tight filters: model gate, net edge, spread, risk.
        Throttled to one warning per 10 minutes.
        """
        assert self._app._settings is not None
        now = datetime.now(tz=UTC)
        if self._app._last_zero_trading_warn_at is not None:
            if (now - self._app._last_zero_trading_warn_at).total_seconds() < 600:
                return
        diag = self.get_snapshot()
        signals = int(diag.get("hour_signals_emitted") or 0)
        placed = int(diag.get("hour_order_placed") or 0)
        shadow_would_place = int(diag.get("hour_shadow_order_would_be_placed") or 0)
        if signals >= max(1, self._app._settings.MIN_SIGNALS_PER_HOUR) and placed == 0 and shadow_would_place == 0:
            if self._app._execution_engine is not None and self._app._execution_engine.is_in_warmup():
                log.info(
                    "zero_trading.suppressed_warmup",
                    hour_signals=signals,
                    warmup_seconds_remaining=round(self._app._execution_engine.warmup_seconds_remaining(), 1),
                )
                return

            self._app._last_zero_trading_warn_at = now
            top_blocker, blockers = self.top_blocker_from_diag(diag, default="unknown")
            log.warning(
                "zero_trading.detected",
                hour_signals=signals,
                hour_orders_placed=placed,
                top_blocker=top_blocker,
                blockers=blockers,
                auto_soften_enabled=self._app._settings.AUTO_SOFTEN_FILTERS_ENABLED,
            )

    def runtime_candle_readiness_counts(self) -> dict[str, int]:
        """In-memory candle counts when Postgres diagnostics are slow or unavailable."""
        if self._app._candle_store is None:
            return {}
        symbols = self._app._screener.active_symbols if self._app._screener is not None else []
        if not symbols:
            return {}
        targets = {"1": 1000, "5": 200, "15": 200, "60": 100}
        counts: dict[str, int] = {}
        for interval, target in targets.items():
            total = sum(self._app._candle_store.count(symbol, interval, confirmed_only=True) for symbol in symbols)
            counts[interval] = min(target, total)
        return counts

    def merge_db_fallbacks(self, diag: dict[str, Any]) -> None:
        """Fill gaps in DB diagnostics from live runtime state (WS candle store, ML registry)."""
        runtime_candles = self.runtime_candle_readiness_counts()
        if runtime_candles:
            diag["runtime_candles_by_interval"] = runtime_candles
            db_candles = dict(diag.get("candles_by_interval") or {})
            merged = dict(db_candles)
            for interval, runtime_count in runtime_candles.items():
                if int(runtime_count) > int(merged.get(interval) or 0):
                    merged[interval] = int(runtime_count)
            if merged:
                diag["candles_by_interval"] = merged
                if db_candles and merged != db_candles:
                    diag["candles_source"] = "db_with_runtime_fallback"
                elif not db_candles:
                    diag["candles_source"] = "runtime_fallback"
        if self._app._last_confirmed_candle_at is not None and not diag.get("latest_candle_1m"):
            diag["latest_candle_1m"] = self._app._last_confirmed_candle_at
            diag["last_confirmed_candle_age_s"] = max(
                0.0,
                (datetime.now(tz=UTC) - self._app._last_confirmed_candle_at).total_seconds(),
            )
        if self._app._model_registry is not None:
            challenger = self._app._model_registry.challenger
            champion = self._app._model_registry.champion
            latest = diag.get("latest_model_version") or {}
            if challenger is not None and not latest.get("version"):
                diag["latest_model_version"] = {
                    "version": challenger.version,
                    "status": "SHADOW_CHALLENGER",
                    "training_samples": challenger.training_samples,
                    "metrics": getattr(challenger, "metrics", {}) or {},
                }
            if champion is not None:
                diag["active_model_version"] = {
                    "version": champion.version,
                    "status": "CHAMPION",
                    "training_samples": champion.training_samples,
                    "metrics": getattr(champion, "metrics", {}) or {},
                }
            elif challenger is not None and not (diag.get("active_model_version") or {}).get("version"):
                diag["active_model_version"] = diag.get("latest_model_version", {})

    def get_snapshot(self) -> dict[str, Any]:
        """Return a diagnostics snapshot for the /diagnostics Telegram command."""
        now = datetime.now(tz=UTC)
        cutoff = now - _DIAG_WINDOW

        # Count events in the last hour
        hour_counts: dict[str, int] = {}
        for ts, event in self._app._diag_events:
            if ts >= cutoff:
                hour_counts[event] = hour_counts.get(event, 0) + 1

        ws_age: float | None = None
        if self._app._health_checker is not None and self._app._health_checker._last_ws_message_at is not None:
            ws_age = (now - self._app._health_checker._last_ws_message_at).total_seconds()
        confirmed_age: float | None = None
        if self._app._last_confirmed_candle_at is not None:
            confirmed_age = (now - self._app._last_confirmed_candle_at).total_seconds()
        telegram_health: dict[str, Any] = {}
        if self._app._telegram_bot is not None and hasattr(self._app._telegram_bot, "health_snapshot"):
            try:
                telegram_health = self._app._telegram_bot.health_snapshot()
            except Exception as exc:
                telegram_health = {"enabled": True, "error": str(exc)}

        from trader.monitoring.deploy_info import get_deploy_info

        return {
            "deploy": get_deploy_info(),
            "subscribe_watchdog": (self._app._subscribe_watchdog.to_dict() if self._app._subscribe_watchdog is not None else {}),
            "last_strategy_loop_at": self._app._last_strategy_loop_at.isoformat() if self._app._last_strategy_loop_at else None,
            "last_ws_message_age_s": ws_age,
            "last_confirmed_candle_age_s": confirmed_age,
            "runtime_candles_by_interval": self.runtime_candle_readiness_counts(),
            "telegram": telegram_health,
            "active_symbols": (self._app._screener.active_symbols if self._app._screener is not None else list(_SYMBOLS)),
            "open_positions": (
                list(self._app._execution_engine._open_positions.keys()) if self._app._execution_engine is not None else []
            ),
            "portfolio_heat_pct": (
                float(self._app._exposure_tracker.total_exposure_pct) if self._app._exposure_tracker is not None else None
            ),
            "hour_signals_emitted": hour_counts.get("signals_emitted", 0),
            "hour_risk_rejected": hour_counts.get("risk_rejected", 0),
            "hour_risk_sizer_rejected": hour_counts.get("risk_sizer_rejected", 0),
            "hour_risk_exposure_rejected": hour_counts.get("risk_exposure_rejected", 0),
            "hour_risk_balance_rejected": hour_counts.get("risk_balance_rejected", 0),
            "hour_risk_market_filter_rejected": hour_counts.get("risk_market_filter_rejected", 0),
            "hour_api_rejected": hour_counts.get("api_rejected", 0),
            "hour_min_notional_rejected": hour_counts.get("post_multiplier_min_notional_rejected", 0),
            "hour_skipped_open_position": hour_counts.get("skipped_open_position", 0),
            "hour_skipped_entry_cooldown": hour_counts.get("skipped_entry_cooldown", 0),
            "hour_skipped_failure_cooldown": hour_counts.get("skipped_failure_cooldown", 0),
            "hour_model_gate_canary_blocked": hour_counts.get("model_gate_canary_blocked", 0),
            "hour_ml_replacement": hour_counts.get("ml_replacement", 0),
            "hour_rule_fallback_signals": hour_counts.get("rule_fallback_signal", 0),
            "hour_spread_rejected": hour_counts.get("spread_rejected", 0),
            "hour_scalp_net_edge_rejected": hour_counts.get("scalp_net_edge_rejected", 0),
            "hour_imbalance_rejected": hour_counts.get("imbalance_rejected", 0),
            "hour_bucket_blocked": hour_counts.get("bucket_blocked", 0),
            "hour_symbol_side_blocked": hour_counts.get("symbol_side_blocked", 0),
            "hour_trend_confirmation_blocked": hour_counts.get("trend_confirmation_blocked", 0),
            "drift_status": self._app._drift_status,
            "strategy_cycle_ms": round(self._app._last_strategy_cycle_ms, 1),
            "last_retention_run_at": (
                self._app._last_retention_run_at.isoformat() if self._app._last_retention_run_at is not None else None
            ),
            "hour_shadow_loss_guard_blocked": hour_counts.get("shadow_loss_guard_blocked", 0),
            # Engine-level counters (cumulative since startup, read from execution engine)
            "hour_skipped_pending_entries": (
                self._app._execution_engine.get_diag_counts().get("skipped_pending_entries", 0)
                if self._app._execution_engine is not None
                else 0
            ),
            "hour_skipped_startup_warmup": (
                self._app._execution_engine.get_diag_counts().get("skipped_startup_warmup", 0)
                if self._app._execution_engine is not None
                else 0
            ),
            "hour_skipped_rate_limit": (
                self._app._execution_engine.get_diag_counts().get("skipped_rate_limit", 0)
                if self._app._execution_engine is not None
                else 0
            ),
            "hour_signal_qty_adjustment_rejected": (
                self._app._execution_engine.get_diag_counts().get("signal_qty_adjustment_rejected", 0)
                if self._app._execution_engine is not None
                else 0
            ),
            "hour_order_placed": (
                self._app._execution_engine.get_diag_counts().get("order_placed", 0)
                if self._app._execution_engine is not None
                else 0
            ),
            "hour_shadow_order_would_be_placed": (
                self._app._execution_engine.get_diag_counts().get("shadow_order_would_be_placed", 0)
                if self._app._execution_engine is not None
                else 0
            ),
            "hour_order_failed": (
                self._app._execution_engine.get_diag_counts().get("order_failed", 0)
                if self._app._execution_engine is not None
                else 0
            ),
            "hour_net_edge_rejected": (
                self._app._execution_engine.get_diag_counts().get("net_edge_rejected", 0)
                if self._app._execution_engine is not None
                else 0
            ),
            "hour_no_take_profit_rejected": (
                self._app._execution_engine.get_diag_counts().get("no_tp_rejected", 0)
                if self._app._execution_engine is not None
                else 0
            ),
            "hour_fee_rate_unavailable_rejected": (
                self._app._execution_engine.get_diag_counts().get("fee_unavailable_rejected", 0)
                if self._app._execution_engine is not None
                else 0
            ),
            # Pending entry details for /diagnostics "why no trades" display
            **(
                self._app._execution_engine.pending_entry_diagnostics()
                if self._app._execution_engine is not None
                else {
                    "pending_entry_count": 0,
                    "pending_entry_ids": [],
                    "pending_entry_symbols": [],
                    "oldest_pending_age_s": None,
                }
            ),
            "model": {
                "last_training": self._app._last_training_message,
                "training_samples": (
                    self._app._model_registry.champion.training_samples
                    if self._app._model_registry is not None and self._app._model_registry.champion is not None
                    else (
                        self._app._model_registry.challenger.training_samples
                        if self._app._model_registry is not None and self._app._model_registry.challenger is not None
                        else 0
                    )
                ),
                "champion_version": (
                    self._app._model_registry.champion.version
                    if self._app._model_registry is not None and self._app._model_registry.champion is not None
                    else "none"
                ),
                "challenger_version": (
                    self._app._model_registry.challenger.version
                    if self._app._model_registry is not None and self._app._model_registry.challenger is not None
                    else "none"
                ),
                "quality": (
                    str(
                        (
                            getattr(self._app._model_registry.champion, "metrics", {})
                            if self._app._model_registry is not None and self._app._model_registry.champion is not None
                            else (
                                getattr(self._app._model_registry.challenger, "metrics", {})
                                if self._app._model_registry is not None and self._app._model_registry.challenger is not None
                                else {}
                            )
                        ).get("quality")
                        or self._app._model_gate_quality.get("quality")
                        or "n/a"
                    )
                ),
                "lift_bps": (
                    (
                        getattr(self._app._model_registry.champion, "metrics", {})
                        if self._app._model_registry is not None and self._app._model_registry.champion is not None
                        else (
                            getattr(self._app._model_registry.challenger, "metrics", {})
                            if self._app._model_registry is not None and self._app._model_registry.challenger is not None
                            else {}
                        )
                    ).get("lift_bps")
                ),
                "walk_forward_expectancy": (
                    (
                        getattr(self._app._model_registry.champion, "metrics", {})
                        if self._app._model_registry is not None and self._app._model_registry.champion is not None
                        else (
                            getattr(self._app._model_registry.challenger, "metrics", {})
                            if self._app._model_registry is not None and self._app._model_registry.challenger is not None
                            else {}
                        )
                    ).get("walk_forward_expectancy_bps", "n/a")
                ),
                "drift_status": self._app._drift_status.get("status", "n/a"),
                "drift_psi": self._app._drift_status.get("psi"),
                "gate_quality": self._app._model_gate_quality.get("quality"),
            },
        }

