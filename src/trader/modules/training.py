"""training."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import html
import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from trader.monitoring.logging import get_logger
from trader.runtime.constants import (
    _TRAINING_HEARTBEAT_SECONDS,
    _TRAINING_TIMEOUT_SECONDS,
)
from trader.runtime.module import ModuleTaskMixin

log = get_logger(__name__)


class TrainingModule(ModuleTaskMixin):
    name = "training"

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self._schema_milestone_notified: set[str] = set()

    def spawn_background_tasks(self, tasks: list[asyncio.Task[object]]) -> None:
        self._spawn(tasks, self.run_bucket_stats_refresher(), "bucket-stats")
        self._spawn(tasks, self.run_auto_model_trainer(), "auto-model-trainer")
        self._spawn(tasks, self.run_auto_model_promoter(), "auto-model-promoter")
        self._spawn(tasks, self.run_model_progress_reporter(), "model-progress-reporter")

    async def run_auto_model_trainer(self) -> None:
        """Automatically train a shadow challenger when enough new labels accumulate."""
        assert self._app._settings is not None
        if not self._app._settings.MODEL_AUTO_TRAIN_ENABLED:
            log.info("model_auto_training.disabled")
            return

        check_seconds = max(60, int(self._app._settings.MODEL_AUTO_TRAIN_CHECK_SECONDS))
        min_samples = max(50, int(self._app._settings.MODEL_AUTO_TRAIN_MIN_SAMPLES))
        schema_change_min_samples = max(50, int(self._app._settings.MODEL_AUTO_TRAIN_SCHEMA_CHANGE_MIN_SAMPLES))
        increment_samples = max(1, int(self._app._settings.MODEL_AUTO_TRAIN_INCREMENT_SAMPLES))
        min_train_interval_s = max(0, int(getattr(self._app._settings, "MODEL_AUTO_TRAIN_MIN_INTERVAL_SECONDS", 3600)))
        horizon = int(self._app._settings.MODEL_AUTO_TRAIN_HORIZON_MINUTES)
        label_bps = float(self._app._settings.MODEL_AUTO_TRAIN_LABEL_BPS)

        # First check fires after a short grace period so the trade journal has time to
        # connect. Subsequent checks use the full check_seconds interval. Without this,
        # a 300s initial wait means the trainer never runs when deployments restart the
        # bot more frequently than check_seconds.
        first_run = True

        while not self._app._shutdown_event.is_set():
            wait_seconds = min(60, check_seconds) if first_run else check_seconds
            first_run = False
            try:
                await asyncio.wait_for(self._app._shutdown_event.wait(), timeout=wait_seconds)
                break
            except TimeoutError:
                pass

            if self._app._training_task is not None and not self._app._training_task.done():
                continue

            if (
                getattr(self._app._settings, "STARTER_DEFER_TRAINING_UNDER_LOAD", True)
                and self._app._runtime_under_memory_pressure()
            ):
                log.info("model_auto_training.deferred", reason="load_governor_reduced")
                continue

            # Back off for 30 minutes after a training failure to avoid an infinite
            # retry loop (e.g. OOM kill causes returncode -9, then latest_samples=0
            # looks like a fresh trigger on the very next check cycle).
            _training_failure_cooldown = 1800.0
            if self._app._training_failed_at is not None:
                elapsed_since_failure = time.monotonic() - self._app._training_failed_at
                if elapsed_since_failure < _training_failure_cooldown:
                    remaining = int(_training_failure_cooldown - elapsed_since_failure)
                    log.info(
                        "model_auto_training.failure_cooldown",
                        cooldown_remaining_s=remaining,
                    )
                    continue

            if self._app._trade_journal is None:
                log.info("model_auto_training.waiting", reason="trade_journal_not_started")
                continue
            if not self._app._trade_journal.is_enabled:
                await self._app._trade_journal.reconnect_if_needed()
                if not self._app._trade_journal.is_enabled:
                    log.info(
                        "model_auto_training.waiting",
                        reason="trade_journal_unavailable",
                    )
                    continue

            try:
                diag = await self._app._trade_journal.get_db_diagnostics()
                self._app._update_model_gate_quality_from_diag(diag)
                drift_summary = await self._app._evaluate_feature_drift()
                training_by_horizon = diag.get("training_eligible_by_horizon", {}) or {}
                newest_schema_by_horizon = diag.get("newest_training_schema_by_horizon", {}) or {}
                preferred_horizon = horizon
                latest_model = diag.get("latest_model_version", {}) or {}
                latest_run = diag.get("latest_training_run", {}) or {}
                latest_run_status = str(latest_run.get("status") or "").upper()
                latest_run_samples = int(latest_run.get("sample_count") or 0) if latest_run_status == "COMPLETED" else 0
                actual_latest_samples = int(
                    latest_model.get("actual_training_samples", latest_model.get("training_samples", 0)) or 0
                )
                latest_success_samples = max(actual_latest_samples, latest_run_samples)
                from trader.training.auto_train import TrainableSnapshot, resolve_training_horizon

                horizon_snapshots = [
                    TrainableSnapshot(int(horizon_key), int(count or 0))
                    for horizon_key, count in training_by_horizon.items()
                    if str(horizon_key).isdigit()
                ]
                initial_chosen = (
                    resolve_training_horizon(horizon_snapshots, preferred_horizon, min_samples=min_samples)
                    if latest_success_samples == 0
                    else None
                )
                active_horizon = initial_chosen.horizon_minutes if initial_chosen is not None else preferred_horizon
                horizon_schema = newest_schema_by_horizon.get(str(active_horizon), {}) or {}
                trainable = int(
                    horizon_schema.get(
                        "best_schema_count",
                        training_by_horizon.get(
                            str(active_horizon),
                            diag.get("training_eligible_15m", diag.get("labelled_samples_15m", 0)),
                        ),
                    )
                    or 0
                )
                if initial_chosen is not None:
                    trainable = initial_chosen.sample_count
                trainable_filtered_total = int(
                    (diag.get("training_filtered_total_by_horizon", {}) or {}).get(str(active_horizon), 0) or 0
                )
                compatible_latest_samples = int(
                    latest_model.get("training_samples_compatible", latest_model.get("training_samples", 0)) or 0
                )

                # Schema-mismatch trigger: if current model uses an outdated feature schema,
                # fire as soon as the new schema has min_samples — don't wait for increment_samples.
                # This eliminates the silent multi-hour window where predict() always returns None.
                newest_schema_hash = str(
                    horizon_schema.get("feature_schema_hash", diag.get("newest_training_schema_hash", "")) or ""
                )
                newest_schema_samples = int(
                    horizon_schema.get("sample_count", diag.get("newest_training_schema_samples", 0)) or 0
                )
                current_schema_hash = str(latest_model.get("feature_schema_hash", "") or "")
                schema_mismatch = bool(
                    newest_schema_hash and current_schema_hash and newest_schema_hash != current_schema_hash
                )
                label_schema_compatible = bool(latest_model.get("schema_compatible", True))
                schema_incompatible = bool(latest_model) and not label_schema_compatible
                label_schema_mismatch = schema_incompatible
                incompatible_reason = None
                if schema_incompatible:
                    if schema_mismatch and not label_schema_compatible:
                        incompatible_reason = "feature_and_label_schema"
                    elif schema_mismatch:
                        incompatible_reason = "feature_schema"
                    else:
                        incompatible_reason = "label_schema"
                bypass_train_cooldown = schema_incompatible or (
                    schema_mismatch and newest_schema_samples >= schema_change_min_samples
                )
                if min_train_interval_s > 0 and latest_success_samples > 0 and not bypass_train_cooldown:
                    latest_finished_at = latest_model.get("training_finished_at") or latest_model.get("created_at")
                    if latest_finished_at is None:
                        latest_finished_at = latest_run.get("finished_at")
                    if isinstance(latest_finished_at, str):
                        try:
                            latest_finished_at = datetime.fromisoformat(latest_finished_at.replace("Z", "+00:00"))
                        except ValueError:
                            latest_finished_at = None
                    if isinstance(latest_finished_at, datetime):
                        if latest_finished_at.tzinfo is None:
                            latest_finished_at = latest_finished_at.replace(tzinfo=UTC)
                        age_s = (datetime.now(tz=UTC) - latest_finished_at.astimezone(UTC)).total_seconds()
                        if age_s < min_train_interval_s:
                            log.info(
                                "model_auto_training.success_cooldown",
                                cooldown_remaining_s=int(min_train_interval_s - age_s),
                                latest_actual_samples=actual_latest_samples,
                                latest_run_samples=latest_run_samples,
                                latest_compatible_samples=compatible_latest_samples,
                                schema_mismatch=schema_mismatch,
                                label_schema_mismatch=label_schema_mismatch,
                            )
                            continue

                enough_initial = initial_chosen is not None
                enough_increment = (
                    not schema_mismatch
                    and not label_schema_mismatch
                    and compatible_latest_samples > 0
                    and (trainable - compatible_latest_samples) >= increment_samples
                )
                enough_schema_change = schema_mismatch and newest_schema_samples >= schema_change_min_samples
                enough_label_schema_change = label_schema_mismatch and trainable >= min_samples

                enough_weak_retrain = bool(
                    getattr(self._app._settings, "MODEL_AUTO_TRAIN_RETRAIN_IF_WEAK", True)
                    and not label_schema_mismatch
                    and compatible_latest_samples > 0
                    and str(self._app._model_gate_quality.get("quality") or "WEAK").upper() in {"WEAK", ""}
                    and trainable >= min_samples
                )
                enough_drift_retrain = bool(
                    self._app._settings.MODEL_DRIFT_AUTO_RETRAIN
                    and drift_summary.get("drift_detected")
                    and trainable >= min_samples
                )
                if not (
                    enough_initial
                    or enough_increment
                    or enough_schema_change
                    or enough_label_schema_change
                    or enough_weak_retrain
                    or enough_drift_retrain
                ):
                    if schema_incompatible and trainable > 0:
                        log.warning(
                            "model_auto_training.schema_incompatible_accumulating",
                            trainable=trainable,
                            min_samples=min_samples,
                            compatible_latest_samples=compatible_latest_samples,
                            incompatible_reason=incompatible_reason,
                            feature_schema_mismatch=schema_mismatch,
                            label_schema_mismatch=not label_schema_compatible,
                        )
                    elif schema_mismatch and newest_schema_samples > 0:
                        log.warning(
                            "model_auto_training.schema_mismatch_accumulating",
                            current_schema=current_schema_hash,
                            newest_schema=newest_schema_hash,
                            newest_schema_samples=newest_schema_samples,
                            schema_change_min_samples=schema_change_min_samples,
                            min_samples_needed=schema_change_min_samples,
                        )
                        await self._maybe_notify_schema_migration_progress(
                            newest_schema_hash=newest_schema_hash,
                            newest_schema_samples=newest_schema_samples,
                            schema_change_min_samples=schema_change_min_samples,
                            current_schema_hash=current_schema_hash,
                        )
                    else:
                        pool_breakdown = diag.get("training_pool_breakdown", {}) or {}
                        log.info(
                            "model_auto_training.waiting",
                            reason="threshold_not_met",
                            trainable=trainable,
                            trainable_filtered_total=trainable_filtered_total,
                            min_samples=min_samples,
                            preferred_horizon_minutes=preferred_horizon,
                            active_horizon_minutes=active_horizon,
                            training_eligible_by_horizon=training_by_horizon,
                            latest_success_samples=latest_success_samples,
                            newest_schema_samples=newest_schema_samples,
                            schema_mismatch=schema_mismatch,
                            label_schema_mismatch=label_schema_mismatch,
                            candle_baseline=int(pool_breakdown.get("candle_baseline_active_schema", 0) or 0),
                            scalp_micro=int(pool_breakdown.get("scalp_micro_v1_active_schema", 0) or 0),
                        )
                    continue

                trigger_reason = (
                    "drift"
                    if enough_drift_retrain
                    else (
                        "label_schema_change"
                        if enough_label_schema_change
                        else (
                            "schema_change"
                            if enough_schema_change
                            else (
                                "weak_retrain"
                                if enough_weak_retrain
                                else ("initial" if enough_initial else "increment")
                            )
                        )
                    )
                )
                effective_min_samples = schema_change_min_samples if enough_schema_change else min_samples
                if trainable < effective_min_samples:
                    log.warning(
                        "model_auto_training.preflight_blocked",
                        trainable=trainable,
                        trainable_filtered_total=trainable_filtered_total,
                        min_samples=effective_min_samples,
                        horizon_minutes=active_horizon,
                        label_schema_mismatch=label_schema_mismatch,
                    )
                    continue

                train_horizon = active_horizon if enough_initial else preferred_horizon
                msg = await self._app._start_model_training(effective_min_samples, train_horizon, label_bps)
                log.info(
                    "model_auto_training.started",
                    trainable=trainable,
                    horizon_minutes=train_horizon,
                    preferred_horizon_minutes=preferred_horizon,
                    latest_samples=actual_latest_samples,
                    latest_run_samples=latest_run_samples,
                    compatible_latest_samples=compatible_latest_samples,
                    min_samples=effective_min_samples,
                    increment_samples=increment_samples,
                    trigger_reason=trigger_reason,
                    schema_mismatch=schema_mismatch,
                    label_schema_mismatch=label_schema_mismatch,
                )
                if self._app._telegram_bot is not None:
                    await self._app._telegram_bot.notify(
                        "🤖 <b>Auto-training triggered</b>\n"
                        f"trainable_{train_horizon}m=<code>{trainable}</code>, "
                        f"latest_model_samples=<code>{actual_latest_samples}</code>, "
                        f"latest_run_samples=<code>{latest_run_samples}</code>, "
                        f"compatible=<code>{compatible_latest_samples}</code>\n"
                        f"{msg}"
                    )
            except Exception as exc:
                log.warning("model_auto_training.failed", error=str(exc))

    async def _maybe_notify_schema_migration_progress(
        self,
        *,
        newest_schema_hash: str,
        newest_schema_samples: int,
        schema_change_min_samples: int,
        current_schema_hash: str,
    ) -> None:
        """Telegram milestones while a new feature schema accumulates training samples."""
        milestones = (
            10,
            25,
            max(1, schema_change_min_samples // 2),
            schema_change_min_samples,
        )
        for milestone in milestones:
            if newest_schema_samples < milestone:
                continue
            key = f"{newest_schema_hash}:{milestone}"
            if key in self._schema_milestone_notified:
                continue
            self._schema_milestone_notified.add(key)
            if self._app._telegram_bot is None:
                return
            ready = newest_schema_samples >= schema_change_min_samples
            await self._app._telegram_bot.notify(
                "🧬 <b>Schema migration</b>\n"
                f"Модель: <code>{current_schema_hash[:8] or '—'}</code> → "
                f"<code>{newest_schema_hash[:8]}</code>\n"
                f"Новых samples: <code>{newest_schema_samples}</code> / "
                f"<code>{schema_change_min_samples}</code>"
                + ("\n✅ Порог достигнут — auto-train скоро запустится." if ready else "")
            )
            return

    async def run_auto_model_promoter(self) -> None:
        """Promote the best eligible challenger and roll back degraded champions."""
        assert self._app._settings is not None
        if not self._app._settings.MODEL_AUTO_PROMOTE_ENABLED:
            log.info("model_auto_promote.disabled")
            return

        from trader.ml.auto_promotion import AutoPromotionConfig, AutoPromotionEngine

        config = AutoPromotionConfig.from_settings(self._app._settings)
        check_seconds = config.check_seconds
        last_monitor_at = datetime.now(tz=UTC) - timedelta(seconds=config.monitor_seconds)

        async def _reload_registry() -> None:
            if self._app._model_registry is not None:
                await self._app._model_registry.load_active_model()

        while not self._app._shutdown_event.is_set():
            try:
                await asyncio.wait_for(self._app._shutdown_event.wait(), timeout=check_seconds)
                break
            except TimeoutError:
                pass

            if self._app._trade_journal is None or not self._app._trade_journal.is_enabled:
                continue

            try:
                engine = AutoPromotionEngine(
                    trade_journal=self._app._trade_journal,
                    config=config,
                    reload_registry=_reload_registry,
                )
                now = datetime.now(tz=UTC)
                if (now - last_monitor_at).total_seconds() >= config.monitor_seconds:
                    rollback = await engine.rollback_if_needed()
                    last_monitor_at = now
                    if rollback.rollback and self._app._telegram_bot is not None:
                        await self._app._telegram_bot.notify(
                            "↩️ <b>Авто-откат модели</b>\n"
                            f"Было: <code>{rollback.champion_version}</code>\n"
                            f"Стало: <code>{rollback.rollback_version}</code>\n"
                            f"Причины: <code>{html.escape(', '.join(rollback.reasons))}</code>"
                        )

                challenger_version = await engine.best_challenger()
                if not challenger_version:
                    log.debug("model_auto_promote.waiting", reason="no_eligible_challenger")
                    continue

                decision = await engine.promote(challenger_version)
                if not decision.promote:
                    log.info(
                        "model_auto_promote.waiting",
                        version=challenger_version,
                        reasons=decision.reasons,
                        metrics=decision.metrics,
                    )
                    continue

                # Promotion and Canary are separate safety steps. This does not
                # auto-enable MODEL_GATE_CANARY_ENABLED.

                if self._app._telegram_bot is not None:
                    await self._app._telegram_bot.notify(
                        f"🤖 <b>Авто-промоут</b>\n"
                        f"Версия: <code>{challenger_version}</code>\n"
                        f"Сигналов: <code>{decision.metrics.get('total_count')}</code> | "
                        f"Lift: <code>{float(decision.metrics.get('lift_bps') or 0.0):+.2f} bps</code>\n"
                        f"WF: <code>{float(decision.metrics.get('wf_bps') or 0.0):+.2f} bps</code>, "
                        f"p-value: <code>{float(decision.metrics.get('bootstrap_p_value') or 0.0):.4f}</code>\n"
                        f"Предыдущий чемпион: <code>{decision.champion_version or 'none'}</code>"
                    )
            except Exception as exc:
                log.warning("model_auto_promote.failed", error=str(exc))

    async def run_model_progress_reporter(self) -> None:
        """Send an hourly Telegram report on model training progress and promotion readiness."""
        assert self._app._settings is not None
        if self._app._telegram_bot is None:
            return

        report_interval = 3600  # 1 hour

        while not self._app._shutdown_event.is_set():
            try:
                await asyncio.wait_for(self._app._shutdown_event.wait(), timeout=report_interval)
                break
            except TimeoutError:
                pass

            if self._app._trade_journal is None or not self._app._trade_journal.is_enabled:
                continue

            try:
                from trader.training.labels import active_label_schema_version

                diag = await self._app._trade_journal.get_db_diagnostics()
                self._app._update_model_gate_quality_from_diag(diag)

                latest_model = diag.get("latest_model_version", {}) or {}
                champion_wf_bps = await self._app._get_champion_walk_forward_bps()

                version = str(latest_model.get("version", "—") or "—")
                status = str(latest_model.get("status", "—") or "—")
                training_samples = int(latest_model.get("training_samples", 0) or 0)
                actual_training_samples = int(latest_model.get("actual_training_samples", training_samples) or 0)
                compatible_training_samples = int(
                    latest_model.get("training_samples_compatible", training_samples) or 0
                )
                report_horizon = int(self._app._settings.MODEL_AUTO_TRAIN_HORIZON_MINUTES)
                training_by_horizon = diag.get("training_eligible_by_horizon", {}) or {}
                filtered_by_horizon = diag.get("training_filtered_total_by_horizon", {}) or {}
                min_train_samples = max(50, int(self._app._settings.MODEL_AUTO_TRAIN_MIN_SAMPLES))
                labelled = int(training_by_horizon.get(str(report_horizon), diag.get("labelled_samples_15m", 0)) or 0)
                pool_breakdown = diag.get("training_pool_breakdown", {}) or {}
                candle_pool = int(pool_breakdown.get("candle_baseline_active_schema", 0) or 0) + int(
                    pool_breakdown.get("candle_sampler_v1_active_schema", 0) or 0
                )
                scalp_pool = int(pool_breakdown.get("scalp_micro_v1_active_schema", 0) or 0)
                horizon_parts = [
                    f"{horizon_key}m: {int(training_by_horizon.get(horizon_key, 0) or 0)}"
                    for horizon_key in ("5", "15")
                    if horizon_key in training_by_horizon or horizon_key in filtered_by_horizon
                ]
                horizon_summary = ", ".join(horizon_parts) if horizon_parts else f"{report_horizon}m: {labelled}"
                no_trained_model = version in {"", "—", "none"} or actual_training_samples == 0

                # Fetch gate stats for the latest model (challenger), not the active champion.
                # get_db_diagnostics.shadow_gate_15m tracks the active/champion model which can
                # be a different version — producing a misleading "0 signals" for the challenger.
                gate: dict[str, Any] = {}
                gate_events: dict[str, Any] = {}
                if version and version != "—" and self._app._trade_journal is not None:
                    report_label_schema = active_label_schema_version(
                        use_tpsl_exit=bool(self._app._settings.MODEL_LABEL_USE_TPSL_EXIT)
                    )
                    gate = await self._app._trade_journal.get_shadow_gate_stats(
                        version,
                        report_horizon,
                        report_label_schema,
                    )
                    gate_event_counter = getattr(self._app._trade_journal, "get_shadow_gate_event_counts", None)
                    if gate_event_counter is not None:
                        gate_events = await gate_event_counter(
                            version,
                            report_horizon,
                            report_label_schema,
                        )
                else:
                    gate = diag.get("shadow_gate_15m", {}) or {}

                resolved_count = int(gate.get("total_count", 0) or 0)
                observed_count = int(gate_events.get("total_count", resolved_count) or 0)
                pending_count = int(gate_events.get("pending_count", 0) or 0)
                lift_bps = gate.get("lift_vs_all_bps")
                pass_precision = gate.get("pass_precision")

                newest_schema_by_horizon = diag.get("newest_training_schema_by_horizon", {}) or {}
                horizon_schema = newest_schema_by_horizon.get(str(report_horizon), {}) or {}
                newest_schema_hash = str(
                    horizon_schema.get("feature_schema_hash", diag.get("newest_training_schema_hash", "")) or ""
                )
                newest_schema_samples = int(
                    horizon_schema.get("sample_count", diag.get("newest_training_schema_samples", 0)) or 0
                )
                current_schema_hash = str(latest_model.get("feature_schema_hash", "") or "")
                schema_drift = bool(
                    newest_schema_hash and current_schema_hash and newest_schema_hash != current_schema_hash
                )
                _sc_min = max(50, int(self._app._settings.MODEL_AUTO_TRAIN_SCHEMA_CHANGE_MIN_SAMPLES))

                log.info(
                    "model_progress_reporter.stats",
                    challenger_version=version,
                    challenger_status=status,
                    training_samples=training_samples,
                    actual_training_samples=actual_training_samples,
                    compatible_training_samples=compatible_training_samples,
                    labelled_horizon=labelled,
                    training_horizon_minutes=report_horizon,
                    gate_total=observed_count,
                    gate_resolved=resolved_count,
                    gate_pending=pending_count,
                    gate_pass=gate_events.get("pass_count", gate.get("pass_count")),
                    gate_block=gate_events.get("block_count", gate.get("block_count")),
                    lift_bps=round(float(lift_bps), 3) if lift_bps is not None else None,
                    pass_precision=round(float(pass_precision), 3) if pass_precision is not None else None,
                    gate_schema_hash=gate.get("feature_schema_hash"),
                    model_schema=current_schema_hash[:8] if current_schema_hash else None,
                    newest_schema=newest_schema_hash[:8] if newest_schema_hash else None,
                    newest_schema_samples=newest_schema_samples,
                    schema_drift=schema_drift,
                    candle_sampler_total=self._app._candle_sampler_total,
                    candle_sampler_scored=self._app._candle_sampler_scored,
                    candle_sampler_no_model=self._app._candle_sampler_no_model,
                )

                min_signals = max(10, int(self._app._settings.MODEL_AUTO_PROMOTE_MIN_SIGNALS))
                min_lift = float(self._app._settings.MODEL_AUTO_PROMOTE_MIN_LIFT_BPS)
                min_wf_bps = float(self._app._settings.MODEL_AUTO_PROMOTE_MIN_WF_BPS)
                min_paper_gate = max(20, int(getattr(self._app._settings, "MODEL_MIN_PASS_COUNT_FOR_PROMOTION", 20)))
                max_paper_drawdown_bps = float(
                    getattr(
                        self._app._settings,
                        "MODEL_AUTO_PROMOTE_MAX_DRAWDOWN_BPS",
                        getattr(self._app._settings, "MODEL_CHAMPION_MAX_DRAWDOWN_BPS", 1500.0),
                    )
                )
                required_quality = str(self._app._settings.MODEL_AUTO_PROMOTE_MIN_QUALITY or "GOOD").upper()

                raw_metrics = latest_model.get("metrics")
                if isinstance(raw_metrics, str) and raw_metrics.strip():
                    try:
                        model_metrics = json.loads(raw_metrics)
                    except json.JSONDecodeError:
                        model_metrics = {}
                elif isinstance(raw_metrics, dict):
                    model_metrics = raw_metrics
                else:
                    model_metrics = {}
                challenger_wf_bps = model_metrics.get("walk_forward_expectancy_bps")
                if challenger_wf_bps is None:
                    challenger_wf_bps = model_metrics.get("walk_forward_bps")
                if challenger_wf_bps is None:
                    challenger_wf_bps = model_metrics.get("wf_mean_bps")
                challenger_wf_bps = float(challenger_wf_bps) if challenger_wf_bps is not None else None
                model_quality = str(model_metrics.get("quality") or "").upper()

                paper_gate_count = 0
                paper_gate_bps = 0.0
                paper_gate_drawdown_bps = 0.0
                if version and version != "—" and self._app._trade_journal is not None:
                    paper = await self._app._trade_journal.get_live_paper_gate_stats(
                        version,
                        horizon_minutes=report_horizon,
                        feature_schema_hash=current_schema_hash,
                    )
                    paper_gate_count = int(paper.get("count") or 0)
                    paper_gate_bps = float(paper.get("total_bps") or 0.0)
                    paper_gate_drawdown_bps = abs(float(paper.get("max_drawdown_bps") or 0.0))

                # Build promotion checklist
                def check(ok: bool, label: str) -> str:
                    return f"{'✅' if ok else '❌'} {label}"

                has_signals = resolved_count >= min_signals
                has_lift = lift_bps is not None and float(lift_bps) >= min_lift
                has_wf = challenger_wf_bps is not None and challenger_wf_bps >= min_wf_bps
                has_paper_drawdown = paper_gate_drawdown_bps <= max_paper_drawdown_bps
                has_paper_gate = paper_gate_count >= min_paper_gate and paper_gate_bps > 0 and has_paper_drawdown
                has_quality = bool(model_quality) and model_quality == required_quality
                beats_champion = lift_bps is not None and float(lift_bps) > champion_wf_bps
                is_challenger = status == "SHADOW_CHALLENGER"
                promotion_reasons: list[str] = []
                promotion_engine_allows = False
                promotion_engine_checked = False
                if is_challenger and version and version != "—" and self._app._trade_journal is not None:
                    try:
                        from trader.ml.auto_promotion import AutoPromotionConfig, AutoPromotionEngine

                        promotion_engine = AutoPromotionEngine(
                            trade_journal=self._app._trade_journal,
                            config=AutoPromotionConfig.from_settings(self._app._settings),
                        )
                        promotion_decision = await promotion_engine.should_promote(None, version)
                        promotion_engine_checked = True
                        promotion_engine_allows = promotion_decision.promote
                        promotion_reasons = list(promotion_decision.reasons)
                    except Exception as promo_exc:
                        promotion_reasons = [f"promotion_check_failed:{promo_exc}"]

                lift_str = f"{float(lift_bps):+.2f} bps" if lift_bps is not None else "н/д"
                wf_str = f"{challenger_wf_bps:+.2f} bps" if challenger_wf_bps is not None else "н/д"
                precision_str = f"{float(pass_precision) * 100:.1f}%" if pass_precision is not None else "н/д"
                paper_gate_str = (
                    f"{paper_gate_count} сделок, {paper_gate_bps:+.1f} bps, "
                    f"DD {paper_gate_drawdown_bps:.1f}/{max_paper_drawdown_bps:.0f} bps"
                )

                lines = [
                    "📊 <b>Прогресс модели</b>",
                    f"Версия: <code>{version}</code> [{status}]",
                    (
                        f"Обучено на: <code>{actual_training_samples}</code> примерах"
                        + (
                            f" | Совместимо: <code>{compatible_training_samples}</code>"
                            if compatible_training_samples != actual_training_samples
                            else ""
                        )
                        + f" | Доступно: <code>{horizon_summary}</code>"
                    ),
                    (
                        f"Пул (5m): candle <code>{candle_pool}</code>, scalp <code>{scalp_pool}</code>"
                        if candle_pool or scalp_pool
                        else ""
                    ),
                    (
                        f"Gate: <code>{resolved_count}</code> resolved / "
                        f"<code>{observed_count}</code> всего"
                        + (f" / <code>{pending_count}</code> ждёт outcome" if pending_count else "")
                    ),
                ]
                lines = [line for line in lines if line]

                if schema_drift:
                    lines.append(
                        f"⚠️ <b>Смена схемы фичей!</b> Модель: <code>{current_schema_hash[:8]}</code> → "
                        f"Новая: <code>{newest_schema_hash[:8]}</code> "
                        f"({newest_schema_samples}/{_sc_min} примеров)"
                    )

                lines += [
                    "",
                    "<b>Условия для авто-промоута:</b>",
                    check(is_challenger, f"Статус SHADOW_CHALLENGER → {status}"),
                    check(has_signals, f"Resolved GATE ≥ {min_signals} → сейчас {resolved_count}"),
                    check(has_lift, f"Lift ≥ {min_lift:+.1f} bps → сейчас {lift_str}"),
                    check(
                        has_paper_gate,
                        f"Paper GATE ≥ {min_paper_gate}, > 0 bps и DD в лимите → сейчас {paper_gate_str}",
                    ),
                    check(has_wf, f"Walk-forward ≥ {min_wf_bps:+.1f} bps → сейчас {wf_str}"),
                    check(has_quality, f"Quality = {required_quality} → сейчас {model_quality or 'н/д'}"),
                    check(
                        beats_champion,
                        f"Лучше чемпиона ({champion_wf_bps:+.2f} bps) → {lift_str}",
                    ),
                    "",
                    f"Точность GATE_PASS: <code>{precision_str}</code>",
                    f"Canary: <code>{'включён' if self._app._settings.MODEL_GATE_CANARY_ENABLED else 'выключен'}</code>",
                ]

                checklist_ready = all([is_challenger, has_signals, has_lift, has_wf, has_quality, beats_champion])

                if promotion_engine_checked and promotion_engine_allows:
                    lines.append("\n🟢 <b>Все условия выполнены — промоут скоро!</b>")
                elif promotion_engine_checked and not promotion_engine_allows:
                    safe_reasons = ", ".join(html.escape(reason) for reason in promotion_reasons[:4])
                    lines.append(f"\n⏳ Auto-promoter ждёт: <code>{safe_reasons}</code>")
                elif not is_challenger and status == "CHAMPION":
                    lines.append("\n🏆 Модель уже чемпион — ждём нового challenger после следующего обучения.")
                elif no_trained_model:
                    best_trainable = max(
                        (int(training_by_horizon.get(key, 0) or 0) for key in ("5", "15", "30", "60")),
                        default=labelled,
                    )
                    lines.append(
                        f"\n⏳ <b>Модель ещё не обучена.</b> Нужно ≥ <code>{min_train_samples}</code> "
                        f"примеров на одной схеме (сейчас лучший горизонт: <code>{best_trainable}</code>). "
                        "Авто-обучение запустится, когда порог будет достигнут."
                    )
                elif schema_drift:
                    lines.append(
                        f"\n⏳ Модель не обучена под текущую схему фичей. "
                        f"Авто-обучение запустится при {newest_schema_samples}/{_sc_min} примерах."
                    )
                else:
                    missing = []
                    if not has_signals:
                        missing.append(f"ещё {min_signals - resolved_count} resolved GATE")
                    if not has_lift:
                        missing.append("lift > 0")
                    if not has_paper_gate:
                        missing.append(
                            f"paper GATE ≥ {min_paper_gate} с > 0 bps и DD ≤ {max_paper_drawdown_bps:.0f} bps"
                        )
                    if not has_wf:
                        missing.append(f"walk-forward ≥ {min_wf_bps:+.1f} bps")
                    if not has_quality:
                        missing.append(f"quality={required_quality}")
                    if not beats_champion and has_lift:
                        missing.append(f"обогнать чемпиона на {champion_wf_bps - float(lift_bps or 0):+.2f} bps")
                    if checklist_ready and not self._app._settings.MODEL_AUTO_PROMOTE_ENABLED:
                        missing.append("включить MODEL_AUTO_PROMOTE_ENABLED")
                    elif checklist_ready and not promotion_engine_checked:
                        missing.append("проверка auto-promoter")
                    lines.append(f"\n⏳ Не хватает: {', '.join(missing)}")

                await self._app._telegram_bot.notify("\n".join(lines))

            except Exception as exc:
                log.debug("model_progress_reporter.failed", error=str(exc))

    async def run_model_training_all(self) -> None:
        """Run training sequentially for all horizons using all available labeled data."""
        horizons = [5, 15, 30, 60]
        assert self._app._settings is not None
        label_bps = float(self._app._settings.MODEL_AUTO_TRAIN_LABEL_BPS)
        min_samples = 100
        results: list[str] = []
        for horizon in horizons:
            if self._app._telegram_bot is not None:
                await self._app._telegram_bot.notify(
                    f"⏳ <b>Training ALL</b>: запускаю горизонт <code>{horizon}m</code>…"
                )
            await self._app._run_model_training(min_samples, horizon, label_bps)
            results.append(f"h{horizon}m: готово")
        if self._app._telegram_bot is not None:
            summary = " | ".join(results)
            await self._app._telegram_bot.notify(f"✅ <b>Training ALL завершено</b>\n{summary}")

    async def run_model_training(self, min_samples: int, horizon: int, label_bps: float) -> None:
        cmd = [
            sys.executable,
            "-m",
            "trader.training.train",
            "--min-samples",
            str(min_samples),
            "--horizon",
            str(horizon),
            "--label-bps",
            str(label_bps),
        ]
        from trader.training.labels import active_label_schema_version

        _settings = self._app._settings
        log.info(
            "model_training.started",
            min_samples=min_samples,
            horizon=horizon,
            label_bps=label_bps,
            strategy_allowlist=_settings.TRAIN_STRATEGY_ALLOWLIST if _settings is not None else "",
            include_candle_baseline=_settings.TRAIN_INCLUDE_CANDLE_BASELINE if _settings is not None else None,
            label_schema=(
                active_label_schema_version(use_tpsl_exit=bool(_settings.MODEL_LABEL_USE_TPSL_EXIT))
                if _settings is not None
                else None
            ),
        )
        started_at = datetime.now(tz=UTC)

        def code_text(value: str, limit: int = 1500) -> str:
            return html.escape(value[-limit:])

        try:
            # Pass only the variables the training subprocess needs.
            # Never forward the full os.environ to avoid leaking secrets
            # (API keys, tokens, DSN passwords) to the child process.
            _safe_env_passthrough = {
                "PATH",
                "HOME",
                "USER",
                "LANG",
                "LC_ALL",
                "LC_CTYPE",
                "TZ",
                "PYTHONPATH",
                "PYTHONDONTWRITEBYTECODE",
                "VIRTUAL_ENV",
                # Postgres DSN is required for training data queries.
                "POSTGRES_DSN",
                "DATABASE_URL",
            }
            train_env = {k: v for k, v in os.environ.items() if k in _safe_env_passthrough}
            # Keep sklearn/BLAS from saturating the single Render starter CPU.
            train_env.update(
                {
                    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
                    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
                    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
                    "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "1"),
                    "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS", "1"),
                    "BLIS_NUM_THREADS": os.environ.get("BLIS_NUM_THREADS", "1"),
                    "TRAIN_STRATEGY_ALLOWLIST": (
                        self._app._settings.TRAIN_STRATEGY_ALLOWLIST if self._app._settings is not None else ""
                    ),
                    "TRAIN_INCLUDE_CANDLE_BASELINE": (
                        "true" if self._app._settings and self._app._settings.TRAIN_INCLUDE_CANDLE_BASELINE else "false"
                    ),
                    "MODEL_LABEL_USE_TPSL_EXIT": (
                        "true" if self._app._settings and self._app._settings.MODEL_LABEL_USE_TPSL_EXIT else "false"
                    ),
                }
            )
            create_kwargs: dict[str, Any] = {"env": train_env}
            if hasattr(os, "nice"):
                create_kwargs["preexec_fn"] = lambda: os.nice(10)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **create_kwargs,
            )
            communicate_task = asyncio.create_task(proc.communicate(), name="model-training-communicate")
            timed_out = False
            _notified_running = False
            try:
                while True:
                    try:
                        stdout_b, stderr_b = await asyncio.wait_for(
                            asyncio.shield(communicate_task),
                            timeout=_TRAINING_HEARTBEAT_SECONDS,
                        )
                        break
                    except TimeoutError:
                        elapsed = (datetime.now(tz=UTC) - started_at).total_seconds()
                        if elapsed >= _TRAINING_TIMEOUT_SECONDS:
                            timed_out = True
                            if proc.returncode is None:
                                proc.kill()
                            try:
                                stdout_b, stderr_b = await asyncio.wait_for(communicate_task, timeout=10.0)
                            except TimeoutError:
                                communicate_task.cancel()
                                with contextlib.suppress(asyncio.CancelledError):
                                    await communicate_task
                                stdout_b = b""
                                stderr_b = f"training timeout after {elapsed:.0f}s".encode()
                            break
                        if self._app._telegram_bot is not None and not _notified_running:
                            await self._app._telegram_bot.notify(f"⏳ <b>Обучение модели...</b> (~{int(elapsed)}с)")
                            _notified_running = True
            except asyncio.CancelledError:
                # Kill the subprocess so it does not become an orphan when the
                # event loop is shutting down.
                if proc.returncode is None:
                    proc.kill()
                communicate_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await communicate_task
                raise
            stdout = stdout_b.decode(errors="replace").strip()
            stderr = stderr_b.decode(errors="replace").strip()
            if timed_out:
                self._app._last_training_message = stderr or stdout or "training timeout"
                self._app._training_failed_at = time.monotonic()
                text = "❌ <b>Training timed out</b>\n" + f"<code>{code_text(self._app._last_training_message)}</code>"
            elif proc.returncode == 0 and "Checkpoint saved" in stdout:
                self._app._last_training_message = stdout.splitlines()[-2] if len(stdout.splitlines()) >= 2 else stdout
                if (
                    self._app._model_registry is not None
                    and self._app._trade_journal is not None
                    and self._app._trade_journal.is_enabled
                ):
                    await self._app._model_registry.load_active_model()
                text = "✅ <b>Training completed</b>\n" + f"<code>{code_text(self._app._last_training_message)}</code>"
            elif proc.returncode == 0:
                self._app._last_training_message = stdout or stderr or "training finished without checkpoint"
                text = (
                    "⚠️ <b>Training finished without checkpoint</b>\n"
                    + f"<code>{code_text(self._app._last_training_message)}</code>"
                )
            else:
                self._app._last_training_message = stderr or stdout or f"exit code {proc.returncode}"
                failure_text = self._app._last_training_message or ""
                if "Insufficient compatible samples" not in failure_text:
                    self._app._training_failed_at = time.monotonic()
                text = "❌ <b>Training failed</b>\n" + f"<code>{code_text(self._app._last_training_message)}</code>"
            log.info(
                "model_training.finished",
                returncode=proc.returncode,
                message=self._app._last_training_message,
            )
        except Exception as exc:
            self._app._last_training_message = str(exc)
            self._app._training_failed_at = time.monotonic()
            text = f"❌ <b>Training crashed</b>\n<code>{code_text(str(exc))}</code>"
            log.warning("model_training.crashed", error=str(exc))
        if self._app._trade_journal is not None and self._app._trade_journal.is_enabled:
            try:
                self._app._update_model_gate_quality_from_diag(await self._app._trade_journal.get_db_diagnostics())
            except Exception as diag_exc:
                log.debug("model_gate.quality_refresh_failed", error=str(diag_exc))
        if self._app._telegram_bot is not None:
            await self._app._telegram_bot.notify(text)
        gc.collect()

    async def maybe_apply_online_learning(self) -> None:
        assert self._app._settings is not None
        if not self._app._settings.MODEL_ONLINE_LEARNING_ENABLED:
            return
        if self._app._trade_journal is None or not self._app._trade_journal.is_enabled:
            return
        if self._app._model_registry is None:
            return
        challenger = self._app._model_registry.challenger
        if challenger is None or not challenger.supports_online_learning:
            return

        from trader.training.labels import active_label_schema_version

        label_schema = active_label_schema_version(use_tpsl_exit=bool(self._app._settings.MODEL_LABEL_USE_TPSL_EXIT))
        limit = int(self._app._settings.MODEL_ONLINE_LEARNING_MAX_UPDATES_PER_CYCLE)
        batch = await self._app._trade_journal.fetch_online_learning_batch(
            limit=limit,
            challenger_version=challenger.version,
            label_schema_version=label_schema,
        )
        if not batch:
            return
        if (
            self._app._model_registry.challenger is None
            or self._app._model_registry.challenger.version != challenger.version
        ):
            # A promotion/retrain swapped the registry's challenger while we
            # were awaiting the batch fetch above. This batch was labelled
            # against the old challenger_version filter — applying it to
            # whatever model is current now would mutate the wrong model.
            log.warning(
                "online_learning.batch_skipped_challenger_changed",
                fetched_for_version=challenger.version,
                current_version=getattr(self._app._model_registry.challenger, "version", None),
            )
            return
        applied = 0
        prediction_ids: list[str] = []
        for row in batch:
            try:
                if self._app._model_registry.partial_fit_challenger(row["features"], int(row["label"])):
                    prediction_ids.append(str(row["prediction_id"]))
                    applied += 1
            except Exception as exc:
                log.debug("online_learning.partial_fit_failed", error=str(exc))
        if prediction_ids:
            await self._app._trade_journal.mark_online_learning_applied(prediction_ids)
        if applied:
            self._app._online_learning_updates_since_checkpoint += applied
            checkpoint_every = max(1, int(self._app._settings.MODEL_ONLINE_LEARNING_CHECKPOINT_EVERY))
            if self._app._online_learning_updates_since_checkpoint >= checkpoint_every:
                # partial_fit_challenger mutates whatever the registry's
                # *current* challenger is; a promotion/retrain running
                # concurrently on the awaits above could have swapped it out
                # from under this stale local reference. Only checkpoint if
                # it's still the same model, otherwise the mutated model
                # would go unpersisted while a different (untouched) one
                # gets checkpointed instead.
                current_challenger = self._app._model_registry.challenger
                if current_challenger is not None and current_challenger.version == challenger.version:
                    await self._app._model_registry.save_checkpoint(current_challenger)
                    self._app._online_learning_updates_since_checkpoint = 0
                    log.info(
                        "online_learning.checkpoint_saved",
                        version=challenger.version,
                        samples=current_challenger.training_samples,
                    )
                else:
                    log.warning(
                        "online_learning.checkpoint_skipped_challenger_changed",
                        applied_to_version=challenger.version,
                        current_version=getattr(current_challenger, "version", None),
                    )
            log.info("online_learning.updated", samples=applied, model_version=challenger.version)

    async def evaluate_feature_drift(self) -> dict[str, Any]:
        assert self._app._settings is not None
        if not self._app._settings.MODEL_DRIFT_DETECTION_ENABLED or self._app._trade_journal is None:
            return {"status": "disabled"}
        if not self._app._trade_journal.is_enabled:
            return {"status": "journal_unavailable"}
        try:
            baseline, current = await self._app._trade_journal.fetch_feature_drift_samples(limit=500)
            min_samples = int(self._app._settings.MODEL_DRIFT_MIN_SAMPLES)
            if len(baseline) < min_samples or len(current) < 50:
                return {
                    "status": "insufficient_samples",
                    "baseline_count": len(baseline),
                    "current_count": len(current),
                }
            from trader.ml.drift import drift_summary_from_samples

            summary = drift_summary_from_samples(
                baseline,
                current,
                psi_threshold=float(self._app._settings.MODEL_DRIFT_PSI_THRESHOLD),
            )
            summary["status"] = "drift" if summary.get("drift_detected") else "stable"
            self._app._drift_status = summary
            return summary
        except Exception as exc:
            self._app._drift_status = {"status": "error", "error": str(exc)}
            return self._app._drift_status

    async def run_bucket_stats_refresher(self) -> None:
        """Refresh in-memory expectancy gates from Postgres periodically."""
        assert self._app._settings is not None
        interval = float(self._app._settings.BUCKET_STATS_REFRESH_SECONDS)

        while not self._app._shutdown_event.is_set():
            if self._app._trade_journal is not None and self._app._trade_journal.is_enabled:
                try:
                    horizon_minutes = int(self._app._settings.MODEL_AUTO_TRAIN_HORIZON_MINUTES)
                    stats = await self._app._trade_journal.get_bucket_stats(
                        horizon_minutes=horizon_minutes,
                    )
                    hour_stats = await self._app._trade_journal.get_hour_stats(
                        horizon_minutes=horizon_minutes,
                    )
                    strategy_stats = await self._app._trade_journal.get_strategy_stats(
                        horizon_minutes=horizon_minutes,
                    )
                    strategy_side_stats = await self._app._trade_journal.get_strategy_side_stats(
                        horizon_minutes=horizon_minutes,
                    )
                    strategy_regime_stats = await self._app._trade_journal.get_strategy_regime_stats(
                        horizon_minutes=horizon_minutes,
                    )
                    symbol_side_stats = await self._app._trade_journal.get_symbol_side_stats(
                        horizon_minutes=horizon_minutes,
                    )
                    probe_side_stats = await self._app._trade_journal.get_shadow_probe_symbol_side_stats(
                        horizon_minutes=horizon_minutes,
                        lookback_days=self._app._settings.SHADOW_PROBE_STATS_LOOKBACK_DAYS,
                    )
                    probe_symbol_stats = await self._app._trade_journal.get_shadow_probe_symbol_stats(
                        horizon_minutes=horizon_minutes,
                        lookback_days=self._app._settings.SHADOW_PROBE_STATS_LOOKBACK_DAYS,
                    )
                    self._app._bucket_stats = stats
                    self._app._hour_stats = hour_stats
                    self._app._strategy_stats = strategy_stats
                    self._app._strategy_side_stats = strategy_side_stats
                    self._app._strategy_regime_stats = strategy_regime_stats
                    self._app._symbol_side_stats = symbol_side_stats
                    self._app._shadow_probe_side_stats = probe_side_stats
                    self._app._shadow_probe_symbol_stats = probe_symbol_stats
                    self._app._shadow_probe_eligible_symbols = (
                        self._app._modules.signal_policy.compute_shadow_probe_eligible_symbols(
                            probe_symbol_stats,
                            top_n=self._app._settings.SHADOW_PROBE_SYMBOL_TOP_N,
                            min_samples=self._app._settings.SHADOW_PROBE_SYMBOL_MIN_SAMPLES,
                            min_avg_bps=self._app._settings.SHADOW_PROBE_SYMBOL_MIN_AVG_BPS,
                        )
                    )
                    self._app._bucket_stats_refreshed_at = datetime.now(tz=UTC)
                    blocked = [
                        key
                        for key, (avg, cnt) in stats.items()
                        if cnt >= self._app._settings.BUCKET_MIN_SAMPLES
                        and avg < self._app._settings.BUCKET_BLOCK_AVG_BPS
                    ]
                    blocked_symbol_sides = [
                        key
                        for key, (avg, cnt) in symbol_side_stats.items()
                        if cnt >= self._app._settings.SYMBOL_SIDE_MIN_SAMPLES
                        and avg < self._app._settings.SYMBOL_SIDE_BLOCK_AVG_BPS
                    ]
                    blocked_hours = [
                        hour
                        for hour, (avg, cnt) in hour_stats.items()
                        if cnt >= self._app._settings.HOUR_MIN_SAMPLES and avg < self._app._settings.HOUR_BLOCK_AVG_BPS
                    ]
                    blocked_strategies = [
                        strategy_id
                        for strategy_id, (avg, cnt) in strategy_stats.items()
                        if cnt >= self._app._settings.STRATEGY_MIN_SAMPLES
                        and avg < self._app._settings.STRATEGY_BLOCK_AVG_BPS
                    ]
                    blocked_strategy_sides = [
                        key
                        for key, (avg, cnt) in strategy_side_stats.items()
                        if cnt >= self._app._settings.STRATEGY_SIDE_MIN_SAMPLES
                        and avg < self._app._settings.STRATEGY_SIDE_BLOCK_AVG_BPS
                    ]
                    blocked_strategy_regimes = [
                        key
                        for key, (avg, cnt) in strategy_regime_stats.items()
                        if cnt >= self._app._settings.STRATEGY_REGIME_MIN_SAMPLES
                        and avg < self._app._settings.STRATEGY_REGIME_BLOCK_AVG_BPS
                    ]
                    blocked_probe_sides = [
                        key
                        for key, (avg, cnt) in probe_side_stats.items()
                        if cnt >= self._app._settings.SHADOW_PROBE_SIDE_MIN_SAMPLES
                        and avg < self._app._settings.SHADOW_PROBE_SIDE_BLOCK_AVG_BPS
                    ]
                    blocked_probe_symbols = [
                        symbol
                        for symbol, (avg, cnt) in probe_symbol_stats.items()
                        if cnt >= self._app._settings.SHADOW_PROBE_SYMBOL_MIN_SAMPLES
                        and avg < self._app._settings.SHADOW_PROBE_SYMBOL_MIN_AVG_BPS
                    ]
                    log.info(
                        "bucket_stats.refreshed",
                        horizon_minutes=horizon_minutes,
                        buckets=len(stats),
                        blocked=len(blocked),
                        blocked_keys=blocked[:10],
                        hours=len(hour_stats),
                        blocked_hours=blocked_hours,
                        strategies=len(strategy_stats),
                        blocked_strategies=blocked_strategies,
                        strategy_sides=len(strategy_side_stats),
                        blocked_strategy_sides=blocked_strategy_sides[:10],
                        strategy_regimes=len(strategy_regime_stats),
                        blocked_strategy_regimes=blocked_strategy_regimes[:10],
                        symbol_sides=len(symbol_side_stats),
                        blocked_symbol_sides=blocked_symbol_sides[:10],
                        probe_symbol_sides=len(probe_side_stats),
                        blocked_probe_sides=blocked_probe_sides[:10],
                        probe_symbols=len(probe_symbol_stats),
                        blocked_probe_symbols=blocked_probe_symbols[:10],
                        probe_eligible_symbols=sorted(self._app._shadow_probe_eligible_symbols or [])[:10],
                    )
                except Exception as exc:
                    log.warning("bucket_stats.refresh_failed", error=str(exc))

            try:
                await asyncio.wait_for(
                    self._app._shutdown_event.wait(),
                    timeout=interval,
                )
            except TimeoutError:
                pass
