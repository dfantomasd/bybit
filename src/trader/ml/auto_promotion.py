"""Safe automatic promotion and rollback for shadow ML challengers."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast

import structlog

from trader.ml.challenger import ModelStatus
from trader.ml.model_selection import model_selection_metrics
from trader.training.bootstrap import bootstrap_pvalue
from trader.training.labels import LABEL_SCHEMA_VERSION, active_label_schema_version

log = structlog.get_logger(__name__)

_PROMOTION_LOCK_KEY = 926_202_606


def _metrics(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _metric_score(metrics: dict[str, Any]) -> float:
    return float(model_selection_metrics(metrics)["model_score"])


def _quality_rank(quality: str) -> int:
    normalized = str(quality or "").upper()
    if normalized == "GOOD":
        return 2
    if normalized == "WEAK":
        return 1
    return 0


def _model_snapshot(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    metrics = _metrics(row.get("metrics"))
    normalized = model_selection_metrics(metrics)
    return {
        "version": row.get("version"),
        "status": row.get("status"),
        "training_samples": row.get("training_samples"),
        "training_finished_at": str(row.get("training_finished_at") or ""),
        "quality": metrics.get("quality"),
        "model_score": metrics.get("model_score", normalized["model_score"]),
        "walk_forward_bps": normalized["walk_forward_bps"],
        "wf_positive_folds": metrics.get("wf_positive_folds"),
        "wf_folds": metrics.get("wf_folds"),
        "wf_std_bps": metrics.get("wf_std_bps"),
        "lift_bps": normalized["lift_bps"],
        "paper_gate_count": normalized["paper_gate_count"],
        "label_schema_version": metrics.get("label_schema_version"),
        "walk_forward_chronology": metrics.get("walk_forward_chronology"),
    }


def _max_drawdown_bps(returns_bps: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in reversed(returns_bps):
        equity += float(value)
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


@dataclass(frozen=True)
class AutoPromotionConfig:
    enabled: bool = False
    check_seconds: int = 600
    monitor_seconds: int = 14_400
    min_training_samples: int = 500
    min_shadow_signals: int = 50
    min_pass_count: int = 20
    min_lift_bps: float = 1.0
    min_pass_expectancy_bps: float = 0.0
    min_wf_bps: float = 0.0
    min_wf_positive_folds: int = 3
    max_wf_std_bps: float = 25.0
    required_quality: str = "GOOD"
    pvalue_threshold: float = 0.05
    bootstrap_iterations: int = 1000
    min_bootstrap_samples: int = 50
    horizon_minutes: int = 15
    max_champion_drawdown_bps: float = 1500.0
    min_champion_wf_bps: float = 0.0
    returns_limit: int = 200
    label_schema_version: str = LABEL_SCHEMA_VERSION

    @classmethod
    def from_settings(cls, settings: Any) -> AutoPromotionConfig:
        return cls(
            enabled=bool(getattr(settings, "MODEL_AUTO_PROMOTE_ENABLED", False)),
            check_seconds=max(120, int(getattr(settings, "MODEL_AUTO_PROMOTE_CHECK_SECONDS", 600))),
            monitor_seconds=max(600, int(getattr(settings, "MODEL_CHAMPION_MONITOR_SECONDS", 14_400))),
            min_training_samples=max(1, int(getattr(settings, "MODEL_MIN_TRAINING_SAMPLES", 500))),
            min_shadow_signals=max(1, int(getattr(settings, "MODEL_AUTO_PROMOTE_MIN_SIGNALS", 50))),
            min_pass_count=max(1, int(getattr(settings, "MODEL_MIN_PASS_COUNT_FOR_PROMOTION", 20))),
            min_lift_bps=float(getattr(settings, "MODEL_AUTO_PROMOTE_MIN_LIFT_BPS", 1.0)),
            min_pass_expectancy_bps=float(getattr(settings, "MODEL_AUTO_PROMOTE_MIN_PASS_EXPECTANCY_BPS", 0.0)),
            min_wf_bps=float(getattr(settings, "MODEL_AUTO_PROMOTE_MIN_WF_BPS", 0.0)),
            min_wf_positive_folds=max(1, int(getattr(settings, "MODEL_AUTO_PROMOTE_MIN_WF_POSITIVE_FOLDS", 3))),
            max_wf_std_bps=float(getattr(settings, "MODEL_AUTO_PROMOTE_MAX_WF_STD_BPS", 25.0)),
            required_quality=str(
                getattr(
                    settings,
                    "MODEL_AUTO_PROMOTE_MIN_QUALITY",
                    getattr(settings, "MODEL_GATE_CANARY_MIN_QUALITY", "GOOD"),
                )
            ).upper(),
            pvalue_threshold=float(getattr(settings, "MODEL_AUTO_PROMOTE_PVALUE_THRESHOLD", 0.05)),
            bootstrap_iterations=max(100, int(getattr(settings, "MODEL_AUTO_PROMOTE_BOOTSTRAP_ITERATIONS", 1000))),
            min_bootstrap_samples=max(2, int(getattr(settings, "MODEL_AUTO_PROMOTE_MIN_BOOTSTRAP_SAMPLES", 50))),
            horizon_minutes=int(getattr(settings, "MODEL_AUTO_TRAIN_HORIZON_MINUTES", 15)),
            label_schema_version=active_label_schema_version(
                use_tpsl_exit=bool(getattr(settings, "MODEL_LABEL_USE_TPSL_EXIT", False))
            ),
            max_champion_drawdown_bps=float(getattr(settings, "MODEL_CHAMPION_MAX_DRAWDOWN_BPS", 1500.0)),
            min_champion_wf_bps=float(getattr(settings, "MODEL_CHAMPION_MIN_WF_BPS", 0.0)),
        )


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    champion_version: str | None
    challenger_version: str | None
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RollbackDecision:
    rollback: bool
    champion_version: str | None
    rollback_version: str | None
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class AutoPromotionEngine:
    """Evaluate, promote, and roll back models using DB-backed evidence."""

    def __init__(
        self,
        *,
        trade_journal: Any,
        config: AutoPromotionConfig,
        reload_registry: Callable[[], Awaitable[Any]] | None = None,
    ) -> None:
        self._journal = trade_journal
        self._config = config
        self._reload_registry = reload_registry

    async def best_challenger(self) -> str | None:
        rows = await self._fetch(
            """
            SELECT version, training_samples, metrics, training_finished_at, created_at
            FROM model_versions
            WHERE status IN ('SHADOW_CHALLENGER', 'VALIDATED')
              AND artifact IS NOT NULL
            ORDER BY training_finished_at DESC NULLS LAST, created_at DESC
            LIMIT 25
            """
        )
        scored: list[tuple[float, str]] = []
        for row in rows:
            version = str(row["version"])
            decision = await self.should_promote(champion_version=None, challenger_version=version)
            score = float(decision.metrics.get("selection_score", -1_000_000.0))
            if decision.promote:
                scored.append((score, version))
        if not scored:
            return None
        return max(scored, key=lambda item: item[0])[1]

    async def should_promote(
        self,
        champion_version: str | None,
        challenger_version: str,
    ) -> PromotionDecision:
        del champion_version
        row = await self._fetchrow(
            """
            SELECT version, status, training_samples, feature_schema_hash, metrics
            FROM model_versions
            WHERE version = $1
            LIMIT 1
            """,
            challenger_version,
        )
        if not row:
            return PromotionDecision(False, None, challenger_version, ["challenger_not_found"])

        metrics = _metrics(row.get("metrics"))
        reasons: list[str] = []
        status = str(row.get("status") or "")
        training_samples = _int_or_zero(row.get("training_samples"))
        quality = str(metrics.get("quality") or "").upper()
        label_schema = str(metrics.get("label_schema_version") or "")
        wf_bps = _float_or_none(metrics.get("walk_forward_expectancy_bps"))
        if wf_bps is None:
            wf_bps = _float_or_none(metrics.get("wf_mean_bps"))
        if wf_bps is None:
            wf_bps = _float_or_none(metrics.get("best_threshold_avg_net_return_bps"))

        if status not in {ModelStatus.SHADOW_CHALLENGER, ModelStatus.VALIDATED}:
            reasons.append(f"bad_status:{status}")
        if label_schema != self._config.label_schema_version:
            reasons.append(f"incompatible_label_schema:{label_schema or 'missing'}")
        if training_samples < self._config.min_training_samples:
            reasons.append(f"insufficient_training_samples:{training_samples}<{self._config.min_training_samples}")
        required_quality = str(self._config.required_quality or "").upper()
        if required_quality and _quality_rank(quality) < _quality_rank(required_quality):
            reasons.append(f"quality_below_{required_quality}:{quality or 'missing'}")
        if wf_bps is None or wf_bps < self._config.min_wf_bps:
            reasons.append(f"weak_walk_forward:{wf_bps}")
        wf_folds = _int_or_zero(metrics.get("wf_folds"))
        wf_positive_folds = _int_or_zero(metrics.get("wf_positive_folds"))
        wf_std_bps = _float_or_none(metrics.get("wf_std_bps"))
        if wf_folds > 0 and wf_positive_folds < self._config.min_wf_positive_folds:
            reasons.append(f"unstable_walk_forward_folds:{wf_positive_folds}<{self._config.min_wf_positive_folds}")
        if wf_std_bps is not None and wf_std_bps > self._config.max_wf_std_bps:
            reasons.append(f"unstable_walk_forward_std:{wf_std_bps:.4f}>{self._config.max_wf_std_bps:.4f}")
        if metrics.get("fold_metrics") and metrics.get("walk_forward_chronology") != "strict_after_train":
            reasons.append("walk_forward_not_strict_after_train")

        gate = await self._journal.get_shadow_gate_stats(
            challenger_version,
            self._config.horizon_minutes,
            self._config.label_schema_version,
        )
        total_count = _int_or_zero(gate.get("total_count"))
        pass_count = _int_or_zero(gate.get("pass_count"))
        lift_bps = _float_or_none(gate.get("lift_vs_all_bps"))
        pass_expectancy = _float_or_none(gate.get("pass_avg_net_return_bps"))
        pass_precision = _float_or_none(gate.get("pass_precision"))
        if total_count < self._config.min_shadow_signals:
            reasons.append(f"insufficient_shadow_signals:{total_count}<{self._config.min_shadow_signals}")
        if pass_count < self._config.min_pass_count:
            reasons.append(f"insufficient_pass_count:{pass_count}<{self._config.min_pass_count}")
        if lift_bps is None or lift_bps < self._config.min_lift_bps:
            reasons.append(f"insufficient_lift:{lift_bps}")
        if pass_expectancy is None or pass_expectancy < self._config.min_pass_expectancy_bps:
            reasons.append(f"weak_pass_expectancy:{pass_expectancy}")

        champion = await self._current_champion()
        champion_version_found = str(champion["version"]) if champion else None
        champion_metrics = _metrics(champion.get("metrics")) if champion else {}
        champion_wf = _float_or_none(champion_metrics.get("walk_forward_expectancy_bps"))
        if champion_wf is None:
            champion_wf = _float_or_none(champion_metrics.get("wf_mean_bps"))
        if champion_wf is None:
            champion_wf = _float_or_none(champion_metrics.get("best_threshold_avg_net_return_bps"))
        if champion_wf is not None and wf_bps is not None and wf_bps <= champion_wf:
            reasons.append(f"not_better_than_champion:{wf_bps:.4f}<={champion_wf:.4f}")

        challenger_returns = await self._journal.get_returns_for_model(
            challenger_version,
            limit=self._config.returns_limit,
            horizon_minutes=self._config.horizon_minutes,
            label_schema_version=self._config.label_schema_version,
        )
        baseline_returns = await self._journal.get_returns_for_model(
            "RULE_BASELINE_V1",
            limit=self._config.returns_limit,
            horizon_minutes=self._config.horizon_minutes,
            label_schema_version=self._config.label_schema_version,
        )
        p_value: float | None = None
        mean_diff_bps: float | None = None
        if (
            len(challenger_returns) >= self._config.min_bootstrap_samples
            and len(baseline_returns) >= self._config.min_bootstrap_samples
        ):
            boot = bootstrap_pvalue(
                challenger_returns,
                baseline_returns,
                n_iter=self._config.bootstrap_iterations,
            )
            p_value = float(boot.p_value)
            mean_diff_bps = float(boot.mean_diff_bps)
            if p_value >= self._config.pvalue_threshold:
                reasons.append(f"bootstrap_not_significant:{p_value:.4f}>={self._config.pvalue_threshold:.4f}")
            if mean_diff_bps <= 0:
                reasons.append(f"non_positive_bootstrap_diff:{mean_diff_bps:.4f}")
        else:
            reasons.append(
                "insufficient_bootstrap_samples:"
                f"{len(challenger_returns)}/{len(baseline_returns)}<{self._config.min_bootstrap_samples}"
            )

        decision_metrics = {
            "training_samples": training_samples,
            "quality": quality,
            "wf_bps": wf_bps,
            "total_count": total_count,
            "pass_count": pass_count,
            "lift_bps": lift_bps,
            "pass_expectancy_bps": pass_expectancy,
            "pass_precision": pass_precision,
            "champion_wf_bps": champion_wf,
            "wf_folds": wf_folds,
            "wf_positive_folds": wf_positive_folds,
            "wf_std_bps": wf_std_bps,
            "walk_forward_chronology": metrics.get("walk_forward_chronology"),
            "bootstrap_p_value": p_value,
            "bootstrap_mean_diff_bps": mean_diff_bps,
            "model_score": _metric_score(metrics),
            "selection_score": _metric_score(metrics) + ((lift_bps or 0.0) * 0.5) + ((pass_expectancy or 0.0) * 0.5),
        }
        decision_metrics["snapshot"] = self._promotion_snapshot(
            champion=champion,
            challenger=row,
            challenger_gate=gate,
            p_value=p_value,
            mean_diff_bps=mean_diff_bps,
        )
        return PromotionDecision(
            promote=not reasons,
            champion_version=champion_version_found,
            challenger_version=challenger_version,
            reasons=reasons or ["criteria_met"],
            metrics=decision_metrics,
        )

    async def promote(self, challenger_version: str) -> PromotionDecision:
        decision = await self.should_promote(None, challenger_version)
        if not decision.promote:
            await self._record_log("PROMOTION_BLOCKED", decision.champion_version, challenger_version, decision)
            return decision

        pool = getattr(self._journal, "_pool", None)
        if pool is None:
            return PromotionDecision(False, decision.champion_version, challenger_version, ["journal_pool_unavailable"])

        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock($1)", _PROMOTION_LOCK_KEY)
                champion_row = await conn.fetchrow(
                    "SELECT version FROM model_versions WHERE status = 'CHAMPION' ORDER BY training_finished_at DESC NULLS LAST LIMIT 1"
                )
                current_champion = str(champion_row["version"]) if champion_row else None
                await conn.execute(
                    """
                    UPDATE model_versions
                    SET status = 'ARCHIVED'
                    WHERE status = 'CHAMPION'
                    """
                )
                updated = await conn.execute(
                    """
                    UPDATE model_versions
                    SET status = 'CHAMPION'
                    WHERE version = $1
                      AND status IN ('SHADOW_CHALLENGER', 'VALIDATED')
                    """,
                    challenger_version,
                )
                if not updated.endswith("1"):
                    raise RuntimeError(f"promotion update affected unexpected rows: {updated}")
                await conn.execute(
                    """
                    INSERT INTO model_promotion_log (
                        event_type, from_version, to_version, reasons, metrics, metrics_snapshot
                    )
                    VALUES ('PROMOTED', $1, $2, $3::jsonb, $4::jsonb, $4::jsonb)
                    """,
                    current_champion,
                    challenger_version,
                    json.dumps(decision.reasons),
                    json.dumps(decision.metrics),
                )

        if self._reload_registry is not None:
            await self._reload_registry()
        log.info("model_auto_promotion.promoted", version=challenger_version, metrics=decision.metrics)
        return decision

    async def should_rollback(self) -> RollbackDecision:
        champion = await self._current_champion()
        if not champion:
            return RollbackDecision(False, None, None, ["no_champion"])
        champion_version = str(champion["version"])
        metrics = _metrics(champion.get("metrics"))
        wf_bps = _float_or_none(metrics.get("walk_forward_expectancy_bps"))
        if wf_bps is None:
            wf_bps = _float_or_none(metrics.get("wf_mean_bps"))
        if wf_bps is None:
            wf_bps = _float_or_none(metrics.get("best_threshold_avg_net_return_bps"))
        returns = await self._journal.get_returns_for_model(
            champion_version,
            limit=self._config.returns_limit,
            horizon_minutes=self._config.horizon_minutes,
            label_schema_version=self._config.label_schema_version,
        )
        drawdown_bps = _float_or_none(metrics.get("max_drawdown_bps"))
        if drawdown_bps is None and returns:
            drawdown_bps = _max_drawdown_bps(returns)

        reasons: list[str] = []
        if wf_bps is not None and wf_bps < self._config.min_champion_wf_bps:
            reasons.append(f"champion_wf_degraded:{wf_bps:.4f}<{self._config.min_champion_wf_bps:.4f}")
        if drawdown_bps is not None and drawdown_bps > self._config.max_champion_drawdown_bps:
            reasons.append(f"champion_drawdown:{drawdown_bps:.4f}>{self._config.max_champion_drawdown_bps:.4f}")
        rollback_row = await self._fetchrow(
            """
            SELECT version
            FROM model_versions
            WHERE status = 'ARCHIVED'
              AND artifact IS NOT NULL
              AND version <> $1
              AND COALESCE(metrics->>'label_schema_version', '') = $3
              AND COALESCE(
                      NULLIF(metrics->>'walk_forward_expectancy_bps', ''),
                      NULLIF(metrics->>'wf_mean_bps', ''),
                      NULLIF(metrics->>'best_threshold_avg_net_return_bps', '')
                  ) IS NOT NULL
              AND COALESCE(
                      NULLIF(metrics#>>'{paper_gate,count}', ''),
                      NULLIF(metrics#>>'{model_gate,count}', ''),
                      NULLIF(metrics->>'paper_gate_count', ''),
                      NULLIF(metrics->>'total_pass_count', ''),
                      NULLIF(metrics->>'best_threshold_pass_count', '')
                  )::integer >= $2
            ORDER BY
                CASE WHEN COALESCE(
                    NULLIF(metrics->>'walk_forward_expectancy_bps', ''),
                    NULLIF(metrics->>'wf_mean_bps', ''),
                    NULLIF(metrics->>'best_threshold_avg_net_return_bps', '')
                )::double precision > 0 THEN 0 ELSE 1 END,
                CASE WHEN COALESCE(
                    NULLIF(metrics->>'walk_forward_expectancy_bps', ''),
                    NULLIF(metrics->>'wf_mean_bps', ''),
                    NULLIF(metrics->>'best_threshold_avg_net_return_bps', '')
                )::double precision > 0 THEN COALESCE(
                    NULLIF(metrics->>'lift_bps', ''),
                    NULLIF(metrics#>>'{paper_gate,lift_bps}', ''),
                    NULLIF(metrics#>>'{model_gate,lift_bps}', ''),
                    '0'
                )::double precision END DESC NULLS LAST,
                CASE WHEN COALESCE(
                    NULLIF(metrics->>'walk_forward_expectancy_bps', ''),
                    NULLIF(metrics->>'wf_mean_bps', ''),
                    NULLIF(metrics->>'best_threshold_avg_net_return_bps', '')
                )::double precision > 0 THEN training_finished_at END DESC NULLS LAST,
                COALESCE(
                    NULLIF(metrics->>'walk_forward_expectancy_bps', ''),
                    NULLIF(metrics->>'wf_mean_bps', ''),
                    NULLIF(metrics->>'best_threshold_avg_net_return_bps', '')
                )::double precision DESC,
                COALESCE(
                    NULLIF(metrics->>'lift_bps', ''),
                    NULLIF(metrics#>>'{paper_gate,lift_bps}', ''),
                    NULLIF(metrics#>>'{model_gate,lift_bps}', ''),
                    '0'
                )::double precision DESC,
                training_finished_at DESC NULLS LAST,
                created_at DESC
            LIMIT 1
            """,
            champion_version,
            self._config.min_shadow_signals,
            self._config.label_schema_version,
        )
        rollback_version = str(rollback_row["version"]) if rollback_row else None
        if reasons and rollback_version is None:
            reasons.append("no_archived_champion_available")
        return RollbackDecision(
            rollback=bool(reasons) and rollback_version is not None,
            champion_version=champion_version,
            rollback_version=rollback_version,
            reasons=reasons or ["champion_healthy"],
            metrics={"champion_wf_bps": wf_bps, "champion_drawdown_bps": drawdown_bps},
        )

    async def rollback_if_needed(self) -> RollbackDecision:
        decision = await self.should_rollback()
        if not decision.rollback:
            return decision
        pool = getattr(self._journal, "_pool", None)
        if pool is None:
            return RollbackDecision(
                False, decision.champion_version, decision.rollback_version, ["journal_pool_unavailable"]
            )

        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock($1)", _PROMOTION_LOCK_KEY)
                await conn.execute(
                    "UPDATE model_versions SET status = 'ROLLED_BACK' WHERE version = $1 AND status = 'CHAMPION'",
                    decision.champion_version,
                )
                updated = await conn.execute(
                    "UPDATE model_versions SET status = 'CHAMPION' WHERE version = $1 AND status = 'ARCHIVED'",
                    decision.rollback_version,
                )
                if not updated.endswith("1"):
                    raise RuntimeError(f"rollback update affected unexpected rows: {updated}")
                await conn.execute(
                    """
                    INSERT INTO model_promotion_log (
                        event_type, from_version, to_version, reasons, metrics
                    )
                    VALUES ('ROLLED_BACK', $1, $2, $3::jsonb, $4::jsonb)
                    """,
                    decision.champion_version,
                    decision.rollback_version,
                    json.dumps(decision.reasons),
                    json.dumps(decision.metrics),
                )

        if self._reload_registry is not None:
            await self._reload_registry()
        log.warning(
            "model_auto_promotion.rolled_back",
            from_version=decision.champion_version,
            to_version=decision.rollback_version,
            reasons=decision.reasons,
        )
        return decision

    async def _current_champion(self) -> dict[str, Any] | None:
        return await self._fetchrow(
            """
            SELECT version, training_samples, metrics, training_finished_at, created_at
            FROM model_versions
            WHERE status = 'CHAMPION'
              AND artifact IS NOT NULL
            ORDER BY training_finished_at DESC NULLS LAST, created_at DESC
            LIMIT 1
            """
        )

    async def _record_log(
        self,
        event_type: str,
        from_version: str | None,
        to_version: str | None,
        decision: PromotionDecision | RollbackDecision,
    ) -> None:
        try:
            await self._journal._execute(
                """
                INSERT INTO model_promotion_log (
                    event_type, from_version, to_version, reasons, metrics, metrics_snapshot
                )
                VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $5::jsonb)
                """,
                event_type,
                from_version,
                to_version,
                json.dumps(decision.reasons),
                json.dumps(decision.metrics),
            )
        except Exception as exc:
            log.debug("model_auto_promotion.log_failed", error=str(exc))

    def _promotion_snapshot(
        self,
        *,
        champion: dict[str, Any] | None,
        challenger: dict[str, Any] | None,
        challenger_gate: dict[str, Any] | None = None,
        p_value: float | None = None,
        mean_diff_bps: float | None = None,
    ) -> dict[str, Any]:
        champion_snapshot = _model_snapshot(champion)
        challenger_snapshot = _model_snapshot(challenger)
        delta: dict[str, Any] = {}
        if champion_snapshot and challenger_snapshot:
            for key in ("model_score", "walk_forward_bps", "lift_bps", "paper_gate_count"):
                champion_value = champion_snapshot.get(key)
                challenger_value = challenger_snapshot.get(key)
                if champion_value is not None and challenger_value is not None:
                    delta[key] = float(challenger_value) - float(champion_value)
        return {
            "champion": champion_snapshot,
            "challenger": challenger_snapshot,
            "delta": delta,
            "challenger_gate": challenger_gate or {},
            "bootstrap": {
                "p_value": p_value,
                "mean_diff_bps": mean_diff_bps,
            },
            "thresholds": {
                "min_wf_bps": self._config.min_wf_bps,
                "min_wf_positive_folds": self._config.min_wf_positive_folds,
                "max_wf_std_bps": self._config.max_wf_std_bps,
                "min_shadow_signals": self._config.min_shadow_signals,
                "min_pass_count": self._config.min_pass_count,
                "min_lift_bps": self._config.min_lift_bps,
                "min_pass_expectancy_bps": self._config.min_pass_expectancy_bps,
            },
        }

    async def _fetch(self, query: str, *args: Any) -> list[Any]:
        return cast(list[Any], await self._journal._fetch(query, *args))

    async def _fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        rows = await self._fetch(query, *args)
        if not rows:
            return None
        return dict(rows[0])
