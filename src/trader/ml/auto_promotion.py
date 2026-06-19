"""AutoPromotionEngine — pure evaluation layer for model promotion and champion health.

No I/O.  The engine takes pre-fetched stats and returns structured decisions.
Actual DB execution (atomic status transitions + audit log) is delegated to
TradeJournal methods so that transactions stay in one place.

Usage pattern (in the app loop):

    engine = AutoPromotionEngine.from_settings(settings)

    # Promotion path
    decision = engine.evaluate_promotion(
        challenger_version=version,
        challenger_status=status,
        gate_stats=gate,
        champion_wf_bps=champion_wf,
        bootstrap_result=boot,
    )
    if decision.approved:
        await journal.promote_challenger_to_champion(version, event_data=decision.metrics_snapshot)
        await model_registry.load_active_model()

    # Degradation path (champion monitor, runs every N hours)
    deg = engine.evaluate_degradation(
        champion_version=version,
        gate_stats=champion_gate,
    )
    if deg.should_rollback:
        new_champ = await journal.rollback_champion(
            current_version=version,
            reason=deg.reason,
            event_data=deg.metrics_snapshot,
        )
        await model_registry.load_active_model()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

if TYPE_CHECKING:
    from trader.config import Settings

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class _BootstrapResult(Protocol):
    p_value: float
    mean_diff_bps: float
    n_iterations: int
    n_challenger: int
    n_baseline: int


# ---------------------------------------------------------------------------
# Decision dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of AutoPromotionEngine.evaluate_promotion()."""

    version: str
    approved: bool
    blocking_reasons: list[str]
    metrics_snapshot: dict[str, Any] = field(default_factory=dict)

    def log_summary(self) -> str:
        if self.approved:
            return (
                f"APPROVED version={self.version} "
                f"lift={self.metrics_snapshot.get('lift_bps', '?'):+.2f}bps "
                f"p={self.metrics_snapshot.get('bootstrap_p_value', '?')}"
            )
        return f"REJECTED version={self.version} reasons={self.blocking_reasons}"


@dataclass(frozen=True)
class DegradationDecision:
    """Outcome of AutoPromotionEngine.evaluate_degradation()."""

    champion_version: str
    should_rollback: bool
    reason: str
    metrics_snapshot: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class AutoPromotionEngine:
    """Evaluation engine for challenger promotion and champion health checks.

    All thresholds come from the constructor; use ``from_settings()`` to
    populate them from the application config.
    """

    # ---- Construction -----------------------------------------------------

    def __init__(
        self,
        *,
        # --- Promotion criteria ---
        min_signals: int = 50,
        min_gate_passes: int = 20,
        min_lift_bps: float = 1.0,
        min_pass_expectancy_bps: float = 0.0,
        min_paper_trades: int = 20,
        min_paper_total_bps: float = 0.0,
        max_paper_drawdown_bps: float = -50.0,
        pvalue_threshold: float = 0.05,
        # --- Champion degradation criteria ---
        champion_degrade_min_signals: int = 100,
        champion_min_lift_bps: float = -5.0,
        champion_min_pass_expectancy_bps: float = -20.0,
    ) -> None:
        self.min_signals = min_signals
        self.min_gate_passes = min_gate_passes
        self.min_lift_bps = min_lift_bps
        self.min_pass_expectancy_bps = min_pass_expectancy_bps
        self.min_paper_trades = min_paper_trades
        self.min_paper_total_bps = min_paper_total_bps
        self.max_paper_drawdown_bps = max_paper_drawdown_bps
        self.pvalue_threshold = pvalue_threshold
        self.champion_degrade_min_signals = champion_degrade_min_signals
        self.champion_min_lift_bps = champion_min_lift_bps
        self.champion_min_pass_expectancy_bps = champion_min_pass_expectancy_bps

    @classmethod
    def from_settings(cls, settings: Settings) -> AutoPromotionEngine:
        return cls(
            min_signals=max(10, int(settings.MODEL_AUTO_PROMOTE_MIN_SIGNALS)),
            min_gate_passes=max(1, int(getattr(settings, "MODEL_AUTO_PROMOTE_MIN_GATE_PASSES", 20))),
            min_lift_bps=float(settings.MODEL_AUTO_PROMOTE_MIN_LIFT_BPS),
            min_pass_expectancy_bps=float(getattr(settings, "MODEL_AUTO_PROMOTE_MIN_PASS_EXPECTANCY_BPS", 0.0)),
            min_paper_trades=max(1, int(getattr(settings, "MODEL_AUTO_PROMOTE_MIN_PAPER_TRADES", 20))),
            min_paper_total_bps=float(getattr(settings, "MODEL_AUTO_PROMOTE_MIN_PAPER_TOTAL_BPS", 0.0)),
            max_paper_drawdown_bps=float(getattr(settings, "MODEL_AUTO_PROMOTE_MAX_PAPER_DRAWDOWN_BPS", -50.0)),
            pvalue_threshold=float(settings.MODEL_AUTO_PROMOTE_PVALUE_THRESHOLD),
            champion_degrade_min_signals=max(
                10, int(getattr(settings, "MODEL_CHAMPION_DEGRADE_MIN_SIGNALS", 100))
            ),
            champion_min_lift_bps=float(
                getattr(settings, "MODEL_CHAMPION_MIN_LIFT_BPS", -5.0)
            ),
        )

    # ---- Promotion evaluation ---------------------------------------------

    def evaluate_promotion(
        self,
        *,
        challenger_version: str,
        challenger_status: str,
        gate_stats: dict[str, Any],
        paper_stats: dict[str, Any] | None = None,
        champion_wf_bps: float,
        bootstrap_result: _BootstrapResult | None,
    ) -> PromotionDecision:
        """Evaluate all promotion criteria (A-D) and return a decision.

        Criteria:
          A. Status must be SHADOW_CHALLENGER.
          B. Enough live shadow-gate observations (>= min_signals).
          C. Live lift >= min_lift_bps AND quality == "GOOD".
          D. Challenger beats champion's walk-forward expectancy.
          E. Bootstrap p-value < pvalue_threshold (statistical significance).
        """
        blocking: list[str] = []

        # A — Status
        if challenger_status != "SHADOW_CHALLENGER":
            blocking.append(f"status_not_shadow_challenger:{challenger_status}")

        total_count = int(gate_stats.get("total_count") or 0)
        pass_count = int(gate_stats.get("pass_count") or 0)
        lift_bps = float(gate_stats.get("lift_vs_all_bps") or 0.0)
        pass_expectancy_raw = gate_stats.get("pass_avg_net_return_bps")
        pass_expectancy_bps = float(pass_expectancy_raw) if pass_expectancy_raw is not None else None
        quality = str(gate_stats.get("quality") or "").upper()
        paper_required = paper_stats is not None
        resolved_paper_stats: dict[str, Any] = paper_stats or {}
        paper_gate_value = resolved_paper_stats.get("model_gate")
        paper_gate: dict[str, Any] = (
            paper_gate_value if isinstance(paper_gate_value, dict) else resolved_paper_stats
        )
        paper_count = int(paper_gate.get("count") or 0)
        paper_total_bps = float(paper_gate.get("total_bps") or 0.0)
        paper_drawdown_bps = float(paper_gate.get("max_drawdown_bps") or 0.0)

        # B — Observations
        if total_count < self.min_signals:
            blocking.append(f"insufficient_signals:{total_count}<{self.min_signals}")
        if pass_count < self.min_gate_passes:
            blocking.append(f"insufficient_gate_passes:{pass_count}<{self.min_gate_passes}")

        # C — Lift and quality
        if lift_bps < self.min_lift_bps:
            blocking.append(f"insufficient_lift:{lift_bps:.2f}<{self.min_lift_bps:.2f}")
        if pass_expectancy_bps is None or pass_expectancy_bps < self.min_pass_expectancy_bps:
            blocking.append(
                f"insufficient_pass_expectancy:{pass_expectancy_bps}<"
                f"{self.min_pass_expectancy_bps:.2f}"
            )
        if quality != "GOOD":
            blocking.append(f"quality_not_good:{quality or 'UNKNOWN'}")

        # C2 — Paper economics must be net positive, not just statistically pretty.
        if paper_required:
            if paper_count < self.min_paper_trades:
                blocking.append(f"insufficient_paper_trades:{paper_count}<{self.min_paper_trades}")
            if paper_total_bps <= self.min_paper_total_bps:
                blocking.append(f"insufficient_paper_total:{paper_total_bps:.2f}<={self.min_paper_total_bps:.2f}")
            if paper_drawdown_bps < self.max_paper_drawdown_bps:
                blocking.append(
                    f"paper_drawdown_too_deep:{paper_drawdown_bps:.2f}<"
                    f"{self.max_paper_drawdown_bps:.2f}"
                )

        # D — Must beat champion
        if lift_bps <= champion_wf_bps:
            blocking.append(
                f"not_better_than_champion:lift={lift_bps:.2f}<=champ_wf={champion_wf_bps:.2f}"
            )

        # E — Statistical significance
        if bootstrap_result is None:
            blocking.append("bootstrap_not_run")
        elif bootstrap_result.p_value >= self.pvalue_threshold:
            blocking.append(
                f"lift_not_significant:p={bootstrap_result.p_value:.4f}>={self.pvalue_threshold}"
            )

        approved = len(blocking) == 0
        snap: dict[str, Any] = {
            "total_count": total_count,
            "pass_count": pass_count,
            "lift_bps": round(lift_bps, 4),
            "pass_expectancy_bps": round(pass_expectancy_bps, 4) if pass_expectancy_bps is not None else None,
            "paper_count": paper_count,
            "paper_total_bps": round(paper_total_bps, 4),
            "paper_drawdown_bps": round(paper_drawdown_bps, 4),
            "quality": quality,
            "champion_wf_bps": round(champion_wf_bps, 4),
            "bootstrap_p_value": (
                round(bootstrap_result.p_value, 6) if bootstrap_result else None
            ),
            "bootstrap_n_challenger": (
                bootstrap_result.n_challenger if bootstrap_result else None
            ),
            "bootstrap_mean_diff_bps": (
                round(bootstrap_result.mean_diff_bps, 4) if bootstrap_result else None
            ),
        }

        if approved:
            log.info("auto_promotion.approved", version=challenger_version, **snap)
        else:
            log.debug(
                "auto_promotion.rejected",
                version=challenger_version,
                reasons=blocking,
            )

        return PromotionDecision(
            version=challenger_version,
            approved=approved,
            blocking_reasons=blocking,
            metrics_snapshot=snap,
        )

    # ---- Degradation / rollback evaluation --------------------------------

    def evaluate_degradation(
        self,
        *,
        champion_version: str,
        gate_stats: dict[str, Any],
    ) -> DegradationDecision:
        """Detect if the current champion has degraded below acceptable thresholds.

        Returns should_rollback=True only after enough live observations so
        that transient noise does not trigger unnecessary rollbacks.
        """
        total_count = int(gate_stats.get("total_count") or 0)
        lift_raw = gate_stats.get("lift_vs_all_bps")
        pass_avg_raw = gate_stats.get("pass_avg_net_return_bps")

        snap: dict[str, Any] = {
            "total_count": total_count,
            "lift_bps": round(float(lift_raw), 4) if lift_raw is not None else None,
            "pass_avg_bps": (
                round(float(pass_avg_raw), 4) if pass_avg_raw is not None else None
            ),
        }

        if total_count < self.champion_degrade_min_signals:
            return DegradationDecision(
                champion_version=champion_version,
                should_rollback=False,
                reason=(
                    f"insufficient_observations:"
                    f"{total_count}<{self.champion_degrade_min_signals}"
                ),
                metrics_snapshot=snap,
            )

        lift_bps = float(lift_raw) if lift_raw is not None else 0.0

        if lift_bps < self.champion_min_lift_bps:
            log.warning(
                "auto_promotion.champion_degraded",
                champion=champion_version,
                lift_bps=lift_bps,
                floor=self.champion_min_lift_bps,
            )
            return DegradationDecision(
                champion_version=champion_version,
                should_rollback=True,
                reason=(
                    f"lift_degraded:{lift_bps:.2f}bps"
                    f"<floor:{self.champion_min_lift_bps:.2f}bps"
                ),
                metrics_snapshot=snap,
            )

        if pass_avg_raw is not None and float(pass_avg_raw) < self.champion_min_pass_expectancy_bps:
            log.warning(
                "auto_promotion.champion_negative_expectancy",
                champion=champion_version,
                pass_avg_bps=float(pass_avg_raw),
            )
            return DegradationDecision(
                champion_version=champion_version,
                should_rollback=True,
                reason=f"negative_pass_expectancy:{float(pass_avg_raw):.2f}bps",
                metrics_snapshot=snap,
            )

        log.debug(
            "auto_promotion.champion_healthy",
            champion=champion_version,
            lift_bps=lift_bps,
            total_count=total_count,
        )
        return DegradationDecision(
            champion_version=champion_version,
            should_rollback=False,
            reason="champion_healthy",
            metrics_snapshot=snap,
        )
