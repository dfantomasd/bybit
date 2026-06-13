"""Unit tests for AutoPromotionEngine (pure evaluation, no I/O)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from trader.ml.auto_promotion import AutoPromotionEngine, DegradationDecision, PromotionDecision


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _engine(**kwargs) -> AutoPromotionEngine:
    defaults = dict(
        min_signals=50,
        min_lift_bps=1.0,
        pvalue_threshold=0.05,
        champion_degrade_min_signals=100,
        champion_min_lift_bps=-5.0,
        champion_min_pass_expectancy_bps=-20.0,
    )
    defaults.update(kwargs)
    return AutoPromotionEngine(**defaults)


@dataclass
class _Boot:
    p_value: float
    mean_diff_bps: float
    n_iterations: int
    n_challenger: int
    n_baseline: int


def _good_gate(*, total=60, lift=3.0, quality="GOOD") -> dict:
    return {
        "total_count": total,
        "lift_vs_all_bps": lift,
        "quality": quality,
        "pass_count": 40,
        "pass_avg_net_return_bps": 5.0,
    }


def _good_boot(p=0.02) -> _Boot:
    return _Boot(p_value=p, mean_diff_bps=2.5, n_iterations=1000, n_challenger=60, n_baseline=60)


# ---------------------------------------------------------------------------
# evaluate_promotion — all criteria pass
# ---------------------------------------------------------------------------


def test_promotion_approved_all_criteria():
    engine = _engine()
    decision = engine.evaluate_promotion(
        challenger_version="v1",
        challenger_status="SHADOW_CHALLENGER",
        gate_stats=_good_gate(),
        champion_wf_bps=1.5,
        bootstrap_result=_good_boot(),
    )
    assert isinstance(decision, PromotionDecision)
    assert decision.approved is True
    assert decision.blocking_reasons == []
    assert decision.version == "v1"
    assert "lift_bps" in decision.metrics_snapshot


# ---------------------------------------------------------------------------
# evaluate_promotion — each individual criterion blocks
# ---------------------------------------------------------------------------


def test_promotion_blocked_wrong_status():
    engine = _engine()
    decision = engine.evaluate_promotion(
        challenger_version="v1",
        challenger_status="CHAMPION",
        gate_stats=_good_gate(),
        champion_wf_bps=1.5,
        bootstrap_result=_good_boot(),
    )
    assert decision.approved is False
    assert any("status_not_shadow_challenger" in r for r in decision.blocking_reasons)


def test_promotion_blocked_insufficient_signals():
    engine = _engine(min_signals=50)
    decision = engine.evaluate_promotion(
        challenger_version="v1",
        challenger_status="SHADOW_CHALLENGER",
        gate_stats=_good_gate(total=30),
        champion_wf_bps=1.5,
        bootstrap_result=_good_boot(),
    )
    assert decision.approved is False
    assert any("insufficient_signals" in r for r in decision.blocking_reasons)


def test_promotion_blocked_insufficient_lift():
    engine = _engine(min_lift_bps=5.0)
    decision = engine.evaluate_promotion(
        challenger_version="v1",
        challenger_status="SHADOW_CHALLENGER",
        gate_stats=_good_gate(lift=2.0),
        champion_wf_bps=0.5,
        bootstrap_result=_good_boot(),
    )
    assert decision.approved is False
    assert any("insufficient_lift" in r for r in decision.blocking_reasons)


def test_promotion_blocked_weak_quality():
    engine = _engine()
    decision = engine.evaluate_promotion(
        challenger_version="v1",
        challenger_status="SHADOW_CHALLENGER",
        gate_stats=_good_gate(quality="WEAK"),
        champion_wf_bps=1.5,
        bootstrap_result=_good_boot(),
    )
    assert decision.approved is False
    assert any("quality_not_good" in r for r in decision.blocking_reasons)


def test_promotion_blocked_doesnt_beat_champion():
    engine = _engine()
    decision = engine.evaluate_promotion(
        challenger_version="v1",
        challenger_status="SHADOW_CHALLENGER",
        gate_stats=_good_gate(lift=3.0),
        champion_wf_bps=4.0,  # champion is better
        bootstrap_result=_good_boot(),
    )
    assert decision.approved is False
    assert any("not_better_than_champion" in r for r in decision.blocking_reasons)


def test_promotion_blocked_bootstrap_not_significant():
    engine = _engine(pvalue_threshold=0.05)
    decision = engine.evaluate_promotion(
        challenger_version="v1",
        challenger_status="SHADOW_CHALLENGER",
        gate_stats=_good_gate(),
        champion_wf_bps=1.5,
        bootstrap_result=_good_boot(p=0.10),  # p >= 0.05
    )
    assert decision.approved is False
    assert any("lift_not_significant" in r for r in decision.blocking_reasons)


def test_promotion_blocked_bootstrap_none():
    engine = _engine()
    decision = engine.evaluate_promotion(
        challenger_version="v1",
        challenger_status="SHADOW_CHALLENGER",
        gate_stats=_good_gate(),
        champion_wf_bps=1.5,
        bootstrap_result=None,
    )
    assert decision.approved is False
    assert any("bootstrap_not_run" in r for r in decision.blocking_reasons)


def test_promotion_multiple_blocking_reasons():
    """All criteria fail simultaneously — blocking_reasons lists all of them."""
    engine = _engine(min_signals=100)
    decision = engine.evaluate_promotion(
        challenger_version="v1",
        challenger_status="VALIDATED",
        gate_stats={"total_count": 5, "lift_vs_all_bps": -1.0, "quality": "WEAK"},
        champion_wf_bps=10.0,
        bootstrap_result=_good_boot(p=0.99),
    )
    assert decision.approved is False
    assert len(decision.blocking_reasons) >= 4


# ---------------------------------------------------------------------------
# log_summary
# ---------------------------------------------------------------------------


def test_log_summary_approved():
    engine = _engine()
    decision = engine.evaluate_promotion(
        challenger_version="v2",
        challenger_status="SHADOW_CHALLENGER",
        gate_stats=_good_gate(),
        champion_wf_bps=1.5,
        bootstrap_result=_good_boot(),
    )
    summary = decision.log_summary()
    assert "APPROVED" in summary
    assert "v2" in summary


def test_log_summary_rejected():
    engine = _engine()
    decision = engine.evaluate_promotion(
        challenger_version="v2",
        challenger_status="CHAMPION",
        gate_stats=_good_gate(),
        champion_wf_bps=1.5,
        bootstrap_result=_good_boot(),
    )
    summary = decision.log_summary()
    assert "REJECTED" in summary
    assert "v2" in summary


# ---------------------------------------------------------------------------
# evaluate_degradation — healthy champion
# ---------------------------------------------------------------------------


def test_degradation_healthy():
    engine = _engine(champion_degrade_min_signals=100, champion_min_lift_bps=-5.0)
    deg = engine.evaluate_degradation(
        champion_version="champ_v1",
        gate_stats={
            "total_count": 150,
            "lift_vs_all_bps": 2.0,
            "pass_avg_net_return_bps": 10.0,
        },
    )
    assert isinstance(deg, DegradationDecision)
    assert deg.should_rollback is False
    assert deg.champion_version == "champ_v1"


def test_degradation_insufficient_observations():
    engine = _engine(champion_degrade_min_signals=100)
    deg = engine.evaluate_degradation(
        champion_version="champ_v1",
        gate_stats={"total_count": 40, "lift_vs_all_bps": -10.0},
    )
    assert deg.should_rollback is False
    assert "insufficient_observations" in deg.reason


def test_degradation_lift_below_floor():
    engine = _engine(champion_degrade_min_signals=50, champion_min_lift_bps=-5.0)
    deg = engine.evaluate_degradation(
        champion_version="champ_v1",
        gate_stats={"total_count": 100, "lift_vs_all_bps": -8.0},
    )
    assert deg.should_rollback is True
    assert "lift_degraded" in deg.reason


def test_degradation_negative_pass_expectancy():
    engine = _engine(
        champion_degrade_min_signals=50,
        champion_min_lift_bps=-5.0,
        champion_min_pass_expectancy_bps=-20.0,
    )
    deg = engine.evaluate_degradation(
        champion_version="champ_v1",
        gate_stats={
            "total_count": 100,
            "lift_vs_all_bps": -3.0,  # above floor, won't trigger lift rollback
            "pass_avg_net_return_bps": -25.0,  # below -20 bps
        },
    )
    assert deg.should_rollback is True
    assert "negative_pass_expectancy" in deg.reason


def test_degradation_metrics_snapshot_populated():
    engine = _engine(champion_degrade_min_signals=50)
    deg = engine.evaluate_degradation(
        champion_version="champ_v1",
        gate_stats={"total_count": 80, "lift_vs_all_bps": 1.0},
    )
    assert "total_count" in deg.metrics_snapshot
    assert "lift_bps" in deg.metrics_snapshot


# ---------------------------------------------------------------------------
# from_settings
# ---------------------------------------------------------------------------


def test_from_settings():
    class FakeSettings:
        MODEL_AUTO_PROMOTE_MIN_SIGNALS = 75
        MODEL_AUTO_PROMOTE_MIN_LIFT_BPS = 2.5
        MODEL_AUTO_PROMOTE_PVALUE_THRESHOLD = 0.03
        MODEL_CHAMPION_DEGRADE_MIN_SIGNALS = 200
        MODEL_CHAMPION_MIN_LIFT_BPS = -3.0

    engine = AutoPromotionEngine.from_settings(FakeSettings())  # type: ignore[arg-type]
    assert engine.min_signals == 75
    assert engine.min_lift_bps == 2.5
    assert engine.pvalue_threshold == 0.03
    assert engine.champion_degrade_min_signals == 200
    assert engine.champion_min_lift_bps == -3.0


def test_from_settings_defaults_for_missing_champion_fields():
    """Settings without MODEL_CHAMPION_* still produce a valid engine via getattr fallback."""

    class MinimalSettings:
        MODEL_AUTO_PROMOTE_MIN_SIGNALS = 50
        MODEL_AUTO_PROMOTE_MIN_LIFT_BPS = 1.0
        MODEL_AUTO_PROMOTE_PVALUE_THRESHOLD = 0.05

    engine = AutoPromotionEngine.from_settings(MinimalSettings())  # type: ignore[arg-type]
    assert engine.champion_degrade_min_signals == 100  # default
    assert engine.champion_min_lift_bps == -5.0  # default
