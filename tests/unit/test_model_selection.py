from __future__ import annotations

from trader.ml.model_selection import model_selection_metrics, selection_reason


def test_model_selection_score_rewards_walk_forward_lift_and_sample_count() -> None:
    weaker = model_selection_metrics(
        {
            "walk_forward_expectancy_bps": 1.0,
            "lift_bps": 1.0,
            "precision": 0.4,
            "paper_gate": {"count": 50},
        }
    )
    stronger = model_selection_metrics(
        {
            "walk_forward_expectancy_bps": 4.0,
            "lift_bps": 6.0,
            "precision": 0.5,
            "paper_gate": {"count": 120},
        }
    )

    assert stronger["model_score"] > weaker["model_score"]
    assert stronger["walk_forward_bps"] == 4.0
    assert stronger["paper_gate_count"] == 120


def test_model_selection_does_not_confuse_walk_forward_pass_with_paper_gate() -> None:
    normalized = model_selection_metrics(
        {
            "walk_forward_expectancy_bps": 3.0,
            "total_pass_count": 120,
            "best_threshold_pass_count": 95,
        }
    )
    assert normalized["paper_gate_count"] == 0
    assert normalized["walk_forward_pass_count"] == 120
    assert normalized["pass_count_for_score"] == 120
    assert (
        selection_reason({"walk_forward_expectancy_bps": 3.0, "total_pass_count": 120})
        == "blocked:paper_gate_count<50"
    )


def test_model_selection_prefers_explicit_paper_gate_over_walk_forward_pass() -> None:
    normalized = model_selection_metrics(
        {
            "walk_forward_expectancy_bps": 3.0,
            "paper_gate": {"count": 55},
            "total_pass_count": 120,
        }
    )
    assert normalized["paper_gate_count"] == 55
    assert normalized["walk_forward_pass_count"] == 120
    assert normalized["pass_count_for_score"] == 55


def test_selection_reason_distinguishes_missing_and_immature_evidence() -> None:
    assert selection_reason({}) == "fallback:no_walk_forward"
    assert (
        selection_reason({"walk_forward_expectancy_bps": 2.0, "paper_gate": {"count": 12}})
        == "blocked:paper_gate_count<50"
    )
    assert (
        selection_reason({"walk_forward_expectancy_bps": 2.0, "paper_gate": {"count": 50}})
        == "selected:positive_walk_forward_lift"
    )
