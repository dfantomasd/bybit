"""Tests for _summarise_walk_forward metrics aggregation."""

from __future__ import annotations

import pytest

from trader.training.train import _summarise_walk_forward


def _fold(bps: float, threshold: float = 0.5, pass_rate: float = 0.1) -> dict:
    return {
        "best_threshold_avg_net_return_bps": bps,
        "best_threshold": threshold,
        "best_threshold_pass_rate": pass_rate,
        "precision": 0.6,
        "lift_bps": bps * 0.5,
        "selected_pass_count": 10,
    }


def test_summarise_wf_mean_bps_is_average() -> None:
    folds = [_fold(10.0), _fold(20.0), _fold(-5.0)]
    result = _summarise_walk_forward(folds)
    assert abs(result["wf_mean_bps"] - (10 + 20 - 5) / 3) < 0.01


def test_summarise_wf_positive_folds_counts_positive() -> None:
    folds = [_fold(5.0), _fold(-1.0), _fold(3.0), _fold(-2.0)]
    result = _summarise_walk_forward(folds)
    assert result["wf_positive_folds"] == 2
    assert result["wf_folds"] == 4


def test_summarise_wf_min_max() -> None:
    folds = [_fold(1.0), _fold(5.0), _fold(-3.0)]
    result = _summarise_walk_forward(folds)
    assert result["wf_min_bps"] == pytest.approx(-3.0, abs=0.01)
    assert result["wf_max_bps"] == pytest.approx(5.0, abs=0.01)


def test_summarise_wf_empty_folds_returns_none_metrics() -> None:
    result = _summarise_walk_forward([])
    assert result["wf_mean_bps"] is None
    assert result["wf_positive_folds"] == 0
    assert result["wf_folds"] == 0


def test_summarise_wf_selected_threshold_is_median() -> None:
    folds = [
        _fold(1.0, threshold=0.3),
        _fold(2.0, threshold=0.7),
        _fold(3.0, threshold=0.5),
    ]
    result = _summarise_walk_forward(folds)
    assert result["selected_score_threshold"] == pytest.approx(0.5, abs=0.01)
