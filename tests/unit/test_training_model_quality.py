from __future__ import annotations

import numpy as np

from trader.training.train import _evaluate_model, _walk_forward_quality


class _FakeBatchModel:
    def __init__(self, scores: list[float], predictions: list[int]) -> None:
        self._scores = np.asarray(scores, dtype=float)
        self._predictions = np.asarray(predictions, dtype=np.int32)

    def predict_batch(self, _x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._scores, self._predictions


def test_negative_net_threshold_remains_weak_even_when_better_than_average() -> None:
    scores = [0.70] * 20 + [0.10] * 100
    predictions = [1] * 20 + [0] * 100
    labels = np.asarray([1] * 20 + [0] * 100, dtype=np.int32)
    returns_bps = np.asarray([-5.0] * 20 + [-30.0] * 100, dtype=float)
    sides = np.asarray(["Buy"] * 120)
    features = np.zeros((120, 2), dtype=float)

    metrics = _evaluate_model(
        _FakeBatchModel(scores=scores, predictions=predictions),
        features,
        labels,
        returns_bps,
        sides,
    )

    assert metrics["best_threshold_avg_net_return_bps"] == -5.0
    assert metrics["avg_net_return_predicted_positive_bps"] == -5.0
    assert metrics["quality"] == "WEAK"


def test_walk_forward_quality_requires_stable_positive_folds() -> None:
    stable = {
        "wf_mean_bps": 4.0,
        "wf_min_bps": 1.0,
        "wf_std_bps": 2.0,
        "wf_positive_folds": 3,
        "total_pass_count": 25,
    }
    high_std = {**stable, "wf_std_bps": 40.0}
    negative_tail = {**stable, "wf_mean_bps": 4.0, "wf_min_bps": -6.0, "wf_positive_folds": 3}

    kwargs = {
        "min_positive_folds": 3,
        "min_pass_count": 20,
        "min_wf_bps": 0.0,
        "max_wf_std_bps": 25.0,
    }

    assert _walk_forward_quality(stable, **kwargs) == "GOOD"
    assert _walk_forward_quality(high_std, **kwargs) == "WEAK"
    assert _walk_forward_quality(negative_tail, **kwargs) == "WEAK"
