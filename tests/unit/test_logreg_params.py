"""Tests for LOGREG model training with different C parameters."""

from __future__ import annotations

import numpy as np
import pytest

from trader.ml.challenger import ChallengerModel


def _make_dataset(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(np.int32)
    x = rng.normal(size=(n, 3)).astype(np.float32)
    x[:, 0] += y * 2.0
    return x, y


@pytest.mark.parametrize("c_value", [0.1, 1.0, 10.0])
def test_logreg_trains_with_different_c(c_value: float) -> None:
    x, y = _make_dataset()
    model = ChallengerModel(
        version="v_test",
        feature_names=["f0", "f1", "f2"],
        model_type="LOGREG",
        model_params={"C": c_value},
    )
    model.fit_batch(x, y)
    assert model.training_samples == len(x)
    pred = model.predict(x[0].tolist())
    assert pred is not None
    assert 0.0 <= pred.score <= 1.0


def test_logreg_params_preserved_after_serialization() -> None:
    model = ChallengerModel(
        version="v_test",
        feature_names=["a", "b"],
        model_type="LOGREG",
        model_params={"C": 0.1},
    )
    x, y = _make_dataset(n=100)
    model.fit_batch(x[:, :2], y)

    restored = ChallengerModel.from_bytes(model.to_bytes(), "v_test")
    assert restored.model_type == "LOGREG"
    assert restored.model_params.get("C") == pytest.approx(0.1)


def test_logreg_high_c_less_regularized() -> None:
    """High C (less regularization) should fit training data better."""
    x, y = _make_dataset(n=500)

    model_tight = ChallengerModel(
        version="v",
        feature_names=["f0", "f1", "f2"],
        model_type="LOGREG",
        model_params={"C": 0.01},
    )
    model_loose = ChallengerModel(
        version="v",
        feature_names=["f0", "f1", "f2"],
        model_type="LOGREG",
        model_params={"C": 100.0},
    )

    model_tight.fit_batch(x, y)
    model_loose.fit_batch(x, y)

    scores_tight = [model_tight.predict(row.tolist()).score for row in x[y == 1][:20]]
    scores_loose = [model_loose.predict(row.tolist()).score for row in x[y == 1][:20]]

    # Loose regularization should score positives higher on training data
    assert np.mean(scores_loose) >= np.mean(scores_tight) - 0.05
