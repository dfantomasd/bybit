"""Tests for batch training of the challenger model."""

from __future__ import annotations

import numpy as np

from trader.ml.challenger import ChallengerModel


def _imbalanced_separable(n: int = 1000, positive_rate: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 10%-positive dataset where feature 0 carries the signal."""

    rng = np.random.default_rng(7)
    y = (rng.random(n) < positive_rate).astype(np.int32)
    signal = y.astype(np.float32) * 2.0 - 1.0
    x = rng.normal(0.0, 1.0, size=(n, 5)).astype(np.float32)
    x[:, 0] = signal + rng.normal(0.0, 0.5, size=n).astype(np.float32)
    # Feature scales differ wildly — the scaler must handle this.
    x[:, 1] *= 1000.0
    x[:, 2] *= 0.001
    return x, y


class TestFitBatch:
    def test_learns_imbalanced_separable_signal(self) -> None:
        x, y = _imbalanced_separable()
        model = ChallengerModel(version="v_test", feature_names=[f"f{i}" for i in range(5)])
        model.fit_batch(x[:800], y[:800])
        assert model.training_samples == 800

        scores = []
        for row in x[800:]:
            pred = model.predict(row.tolist())
            assert pred is not None
            scores.append(pred.score)
        scores_arr = np.array(scores)
        val_y = y[800:]
        # Positives must score clearly higher than negatives on held-out data.
        assert scores_arr[val_y == 1].mean() > scores_arr[val_y == 0].mean() + 0.2

    def test_refit_replaces_previous_state(self) -> None:
        x, y = _imbalanced_separable()
        model = ChallengerModel(version="v_test", feature_names=[f"f{i}" for i in range(5)])
        model.fit_batch(x[:300], y[:300])
        model.fit_batch(x, y)
        assert model.training_samples == len(x)

    def test_online_partial_fit_still_works_after_batch(self) -> None:
        x, y = _imbalanced_separable(n=200)
        model = ChallengerModel(version="v_test", feature_names=[f"f{i}" for i in range(5)])
        model.fit_batch(x, y)
        before = model.training_samples
        model.partial_fit(x[0].tolist(), int(y[0]))
        assert model.training_samples == before + 1
        assert model.predict(x[1].tolist()) is not None

    def test_empty_batch_is_noop(self) -> None:
        model = ChallengerModel(version="v_test", feature_names=["f0"])
        model.fit_batch(np.empty((0, 1), dtype=np.float32), np.empty((0,), dtype=np.int32))
        assert model.training_samples == 0

    def test_mismatched_lengths_are_rejected(self) -> None:
        model = ChallengerModel(version="v_test", feature_names=["f0", "f1"])
        model.fit_batch(np.zeros((10, 2), dtype=np.float32), np.zeros((5,), dtype=np.int32))
        assert model.training_samples == 0


class TestGbdtModel:
    def test_gbdt_learns_separable_signal(self) -> None:
        x, y = _imbalanced_separable()
        model = ChallengerModel(
            version="v_test_gbdt",
            feature_names=[f"f{i}" for i in range(5)],
            model_type="GBDT",
        )
        model.fit_batch(x[:800], y[:800])
        assert model.training_samples == 800

        scores = []
        for row in x[800:]:
            pred = model.predict(row.tolist())
            assert pred is not None
            scores.append(pred.score)
        scores_arr = np.array(scores)
        val_y = y[800:]
        assert scores_arr[val_y == 1].mean() > scores_arr[val_y == 0].mean() + 0.2

    def test_gbdt_partial_fit_is_noop(self) -> None:
        x, y = _imbalanced_separable(n=200)
        model = ChallengerModel(
            version="v_test_gbdt",
            feature_names=[f"f{i}" for i in range(5)],
            model_type="GBDT",
        )
        model.fit_batch(x, y)
        before = model.training_samples
        model.partial_fit(x[0].tolist(), int(y[0]))
        assert model.training_samples == before  # online updates skipped

    def test_gbdt_roundtrip_serialization(self) -> None:
        x, y = _imbalanced_separable(n=300)
        model = ChallengerModel(
            version="v_test_gbdt",
            feature_names=[f"f{i}" for i in range(5)],
            model_type="GBDT",
        )
        model.fit_batch(x, y)
        restored = ChallengerModel.from_bytes(model.to_bytes(), version="v_test_gbdt")
        assert restored.model_type == "GBDT"
        assert restored.training_samples == len(x)
        original = model.predict(x[0].tolist())
        roundtrip = restored.predict(x[0].tolist())
        assert original is not None and roundtrip is not None
        assert roundtrip.score == original.score

    def test_legacy_artifact_defaults_to_sgd(self) -> None:
        x, y = _imbalanced_separable(n=200)
        model = ChallengerModel(version="v_legacy", feature_names=[f"f{i}" for i in range(5)])
        model.fit_batch(x, y)
        payload = model.to_bytes()
        restored = ChallengerModel.from_bytes(payload, version="v_legacy")
        assert restored.model_type == "SGD"
