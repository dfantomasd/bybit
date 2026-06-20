"""Tests for offline walk-forward model-selection helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trader.ml.challenger import ChallengerModel
from trader.training.train import (
    _candidate_specs,
    _filter_timestamps_by_mask,
    _negative_bucket_keep_mask,
    _validate_walk_forward_chronology,
    _walk_forward_splits,
)


def test_walk_forward_splits_expand_train_window() -> None:
    x = np.zeros((20, 3), dtype=np.float32)

    folds = _walk_forward_splits(x, min_train_samples=8, n_folds=3)

    assert len(folds) == 3
    assert folds[0][0].tolist() == list(range(8))
    assert folds[0][1].tolist() == list(range(8, 12))
    assert folds[1][0].tolist() == list(range(12))
    assert folds[1][1].tolist() == list(range(12, 16))


def test_walk_forward_splits_fallback_when_too_small() -> None:
    x = np.zeros((6, 2), dtype=np.float32)

    folds = _walk_forward_splits(x, min_train_samples=5, n_folds=5)

    assert len(folds) == 1
    assert folds[0][0].tolist() == [0, 1, 2, 3, 4]
    assert folds[0][1].tolist() == [5]


def test_walk_forward_chronology_requires_validation_after_train() -> None:
    x = np.zeros((10, 2), dtype=np.float32)
    folds = _walk_forward_splits(x, min_train_samples=5, n_folds=2)
    timestamps = [datetime(2026, 6, 13, tzinfo=UTC) + timedelta(minutes=i) for i in range(10)]

    windows = _validate_walk_forward_chronology(folds, timestamps)

    assert windows[0]["train_end_at"] < windows[0]["val_start_at"]
    assert windows[0]["train_samples"] == 5


def test_walk_forward_chronology_rejects_overlapping_windows() -> None:
    folds = [(np.array([0, 1]), np.array([1, 2]))]
    timestamps = [datetime(2026, 6, 13, tzinfo=UTC)] * 3

    with pytest.raises(RuntimeError, match="chronology violation"):
        _validate_walk_forward_chronology(folds, timestamps)


def test_candidate_specs_include_requested_families_only() -> None:
    specs = _candidate_specs("LOGREG")

    assert specs == [
        {"model_type": "LOGREG", "C": 0.1},
        {"model_type": "LOGREG", "C": 1.0},
        {"model_type": "LOGREG", "C": 10.0},
    ]


def test_negative_bucket_filter_excludes_stable_losers() -> None:
    returns = np.array([-10.0, -8.0, -9.0, 2.0, 3.0, 4.0], dtype=np.float32)
    regimes = np.array(["bad", "bad", "bad", "good", "good", "good"], dtype=object)
    hours = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)
    volatility = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3], dtype=np.float32)

    keep, excluded = _negative_bucket_keep_mask(
        returns_bps=returns,
        regimes=regimes,
        hours=hours,
        volatility_values=volatility,
        min_bucket_samples=3,
        min_bucket_avg_bps=-5.0,
    )

    assert keep.tolist() == [False, False, False, True, True, True]
    assert excluded[0]["regime"] == "bad"
    assert excluded[0]["count"] == 3


def test_negative_bucket_filter_keeps_timestamps_aligned() -> None:
    timestamps = [datetime(2026, 6, 13, tzinfo=UTC) + timedelta(minutes=i) for i in range(6)]
    keep = np.array([False, True, False, True, True, False], dtype=bool)

    filtered = _filter_timestamps_by_mask(timestamps, keep)

    assert filtered == [timestamps[1], timestamps[3], timestamps[4]]


def test_negative_bucket_filter_rejects_timestamp_length_mismatch() -> None:
    timestamps = [datetime(2026, 6, 13, tzinfo=UTC)]
    keep = np.array([True, False], dtype=bool)

    with pytest.raises(RuntimeError, match="timestamp/filter length mismatch"):
        _filter_timestamps_by_mask(timestamps, keep)


def test_walk_forward_chronology_rejects_missing_timestamps() -> None:
    folds = [(np.array([0, 1]), np.array([2]))]
    timestamps = [datetime(2026, 6, 13, tzinfo=UTC), datetime(2026, 6, 13, 0, 1, tzinfo=UTC)]

    with pytest.raises(RuntimeError, match="timestamp mismatch"):
        _validate_walk_forward_chronology(folds, timestamps)


def test_challenger_model_params_roundtrip() -> None:
    model = ChallengerModel(
        version="v-test",
        feature_names=["a", "b"],
        model_type="LOGREG",
        model_params={"C": 0.1},
    )

    restored = ChallengerModel.from_bytes(model.to_bytes(), "v-test")

    assert restored.model_type == "LOGREG"
    assert restored.model_params == {"C": 0.1}
