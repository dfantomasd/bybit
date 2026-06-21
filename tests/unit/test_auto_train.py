"""Tests for training eligibility and auto-train horizon selection."""

from __future__ import annotations

from trader.training.auto_train import TrainableSnapshot, resolve_training_horizon
from trader.training.eligibility import training_strategy_filter_sql


def test_training_strategy_filter_includes_candle_baselines_when_enabled() -> None:
    sql = training_strategy_filter_sql("$4", "$5")
    assert "SHADOW_CANDLE" in sql
    assert "HISTORICAL_REAL" in sql
    assert "$5::boolean IS TRUE" in sql


def test_training_strategy_filter_exclusive_when_allowlist_set() -> None:
    sql = training_strategy_filter_sql("$4", "$5")
    assert "strategy_id' = ANY($4::text[])" in sql
    assert "cardinality($4::text[]) = 0" in sql


def test_resolve_training_horizon_prefers_configured_horizon() -> None:
    snapshots = [
        TrainableSnapshot(horizon_minutes=5, sample_count=141_808),
        TrainableSnapshot(horizon_minutes=15, sample_count=10_000),
    ]
    chosen = resolve_training_horizon(snapshots, preferred=5, min_samples=500)
    assert chosen is not None
    assert chosen.horizon_minutes == 5


def test_resolve_training_horizon_falls_back_when_preferred_missing() -> None:
    snapshots = [TrainableSnapshot(horizon_minutes=5, sample_count=141_808)]
    chosen = resolve_training_horizon(snapshots, preferred=30, min_samples=500)
    assert chosen is not None
    assert chosen.horizon_minutes == 5


def test_auto_train_module_importable() -> None:
    from trader.training import auto_train

    assert hasattr(auto_train, "main")
