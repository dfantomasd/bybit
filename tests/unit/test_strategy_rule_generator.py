from __future__ import annotations

import numpy as np

from trader.strategy_lab.rule_generator import RuleSearchConfig, discover_rules


def test_discover_rules_finds_positive_out_of_sample_feature_threshold() -> None:
    rng = np.random.default_rng(7)
    momentum = np.tile(np.linspace(-1.0, 1.0, 60), 4)
    noise = rng.normal(0.0, 1.0, 240)
    values = np.column_stack([momentum, noise])
    returns = np.where(momentum >= 0.55, 24.0, -18.0)

    report = discover_rules(
        values=values,
        returns_bps=returns,
        feature_names=["momentum", "noise"],
        config=RuleSearchConfig(
            min_train_count=20,
            min_validation_count=10,
            min_validation_avg_net_bps=1.0,
            top_n=5,
        ),
    )

    assert report["status"] == "ok"
    assert report["rules"]
    best = report["rules"][0]
    assert best["validation_avg_net_bps"] > 0
    assert best["validation_positive_folds"] >= 2
    assert best["validation_worst_fold_avg_net_bps"] > 0
    assert any(condition["feature"] == "momentum" and condition["op"] == ">=" for condition in best["conditions"])


def test_discover_rules_rejects_rules_that_remain_negative_after_costs() -> None:
    feature = np.linspace(-1.0, 1.0, 200)
    values = feature.reshape(-1, 1)
    returns = np.where(feature >= 0.5, -2.0, -30.0)

    report = discover_rules(
        values=values,
        returns_bps=returns,
        feature_names=["weak_edge"],
        config=RuleSearchConfig(
            min_train_count=20,
            min_validation_count=10,
            min_validation_avg_net_bps=0.0,
        ),
    )

    assert report["status"] == "ok"
    assert report["rules"] == []


def test_discover_rules_rejects_one_off_validation_spike() -> None:
    feature = np.ones(240)
    values = feature.reshape(-1, 1)
    returns = np.full(240, -8.0)
    returns[:168] = 4.0  # train looks mildly profitable
    returns[168:192] = 40.0  # one validation fold spikes
    returns[192:] = -10.0  # later validation folds fail

    report = discover_rules(
        values=values,
        returns_bps=returns,
        feature_names=["always_on"],
        config=RuleSearchConfig(
            min_train_count=20,
            min_validation_count=10,
            min_validation_avg_net_bps=0.0,
            min_validation_positive_folds=2,
            validation_folds=3,
        ),
    )

    assert report["status"] == "ok"
    assert report["rules"] == []
