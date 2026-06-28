from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np

from trader.domain.enums import OrderSide
from trader.domain.models import FeatureVector
from trader.strategies.discovered_rule import DiscoveredRuleStrategy, load_discovered_rules
from trader.strategy_lab.rule_generator import RuleSearchConfig, discover_segmented_rules


def _vec(symbol: str = "DOGEUSDT", value: float = 1.0) -> FeatureVector:
    return FeatureVector(
        symbol=symbol,
        timestamp=datetime.now(tz=UTC),
        feature_names=["edge_feature", "noise"],
        values=[value, 0.0],
        quality_score=1.0,
        lookback_bars=60,
    )


def test_segmented_discovery_emits_side_aware_positive_rules() -> None:
    values = np.asarray([[i / 100.0, 0.0] for i in range(120)], dtype=float)
    returns = np.asarray([20.0 if i >= 50 else -5.0 for i in range(120)], dtype=float)
    symbols = ["DOGEUSDT"] * 120
    sides = ["Buy"] * 120

    report = discover_segmented_rules(
        values=values,
        returns_bps=returns,
        feature_names=["edge_feature", "noise"],
        symbols=symbols,
        sides=sides,
        config=RuleSearchConfig(
            min_train_count=20,
            min_validation_count=10,
            min_validation_avg_net_bps=1.0,
            top_n=20,
        ),
    )

    assert report["status"] == "ok"
    assert report["rules"]
    assert report["rules"][0]["side"] == "Buy"
    assert any(rule["segment"] == "side:Buy" for rule in report["rules"])
    assert all(rule["validation_avg_net_bps"] > 0 for rule in report["rules"])


def test_discovered_rule_strategy_loads_and_emits_matching_shadow_signal(tmp_path) -> None:
    rules_path = tmp_path / "strategy_lab.json"
    rules_path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "rule_id": "r1",
                        "side": "Buy",
                        "symbol": "DOGEUSDT",
                        "validation_count": 25,
                        "validation_avg_net_bps": 12.5,
                        "validation_lift_bps": 15.0,
                        "score": 16.0,
                        "conditions": [{"feature": "edge_feature", "op": ">=", "threshold": 0.5}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rules = load_discovered_rules(rules_path)
    strategy = DiscoveredRuleStrategy(rules=rules)
    proposal = strategy.evaluate(_vec(value=0.7), current_price=0.10, available_balance_usd=1000.0)

    assert proposal is not None
    assert proposal.strategy_id == "discovered_rule_v1"
    assert proposal.side == OrderSide.BUY
    assert proposal.symbol == "DOGEUSDT"
    assert proposal.take_profit is not None
    assert proposal.stop_loss is not None


def test_discovered_rule_strategy_ignores_negative_or_wrong_side_rules(tmp_path) -> None:
    rules_path = tmp_path / "strategy_lab.json"
    rules_path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "rule_id": "bad",
                        "side": "Buy",
                        "validation_count": 25,
                        "validation_avg_net_bps": -1.0,
                        "conditions": [{"feature": "edge_feature", "op": ">=", "threshold": 0.5}],
                    },
                    {
                        "rule_id": "missing_side",
                        "validation_count": 25,
                        "validation_avg_net_bps": 10.0,
                        "conditions": [{"feature": "edge_feature", "op": ">=", "threshold": 0.5}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_discovered_rules(rules_path) == []
