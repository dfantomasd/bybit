"""Cost-aware offline rule generator for strategy discovery.

The generator searches simple, explainable feature-threshold rules over
already-labelled prediction outcomes. It is deliberately offline-only: rules
found here are candidates for review/shadow collection, not live trading.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RuleCondition:
    feature: str
    op: str
    threshold: float

    def describe(self) -> str:
        return f"{self.feature} {self.op} {self.threshold:.6g}"


@dataclass(frozen=True)
class RuleCandidate:
    rule_id: str
    conditions: tuple[RuleCondition, ...]
    train_count: int
    train_avg_net_bps: float
    train_lift_bps: float
    validation_count: int
    validation_avg_net_bps: float | None
    validation_lift_bps: float | None
    pass_rate: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["conditions"] = [asdict(condition) for condition in self.conditions]
        data["description"] = " AND ".join(condition.describe() for condition in self.conditions)
        return data


@dataclass(frozen=True)
class RuleSearchConfig:
    min_train_count: int = 30
    min_validation_count: int = 10
    min_train_avg_net_bps: float = 0.0
    min_validation_avg_net_bps: float = 0.0
    max_candidates_per_feature: int = 8
    top_n: int = 20
    quantiles: tuple[float, ...] = (0.10, 0.20, 0.30, 0.70, 0.80, 0.90)
    pair_top_n: int = 30


def _finite_column_mask(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values)


def _condition_mask(values: np.ndarray, condition: RuleCondition, feature_index: dict[str, int]) -> np.ndarray:
    column = values[:, feature_index[condition.feature]]
    finite = _finite_column_mask(column)
    if condition.op == ">=":
        return finite & (column >= condition.threshold)
    if condition.op == "<=":
        return finite & (column <= condition.threshold)
    raise ValueError(f"unsupported condition op: {condition.op!r}")


def _rule_mask(values: np.ndarray, conditions: tuple[RuleCondition, ...], feature_index: dict[str, int]) -> np.ndarray:
    if not conditions:
        return np.ones(len(values), dtype=bool)
    mask = np.ones(len(values), dtype=bool)
    for condition in conditions:
        mask &= _condition_mask(values, condition, feature_index)
    return mask


def _score_rule(
    *,
    values: np.ndarray,
    returns_bps: np.ndarray,
    feature_names: list[str],
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    conditions: tuple[RuleCondition, ...],
    config: RuleSearchConfig,
) -> RuleCandidate | None:
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    mask = _rule_mask(values, conditions, feature_index)
    train_mask = mask[train_idx]
    validation_mask = mask[validation_idx]
    train_count = int(train_mask.sum())
    validation_count = int(validation_mask.sum())
    if train_count < config.min_train_count or validation_count < config.min_validation_count:
        return None

    train_returns = returns_bps[train_idx][train_mask]
    validation_returns = returns_bps[validation_idx][validation_mask]
    train_avg = float(np.mean(train_returns))
    validation_avg = float(np.mean(validation_returns))
    train_baseline = float(np.mean(returns_bps[train_idx]))
    validation_baseline = float(np.mean(returns_bps[validation_idx]))
    train_lift = train_avg - train_baseline
    validation_lift = validation_avg - validation_baseline
    if train_avg < config.min_train_avg_net_bps:
        return None
    if validation_avg < config.min_validation_avg_net_bps:
        return None

    pass_rate = float(validation_count / len(validation_idx)) if len(validation_idx) else 0.0
    scarcity_penalty = 5.0 / max(validation_count, 1) ** 0.5
    score = validation_avg + min(validation_lift, 50.0) * 0.25 - scarcity_penalty
    rule_id = "rule_" + "_and_".join(
        f"{condition.feature}_{'ge' if condition.op == '>=' else 'le'}_{condition.threshold:.6g}"
        for condition in conditions
    )
    return RuleCandidate(
        rule_id=rule_id,
        conditions=conditions,
        train_count=train_count,
        train_avg_net_bps=train_avg,
        train_lift_bps=train_lift,
        validation_count=validation_count,
        validation_avg_net_bps=validation_avg,
        validation_lift_bps=validation_lift,
        pass_rate=pass_rate,
        score=score,
    )


def _single_feature_conditions(
    values: np.ndarray,
    feature_names: list[str],
    train_idx: np.ndarray,
    quantiles: tuple[float, ...],
) -> list[RuleCondition]:
    conditions: list[RuleCondition] = []
    for col_idx, feature in enumerate(feature_names):
        column = values[train_idx, col_idx]
        column = column[np.isfinite(column)]
        if len(column) < 10:
            continue
        thresholds = sorted({float(np.quantile(column, q)) for q in quantiles})
        for threshold in thresholds:
            conditions.append(RuleCondition(feature=feature, op=">=", threshold=threshold))
            conditions.append(RuleCondition(feature=feature, op="<=", threshold=threshold))
    return conditions


def discover_rules(
    *,
    values: np.ndarray,
    returns_bps: np.ndarray,
    feature_names: list[str],
    train_fraction: float = 0.70,
    config: RuleSearchConfig | None = None,
) -> dict[str, Any]:
    """Return top cost-aware rules using chronological train/validation split."""

    cfg = config or RuleSearchConfig()
    x = np.asarray(values, dtype=float)
    y = np.asarray(returns_bps, dtype=float)
    if x.ndim != 2:
        raise ValueError("values must be a 2D array")
    if len(x) != len(y):
        raise ValueError(f"values/returns length mismatch: {len(x)} != {len(y)}")
    if x.shape[1] != len(feature_names):
        raise ValueError(f"feature count mismatch: {x.shape[1]} != {len(feature_names)}")
    if len(x) < cfg.min_train_count + cfg.min_validation_count:
        return {
            "status": "insufficient_samples",
            "sample_count": int(len(x)),
            "required_samples": int(cfg.min_train_count + cfg.min_validation_count),
            "rules": [],
        }

    split = min(max(int(len(x) * train_fraction), cfg.min_train_count), len(x) - cfg.min_validation_count)
    train_idx = np.arange(0, split)
    validation_idx = np.arange(split, len(x))
    baseline_train = float(np.mean(y[train_idx])) if len(train_idx) else None
    baseline_validation = float(np.mean(y[validation_idx])) if len(validation_idx) else None

    scored: list[RuleCandidate] = []
    single_conditions = _single_feature_conditions(x, feature_names, train_idx, cfg.quantiles)
    for condition in single_conditions:
        candidate = _score_rule(
            values=x,
            returns_bps=y,
            feature_names=feature_names,
            train_idx=train_idx,
            validation_idx=validation_idx,
            conditions=(condition,),
            config=cfg,
        )
        if candidate is not None:
            scored.append(candidate)

    top_single = sorted(scored, key=lambda item: item.score, reverse=True)[: cfg.pair_top_n]
    pair_candidates: list[RuleCandidate] = []
    for left_idx, left in enumerate(top_single):
        for right in top_single[left_idx + 1 :]:
            left_features = {condition.feature for condition in left.conditions}
            right_features = {condition.feature for condition in right.conditions}
            if left_features & right_features:
                continue
            conditions = left.conditions + right.conditions
            candidate = _score_rule(
                values=x,
                returns_bps=y,
                feature_names=feature_names,
                train_idx=train_idx,
                validation_idx=validation_idx,
                conditions=conditions,
                config=cfg,
            )
            if candidate is not None:
                pair_candidates.append(candidate)

    all_candidates = sorted(scored + pair_candidates, key=lambda item: item.score, reverse=True)
    return {
        "status": "ok",
        "sample_count": int(len(x)),
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(validation_idx)),
        "baseline_train_avg_net_bps": baseline_train,
        "baseline_validation_avg_net_bps": baseline_validation,
        "rules_tested": int(len(single_conditions) + len(top_single) * max(len(top_single) - 1, 0) / 2),
        "rules": [candidate.to_dict() for candidate in all_candidates[: cfg.top_n]],
    }
