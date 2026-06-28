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
    symbol: str | None = None
    side: str | None = None
    segment: str = "all"

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


def discover_segmented_rules(
    *,
    values: np.ndarray,
    returns_bps: np.ndarray,
    feature_names: list[str],
    symbols: list[str] | None = None,
    sides: list[str] | None = None,
    train_fraction: float = 0.70,
    config: RuleSearchConfig | None = None,
    top_n_per_segment: int = 5,
) -> dict[str, Any]:
    """Discover rules for all/side/symbol+side segments.

    The live runtime needs a direction. Searching side-aware segments prevents a
    single aggregate rule from mixing Buy and Sell economics. Symbol+side
    segments are included only when they have enough samples for the configured
    train/validation split.
    """

    cfg = config or RuleSearchConfig()
    x = np.asarray(values, dtype=float)
    y = np.asarray(returns_bps, dtype=float)
    if symbols is not None and len(symbols) != len(x):
        raise ValueError(f"symbols length mismatch: {len(symbols)} != {len(x)}")
    if sides is not None and len(sides) != len(x):
        raise ValueError(f"sides length mismatch: {len(sides)} != {len(x)}")

    segments: list[tuple[str, str | None, str | None, np.ndarray]] = [
        ("all", None, None, np.arange(len(x))),
    ]
    if sides is not None:
        for side in sorted({str(item) for item in sides if item}):
            idx = np.asarray([i for i, item in enumerate(sides) if str(item) == side], dtype=int)
            segments.append((f"side:{side}", None, side, idx))
    if symbols is not None and sides is not None:
        keys = sorted({(str(symbols[i]), str(sides[i])) for i in range(len(x)) if symbols[i] and sides[i]})
        for symbol, side in keys:
            idx = np.asarray(
                [i for i in range(len(x)) if str(symbols[i]) == symbol and str(sides[i]) == side],
                dtype=int,
            )
            segments.append((f"symbol_side:{symbol}:{side}", symbol, side, idx))

    discovered: list[dict[str, Any]] = []
    segment_reports: dict[str, dict[str, Any]] = {}
    min_required = cfg.min_train_count + cfg.min_validation_count
    for segment_name, symbol, side, idx in segments:
        if len(idx) < min_required:
            segment_reports[segment_name] = {
                "status": "insufficient_samples",
                "sample_count": int(len(idx)),
                "required_samples": int(min_required),
            }
            continue
        report = discover_rules(
            values=x[idx],
            returns_bps=y[idx],
            feature_names=feature_names,
            train_fraction=train_fraction,
            config=cfg,
        )
        segment_reports[segment_name] = {
            key: value for key, value in report.items() if key != "rules"
        }
        if side is None:
            # Runtime-discovered rules must be directional. Keep aggregate
            # segment diagnostics, but do not emit side-less rules into the
            # executable rule list.
            continue
        for rule in list(report.get("rules") or [])[:top_n_per_segment]:
            enriched = dict(rule)
            enriched["segment"] = segment_name
            enriched["symbol"] = symbol
            enriched["side"] = side
            # Keep ids stable but readable across segments.
            enriched["rule_id"] = f"{segment_name}:{rule.get('rule_id', 'rule')}"
            discovered.append(enriched)

    discovered.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return {
        "status": "ok" if discovered else "no_rules",
        "sample_count": int(len(x)),
        "segments_tested": len(segments),
        "segment_reports": segment_reports,
        "rules": discovered[: cfg.top_n],
    }
