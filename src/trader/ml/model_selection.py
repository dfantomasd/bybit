"""Shared scoring helpers for model selection and diagnostics."""

from __future__ import annotations

from typing import Any


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def metric_float(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value: Any = metrics
        for part in key.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        parsed = _float_or_none(value)
        if parsed is not None:
            return parsed
    return None


def metric_int(metrics: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value: Any = metrics
        for part in key.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        parsed = _int_or_zero(value)
        if parsed:
            return parsed
    return 0


def model_selection_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return normalized selection metrics used by registry, promotion, and UI."""

    wf_bps = metric_float(
        metrics,
        "walk_forward_expectancy_bps",
        "wf_mean_bps",
        "best_threshold_avg_net_return_bps",
    )
    lift_bps = metric_float(metrics, "lift_bps", "paper_gate.lift_bps", "model_gate.lift_bps") or 0.0
    precision = metric_float(metrics, "precision", "paper_gate.precision", "model_gate.precision") or 0.0
    paper_gate_count = metric_int(
        metrics,
        "paper_gate.count",
        "model_gate.count",
        "paper_gate_count",
    )
    walk_forward_pass_count = metric_int(
        metrics,
        "total_pass_count",
        "best_threshold_pass_count",
        "walk_forward_pass_count",
    )
    pass_count = paper_gate_count or walk_forward_pass_count
    drawdown_bps = abs(
        metric_float(
            metrics,
            "max_drawdown_bps",
            "paper_gate.max_drawdown_bps",
            "model_gate.max_drawdown_bps",
        )
        or 0.0
    )
    wf_component = wf_bps if wf_bps is not None else -1_000_000.0
    score = wf_component + (lift_bps * 0.35) + (precision * 2.0) + min(pass_count, 500) / 25.0
    score -= min(drawdown_bps, 3000.0) / 300.0
    return {
        "model_score": round(float(score), 6),
        "walk_forward_bps": wf_bps,
        "lift_bps": lift_bps,
        "precision": precision,
        "paper_gate_count": paper_gate_count,
        "walk_forward_pass_count": walk_forward_pass_count,
        "pass_count_for_score": pass_count,
        "drawdown_bps": drawdown_bps,
    }


def selection_reason(metrics: dict[str, Any], *, min_paper_gate_count: int = 50) -> str:
    normalized = model_selection_metrics(metrics)
    wf_bps = normalized["walk_forward_bps"]
    paper_count = int(normalized["paper_gate_count"] or 0)
    if wf_bps is None:
        return "fallback:no_walk_forward"
    if paper_count < min_paper_gate_count:
        return f"blocked:paper_gate_count<{min_paper_gate_count}"
    if float(wf_bps) > 0:
        return "selected:positive_walk_forward_lift"
    return "selected:least_negative_walk_forward"
