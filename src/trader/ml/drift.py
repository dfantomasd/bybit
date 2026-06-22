"""Feature-distribution drift detection (PSI) for auto-retrain triggers."""

from __future__ import annotations

import math
from typing import Any, Sequence


def population_stability_index(
    baseline: Sequence[float],
    current: Sequence[float],
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute PSI between two numeric samples.

    Returns 0 when distributions match; >0.25 is commonly treated as significant drift.
    """
    if not baseline or not current:
        return 0.0

    b = [float(x) for x in baseline if x is not None and math.isfinite(float(x))]
    c = [float(x) for x in current if x is not None and math.isfinite(float(x))]
    if len(b) < 5 or len(c) < 5:
        return 0.0

    lo = min(min(b), min(c))
    hi = max(max(b), max(c))
    if hi <= lo:
        return 0.0

    width = (hi - lo) / bins
    if width <= 0:
        return 0.0

    def _hist(values: list[float]) -> list[float]:
        counts = [0] * bins
        for v in values:
            idx = min(bins - 1, max(0, int((v - lo) / width)))
            counts[idx] += 1
        total = float(len(values))
        return [max(epsilon, c / total) for c in counts]

    base_pct = _hist(b)
    curr_pct = _hist(c)
    psi = 0.0
    for bp, cp in zip(base_pct, curr_pct, strict=True):
        psi += (cp - bp) * math.log(cp / bp)
    return max(0.0, psi)


def drift_summary_from_samples(
    baseline_samples: Sequence[Sequence[float]],
    current_samples: Sequence[Sequence[float]],
    *,
    feature_index: int = 0,
    psi_threshold: float = 0.25,
) -> dict[str, Any]:
    """Compare one feature column across baseline vs current snapshot vectors."""
    baseline_vals = [
        row[feature_index] for row in baseline_samples if isinstance(row, (list, tuple)) and len(row) > feature_index
    ]
    current_vals = [
        row[feature_index] for row in current_samples if isinstance(row, (list, tuple)) and len(row) > feature_index
    ]
    psi = population_stability_index(baseline_vals, current_vals)
    return {
        "feature_index": feature_index,
        "psi": round(psi, 4),
        "drift_detected": psi >= psi_threshold,
        "baseline_count": len(baseline_vals),
        "current_count": len(current_vals),
    }
