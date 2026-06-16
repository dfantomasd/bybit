"""Bootstrap significance test for model promotion.

Used by the auto-promoter to verify that a challenger's observed lift over the
baseline is statistically significant rather than noise. A simple bootstrap
difference-of-means: resample both return distributions with replacement,
compute the mean difference per iteration, and report the one-sided p-value as
the fraction of iterations where the challenger does NOT beat the baseline.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BootstrapResult:
    p_value: float
    mean_diff_bps: float
    n_iterations: int
    n_challenger: int
    n_baseline: int


def bootstrap_pvalue(
    samples_challenger: Sequence[float],
    samples_baseline: Sequence[float],
    n_iter: int = 1000,
    seed: int | None = 42,
) -> BootstrapResult:
    """One-sided bootstrap test of mean(challenger) > mean(baseline).

    Args:
        samples_challenger: Net returns (bps) of challenger-approved signals.
        samples_baseline:   Net returns (bps) of all baseline signals.
        n_iter:             Bootstrap iterations (default 1000).
        seed:               RNG seed for reproducible decisions (None = random).

    Returns:
        BootstrapResult with p_value = fraction of iterations where the
        resampled mean difference was <= 0. p_value < 0.05 means the
        challenger's edge is unlikely to be noise.

    Raises:
        ValueError: if either sample is empty.
    """
    challenger = [float(x) for x in samples_challenger]
    baseline = [float(x) for x in samples_baseline]
    if not challenger or not baseline:
        raise ValueError(
            f"bootstrap_pvalue requires non-empty samples (challenger={len(challenger)}, baseline={len(baseline)})"
        )

    rng = random.Random(seed)  # noqa: S311 - deterministic statistical resampling, not security-sensitive.
    n_c = len(challenger)
    n_b = len(baseline)
    observed_diff = (sum(challenger) / n_c) - (sum(baseline) / n_b)

    not_better = 0
    for _ in range(n_iter):
        c_mean = sum(rng.choices(challenger, k=n_c)) / n_c
        b_mean = sum(rng.choices(baseline, k=n_b)) / n_b
        if c_mean - b_mean <= 0:
            not_better += 1

    return BootstrapResult(
        p_value=not_better / n_iter,
        mean_diff_bps=observed_diff,
        n_iterations=n_iter,
        n_challenger=n_c,
        n_baseline=n_b,
    )
