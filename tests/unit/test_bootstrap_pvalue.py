"""Tests for the bootstrap significance test used by the auto-promoter."""

from __future__ import annotations

import random

import pytest

from trader.training.bootstrap import bootstrap_pvalue


class TestBootstrapPvalue:
    def test_strong_edge_is_significant(self) -> None:
        rng = random.Random(2)
        challenger = [rng.gauss(10, 10) for _ in range(200)]
        baseline = [rng.gauss(0, 10) for _ in range(200)]
        result = bootstrap_pvalue(challenger, baseline)
        assert result.p_value < 0.05
        assert result.mean_diff_bps > 0
        assert result.n_iterations == 1000

    def test_no_edge_is_not_significant(self) -> None:
        rng = random.Random(3)
        challenger = [rng.gauss(0, 10) for _ in range(200)]
        baseline = [rng.gauss(0, 10) for _ in range(200)]
        result = bootstrap_pvalue(challenger, baseline)
        assert result.p_value > 0.05

    def test_deterministic_with_seed(self) -> None:
        challenger = [1.0, 2.0, 3.0, 4.0, 5.0]
        baseline = [0.0, 1.0, 2.0, 3.0, 4.0]
        r1 = bootstrap_pvalue(challenger, baseline, seed=42)
        r2 = bootstrap_pvalue(challenger, baseline, seed=42)
        assert r1.p_value == r2.p_value

    def test_empty_samples_raise(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_pvalue([], [1.0, 2.0])
        with pytest.raises(ValueError):
            bootstrap_pvalue([1.0, 2.0], [])

    def test_sample_counts_reported(self) -> None:
        result = bootstrap_pvalue([1.0] * 60, [0.5] * 80, n_iter=100)
        assert result.n_challenger == 60
        assert result.n_baseline == 80
        assert result.n_iterations == 100
