"""Tests for data retention and drift helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from trader.ml.drift import drift_summary_from_samples, population_stability_index
from trader.storage.retention import RetentionSettings, get_pnl_attribution, run_data_retention


class TestDriftPsi:
    def test_similar_distributions_have_low_psi(self) -> None:
        baseline = [float(i % 10) for i in range(100)]
        current = [float(i % 10) for i in range(100)]
        psi = population_stability_index(baseline, current)
        assert psi < 0.25

    def test_shifted_distribution_raises_psi(self) -> None:
        baseline = [1.0] * 100
        current = [10.0] * 100
        psi = population_stability_index(baseline, current)
        assert psi >= 0.25

    def test_drift_summary_flags_shift(self) -> None:
        baseline = [[1.0]] * 100
        current = [[10.0]] * 100
        summary = drift_summary_from_samples(baseline, current, psi_threshold=0.25)
        assert summary["drift_detected"] is True


class TestRetentionRunner:
    @pytest.mark.asyncio
    async def test_run_data_retention_deletes_invalid_snapshots(self) -> None:
        store = AsyncMock()
        store._execute = AsyncMock(return_value="DELETE 3")
        store._fetch = AsyncMock(return_value=[])

        report = await run_data_retention(
            store,
            RetentionSettings(
                candle_retention_days={"1": 30},
                feature_snapshot_invalid_retention_days=7,
                export_enabled=False,
            ),
        )
        assert report.invalid_snapshots_deleted == 3
        assert any("feature_snapshots" in str(call.args[0]) for call in store._execute.await_args_list)

    @pytest.mark.asyncio
    async def test_get_pnl_attribution_shadow_fallback(self) -> None:
        store = AsyncMock()
        store._fetch = AsyncMock(
            side_effect=[
                [],
                [
                    {
                        "symbol": "DOGEUSDT",
                        "wins": 3,
                        "losses": 2,
                        "avg_bps": 1.5,
                        "samples": 5,
                        "source": "shadow",
                    }
                ],
            ]
        )
        rows = await get_pnl_attribution(store, days=7)
        assert rows[0]["symbol"] == "DOGEUSDT"
        assert rows[0]["source"] == "shadow"
