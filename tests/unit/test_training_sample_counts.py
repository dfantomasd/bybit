"""Tests for canonical training sample counting."""

from __future__ import annotations

from typing import Any

import pytest

from trader.training.sample_counts import fetch_training_sample_snapshot


@pytest.mark.asyncio
async def test_training_snapshot_uses_best_schema_count_for_readiness() -> None:
    calls: list[str] = []

    async def mock_fetch(query: str, *args: Any) -> list[dict[str, Any]]:
        calls.append(query)
        if "GROUP BY feature_schema_hash" in query:
            return [
                {"feature_schema_hash": "new_schema", "sample_count": 120, "latest_at": "2026-06-21"},
                {"feature_schema_hash": "old_schema", "sample_count": 900, "latest_at": "2026-06-20"},
            ]
        if "GROUP BY pool" in query:
            return [{"pool": "scalp_micro_v1", "sample_count": 120}]
        if "GROUP BY label_threshold_bps" in query:
            return [{"threshold": "2.0", "sample_count": 120}]
        return []

    snapshot = await fetch_training_sample_snapshot(
        mock_fetch,
        horizon_minutes=5,
        label_schema_version="directional_net_v2",
        label_threshold_bps=2.0,
        strategy_allowlist=["scalp_micro_v1"],
        include_candle_baseline=False,
        min_samples=1000,
    )

    assert snapshot.filtered_distinct_candles == 1020
    assert snapshot.best_schema_count == 900
    assert snapshot.best_schema_hash == "old_schema"
    assert snapshot.newest_schema_count == 120
    assert snapshot.newest_schema_hash == "new_schema"
    assert snapshot.training_ready is False
    assert snapshot.by_strategy_pool["scalp_micro_v1"] == 120
    assert len(calls) == 3
