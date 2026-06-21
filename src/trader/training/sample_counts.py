"""Canonical training-sample counts shared by diagnostics, auto-train, and train.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from trader.training.eligibility import training_strategy_filter_sql

FetchRows = Callable[..., Awaitable[list[Any]]]


@dataclass(frozen=True)
class TrainingSampleSnapshot:
    """Counts that mirror ``train.py`` eligibility and schema selection."""

    horizon_minutes: int
    filtered_distinct_candles: int
    best_schema_count: int
    best_schema_hash: str
    newest_schema_count: int
    newest_schema_hash: str
    training_ready: bool
    by_schema: dict[str, int]
    by_strategy_pool: dict[str, int]
    by_label_threshold: dict[str, int]


def _eligible_samples_cte(strategy_filter: str) -> str:
    return f"""
        WITH eligible_samples AS (
            SELECT
                fs.symbol,
                fs.interval,
                fs.candle_open_time,
                fs.feature_schema_hash,
                fs.created_at,
                pe.metadata,
                pe.decision,
                po.label_threshold_bps,
                ROW_NUMBER() OVER (
                    PARTITION BY fs.symbol, fs.interval, fs.candle_open_time, fs.feature_schema_hash
                    ORDER BY fs.created_at DESC, pe.created_at DESC
                ) AS candle_rank
            FROM feature_snapshots fs
            JOIN prediction_events pe ON pe.feature_snapshot_id = fs.snapshot_id
            JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
            WHERE po.horizon_minutes = $1
              AND po.label IS NOT NULL
              AND po.label_schema_version = $2
              AND po.label_threshold_bps = $3
              AND fs.feature_values IS NOT NULL
              AND fs.training_eligible = true
              AND pe.model_version = 'RULE_BASELINE_V1'
              AND pe.strategy_signal IN ('Buy', 'Sell')
              AND {strategy_filter}
        ),
        deduped AS (
            SELECT *
            FROM eligible_samples
            WHERE candle_rank = 1
        )
    """


async def fetch_training_sample_snapshot(
    fetch: FetchRows,
    *,
    horizon_minutes: int,
    label_schema_version: str,
    label_threshold_bps: float,
    strategy_allowlist: list[str] | None,
    include_candle_baseline: bool,
    min_samples: int,
) -> TrainingSampleSnapshot:
    """Return train-compatible sample counts for one horizon."""

    strategy_filter = training_strategy_filter_sql("$4", "$5")
    cte = _eligible_samples_cte(strategy_filter)
    bind = (
        horizon_minutes,
        label_schema_version,
        float(label_threshold_bps),
        strategy_allowlist,
        include_candle_baseline,
    )

    schema_rows = await fetch(
        f"""
        {cte}
        SELECT feature_schema_hash, count(*)::int AS sample_count, max(created_at) AS latest_at
        FROM deduped
        GROUP BY feature_schema_hash
        ORDER BY latest_at DESC
        """,
        *bind,
    )

    pool_rows = await fetch(
        f"""
        {cte}
        SELECT
            CASE
                WHEN metadata->>'strategy_id' = 'scalp_micro_v1' THEN 'scalp_micro_v1'
                WHEN metadata->>'strategy_id' IS NULL
                     AND COALESCE(decision, '') IN ('SHADOW_CANDLE', 'HISTORICAL_REAL')
                    THEN 'candle_baseline'
                ELSE COALESCE(metadata->>'strategy_id', 'other')
            END AS pool,
            count(*)::int AS sample_count
        FROM deduped
        GROUP BY pool
        ORDER BY pool
        """,
        *bind,
    )

    threshold_rows = await fetch(
        f"""
        WITH eligible_samples AS (
            SELECT
                po.label_threshold_bps,
                ROW_NUMBER() OVER (
                    PARTITION BY fs.symbol, fs.interval, fs.candle_open_time, fs.feature_schema_hash
                    ORDER BY fs.created_at DESC, pe.created_at DESC
                ) AS candle_rank
            FROM feature_snapshots fs
            JOIN prediction_events pe ON pe.feature_snapshot_id = fs.snapshot_id
            JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
            WHERE po.horizon_minutes = $1
              AND po.label IS NOT NULL
              AND po.label_schema_version = $2
              AND fs.feature_values IS NOT NULL
              AND fs.training_eligible = true
              AND pe.model_version = 'RULE_BASELINE_V1'
              AND pe.strategy_signal IN ('Buy', 'Sell')
              AND {strategy_filter}
        )
        SELECT label_threshold_bps::text AS threshold, count(*)::int AS sample_count
        FROM eligible_samples
        WHERE candle_rank = 1
        GROUP BY label_threshold_bps
        ORDER BY label_threshold_bps
        """,
        horizon_minutes,
        label_schema_version,
        strategy_allowlist,
        include_candle_baseline,
    )

    by_schema = {str(row["feature_schema_hash"]): int(row["sample_count"]) for row in schema_rows}
    filtered_distinct = sum(by_schema.values())
    newest_schema_hash = str(schema_rows[0]["feature_schema_hash"]) if schema_rows else ""
    newest_schema_count = int(schema_rows[0]["sample_count"]) if schema_rows else 0
    best_schema_hash = ""
    best_schema_count = 0
    if schema_rows:
        best_row = max(schema_rows, key=lambda row: int(row["sample_count"]))
        best_schema_hash = str(best_row["feature_schema_hash"])
        best_schema_count = int(best_row["sample_count"])

    return TrainingSampleSnapshot(
        horizon_minutes=horizon_minutes,
        filtered_distinct_candles=filtered_distinct,
        best_schema_count=best_schema_count,
        best_schema_hash=best_schema_hash,
        newest_schema_count=newest_schema_count,
        newest_schema_hash=newest_schema_hash,
        training_ready=best_schema_count >= min_samples,
        by_schema=by_schema,
        by_strategy_pool={str(row["pool"]): int(row["sample_count"]) for row in pool_rows},
        by_label_threshold={str(row["threshold"]): int(row["sample_count"]) for row in threshold_rows},
    )


async def fetch_training_snapshots_for_horizons(
    fetch: FetchRows,
    *,
    horizons: tuple[int, ...],
    label_schema_version: str,
    label_threshold_bps: float,
    strategy_allowlist: list[str] | None,
    include_candle_baseline: bool,
    min_samples: int,
) -> dict[str, TrainingSampleSnapshot]:
    snapshots: dict[str, TrainingSampleSnapshot] = {}
    for horizon in horizons:
        snapshots[str(horizon)] = await fetch_training_sample_snapshot(
            fetch,
            horizon_minutes=horizon,
            label_schema_version=label_schema_version,
            label_threshold_bps=label_threshold_bps,
            strategy_allowlist=strategy_allowlist,
            include_candle_baseline=include_candle_baseline,
            min_samples=min_samples,
        )
    return snapshots
