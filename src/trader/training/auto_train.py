"""Automatic model training when labelled real data is ready.

Usage:
    python -m trader.training.auto_train
    python -m trader.training.auto_train --force
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import click

from trader.training.labels import active_label_schema_version

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_CANDIDATE_HORIZONS = (5, 15, 30, 60)


@dataclass(frozen=True)
class TrainableSnapshot:
    horizon_minutes: int
    sample_count: int


@dataclass(frozen=True)
class AutoTrainDecision:
    should_train: bool
    reason: str
    horizon_minutes: int
    trainable_samples: int
    min_samples: int


async def _pool() -> Any:
    import asyncpg

    from trader.config import Settings

    settings = Settings()
    dsn = settings.POSTGRES_DSN.get_secret_value().replace("postgresql+asyncpg://", "postgresql://", 1)
    return await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2, statement_cache_size=0)


async def count_trainable_by_horizon(pool: Any) -> list[TrainableSnapshot]:
    from trader.config import Settings
    from trader.training.sample_counts import fetch_training_snapshots_for_horizons

    settings = Settings()
    allowlist = [item.strip() for item in settings.TRAIN_STRATEGY_ALLOWLIST.split(",") if item.strip()] or None
    include_candle = bool(settings.TRAIN_INCLUDE_CANDLE_BASELINE)
    label_schema = active_label_schema_version(use_tpsl_exit=bool(settings.MODEL_LABEL_USE_TPSL_EXIT))
    label_threshold = float(settings.MODEL_AUTO_TRAIN_LABEL_BPS)

    async def fetch(query: str, *args: Any) -> list[Any]:
        return list(await pool.fetch(query, *args))

    snapshots = await fetch_training_snapshots_for_horizons(
        fetch,
        horizons=_CANDIDATE_HORIZONS,
        label_schema_version=label_schema,
        label_threshold_bps=label_threshold,
        strategy_allowlist=allowlist,
        include_candle_baseline=include_candle,
        min_samples=1,
    )
    return [
        TrainableSnapshot(horizon_minutes=int(horizon), sample_count=snapshot.best_schema_count)
        for horizon, snapshot in snapshots.items()
    ]


def resolve_training_horizon(
    snapshots: list[TrainableSnapshot], preferred: int, *, min_samples: int
) -> TrainableSnapshot | None:
    eligible = [item for item in snapshots if item.sample_count >= min_samples]
    if not eligible:
        return None
    by_horizon = {item.horizon_minutes: item for item in eligible}
    if preferred in by_horizon:
        return by_horizon[preferred]
    for horizon in _CANDIDATE_HORIZONS:
        if horizon in by_horizon:
            return by_horizon[horizon]
    return eligible[0]


async def latest_model_quality(pool: Any) -> str:
    row = await pool.fetchrow(
        """
        SELECT COALESCE(metrics->>'quality', 'WEAK') AS quality
        FROM model_versions
        WHERE artifact IS NOT NULL
        ORDER BY training_finished_at DESC NULLS LAST, created_at DESC
        LIMIT 1
        """
    )
    return str(row["quality"]).upper() if row else "NONE"


async def decide_auto_train(
    pool: Any,
    *,
    min_samples: int,
    preferred_horizon: int,
    force: bool = False,
) -> AutoTrainDecision:
    snapshots = await count_trainable_by_horizon(pool)
    chosen = resolve_training_horizon(snapshots, preferred_horizon, min_samples=min_samples)
    if chosen is None:
        return AutoTrainDecision(
            should_train=False,
            reason="no_horizon_with_enough_labelled_samples",
            horizon_minutes=preferred_horizon,
            trainable_samples=0,
            min_samples=min_samples,
        )

    if force:
        return AutoTrainDecision(
            should_train=True,
            reason="forced",
            horizon_minutes=chosen.horizon_minutes,
            trainable_samples=chosen.sample_count,
            min_samples=min_samples,
        )

    quality = await latest_model_quality(pool)
    if quality in {"NONE", "WEAK"}:
        return AutoTrainDecision(
            should_train=True,
            reason=f"model_quality_{quality.lower()}",
            horizon_minutes=chosen.horizon_minutes,
            trainable_samples=chosen.sample_count,
            min_samples=min_samples,
        )

    return AutoTrainDecision(
        should_train=False,
        reason=f"model_quality_{quality.lower()}_and_not_forced",
        horizon_minutes=chosen.horizon_minutes,
        trainable_samples=chosen.sample_count,
        min_samples=min_samples,
    )


def run_training_subprocess(*, min_samples: int, horizon: int, label_bps: float) -> int:
    cmd = [
        sys.executable,
        "-m",
        "trader.training.train",
        "--min-samples",
        str(min_samples),
        "--horizon",
        str(horizon),
        "--label-bps",
        str(label_bps),
    ]
    log.info("Starting training: %s", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


async def _auto_train(force: bool, min_samples: int, horizon: int, label_bps: float) -> int:
    from trader.config import Settings

    settings = Settings()
    min_samples = max(50, int(min_samples or settings.MODEL_AUTO_TRAIN_MIN_SAMPLES))
    preferred_horizon = int(horizon or settings.MODEL_AUTO_TRAIN_HORIZON_MINUTES or settings.MODEL_LABEL_HORIZON)
    label_bps = float(label_bps or settings.MODEL_AUTO_TRAIN_LABEL_BPS)

    pool = await _pool()
    try:
        snapshots = await count_trainable_by_horizon(pool)
        for item in snapshots:
            log.info("Trainable horizon=%sm samples=%s", item.horizon_minutes, item.sample_count)

        decision = await decide_auto_train(
            pool,
            min_samples=min_samples,
            preferred_horizon=preferred_horizon,
            force=force,
        )
        log.info(
            "Auto-train decision: train=%s reason=%s horizon=%sm samples=%s min=%s",
            decision.should_train,
            decision.reason,
            decision.horizon_minutes,
            decision.trainable_samples,
            decision.min_samples,
        )
        if not decision.should_train:
            return 0
        return run_training_subprocess(
            min_samples=min(decision.min_samples, decision.trainable_samples),
            horizon=decision.horizon_minutes,
            label_bps=label_bps,
        )
    finally:
        await pool.close()


@click.command()
@click.option("--force", is_flag=True, help="Train even if the latest model is not WEAK")
@click.option(
    "--min-samples", default=0, type=int, help="Minimum labelled samples (default: MODEL_AUTO_TRAIN_MIN_SAMPLES)"
)
@click.option(
    "--horizon", default=0, type=int, help="Preferred horizon minutes (default: MODEL_AUTO_TRAIN_HORIZON_MINUTES)"
)
@click.option(
    "--label-bps", default=0.0, type=float, help="Label threshold in bps (default: MODEL_AUTO_TRAIN_LABEL_BPS)"
)
def main(force: bool, min_samples: int, horizon: int, label_bps: float) -> None:
    """Train automatically when real labelled samples are ready."""
    raise SystemExit(asyncio.run(_auto_train(force, min_samples, horizon, label_bps)))


if __name__ == "__main__":
    main()
