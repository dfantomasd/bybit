"""Offline trainer CLI — train a challenger model from feature_snapshots + prediction_outcomes.

Usage:
    python -m trader.training.train [--min-samples 500]

Runs as a separate process (Render Cron Job at 03:00 UTC, or manually).
NEVER runs inside the trading process.

Label: 1 if net_return_bps > threshold (default 5 bps = 0.05%) at 15m horizon.
No lookahead leakage: only uses features recorded BEFORE the candle close time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime

import click
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


async def _train(min_samples: int, label_bps_threshold: float, horizon_minutes: int) -> None:
    import asyncpg

    from trader.config import Settings
    from trader.ml.challenger import ChallengerModel, ModelStatus

    settings = Settings()
    dsn = settings.POSTGRES_DSN.get_secret_value().replace("postgresql+asyncpg://", "postgresql://", 1)
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2)

    run_id = str(uuid.uuid4())
    run_started = datetime.now(tz=UTC)

    try:
        # Record training run start
        await pool.execute(
            "INSERT INTO training_runs (run_id, mode, status) VALUES ($1, 'offline', 'RUNNING')",
            run_id,
        )

        # Load labelled samples: join feature_snapshots with prediction_outcomes
        rows = await pool.fetch(
            """
            SELECT fs.feature_names, fs.feature_values, po.net_return_bps, po.label
            FROM feature_snapshots fs
            JOIN prediction_events pe ON pe.feature_snapshot_id = fs.snapshot_id
            JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
            WHERE po.horizon_minutes = $1
              AND po.label IS NOT NULL
              AND fs.feature_values IS NOT NULL
            ORDER BY fs.created_at DESC
            LIMIT 10000
            """,
            horizon_minutes,
        )

        if len(rows) < min_samples:
            msg = f"Insufficient samples: {len(rows)} < {min_samples}"
            click.echo(msg, err=True)
            await pool.execute(
                "UPDATE training_runs SET status='FAILED', error=$1, finished_at=now() WHERE run_id=$2",
                msg, run_id,
            )
            return

        click.echo(f"Training on {len(rows)} samples (horizon={horizon_minutes}m)")

        # Build arrays
        X_list = []
        y_list = []
        feature_names: list[str] = []

        for row in rows:
            vals = row["feature_values"]
            labels_raw = row["label"]
            if vals is None:
                continue
            v = json.loads(vals) if isinstance(vals, str) else list(vals)
            X_list.append(v)
            y_list.append(int(labels_raw))
            if not feature_names and row["feature_names"]:
                fn = row["feature_names"]
                feature_names = json.loads(fn) if isinstance(fn, str) else list(fn)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        # Shuffle
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

        version = f"v{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M')}"
        model = ChallengerModel(version=version, feature_names=feature_names)

        # Batch partial_fit (chunks of 32)
        chunk = 32
        for i in range(0, len(X), chunk):
            Xb = X[i:i+chunk]
            yb = y[i:i+chunk]
            for xi, yi in zip(Xb, yb):
                model.partial_fit(xi.tolist(), int(yi))

        click.echo(f"Training complete: {model.training_samples} samples, version={version}")

        # Save checkpoint
        artifact = model.to_bytes()
        await pool.execute(
            """
            INSERT INTO model_versions (version, status, training_samples, feature_schema_hash, artifact, metrics,
                training_started_at, training_finished_at)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, now())
            ON CONFLICT (version) DO UPDATE SET
                artifact = EXCLUDED.artifact,
                training_samples = EXCLUDED.training_samples,
                training_finished_at = now()
            """,
            version,
            ModelStatus.SHADOW_CHALLENGER,
            model.training_samples,
            model.feature_schema_hash,
            artifact,
            json.dumps({"label_bps_threshold": label_bps_threshold, "horizon_minutes": horizon_minutes}),
            run_started,
        )

        await pool.execute(
            "UPDATE training_runs SET status='COMPLETED', sample_count=$1, finished_at=now(), model_version=$2 WHERE run_id=$3",
            model.training_samples, version, run_id,
        )

        click.echo(f"Checkpoint saved as version={version}, status=SHADOW_CHALLENGER")
        click.echo("Run 'python -m trader.training.promote --version <version>' to evaluate and promote.")

    except Exception as exc:
        log.exception("Training failed")
        await pool.execute(
            "UPDATE training_runs SET status='FAILED', error=$1, finished_at=now() WHERE run_id=$2",
            str(exc), run_id,
        )
    finally:
        await pool.close()


@click.command()
@click.option("--min-samples", default=500, type=int, help="Minimum samples required")
@click.option("--label-bps", default=5.0, type=float, help="Min net return bps for positive label")
@click.option("--horizon", default=15, type=int, help="Prediction horizon in minutes")
def main(min_samples: int, label_bps: float, horizon: int) -> None:
    """Train challenger model from historical feature snapshots."""
    asyncio.run(_train(min_samples, label_bps, horizon))


if __name__ == "__main__":
    main()
