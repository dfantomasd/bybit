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
from typing import Any

import click
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _fit_model(model: Any, x_arr: np.ndarray, y: np.ndarray) -> None:
    chunk = 32
    for i in range(0, len(x_arr), chunk):
        xb = x_arr[i : i + chunk]
        yb = y[i : i + chunk]
        for xi, yi in zip(xb, yb, strict=False):
            model.partial_fit(xi.tolist(), int(yi))


def _evaluate_model(model: Any, x_val: np.ndarray, y_val: np.ndarray, returns_bps: np.ndarray) -> dict[str, Any]:
    if len(x_val) == 0:
        return {
            "quality": "n/a",
            "validation_samples": 0,
        }

    scores: list[float] = []
    preds: list[int] = []
    for features in x_val:
        pred = model.predict(features.tolist())
        if pred is None:
            scores.append(0.0)
            preds.append(0)
            continue
        scores.append(float(pred.score))
        preds.append(int(pred.label))

    pred_arr = np.array(preds, dtype=np.int32)
    score_arr = np.array(scores, dtype=np.float32)
    positives = y_val == 1
    predicted_positive = pred_arr == 1
    true_positive = predicted_positive & positives

    precision = float(true_positive.sum() / predicted_positive.sum()) if predicted_positive.sum() else 0.0
    recall = float(true_positive.sum() / positives.sum()) if positives.sum() else 0.0
    accuracy = float((pred_arr == y_val).mean())
    positive_rate = float(positives.mean())
    predicted_positive_rate = float(predicted_positive.mean())
    avg_all = float(np.mean(returns_bps)) if len(returns_bps) else 0.0
    avg_predicted_positive = float(np.mean(returns_bps[predicted_positive])) if predicted_positive.sum() else None
    lift_bps = (avg_predicted_positive - avg_all) if avg_predicted_positive is not None else None

    quality = "INSUFFICIENT_VALIDATION"
    if len(x_val) >= 100:
        quality = "GOOD" if precision > positive_rate and lift_bps is not None and lift_bps > 0 else "WEAK"

    return {
        "accuracy": accuracy,
        "avg_model_score": float(score_arr.mean()) if len(score_arr) else 0.0,
        "avg_net_return_all_bps": avg_all,
        "avg_net_return_predicted_positive_bps": avg_predicted_positive,
        "lift_bps": lift_bps,
        "positive_rate": positive_rate,
        "precision": precision,
        "predicted_positive_rate": predicted_positive_rate,
        "quality": quality,
        "recall": recall,
        "validation_samples": int(len(x_val)),
        "walk_forward_expectancy_bps": avg_predicted_positive,
    }


async def _train(min_samples: int, label_bps_threshold: float, horizon_minutes: int) -> int:
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

        # Load labelled samples oldest-first, then use the newest 20% as holdout.
        rows = await pool.fetch(
            """
            SELECT fs.feature_names, fs.feature_values, po.net_return_bps, po.label
            FROM feature_snapshots fs
            JOIN prediction_events pe ON pe.feature_snapshot_id = fs.snapshot_id
            JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
            WHERE po.horizon_minutes = $1
              AND po.label IS NOT NULL
              AND fs.feature_values IS NOT NULL
            ORDER BY fs.created_at ASC
            LIMIT 10000
            """,
            horizon_minutes,
        )

        if len(rows) < min_samples:
            msg = f"Insufficient samples: {len(rows)} < {min_samples}"
            click.echo(msg, err=True)
            await pool.execute(
                "UPDATE training_runs SET status='FAILED', error=$1, finished_at=now() WHERE run_id=$2",
                msg,
                run_id,
            )
            return 1

        click.echo(f"Training on {len(rows)} samples (horizon={horizon_minutes}m)")

        # Build arrays
        x_list = []
        y_list = []
        returns_list = []
        feature_names: list[str] = []

        for row in rows:
            vals = row["feature_values"]
            labels_raw = row["label"]
            if vals is None:
                continue
            v = json.loads(vals) if isinstance(vals, str) else list(vals)
            x_list.append(v)
            y_list.append(int(labels_raw))
            returns_list.append(float(row["net_return_bps"] or 0.0))
            if not feature_names and row["feature_names"]:
                fn = row["feature_names"]
                feature_names = json.loads(fn) if isinstance(fn, str) else list(fn)

        x_arr = np.array(x_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        returns_bps = np.array(returns_list, dtype=np.float32)
        if len(x_arr) < min_samples:
            msg = f"Insufficient usable samples: {len(x_arr)} < {min_samples}"
            click.echo(msg, err=True)
            await pool.execute(
                "UPDATE training_runs SET status='FAILED', error=$1, finished_at=now() WHERE run_id=$2",
                msg,
                run_id,
            )
            return 1

        validation_size = max(1, int(len(x_arr) * 0.2))
        if len(x_arr) - validation_size < 50:
            validation_size = max(1, len(x_arr) - 50)
        train_size = len(x_arr) - validation_size
        x_train, y_train = x_arr[:train_size], y[:train_size]
        x_val, y_val = x_arr[train_size:], y[train_size:]
        returns_val = returns_bps[train_size:]

        version = f"v{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M')}"
        model = ChallengerModel(version=version, feature_names=feature_names)

        _fit_model(model, x_train, y_train)
        metrics = _evaluate_model(model, x_val, y_val, returns_val)
        _fit_model(model, x_val, y_val)

        click.echo(f"Training complete: {model.training_samples} samples, version={version}")
        click.echo(
            "Quality: "
            f"{metrics.get('quality')} "
            f"precision={metrics.get('precision', 0):.3f} "
            f"lift_bps={metrics.get('lift_bps')}"
        )

        # Save checkpoint
        artifact = model.to_bytes()
        await pool.execute(
            """
            INSERT INTO model_versions (version, status, training_samples, feature_schema_hash, artifact, metrics,
                training_started_at, training_finished_at)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, now())
            ON CONFLICT (version) DO UPDATE SET
                status = EXCLUDED.status,
                artifact = EXCLUDED.artifact,
                training_samples = EXCLUDED.training_samples,
                feature_schema_hash = EXCLUDED.feature_schema_hash,
                metrics = EXCLUDED.metrics,
                training_started_at = EXCLUDED.training_started_at,
                training_finished_at = now()
            """,
            version,
            ModelStatus.SHADOW_CHALLENGER,
            model.training_samples,
            model.feature_schema_hash,
            artifact,
            json.dumps(
                metrics
                | {
                    "features": len(feature_names),
                    "horizon_minutes": horizon_minutes,
                    "label_bps_threshold": label_bps_threshold,
                    "sample_count": int(len(x_arr)),
                    "train_samples": int(len(x_train)),
                    "run_id": run_id,
                }
            ),
            run_started,
        )

        await pool.execute(
            """
            UPDATE training_runs
            SET status='COMPLETED', sample_count=$1, finished_at=now(), model_version=$2, metrics=$3::jsonb
            WHERE run_id=$4
            """,
            model.training_samples,
            version,
            json.dumps(metrics),
            run_id,
        )

        click.echo(f"Checkpoint saved as version={version}, status=SHADOW_CHALLENGER")
        click.echo("Run 'python -m trader.training.promote --version <version>' to evaluate and promote.")
        return 0

    except Exception as exc:
        log.exception("Training failed")
        try:
            await pool.execute(
                "UPDATE training_runs SET status='FAILED', error=$1, finished_at=now() WHERE run_id=$2",
                str(exc),
                run_id,
            )
        except Exception as update_exc:
            log.debug("Training run failure update failed: %s", update_exc)
        click.echo(f"Training failed: {exc}", err=True)
        return 1
    finally:
        await pool.close()


@click.command()
@click.option("--min-samples", default=500, type=int, help="Minimum samples required")
@click.option("--label-bps", default=5.0, type=float, help="Min net return bps for positive label")
@click.option("--horizon", default=15, type=int, help="Prediction horizon in minutes")
def main(min_samples: int, label_bps: float, horizon: int) -> None:
    """Train challenger model from historical feature snapshots."""
    raise SystemExit(asyncio.run(_train(min_samples, label_bps, horizon)))


if __name__ == "__main__":
    main()
