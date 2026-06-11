"""Offline trainer CLI for the directional, cost-aware challenger model.

Usage:
    python -m trader.training.train [--min-samples 500]

Runs as a separate process (Render Cron Job at 03:00 UTC, or manually).
NEVER runs inside the trading process.

Labels are resolved by ``DirectionalTradeJournal``. Training accepts only the
current label schema, the requested threshold, and one feature schema hash.
The newest 10,000 compatible samples are retained and then ordered oldest-first
for a chronological train/validation split.
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

from trader.training.labels import LABEL_SCHEMA_VERSION

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _fit_model(model: Any, x_arr: np.ndarray, y: np.ndarray) -> None:
    chunk = 32
    for i in range(0, len(x_arr), chunk):
        xb = x_arr[i : i + chunk]
        yb = y[i : i + chunk]
        for xi, yi in zip(xb, yb, strict=False):
            model.partial_fit(xi.tolist(), int(yi))


def _validation_metrics_by_side(
    *,
    sides: np.ndarray,
    predicted_positive: np.ndarray,
    returns_bps: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Return validation expectancy split by Buy and Sell."""

    result: dict[str, dict[str, Any]] = {}
    for side in ("Buy", "Sell"):
        side_mask = sides == side
        passed_mask = side_mask & predicted_positive
        side_count = int(side_mask.sum())
        pass_count = int(passed_mask.sum())
        result[side.lower()] = {
            "validation_count": side_count,
            "pass_count": pass_count,
            "avg_net_return_all_bps": float(np.mean(returns_bps[side_mask])) if side_count else None,
            "avg_net_return_pass_bps": float(np.mean(returns_bps[passed_mask])) if pass_count else None,
        }
    return result


def _evaluate_model(
    model: Any,
    x_val: np.ndarray,
    y_val: np.ndarray,
    returns_bps: np.ndarray,
    sides: np.ndarray,
) -> dict[str, Any]:
    if len(x_val) == 0:
        return {
            "quality": "n/a",
            "validation_samples": 0,
            "validation_by_side": {},
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
    threshold_candidates = [0.50, 0.55, 0.60, 0.65, 0.70]
    threshold_metrics: list[dict[str, Any]] = []
    min_pass = max(10, int(len(x_val) * 0.05))
    best_threshold: float | None = None
    best_threshold_avg_bps: float | None = None
    best_threshold_pass_rate: float | None = None
    for threshold in threshold_candidates:
        passed = score_arr >= threshold
        pass_count = int(passed.sum())
        pass_rate = float(pass_count / len(score_arr)) if len(score_arr) else 0.0
        avg_pass_bps = float(np.mean(returns_bps[passed])) if pass_count else None
        threshold_metrics.append(
            {
                "threshold": threshold,
                "pass_count": pass_count,
                "pass_rate": pass_rate,
                "avg_net_return_bps": avg_pass_bps,
            }
        )
        if pass_count < min_pass or avg_pass_bps is None:
            continue
        if best_threshold_avg_bps is None or avg_pass_bps > best_threshold_avg_bps:
            best_threshold = threshold
            best_threshold_avg_bps = avg_pass_bps
            best_threshold_pass_rate = pass_rate

    quality = "INSUFFICIENT_VALIDATION"
    if len(x_val) >= 100:
        good_default = precision > positive_rate and lift_bps is not None and lift_bps > 0
        good_best_threshold = (
            best_threshold_avg_bps is not None
            and best_threshold_avg_bps > avg_all
            and best_threshold_pass_rate is not None
            and best_threshold_pass_rate >= 0.05
        )
        quality = "GOOD" if good_default or good_best_threshold else "WEAK"

    return {
        "accuracy": accuracy,
        "avg_model_score": float(score_arr.mean()) if len(score_arr) else 0.0,
        "avg_net_return_all_bps": avg_all,
        "avg_net_return_predicted_positive_bps": avg_predicted_positive,
        "best_threshold": best_threshold,
        "best_threshold_avg_net_return_bps": best_threshold_avg_bps,
        "best_threshold_pass_rate": best_threshold_pass_rate,
        "lift_bps": lift_bps,
        "positive_rate": positive_rate,
        "precision": precision,
        "predicted_positive_rate": predicted_positive_rate,
        "quality": quality,
        "recall": recall,
        "threshold_metrics": threshold_metrics,
        "validation_by_side": _validation_metrics_by_side(
            sides=sides,
            predicted_positive=predicted_positive,
            returns_bps=returns_bps,
        ),
        "validation_samples": int(len(x_val)),
        "walk_forward_expectancy_bps": avg_predicted_positive,
    }


async def _train(min_samples: int, label_bps_threshold: float, horizon_minutes: int) -> int:
    import asyncpg

    from trader.config import Settings
    from trader.ml.challenger import ChallengerModel, ModelStatus

    settings = Settings()
    dsn = settings.POSTGRES_DSN.get_secret_value().replace("postgresql+asyncpg://", "postgresql://", 1)
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2, statement_cache_size=0)

    run_id = str(uuid.uuid4())
    run_started = datetime.now(tz=UTC)

    try:
        await pool.execute(
            "INSERT INTO training_runs (run_id, mode, status) VALUES ($1, 'offline', 'RUNNING')",
            run_id,
        )

        rows = await pool.fetch(
            """
            WITH schema_counts AS (
                SELECT
                    fs.feature_schema_hash,
                    count(DISTINCT fs.snapshot_id) AS sample_count,
                    max(fs.created_at) AS latest_at
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
                GROUP BY fs.feature_schema_hash
            ),
            selected_schema AS (
                SELECT feature_schema_hash
                FROM schema_counts
                WHERE sample_count >= $4
                ORDER BY latest_at DESC
                LIMIT 1
            ),
            labelled AS (
                SELECT DISTINCT ON (fs.symbol, fs.interval, fs.candle_open_time)
                       fs.feature_names,
                       fs.feature_values,
                       fs.feature_schema_hash,
                       po.net_return_bps,
                       po.label,
                       pe.strategy_signal,
                       fs.created_at
                FROM feature_snapshots fs
                JOIN prediction_events pe ON pe.feature_snapshot_id = fs.snapshot_id
                JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                JOIN selected_schema ss ON ss.feature_schema_hash = fs.feature_schema_hash
                WHERE po.horizon_minutes = $1
                  AND po.label IS NOT NULL
                  AND po.label_schema_version = $2
                  AND po.label_threshold_bps = $3
                  AND fs.feature_values IS NOT NULL
                  AND fs.training_eligible = true
                  AND pe.model_version = 'RULE_BASELINE_V1'
                  AND pe.strategy_signal IN ('Buy', 'Sell')
                ORDER BY fs.symbol, fs.interval, fs.candle_open_time, fs.created_at DESC
            ),
            latest_window AS (
                SELECT *
                FROM labelled
                ORDER BY created_at DESC
                LIMIT 10000
            )
            SELECT feature_names, feature_values, feature_schema_hash,
                   net_return_bps, label, strategy_signal, created_at
            FROM latest_window
            ORDER BY created_at ASC
            """,
            horizon_minutes,
            LABEL_SCHEMA_VERSION,
            label_bps_threshold,
            min_samples,
        )

        if len(rows) < min_samples:
            msg = (
                f"Insufficient compatible samples: {len(rows)} < {min_samples}; "
                f"schema={LABEL_SCHEMA_VERSION}, threshold={label_bps_threshold:g}bps, "
                f"horizon={horizon_minutes}m"
            )
            click.echo(msg, err=True)
            await pool.execute(
                "UPDATE training_runs SET status='FAILED', error=$1, finished_at=now() WHERE run_id=$2",
                msg,
                run_id,
            )
            return 1

        click.echo(
            f"Training on {len(rows)} compatible samples "
            f"(schema={LABEL_SCHEMA_VERSION}, threshold={label_bps_threshold:g}bps, horizon={horizon_minutes}m)"
        )

        x_list: list[list[float]] = []
        y_list: list[int] = []
        returns_list: list[float] = []
        sides_list: list[str] = []
        feature_names: list[str] = []
        feature_schema_hash = ""
        expected_vector_size: int | None = None

        for row in rows:
            vals = row["feature_values"]
            labels_raw = row["label"]
            row_schema_hash = str(row["feature_schema_hash"] or "")
            if vals is None:
                continue
            if not feature_schema_hash:
                feature_schema_hash = row_schema_hash
            elif row_schema_hash != feature_schema_hash:
                raise RuntimeError("mixed feature schema hashes in selected training window")

            v = json.loads(vals) if isinstance(vals, str) else list(vals)
            if expected_vector_size is None:
                expected_vector_size = len(v)
            elif len(v) != expected_vector_size:
                raise RuntimeError("mixed feature vector lengths in selected training window")

            current_names = row["feature_names"]
            parsed_names = json.loads(current_names) if isinstance(current_names, str) else list(current_names)
            if not feature_names:
                feature_names = parsed_names
            elif parsed_names != feature_names:
                raise RuntimeError("mixed feature names or order in selected training window")

            side = str(row["strategy_signal"])
            if side not in {"Buy", "Sell"}:
                raise RuntimeError(f"unsupported strategy signal in training data: {side!r}")

            x_list.append(v)
            y_list.append(int(labels_raw))
            returns_list.append(float(row["net_return_bps"] or 0.0))
            sides_list.append(side)

        x_arr = np.array(x_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        returns_bps = np.array(returns_list, dtype=np.float32)
        sides = np.array(sides_list, dtype=object)
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
        sides_val = sides[train_size:]

        version = f"v{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M')}_h{horizon_minutes}m_dnv1"
        model = ChallengerModel(version=version, feature_names=feature_names)

        _fit_model(model, x_train, y_train)
        metrics = _evaluate_model(model, x_val, y_val, returns_val, sides_val)
        _fit_model(model, x_val, y_val)

        click.echo(f"Training complete: {model.training_samples} samples, version={version}")
        click.echo(
            "Quality: "
            f"{metrics.get('quality')} "
            f"precision={metrics.get('precision', 0):.3f} "
            f"lift_bps={metrics.get('lift_bps')}"
        )

        artifact = model.to_bytes()
        stored_metrics = metrics | {
            "features": len(feature_names),
            "feature_schema_hash": feature_schema_hash,
            "horizon_minutes": horizon_minutes,
            "label_bps_threshold": label_bps_threshold,
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "sample_count": int(len(x_arr)),
            "train_samples": int(len(x_train)),
            "run_id": run_id,
        }
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
            feature_schema_hash,
            artifact,
            json.dumps(stored_metrics),
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
            json.dumps(stored_metrics),
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
@click.option("--min-samples", default=500, type=int, help="Minimum compatible samples required")
@click.option("--label-bps", default=5.0, type=float, help="Resolved net-return threshold in bps")
@click.option("--horizon", default=15, type=int, help="Prediction horizon in minutes")
def main(min_samples: int, label_bps: float, horizon: int) -> None:
    """Train a challenger from directional, cost-aware historical labels."""

    raise SystemExit(asyncio.run(_train(min_samples, label_bps, horizon)))


if __name__ == "__main__":
    main()
