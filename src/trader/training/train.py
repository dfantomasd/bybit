"""Offline trainer CLI for the directional, cost-aware challenger model.

Usage:
    python -m trader.training.train [--min-samples 500]

Runs as a separate process (Render Cron Job at 03:00 UTC, or manually).
NEVER runs inside the trading process.

Labels are resolved by ``DirectionalTradeJournal``. Training accepts only the
current label schema and one feature schema hash. Label thresholds are
recomputed from stored net_return_bps during walk-forward candidate selection.
The newest 10,000 compatible samples are retained and then ordered oldest-first
for chronological walk-forward validation.
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

from trader.ml.model_selection import model_selection_metrics
from trader.training.eligibility import training_strategy_filter_sql
from trader.training.labels import LABEL_SCHEMA_VERSION

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _settings_horizon() -> int:
    """Return MODEL_LABEL_HORIZON from settings, falling back to 15."""
    try:
        from trader.config import Settings

        return int(getattr(Settings(), "MODEL_LABEL_HORIZON", 15))
    except Exception:
        import os

        try:
            return int(os.environ.get("MODEL_LABEL_HORIZON", 15))
        except (ValueError, TypeError):
            return 15


def _parse_float_csv(raw: str, default: list[float]) -> list[float]:
    values: list[float] = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError:
            log.warning("Ignoring invalid float in CSV setting: %s", item)
    return values or default


def _parse_str_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _walk_forward_splits(
    x: np.ndarray,
    min_train_samples: int = 500,
    n_folds: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return chronological expanding-window walk-forward splits."""

    total = len(x)
    if total <= 1:
        return []
    min_train = min(max(1, int(min_train_samples)), total - 1)
    folds = max(1, int(n_folds))
    fold_size = (total - min_train) // folds
    if fold_size <= 0:
        return [(np.arange(0, min_train), np.arange(min_train, total))]

    result: list[tuple[np.ndarray, np.ndarray]] = []
    train_end = min_train
    for _ in range(folds):
        val_start = train_end
        val_end = min(val_start + fold_size, total)
        if val_start >= total or val_end <= val_start:
            break
        result.append((np.arange(0, train_end), np.arange(val_start, val_end)))
        train_end = val_end
    return result


def _validate_walk_forward_chronology(
    folds: list[tuple[np.ndarray, np.ndarray]],
    timestamps: list[datetime],
) -> list[dict[str, Any]]:
    """Assert every validation fold starts strictly after its train window."""

    windows: list[dict[str, Any]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise RuntimeError(f"empty walk-forward fold: {fold_idx}")
        max_idx = int(max(train_idx[-1], val_idx[-1]))
        if max_idx >= len(timestamps):
            raise RuntimeError(
                "walk-forward chronology timestamp mismatch: "
                f"fold={fold_idx} max_idx={max_idx} timestamps={len(timestamps)}"
            )
        train_end = timestamps[int(train_idx[-1])]
        val_start = timestamps[int(val_idx[0])]
        val_end = timestamps[int(val_idx[-1])]
        if val_start <= train_end:
            raise RuntimeError(
                "walk-forward chronology violation: "
                f"fold={fold_idx} train_end={train_end.isoformat()} val_start={val_start.isoformat()}"
            )
        windows.append(
            {
                "fold": int(fold_idx),
                "train_start_at": timestamps[int(train_idx[0])].isoformat(),
                "train_end_at": train_end.isoformat(),
                "val_start_at": val_start.isoformat(),
                "val_end_at": val_end.isoformat(),
                "train_samples": int(len(train_idx)),
                "validation_samples": int(len(val_idx)),
            }
        )
    return windows


def _filter_timestamps_by_mask(timestamps: list[datetime], keep_mask: np.ndarray) -> list[datetime]:
    """Apply the same sample mask used for training arrays to timestamp metadata."""

    if len(timestamps) != len(keep_mask):
        raise RuntimeError(f"timestamp/filter length mismatch: {len(timestamps)}!={len(keep_mask)}")
    return [ts for ts, keep in zip(timestamps, keep_mask, strict=True) if bool(keep)]


def _candidate_specs(enabled: str) -> list[dict[str, Any]]:
    families = {item.strip().upper() for item in str(enabled or "").split(",") if item.strip()}
    if not families:
        families = {"GBDT", "LOGREG"}

    specs: list[dict[str, Any]] = []
    if "GBDT" in families:
        specs.extend(
            [
                {
                    "model_type": "GBDT",
                    "max_iter": 150,
                    "learning_rate": 0.05,
                    "max_leaf_nodes": 31,
                    "l2_regularization": 0.0,
                },
                {
                    "model_type": "GBDT",
                    "max_iter": 300,
                    "learning_rate": 0.03,
                    "max_leaf_nodes": 31,
                    "l2_regularization": 0.1,
                },
                {
                    "model_type": "GBDT",
                    "max_iter": 80,
                    "learning_rate": 0.08,
                    "max_leaf_nodes": 15,
                    "l2_regularization": 0.0,
                },
            ]
        )
    if "LOGREG" in families:
        specs.extend(
            [
                {"model_type": "LOGREG", "C": 0.1},
                {"model_type": "LOGREG", "C": 1.0},
                {"model_type": "LOGREG", "C": 10.0},
            ]
        )
    if "SGD" in families:
        specs.append({"model_type": "SGD"})
    if "MLP" in families:
        specs.extend(
            [
                {
                    "model_type": "MLP",
                    "hidden_layer_sizes": (64, 32),
                    "learning_rate_init": 0.001,
                    "alpha": 0.0001,
                },
                {
                    "model_type": "MLP",
                    "hidden_layer_sizes": (128, 64, 32),
                    "learning_rate_init": 0.0005,
                    "alpha": 0.001,
                },
            ]
        )
    return specs


def _summarise_walk_forward(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    expectancies = [
        float(m["best_threshold_avg_net_return_bps"])
        for m in fold_metrics
        if m.get("best_threshold_avg_net_return_bps") is not None
    ]
    if not expectancies:
        expectancies = [
            float(m["walk_forward_expectancy_bps"])
            for m in fold_metrics
            if m.get("walk_forward_expectancy_bps") is not None
        ]
    precision_values = [float(m["precision"]) for m in fold_metrics if m.get("precision") is not None]
    lift_values = [float(m["lift_bps"]) for m in fold_metrics if m.get("lift_bps") is not None]
    pass_counts = [int(m.get("selected_pass_count") or 0) for m in fold_metrics]
    score_thresholds = [float(m["best_threshold"]) for m in fold_metrics if m.get("best_threshold") is not None]
    pass_rates = [
        float(m["best_threshold_pass_rate"]) for m in fold_metrics if m.get("best_threshold_pass_rate") is not None
    ]

    arr = np.array(expectancies, dtype=np.float32)
    return {
        "wf_folds": int(len(fold_metrics)),
        "wf_mean_bps": float(arr.mean()) if len(arr) else None,
        "wf_median_bps": float(np.median(arr)) if len(arr) else None,
        "wf_positive_folds": int((arr > 0).sum()) if len(arr) else 0,
        "wf_min_bps": float(arr.min()) if len(arr) else None,
        "wf_max_bps": float(arr.max()) if len(arr) else None,
        "wf_std_bps": float(arr.std(ddof=0)) if len(arr) else None,
        "precision": float(np.mean(precision_values)) if precision_values else 0.0,
        "lift_bps": float(np.mean(lift_values)) if lift_values else None,
        "selected_score_threshold": float(np.median(score_thresholds)) if score_thresholds else None,
        "selected_score_pass_rate": float(np.mean(pass_rates)) if pass_rates else None,
        "walk_forward_expectancy_bps": float(arr.mean()) if len(arr) else None,
        "total_pass_count": int(sum(pass_counts)),
    }


def _negative_bucket_keep_mask(
    *,
    returns_bps: np.ndarray,
    regimes: np.ndarray,
    hours: np.ndarray,
    volatility_values: np.ndarray,
    min_bucket_samples: int,
    min_bucket_avg_bps: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return mask excluding regime/hour/volatility buckets with stable losses."""

    if len(returns_bps) == 0:
        return np.array([], dtype=bool), []
    median_vol = float(np.median(volatility_values)) if len(volatility_values) else 0.0
    buckets: dict[tuple[str, int, str], list[int]] = {}
    for idx, ret in enumerate(returns_bps):
        _ = ret
        vol_bucket = "high_vol" if float(volatility_values[idx]) >= median_vol else "low_vol"
        key = (str(regimes[idx] or "unknown"), int(hours[idx]), vol_bucket)
        buckets.setdefault(key, []).append(idx)

    keep = np.ones(len(returns_bps), dtype=bool)
    excluded: list[dict[str, Any]] = []
    for (regime, hour, vol_bucket), indexes in buckets.items():
        if len(indexes) < min_bucket_samples:
            continue
        avg_bps = float(np.mean(returns_bps[indexes]))
        if avg_bps < min_bucket_avg_bps:
            keep[indexes] = False
            excluded.append(
                {
                    "regime": regime,
                    "hour": hour,
                    "volatility": vol_bucket,
                    "count": int(len(indexes)),
                    "avg_net_return_bps": avg_bps,
                }
            )
    return keep, excluded


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

    # Batch prediction — single sklearn call instead of a Python loop per sample
    score_arr, pred_arr = model.predict_batch(x_val)
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
    best_threshold_pass_count = 0
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
            best_threshold_pass_count = pass_count

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
        "best_threshold_pass_count": best_threshold_pass_count,
        "best_threshold_pass_rate": best_threshold_pass_rate,
        "lift_bps": lift_bps,
        "positive_rate": positive_rate,
        "precision": precision,
        "predicted_positive_rate": predicted_positive_rate,
        "quality": quality,
        "recall": recall,
        "threshold_metrics": threshold_metrics,
        "selected_pass_count": best_threshold_pass_count or int(predicted_positive.sum()),
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
    strategy_allowlist = _parse_str_csv(settings.TRAIN_STRATEGY_ALLOWLIST)
    strategy_filter = training_strategy_filter_sql("$4")
    log.info("Training strategy allowlist: %s", strategy_allowlist or "ALL")
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
            f"""
            WITH eligible_samples AS (
                SELECT
                    fs.symbol,
                    fs.interval,
                    fs.candle_open_time,
                    fs.feature_names,
                    fs.feature_values,
                    fs.feature_schema_hash,
                    po.net_return_bps,
                    po.label,
                    pe.strategy_signal,
                    pe.metadata->>'strategy_id' AS strategy_id,
                    COALESCE(pe.metadata->>'regime', 'unknown') AS regime,
                    fs.created_at,
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
            ),
            schema_counts AS (
                SELECT
                    feature_schema_hash,
                    count(*) AS sample_count,
                    max(created_at) AS latest_at
                FROM eligible_samples
                WHERE candle_rank = 1
                GROUP BY feature_schema_hash
            ),
            selected_schema AS (
                SELECT feature_schema_hash
                FROM schema_counts
                WHERE sample_count >= $3
                ORDER BY latest_at DESC
                LIMIT 1
            ),
            labelled AS (
                SELECT es.feature_names,
                       es.feature_values,
                       es.feature_schema_hash,
                       es.net_return_bps,
                       es.label,
                       es.strategy_signal,
                       es.strategy_id,
                       es.regime,
                       es.created_at
                FROM eligible_samples es
                JOIN selected_schema ss ON ss.feature_schema_hash = es.feature_schema_hash
                WHERE es.candle_rank = 1
            ),
            latest_window AS (
                SELECT *
                FROM labelled
                ORDER BY created_at DESC
                LIMIT 10000
            )
            SELECT feature_names, feature_values, feature_schema_hash,
                   net_return_bps, label, strategy_signal, strategy_id, regime, created_at
            FROM latest_window
            ORDER BY created_at ASC
            """,
            horizon_minutes,
            LABEL_SCHEMA_VERSION,
            min_samples,
            strategy_allowlist or None,
        )

        if len(rows) < min_samples:
            # The main query returns rows only when ONE schema bucket reaches
            # min_samples, so "0" alone is misleading — report the real
            # accumulation progress per feature schema.
            schema_rows = await pool.fetch(
                f"""
                SELECT fs.feature_schema_hash,
                       count(DISTINCT (fs.symbol, fs.interval, fs.candle_open_time)) AS cnt
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
                  AND {strategy_filter.replace("$4", "$3")}
                GROUP BY fs.feature_schema_hash
                ORDER BY cnt DESC
                """,
                horizon_minutes,
                LABEL_SCHEMA_VERSION,
                strategy_allowlist or None,
            )
            total_labelled = sum(int(r["cnt"]) for r in schema_rows)
            top = ", ".join(f"{str(r['feature_schema_hash'])[:8]}:{r['cnt']}" for r in schema_rows[:4])
            msg = (
                f"Insufficient compatible samples: no feature schema has {min_samples} yet "
                f"(unique labelled candles={total_labelled}, per-schema=[{top or 'none'}]); "
                f"schema={LABEL_SCHEMA_VERSION}, "
                f"horizon={horizon_minutes}m, "
                f"strategy_allowlist={strategy_allowlist or 'ALL'}"
            )
            click.echo(msg, err=True)
            await pool.execute(
                "UPDATE training_runs SET status='FAILED', error=$1, finished_at=now() WHERE run_id=$2",
                msg,
                run_id,
            )
            return 1

        click.echo(
            f"Training on {len(rows)} compatible samples (schema={LABEL_SCHEMA_VERSION}, horizon={horizon_minutes}m)"
        )

        x_list: list[list[float]] = []
        returns_list: list[float] = []
        sides_list: list[str] = []
        regimes_list: list[str] = []
        hours_list: list[int] = []
        created_at_list: list[datetime] = []
        feature_names: list[str] = []
        feature_schema_hash = ""
        expected_vector_size: int | None = None

        for row in rows:
            vals = row["feature_values"]
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

            side = str(row["strategy_signal"])
            if side not in {"Buy", "Sell"}:
                raise RuntimeError(f"unsupported strategy signal in training data: {side!r}")

            current_names = row["feature_names"]
            parsed_names = json.loads(current_names) if isinstance(current_names, str) else list(current_names)
            if "proposal_side" not in parsed_names:
                by_name = dict(zip(parsed_names, v, strict=True))
                by_name["proposal_side"] = 1.0 if side == "Buy" else -1.0
                parsed_names = sorted(by_name.keys())
                v = [float(by_name[name]) for name in parsed_names]
            if not feature_names:
                feature_names = parsed_names
            elif parsed_names != feature_names:
                raise RuntimeError("mixed feature names or order in selected training window")

            x_list.append(v)
            returns_list.append(float(row["net_return_bps"] or 0.0))
            sides_list.append(side)
            regimes_list.append(str(row["regime"] or "unknown"))
            created_at = row["created_at"]
            created_dt = created_at if isinstance(created_at, datetime) else datetime.now(tz=UTC)
            hours_list.append(int(created_dt.hour))
            created_at_list.append(created_dt)

        x_arr = np.array(x_list, dtype=np.float32)
        returns_bps = np.array(returns_list, dtype=np.float32)
        sides = np.array(sides_list, dtype=object)
        regimes = np.array(regimes_list, dtype=object)
        hours = np.array(hours_list, dtype=np.int32)
        if len(x_arr) < min_samples:
            msg = f"Insufficient usable samples: {len(x_arr)} < {min_samples}"
            click.echo(msg, err=True)
            await pool.execute(
                "UPDATE training_runs SET status='FAILED', error=$1, finished_at=now() WHERE run_id=$2",
                msg,
                run_id,
            )
            return 1

        excluded_buckets: list[dict[str, Any]] = []
        if bool(settings.TRAIN_EXCLUDE_NEGATIVE_BUCKETS):
            atr_values = np.zeros(len(x_arr), dtype=np.float32)
            if "atr_14_pct" in feature_names:
                atr_idx = feature_names.index("atr_14_pct")
                atr_values = x_arr[:, atr_idx].astype(np.float32)
            keep_mask, excluded_buckets = _negative_bucket_keep_mask(
                returns_bps=returns_bps,
                regimes=regimes,
                hours=hours,
                volatility_values=atr_values,
                min_bucket_samples=max(1, int(settings.TRAIN_MIN_BUCKET_SAMPLES)),
                min_bucket_avg_bps=float(settings.TRAIN_BUCKET_MIN_AVG_BPS),
            )
            if int(keep_mask.sum()) >= min_samples:
                x_arr = x_arr[keep_mask]
                returns_bps = returns_bps[keep_mask]
                sides = sides[keep_mask]
                regimes = regimes[keep_mask]
                hours = hours[keep_mask]
                created_at_list = _filter_timestamps_by_mask(created_at_list, keep_mask)
                click.echo(f"Excluded {len(keep_mask) - int(keep_mask.sum())} samples from negative buckets")
            else:
                click.echo("Negative bucket filter skipped: it would leave too few samples", err=True)
                excluded_buckets = []

        label_thresholds = sorted(
            set(
                _parse_float_csv(str(settings.MODEL_THRESHOLD_GRID), [0.0, 2.0, 5.0, 8.0, 12.0])
                + [float(label_bps_threshold)]
            )
        )
        candidates = _candidate_specs(str(settings.MODEL_CANDIDATES))
        min_train_for_wf = max(min_samples, int(settings.MODEL_WF_MIN_TRAIN_SAMPLES))
        folds = _walk_forward_splits(
            x_arr,
            min_train_samples=min_train_for_wf,
            n_folds=max(1, int(settings.MODEL_WF_FOLDS)),
        )
        if not folds:
            raise RuntimeError("not enough samples for walk-forward validation")
        wf_windows = _validate_walk_forward_chronology(folds, created_at_list)

        def _run_candidate(spec: dict[str, Any], selected_label_threshold: float) -> dict[str, Any] | None:
            y = (returns_bps > float(selected_label_threshold)).astype(np.int32)
            if len(np.unique(y)) < 2:
                return None
            fold_metrics: list[dict[str, Any]] = []
            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                y_train = y[train_idx]
                y_val_fold = y[val_idx]
                if len(np.unique(y_train)) < 2 or len(np.unique(y_val_fold)) < 2:
                    continue
                fold_model = ChallengerModel(
                    version=f"{run_id}_fold{fold_idx}",
                    feature_names=feature_names,
                    model_type=str(spec["model_type"]),
                    model_params={k: v for k, v in spec.items() if k != "model_type"},
                )
                try:
                    fold_model.fit_batch(x_arr[train_idx], y_train, params=fold_model.model_params)
                    metrics = _evaluate_model(
                        fold_model, x_arr[val_idx], y_val_fold, returns_bps[val_idx], sides[val_idx]
                    )
                except Exception as exc:
                    log.warning(
                        "Candidate fold failed: model_type=%s label_threshold=%s fold=%s error=%s",
                        spec.get("model_type"),
                        selected_label_threshold,
                        fold_idx,
                        exc,
                    )
                    continue
                metrics["fold"] = fold_idx
                metrics["val_start_idx"] = int(val_idx[0])
                metrics["val_end_idx"] = int(val_idx[-1])
                metrics.update(wf_windows[fold_idx])
                fold_metrics.append(metrics)
            if not fold_metrics:
                return None
            summary = _summarise_walk_forward(fold_metrics)
            summary.update(
                {
                    "candidate": spec,
                    "model_type": spec["model_type"],
                    "model_params": {k: v for k, v in spec.items() if k != "model_type"},
                    "selected_label_threshold_bps": float(selected_label_threshold),
                    "fold_metrics": fold_metrics,
                }
            )
            return summary

        tasks = [(spec, threshold) for threshold in label_thresholds for spec in candidates]

        try:
            from joblib import Parallel
            from joblib import delayed as jdelayed

            raw_results = Parallel(n_jobs=1)(jdelayed(_run_candidate)(spec, threshold) for spec, threshold in tasks)
        except (ImportError, ModuleNotFoundError):
            raw_results = [_run_candidate(spec, threshold) for spec, threshold in tasks]

        candidate_results: list[dict[str, Any]] = [r for r in raw_results if r is not None]

        if not candidate_results:
            raise RuntimeError("no trainable model candidate survived walk-forward validation")

        min_positive_folds = min(3, len(folds))
        eligible_results = [r for r in candidate_results if int(r.get("wf_positive_folds") or 0) >= min_positive_folds]
        selection_pool = eligible_results or candidate_results

        def _selection_key(result: dict[str, Any]) -> tuple[float, int]:
            wf_mean = result.get("wf_mean_bps")
            return (
                float(wf_mean) if wf_mean is not None else -1_000_000.0,
                int(result.get("wf_positive_folds") or 0),
            )

        best = max(
            selection_pool,
            key=_selection_key,
        )

        selected_label_threshold = float(best["selected_label_threshold_bps"])
        y = (returns_bps > selected_label_threshold).astype(np.int32)
        version = f"v{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M')}_h{horizon_minutes}m_dnv1"
        model = ChallengerModel(
            version=version,
            feature_names=feature_names,
            model_type=str(best["model_type"]),
            model_params=dict(best.get("model_params") or {}),
        )
        model_feature_schema_hash = model.feature_schema_hash
        snapshot_feature_schema_hash = feature_schema_hash

        # The saved artifact is trained on all usable samples with the candidate
        # selected only from out-of-sample walk-forward metrics.
        model.fit_batch(x_arr, y, params=model.model_params)
        metrics = {
            "quality": "GOOD"
            if (
                best.get("wf_mean_bps") is not None
                and float(best["wf_mean_bps"]) > 0
                and int(best.get("wf_positive_folds") or 0) >= min_positive_folds
                and int(best.get("total_pass_count") or 0) >= int(settings.MODEL_MIN_PASS_COUNT_FOR_PROMOTION)
            )
            else "WEAK",
            "accuracy": None,
            "avg_net_return_all_bps": float(np.mean(returns_bps)) if len(returns_bps) else 0.0,
            "best_threshold": best.get("selected_score_threshold"),
            "best_threshold_avg_net_return_bps": best.get("wf_mean_bps"),
            "best_threshold_pass_count": best.get("total_pass_count"),
            "best_threshold_pass_rate": best.get("selected_score_pass_rate"),
            "candidate_count": len(candidate_results),
            "candidate_summaries": [
                {
                    k: v
                    for k, v in r.items()
                    if k
                    not in {
                        "fold_metrics",
                    }
                }
                for r in candidate_results
            ],
            "excluded_negative_buckets": excluded_buckets[:20],
            "label_threshold_candidates_bps": label_thresholds,
            "selected_label_threshold_bps": selected_label_threshold,
            "selected_model_params": model.model_params,
            "validation_samples": int(sum(len(v) for _, v in folds)),
            "walk_forward_windows": wf_windows,
            "walk_forward_chronology": "strict_after_train",
            **{k: v for k, v in best.items() if k != "fold_metrics"},
            "fold_metrics": best["fold_metrics"],
        }

        click.echo(f"Training complete: {model.training_samples} samples, version={version}")
        click.echo(
            "Quality: "
            f"{metrics.get('quality')} "
            f"precision={metrics.get('precision', 0):.3f} "
            f"lift_bps={metrics.get('lift_bps')} "
            f"wf_mean_bps={metrics.get('wf_mean_bps')} "
            f"label_threshold={selected_label_threshold:g}"
        )

        artifact = model.to_bytes()
        stored_metrics = metrics | {
            "features": len(feature_names),
            "model_type": model.model_type,
            "feature_schema_hash": snapshot_feature_schema_hash,
            "model_feature_schema_hash": model_feature_schema_hash,
            "source_feature_schema_hash": snapshot_feature_schema_hash,
            "horizon_minutes": horizon_minutes,
            "label_bps_threshold": selected_label_threshold,
            "requested_label_bps_threshold": label_bps_threshold,
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "sample_count": int(len(x_arr)),
            "train_samples": int(len(x_arr)),
            "run_id": run_id,
        }
        stored_metrics |= model_selection_metrics(stored_metrics)
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
            snapshot_feature_schema_hash,
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
@click.option(
    "--horizon",
    default=lambda: _settings_horizon(),
    type=int,
    help="Prediction horizon in minutes (default: MODEL_LABEL_HORIZON config, fallback 15)",
)
def main(min_samples: int, label_bps: float, horizon: int) -> None:
    """Train a challenger from directional, cost-aware historical labels."""

    raise SystemExit(asyncio.run(_train(min_samples, label_bps, horizon)))


if __name__ == "__main__":
    main()
