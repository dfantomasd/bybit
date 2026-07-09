"""CLI for offline strategy-rule discovery.

Examples:
    python -m trader.strategy_lab.discover --from-db --horizon 5 --min-samples 1000
    python -m trader.strategy_lab.discover --csv samples.csv --output strategy_lab.json
"""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import Any

import click
import numpy as np

from trader.storage.trade_journal import asyncpg_pool_connect_kwargs
from trader.strategy_lab.rule_generator import RuleSearchConfig, discover_rules, discover_segmented_rules
from trader.training.eligibility import training_decision_filter_sql, training_strategy_filter_sql
from trader.training.labels import active_label_schema_version
from trader.training.train import _settings_horizon, _settings_label_bps


def _parse_csv_samples(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str] | None, list[str] | None]:
    rows: list[list[float]] = []
    returns: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise click.ClickException("CSV has no header")
        if "net_return_bps" not in reader.fieldnames:
            raise click.ClickException("CSV must include net_return_bps column")
        feature_names = [name for name in reader.fieldnames if name not in {"net_return_bps", "symbol", "side"}]
        symbols: list[str] = []
        sides: list[str] = []
        for row in reader:
            try:
                returns.append(float(row["net_return_bps"]))
                rows.append([float(row[name]) for name in feature_names])
                if "symbol" in row:
                    symbols.append(str(row.get("symbol") or ""))
                if "side" in row:
                    sides.append(str(row.get("side") or ""))
            except (TypeError, ValueError) as exc:
                raise click.ClickException(f"invalid numeric CSV row: {exc}") from exc
    return (
        np.asarray(rows, dtype=float),
        np.asarray(returns, dtype=float),
        feature_names,
        symbols if symbols else None,
        sides if sides else None,
    )


async def _load_db_samples(
    *, horizon: int, min_samples: int
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str], dict[str, Any]]:
    import asyncpg

    from trader.config import Settings

    settings = Settings()
    strategy_allowlist = [
        item.strip() for item in str(settings.TRAIN_STRATEGY_ALLOWLIST or "").split(",") if item.strip()
    ]
    include_candle_baseline = bool(settings.TRAIN_INCLUDE_CANDLE_BASELINE)
    label_schema_version = active_label_schema_version(use_tpsl_exit=bool(settings.MODEL_LABEL_USE_TPSL_EXIT))
    label_threshold_bps = float(_settings_label_bps())
    pool = await asyncpg.create_pool(
        **asyncpg_pool_connect_kwargs(settings.POSTGRES_DSN.get_secret_value()),
        min_size=1,
        max_size=2,
        statement_cache_size=0,
    )
    query_template = """
            WITH eligible_samples AS (
                SELECT
                    fs.feature_names,
                    fs.feature_values,
                    fs.feature_schema_hash,
                    pe.symbol,
                    pe.strategy_signal,
                    po.net_return_bps,
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
                  AND po.label_threshold_bps = $5
                  AND po.net_return_bps IS NOT NULL
                  AND fs.feature_values IS NOT NULL
                  AND fs.training_eligible = true
                  AND pe.model_version = 'RULE_BASELINE_V1'
                  AND pe.strategy_signal IN ('Buy', 'Sell')
                  AND __TRAINING_DECISION_FILTER__
                  AND __TRAINING_STRATEGY_FILTER__
            ),
            schema_counts AS (
                SELECT feature_schema_hash, count(*) AS sample_count, max(created_at) AS latest_at
                FROM eligible_samples
                WHERE candle_rank = 1
                GROUP BY feature_schema_hash
            ),
            selected_schema AS (
                SELECT feature_schema_hash
                FROM schema_counts
                WHERE sample_count >= $6
                ORDER BY latest_at DESC
                LIMIT 1
            ),
            latest_window AS (
                SELECT es.feature_names, es.feature_values, es.symbol, es.strategy_signal, es.net_return_bps, es.created_at
                FROM eligible_samples es
                JOIN selected_schema ss ON ss.feature_schema_hash = es.feature_schema_hash
                WHERE es.candle_rank = 1
                ORDER BY es.created_at DESC
                LIMIT 10000
            )
            SELECT feature_names, feature_values, symbol, strategy_signal, net_return_bps, created_at
            FROM latest_window
            ORDER BY created_at ASC
            """
    query = query_template.replace("__TRAINING_DECISION_FILTER__", training_decision_filter_sql("$4")).replace(
        "__TRAINING_STRATEGY_FILTER__",
        training_strategy_filter_sql("$3", "$4"),
    )
    try:
        rows = await pool.fetch(
            query,
            horizon,
            label_schema_version,
            strategy_allowlist or None,
            include_candle_baseline,
            label_threshold_bps,
            min_samples,
        )
    finally:
        await pool.close()

    values: list[list[float]] = []
    returns: list[float] = []
    symbols: list[str] = []
    sides: list[str] = []
    feature_names: list[str] = []
    for row in rows:
        parsed_names = row["feature_names"]
        names = json.loads(parsed_names) if isinstance(parsed_names, str) else list(parsed_names)
        parsed_values = row["feature_values"]
        vals = json.loads(parsed_values) if isinstance(parsed_values, str) else list(parsed_values)
        if not feature_names:
            feature_names = [str(name) for name in names]
        elif feature_names != list(names):
            raise click.ClickException("mixed feature names in selected DB sample window")
        values.append([float(value) for value in vals])
        returns.append(float(row["net_return_bps"] or 0.0))
        symbols.append(str(row["symbol"] or ""))
        sides.append(str(row["strategy_signal"] or ""))

    meta = {
        "source": "db",
        "horizon_minutes": horizon,
        "label_schema_version": label_schema_version,
        "label_threshold_bps": label_threshold_bps,
        "strategy_allowlist": strategy_allowlist or "ALL",
        "include_candle_baseline": include_candle_baseline,
    }
    return np.asarray(values, dtype=float), np.asarray(returns, dtype=float), feature_names, symbols, sides, meta


async def build_strategy_lab_report_from_db(
    *,
    horizon: int | None = None,
    min_samples: int = 1000,
    min_train_count: int = 30,
    min_validation_count: int = 10,
    min_validation_net_bps: float = 0.0,
    top_n: int = 20,
    segmented: bool = True,
) -> dict[str, Any]:
    """Build a strategy-lab report from compatible Postgres outcomes.

    This is the programmatic counterpart of the CLI. Runtime code uses it in
    SHADOW mode to avoid the common operational trap where discovered-rule
    validation is enabled but the JSON file was never generated after deploy.
    """

    values, returns_bps, feature_names, symbols, sides, meta = await _load_db_samples(
        horizon=horizon or _settings_horizon(),
        min_samples=min_samples,
    )
    sample_count = int(len(returns_bps))
    if sample_count < min_samples or not feature_names:
        return {
            "status": "insufficient_samples",
            "sample_count": sample_count,
            "required_samples": int(min_samples),
            "rules": [],
            "meta": meta,
        }
    config = RuleSearchConfig(
        min_train_count=min_train_count,
        min_validation_count=min_validation_count,
        min_validation_avg_net_bps=min_validation_net_bps,
        top_n=top_n,
    )
    if segmented:
        report = discover_segmented_rules(
            values=values,
            returns_bps=returns_bps,
            feature_names=feature_names,
            symbols=symbols,
            sides=sides,
            config=config,
        )
    else:
        report = discover_rules(values=values, returns_bps=returns_bps, feature_names=feature_names, config=config)
    report["meta"] = meta
    return report


@click.command()
@click.option("--from-db", "from_db", is_flag=True, help="Load compatible samples from Postgres.")
@click.option(
    "--csv", "csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="CSV sample file."
)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), help="Write JSON report to this file.")
@click.option("--horizon", type=int, default=None, help="Outcome horizon in minutes for DB mode.")
@click.option("--min-samples", type=int, default=1000, show_default=True)
@click.option("--min-train-count", type=int, default=30, show_default=True)
@click.option("--min-validation-count", type=int, default=10, show_default=True)
@click.option("--min-validation-net-bps", type=float, default=0.0, show_default=True)
@click.option("--top-n", type=int, default=20, show_default=True)
@click.option("--segmented/--no-segmented", default=True, show_default=True)
def main(
    *,
    from_db: bool,
    csv_path: Path | None,
    output: Path | None,
    horizon: int | None,
    min_samples: int,
    min_train_count: int,
    min_validation_count: int,
    min_validation_net_bps: float,
    top_n: int,
    segmented: bool,
) -> None:
    """Discover explainable cost-aware strategy rules offline."""

    if from_db == bool(csv_path):
        raise click.ClickException("choose exactly one source: --from-db or --csv")
    meta: dict[str, Any]
    if from_db:
        report = asyncio.run(
            build_strategy_lab_report_from_db(
                horizon=horizon,
                min_samples=min_samples,
                min_train_count=min_train_count,
                min_validation_count=min_validation_count,
                min_validation_net_bps=min_validation_net_bps,
                top_n=top_n,
                segmented=segmented,
            )
        )
    else:
        assert csv_path is not None
        values, returns_bps, feature_names, symbols, sides = _parse_csv_samples(csv_path)
        meta = {"source": "csv", "path": str(csv_path)}
        config = RuleSearchConfig(
            min_train_count=min_train_count,
            min_validation_count=min_validation_count,
            min_validation_avg_net_bps=min_validation_net_bps,
            top_n=top_n,
        )
        if segmented:
            report = discover_segmented_rules(
                values=values,
                returns_bps=returns_bps,
                feature_names=feature_names,
                symbols=symbols,
                sides=sides,
                config=config,
            )
        else:
            report = discover_rules(values=values, returns_bps=returns_bps, feature_names=feature_names, config=config)
        report["meta"] = meta
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if output is not None:
        output.write_text(text + "\n", encoding="utf-8")
        click.echo(f"Wrote strategy-lab report: {output}")
    else:
        click.echo(text)


if __name__ == "__main__":
    main()
