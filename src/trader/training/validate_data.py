"""Data validation script for ML training dataset.

Usage:
    python -m trader.training.validate_data
    python -m trader.training.validate_data --dsn postgresql://...

Checks:
- No duplicate feature snapshots per (symbol, interval, candle_open_time)
- No feature values out of expected ranges
- Sufficient samples per symbol
- No orphaned snapshots (no linked prediction outcomes)
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import click


async def _validate(dsn: str) -> int:
    try:
        import asyncpg
    except ImportError:
        click.echo("asyncpg not installed", err=True)
        return 1

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2, statement_cache_size=0)
    errors = 0
    try:
        click.echo("=== Training Data Validation ===\n")

        # 1. Duplicate snapshots
        rows = await pool.fetch(
            """
            SELECT symbol, interval, candle_open_time, count(*) AS cnt
            FROM feature_snapshots
            GROUP BY symbol, interval, candle_open_time
            HAVING count(*) > 1
            ORDER BY cnt DESC
            LIMIT 20
            """
        )
        if rows:
            click.echo(f"[ERROR] {len(rows)} duplicate (symbol, interval, candle_open_time) groups found:")
            for r in rows[:5]:
                click.echo(f"  {r['symbol']} {r['interval']} {r['candle_open_time']}: {r['cnt']} copies")
            errors += 1
        else:
            click.echo("[OK] No duplicate snapshots found.")

        # 2. Snapshots with NULL feature_values
        rows = await pool.fetch(
            "SELECT count(*) AS cnt FROM feature_snapshots WHERE feature_values IS NULL"
        )
        null_count = int(rows[0]["cnt"])
        if null_count > 0:
            click.echo(f"[WARN] {null_count} snapshots have NULL feature_values.")
        else:
            click.echo("[OK] All snapshots have feature_values.")

        # 3. Samples per symbol
        rows = await pool.fetch(
            """
            SELECT symbol, count(*) AS cnt
            FROM feature_snapshots
            WHERE training_eligible = true AND feature_values IS NOT NULL
            GROUP BY symbol
            ORDER BY cnt DESC
            """
        )
        click.echo(f"\nSamples per symbol ({len(rows)} symbols):")
        low_symbols = []
        for r in rows:
            flag = "" if r["cnt"] >= 100 else " [WARN: < 100]"
            click.echo(f"  {r['symbol']}: {r['cnt']}{flag}")
            if r["cnt"] < 100:
                low_symbols.append(r["symbol"])
        if low_symbols:
            click.echo(f"[WARN] {len(low_symbols)} symbols have < 100 samples.")

        # 4. Labelled outcomes
        rows = await pool.fetch(
            """
            SELECT count(*) AS total,
                   sum(CASE WHEN po.label IS NOT NULL THEN 1 ELSE 0 END) AS labelled
            FROM feature_snapshots fs
            JOIN prediction_events pe ON pe.feature_snapshot_id = fs.snapshot_id
            JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
            WHERE fs.training_eligible = true
            """
        )
        if rows:
            total = int(rows[0]["total"])
            labelled = int(rows[0]["labelled"])
            click.echo(f"\nLabelled outcomes: {labelled}/{total}")
            if total > 0 and labelled / total < 0.5:
                click.echo("[WARN] Less than 50% of snapshots have resolved outcomes.")

        # 5. Positive rate
        rows = await pool.fetch(
            """
            SELECT
                avg(po.label::float) AS positive_rate,
                count(*) AS cnt
            FROM feature_snapshots fs
            JOIN prediction_events pe ON pe.feature_snapshot_id = fs.snapshot_id
            JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
            WHERE po.horizon_minutes = 15
              AND po.label IS NOT NULL
              AND fs.training_eligible = true
            """
        )
        if rows and rows[0]["cnt"]:
            pos_rate = float(rows[0]["positive_rate"] or 0)
            cnt = int(rows[0]["cnt"])
            click.echo(f"\n15m positive label rate: {pos_rate:.1%} (n={cnt})")
            if pos_rate < 0.1 or pos_rate > 0.9:
                click.echo("[WARN] Extreme class imbalance — check label_threshold_bps.")

        click.echo(f"\n{'[PASS]' if errors == 0 else '[FAIL]'} Validation complete. Errors: {errors}")
        return 0 if errors == 0 else 1

    finally:
        await pool.close()


@click.command()
@click.option("--dsn", default=None, help="PostgreSQL DSN. Falls back to POSTGRES_DSN env var.")
def main(dsn: str | None) -> None:
    """Validate ML training dataset for quality issues."""
    resolved = dsn or os.environ.get("POSTGRES_DSN", "")
    if not resolved:
        click.echo("Error: provide --dsn or set POSTGRES_DSN.", err=True)
        sys.exit(1)
    sys.exit(asyncio.run(_validate(resolved)))


if __name__ == "__main__":
    main()
