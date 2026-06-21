"""Non-destructive audit and repair tool for the ML training dataset.

Identifies feature snapshots that were created before their corresponding
candle had fully closed (and therefore may contain intermediate prices),
then marks them as training_eligible=false in --apply mode.

Usage:
    python scripts/audit_repair_training_data.py          # dry-run (default)
    python scripts/audit_repair_training_data.py --apply  # apply changes

The script NEVER deletes any rows — it only sets training_eligible=false
on suspicious snapshots and reports counts. All changes are idempotent.
"""

from __future__ import annotations

import asyncio
import os
import sys

# Allow running as a script from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import click


async def _run(apply: bool, dsn: str) -> None:
    try:
        import asyncpg
    except ImportError:
        click.echo("asyncpg not installed — run: pip install asyncpg", err=True)
        sys.exit(1)

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        await _report(pool, apply=apply)
    finally:
        await pool.close()


async def _report(pool: object, *, apply: bool) -> None:
    import asyncpg  # type: ignore[import]

    assert isinstance(pool, asyncpg.Pool)

    click.echo("=" * 60)
    click.echo(f"Mode: {'APPLY' if apply else 'DRY-RUN'}")
    click.echo("=" * 60)

    # ------------------------------------------------------------------
    # 1. Candle counts by interval / source / confirmed
    # ------------------------------------------------------------------
    click.echo("\n--- market_candles summary ---")
    rows = await pool.fetch(
        """
        SELECT interval, source, confirmed, count(*) AS cnt
        FROM market_candles
        GROUP BY interval, source, confirmed
        ORDER BY interval, source, confirmed
        """
    )
    for r in rows:
        click.echo(
            f"  interval={r['interval']:>4}  source={r['source']:<10}"
            f"  confirmed={str(r['confirmed']):<5}  count={r['cnt']}"
        )

    # ------------------------------------------------------------------
    # 2. Potentially suspicious candles (created before close_time)
    # ------------------------------------------------------------------
    click.echo("\n--- suspicious candles (created_at < close_time) ---")
    rows = await pool.fetch(
        """
        SELECT count(*) AS suspicious_candles,
               min(open_time) AS first_open_time,
               max(open_time) AS last_open_time
        FROM market_candles
        WHERE created_at < close_time
        """
    )
    row = rows[0]
    click.echo(f"  suspicious_candles: {row['suspicious_candles']}")
    click.echo(f"  first_open_time:    {row['first_open_time']}")
    click.echo(f"  last_open_time:     {row['last_open_time']}")

    # How many were later corrected by WebSocket vs still original
    rows = await pool.fetch(
        """
        SELECT
            CASE WHEN updated_at > created_at THEN 'updated_later'
                 ELSE 'not_updated'
            END AS update_status,
            count(*) AS cnt
        FROM market_candles
        WHERE created_at < close_time
        GROUP BY 1
        ORDER BY 1
        """
    )
    for r in rows:
        click.echo(f"    {r['update_status']}: {r['cnt']}")

    rows = await pool.fetch(
        """
        SELECT source, count(*) AS cnt
        FROM market_candles
        WHERE created_at < close_time
        GROUP BY source
        ORDER BY source
        """
    )
    click.echo("  by source:")
    for r in rows:
        click.echo(f"    source={r['source']:<12} count={r['cnt']}")

    # ------------------------------------------------------------------
    # 3. Suspicious feature snapshots
    # ------------------------------------------------------------------
    click.echo("\n--- feature_snapshots eligibility ---")
    rows = await pool.fetch(
        """
        SELECT training_eligible, count(*) AS cnt
        FROM feature_snapshots
        GROUP BY training_eligible
        ORDER BY training_eligible DESC
        """
    )
    for r in rows:
        click.echo(f"  training_eligible={str(r['training_eligible']):<5}  count={r['cnt']}")

    rows = await pool.fetch(
        """
        SELECT count(*) AS suspicious_snapshots
        FROM feature_snapshots fs
        JOIN market_candles mc
            ON mc.symbol = fs.symbol
            AND mc.interval = fs.interval
            AND mc.open_time = fs.candle_open_time
        WHERE fs.created_at < mc.close_time
          AND fs.training_eligible = true
        """
    )
    suspicious = int(rows[0]["suspicious_snapshots"])
    click.echo(f"\n  snapshots created before candle close (still eligible): {suspicious}")

    if rows[0]["invalid_reason"] if False else False:
        pass  # type hint workaround

    rows = await pool.fetch(
        """
        SELECT invalid_reason, count(*) AS cnt
        FROM feature_snapshots
        WHERE training_eligible = false
        GROUP BY invalid_reason
        ORDER BY cnt DESC
        """
    )
    if rows:
        click.echo("  already-excluded snapshots by reason:")
        for r in rows:
            click.echo(f"    reason={r['invalid_reason']}  count={r['cnt']}")

    duplicate_rows = await pool.fetch(
        """
        SELECT
            symbol,
            interval,
            candle_open_time,
            feature_schema_hash,
            count(*) AS dup_count
        FROM feature_snapshots
        WHERE training_eligible = true
        GROUP BY symbol, interval, candle_open_time, feature_schema_hash
        HAVING count(*) > 1
        ORDER BY dup_count DESC, candle_open_time ASC
        """
    )
    duplicate_groups = len(duplicate_rows)
    total_duplicate_snapshots = sum(int(r["dup_count"]) - 1 for r in duplicate_rows)
    click.echo("\n--- duplicate eligible snapshots (same candle/schema) ---")
    click.echo(f"  duplicate groups: {duplicate_groups}")
    click.echo(f"  extra snapshots to invalidate: {total_duplicate_snapshots}")

    if not apply and duplicate_rows:
        duplicate_bounds = await pool.fetch(
            """
            SELECT min(candle_open_time) AS min_open,
                   max(candle_open_time) AS max_open
            FROM feature_snapshots
            WHERE training_eligible = true
              AND (symbol, interval, candle_open_time, feature_schema_hash) IN (
                  SELECT symbol, interval, candle_open_time, feature_schema_hash
                  FROM feature_snapshots
                  WHERE training_eligible = true
                  GROUP BY symbol, interval, candle_open_time, feature_schema_hash
                  HAVING count(*) > 1
              )
            """
        )
        if duplicate_bounds and duplicate_bounds[0]["min_open"]:
            click.echo(f"  min candle_open_time: {duplicate_bounds[0]['min_open']}")
            click.echo(f"  max candle_open_time: {duplicate_bounds[0]['max_open']}")

        by_symbol = await pool.fetch(
            """
            SELECT symbol,
                   count(*) AS dup_group_count,
                   sum(dup_count) AS total_rows
            FROM (
                SELECT symbol,
                       interval,
                       candle_open_time,
                       feature_schema_hash,
                       count(*) AS dup_count
                FROM feature_snapshots
                WHERE training_eligible = true
                GROUP BY symbol, interval, candle_open_time, feature_schema_hash
                HAVING count(*) > 1
            ) sub
            GROUP BY symbol
            ORDER BY total_rows DESC
            """
        )
        click.echo("  breakdown by symbol:")
        for r in by_symbol:
            click.echo(f"    symbol={r['symbol']:<12} groups={r['dup_group_count']:<4} total_rows={r['total_rows']}")

        for r in duplicate_rows[:10]:
            click.echo(
                f"    symbol={r['symbol']} interval={r['interval']} "
                f"candle_open_time={r['candle_open_time']} schema={r['feature_schema_hash']} "
                f"count={r['dup_count']}"
            )
        if len(duplicate_rows) > 10:
            click.echo(f"    ... and {len(duplicate_rows) - 10} more groups")

    click.echo(f"\n  to-be-invalidated in this run: {suspicious}")

    # ------------------------------------------------------------------
    # 4. Apply (or report) the invalidation
    # ------------------------------------------------------------------
    if apply and suspicious > 0:
        click.echo("\n[APPLY] Marking suspicious snapshots training_eligible=false ...")
        result = await pool.execute(
            """
            UPDATE feature_snapshots fs
            SET training_eligible = false,
                invalid_reason    = 'snapshot_created_before_candle_close',
                invalidated_at    = now()
            FROM market_candles mc
            WHERE mc.symbol       = fs.symbol
              AND mc.interval     = fs.interval
              AND mc.open_time    = fs.candle_open_time
              AND fs.created_at   < mc.close_time
              AND fs.training_eligible = true
            """
        )
        click.echo(f"[APPLY] Done: {result}")
    elif not apply and suspicious > 0:
        click.echo("\n[DRY-RUN] No changes made. Re-run with --apply to invalidate.")

    if apply and total_duplicate_snapshots > 0:
        click.echo("\n[APPLY] Marking duplicate eligible snapshots training_eligible=false ...")
        result = await pool.execute(
            """
            WITH ranked AS (
                SELECT snapshot_id,
                       ROW_NUMBER() OVER (
                           PARTITION BY symbol, interval, candle_open_time, feature_schema_hash
                           ORDER BY created_at ASC, snapshot_id ASC
                       ) AS rn
                FROM feature_snapshots
                WHERE training_eligible = true
            )
            UPDATE feature_snapshots fs
            SET training_eligible = false,
                invalid_reason    = 'duplicate_snapshot_same_candle',
                invalidated_at    = now()
            FROM ranked
            WHERE fs.snapshot_id = ranked.snapshot_id
              AND ranked.rn > 1
            """
        )
        click.echo(f"[APPLY] Done: {result}")

        verify = await pool.fetch(
            """
            SELECT count(*) AS remaining_duplicate_groups
            FROM (
                SELECT symbol, interval, candle_open_time, feature_schema_hash
                FROM feature_snapshots
                WHERE training_eligible = true
                GROUP BY symbol, interval, candle_open_time, feature_schema_hash
                HAVING count(*) > 1
            ) duplicates
            """
        )
        remaining = int(verify[0]["remaining_duplicate_groups"])
        click.echo(f"\n  remaining duplicate eligible groups: {remaining}")
        if remaining != 0:
            click.echo("\n[ERROR] Duplicate elimination incomplete — remaining groups > 0", err=True)
            sys.exit(1)
    elif not apply and total_duplicate_snapshots > 0:
        click.echo("\n[DRY-RUN] No duplicate changes made. Re-run with --apply to invalidate duplicates.")

    # ------------------------------------------------------------------
    # 5. Final eligible snapshot count
    # ------------------------------------------------------------------
    rows = await pool.fetch("SELECT count(*) AS cnt FROM feature_snapshots WHERE training_eligible = true")
    click.echo(f"\n  eligible snapshots after run: {rows[0]['cnt']}")

    # ------------------------------------------------------------------
    # 6. SQL for manual operator verification
    # ------------------------------------------------------------------
    click.echo("\n" + "=" * 60)
    click.echo("SQL for manual operator verification:")
    click.echo("=" * 60)
    click.echo("""
-- 1. Suspicious candles
SELECT count(*) AS suspicious_candles,
       min(open_time) AS first_open_time,
       max(open_time) AS last_open_time
FROM market_candles
WHERE created_at < close_time;

-- 2. Suspicious snapshots
SELECT count(*) AS suspicious_snapshots
FROM feature_snapshots fs
JOIN market_candles mc
    ON mc.symbol = fs.symbol AND mc.interval = fs.interval
    AND mc.open_time = fs.candle_open_time
WHERE fs.created_at < mc.close_time;

-- 3. Excluded snapshots by reason
SELECT invalid_reason, count(*) AS cnt
FROM feature_snapshots
WHERE training_eligible = false
GROUP BY invalid_reason ORDER BY cnt DESC;

-- 4. Eligible snapshots for training
SELECT count(*) AS eligible_snapshots
FROM feature_snapshots WHERE training_eligible = true;

-- 5. Duplicate eligible snapshots by source candle/schema
SELECT symbol, interval, candle_open_time, feature_schema_hash, count(*) AS dup_count
FROM feature_snapshots
WHERE training_eligible = true
GROUP BY symbol, interval, candle_open_time, feature_schema_hash
HAVING count(*) > 1
ORDER BY dup_count DESC, candle_open_time ASC;
""")


@click.command()
@click.option("--apply", is_flag=True, default=False, help="Apply changes (default: dry-run).")
@click.option(
    "--dsn",
    default=None,
    help="PostgreSQL DSN. Falls back to POSTGRES_DSN env var.",
)
def main(apply: bool, dsn: str | None) -> None:
    """Audit and (optionally) repair the ML training dataset.

    By default runs in dry-run mode — pass --apply to make changes.
    Changes are idempotent and non-destructive (no rows are deleted).
    """
    resolved_dsn = dsn or os.environ.get("POSTGRES_DSN", "")
    if not resolved_dsn:
        click.echo("Error: provide --dsn or set POSTGRES_DSN env var.", err=True)
        sys.exit(1)
    asyncio.run(_run(apply=apply, dsn=resolved_dsn))


if __name__ == "__main__":
    main()
