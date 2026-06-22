"""Postgres data retention, export-before-delete, and storage statistics."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import structlog

log = structlog.get_logger(__name__)


class RetentionStore(Protocol):
    async def _fetch(self, query: str, *args: Any) -> list[Any]: ...
    async def _execute(self, query: str, *args: Any) -> str: ...


@dataclass
class RetentionSettings:
    candle_retention_days: dict[str, int] = field(default_factory=lambda: {"1": 30, "5": 180, "15": 365, "60": 730})
    feature_snapshot_retention_days: int = 90
    feature_snapshot_invalid_retention_days: int = 7
    prediction_event_orphan_retention_days: int = 30
    shadow_signal_retention_days: int = 30
    resolved_snapshot_export_before_delete_days: int = 90
    export_enabled: bool = True
    export_dir: str = "data/retention_exports"


@dataclass
class RetentionReport:
    candles_deleted: int = 0
    invalid_snapshots_deleted: int = 0
    orphan_predictions_deleted: int = 0
    shadow_signals_deleted: int = 0
    archived_snapshots_deleted: int = 0
    exported_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candles_deleted": self.candles_deleted,
            "invalid_snapshots_deleted": self.invalid_snapshots_deleted,
            "orphan_predictions_deleted": self.orphan_predictions_deleted,
            "shadow_signals_deleted": self.shadow_signals_deleted,
            "archived_snapshots_deleted": self.archived_snapshots_deleted,
            "exported_files": self.exported_files,
            "errors": self.errors,
        }


def _parse_delete_count(result: str) -> int:
    # asyncpg returns 'DELETE 42'
    parts = str(result or "").strip().split()
    if len(parts) >= 2 and parts[-1].isdigit():
        return int(parts[-1])
    return 0


async def get_storage_stats(store: RetentionStore) -> dict[str, Any]:
    """Row counts and approximate Postgres database size."""
    stats: dict[str, Any] = {
        "tables": {},
        "database_size_bytes": None,
        "database_size_mb": None,
    }
    table_names = (
        "market_candles",
        "feature_snapshots",
        "prediction_events",
        "prediction_outcomes",
        "trade_signals",
        "order_events",
        "execution_events",
        "closed_pnl",
        "model_versions",
    )
    for table in table_names:
        try:
            rows = await store._fetch(f"SELECT count(*) AS cnt FROM {table}")
            stats["tables"][table] = int(rows[0]["cnt"]) if rows else 0
        except Exception as exc:
            stats["tables"][table] = None
            stats.setdefault("errors", []).append(f"{table}: {exc}")

    try:
        rows = await store._fetch("SELECT pg_database_size(current_database()) AS sz")
        if rows:
            size_b = int(rows[0]["sz"])
            stats["database_size_bytes"] = size_b
            stats["database_size_mb"] = round(size_b / (1024 * 1024), 2)
    except Exception as exc:
        stats.setdefault("errors", []).append(f"pg_database_size: {exc}")

    eligible = await store._fetch("SELECT count(*) AS cnt FROM feature_snapshots WHERE training_eligible = true")
    stats["feature_snapshots_eligible"] = int(eligible[0]["cnt"]) if eligible else 0
    invalid = await store._fetch("SELECT count(*) AS cnt FROM feature_snapshots WHERE training_eligible = false")
    stats["feature_snapshots_invalid"] = int(invalid[0]["cnt"]) if invalid else 0
    return stats


async def get_pnl_attribution(store: RetentionStore, *, days: int = 7) -> list[dict[str, Any]]:
    """PnL attribution by symbol for the last N days (closed trades + shadow outcomes)."""
    days = max(1, int(days))
    rows = await store._fetch(
        """
        SELECT symbol,
               count(*) FILTER (WHERE closed_pnl > 0) AS wins,
               count(*) FILTER (WHERE closed_pnl <= 0) AS losses,
               COALESCE(sum(closed_pnl), 0) AS total_pnl,
               'live' AS source
        FROM closed_pnl
        WHERE created_at >= now() - ($1::text || ' days')::interval
        GROUP BY symbol
        """,
        str(days),
    )
    result = [
        {
            "symbol": r["symbol"],
            "wins": int(r["wins"] or 0),
            "losses": int(r["losses"] or 0),
            "total_pnl": float(r["total_pnl"] or 0),
            "source": r["source"],
        }
        for r in rows
    ]
    if result:
        return sorted(result, key=lambda x: x["total_pnl"], reverse=True)

    # SHADOW fallback: aggregate resolved paper outcomes by symbol
    shadow_rows = await store._fetch(
        """
        SELECT pe.symbol,
               count(*) FILTER (WHERE po.label = 1) AS wins,
               count(*) FILTER (WHERE po.label = 0) AS losses,
               COALESCE(avg(po.net_return_bps), 0) AS avg_bps,
               count(*) AS samples,
               'shadow' AS source
        FROM prediction_outcomes po
        JOIN prediction_events pe ON pe.prediction_id = po.prediction_id
        WHERE po.created_at >= now() - ($1::text || ' days')::interval
          AND po.net_return_bps IS NOT NULL
        GROUP BY pe.symbol
        ORDER BY avg_bps DESC
        """,
        str(days),
    )
    return [
        {
            "symbol": r["symbol"],
            "wins": int(r["wins"] or 0),
            "losses": int(r["losses"] or 0),
            "total_pnl": float(r["avg_bps"] or 0),
            "samples": int(r["samples"] or 0),
            "source": r["source"],
        }
        for r in shadow_rows
    ]


async def _export_rows_jsonl_gz(
    store: RetentionStore,
    *,
    export_dir: Path,
    stem: str,
    query: str,
    query_args: tuple[Any, ...] = (),
) -> str | None:
    rows = await store._fetch(query, *query_args)
    if not rows:
        return None
    export_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    path = export_dir / f"{stem}_{ts}.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            payload = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in dict(row).items()}
            fh.write(json.dumps(payload, default=str) + "\n")
    return str(path)


async def run_data_retention(store: RetentionStore, settings: RetentionSettings) -> RetentionReport:
    """Apply retention policy; export cold rows before delete when enabled."""
    report = RetentionReport()
    export_dir = Path(settings.export_dir)

    for interval, days in settings.candle_retention_days.items():
        try:
            result = await store._execute(
                "DELETE FROM market_candles WHERE interval = $1 AND open_time < now() - ($2::text || ' days')::interval",
                interval,
                str(days),
            )
            report.candles_deleted += _parse_delete_count(result)
        except Exception as exc:
            report.errors.append(f"candles:{interval}: {exc}")

    try:
        result = await store._execute(
            """
            DELETE FROM feature_snapshots
            WHERE training_eligible = false
              AND created_at < now() - ($1::text || ' days')::interval
            """,
            str(settings.feature_snapshot_invalid_retention_days),
        )
        report.invalid_snapshots_deleted = _parse_delete_count(result)
    except Exception as exc:
        report.errors.append(f"invalid_snapshots: {exc}")

    try:
        result = await store._execute(
            """
            DELETE FROM prediction_events pe
            WHERE pe.created_at < now() - ($1::text || ' days')::interval
              AND NOT EXISTS (
                  SELECT 1 FROM prediction_outcomes po WHERE po.prediction_id = pe.prediction_id
              )
            """,
            str(settings.prediction_event_orphan_retention_days),
        )
        report.orphan_predictions_deleted = _parse_delete_count(result)
    except Exception as exc:
        report.errors.append(f"orphan_predictions: {exc}")

    try:
        result = await store._execute(
            """
            DELETE FROM trade_signals
            WHERE created_at < now() - ($1::text || ' days')::interval
            """,
            str(settings.shadow_signal_retention_days),
        )
        report.shadow_signals_deleted = _parse_delete_count(result)
    except Exception as exc:
        report.errors.append(f"shadow_signals: {exc}")

    for table in ("order_events", "execution_events", "risk_decisions"):
        try:
            result = await store._execute(
                f"""
                DELETE FROM {table}
                WHERE created_at < now() - ($1::text || ' days')::interval
                """,
                str(settings.shadow_signal_retention_days),
            )
            report.shadow_signals_deleted += _parse_delete_count(result)
        except Exception as exc:
            report.errors.append(f"{table}: {exc}")

    export_days = settings.resolved_snapshot_export_before_delete_days
    retain_days = settings.feature_snapshot_retention_days
    if retain_days > 0:
        try:
            if settings.export_enabled:
                exported = await _export_rows_jsonl_gz(
                    store,
                    export_dir=export_dir,
                    stem="feature_snapshots_archived",
                    query="""
                        SELECT fs.*
                        FROM feature_snapshots fs
                        WHERE fs.created_at < now() - ($1::text || ' days')::interval
                          AND EXISTS (
                              SELECT 1 FROM prediction_outcomes po
                              JOIN prediction_events pe ON pe.prediction_id = po.prediction_id
                              WHERE pe.feature_snapshot_id = fs.snapshot_id
                          )
                        LIMIT 5000
                    """,
                    query_args=(str(export_days),),
                )
                if exported:
                    report.exported_files.append(exported)

            result = await store._execute(
                """
                DELETE FROM feature_snapshots fs
                WHERE fs.created_at < now() - ($1::text || ' days')::interval
                  AND EXISTS (
                      SELECT 1 FROM prediction_outcomes po
                      JOIN prediction_events pe ON pe.prediction_id = po.prediction_id
                      WHERE pe.feature_snapshot_id = fs.snapshot_id
                  )
                """,
                str(retain_days),
            )
            report.archived_snapshots_deleted = _parse_delete_count(result)
        except Exception as exc:
            report.errors.append(f"archived_snapshots: {exc}")

    if report.candles_deleted or report.invalid_snapshots_deleted or report.archived_snapshots_deleted:
        log.info("data_retention.completed", **report.to_dict())
    return report
