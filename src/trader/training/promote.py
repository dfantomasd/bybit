"""Promote CLI for a directional, cost-aware challenger model.

Usage:
    python -m trader.training.promote --version v20260608_1200_h15m_dnv1

Promotion is deliberately conservative. A model must have been trained on the
current directional label schema and must accumulate resolved shadow gate
observations for its own exact model version before it can become CHAMPION.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import click

from trader.training.labels import LABEL_SCHEMA_VERSION


def _parse_metrics(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _required_gate_float(gate: dict[str, Any], key: str, reason: str) -> tuple[float | None, str | None]:
    value = gate.get(key)
    if value is None:
        return None, reason
    return float(value), None


async def _shadow_gate_stats(
    pool: Any,
    *,
    version: str,
    horizon_minutes: int,
) -> dict[str, Any]:
    """Return resolved shadow-gate statistics for one exact model version."""

    rows = await pool.fetch(
        """
        SELECT
            pe.decision,
            count(*) AS cnt,
            avg(po.net_return_bps) AS avg_net_return_bps,
            avg(po.label::double precision) AS precision
        FROM prediction_events pe
        JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
        LEFT JOIN feature_snapshots fs ON fs.snapshot_id = pe.feature_snapshot_id
        WHERE pe.model_version = $1
          AND pe.decision IN ('GATE_PASS', 'GATE_BLOCK')
          AND po.horizon_minutes = $2
          AND po.label IS NOT NULL
          AND po.label_schema_version = $3
          AND COALESCE(fs.training_eligible, true) = true
        GROUP BY pe.decision
        """,
        version,
        horizon_minutes,
        LABEL_SCHEMA_VERSION,
    )

    stats: dict[str, Any] = {
        "model_version": version,
        "horizon_minutes": horizon_minutes,
        "total_count": 0,
        "pass_count": 0,
        "block_count": 0,
        "pass_avg_net_return_bps": None,
        "block_avg_net_return_bps": None,
        "pass_precision": None,
        "lift_vs_all_bps": None,
    }
    weighted_total = 0.0
    weighted_count = 0
    for row in rows:
        decision = str(row["decision"])
        count = int(row["cnt"] or 0)
        avg_return = float(row["avg_net_return_bps"]) if row["avg_net_return_bps"] is not None else None
        precision = float(row["precision"]) if row["precision"] is not None else None
        stats["total_count"] += count
        if avg_return is not None:
            weighted_total += avg_return * count
            weighted_count += count
        if decision == "GATE_PASS":
            stats["pass_count"] = count
            stats["pass_avg_net_return_bps"] = avg_return
            stats["pass_precision"] = precision
        elif decision == "GATE_BLOCK":
            stats["block_count"] = count
            stats["block_avg_net_return_bps"] = avg_return

    pass_avg = stats["pass_avg_net_return_bps"]
    if weighted_count and pass_avg is not None:
        all_avg = weighted_total / weighted_count
        stats["all_avg_net_return_bps"] = all_avg
        stats["lift_vs_all_bps"] = float(pass_avg) - all_avg
    return stats


async def _promote(version: str, confirm: bool) -> None:
    from trader.config import Settings
    from trader.ml.auto_promotion import AutoPromotionConfig, AutoPromotionEngine
    from trader.storage.trade_journal import TradeJournal

    settings = Settings()
    journal = TradeJournal(
        postgres_dsn=settings.POSTGRES_DSN.get_secret_value(),
        enabled=settings.TRADE_JOURNAL_ENABLED,
        fetch_timeout_seconds=settings.TRADE_JOURNAL_FETCH_TIMEOUT_SECONDS,
        pool_max_size=settings.TRADE_JOURNAL_POOL_MAX_SIZE,
    )
    await journal.connect()

    try:
        engine = AutoPromotionEngine(
            trade_journal=journal,
            config=AutoPromotionConfig.from_settings(settings),
        )
        decision = await engine.should_promote(None, version)
        if not decision.promote:
            click.echo(f"Promotion criteria not met: {', '.join(decision.reasons)}", err=True)
            return

        click.echo(f"Model {version} meets promotion criteria ({decision.metrics})")

        if not confirm:
            click.echo("Add --confirm to actually promote this model to CHAMPION")
            return

        promoted = await engine.promote(version)
        if not promoted.promote:
            click.echo(
                f"Promotion criteria not met after re-check: {', '.join(promoted.reasons)}",
                err=True,
            )
            return

        click.echo(f"Model {version} promoted to CHAMPION")

    finally:
        await journal.close()


@click.command()
@click.option("--version", required=True, help="Model version to promote")
@click.option("--confirm", is_flag=True, default=False, help="Actually execute promotion")
def main(version: str, confirm: bool) -> None:
    """Evaluate and promote a compatible challenger model."""

    asyncio.run(_promote(version, confirm))


if __name__ == "__main__":
    main()
