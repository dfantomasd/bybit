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
          AND fs.training_eligible = true
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
    import asyncpg

    from trader.config import Settings
    from trader.ml.challenger import ChallengerModel, ModelStatus

    settings = Settings()
    dsn = settings.POSTGRES_DSN.get_secret_value().replace("postgresql+asyncpg://", "postgresql://", 1)
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2, statement_cache_size=0)

    try:
        row = await pool.fetchrow(
            "SELECT version, status, training_samples, artifact, metrics FROM model_versions WHERE version = $1",
            version,
        )
        if not row:
            click.echo(f"Model version {version!r} not found", err=True)
            return

        if row["status"] not in (ModelStatus.SHADOW_CHALLENGER, ModelStatus.VALIDATED):
            click.echo(f"Cannot promote model in status {row['status']!r}", err=True)
            return

        if row["artifact"] is None:
            click.echo(f"Model version {version!r} has no artifact", err=True)
            return

        metrics = _parse_metrics(row["metrics"] or {})
        label_schema_version = str(metrics.get("label_schema_version") or "")
        if label_schema_version != LABEL_SCHEMA_VERSION:
            click.echo(
                f"Promotion blocked: incompatible label schema {label_schema_version!r}; "
                f"required {LABEL_SCHEMA_VERSION!r}",
                err=True,
            )
            return

        horizon_minutes = int(metrics.get("horizon_minutes") or 15)
        gate = await _shadow_gate_stats(pool, version=version, horizon_minutes=horizon_minutes)
        resolved_observations = int(gate.get("total_count") or 0)
        pass_count = int(gate.get("pass_count") or 0)
        expectancy, missing_reason = _required_gate_float(
            gate,
            "pass_avg_net_return_bps",
            "missing_shadow_pass_expectancy",
        )
        if missing_reason is not None:
            click.echo(f"Promotion criteria not met: {missing_reason}", err=True)
            return
        assert expectancy is not None
        lift_bps, missing_reason = _required_gate_float(
            gate,
            "lift_vs_all_bps",
            "missing_shadow_lift",
        )
        if missing_reason is not None:
            click.echo(f"Promotion criteria not met: {missing_reason}", err=True)
            return
        assert lift_bps is not None
        quality = str(metrics.get("quality") or "")

        model = ChallengerModel.from_bytes(bytes(row["artifact"]), version=version)
        model.training_samples = int(row["training_samples"] or model.training_samples)
        model.label_schema_version = label_schema_version

        can, reason = model.can_promote(
            min_samples=settings.MODEL_MIN_TRAINING_SAMPLES,
            min_resolved_observations=settings.MODEL_MIN_CLOSED_TRADES_FOR_PROMOTION,
            resolved_observations=resolved_observations,
            walk_forward_expectancy=float(expectancy),
            quality=quality,
            required_quality=settings.MODEL_GATE_CANARY_MIN_QUALITY,
        )
        if not can:
            click.echo(f"Promotion criteria not met: {reason}", err=True)
            return

        # Require at minimum 30 observed outcomes for statistical significance
        if int(gate.get("total_count", 0)) < 30:
            click.echo("Promotion criteria not met: insufficient_gate_observations", err=True)
            return

        min_pass_count = max(10, settings.MODEL_MIN_CLOSED_TRADES_FOR_PROMOTION // 3)
        if pass_count < min_pass_count:
            click.echo(
                f"Promotion criteria not met: insufficient_gate_passes: {pass_count} < {min_pass_count}",
                err=True,
            )
            return
        if lift_bps <= 0:
            click.echo(
                f"Promotion criteria not met: non_positive_shadow_lift: {lift_bps:+.4f} bps",
                err=True,
            )
            return

        click.echo(
            f"Model {version} meets promotion criteria "
            f"({model.training_samples} training samples, {resolved_observations} resolved shadow observations, "
            f"{pass_count} gate passes, pass expectancy={expectancy:+.2f} bps, lift={lift_bps:+.2f} bps)"
        )

        if not confirm:
            click.echo("Add --confirm to actually promote this model to CHAMPION")
            return

        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("UPDATE model_versions SET status='ROLLED_BACK' WHERE status='CHAMPION'")
                await conn.execute(
                    "UPDATE model_versions SET status='CHAMPION' WHERE version=$1 AND status IN ('SHADOW_CHALLENGER','VALIDATED')",
                    version,
                )

        click.echo(f"Model {version} promoted to CHAMPION")

    finally:
        await pool.close()


@click.command()
@click.option("--version", required=True, help="Model version to promote")
@click.option("--confirm", is_flag=True, default=False, help="Actually execute promotion")
def main(version: str, confirm: bool) -> None:
    """Evaluate and promote a compatible challenger model."""

    asyncio.run(_promote(version, confirm))


if __name__ == "__main__":
    main()
