"""Promote CLI — evaluate a challenger model and promote to CHAMPION if criteria met.

Usage:
    python -m trader.training.promote --version v20260606_0300

Promotion criteria:
  - samples >= MODEL_MIN_TRAINING_SAMPLES
  - walk-forward net expectancy > 0
  - manual confirmation required (--confirm flag)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import click


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

        metrics_raw = row["metrics"] or {}
        metrics: dict[str, Any] = json.loads(metrics_raw) if isinstance(metrics_raw, str) else dict(metrics_raw)
        walk_forward_expectancy = float(
            metrics.get("walk_forward_expectancy_bps")
            or metrics.get("best_threshold_avg_net_return_bps")
            or metrics.get("avg_net_return_predicted_positive_bps")
            or 0.0
        )

        model = ChallengerModel.from_bytes(bytes(row["artifact"]), version=version)
        model.training_samples = row["training_samples"]

        can, reason = model.can_promote(
            min_samples=settings.MODEL_MIN_TRAINING_SAMPLES,
            walk_forward_expectancy=walk_forward_expectancy,
        )
        if not can:
            click.echo(f"Promotion criteria not met: {reason}", err=True)
            return

        click.echo(
            f"Model {version} meets promotion criteria "
            f"({model.training_samples} samples, walk_forward={walk_forward_expectancy:+.2f} bps)"
        )

        if not confirm:
            click.echo("Add --confirm to actually promote this model to CHAMPION")
            return

        # Demote current champion
        await pool.execute("UPDATE model_versions SET status='ROLLED_BACK' WHERE status='CHAMPION'")

        # Promote challenger
        await pool.execute(
            "UPDATE model_versions SET status='CHAMPION' WHERE version=$1",
            version,
        )

        click.echo(f"Model {version} promoted to CHAMPION")

    finally:
        await pool.close()


@click.command()
@click.option("--version", required=True, help="Model version to promote")
@click.option("--confirm", is_flag=True, default=False, help="Actually execute promotion")
def main(version: str, confirm: bool) -> None:
    """Evaluate and promote a challenger model to CHAMPION status."""
    asyncio.run(_promote(version, confirm))


if __name__ == "__main__":
    main()
