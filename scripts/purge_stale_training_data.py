#!/usr/bin/env python3
"""Purge stale Postgres training/journal rows according to retention policy.

Usage:
    python scripts/purge_stale_training_data.py
    python scripts/purge_stale_training_data.py --dry-run
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import click

from trader.config import Settings
from trader.storage.trade_journal import TradeJournal


async def _run(*, dry_run: bool) -> None:
    settings = Settings()
    journal = TradeJournal(
        postgres_dsn=settings.POSTGRES_DSN.get_secret_value(),
        enabled=True,
        pool_max_size=2,
    )
    await journal.connect()
    try:
        if dry_run:
            stats = await journal.get_storage_stats()
            click.echo("DRY-RUN — current storage stats:")
            for key, value in stats.items():
                click.echo(f"  {key}: {value}")
            click.echo("Re-run without --dry-run to apply retention policy.")
            return
        report = await journal.run_data_retention_policy(settings)
        click.echo("Retention complete:")
        for key, value in report.items():
            click.echo(f"  {key}: {value}")
    finally:
        await journal.close()


@click.command()
@click.option("--dry-run", is_flag=True, help="Show storage stats without deleting rows.")
def main(dry_run: bool) -> None:
    asyncio.run(_run(dry_run=dry_run))


if __name__ == "__main__":
    main()
