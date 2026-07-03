"""Background trainer worker.

Automatically trains the challenger when labelled real data is ready.

Usage:
    python -m trader.workers.trainer
    python -m trader.workers.trainer --force
    python -m trader.workers.trainer --once
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time

import click

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _run_module(module: str, args: list[str]) -> int:
    cmd = [sys.executable, "-m", module, *args]
    log.info("Running %s", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


@click.command()
@click.option("--force", is_flag=True, help="Force training even if the latest model is not WEAK")
@click.option("--once", is_flag=True, help="Run one cycle and exit (no loop)")
@click.option("--interval-hours", default=6, type=int, show_default=True, help="Loop interval when not --once")
def main(force: bool, once: bool, interval_hours: int) -> None:
    """Train automatically from existing labelled market data."""
    args = ["--force"] if force else []

    if once:
        raise SystemExit(_run_module("trader.training.auto_train", args))

    while True:
        code = _run_module("trader.training.auto_train", args)
        if code != 0:
            log.warning("Auto-train finished with code %s", code)
        sleep_s = max(300, int(interval_hours) * 3600)
        log.info("Sleeping %s seconds until next auto-train check", sleep_s)
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
