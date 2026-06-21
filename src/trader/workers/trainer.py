"""Background trainer worker.

Chains real-data preparation with offline model training:
  backfill (optional) -> historical_seed -> train

Usage:
    python -m trader.workers.trainer
    python -m trader.workers.trainer --retrain
    python -m trader.workers.trainer --once
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from typing import Any

import click

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _parse_csv(raw: str) -> list[str]:
    return [item.strip().upper() for item in str(raw or "").split(",") if item.strip()]


def _settings_value(name: str, default: Any) -> Any:
    try:
        from trader.config import Settings

        return getattr(Settings(), name, default)
    except Exception:
        return os.environ.get(name, default)


def _run_module(module: str, args: list[str]) -> int:
    cmd = [sys.executable, "-m", module, *args]
    log.info("Running %s", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def _trainer_cycle(
    *,
    symbols: list[str],
    backfill_days: int,
    run_backfill: bool,
    run_seed: bool,
    run_train: bool,
    min_samples: int,
    horizon: int,
    label_bps: float,
) -> int:
    if run_backfill and symbols:
        code = _run_module(
            "trader.training.backfill",
            [
                "--symbols",
                ",".join(symbols),
                "--intervals",
                "1",
                "--days",
                str(backfill_days),
            ],
        )
        if code != 0:
            log.warning("Backfill exited with code %s", code)

    if run_seed and symbols:
        code = _run_module(
            "trader.training.historical_seed",
            [
                "--symbols",
                ",".join(symbols),
                "--interval",
                "1",
                "--horizons",
                str(horizon),
                "--label-bps",
                str(label_bps),
            ],
        )
        if code != 0:
            log.warning("Historical seed exited with code %s", code)

    if run_train:
        return _run_module(
            "trader.training.train",
            [
                "--min-samples",
                str(min_samples),
                "--horizon",
                str(horizon),
                "--label-bps",
                str(label_bps),
            ],
        )
    return 0


@click.command()
@click.option("--retrain", is_flag=True, help="Force one training cycle immediately")
@click.option("--once", is_flag=True, help="Run one cycle and exit (no loop)")
@click.option("--symbols", default="", help="Comma-separated symbols (default: TRAINER_SYMBOLS or BTCUSDT)")
@click.option("--backfill-days", default=0, type=int, help="Days of REST backfill before seeding (0=skip)")
@click.option("--interval-hours", default=6, type=int, show_default=True, help="Loop interval when not --once")
@click.option("--min-samples", default=0, type=int, help="Minimum samples for train (default: MODEL_MIN_TRAINING_SAMPLES)")
@click.option("--horizon", default=0, type=int, help="Label horizon minutes (default: MODEL_LABEL_HORIZON)")
@click.option("--label-bps", default=5.0, type=float, show_default=True)
@click.option("--no-backfill", is_flag=True, help="Skip REST backfill even when --backfill-days > 0")
@click.option("--no-seed", is_flag=True, help="Skip historical_seed step")
@click.option("--no-train", is_flag=True, help="Only backfill/seed, do not train")
def main(
    retrain: bool,
    once: bool,
    symbols: str,
    backfill_days: int,
    interval_hours: int,
    min_samples: int,
    horizon: int,
    label_bps: float,
    no_backfill: bool,
    no_seed: bool,
    no_train: bool,
) -> None:
    """Prepare real candle data and train the challenger model offline."""
    symbol_list = _parse_csv(symbols) or _parse_csv(str(_settings_value("TRAINER_SYMBOLS", "BTCUSDT")))
    min_samples = int(min_samples or _settings_value("MODEL_MIN_TRAINING_SAMPLES", 500))
    horizon = int(horizon or _settings_value("MODEL_LABEL_HORIZON", 15))
    if backfill_days <= 0:
        backfill_days = int(_settings_value("TRAINER_BACKFILL_DAYS", 0))

    run_backfill = backfill_days > 0 and not no_backfill
    run_seed = not no_seed
    run_train = (retrain or not no_train) and not (no_train and not retrain)

    if once or retrain:
        code = _trainer_cycle(
            symbols=symbol_list,
            backfill_days=backfill_days,
            run_backfill=run_backfill,
            run_seed=run_seed,
            run_train=run_train,
            min_samples=min_samples,
            horizon=horizon,
            label_bps=label_bps,
        )
        raise SystemExit(code)

    while True:
        code = _trainer_cycle(
            symbols=symbol_list,
            backfill_days=backfill_days,
            run_backfill=run_backfill,
            run_seed=run_seed,
            run_train=run_train,
            min_samples=min_samples,
            horizon=horizon,
            label_bps=label_bps,
        )
        if code != 0:
            log.warning("Trainer cycle finished with code %s", code)
        sleep_s = max(60, int(interval_hours) * 3600)
        log.info("Sleeping %s seconds until next trainer cycle", sleep_s)
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
