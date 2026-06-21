"""Tests for historical training seed from real candle replay."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trader.training.historical_seed import (
    DbCandle,
    _forward_path,
    _rule_side_from_features,
    seed_candles_for_symbol,
)


def _synthetic_candles(count: int, *, start_price: float = 100.0, step: float = 0.1) -> list[DbCandle]:
    start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    candles: list[DbCandle] = []
    price = start_price
    for idx in range(count):
        open_time = start + timedelta(minutes=idx)
        open_price = price
        close_price = price + step
        candles.append(
            DbCandle(
                open_time=open_time,
                open=open_price,
                high=max(open_price, close_price) + 0.05,
                low=min(open_price, close_price) - 0.05,
                close=close_price,
                volume=1000.0,
            )
        )
        price = close_price
    return candles


def test_forward_path_requires_exact_horizon_alignment() -> None:
    candles = _synthetic_candles(40)
    path = _forward_path(candles, 10, 5)
    assert path is not None
    assert len(path) == 5
    assert path[-1].open_time == candles[10].open_time + timedelta(minutes=5)


def test_forward_path_rejects_incomplete_tail() -> None:
    candles = _synthetic_candles(35)
    assert _forward_path(candles, 30, 5) is None


def test_rule_side_uses_ema_ordering() -> None:
    assert _rule_side_from_features(["ema_9", "ema_21"], [0.01, -0.01]) == "Buy"
    assert _rule_side_from_features(["ema_9", "ema_21"], [-0.01, 0.01]) == "Sell"
    assert _rule_side_from_features(["rsi_14"], [50.0]) is None


def test_seed_candles_for_symbol_generates_labelled_samples() -> None:
    candles = _synthetic_candles(120, step=0.2)
    pending, stats = seed_candles_for_symbol(
        symbol="BTCUSDT",
        interval="1",
        candles=candles,
        horizons=[5],
        label_bps_threshold=5.0,
        skip_existing=False,
    )

    assert stats.candles_loaded == 120
    assert stats.samples_written > 0
    assert stats.outcomes_resolved == stats.samples_written
    assert pending
    sample = pending[0]
    assert sample["symbol"] == "BTCUSDT"
    assert sample["side"] in {"Buy", "Sell"}
    assert "proposal_side" in sample["feature_names"]
    assert len(sample["outcomes"]) == 1
    assert sample["outcomes"][0]["horizon_minutes"] == 5
    assert sample["outcomes"][0]["label"] in {0, 1}


def test_seed_rejects_non_1m_interval() -> None:
    candles = _synthetic_candles(80)
    with pytest.raises(ValueError, match="only 1-minute"):
        seed_candles_for_symbol(
            symbol="BTCUSDT",
            interval="5",
            candles=candles,
            horizons=[5],
            label_bps_threshold=5.0,
            skip_existing=False,
        )


def test_historical_seed_cli_help() -> None:
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).parent.parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "trader.training.historical_seed", "--help"],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    assert result.returncode == 0
    assert "historical_seed" in result.stdout.lower() or "market_candles" in result.stdout


def test_trainer_worker_module_importable() -> None:
    from trader.workers import trainer

    assert hasattr(trainer, "main")
