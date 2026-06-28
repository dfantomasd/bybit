"""Regression tests for post-audit wiring fixes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.features.candle_patterns import score_hammer
from trader.modules.operator_controls import OperatorControlsModule
from trader.modules.signal_policy import SignalPolicyModule


@dataclass
class _Bar:
    open: float
    high: float
    low: float
    close: float


@pytest.mark.asyncio
async def test_start_model_training_uses_create_task_not_await() -> None:
    app = MagicMock()
    app._training_start_lock = asyncio.Lock()
    app._training_task = None
    app._background_tasks = []
    journal = MagicMock()
    journal.is_enabled = True
    app._trade_journal = journal

    mod = OperatorControlsModule(app)
    mod._run_model_training = AsyncMock()  # type: ignore[method-assign]

    msg = await mod.start_model_training(min_samples=10, horizon=5, label_bps=1.0)
    assert "Обучение запущено" in msg
    assert app._training_task is not None
    assert not app._training_task.done()
    await app._training_task
    mod._run_model_training.assert_awaited_once()


def test_initial_shadow_mode_follows_execution_engine() -> None:
    app = MagicMock()
    settings = MagicMock()
    settings.SHADOW_MODE = True
    app._settings = settings
    engine = MagicMock()
    engine._shadow_mode = False
    app._execution_engine = engine

    mod = SignalPolicyModule(app)
    assert mod.initial_shadow_mode() is False


@pytest.mark.asyncio
async def test_set_shadow_mode_updates_settings_and_engine() -> None:
    app = MagicMock()
    settings = MagicMock()
    settings.SHADOW_MODE = True
    app._settings = settings
    engine = MagicMock()
    engine._shadow_mode = True
    app._execution_engine = engine
    app._fee_provider = MagicMock()

    mod = OperatorControlsModule(app)
    mod._active_execution_allowed = MagicMock(return_value=True)  # type: ignore[method-assign]

    await mod.set_shadow_mode(False)
    assert settings.SHADOW_MODE is False
    assert engine._shadow_mode is False


def test_hammer_score_can_reach_one() -> None:
    bar = _Bar(open=100.0, high=101.0, low=90.0, close=100.5)
    assert score_hammer(bar) >= 0.9


def test_runtime_settings_reports_probe_edge_gate_when_strict_shadow() -> None:
    app = MagicMock()
    app._trading_paused = False
    app._current_risk_profile_str = "scalp"
    app._model_gate_quality = None
    app._shadow_probe_eligible_symbols = set()
    app._shadow_probe_side_stats = {}
    app._scalp_strict_shadow.return_value = True
    app._execution_engine = SimpleNamespace(
        _shadow_mode=True,
        _shadow_apply_net_edge_gate=True,
        _max_entries_per_minute=1,
        _max_concurrent_pending=1,
        _max_same_side=1,
        _max_open_positions=1,
    )
    app._screener = SimpleNamespace(
        _feature_max=3,
        _exec_candidates=3,
        manual_symbols=[],
    )
    app._settings = SimpleNamespace(
        MAX_POSITIONS=1,
        SCREENER_MAX_PRICE_USD=25.0,
        MODEL_GATE_CANARY_ENABLED=False,
        MODEL_SHADOW_GATE_THRESHOLD=0.55,
        MIN_EXPECTED_NET_EDGE_PCT=0.25,
        NET_EDGE_SAFETY_MARGIN_PCT=0.05,
        SHADOW_PROBE_ENABLED=True,
        SHADOW_PROBE_PAPER_COLLECTION_MODE=True,
        SHADOW_PROBE_PAPER_REGIMES="SIDEWAYS",
        SHADOW_PROBE_MIN_NET_RETURN_PCT=0.12,
        SHADOW_PROBE_SYMBOL_TOP_N=10,
        SHADOW_PROBE_SYMBOL_WARMUP_SECONDS=60,
        SHADOW_PROBE_SELL_ENABLED=True,
        SHADOW_PROBE_SIDE_BLOCK_ENABLED=False,
        SHADOW_PROBE_SIDE_MIN_SAMPLES=8,
        SHADOW_PROBE_SIDE_BLOCK_AVG_BPS=-3.0,
        MODEL_AUTO_TRAIN_MIN_SAMPLES=1000,
        MODEL_AUTO_TRAIN_HORIZON_MINUTES=5,
        MODEL_AUTO_TRAIN_LABEL_BPS=2.0,
        MODEL_LABEL_USE_TPSL_EXIT=True,
        STRATEGY_PRIORITY_ORDER="",
        SCALP_STRATEGY_PRIORITY_ORDER="",
    )

    settings = OperatorControlsModule(app).runtime_settings()

    assert settings["shadow_apply_net_edge_gate"] is True
    assert settings["shadow_probe_bypasses_live_edge_gate"] is True
    assert settings["shadow_probe_effective_min_net_return_pct"] == settings["shadow_probe_min_net_return_pct"]
