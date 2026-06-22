"""Regression tests for post-audit wiring fixes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
