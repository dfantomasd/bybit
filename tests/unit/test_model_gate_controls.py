"""Tests for Telegram model_gate controls (P0.3)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trader.app import TradingApplication


def _make_app() -> TradingApplication:
    app = TradingApplication()
    app._settings = SimpleNamespace(
        MODEL_GATE_CANARY_ENABLED=False,
        MODEL_SHADOW_GATE_THRESHOLD=0.60,
        MAX_NEW_ENTRIES_PER_MINUTE=60,
        MAX_CONCURRENT_PENDING_ENTRIES=10,
        MAX_SAME_SIDE_POSITIONS=10,
        SCREENER_FEATURE_MAX_SYMBOLS=20,
        SCREENER_WIDE_MAX_SYMBOLS=40,
        SCREENER_EXECUTION_CANDIDATES=5,
    )
    app._screener = None
    return app


@pytest.mark.asyncio
async def test_model_gate_off_works() -> None:
    app = _make_app()
    result = await app._set_runtime_setting("model_gate", "off")
    assert "OFF" in result
    assert app._settings.MODEL_GATE_CANARY_ENABLED is False


@pytest.mark.asyncio
async def test_model_gate_false_works() -> None:
    app = _make_app()
    result = await app._set_runtime_setting("model_gate", "false")
    assert "OFF" in result
    assert app._settings.MODEL_GATE_CANARY_ENABLED is False


@pytest.mark.asyncio
async def test_model_gate_on_rejected() -> None:
    app = _make_app()
    with pytest.raises(ValueError, match="Render env"):
        await app._set_runtime_setting("model_gate", "on")
    assert app._settings.MODEL_GATE_CANARY_ENABLED is False


@pytest.mark.asyncio
async def test_model_gate_true_rejected() -> None:
    app = _make_app()
    with pytest.raises(ValueError, match="Render env"):
        await app._set_runtime_setting("model_gate", "true")
    assert app._settings.MODEL_GATE_CANARY_ENABLED is False


@pytest.mark.asyncio
async def test_model_gate_1_rejected() -> None:
    app = _make_app()
    with pytest.raises(ValueError, match="Render env"):
        await app._set_runtime_setting("model_gate", "1")
    assert app._settings.MODEL_GATE_CANARY_ENABLED is False


@pytest.mark.asyncio
async def test_model_gate_on_leaves_canary_false_on_failure() -> None:
    """The setting must remain False even after a rejected attempt."""
    app = _make_app()
    app._settings.MODEL_GATE_CANARY_ENABLED = False
    try:
        await app._set_runtime_setting("model_gate", "on")
    except ValueError:
        pass
    assert app._settings.MODEL_GATE_CANARY_ENABLED is False
