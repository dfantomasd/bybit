"""Tests for auto-promoter safety checks (P0.1, P0.2)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trader.app import TradingApplication


def _make_journal(gate: dict, latest: dict) -> MagicMock:
    j = MagicMock()
    j.is_enabled = True
    j.get_db_diagnostics = AsyncMock(
        return_value={
            "latest_model_version": latest,
            "shadow_gate_15m": {
                "model_version": "champion_v1",
                "total_count": 100,
                "lift_vs_all_bps": 5.0,
            },
        }
    )
    j.get_shadow_gate_stats_for_version = AsyncMock(return_value=gate)
    return j


def _make_app(journal: MagicMock) -> TradingApplication:
    app = TradingApplication()
    app._settings = SimpleNamespace(
        MODEL_AUTO_PROMOTE_ENABLED=True,
        MODEL_AUTO_PROMOTE_CHECK_SECONDS=120,
        MODEL_AUTO_PROMOTE_MIN_SIGNALS=10,
        MODEL_AUTO_PROMOTE_MIN_LIFT_BPS=2.0,
        MODEL_GATE_CANARY_ENABLED=False,
    )
    shutdown = MagicMock()
    shutdown.is_set = MagicMock(side_effect=[False, True])
    app._shutdown_event = shutdown
    app._telegram_bot = None
    app._trade_journal = journal
    app._start_model_promote = AsyncMock(return_value="promoted")
    app._get_champion_walk_forward_bps = AsyncMock(return_value=1.0)
    return app


@pytest.mark.asyncio
async def test_auto_promote_blocked_when_challenger_lacks_stats() -> None:
    """Challenger with no shadow gate stats must not be promoted."""
    journal = _make_journal(
        gate={},  # empty → insufficient data for challenger
        latest={
            "version": "challenger_v2",
            "status": "SHADOW_CHALLENGER",
            "metrics": {
                "label_schema_version": "directional_net_v1",
                "quality": "GOOD",
                "horizon_minutes": 15,
            },
        },
    )
    app = _make_app(journal)

    with patch.object(asyncio, "wait_for", side_effect=TimeoutError):
        await app._run_auto_model_promoter()

    app._start_model_promote.assert_not_called()
    assert app._settings.MODEL_GATE_CANARY_ENABLED is False


@pytest.mark.asyncio
async def test_auto_promote_canary_never_auto_enabled() -> None:
    """Even after a successful promote, Canary must remain False."""
    journal = _make_journal(
        gate={
            "total_count": 50,
            "lift_vs_all_bps": 8.0,
            "pass_avg_net_return_bps": 3.0,
        },
        latest={
            "version": "challenger_v2",
            "status": "SHADOW_CHALLENGER",
            "metrics": {
                "label_schema_version": "directional_net_v1",
                "quality": "GOOD",
                "horizon_minutes": 15,
            },
        },
    )
    app = _make_app(journal)

    with patch.object(asyncio, "wait_for", side_effect=TimeoutError):
        await app._run_auto_model_promoter()

    assert app._settings.MODEL_GATE_CANARY_ENABLED is False


@pytest.mark.asyncio
async def test_auto_promote_blocked_for_champion_stats_mismatch() -> None:
    """Champion has stats but Challenger (different version) must use its own stats."""
    # Champion has 200 signals, but Challenger version has none
    journal = _make_journal(
        gate={},  # no stats for challenger version
        latest={
            "version": "challenger_v3",
            "status": "SHADOW_CHALLENGER",
            "metrics": {
                "label_schema_version": "directional_net_v1",
                "quality": "GOOD",
                "horizon_minutes": 15,
            },
        },
    )
    app = _make_app(journal)

    with patch.object(asyncio, "wait_for", side_effect=TimeoutError):
        await app._run_auto_model_promoter()

    app._start_model_promote.assert_not_called()


@pytest.mark.asyncio
async def test_auto_promote_skipped_for_incompatible_schema() -> None:
    journal = _make_journal(
        gate={"total_count": 50, "lift_vs_all_bps": 8.0, "pass_avg_net_return_bps": 3.0},
        latest={
            "version": "challenger_v2",
            "status": "SHADOW_CHALLENGER",
            "metrics": {"label_schema_version": "old_schema_v0", "quality": "GOOD", "horizon_minutes": 15},
        },
    )
    app = _make_app(journal)

    with patch.object(asyncio, "wait_for", side_effect=TimeoutError):
        await app._run_auto_model_promoter()

    app._start_model_promote.assert_not_called()


@pytest.mark.asyncio
async def test_auto_promote_skipped_for_weak_quality() -> None:
    journal = _make_journal(
        gate={"total_count": 50, "lift_vs_all_bps": 8.0, "pass_avg_net_return_bps": 3.0},
        latest={
            "version": "challenger_v2",
            "status": "SHADOW_CHALLENGER",
            "metrics": {"label_schema_version": "directional_net_v1", "quality": "WEAK", "horizon_minutes": 15},
        },
    )
    app = _make_app(journal)

    with patch.object(asyncio, "wait_for", side_effect=TimeoutError):
        await app._run_auto_model_promoter()

    app._start_model_promote.assert_not_called()
