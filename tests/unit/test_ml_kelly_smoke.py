"""Smoke tests for KellyAdapter and UnifiedMLController persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from trader.ml.unified_controller import UnifiedMLController
from trader.risk.kelly_adapter import KellyAdapter, KellyAdapterContext


def _make_context(*, recent_trades: list[dict] | None = None) -> KellyAdapterContext:
    return KellyAdapterContext(
        recent_trades=recent_trades or [],
        current_price=Decimal("100"),
        recent_returns_bps=[1.0, -0.5, 2.0],
        all_returns_bps=[1.0, -0.5, 2.0, 0.3],
        volatility_regime=1,
        current_drawdown_pct=-0.05,
        max_drawdown_pct=-0.2,
        strategy_id="scalp_micro_v1",
        symbol="BTCUSDT",
        total_trades=len(recent_trades or []),
        timestamp=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_kelly_adapter_fallback_with_empty_trades() -> None:
    """Empty trade history must not raise and should return conservative defaults."""
    adapter = KellyAdapter()
    kelly, frac, reasoning = await adapter.predict_kelly_sizing(_make_context())

    assert kelly == Decimal("0.10")
    assert frac == Decimal("0.25")
    assert "No trade history" in reasoning


@pytest.mark.asyncio
async def test_kelly_adapter_fallback_with_wins_only() -> None:
    """Win-only history should still produce valid sizing without errors."""
    adapter = KellyAdapter()
    trades = [{"pnl_bps": 12.0}, {"pnl_bps": 8.0}, {"pnl_bps": 5.0}]
    kelly, frac, reasoning = await adapter.predict_kelly_sizing(_make_context(recent_trades=trades))

    assert Decimal("0.01") <= kelly <= Decimal("0.25")
    assert Decimal("0.1") <= frac <= Decimal("0.5")
    assert "Statistical Kelly" in reasoning


@pytest.mark.asyncio
async def test_unified_ml_controller_save_load_roundtrip(tmp_path) -> None:
    """Saved model artifacts should reload into the controller."""
    marker = {"saved": True, "version": 1}
    kelly = SimpleNamespace(kelly_model=marker.copy(), fractional_model=None)
    regime = SimpleNamespace(regime_model=marker.copy())
    signals = SimpleNamespace(outcome_model=marker.copy())
    spread = SimpleNamespace(spread_model=marker.copy())
    stoploss = SimpleNamespace(model=marker.copy())

    controller = UnifiedMLController(
        kelly_predictor=kelly,
        regime_predictor=regime,
        signal_fusion=signals,
        spread_predictor=spread,
        stoploss_optimizer=stoploss,
        model_dir=str(tmp_path),
        auto_save=False,
    )

    await controller.save_models()

    kelly.kelly_model = None
    regime.regime_model = None
    signals.outcome_model = None
    spread.spread_model = None
    stoploss.model = None

    await controller.load_models()

    assert kelly.kelly_model == marker
    assert regime.regime_model == marker
    assert signals.outcome_model == marker
    assert spread.spread_model == marker
    assert stoploss.model == marker
    assert (tmp_path / "metadata.json").exists()
