from __future__ import annotations

from trader.domain.enums import OrderSide
from trader.domain.models import FeatureVector
from trader.strategies.shadow_probe import ShadowProbeStrategy


def _feature_vector(**overrides: float) -> FeatureVector:
    features = {
        "ema_9": 1.01,
        "ema_21": 1.0,
        "rsi_14": 55.0,
        "atr_14_pct": 0.004,
    }
    features.update(overrides)
    return FeatureVector(
        symbol="XRPUSDT",
        feature_names=list(features.keys()),
        values=list(features.values()),
        quality_score=0.9,
        lookback_bars=100,
    )


def test_shadow_probe_emits_from_orderbook_imbalance() -> None:
    strategy = ShadowProbeStrategy(
        imbalance_provider=lambda _symbol: -0.05,
        min_abs_imbalance=0.03,
        cooldown_seconds=30,
    )

    proposal = strategy.evaluate(_feature_vector(ema_9=0.99, ema_21=1.0), current_price=0.5, available_balance_usd=25.0)

    assert proposal is not None
    assert proposal.strategy_id == "shadow_probe_v1"
    assert proposal.side == OrderSide.SELL
    assert proposal.take_profit is not None
    assert proposal.stop_loss is not None
    assert proposal.requested_notional_usd is not None
    assert proposal.requested_notional_usd >= 5
    assert proposal.take_profit < proposal.entry_price
    assert proposal.stop_loss > proposal.entry_price
    assert proposal.expected_return is not None
    assert proposal.expected_return >= 0.45


def test_shadow_probe_uses_ema_bias_without_orderbook() -> None:
    strategy = ShadowProbeStrategy(imbalance_provider=lambda _symbol: None)

    proposal = strategy.evaluate(_feature_vector(ema_9=0.99, ema_21=1.0), current_price=2.0, available_balance_usd=25.0)

    assert proposal is not None
    assert proposal.side == OrderSide.SELL
    assert "ema9<ema21" in proposal.rationale


def test_shadow_probe_cooldown_suppresses_duplicate_symbol() -> None:
    strategy = ShadowProbeStrategy(imbalance_provider=lambda _symbol: 0.05, cooldown_seconds=300)
    vec = _feature_vector()

    assert strategy.evaluate(vec, current_price=1.0, available_balance_usd=25.0) is not None
    assert strategy.evaluate(vec, current_price=1.0, available_balance_usd=25.0) is None


def test_shadow_probe_blocks_book_ema_conflict() -> None:
    strategy = ShadowProbeStrategy(imbalance_provider=lambda _symbol: -0.08)

    proposal = strategy.evaluate(_feature_vector(ema_9=1.01, ema_21=1.0), current_price=1.0, available_balance_usd=25.0)

    assert proposal is None
