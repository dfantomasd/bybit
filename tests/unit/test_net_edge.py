"""Tests for net-edge helpers and advanced-alpha cost gate."""

from __future__ import annotations

from trader.risk.net_edge import NetEdgeParams, net_edge_from_tp_distance, passes_min_net_edge
from trader.strategies.advanced_alpha import MarketMakingStrategy


def test_net_edge_rejects_tiny_tp_after_costs() -> None:
    params = NetEdgeParams(
        taker_fee_pct=0.055,
        expected_slippage_pct=0.03,
        max_spread_bps=8.0,
        funding_buffer_pct=0.01,
        safety_margin_pct=0.01,
    )
    # 0.12% gross TP cannot clear ~0.28% round-trip costs at default assumptions.
    assert not passes_min_net_edge(0.0012, params, 0.08)


def test_net_edge_allows_wide_tp() -> None:
    params = NetEdgeParams(
        taker_fee_pct=0.055,
        expected_slippage_pct=0.03,
        max_spread_bps=5.0,
        funding_buffer_pct=0.01,
        safety_margin_pct=0.01,
    )
    assert passes_min_net_edge(0.006, params, 0.08)
    assert net_edge_from_tp_distance(0.006, params) > 0.08


def test_market_making_rejects_low_atr_setup() -> None:
    import uuid
    from datetime import UTC, datetime

    from trader.domain.models import FeatureVector

    strategy = MarketMakingStrategy(
        spread_provider=lambda _s: 2.0,
        cost_params=NetEdgeParams(
            taker_fee_pct=0.055,
            expected_slippage_pct=0.03,
            max_spread_bps=8.0,
        ),
        min_net_return_pct=0.08,
    )
    vec = FeatureVector(
        feature_id=uuid.uuid4(),
        symbol="TESTUSDT",
        timestamp=datetime.now(tz=UTC),
        feature_names=["atr_14_pct", "rsi_14", "log_return_1"],
        values=[0.0015, 0.30, -0.002],
        quality_score=1.0,
        lookback_bars=60,
    )
    assert strategy.evaluate(vec, 100.0, 1000.0) is None
