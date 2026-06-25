from __future__ import annotations

from decimal import Decimal

from trader.domain.enums import MarketType, OrderSide
from trader.domain.models import FeatureVector, InstrumentInfo
from trader.risk.net_edge import NetEdgeParams
from trader.strategies.shadow_probe import ShadowProbeStrategy, probe_notional_viable


def _feature_vector(*, symbol: str = "XRPUSDT", **overrides: float) -> FeatureVector:
    features = {
        "ema_9": 1.01,
        "ema_21": 1.0,
        "rsi_14": 55.0,
        "atr_14_pct": 0.004,
    }
    features.update(overrides)
    return FeatureVector(
        symbol=symbol,
        feature_names=list(features.keys()),
        values=list(features.values()),
        quality_score=0.9,
        lookback_bars=100,
    )


def _instrument_info(*, min_notional: str = "5", qty_step: str = "0.1") -> InstrumentInfo:
    return InstrumentInfo(
        symbol="XRPUSDT",
        market_type=MarketType.LINEAR,
        base_coin="XRP",
        quote_coin="USDT",
        min_order_qty=Decimal("0.1"),
        max_order_qty=Decimal("1000000"),
        qty_step=Decimal(qty_step),
        tick_size=Decimal("0.0001"),
        min_notional=Decimal(min_notional),
    )


def test_shadow_probe_blocks_disallowed_regime_before_proposal() -> None:
    strategy = ShadowProbeStrategy(
        imbalance_provider=lambda _symbol: 0.10,
        regime_allows=lambda _vec: False,
        cooldown_seconds=30,
    )

    proposal = strategy.evaluate(_feature_vector(), current_price=1.0, available_balance_usd=25.0)

    assert proposal is None


def test_shadow_probe_allows_matching_regime() -> None:
    strategy = ShadowProbeStrategy(
        imbalance_provider=lambda _symbol: 0.10,
        regime_allows=lambda _vec: True,
        cooldown_seconds=30,
    )

    proposal = strategy.evaluate(_feature_vector(), current_price=1.0, available_balance_usd=25.0)

    assert proposal is not None


def test_shadow_probe_emits_from_orderbook_imbalance() -> None:
    strategy = ShadowProbeStrategy(
        imbalance_provider=lambda _symbol: 0.10,
        min_abs_imbalance=0.03,
        cooldown_seconds=30,
    )

    proposal = strategy.evaluate(_feature_vector(ema_9=1.01, ema_21=1.0), current_price=0.5, available_balance_usd=25.0)

    assert proposal is not None
    assert proposal.strategy_id == "shadow_probe_hv_v2"
    assert proposal.side == OrderSide.BUY
    assert proposal.take_profit is not None
    assert proposal.stop_loss is not None
    assert proposal.requested_notional_usd is not None
    assert proposal.requested_notional_usd >= 5
    assert proposal.take_profit > proposal.entry_price
    assert proposal.stop_loss < proposal.entry_price
    assert proposal.expected_return is not None
    assert proposal.expected_return >= 0.45
    assert proposal.expected_risk == 1.0


def test_shadow_probe_rejects_ema_only_without_obi() -> None:
    strategy = ShadowProbeStrategy(imbalance_provider=lambda _symbol: None)

    proposal = strategy.evaluate(_feature_vector(ema_9=1.01, ema_21=1.0), current_price=2.0, available_balance_usd=25.0)

    assert proposal is None


def test_shadow_probe_requires_obi_above_threshold() -> None:
    strategy = ShadowProbeStrategy(
        imbalance_provider=lambda _symbol: 0.05,
        min_abs_imbalance=0.08,
    )

    assert strategy.evaluate(_feature_vector(), current_price=1.0, available_balance_usd=25.0) is None


def test_shadow_probe_cooldown_suppresses_duplicate_symbol() -> None:
    strategy = ShadowProbeStrategy(imbalance_provider=lambda _symbol: 0.05, cooldown_seconds=300)
    vec = _feature_vector()

    assert strategy.evaluate(vec, current_price=1.0, available_balance_usd=25.0) is not None
    assert strategy.evaluate(vec, current_price=1.0, available_balance_usd=25.0) is None


def test_shadow_probe_blocks_book_ema_conflict() -> None:
    strategy = ShadowProbeStrategy(imbalance_provider=lambda _symbol: -0.08)

    proposal = strategy.evaluate(_feature_vector(ema_9=1.01, ema_21=1.0), current_price=1.0, available_balance_usd=25.0)

    assert proposal is None


def test_shadow_probe_rejects_weak_net_edge() -> None:
    cost_params = NetEdgeParams(
        taker_fee_pct=0.11,
        expected_slippage_pct=0.06,
        max_spread_bps=8.0,
        funding_buffer_pct=0.01,
        safety_margin_pct=0.01,
    )
    strategy = ShadowProbeStrategy(
        imbalance_provider=lambda _symbol: 0.08,
        min_tp_pct=0.10,
        min_sl_pct=0.05,
        min_net_return_pct=0.50,
        cost_params=cost_params,
    )

    proposal = strategy.evaluate(_feature_vector(), current_price=1.0, available_balance_usd=25.0)

    assert proposal is None


def test_shadow_probe_skips_symbol_failing_min_notional() -> None:
    strategy = ShadowProbeStrategy(
        imbalance_provider=lambda _symbol: 0.08,
        instrument_info_provider=lambda _symbol: _instrument_info(min_notional="100"),
        max_notional_usd=8.0,
    )

    proposal = strategy.evaluate(_feature_vector(), current_price=50000.0, available_balance_usd=25.0)

    assert proposal is None


def test_shadow_probe_blocks_sell_when_disabled() -> None:
    strategy = ShadowProbeStrategy(
        imbalance_provider=lambda _symbol: -0.12,
        sell_enabled=False,
    )

    proposal = strategy.evaluate(_feature_vector(ema_9=0.99, ema_21=1.0), current_price=1.0, available_balance_usd=25.0)

    assert proposal is None


def test_shadow_probe_respects_side_and_symbol_filters() -> None:
    strategy = ShadowProbeStrategy(
        imbalance_provider=lambda _symbol: 0.08,
        side_blocked=lambda _symbol, side: side == "Buy",
        symbol_allowed=lambda symbol: symbol == "XRPUSDT",
    )

    assert strategy.evaluate(_feature_vector(), current_price=1.0, available_balance_usd=25.0) is None

    buy_strategy = ShadowProbeStrategy(
        imbalance_provider=lambda _symbol: 0.08,
        side_blocked=lambda _symbol, side: side == "Sell",
        symbol_allowed=lambda symbol: symbol == "XRPUSDT",
    )
    assert buy_strategy.evaluate(_feature_vector(), current_price=1.0, available_balance_usd=25.0) is not None


def test_shadow_probe_respects_max_open_positions() -> None:
    strategy = ShadowProbeStrategy(
        imbalance_provider=lambda _symbol: 0.10,
        open_positions_count=lambda: 2,
        max_open_positions=2,
    )

    assert strategy.evaluate(_feature_vector(), current_price=1.0, available_balance_usd=25.0) is None


def test_shadow_probe_burst_limit_blocks_fourth_signal() -> None:
    strategy = ShadowProbeStrategy(
        imbalance_provider=lambda _symbol: 0.10,
        cooldown_seconds=30,
        burst_max_signals=3,
        burst_window_seconds=300,
        burst_cooldown_seconds=600,
    )

    for symbol in ("XRPUSDT", "DOGEUSDT", "SOLUSDT"):
        vec = _feature_vector(symbol=symbol)
        assert strategy.evaluate(vec, current_price=1.0, available_balance_usd=25.0) is not None
    vec = _feature_vector(symbol="ADAUSDT")
    assert strategy.evaluate(vec, current_price=1.0, available_balance_usd=25.0) is None


def test_shadow_probe_net_edge_with_new_min_tp() -> None:
    costs = NetEdgeParams(
        taker_fee_pct=0.11,
        expected_slippage_pct=0.06,
        max_spread_bps=8.0,
        funding_buffer_pct=0.01,
        safety_margin_pct=0.01,
    )
    passing = ShadowProbeStrategy(
        imbalance_provider=lambda _symbol: 0.10,
        min_tp_pct=0.75,
        min_sl_pct=0.40,
        min_net_return_pct=0.30,
        cost_params=costs,
    )
    failing = ShadowProbeStrategy(
        imbalance_provider=lambda _symbol: 0.10,
        min_tp_pct=0.45,
        min_sl_pct=0.40,
        min_net_return_pct=0.30,
        cost_params=costs,
    )

    assert passing.evaluate(_feature_vector(), current_price=1.0, available_balance_usd=25.0) is not None
    assert failing.evaluate(_feature_vector(), current_price=1.0, available_balance_usd=25.0) is None


def test_probe_notional_viable_requires_buffer_after_penalties() -> None:
    info = _instrument_info(min_notional="5", qty_step="0.01")

    assert probe_notional_viable(
        price=1.0,
        notional_usd=25.0,
        info=info,
        min_notional_buffer_pct=3.0,
    )
    assert not probe_notional_viable(
        price=1.0,
        notional_usd=8.0,
        info=info,
        min_notional_buffer_pct=3.0,
    )
