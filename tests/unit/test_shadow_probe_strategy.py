from __future__ import annotations

from decimal import Decimal

from trader.domain.enums import MarketType, OrderSide
from trader.domain.models import FeatureVector, InstrumentInfo
from trader.risk.net_edge import NetEdgeParams
from trader.strategies.shadow_probe import ShadowProbeStrategy, probe_notional_viable


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


def test_shadow_probe_emits_from_orderbook_imbalance() -> None:
    strategy = ShadowProbeStrategy(
        imbalance_provider=lambda _symbol: 0.10,
        min_abs_imbalance=0.03,
        cooldown_seconds=30,
    )

    proposal = strategy.evaluate(_feature_vector(ema_9=1.01, ema_21=1.0), current_price=0.5, available_balance_usd=25.0)

    assert proposal is not None
    assert proposal.strategy_id == "shadow_probe_v1"
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


def test_shadow_probe_uses_ema_bias_without_orderbook() -> None:
    strategy = ShadowProbeStrategy(imbalance_provider=lambda _symbol: None)

    proposal = strategy.evaluate(_feature_vector(ema_9=1.01, ema_21=1.0), current_price=2.0, available_balance_usd=25.0)

    assert proposal is not None
    assert proposal.side == OrderSide.BUY
    assert "ema9>ema21" in proposal.rationale


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
