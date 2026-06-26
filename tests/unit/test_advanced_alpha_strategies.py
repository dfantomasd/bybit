from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from trader.data.flow_tracker import FlowTracker
from trader.domain.enums import OrderSide
from trader.domain.models import FeatureVector
from trader.strategies.advanced_alpha import (
    FundingArbitrageStrategy,
    LiquidationHuntingStrategy,
    MarketMakingStrategy,
    OrderFlowStrategy,
    StatisticalArbitrageStrategy,
)

_SYMBOL = "TESTUSDT"


class _Book:
    def latest_imbalance(self, _symbol: str) -> float:
        return 0.35

    def microprice_deviation_bps(self, _symbol: str) -> float:
        return 1.5


class _StaleBook:
    def latest_imbalance(self, _symbol: str) -> None:
        return None

    def microprice_deviation_bps(self, _symbol: str) -> None:
        return None


def _vector(**overrides: float) -> FeatureVector:
    features = {
        "atr_14_pct": 0.004,
        "rsi_14": 0.5,
        "log_return_1": 0.0,
        "realized_vol_20": 0.001,
        "adx_14": 0.2,
        "funding_rate_bps_clipped": 0.0,
        "oi_change_pct_60m_clipped": 0.0,
    }
    features.update(overrides)
    return FeatureVector(
        feature_id=uuid.uuid4(),
        symbol=_SYMBOL,
        timestamp=datetime.now(tz=UTC),
        values=list(features.values()),
        feature_names=list(features.keys()),
        quality_score=1.0,
        lookback_bars=60,
    )


def _vector_without(*missing: str, **overrides: float) -> FeatureVector:
    features = {
        "atr_14_pct": 0.004,
        "rsi_14": 0.5,
        "log_return_1": 0.0,
        "realized_vol_20": 0.001,
        "adx_14": 0.2,
        "funding_rate_bps_clipped": 0.0,
        "oi_change_pct_60m_clipped": 0.0,
    }
    features.update(overrides)
    for name in missing:
        features.pop(name, None)
    return FeatureVector(
        feature_id=uuid.uuid4(),
        symbol=_SYMBOL,
        timestamp=datetime.now(tz=UTC),
        values=list(features.values()),
        feature_names=list(features.keys()),
        quality_score=1.0,
        lookback_bars=60,
    )


def test_flow_tracker_computes_trade_and_liquidation_pressure() -> None:
    tracker = FlowTracker(window_s=60, large_trade_notional_usd=1000)
    tracker.record_trade(_SYMBOL, OrderSide.BUY, Decimal("10"), Decimal("200"))
    tracker.record_trade(_SYMBOL, OrderSide.SELL, Decimal("10"), Decimal("50"))
    tracker.record_liquidation(_SYMBOL, OrderSide.SELL, Decimal("10"), Decimal("3000"))

    trades = tracker.trade_stats(_SYMBOL)
    liquidations = tracker.liquidation_stats(_SYMBOL)

    assert trades is not None
    assert trades.imbalance > 0.5
    assert trades.large_trade_count == 1
    assert liquidations is not None
    assert liquidations.imbalance == -1.0


def test_order_flow_strategy_buys_aligned_tape_and_book_pressure() -> None:
    tracker = FlowTracker()
    tracker.record_trade(_SYMBOL, OrderSide.BUY, Decimal("10"), Decimal("500"))
    tracker.record_trade(_SYMBOL, OrderSide.SELL, Decimal("10"), Decimal("50"))

    proposal = OrderFlowStrategy(tracker, _Book()).evaluate(_vector(), 10.0, 1000.0)

    assert proposal is not None
    assert proposal.side == OrderSide.BUY


def test_order_flow_strategy_rejects_missing_book_confirmation() -> None:
    tracker = FlowTracker()
    tracker.record_trade(_SYMBOL, OrderSide.BUY, Decimal("10"), Decimal("500"))
    tracker.record_trade(_SYMBOL, OrderSide.SELL, Decimal("10"), Decimal("50"))

    assert OrderFlowStrategy(tracker, None).evaluate(_vector(), 10.0, 1000.0) is None
    assert OrderFlowStrategy(tracker, _StaleBook()).evaluate(_vector(), 10.0, 1000.0) is None


def test_funding_arbitrage_fades_positive_funding() -> None:
    # RSI must be elevated (>=0.60) for SELL fade; OI must be rising
    proposal = FundingArbitrageStrategy(min_abs_funding_bps=5.0).evaluate(
        _vector(funding_rate_bps_clipped=8.0, oi_change_pct_60m_clipped=0.2, rsi_14=0.70),
        10.0,
        1000.0,
    )

    assert proposal is not None
    assert proposal.side == OrderSide.SELL


def test_funding_arbitrage_requires_oi_and_momentum_features() -> None:
    assert (
        FundingArbitrageStrategy(min_abs_funding_bps=5.0).evaluate(
            _vector_without("oi_change_pct_60m_clipped", funding_rate_bps_clipped=8.0),
            10.0,
            1000.0,
        )
        is None
    )
    assert (
        FundingArbitrageStrategy(min_abs_funding_bps=5.0).evaluate(
            _vector_without("log_return_1", funding_rate_bps_clipped=8.0),
            10.0,
            1000.0,
        )
        is None
    )


def test_liquidation_hunting_fades_sell_liquidation_cluster() -> None:
    # SELL liquidations = shorts closed → price fell too far → fade by BUY
    # RSI must be oversold (<= 0.30); volume_zscore absent → check skipped
    tracker = FlowTracker()
    tracker.record_liquidation(_SYMBOL, OrderSide.SELL, Decimal("10"), Decimal("3000"))

    proposal = LiquidationHuntingStrategy(tracker, min_liq_notional_usd=10_000).evaluate(
        _vector(rsi_14=0.25), 10.0, 1000.0
    )

    assert proposal is not None
    assert proposal.side == OrderSide.BUY


def test_market_making_proxy_fades_oversold_move_when_spread_is_worth_it() -> None:
    proposal = MarketMakingStrategy(lambda _s: 2.0).evaluate(
        _vector(rsi_14=0.25, log_return_1=-0.002),
        10.0,
        1000.0,
    )

    assert proposal is not None
    assert proposal.side == OrderSide.BUY


def test_stat_arb_fades_positive_return_zscore() -> None:
    proposal = StatisticalArbitrageStrategy(min_zscore=2.0).evaluate(
        _vector(log_return_1=0.004, realized_vol_20=0.001, adx_14=0.2),
        10.0,
        1000.0,
    )

    assert proposal is not None
    assert proposal.side == OrderSide.SELL
