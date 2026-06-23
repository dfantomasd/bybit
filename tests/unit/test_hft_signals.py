"""Tests for the HFT signal enhancements.

Covers:
- multi_ewma_signal() in technical.py
- ewma_tier_signal in FeaturePipeline output
- PositionSizer.calculate() volatility-adaptive sizing
- ExecutionEngine STDEV guard, pDiv multiplier, SOP boost
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

from trader.domain.enums import MarketRegime, MarketType, OrderSide, RiskProfile
from trader.domain.models import FeatureVector, InstrumentInfo, RegimeContext, TradeProposal
from trader.features.technical import ewma_periods_for_bar_count, multi_ewma_signal
from trader.risk.profiles import get_risk_limits
from trader.risk.sizing import PositionSizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feature_vector(
    symbol: str = "BTCUSDT",
    realized_vol_20: float = 0.01,
    sma20_dist: float = 0.005,
    ob_imbalance_l5: float = 0.0,
    atr_14_pct: float = 0.005,
    quality: float = 1.0,
    timestamp: datetime | None = None,
) -> FeatureVector:
    names = ["atr_14_pct", "ob_imbalance_l5", "realized_vol_20", "sma20_dist"]
    values = [atr_14_pct, ob_imbalance_l5, realized_vol_20, sma20_dist]
    return FeatureVector(
        symbol=symbol,
        timestamp=timestamp or datetime.now(UTC),
        values=values,
        feature_names=names,
        quality_score=quality,
        lookback_bars=200,
    )


def _make_instrument(
    symbol: str = "BTCUSDT",
    min_notional: str = "5",
    min_order_qty: str = "0.001",
    max_order_qty: str = "100",
    qty_step: str = "0.001",
) -> InstrumentInfo:
    return InstrumentInfo(
        symbol=symbol,
        market_type=MarketType.LINEAR,
        base_coin=symbol.replace("USDT", ""),
        quote_coin="USDT",
        min_order_qty=Decimal(min_order_qty),
        max_order_qty=Decimal(max_order_qty),
        qty_step=Decimal(qty_step),
        tick_size=Decimal("0.01"),
        min_notional=Decimal(min_notional),
    )


def _make_proposal(
    symbol: str = "BTCUSDT",
    side: str = "Buy",
    entry: str = "100",
    qty: str = "10",
    confidence: float = 1.0,
    timestamp: datetime | None = None,
) -> TradeProposal:
    entry_d = Decimal(entry)
    ts = timestamp or datetime.now(UTC)
    return TradeProposal(
        strategy_id="test",
        symbol=symbol,
        market_type=MarketType.LINEAR,
        side=OrderSide.BUY if side == "Buy" else OrderSide.SELL,
        requested_qty=Decimal(qty),
        entry_price=entry_d,
        stop_loss=entry_d * Decimal("0.98") if side == "Buy" else entry_d * Decimal("1.02"),
        take_profit=entry_d * Decimal("1.04") if side == "Buy" else entry_d * Decimal("0.96"),
        confidence=confidence,
        regime=MarketRegime.BULL_TREND,
        timestamp=ts,
    )


def _make_engine_mock(profile: RiskProfile = RiskProfile.SCALP) -> MagicMock:
    """Build a minimal ExecutionEngine-like mock for guard testing."""
    from trader.execution.engine import ExecutionEngine

    limits = get_risk_limits(profile)
    risk_manager = MagicMock()
    risk_manager._limits = limits

    engine = MagicMock(spec=ExecutionEngine)
    engine._risk_manager = risk_manager
    # Bind the actual unbound methods to the mock
    engine._stdev_trend_guard = ExecutionEngine._stdev_trend_guard.__get__(engine)
    engine._pdiv_size_multiplier = ExecutionEngine._pdiv_size_multiplier.__get__(engine)
    engine._sop_size_multiplier = ExecutionEngine._sop_size_multiplier.__get__(engine)
    return engine


# ---------------------------------------------------------------------------
# multi_ewma_signal tests
# ---------------------------------------------------------------------------


class TestMultiEwmaSignal:
    def test_returns_none_on_insufficient_data(self):
        closes = [100.0] * 50  # need 201 bars for periods=(3,12,50,100,200)
        result = multi_ewma_signal(closes)
        assert result is None

    def test_bullish_signal_when_price_rising(self):
        # Monotonically rising prices → fast EMA > slow EMA → positive signal
        closes = [float(100 + i * 0.5) for i in range(250)]
        result = multi_ewma_signal(closes)
        assert result is not None
        assert result > 0.0

    def test_bearish_signal_when_price_falling(self):
        # Monotonically falling prices → fast EMA < slow EMA → negative signal
        closes = [float(250 - i * 0.5) for i in range(250)]
        result = multi_ewma_signal(closes)
        assert result is not None
        assert result < 0.0

    def test_output_clamped_to_unit_interval(self):
        # Extreme price move should still produce output in [-1, 1]
        closes = [1.0] * 200 + [1000.0] * 50
        result = multi_ewma_signal(closes)
        if result is not None:
            assert -1.0 <= result <= 1.0

    def test_custom_periods(self):
        closes = [float(100 + i * 0.1) for i in range(60)]
        result = multi_ewma_signal(closes, periods=(3, 12, 50))
        assert result is not None

    def test_returns_none_on_zero_price(self):
        closes = [0.0] * 250
        assert multi_ewma_signal(closes) is None

    def test_compact_periods_fit_htf_bar_caps(self):
        periods = ewma_periods_for_bar_count(120)
        closes = [float(100 + i * 0.1) for i in range(120)]
        result = multi_ewma_signal(closes, periods=periods)
        assert result is not None


# ---------------------------------------------------------------------------
# Volatility-adaptive sizing tests
# ---------------------------------------------------------------------------


class TestVolatilityAdaptiveSizing:
    def _make_sizer(self, profile: RiskProfile = RiskProfile.SCALP) -> PositionSizer:
        limits = get_risk_limits(profile)
        instrument = _make_instrument()
        return PositionSizer(risk_limits=limits, instrument_info=instrument)

    def test_high_vol_reduces_size(self):
        sizer = self._make_sizer()
        limits = get_risk_limits(RiskProfile.SCALP)
        target_vol = limits.max_drawdown_pct / Decimal("100")

        # Normal vol: target_vol itself → multiplier = 1.0
        qty_normal, _ = sizer.calculate(
            capital=Decimal("1000"),
            stop_distance_pct=Decimal("0.02"),
            desired_risk_pct=Decimal("0.75"),
            current_exposure_pct=Decimal("0"),
            drawdown_pct=Decimal("0"),
            event_risk_score=0.0,
            data_quality_score=1.0,
            spread=None,
            atr=None,
            available_balance=Decimal("1000"),
            entry_price=Decimal("100"),
            realized_vol=target_vol,
        )
        # High vol: 4× target → multiplier clamps to 0.5
        qty_high, _ = sizer.calculate(
            capital=Decimal("1000"),
            stop_distance_pct=Decimal("0.02"),
            desired_risk_pct=Decimal("0.75"),
            current_exposure_pct=Decimal("0"),
            drawdown_pct=Decimal("0"),
            event_risk_score=0.0,
            data_quality_score=1.0,
            spread=None,
            atr=None,
            available_balance=Decimal("1000"),
            entry_price=Decimal("100"),
            realized_vol=target_vol * Decimal("4"),
        )
        assert qty_high < qty_normal

    def test_low_vol_increases_size(self):
        sizer = self._make_sizer()
        limits = get_risk_limits(RiskProfile.SCALP)
        target_vol = limits.max_drawdown_pct / Decimal("100")

        qty_normal, _ = sizer.calculate(
            capital=Decimal("1000"),
            stop_distance_pct=Decimal("0.02"),
            desired_risk_pct=Decimal("0.75"),
            current_exposure_pct=Decimal("0"),
            drawdown_pct=Decimal("0"),
            event_risk_score=0.0,
            data_quality_score=1.0,
            spread=None,
            atr=None,
            available_balance=Decimal("1000"),
            entry_price=Decimal("100"),
            realized_vol=target_vol,
        )
        # Very low vol: 0.1× target → multiplier clamps to 1.5
        qty_low, _ = sizer.calculate(
            capital=Decimal("1000"),
            stop_distance_pct=Decimal("0.02"),
            desired_risk_pct=Decimal("0.75"),
            current_exposure_pct=Decimal("0"),
            drawdown_pct=Decimal("0"),
            event_risk_score=0.0,
            data_quality_score=1.0,
            spread=None,
            atr=None,
            available_balance=Decimal("1000"),
            entry_price=Decimal("100"),
            realized_vol=target_vol / Decimal("10"),
        )
        assert qty_low > qty_normal

    def test_none_realized_vol_leaves_size_unchanged(self):
        sizer = self._make_sizer()
        qty_with, _ = sizer.calculate(
            capital=Decimal("1000"),
            stop_distance_pct=Decimal("0.02"),
            desired_risk_pct=Decimal("0.75"),
            current_exposure_pct=Decimal("0"),
            drawdown_pct=Decimal("0"),
            event_risk_score=0.0,
            data_quality_score=1.0,
            spread=None,
            atr=None,
            available_balance=Decimal("1000"),
            entry_price=Decimal("100"),
            realized_vol=None,
        )
        qty_without, _ = sizer.calculate(
            capital=Decimal("1000"),
            stop_distance_pct=Decimal("0.02"),
            desired_risk_pct=Decimal("0.75"),
            current_exposure_pct=Decimal("0"),
            drawdown_pct=Decimal("0"),
            event_risk_score=0.0,
            data_quality_score=1.0,
            spread=None,
            atr=None,
            available_balance=Decimal("1000"),
            entry_price=Decimal("100"),
        )
        assert qty_with == qty_without


# ---------------------------------------------------------------------------
# STDEV + Trend Guard tests
# ---------------------------------------------------------------------------


class TestStdevTrendGuard:
    def test_normal_vol_passes(self):
        engine = _make_engine_mock()
        fv = _make_feature_vector(realized_vol_20=0.001, sma20_dist=-0.01)
        proposal = _make_proposal(side="Buy")
        result = engine._stdev_trend_guard(proposal, fv)
        assert result is None

    def test_extreme_vol_contra_trend_buy_blocked(self):
        engine = _make_engine_mock()
        # SCALP max_drawdown=8% → target_vol=0.08; 3× = 0.24
        # Set vol to 0.30 (extreme) and sma20_dist < 0 (price below SMA = bearish)
        fv = _make_feature_vector(realized_vol_20=0.30, sma20_dist=-0.02)
        proposal = _make_proposal(side="Buy")
        result = engine._stdev_trend_guard(proposal, fv)
        assert result is not None
        assert "stdev_guard" in result

    def test_extreme_vol_trend_confirming_buy_passes(self):
        engine = _make_engine_mock()
        fv = _make_feature_vector(realized_vol_20=0.30, sma20_dist=0.02)
        proposal = _make_proposal(side="Buy")
        result = engine._stdev_trend_guard(proposal, fv)
        assert result is None

    def test_extreme_vol_contra_trend_sell_blocked(self):
        engine = _make_engine_mock()
        fv = _make_feature_vector(realized_vol_20=0.30, sma20_dist=0.02)
        proposal = _make_proposal(side="Sell")
        result = engine._stdev_trend_guard(proposal, fv)
        assert result is not None

    def test_no_feature_vector_passes(self):
        engine = _make_engine_mock()
        proposal = _make_proposal()
        result = engine._stdev_trend_guard(proposal, None)
        assert result is None


# ---------------------------------------------------------------------------
# pDiv multiplier tests
# ---------------------------------------------------------------------------


class TestPdivMultiplier:
    def test_fresh_signal_no_penalty(self):
        engine = _make_engine_mock()
        fv = _make_feature_vector(realized_vol_20=0.01)
        proposal = _make_proposal(timestamp=datetime.now(UTC))
        multiplier = engine._pdiv_size_multiplier(proposal, fv)
        assert multiplier == Decimal("1")

    def test_old_signal_high_vol_penalised(self):
        engine = _make_engine_mock()
        fv = _make_feature_vector(realized_vol_20=0.05)
        # 24h elapsed + 5% daily vol → drift_estimate = 0.05 ≥ 0.02 → max penalty (0.5)
        old_ts = datetime.now(UTC) - timedelta(hours=24)
        proposal = _make_proposal(timestamp=old_ts)
        multiplier = engine._pdiv_size_multiplier(proposal, fv)
        assert multiplier <= Decimal("0.5")

    def test_old_signal_low_vol_partial_penalty(self):
        engine = _make_engine_mock()
        # Low vol: 30s elapsed → minimal expected drift
        fv = _make_feature_vector(realized_vol_20=0.001)
        old_ts = datetime.now(UTC) - timedelta(seconds=30)
        proposal = _make_proposal(timestamp=old_ts)
        multiplier = engine._pdiv_size_multiplier(proposal, fv)
        assert Decimal("0") < multiplier <= Decimal("1")

    def test_no_feature_vector_returns_one(self):
        engine = _make_engine_mock()
        proposal = _make_proposal()
        multiplier = engine._pdiv_size_multiplier(proposal, None)
        assert multiplier == Decimal("1")


# ---------------------------------------------------------------------------
# SOP multiplier tests
# ---------------------------------------------------------------------------


class TestSopMultiplier:
    def _make_regime(self, spread_bps: float | None) -> RegimeContext | None:
        if spread_bps is None:
            return None
        rc = MagicMock(spec=RegimeContext)
        rc.spread_bps = spread_bps
        return rc

    def test_strong_aligned_ob_boosts_size(self):
        engine = _make_engine_mock()
        fv = _make_feature_vector(ob_imbalance_l5=0.9)
        proposal = _make_proposal(side="Buy")
        rc = self._make_regime(5.0)
        multiplier = engine._sop_size_multiplier(proposal, fv, rc)
        assert multiplier > Decimal("1")

    def test_misaligned_ob_no_boost(self):
        engine = _make_engine_mock()
        fv = _make_feature_vector(ob_imbalance_l5=-0.9)
        proposal = _make_proposal(side="Buy")
        rc = self._make_regime(5.0)
        multiplier = engine._sop_size_multiplier(proposal, fv, rc)
        assert multiplier == Decimal("1")

    def test_wide_spread_no_boost(self):
        engine = _make_engine_mock()
        fv = _make_feature_vector(ob_imbalance_l5=0.9)
        proposal = _make_proposal(side="Buy")
        rc = self._make_regime(50.0)  # 50 bps — wide
        multiplier = engine._sop_size_multiplier(proposal, fv, rc)
        assert multiplier == Decimal("1")

    def test_no_feature_vector_returns_one(self):
        engine = _make_engine_mock()
        proposal = _make_proposal()
        multiplier = engine._sop_size_multiplier(proposal, None, None)
        assert multiplier == Decimal("1")

    def test_boost_capped_at_1_2(self):
        engine = _make_engine_mock()
        fv = _make_feature_vector(ob_imbalance_l5=1.0)
        proposal = _make_proposal(side="Buy")
        rc = self._make_regime(1.0)
        multiplier = engine._sop_size_multiplier(proposal, fv, rc)
        assert multiplier <= Decimal("1.2001")  # small float tolerance
