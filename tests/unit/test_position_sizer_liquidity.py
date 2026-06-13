"""Tests for PositionSizer liquidity caps."""

from __future__ import annotations

from decimal import Decimal

from trader.domain.enums import MarketType, RiskProfile
from trader.domain.models import InstrumentInfo
from trader.risk.profiles import get_risk_limits
from trader.risk.sizing import PositionSizer


def test_position_sizer_caps_notional_to_five_pct_avg_hourly_turnover():
    info = InstrumentInfo(
        symbol="THINUSDT",
        market_type=MarketType.LINEAR,
        base_coin="THIN",
        quote_coin="USDT",
        min_order_qty=Decimal("0.001"),
        max_order_qty=Decimal("1000000"),
        qty_step=Decimal("0.001"),
        tick_size=Decimal("0.0001"),
        min_notional=Decimal("5"),
        turnover_24h=Decimal("2400"),
    )
    sizer = PositionSizer(get_risk_limits(RiskProfile.MODERATE), info)

    qty, reason = sizer.calculate(
        capital=Decimal("10000"),
        stop_distance_pct=Decimal("0.02"),
        desired_risk_pct=Decimal("1"),
        current_exposure_pct=Decimal("0"),
        drawdown_pct=Decimal("0"),
        event_risk_score=0.0,
        data_quality_score=1.0,
        spread=None,
        atr=None,
        available_balance=Decimal("10000"),
        entry_price=Decimal("1"),
    )

    assert reason == ""
    # turnover_24h / 24 * 5% = 2400 / 24 * 0.05 = 5 USDT
    assert qty == Decimal("5.000")


def test_position_sizer_compares_fractional_atr_to_fractional_stop_distance():
    info = InstrumentInfo(
        symbol="ATRUSDT",
        market_type=MarketType.LINEAR,
        base_coin="ATR",
        quote_coin="USDT",
        min_order_qty=Decimal("0.001"),
        max_order_qty=Decimal("1000000"),
        qty_step=Decimal("0.001"),
        tick_size=Decimal("0.0001"),
        min_notional=Decimal("5"),
    )
    sizer = PositionSizer(get_risk_limits(RiskProfile.MODERATE), info)

    qty, reason = sizer.calculate(
        capital=Decimal("1000"),
        stop_distance_pct=Decimal("0.002"),
        desired_risk_pct=Decimal("1"),
        current_exposure_pct=Decimal("0"),
        drawdown_pct=Decimal("0"),
        event_risk_score=0.0,
        data_quality_score=1.0,
        spread=None,
        atr=Decimal("0.004"),
        available_balance=Decimal("1000"),
        entry_price=Decimal("100"),
    )

    assert reason == ""
    assert qty > 0


def test_position_sizer_rejects_stop_below_fractional_atr_floor():
    info = InstrumentInfo(
        symbol="ATRUSDT",
        market_type=MarketType.LINEAR,
        base_coin="ATR",
        quote_coin="USDT",
        min_order_qty=Decimal("0.001"),
        max_order_qty=Decimal("1000000"),
        qty_step=Decimal("0.001"),
        tick_size=Decimal("0.0001"),
        min_notional=Decimal("5"),
    )
    sizer = PositionSizer(get_risk_limits(RiskProfile.MODERATE), info)

    qty, reason = sizer.calculate(
        capital=Decimal("1000"),
        stop_distance_pct=Decimal("0.0019"),
        desired_risk_pct=Decimal("1"),
        current_exposure_pct=Decimal("0"),
        drawdown_pct=Decimal("0"),
        event_risk_score=0.0,
        data_quality_score=1.0,
        spread=None,
        atr=Decimal("0.004"),
        available_balance=Decimal("1000"),
        entry_price=Decimal("100"),
    )

    assert qty == Decimal("0")
    assert "min ATR multiple" in reason
