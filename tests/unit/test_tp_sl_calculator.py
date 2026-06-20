"""Tests for conservative TP/SL tick rounding."""

from __future__ import annotations

from decimal import Decimal

from trader.risk.tp_sl import TPSLCalculator


def test_buy_tp_sl_round_down_to_avoid_overstating_exits() -> None:
    calc = TPSLCalculator()

    sl, tp = calc.calculate(
        side="Buy",
        entry_price=Decimal("100"),
        stop_distance_pct=Decimal("0.0126"),
        take_profit_distance_pct=Decimal("0.0126"),
        tick_size=Decimal("0.1"),
    )

    assert sl == Decimal("98.7")
    assert tp == Decimal("101.2")


def test_sell_tp_sl_round_up_to_avoid_overstating_exits() -> None:
    calc = TPSLCalculator()

    sl, tp = calc.calculate(
        side="Sell",
        entry_price=Decimal("100"),
        stop_distance_pct=Decimal("0.0124"),
        take_profit_distance_pct=Decimal("0.0124"),
        tick_size=Decimal("0.1"),
    )

    assert sl == Decimal("101.3")
    assert tp == Decimal("98.8")
