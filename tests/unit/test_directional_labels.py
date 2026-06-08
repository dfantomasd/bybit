"""Regression tests for directional, cost-aware ML labels."""

from __future__ import annotations

import pytest

from trader.training.labels import (
    CostModelBps,
    build_directional_outcome,
    directional_excursions_bps,
    directional_return_bps,
)


def test_buy_rise_is_profitable() -> None:
    assert directional_return_bps(side="Buy", entry_price=100.0, exit_price=101.0) == pytest.approx(100.0)


def test_sell_fall_is_profitable() -> None:
    assert directional_return_bps(side="Sell", entry_price=100.0, exit_price=99.0) == pytest.approx(100.0)


def test_sell_rise_is_loss() -> None:
    assert directional_return_bps(side="Sell", entry_price=100.0, exit_price=101.0) == pytest.approx(-100.0)


def test_costs_can_flip_small_gross_profit_to_negative_label() -> None:
    costs = CostModelBps(
        entry_fee_bps=5.5,
        exit_fee_bps=5.5,
        spread_bps=8.0,
        entry_slippage_bps=3.0,
        exit_slippage_bps=3.0,
        funding_bps=1.0,
    )
    outcome = build_directional_outcome(
        side="Buy",
        entry_price=100.0,
        exit_price=100.20,
        highs=[100.25],
        lows=[99.95],
        cost_model=costs,
        label_threshold_bps=5.0,
    )
    assert outcome.gross_return_bps == pytest.approx(20.0)
    assert costs.total_bps == pytest.approx(26.0)
    assert outcome.net_return_bps == pytest.approx(-6.0)
    assert outcome.label == 0


def test_sell_excursions_are_directional_and_path_aware() -> None:
    favorable, adverse = directional_excursions_bps(
        side="Sell",
        entry_price=100.0,
        highs=[100.2, 100.7, 100.1],
        lows=[99.8, 98.5, 99.1],
    )
    assert favorable == pytest.approx(150.0)
    assert adverse == pytest.approx(-70.0)


def test_unknown_side_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported trade side"):
        directional_return_bps(side="Hold", entry_price=100.0, exit_price=101.0)
