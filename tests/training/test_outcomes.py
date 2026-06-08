from decimal import Decimal

import pytest

from trader.training.outcomes import TradingCostsBps, calculate_directional_outcome


def test_buy_profit_is_positive_label() -> None:
    outcome = calculate_directional_outcome(
        side="Buy",
        entry_price="100",
        horizon_close="101",
        path_highs=["101.5"],
        path_lows=["99.8"],
        label_bps_threshold="5",
    )

    assert outcome.label == 1
    assert outcome.net_return_bps == Decimal("100")
    assert outcome.max_favorable_excursion_bps == Decimal("150")
    assert outcome.max_adverse_excursion_bps == Decimal("-20")


def test_buy_loss_is_negative_label() -> None:
    outcome = calculate_directional_outcome(
        side="Buy",
        entry_price="100",
        horizon_close="99",
        path_highs=["100.2"],
        path_lows=["98.5"],
        label_bps_threshold="5",
    )

    assert outcome.label == 0
    assert outcome.net_return_bps == Decimal("-100")


def test_sell_profit_is_positive_label() -> None:
    outcome = calculate_directional_outcome(
        side="Sell",
        entry_price="100",
        horizon_close="99",
        path_highs=["100.2"],
        path_lows=["98.5"],
        label_bps_threshold="5",
    )

    assert outcome.label == 1
    assert outcome.net_return_bps == Decimal("100")
    assert outcome.max_favorable_excursion_bps == Decimal("150")
    assert outcome.max_adverse_excursion_bps == Decimal("-20")


def test_sell_loss_is_negative_label() -> None:
    outcome = calculate_directional_outcome(
        side="Sell",
        entry_price="100",
        horizon_close="101",
        path_highs=["101.5"],
        path_lows=["99.8"],
        label_bps_threshold="5",
    )

    assert outcome.label == 0
    assert outcome.net_return_bps == Decimal("-100")


def test_round_trip_costs_can_turn_small_gross_profit_into_negative_label() -> None:
    outcome = calculate_directional_outcome(
        side="Buy",
        entry_price="100",
        horizon_close="100.10",
        path_highs=["100.10"],
        path_lows=["100"],
        label_bps_threshold="0",
        costs=TradingCostsBps(
            entry_fee_bps=Decimal("5.5"),
            exit_fee_bps=Decimal("5.5"),
        ),
    )

    assert outcome.gross_return_bps == Decimal("10")
    assert outcome.net_return_bps == Decimal("-1")
    assert outcome.label == 0


def test_unknown_side_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported trade side"):
        calculate_directional_outcome(
            side="Flat",
            entry_price="100",
            horizon_close="101",
            path_highs=["101"],
            path_lows=["100"],
            label_bps_threshold="5",
        )


def test_negative_cost_is_rejected() -> None:
    with pytest.raises(ValueError, match="entry_fee_bps must be non-negative"):
        TradingCostsBps(entry_fee_bps=Decimal("-1"))
