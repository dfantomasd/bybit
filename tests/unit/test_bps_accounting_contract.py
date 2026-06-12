"""Contract tests for bps accounting across analytics, labels, and storage."""

from __future__ import annotations

from decimal import Decimal

import pytest

from trader.analytics.outcome_labeler import label_outcome
from trader.training.labels import CostModelBps, build_directional_outcome


def test_legacy_labeler_matches_canonical_sell_bps_contract() -> None:
    """Sell profit, costs, MFE, and MAE must keep the same sign convention."""
    legacy = label_outcome(
        side="Sell",
        entry_price=Decimal("100"),
        exit_price=Decimal("99"),
        horizon_candles=[{"high": "100.7", "low": "98.5"}],
        entry_fee_bps=Decimal("1.5"),
        exit_fee_bps=Decimal("2.5"),
        slippage_bps=Decimal("3.0"),
        spread_bps=Decimal("4.0"),
        funding_bps=Decimal("-1.0"),
    )
    canonical = build_directional_outcome(
        side="Sell",
        entry_price=100.0,
        exit_price=99.0,
        highs=[100.7],
        lows=[98.5],
        cost_model=CostModelBps(
            entry_fee_bps=1.5,
            exit_fee_bps=2.5,
            entry_slippage_bps=3.0,
            spread_bps=4.0,
            funding_bps=-1.0,
        ),
        label_threshold_bps=5.0,
    )
    assert float(legacy.gross_return_bps) == pytest.approx(canonical.gross_return_bps)
    assert float(legacy.net_return_bps) == pytest.approx(canonical.net_return_bps)
    assert float(legacy.mfe_bps) == pytest.approx(canonical.max_favorable_excursion_bps)
    assert float(legacy.mae_bps) == pytest.approx(canonical.max_adverse_excursion_bps)
    assert legacy.gross_return_bps == Decimal("100.0")
    assert legacy.total_cost_bps == Decimal("10.0")
    assert legacy.net_return_bps == Decimal("90.0")
    assert legacy.mfe_bps == Decimal("150.0")
    assert float(legacy.mae_bps) == pytest.approx(-70.0)