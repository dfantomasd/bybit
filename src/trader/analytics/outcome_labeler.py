"""Fee-aware outcome labeler for ML training dataset generation.

Computes net_return_bps, MFE, MAE for a trade given OHLC candles over
the horizon period. Both long and short directions are handled correctly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from trader.training.labels import CostModelBps, build_directional_outcome


def _env_bps(key: str, default: str) -> Decimal:
    return Decimal(os.environ.get(key, default))


MODEL_FALLBACK_ENTRY_FEE_BPS: Decimal = _env_bps("MODEL_FALLBACK_ENTRY_FEE_BPS", "5.5")
MODEL_FALLBACK_EXIT_FEE_BPS: Decimal = _env_bps("MODEL_FALLBACK_EXIT_FEE_BPS", "5.5")
MODEL_FALLBACK_SLIPPAGE_BPS: Decimal = _env_bps("MODEL_FALLBACK_SLIPPAGE_BPS", "2.0")
MODEL_FALLBACK_SPREAD_BPS: Decimal = _env_bps("MODEL_FALLBACK_SPREAD_BPS", "2.0")


@dataclass
class OutcomeLabel:
    """Labelled outcome for a single trade horizon."""

    gross_return_bps: Decimal
    net_return_bps: Decimal
    mfe_bps: Decimal
    mae_bps: Decimal
    side: str
    entry_price: Decimal
    exit_price: Decimal
    total_cost_bps: Decimal
    funding_bps: Decimal = field(default=Decimal("0"))


def label_outcome(
    *,
    side: str,
    entry_price: Decimal,
    exit_price: Decimal,
    horizon_candles: list[dict[str, Any]],
    entry_fee_bps: Decimal | None = None,
    exit_fee_bps: Decimal | None = None,
    slippage_bps: Decimal | None = None,
    spread_bps: Decimal | None = None,
    funding_bps: Decimal = Decimal("0"),
) -> OutcomeLabel:
    """Compute fee-net outcome label for a single trade."""
    e_fee = entry_fee_bps if entry_fee_bps is not None else MODEL_FALLBACK_ENTRY_FEE_BPS
    x_fee = exit_fee_bps if exit_fee_bps is not None else MODEL_FALLBACK_EXIT_FEE_BPS
    slip = slippage_bps if slippage_bps is not None else MODEL_FALLBACK_SLIPPAGE_BPS
    sprd = spread_bps if spread_bps is not None else MODEL_FALLBACK_SPREAD_BPS

    highs = [float(Decimal(str(candle.get("high", entry_price)))) for candle in horizon_candles]
    lows = [float(Decimal(str(candle.get("low", entry_price)))) for candle in horizon_candles]
    if not highs or not lows:
        highs = [float(entry_price)]
        lows = [float(entry_price)]

    cost_model = CostModelBps(
        entry_fee_bps=float(e_fee),
        exit_fee_bps=float(x_fee),
        spread_bps=float(sprd),
        entry_slippage_bps=float(slip),
        funding_bps=float(funding_bps),
    )
    outcome = build_directional_outcome(
        side=side,
        entry_price=float(entry_price),
        exit_price=float(exit_price),
        highs=highs,
        lows=lows,
        cost_model=cost_model,
        label_threshold_bps=0.0,
    )
    gross_bps = Decimal(str(outcome.gross_return_bps))
    net_return_bps = Decimal(str(outcome.net_return_bps))
    mfe_bps = Decimal(str(outcome.max_favorable_excursion_bps))
    mae_bps = Decimal(str(outcome.max_adverse_excursion_bps))
    total_cost_bps = Decimal(str(cost_model.total_bps))

    return OutcomeLabel(
        gross_return_bps=gross_bps,
        net_return_bps=net_return_bps,
        mfe_bps=mfe_bps,
        mae_bps=mae_bps,
        side=side,
        entry_price=entry_price,
        exit_price=exit_price,
        total_cost_bps=total_cost_bps,
        funding_bps=funding_bps,
    )