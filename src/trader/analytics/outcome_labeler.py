"""Fee-aware outcome labeler for ML training dataset generation.

Computes net_return_bps, MFE, MAE for a trade given OHLC candles over
the horizon period. Both long and short directions are handled correctly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


# Default cost assumptions (overridable via environment variables)
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
    mfe_bps: Decimal  # max favourable excursion (always positive)
    mae_bps: Decimal  # max adverse excursion (always positive)
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
    """Compute fee-net outcome label for a single trade.

    Args:
        side: "Buy" or "Sell"
        entry_price: actual fill price
        exit_price: price at horizon exit
        horizon_candles: list of OHLCV dicts with keys "high", "low" covering
                         the full horizon period (used for MFE/MAE)
        entry_fee_bps: override entry fee (default: MODEL_FALLBACK_ENTRY_FEE_BPS)
        exit_fee_bps: override exit fee (default: MODEL_FALLBACK_EXIT_FEE_BPS)
        slippage_bps: override slippage (default: MODEL_FALLBACK_SLIPPAGE_BPS)
        spread_bps: override spread (default: MODEL_FALLBACK_SPREAD_BPS)
        funding_bps: optional funding cost (default: 0, positive = cost)
    """
    e_fee = entry_fee_bps if entry_fee_bps is not None else MODEL_FALLBACK_ENTRY_FEE_BPS
    x_fee = exit_fee_bps if exit_fee_bps is not None else MODEL_FALLBACK_EXIT_FEE_BPS
    slip = slippage_bps if slippage_bps is not None else MODEL_FALLBACK_SLIPPAGE_BPS
    sprd = spread_bps if spread_bps is not None else MODEL_FALLBACK_SPREAD_BPS

    total_cost_bps = e_fee + x_fee + slip + sprd + funding_bps

    is_long = side.lower() in ("buy", "long")

    # Gross directional return in bps
    if is_long:
        gross_bps = (exit_price - entry_price) / entry_price * Decimal("10000")
    else:
        # Short: profit when price falls
        gross_bps = (entry_price - exit_price) / entry_price * Decimal("10000")

    net_return_bps = gross_bps - total_cost_bps

    # MFE/MAE over full horizon candle range
    mfe_bps = Decimal("0")
    mae_bps = Decimal("0")
    for candle in horizon_candles:
        high = Decimal(str(candle.get("high", entry_price)))
        low = Decimal(str(candle.get("low", entry_price)))
        if is_long:
            fav = (high - entry_price) / entry_price * Decimal("10000")
            adv = (entry_price - low) / entry_price * Decimal("10000")
        else:
            fav = (entry_price - low) / entry_price * Decimal("10000")
            adv = (high - entry_price) / entry_price * Decimal("10000")
        if fav > mfe_bps:
            mfe_bps = fav
        if adv > mae_bps:
            mae_bps = adv

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
