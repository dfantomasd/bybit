"""Directional, cost-aware labels for ML training.

This module is intentionally pure: it contains no database or exchange access.
It defines the one canonical formula for evaluating Buy and Sell signals so
training, diagnostics, and promotion checks cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

LABEL_SCHEMA_VERSION = "directional_net_v1"


@dataclass(frozen=True)
class CostModelBps:
    """Estimated round-trip trading costs expressed in basis points.

    All values are positive deductions from gross return. Funding may be
    negative when the position receives funding instead of paying it.
    """

    entry_fee_bps: float = 0.0
    exit_fee_bps: float = 0.0
    spread_bps: float = 0.0
    entry_slippage_bps: float = 0.0
    exit_slippage_bps: float = 0.0
    funding_bps: float = 0.0

    @property
    def total_bps(self) -> float:
        """Return total estimated round-trip cost in basis points."""
        return (
            self.entry_fee_bps
            + self.exit_fee_bps
            + self.spread_bps
            + self.entry_slippage_bps
            + self.exit_slippage_bps
            + self.funding_bps
        )


@dataclass(frozen=True)
class DirectionalOutcome:
    """Resolved outcome for one strategy signal."""

    side: str
    gross_return_bps: float
    net_return_bps: float
    max_favorable_excursion_bps: float
    max_adverse_excursion_bps: float
    label: int
    label_schema_version: str = LABEL_SCHEMA_VERSION


def normalize_side(side: str) -> str:
    """Normalize exchange side and reject unknown values."""
    normalized = side.strip().lower()
    if normalized == "buy":
        return "Buy"
    if normalized == "sell":
        return "Sell"
    raise ValueError(f"unsupported trade side: {side!r}")


def direction_multiplier(side: str) -> int:
    """Return +1 for Buy and -1 for Sell."""
    return 1 if normalize_side(side) == "Buy" else -1


def directional_return_bps(*, side: str, entry_price: float, exit_price: float) -> float:
    """Return gross PnL in bps from the signal direction.

    A profitable Sell therefore produces a positive value when price falls.
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if exit_price <= 0:
        raise ValueError("exit_price must be positive")
    direction = direction_multiplier(side)
    return direction * (exit_price - entry_price) / entry_price * 10_000.0


def directional_excursions_bps(
    *,
    side: str,
    entry_price: float,
    highs: Iterable[float],
    lows: Iterable[float],
) -> tuple[float, float]:
    """Return path-aware MFE and MAE in bps for Buy or Sell.

    MFE is always non-negative and MAE is always non-positive. The caller
    should pass highs/lows for the full horizon path, not only the last bar.
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")

    high_values = list(highs)
    low_values = list(lows)
    if not high_values or not low_values:
        raise ValueError("highs and lows must not be empty")

    normalized_side = normalize_side(side)
    if normalized_side == "Buy":
        favorable = (max(high_values) - entry_price) / entry_price * 10_000.0
        adverse = (min(low_values) - entry_price) / entry_price * 10_000.0
    else:
        favorable = (entry_price - min(low_values)) / entry_price * 10_000.0
        adverse = (entry_price - max(high_values)) / entry_price * 10_000.0

    return max(0.0, favorable), min(0.0, adverse)


def build_directional_outcome(
    *,
    side: str,
    entry_price: float,
    exit_price: float,
    highs: Iterable[float],
    lows: Iterable[float],
    cost_model: CostModelBps,
    label_threshold_bps: float,
) -> DirectionalOutcome:
    """Build one canonical cost-aware ML outcome."""
    normalized_side = normalize_side(side)
    gross_bps = directional_return_bps(side=normalized_side, entry_price=entry_price, exit_price=exit_price)
    mfe_bps, mae_bps = directional_excursions_bps(
        side=normalized_side,
        entry_price=entry_price,
        highs=highs,
        lows=lows,
    )
    net_bps = gross_bps - cost_model.total_bps
    return DirectionalOutcome(
        side=normalized_side,
        gross_return_bps=gross_bps,
        net_return_bps=net_bps,
        max_favorable_excursion_bps=mfe_bps,
        max_adverse_excursion_bps=mae_bps,
        label=1 if net_bps > label_threshold_bps else 0,
    )
