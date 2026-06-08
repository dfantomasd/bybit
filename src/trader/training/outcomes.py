"""Direction-aware outcome calculation for ML labels.

This module contains the economic meaning of a labelled trading example.  It is
kept separate from PostgreSQL access so the calculation can be tested without a
database and reused by offline diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

LABEL_SCHEMA_VERSION = "directional_net_v2"
_BPS = Decimal("10000")
_ZERO = Decimal("0")


@dataclass(frozen=True)
class TradingCostsBps:
    """Expected round-trip trading costs expressed in basis points.

    All values must be non-negative.  Fees, spread, slippage and funding are
    subtracted from the direction-adjusted gross return when the ML label is
    calculated.
    """

    entry_fee_bps: Decimal = _ZERO
    exit_fee_bps: Decimal = _ZERO
    spread_bps: Decimal = _ZERO
    entry_slippage_bps: Decimal = _ZERO
    exit_slippage_bps: Decimal = _ZERO
    funding_bps: Decimal = _ZERO

    def __post_init__(self) -> None:
        values = {
            "entry_fee_bps": self.entry_fee_bps,
            "exit_fee_bps": self.exit_fee_bps,
            "spread_bps": self.spread_bps,
            "entry_slippage_bps": self.entry_slippage_bps,
            "exit_slippage_bps": self.exit_slippage_bps,
            "funding_bps": self.funding_bps,
        }
        for name, value in values.items():
            if value < _ZERO:
                raise ValueError(f"{name} must be non-negative")

    @property
    def total_bps(self) -> Decimal:
        """Return expected total round-trip cost in basis points."""

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
    """Resolved direction-adjusted result for one prediction event."""

    gross_return_bps: Decimal
    net_return_bps: Decimal
    max_favorable_excursion_bps: Decimal
    max_adverse_excursion_bps: Decimal
    label: int
    label_schema_version: str = LABEL_SCHEMA_VERSION


def _as_decimal(value: Decimal | int | float | str) -> Decimal:
    """Convert numeric input without importing binary float artefacts."""

    return value if isinstance(value, Decimal) else Decimal(str(value))


def _direction(side: str) -> Decimal:
    """Map Bybit-style side names to a signed price direction."""

    normalized = side.strip().upper()
    if normalized == "BUY":
        return Decimal("1")
    if normalized == "SELL":
        return Decimal("-1")
    raise ValueError(f"unsupported trade side: {side!r}")


def _extrema(values: Iterable[Decimal | int | float | str], *, name: str) -> list[Decimal]:
    converted = [_as_decimal(value) for value in values]
    if not converted:
        raise ValueError(f"{name} must contain at least one price")
    if any(value <= _ZERO for value in converted):
        raise ValueError(f"{name} must contain only positive prices")
    return converted


def calculate_directional_outcome(
    *,
    side: str,
    entry_price: Decimal | int | float | str,
    horizon_close: Decimal | int | float | str,
    path_highs: Iterable[Decimal | int | float | str],
    path_lows: Iterable[Decimal | int | float | str],
    label_bps_threshold: Decimal | int | float | str,
    costs: TradingCostsBps | None = None,
) -> DirectionalOutcome:
    """Calculate a side-aware ML label and path excursions.

    ``net_return_bps`` is positive only when the trade direction was correct
    after expected round-trip costs.  MFE is non-negative and MAE is
    non-positive, both measured from the proposed entry price over the complete
    path up to the requested horizon.
    """

    entry = _as_decimal(entry_price)
    close = _as_decimal(horizon_close)
    threshold = _as_decimal(label_bps_threshold)
    direction = _direction(side)
    highs = _extrema(path_highs, name="path_highs")
    lows = _extrema(path_lows, name="path_lows")

    if entry <= _ZERO:
        raise ValueError("entry_price must be positive")
    if close <= _ZERO:
        raise ValueError("horizon_close must be positive")
    if threshold < _ZERO:
        raise ValueError("label_bps_threshold must be non-negative")

    gross_return_bps = direction * (close - entry) / entry * _BPS
    cost_bps = costs.total_bps if costs is not None else _ZERO
    net_return_bps = gross_return_bps - cost_bps

    if direction > _ZERO:
        favorable = (max(highs) - entry) / entry * _BPS
        adverse = (min(lows) - entry) / entry * _BPS
    else:
        favorable = (entry - min(lows)) / entry * _BPS
        adverse = (entry - max(highs)) / entry * _BPS

    return DirectionalOutcome(
        gross_return_bps=gross_return_bps,
        net_return_bps=net_return_bps,
        max_favorable_excursion_bps=max(_ZERO, favorable),
        max_adverse_excursion_bps=min(_ZERO, adverse),
        label=int(net_return_bps >= threshold),
    )
