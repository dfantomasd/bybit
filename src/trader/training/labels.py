"""Directional, cost-aware labels for ML training.

This module is intentionally pure: it contains no database or exchange access.
It defines the one canonical formula for evaluating Buy and Sell signals so
training, diagnostics, and promotion checks cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from typing import Any

LABEL_SCHEMA_VERSION = "directional_net_v1"
"""Legacy close-at-horizon labels."""

LABEL_SCHEMA_VERSION_TPSL = "directional_net_v2"
"""TP/SL first-touch path labels aligned with scalp_micro exits."""


def active_label_schema_version(*, use_tpsl_exit: bool) -> str:
    """Return the label schema version for new outcome writes and training."""
    return LABEL_SCHEMA_VERSION_TPSL if use_tpsl_exit else LABEL_SCHEMA_VERSION


@dataclass(frozen=True)
class CostModelBps:
    """Estimated round-trip trading costs expressed in basis points.

    All values are positive deductions from gross return. Funding may be
    negative when the position receives funding instead of paying it.
    ``safety_margin_bps`` mirrors the engine's ``net_edge_safety_margin_pct``
    so that training labels use the same effective hurdle as the live gate.
    """

    entry_fee_bps: float = 0.0
    exit_fee_bps: float = 0.0
    spread_bps: float = 0.0
    entry_slippage_bps: float = 0.0
    exit_slippage_bps: float = 0.0
    funding_bps: float = 0.0
    safety_margin_bps: float = 0.0

    def __post_init__(self) -> None:
        non_negative_fields = (
            "entry_fee_bps",
            "exit_fee_bps",
            "spread_bps",
            "entry_slippage_bps",
            "exit_slippage_bps",
            "safety_margin_bps",
        )
        for name in non_negative_fields:
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative bps value")
        funding = float(self.funding_bps)
        if not isfinite(funding):
            raise ValueError("funding_bps must be finite")

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
            + self.safety_margin_bps
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
    if not isfinite(entry_price) or entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if not isfinite(exit_price) or exit_price <= 0:
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
    if not isfinite(entry_price) or entry_price <= 0:
        raise ValueError("entry_price must be positive")

    high_values = list(highs)
    low_values = list(lows)
    if not high_values or not low_values:
        raise ValueError("highs and lows must not be empty")
    if any(not isfinite(value) or value <= 0 for value in high_values):
        raise ValueError("highs must contain only positive finite prices")
    if any(not isfinite(value) or value <= 0 for value in low_values):
        raise ValueError("lows must contain only positive finite prices")

    normalized_side = normalize_side(side)
    if normalized_side == "Buy":
        favorable = (max(high_values) - entry_price) / entry_price * 10_000.0
        adverse = (min(low_values) - entry_price) / entry_price * 10_000.0
    else:
        favorable = (entry_price - min(low_values)) / entry_price * 10_000.0
        adverse = (entry_price - max(high_values)) / entry_price * 10_000.0

    return max(0.0, favorable), min(0.0, adverse)


def resolve_tpsl_exit_price(
    *,
    side: str,
    entry_price: float,
    highs: Iterable[float],
    lows: Iterable[float],
    atr_pct: float,
    tp_atr_mult: float,
    sl_atr_mult: float,
    horizon_exit_price: float,
) -> float:
    """Return the first TP/SL touch price, else the horizon close.

    On each bar SL is checked before TP (conservative intrabar ordering).
    ``atr_pct`` is ATR as a fraction of price (same units as ``atr_14_pct``).
    """
    if not isfinite(atr_pct) or atr_pct <= 0:
        return horizon_exit_price
    if not isfinite(tp_atr_mult) or tp_atr_mult <= 0:
        return horizon_exit_price
    if not isfinite(sl_atr_mult) or sl_atr_mult <= 0:
        return horizon_exit_price

    high_values = list(highs)
    low_values = list(lows)
    if not high_values or not low_values:
        return horizon_exit_price

    normalized_side = normalize_side(side)
    tp_distance = entry_price * atr_pct * tp_atr_mult
    sl_distance = entry_price * atr_pct * sl_atr_mult

    if normalized_side == "Buy":
        tp_price = entry_price + tp_distance
        sl_price = entry_price - sl_distance
        for high, low in zip(high_values, low_values, strict=True):
            if low <= sl_price:
                return sl_price
            if high >= tp_price:
                return tp_price
    else:
        tp_price = entry_price - tp_distance
        sl_price = entry_price + sl_distance
        for high, low in zip(high_values, low_values, strict=True):
            if high >= sl_price:
                return sl_price
            if low <= tp_price:
                return tp_price

    return horizon_exit_price


def atr_pct_from_feature_payload(
    feature_names: Any,
    feature_values: Any,
) -> float | None:
    """Extract ``atr_14_pct`` from a feature snapshot payload."""
    import json

    if feature_names is None or feature_values is None:
        return None
    try:
        names = json.loads(feature_names) if isinstance(feature_names, str) else list(feature_names)
        vals = json.loads(feature_values) if isinstance(feature_values, str) else list(feature_values)
        features = dict(zip(names, vals, strict=False))
        raw = features.get("atr_14_pct")
        if raw is None:
            return None
        atr_pct = float(raw)
        return atr_pct if isfinite(atr_pct) and atr_pct > 0 else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def build_directional_outcome(
    *,
    side: str,
    entry_price: float,
    exit_price: float,
    highs: Iterable[float],
    lows: Iterable[float],
    cost_model: CostModelBps,
    label_threshold_bps: float,
    atr_pct: float | None = None,
    tp_atr_mult: float = 1.0,
    sl_atr_mult: float = 0.5,
    use_tpsl_exit: bool = False,
) -> DirectionalOutcome:
    """Build one canonical cost-aware ML outcome."""
    if not isfinite(label_threshold_bps):
        raise ValueError("label_threshold_bps must be finite")
    normalized_side = normalize_side(side)
    resolved_exit = exit_price
    if use_tpsl_exit and atr_pct is not None:
        resolved_exit = resolve_tpsl_exit_price(
            side=normalized_side,
            entry_price=entry_price,
            highs=highs,
            lows=lows,
            atr_pct=atr_pct,
            tp_atr_mult=tp_atr_mult,
            sl_atr_mult=sl_atr_mult,
            horizon_exit_price=exit_price,
        )
    gross_bps = directional_return_bps(side=normalized_side, entry_price=entry_price, exit_price=resolved_exit)
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
