"""Shared net-edge calculations for strategy and execution layers.

Keeps the cost formula aligned with ``trader.training.labels.CostModelBps`` and
the LIVE entry gate in ``execution.engine``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NetEdgeParams:
    """Round-trip cost assumptions expressed in percent (not bps)."""

    taker_fee_pct: float
    expected_slippage_pct: float
    max_spread_bps: float
    funding_buffer_pct: float = 0.01
    safety_margin_pct: float = 0.01


def gross_edge_pct_from_distance(tp_distance_frac: float) -> float:
    """Convert fractional TP distance (e.g. 0.004) to gross percent (0.4%)."""
    return tp_distance_frac * 100.0


def net_edge_pct(
    gross_edge_pct_val: float,
    *,
    taker_fee_pct: float,
    spread_bps: float,
    expected_slippage_pct: float,
    funding_buffer_pct: float = 0.01,
    safety_margin_pct: float = 0.0,
) -> float:
    """Return expected net edge in percent after round-trip costs."""
    spread_pct = spread_bps / 100.0
    round_trip_fee_pct = taker_fee_pct * 2.0
    round_trip_slippage_pct = expected_slippage_pct * 2.0
    return (
        gross_edge_pct_val
        - round_trip_fee_pct
        - spread_pct
        - round_trip_slippage_pct
        - funding_buffer_pct
        - safety_margin_pct
    )


def net_edge_from_tp_distance(
    tp_distance_frac: float,
    params: NetEdgeParams,
    *,
    spread_bps: float | None = None,
) -> float:
    """Net edge percent for a TP distance expressed as a price fraction."""
    spread = params.max_spread_bps if spread_bps is None else spread_bps
    return net_edge_pct(
        gross_edge_pct_from_distance(tp_distance_frac),
        taker_fee_pct=params.taker_fee_pct,
        spread_bps=spread,
        expected_slippage_pct=params.expected_slippage_pct,
        funding_buffer_pct=params.funding_buffer_pct,
        safety_margin_pct=params.safety_margin_pct,
    )


def passes_min_net_edge(
    tp_distance_frac: float,
    params: NetEdgeParams,
    min_net_return_pct: float,
    *,
    spread_bps: float | None = None,
) -> bool:
    """Return True when expected net edge clears the minimum hurdle."""
    return net_edge_from_tp_distance(tp_distance_frac, params, spread_bps=spread_bps) >= min_net_return_pct
