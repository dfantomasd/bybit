"""Effective capacity calculator.

Translates raw profile limits + current balance into a concrete answer:
"how many positions can I actually open right now?"

This is distinct from the configured profile maximum, which only makes sense
when the account has sufficient capital for each position to clear min-notional,
margin, and stop-loss budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class CapacitySnapshot:
    """Effective slot calculation result."""

    configured_max_positions: int
    """Max positions from RiskLimits.max_simultaneous_positions."""

    capital_limited_positions: int
    """How many positions the account can fund above min_notional."""

    gross_exposure_limited_positions: int
    """How many positions fit within max_total_exposure_pct."""

    margin_limited_positions: int
    """How many positions fit within available margin (leverage-adjusted)."""

    risk_at_stop_limited_positions: int
    """How many positions fit within daily loss limit (risk_per_trade_max_pct)."""

    effective_max_positions: int
    """min(configured, capital_limited, gross_exposure_limited, margin_limited,
    risk_at_stop_limited) — the tightest real constraint."""

    current_open_positions: int
    """Number of positions currently open."""

    free_slots: int
    """effective_max_positions - current_open_positions (floor 0)."""

    remaining_gross_exposure_usd: Decimal
    """USD notional that can still be added before hitting max_total_exposure_pct."""

    remaining_margin_usd: Decimal
    """Available balance remaining after accounting for open positions."""

    remaining_risk_at_stop_usd: Decimal
    """USD risk headroom before hitting risk_per_trade_max_pct limit."""


class EffectiveCapacityCalculator:
    """Calculates how many positions are actually openable given current state.

    All parameters are passed per-call so the calculator is stateless and
    easily testable. Callers should refresh the snapshot every strategy cycle.
    """

    def calculate(
        self,
        *,
        equity: Decimal,
        available_balance: Decimal,
        configured_max_positions: int,
        max_total_exposure_pct: Decimal,
        max_capital_per_position_pct: Decimal,
        risk_per_trade_max_pct: Decimal,
        current_open_positions: int,
        current_gross_exposure_usd: Decimal,
        avg_leverage: Decimal,
        min_notional_usd: Decimal,
        fee_reserve_pct: Decimal = Decimal("0.5"),
    ) -> CapacitySnapshot:
        """Calculate effective capacity.

        Args:
            equity: Total wallet equity (USDT).
            available_balance: Unreserved balance available for new margin.
            configured_max_positions: Profile max_simultaneous_positions.
            max_total_exposure_pct: Max gross notional as % of equity.
            max_capital_per_position_pct: Max single position notional as % of equity.
            risk_per_trade_max_pct: Max risk (stop loss distance) as % of capital.
            current_open_positions: Count of currently open positions.
            current_gross_exposure_usd: Sum of notional values of open positions.
            avg_leverage: Leverage multiplier for margin estimation.
            min_notional_usd: Min position notional (with safety buffer applied).
            fee_reserve_pct: % of equity to keep back as fee reserve (default 0.5%).
        """
        if equity <= Decimal("0"):
            return CapacitySnapshot(
                configured_max_positions=configured_max_positions,
                capital_limited_positions=0,
                gross_exposure_limited_positions=0,
                margin_limited_positions=0,
                risk_at_stop_limited_positions=0,
                effective_max_positions=0,
                current_open_positions=current_open_positions,
                free_slots=0,
                remaining_gross_exposure_usd=Decimal("0"),
                remaining_margin_usd=Decimal("0"),
                remaining_risk_at_stop_usd=Decimal("0"),
            )

        leverage = max(avg_leverage, Decimal("1"))
        fee_reserve_usd = equity * fee_reserve_pct / Decimal("100")
        usable_balance = max(available_balance - fee_reserve_usd, Decimal("0"))

        # --- Limit 1: capital — how many positions does our balance fund? ---
        # Each position needs at minimum: min_notional / leverage in margin
        min_margin_per_position = min_notional_usd / leverage
        if min_margin_per_position > Decimal("0"):
            capital_limited = int(usable_balance / min_margin_per_position)
        else:
            capital_limited = configured_max_positions

        # --- Limit 2: gross exposure ---
        max_gross_usd = equity * max_total_exposure_pct / Decimal("100")
        remaining_gross = max(max_gross_usd - current_gross_exposure_usd, Decimal("0"))
        # Per position: up to max_capital_per_position_pct of equity
        max_per_position_notional = equity * max_capital_per_position_pct / Decimal("100")
        slot_notional = max(min(max_per_position_notional, remaining_gross), min_notional_usd)
        if slot_notional > Decimal("0"):
            gross_limited = int(remaining_gross / slot_notional)
        else:
            gross_limited = 0

        # --- Limit 3: margin headroom ---
        # usable_balance / min_margin_per_position (same as capital_limited but
        # uses usable not total equity — they differ when open positions use margin)
        margin_limited = capital_limited  # already uses available_balance

        # --- Limit 4: risk at stop ---
        # Each trade risks risk_per_trade_max_pct of equity
        max_risk_per_trade_usd = equity * risk_per_trade_max_pct / Decimal("100")
        # Rough proxy: assume 2% stop distance to estimate risk-at-stop budget
        # (exact stop distance is per-proposal; this is a conservative estimate)
        _assumed_stop_pct = Decimal("2")
        risk_usd_per_position = min_notional_usd * _assumed_stop_pct / Decimal("100")
        remaining_risk_at_stop_usd = max(
            max_risk_per_trade_usd * Decimal(str(configured_max_positions))
            - (equity * risk_per_trade_max_pct / Decimal("100") * Decimal(str(current_open_positions))),
            Decimal("0"),
        )
        if risk_usd_per_position > Decimal("0"):
            risk_at_stop_limited = int(remaining_risk_at_stop_usd / risk_usd_per_position)
        else:
            risk_at_stop_limited = configured_max_positions

        # --- Effective max: tightest constraint ---
        effective = min(
            configured_max_positions,
            capital_limited,
            gross_limited,
            margin_limited,
            risk_at_stop_limited,
        )
        effective = max(effective, 0)

        free_slots = max(effective - current_open_positions, 0)

        remaining_margin = usable_balance

        return CapacitySnapshot(
            configured_max_positions=configured_max_positions,
            capital_limited_positions=capital_limited,
            gross_exposure_limited_positions=gross_limited,
            margin_limited_positions=margin_limited,
            risk_at_stop_limited_positions=risk_at_stop_limited,
            effective_max_positions=effective,
            current_open_positions=current_open_positions,
            free_slots=free_slots,
            remaining_gross_exposure_usd=remaining_gross,
            remaining_margin_usd=remaining_margin,
            remaining_risk_at_stop_usd=remaining_risk_at_stop_usd,
        )
