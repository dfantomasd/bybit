"""Compatibility patch for direction-aware ML outcome labels.

The existing trade journal is intentionally left intact while the corrected
resolver is rolled out.  Importing ``trader.storage`` installs these methods on
``TradeJournal`` exactly once.  After the rollout is verified in shadow mode,
this module can be folded into ``trade_journal.py``.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import timedelta
from decimal import Decimal
from typing import Any

from trader.storage import trade_journal as _trade_journal
from trader.training.outcomes import (
    LABEL_SCHEMA_VERSION,
    TradingCostsBps,
    calculate_directional_outcome,
)

# Conservative expected round-trip costs for 1-minute market-order labels.
# These defaults intentionally avoid teaching the model that tiny gross moves
# are profitable.  A later config wiring patch can override them explicitly.
DEFAULT_TRAINING_COSTS_BPS = TradingCostsBps(
    entry_fee_bps=Decimal("5.5"),
    exit_fee_bps=Decimal("5.5"),
    spread_bps=Decimal("8"),
    entry_slippage_bps=Decimal("3"),
    exit_slippage_bps=Decimal("3"),
    funding_bps=Decimal("1"),
)

_ORIGINAL_ENSURE_SCHEMA = _trade_journal.TradeJournal._ensure_schema
_PATCH_INSTALLED = False


async def _ensure_schema_with_directional_labels(self: Any) -> None:
    """Create the original schema and add versioned directional-label columns."""

    await _ORIGINAL_ENSURE_SCHEMA(self)
    assert self._pool is not None
    async with self._pool.acquire() as conn:
        await conn.execute(
            """
            ALTER TABLE prediction_outcomes
                ADD COLUMN IF NOT EXISTS gross_return_bps double precision;
            ALTER TABLE prediction_outcomes
                ADD COLUMN IF NOT EXISTS cost_bps double precision;
            ALTER TABLE prediction_outcomes
                ADD COLUMN IF NOT EXISTS label_schema_version text;
            CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_schema_horizon
                ON prediction_outcomes (label_schema_version, horizon_minutes);
            """
        )


async def _resolve_prediction_outcomes_directional(
    self: Any,
    *,
    prediction_id: str,
    horizon_minutes: int,
    net_return_bps: float,
    max_favorable_excursion_bps: float,
    max_adverse_excursion_bps: float,
    label: int,
    gross_return_bps: float | None = None,
    cost_bps: float | None = None,
    label_schema_version: str = LABEL_SCHEMA_VERSION,
) -> None:
    """Persist a versioned directional outcome and refresh all derived fields."""

    await self._execute(
        """
        INSERT INTO prediction_outcomes (
            prediction_id, horizon_minutes, gross_return_bps, cost_bps,
            net_return_bps, max_favorable_excursion_bps,
            max_adverse_excursion_bps, label, label_schema_version, resolved_at
        )
        VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, now())
        ON CONFLICT (prediction_id, horizon_minutes) DO UPDATE SET
            gross_return_bps = EXCLUDED.gross_return_bps,
            cost_bps = EXCLUDED.cost_bps,
            net_return_bps = EXCLUDED.net_return_bps,
            max_favorable_excursion_bps = EXCLUDED.max_favorable_excursion_bps,
            max_adverse_excursion_bps = EXCLUDED.max_adverse_excursion_bps,
            label = EXCLUDED.label,
            label_schema_version = EXCLUDED.label_schema_version,
            resolved_at = now()
        """,
        prediction_id,
        horizon_minutes,
        gross_return_bps,
        cost_bps,
        net_return_bps,
        max_favorable_excursion_bps,
        max_adverse_excursion_bps,
        label,
        label_schema_version,
    )


async def _resolve_outcomes_from_candles_directional(
    self: Any,
    *,
    horizon_minutes: int,
    label_bps_threshold: float = 5.0,
    limit: int = 200,
    costs: TradingCostsBps | None = None,
) -> int:
    """Resolve labels using signal direction, full candle path and costs.

    Old unversioned labels are deliberately selected again and overwritten with
    ``directional_net_v2`` results.  Only confirmed one-minute candles are used.
    """

    expected_costs = costs or DEFAULT_TRAINING_COSTS_BPS
    rows = await self._fetch(
        """
        SELECT
            pe.prediction_id,
            pe.symbol,
            pe.strategy_signal,
            fs.candle_open_time AS entry_time
        FROM prediction_events pe
        JOIN feature_snapshots fs ON fs.snapshot_id = pe.feature_snapshot_id
        LEFT JOIN prediction_outcomes po
            ON po.prediction_id = pe.prediction_id
            AND po.horizon_minutes = $1
            AND po.label_schema_version = $3
        WHERE po.prediction_id IS NULL
          AND pe.created_at < now() - ($1 * interval '1 minute')
          AND pe.feature_snapshot_id IS NOT NULL
          AND upper(COALESCE(pe.strategy_signal, '')) IN ('BUY', 'SELL')
        ORDER BY pe.created_at ASC
        LIMIT $2
        """,
        horizon_minutes,
        limit,
        LABEL_SCHEMA_VERSION,
    )

    resolved = 0
    for row in rows:
        prediction_id = str(row["prediction_id"])
        symbol = str(row["symbol"])
        side = str(row["strategy_signal"])
        entry_time = row["entry_time"]
        horizon_time = entry_time + timedelta(minutes=horizon_minutes)

        entry_rows = await self._fetch(
            """
            SELECT close
            FROM market_candles
            WHERE symbol = $1
              AND interval = '1'
              AND confirmed = true
              AND open_time <= $2
            ORDER BY open_time DESC
            LIMIT 1
            """,
            symbol,
            entry_time,
        )
        if not entry_rows:
            continue

        path_rows = await self._fetch(
            """
            SELECT close, high, low
            FROM market_candles
            WHERE symbol = $1
              AND interval = '1'
              AND confirmed = true
              AND open_time > $2
              AND open_time <= $3
            ORDER BY open_time ASC
            """,
            symbol,
            entry_time,
            horizon_time,
        )
        if not path_rows:
            continue

        outcome = calculate_directional_outcome(
            side=side,
            entry_price=entry_rows[0]["close"],
            horizon_close=path_rows[-1]["close"],
            path_highs=[path_row["high"] for path_row in path_rows],
            path_lows=[path_row["low"] for path_row in path_rows],
            label_bps_threshold=label_bps_threshold,
            costs=expected_costs,
        )
        await self.resolve_prediction_outcomes(
            prediction_id=prediction_id,
            horizon_minutes=horizon_minutes,
            gross_return_bps=float(outcome.gross_return_bps),
            cost_bps=float(expected_costs.total_bps),
            net_return_bps=float(outcome.net_return_bps),
            max_favorable_excursion_bps=float(outcome.max_favorable_excursion_bps),
            max_adverse_excursion_bps=float(outcome.max_adverse_excursion_bps),
            label=outcome.label,
            label_schema_version=outcome.label_schema_version,
        )
        resolved += 1

    return resolved


def install_directional_outcomes_patch() -> None:
    """Install corrected TradeJournal methods once per Python process."""

    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return

    journal_cls = _trade_journal.TradeJournal
    journal_cls._ensure_schema = _ensure_schema_with_directional_labels
    journal_cls.resolve_prediction_outcomes = _resolve_prediction_outcomes_directional
    journal_cls.resolve_outcomes_from_candles = _resolve_outcomes_from_candles_directional
    _PATCH_INSTALLED = True


__all__: Iterable[str] = (
    "DEFAULT_TRAINING_COSTS_BPS",
    "install_directional_outcomes_patch",
)
