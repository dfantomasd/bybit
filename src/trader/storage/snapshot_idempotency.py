"""Idempotency helpers for ML feature snapshot persistence."""

from __future__ import annotations

from trader.storage.directional_trade_journal import _BaseTradeJournal

_ORIGINAL_ENSURE_SCHEMA = _BaseTradeJournal._ensure_schema


async def _ensure_schema_with_snapshot_idempotency(self: _BaseTradeJournal) -> None:
    """Invalidate extra eligible duplicates before installing the unique index."""

    await _ORIGINAL_ENSURE_SCHEMA(self)
    assert self._pool is not None
    async with self._pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                WITH ranked AS (
                    SELECT
                        snapshot_id,
                        row_number() OVER (
                            PARTITION BY symbol, interval, candle_open_time, feature_schema_hash
                            ORDER BY created_at ASC, snapshot_id ASC
                        ) AS duplicate_rank
                    FROM feature_snapshots
                    WHERE training_eligible = true
                )
                UPDATE feature_snapshots fs
                SET training_eligible = false,
                    invalid_reason = 'duplicate_snapshot_same_candle',
                    invalidated_at = now()
                FROM ranked
                WHERE fs.snapshot_id = ranked.snapshot_id
                  AND ranked.duplicate_rank > 1;

                CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_snapshots_unique_eligible_candle_schema
                    ON feature_snapshots (symbol, interval, candle_open_time, feature_schema_hash)
                    WHERE training_eligible = true;
                """
            )


def install_snapshot_idempotency() -> None:
    """Patch the base schema migration used by the directional journal."""

    _BaseTradeJournal._ensure_schema = _ensure_schema_with_snapshot_idempotency
