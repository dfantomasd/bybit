"""Idempotent runtime writes for ML feature snapshots."""

from __future__ import annotations

from trader.storage.directional_trade_journal import DirectionalTradeJournal


class SnapshotIdempotentTradeJournal(DirectionalTradeJournal):
    """Directional journal that reuses one eligible snapshot per candle schema."""
