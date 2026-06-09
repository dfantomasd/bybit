from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime, timedelta
from typing import Any


def test_directional_journal_alias() -> None:
    from trader.storage.directional_trade_journal import DirectionalTradeJournal
    from trader.storage.trade_journal import TradeJournal

    assert TradeJournal is DirectionalTradeJournal


def test_source_candle_feature_guard_alias() -> None:
    from trader.features.pipeline import FeaturePipeline
    from trader.features.source_candle_guard import SourceCandleFeaturePipeline

    assert FeaturePipeline is SourceCandleFeaturePipeline


def test_snapshot_source_guard_rejects_mismatch_before_write() -> None:
    from trader.storage.directional_trade_journal import DirectionalTradeJournal

    source = inspect.getsource(DirectionalTradeJournal.record_feature_snapshot)
    assert "feature_snapshot_source_mismatch" in source
    assert 'return ""' in source
    assert "_CURRENT_SOURCE_BINDING.set(None)" in source


def test_snapshot_source_guard_rejects_mismatch_at_runtime() -> None:
    from trader.storage.directional_trade_journal import _CURRENT_SOURCE_BINDING, DirectionalTradeJournal

    class FakeJournal(DirectionalTradeJournal):
        def __init__(self) -> None:
            self.called = False

        async def _fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
            del query, args
            self.called = True
            return [{"snapshot_id": "snapshot-1"}]

    async def exercise() -> tuple[str, bool, object]:
        source_time = datetime(2026, 6, 9, 12, 0, tzinfo=UTC)
        journal = FakeJournal()
        _CURRENT_SOURCE_BINDING.set(("BTCUSDT", "1", source_time))
        snapshot_id = await journal.record_feature_snapshot(
            symbol="BTCUSDT",
            interval="1",
            candle_open_time=source_time + timedelta(minutes=1),
            feature_schema_hash="schema",
            feature_names=["rsi"],
            feature_values=[0.5],
        )
        return snapshot_id, journal.called, _CURRENT_SOURCE_BINDING.get()

    snapshot_id, called, binding = asyncio.run(exercise())
    assert snapshot_id == ""
    assert called is False
    assert binding is None


def test_snapshot_source_guard_allows_exact_candle_at_runtime() -> None:
    from trader.storage.directional_trade_journal import _CURRENT_SOURCE_BINDING, DirectionalTradeJournal

    class FakeJournal(DirectionalTradeJournal):
        def __init__(self) -> None:
            self.called = False

        async def _fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
            del query, args
            self.called = True
            return [{"snapshot_id": "snapshot-1"}]

    async def exercise() -> tuple[str, bool, object]:
        source_time = datetime(2026, 6, 9, 12, 0, tzinfo=UTC)
        journal = FakeJournal()
        _CURRENT_SOURCE_BINDING.set(("BTCUSDT", "1", source_time))
        snapshot_id = await journal.record_feature_snapshot(
            symbol="BTCUSDT",
            interval="1",
            candle_open_time=source_time,
            feature_schema_hash="schema",
            feature_names=["rsi"],
            feature_values=[0.5],
        )
        return snapshot_id, journal.called, _CURRENT_SOURCE_BINDING.get()

    snapshot_id, called, binding = asyncio.run(exercise())
    assert snapshot_id == "snapshot-1"
    assert called is True
    assert binding is None
