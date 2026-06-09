from __future__ import annotations

import inspect


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
