from __future__ import annotations


def test_directional_journal_alias() -> None:
    from trader.storage.directional_trade_journal import DirectionalTradeJournal
    from trader.storage.trade_journal import TradeJournal

    assert TradeJournal is DirectionalTradeJournal
