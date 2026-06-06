"""Tests for screener dynamic WS subscription and open-position protection."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.features.screener import MarketScreener


def _make_ticker(symbol: str, volume: float, price: float = 1.0) -> dict:
    b = price * 0.9999
    a = price * 1.0001
    return {
        "symbol": symbol,
        "turnover24h": str(volume),
        "lastPrice": str(price),
        "bid1Price": str(b),
        "ask1Price": str(a),
        "bid1Size": "100000",
        "ask1Size": "100000",
        "price24hPcnt": "0.01",
        "curPreListingPhase": "",
    }


def _make_rest(tickers: list[dict]) -> MagicMock:
    rest = MagicMock()
    rest.get_tickers = AsyncMock(return_value={"result": {"list": tickers}})
    return rest


def _loose_screener(**kw) -> MarketScreener:
    """Build a screener with relaxed filters so all tickers pass."""
    defaults = {
        "min_volume_usd": 1,
        "max_spread_bps": 9999.0,
        "min_top_book_depth_usd": 0.0,
    }
    defaults.update(kw)
    return MarketScreener(**defaults)


class TestScreenerWSIntegration:
    @pytest.mark.asyncio
    async def test_on_symbols_added_callback_called_for_new_symbols(self):
        """Callback fires exactly for newly appearing symbols."""
        added_calls: list[list[str]] = []

        async def on_added(symbols: list[str]) -> None:
            added_calls.append(symbols)

        tickers = [_make_ticker("BTCUSDT", 500_000_000), _make_ticker("ETHUSDT", 300_000_000)]
        screener = _loose_screener(
            rest_client=_make_rest(tickers),
            feature_max_symbols=5,
            on_symbols_added=on_added,
        )
        # Override initial feature list so everything looks "new"
        screener._feature_universe = []

        await screener._refresh()

        assert len(added_calls) == 1
        assert set(added_calls[0]) == {"BTCUSDT", "ETHUSDT"}

    @pytest.mark.asyncio
    async def test_on_symbols_removed_callback_called_for_dropped_symbols(self):
        """Callback fires for symbols that leave the feature universe."""
        removed_calls: list[list[str]] = []

        async def on_removed(symbols: list[str]) -> None:
            removed_calls.append(symbols)

        tickers = [_make_ticker("BTCUSDT", 500_000_000)]
        screener = _loose_screener(
            rest_client=_make_rest(tickers),
            feature_max_symbols=5,
            on_symbols_removed=on_removed,
        )
        screener._feature_universe = ["SOLUSDT", "BTCUSDT"]

        await screener._refresh()

        assert len(removed_calls) == 1
        assert "SOLUSDT" in removed_calls[0]

    @pytest.mark.asyncio
    async def test_open_position_symbol_not_removed(self):
        """Symbol with an open position is kept even when screener drops it."""
        has_pos = {"SOLUSDT"}

        tickers = [_make_ticker("BTCUSDT", 500_000_000)]
        screener = _loose_screener(
            rest_client=_make_rest(tickers),
            feature_max_symbols=5,
            has_open_position=lambda s: s in has_pos,
        )
        screener._feature_universe = ["SOLUSDT", "BTCUSDT"]

        await screener._refresh()

        assert "SOLUSDT" in screener.active_symbols

    @pytest.mark.asyncio
    async def test_symbol_without_open_position_removed_normally(self):
        """Symbol without open position is removed when screener drops it."""
        tickers = [_make_ticker("BTCUSDT", 500_000_000)]
        screener = _loose_screener(
            rest_client=_make_rest(tickers),
            feature_max_symbols=5,
            has_open_position=lambda s: False,
        )
        screener._feature_universe = ["XRPUSDT", "BTCUSDT"]

        await screener._refresh()

        assert "XRPUSDT" not in screener.active_symbols
        assert "BTCUSDT" in screener.active_symbols

    @pytest.mark.asyncio
    async def test_added_callback_receives_only_truly_new_symbols(self):
        """Symbols already in active list are not passed to on_symbols_added."""
        added_calls: list[list[str]] = []

        async def on_added(symbols: list[str]) -> None:
            added_calls.append(symbols)

        tickers = [
            _make_ticker("BTCUSDT", 500_000_000),  # already in list
            _make_ticker("ETHUSDT", 300_000_000),  # new
        ]
        screener = _loose_screener(
            rest_client=_make_rest(tickers),
            feature_max_symbols=5,
            on_symbols_added=on_added,
        )
        screener._feature_universe = ["BTCUSDT"]

        await screener._refresh()

        assert len(added_calls) == 1
        assert added_calls[0] == ["ETHUSDT"]
