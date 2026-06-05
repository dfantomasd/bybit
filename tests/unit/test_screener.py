"""Tests for MarketScreener."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.features.screener import MarketScreener


def _make_ticker(symbol: str, volume: float, price: float = 100.0) -> dict:
    return {
        "symbol": symbol,
        "turnover24h": str(volume),
        "lastPrice": str(price),
    }


def _make_rest(tickers: list[dict]):
    rest = MagicMock()
    rest.get_tickers = AsyncMock(
        return_value={"result": {"list": tickers}}
    )
    return rest


class TestMarketScreener:
    @pytest.mark.asyncio
    async def test_returns_top_by_volume(self):
        tickers = [
            _make_ticker("BTCUSDT", 500_000_000),
            _make_ticker("ETHUSDT", 300_000_000),
            _make_ticker("SOLUSDT", 100_000_000),
            _make_ticker("XRPUSDT",  25_000_000),
            _make_ticker("LOWUSDT",   1_000_000),  # below min volume
        ]
        screener = MarketScreener(
            rest_client=_make_rest(tickers),
            max_symbols=3,
            min_volume_usd=20_000_000,
        )
        symbols = await screener._screen()
        assert symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    @pytest.mark.asyncio
    async def test_filters_below_min_volume(self):
        tickers = [_make_ticker("BTCUSDT", 1_000_000)]  # below threshold
        screener = MarketScreener(
            rest_client=_make_rest(tickers),
            min_volume_usd=20_000_000,
        )
        symbols = await screener._screen()
        assert symbols == []

    @pytest.mark.asyncio
    async def test_filters_non_usdt(self):
        tickers = [
            _make_ticker("BTCUSDT", 500_000_000),
            _make_ticker("BTCETH",  500_000_000),   # non-USDT pair
        ]
        screener = MarketScreener(
            rest_client=_make_rest(tickers),
            min_volume_usd=1,
        )
        symbols = await screener._screen()
        assert "BTCETH" not in symbols
        assert "BTCUSDT" in symbols

    @pytest.mark.asyncio
    async def test_filters_stablecoins(self):
        tickers = [
            _make_ticker("BTCUSDT", 500_000_000),
            _make_ticker("USDCUSDT", 500_000_000),  # stablecoin base
        ]
        screener = MarketScreener(
            rest_client=_make_rest(tickers),
            min_volume_usd=1,
        )
        symbols = await screener._screen()
        assert "USDCUSDT" not in symbols

    @pytest.mark.asyncio
    async def test_filters_zero_price(self):
        tickers = [_make_ticker("DEADUSDT", 500_000_000, price=0.0)]
        screener = MarketScreener(
            rest_client=_make_rest(tickers),
            min_volume_usd=1,
        )
        symbols = await screener._screen()
        assert symbols == []

    @pytest.mark.asyncio
    async def test_max_symbols_respected(self):
        tickers = [_make_ticker(f"SYM{i}USDT", float(100 - i) * 1_000_000) for i in range(20)]
        screener = MarketScreener(
            rest_client=_make_rest(tickers),
            max_symbols=5,
            min_volume_usd=1,
        )
        symbols = await screener._screen()
        assert len(symbols) == 5

    @pytest.mark.asyncio
    async def test_fallback_on_api_error(self):
        rest = MagicMock()
        rest.get_tickers = AsyncMock(side_effect=Exception("network error"))
        screener = MarketScreener(rest_client=rest, min_volume_usd=1)
        # _refresh() should not raise
        await screener._refresh()
        # Active symbols remain at fallback
        assert len(screener.active_symbols) > 0

    @pytest.mark.asyncio
    async def test_wait_ready_after_run(self):
        tickers = [_make_ticker("BTCUSDT", 500_000_000)]
        screener = MarketScreener(
            rest_client=_make_rest(tickers),
            min_volume_usd=1,
            interval_s=9999,  # won't re-run
        )
        task = asyncio.create_task(screener.run())
        await screener.wait_ready()
        screener.stop()
        await asyncio.gather(task, return_exceptions=True)
        assert "BTCUSDT" in screener.active_symbols
