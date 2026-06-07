"""Tests for MarketScreener (updated for multi-tier refactor)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.features.screener import MarketScreener


def _make_ticker(
    symbol: str,
    volume: float,
    price: float = 100.0,
    bid: float | None = None,
    ask: float | None = None,
) -> dict:
    b = bid if bid is not None else price * 0.9999
    a = ask if ask is not None else price * 1.0001
    return {
        "symbol": symbol,
        "turnover24h": str(volume),
        "lastPrice": str(price),
        "bid1Price": str(b),
        "ask1Price": str(a),
        "bid1Size": "10000",
        "ask1Size": "10000",
        "price24hPcnt": "0.01",
        "curPreListingPhase": "",
    }


def _make_rest(tickers: list[dict]) -> MagicMock:
    rest = MagicMock()
    rest.get_tickers = AsyncMock(return_value={"result": {"list": tickers}})
    return rest


class TestMarketScreener:
    @pytest.mark.asyncio
    async def test_returns_top_by_volume(self):
        tickers = [
            _make_ticker("BTCUSDT", 500_000_000),
            _make_ticker("ETHUSDT", 300_000_000),
            _make_ticker("SOLUSDT", 100_000_000),
            _make_ticker("XRPUSDT", 25_000_000),
            _make_ticker("LOWUSDT", 1_000_000),  # below min volume
        ]
        screener = MarketScreener(
            rest_client=_make_rest(tickers),
            feature_max_symbols=3,
            min_volume_usd=20_000_000,
            max_spread_bps=100.0,
            min_top_book_depth_usd=0.0,
        )
        await screener._refresh()
        symbols = screener.feature_universe
        assert len(symbols) <= 3
        assert "LOWUSDT" not in symbols

    @pytest.mark.asyncio
    async def test_filters_below_min_volume(self):
        tickers = [_make_ticker("BTCUSDT", 1_000_000)]  # below threshold
        screener = MarketScreener(
            rest_client=_make_rest(tickers),
            min_volume_usd=20_000_000,
        )
        await screener._refresh()
        # No symbols should be added (fallback stays)
        assert "BTCUSDT" not in screener.feature_universe

    @pytest.mark.asyncio
    async def test_filters_non_usdt(self):
        tickers = [
            _make_ticker("BTCUSDT", 500_000_000),
            {
                "symbol": "BTCETH",
                "turnover24h": "500000000",
                "lastPrice": "1",
                "bid1Price": "0.9999",
                "ask1Price": "1.0001",
                "bid1Size": "10000",
                "ask1Size": "10000",
                "price24hPcnt": "0.01",
                "curPreListingPhase": "",
            },
        ]
        screener = MarketScreener(
            rest_client=_make_rest(tickers),
            min_volume_usd=1,
            max_spread_bps=100.0,
            min_top_book_depth_usd=0.0,
        )
        await screener._refresh()
        symbols = screener.feature_universe
        assert "BTCETH" not in symbols
        assert "BTCUSDT" in symbols

    @pytest.mark.asyncio
    async def test_filters_stablecoins(self):
        tickers = [
            _make_ticker("BTCUSDT", 500_000_000),
            _make_ticker("USDCUSDT", 500_000_000),
        ]
        screener = MarketScreener(
            rest_client=_make_rest(tickers),
            min_volume_usd=1,
            max_spread_bps=100.0,
            min_top_book_depth_usd=0.0,
        )
        await screener._refresh()
        assert "USDCUSDT" not in screener.feature_universe

    @pytest.mark.asyncio
    async def test_filters_zero_price(self):
        tickers = [_make_ticker("DEADUSDT", 500_000_000, price=0.0)]
        screener = MarketScreener(
            rest_client=_make_rest(tickers),
            min_volume_usd=1,
            max_spread_bps=100.0,
            min_top_book_depth_usd=0.0,
        )
        await screener._refresh()
        assert "DEADUSDT" not in screener.feature_universe

    @pytest.mark.asyncio
    async def test_filters_above_max_price_for_small_accounts(self):
        tickers = [
            _make_ticker("BTCUSDT", 500_000_000, price=65000.0),
            _make_ticker("XRPUSDT", 300_000_000, price=0.5),
            _make_ticker("ADAUSDT", 250_000_000, price=0.4),
        ]
        screener = MarketScreener(
            rest_client=_make_rest(tickers),
            min_volume_usd=1,
            max_spread_bps=100.0,
            min_top_book_depth_usd=0.0,
            max_price_usd=25.0,
        )
        await screener._refresh()

        assert "BTCUSDT" not in screener.feature_universe
        assert "XRPUSDT" in screener.feature_universe
        assert "ADAUSDT" in screener.feature_universe
        assert screener.metrics.rejection_reasons["above_max_price"] == 1

    @pytest.mark.asyncio
    async def test_max_symbols_respected(self):
        tickers = [_make_ticker(f"SYM{i}USDT", float(100 - i) * 1_000_000) for i in range(20)]
        screener = MarketScreener(
            rest_client=_make_rest(tickers),
            feature_max_symbols=5,
            min_volume_usd=1,
            max_spread_bps=100.0,
            min_top_book_depth_usd=0.0,
        )
        await screener._refresh()
        assert len(screener.feature_universe) <= 5

    @pytest.mark.asyncio
    async def test_manual_symbols_are_prioritised_when_eligible(self):
        tickers = [_make_ticker(f"SYM{i}USDT", float(100 - i) * 1_000_000) for i in range(10)]
        screener = MarketScreener(
            rest_client=_make_rest(tickers),
            feature_max_symbols=3,
            execution_candidates=2,
            min_volume_usd=1,
            max_spread_bps=100.0,
            min_top_book_depth_usd=0.0,
        )
        screener.set_manual_symbols(["SYM8USDT"])
        await screener._refresh()

        assert screener.manual_symbols == ["SYM8USDT"]
        assert screener.feature_universe[0] == "SYM8USDT"
        assert screener.execution_candidates[0] == "SYM8USDT"

    @pytest.mark.asyncio
    async def test_fallback_on_api_error(self):
        rest = MagicMock()
        rest.get_tickers = AsyncMock(side_effect=Exception("network error"))
        screener = MarketScreener(rest_client=rest, min_volume_usd=1)
        await screener._refresh()
        assert len(screener.active_symbols) > 0

    @pytest.mark.asyncio
    async def test_wait_ready_after_run(self):
        tickers = [_make_ticker("BTCUSDT", 500_000_000)]
        screener = MarketScreener(
            rest_client=_make_rest(tickers),
            min_volume_usd=1,
            max_spread_bps=100.0,
            min_top_book_depth_usd=0.0,
            interval_s=9999,
        )
        task = asyncio.create_task(screener.run())
        await screener.wait_ready()
        screener.stop()
        await asyncio.gather(task, return_exceptions=True)
        assert "BTCUSDT" in screener.active_symbols
