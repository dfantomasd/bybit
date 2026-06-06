"""Tests: P0.6 – conservative market price uses ask for Buy, bid for Sell."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

import pytest

from trader.exchange.bybit_rest import BybitRestClient


def _make_client() -> BybitRestClient:
    from trader.exchange.endpoint_selector import EndpointSelector
    from trader.exchange.rate_limiter import RateLimiter

    es = EndpointSelector(use_testnet=False, region="GLOBAL")
    rl = RateLimiter()
    return BybitRestClient(
        api_key="testkey",
        api_secret="testsecret",
        endpoint_selector=es,
        rate_limiter=rl,
        use_testnet=False,
    )


def _ticker_response(ask: str, bid: str, last: str) -> dict:
    return {
        "retCode": 0,
        "result": {
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "ask1Price": ask,
                    "bid1Price": bid,
                    "lastPrice": last,
                }
            ]
        },
    }


@pytest.mark.asyncio
async def test_buy_guard_uses_ask():
    """P0.6: Buy side should use ask1Price."""
    client = _make_client()
    ticker_resp = _ticker_response(ask="50100", bid="50050", last="50075")

    with patch.object(client, "get_tickers", return_value=ticker_resp):
        price = await client.get_conservative_market_price("linear", "BTCUSDT", "Buy")

    assert price == Decimal("50100"), f"Expected ask 50100, got {price}"


@pytest.mark.asyncio
async def test_sell_guard_uses_bid():
    """P0.6: Sell side should use bid1Price."""
    client = _make_client()
    ticker_resp = _ticker_response(ask="50100", bid="50050", last="50075")

    with patch.object(client, "get_tickers", return_value=ticker_resp):
        price = await client.get_conservative_market_price("linear", "BTCUSDT", "Sell")

    assert price == Decimal("50050"), f"Expected bid 50050, got {price}"


@pytest.mark.asyncio
async def test_buy_fallback_to_last_price_when_ask_missing():
    """P0.6: fallback to lastPrice when ask1Price is absent."""
    client = _make_client()
    ticker_resp = {
        "retCode": 0,
        "result": {
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "ask1Price": "",
                    "bid1Price": "50050",
                    "lastPrice": "50075",
                }
            ]
        },
    }

    with patch.object(client, "get_tickers", return_value=ticker_resp):
        price = await client.get_conservative_market_price("linear", "BTCUSDT", "BUY")

    assert price == Decimal("50075"), "Should fall back to lastPrice"


@pytest.mark.asyncio
async def test_long_side_treated_as_buy():
    """LONG side variant is treated as Buy → ask."""
    client = _make_client()
    ticker_resp = _ticker_response(ask="50100", bid="50050", last="50075")

    with patch.object(client, "get_tickers", return_value=ticker_resp):
        price = await client.get_conservative_market_price("linear", "BTCUSDT", "LONG")

    assert price == Decimal("50100")


@pytest.mark.asyncio
async def test_zero_price_raises():
    """Zero conservative price must raise an error."""
    from trader.domain.errors import TradingSystemError

    client = _make_client()
    ticker_resp = {
        "retCode": 0,
        "result": {"list": [{"symbol": "X", "ask1Price": "0", "bid1Price": "0", "lastPrice": "0"}]},
    }

    with patch.object(client, "get_tickers", return_value=ticker_resp):
        with pytest.raises(TradingSystemError):
            await client.get_conservative_market_price("linear", "X", "Buy")
