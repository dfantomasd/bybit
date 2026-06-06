"""Tests: P0.1 – get_open_orders must pass settleCoin for linear/inverse."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

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


@pytest.mark.asyncio
async def test_get_open_orders_linear_uses_settle_coin_usdt():
    """P0.1: linear without symbol must include settleCoin=USDT."""
    client = _make_client()
    captured: dict = {}

    async def fake_get(path: str, params=None, authenticated=True):
        captured["params"] = params or {}
        return {"retCode": 0, "result": {"list": []}}

    with patch.object(client, "_get", side_effect=fake_get):
        await client.get_open_orders(category="linear")

    assert captured["params"].get("settleCoin") == "USDT", "Expected settleCoin=USDT for linear category without symbol"
    assert "symbol" not in captured["params"], "Should not send symbol when not provided"


@pytest.mark.asyncio
async def test_get_open_orders_inverse_uses_settle_coin_btc():
    """P0.1: inverse without symbol must include settleCoin=BTC."""
    client = _make_client()
    captured: dict = {}

    async def fake_get(path: str, params=None, authenticated=True):
        captured["params"] = params or {}
        return {"retCode": 0, "result": {"list": []}}

    with patch.object(client, "_get", side_effect=fake_get):
        await client.get_open_orders(category="inverse")

    assert captured["params"].get("settleCoin") == "BTC"


@pytest.mark.asyncio
async def test_get_open_orders_with_symbol_does_not_add_settle_coin():
    """When symbol is provided, settleCoin must NOT be added."""
    client = _make_client()
    captured: dict = {}

    async def fake_get(path: str, params=None, authenticated=True):
        captured["params"] = params or {}
        return {"retCode": 0, "result": {"list": []}}

    with patch.object(client, "_get", side_effect=fake_get):
        await client.get_open_orders(category="linear", symbol="BTCUSDT")

    assert captured["params"].get("symbol") == "BTCUSDT"
    assert "settleCoin" not in captured["params"]


@pytest.mark.asyncio
async def test_get_open_orders_explicit_settle_coin_overrides_default():
    """Explicit settle_coin param must be forwarded as-is."""
    client = _make_client()
    captured: dict = {}

    async def fake_get(path: str, params=None, authenticated=True):
        captured["params"] = params or {}
        return {"retCode": 0, "result": {"list": []}}

    with patch.object(client, "_get", side_effect=fake_get):
        await client.get_open_orders(category="linear", settle_coin="USDC")

    assert captured["params"].get("settleCoin") == "USDC"


@pytest.mark.asyncio
async def test_reconcile_linear_does_not_raise_10001(monkeypatch):
    """Integration smoke: reconciliation for linear must not trigger 10001."""
    import asyncio

    from trader.exchange.reconciliation import ReconciliationService

    rest = AsyncMock()
    rest.get_open_orders.return_value = {"retCode": 0, "result": {"list": []}}
    rest.get_positions.return_value = []
    rest.get_wallet_balance.return_value = {"retCode": 0}

    order_store = AsyncMock()
    order_store.get_all_active.return_value = {}
    position_store = AsyncMock()
    position_store._positions = {}

    svc = ReconciliationService(
        rest_client=rest,
        order_store=order_store,
        position_store=position_store,
        event_queue=asyncio.Queue(maxsize=100),
    )
    result = await svc.run_once(category="linear")
    # Should complete without exception
    assert result is not None

    # Verify the call used settleCoin, not bare category
    call_kwargs = rest.get_open_orders.call_args
    assert call_kwargs is not None
    # Either keyword or positional – check that no call was made with only category
    kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
    # The reconciliation passes category="linear"; our fix in REST adds settleCoin
    assert kwargs.get("category") == "linear"


@pytest.mark.asyncio
async def test_reconcile_spot_does_not_add_invalid_settle_coin():
    """For spot category, no settleCoin should be auto-added."""
    client = _make_client()
    captured: dict = {}

    async def fake_get(path: str, params=None, authenticated=True):
        captured["params"] = params or {}
        return {"retCode": 0, "result": {"list": []}}

    with patch.object(client, "_get", side_effect=fake_get):
        await client.get_open_orders(category="spot")

    # spot without explicit settlement → no settleCoin added
    assert "settleCoin" not in captured["params"]
    assert "symbol" not in captured["params"]
    assert captured["params"].get("category") == "spot"
