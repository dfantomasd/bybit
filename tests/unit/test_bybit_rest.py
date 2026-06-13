"""Tests for the Bybit REST wrapper."""

from __future__ import annotations

from typing import Any

import pytest

from trader.exchange.bybit_rest import BybitRestClient


class _FakeRateLimiter:
    def __init__(self) -> None:
        self.acquired: list[tuple[str, str]] = []
        self.recorded: list[tuple[str, str]] = []

    async def acquire(self, endpoint: str, method: str = "GET") -> None:
        self.acquired.append((endpoint, method))

    def record_response(self, endpoint: str, headers: dict[str, Any], method: str = "GET") -> None:
        self.recorded.append((endpoint, method))


class _FakeSelector:
    rest_base = "https://api.bybit.com"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> BybitRestClient:
    monkeypatch.setattr(BybitRestClient, "_build_session", lambda self, *_args: _FakeSession())
    return BybitRestClient(
        api_key="key",
        api_secret="secret",
        endpoint_selector=_FakeSelector(),  # type: ignore[arg-type]
        rate_limiter=_FakeRateLimiter(),  # type: ignore[arg-type]
        use_testnet=True,
        max_workers=1,
    )


class _FakeSession:
    def place_order(self, **_kwargs: Any) -> dict[str, Any]:
        return {"retCode": 0, "result": {"orderId": "order-1"}}

    def cancel_order(self, **_kwargs: Any) -> dict[str, Any]:
        return {"retCode": 0, "result": {}}

    def set_leverage(self, **_kwargs: Any) -> dict[str, Any]:
        return {"retCode": 0, "result": {}}

    def get_server_time(self, **_kwargs: Any) -> dict[str, Any]:
        return {"retCode": 0, "result": {"timeSecond": "1800000000"}}


class TestBybitRestMethods:
    async def test_write_endpoints_use_post_rate_limit_key(self, client: BybitRestClient) -> None:
        await client.place_order(category="linear", symbol="BTCUSDT", side="Buy", orderType="Market", qty="0.001")
        await client.cancel_order(category="linear", symbol="BTCUSDT", order_link_id="TN-1")
        await client.set_leverage(category="linear", symbol="BTCUSDT", buy_leverage="1", sell_leverage="1")

        limiter = client._rate_limiter
        assert limiter.acquired == [
            ("/v5/order/create", "POST"),
            ("/v5/order/cancel", "POST"),
            ("/v5/position/set-leverage", "POST"),
        ]

    async def test_read_endpoints_keep_get_rate_limit_key(self, client: BybitRestClient) -> None:
        await client.get_server_time()
        assert client._rate_limiter.acquired == [("/v5/market/time", "GET")]
