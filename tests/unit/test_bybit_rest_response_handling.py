from __future__ import annotations

import pytest

from trader.domain.errors import TradingSystemError
from trader.exchange.bybit_rest import BybitRestClient


class _Limiter:
    async def acquire(self, *_args: object, **_kwargs: object) -> None:
        return None

    def record_response(self, *_args: object, **_kwargs: object) -> None:
        return None

    def handle_rate_limit_error(self, *_args: object, **_kwargs: object) -> float:
        return 0.0


class _Response:
    def __init__(self, *, status: int, body: str, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self.headers = headers or {}
        self._body = body

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def text(self) -> str:
        return self._body


class _Session:
    def __init__(self, response: _Response) -> None:
        self._response = response
        self.closed = False

    def get(self, *_args: object, **_kwargs: object) -> _Response:
        return self._response

    def post(self, *_args: object, **_kwargs: object) -> _Response:
        return self._response


def _client(response: _Response) -> BybitRestClient:
    client = object.__new__(BybitRestClient)
    client._api_key = ""
    client._api_secret = ""
    client._base_url = "https://api.bybit.com"
    client._recv_window = 5000
    client._rate_limiter = _Limiter()
    client._session = _Session(response)
    return client


@pytest.mark.asyncio
async def test_get_non_json_http_error_is_classified() -> None:
    client = _client(
        _Response(
            status=403,
            body="<html><title>Forbidden</title></html>",
            headers={"Content-Type": "text/html"},
        )
    )

    with pytest.raises(TradingSystemError) as exc_info:
        await client._get("/v5/market/tickers", authenticated=False)

    assert exc_info.value.code == "HTTP_403"
    assert "non-JSON response" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_json_http_error_is_classified_before_retcode_zero_default() -> None:
    client = _client(
        _Response(
            status=403,
            body='{"retMsg":"Forbidden"}',
            headers={"Content-Type": "application/json"},
        )
    )

    with pytest.raises(TradingSystemError) as exc_info:
        await client._get("/v5/market/tickers", authenticated=False)

    assert exc_info.value.code == "HTTP_403"
    assert "Forbidden" in str(exc_info.value)


@pytest.mark.asyncio
async def test_post_non_dict_json_response_is_rejected() -> None:
    client = _client(
        _Response(
            status=200,
            body='["unexpected"]',
            headers={"Content-Type": "application/json"},
        )
    )

    with pytest.raises(TradingSystemError) as exc_info:
        await client._post("/v5/order/create", body={})

    assert exc_info.value.code == "INVALID_RESPONSE"
