"""Tests for LLMClient risk-multiplier scoring."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.llm.client import LLMClient


def _make_client(budget_cap_usd: float = 5.0, timeout_s: float = 5.0) -> LLMClient:
    return LLMClient(
        base_url="http://localhost:11434",
        model="llama3",
        budget_cap_usd=budget_cap_usd,
        timeout_s=timeout_s,
    )


def _mock_response(multiplier: float) -> MagicMock:
    resp = MagicMock()
    resp.status = 200
    resp.json = AsyncMock(return_value={"response": json.dumps({"risk_multiplier": multiplier})})
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


class TestLLMClientHappyPath:
    @pytest.mark.asyncio
    async def test_returns_multiplier_from_api(self):
        client = _make_client()
        mock_resp = _mock_response(0.75)

        session = MagicMock()
        session.post = MagicMock(return_value=mock_resp)
        session.closed = False

        client._session = session

        result = await client.get_risk_multiplier(
            symbol="BTCUSDT",
            side="Buy",
            regime="BULL_TREND",
            confidence=0.8,
            rationale="strong momentum",
        )

        assert result == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_clamps_multiplier_above_1(self):
        client = _make_client()
        mock_resp = _mock_response(1.5)

        session = MagicMock()
        session.post = MagicMock(return_value=mock_resp)
        session.closed = False
        client._session = session

        result = await client.get_risk_multiplier("BTCUSDT", "Buy", "RANGING", 0.6, "test")
        assert result == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_clamps_multiplier_below_0(self):
        client = _make_client()
        mock_resp = _mock_response(-0.5)

        session = MagicMock()
        session.post = MagicMock(return_value=mock_resp)
        session.closed = False
        client._session = session

        result = await client.get_risk_multiplier("BTCUSDT", "Sell", "RANGING", 0.6, "test")
        assert result == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_records_daily_spend(self):
        client = _make_client()
        mock_resp = _mock_response(0.9)

        session = MagicMock()
        session.post = MagicMock(return_value=mock_resp)
        session.closed = False
        client._session = session

        initial_spend = client._daily_spend_usd
        await client.get_risk_multiplier("ETHUSDT", "Buy", "BULL_TREND", 0.7, "")
        assert client._daily_spend_usd == pytest.approx(initial_spend + client._cost_per_call_usd)


class TestLLMClientBudgetCap:
    @pytest.mark.asyncio
    async def test_returns_1_when_budget_exhausted(self):
        client = _make_client(budget_cap_usd=0.0)

        result = await client.get_risk_multiplier("BTCUSDT", "Buy", "BULL_TREND", 0.9, "test")
        assert result == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_resets_spend_on_new_day(self):
        client = _make_client(budget_cap_usd=0.001)
        client._spend_date = "2020-01-01"
        client._daily_spend_usd = 999.0

        # Calling _check_budget for today resets the counter
        result = client._check_budget()
        assert result is True
        assert client._daily_spend_usd == pytest.approx(0.0)


class TestLLMClientFailOpen:
    @pytest.mark.asyncio
    async def test_returns_1_on_non_200_response(self):
        client = _make_client()
        resp = MagicMock()
        resp.status = 503
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=False)

        session = MagicMock()
        session.post = MagicMock(return_value=resp)
        session.closed = False
        client._session = session

        result = await client.get_risk_multiplier("BTCUSDT", "Buy", "RANGING", 0.5, "test")
        assert result == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_returns_1_on_network_exception(self):
        client = _make_client()

        session = MagicMock()
        session.post = MagicMock(side_effect=Exception("connection refused"))
        session.closed = False
        client._session = session

        result = await client.get_risk_multiplier("BTCUSDT", "Buy", "RANGING", 0.5, "test")
        assert result == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_returns_1_on_invalid_json(self):
        client = _make_client()
        resp = MagicMock()
        resp.status = 200
        resp.json = AsyncMock(return_value={"response": "not json {"})
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=False)

        session = MagicMock()
        session.post = MagicMock(return_value=resp)
        session.closed = False
        client._session = session

        result = await client.get_risk_multiplier("BTCUSDT", "Buy", "RANGING", 0.5, "test")
        assert result == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_close_is_safe_when_no_session(self):
        client = _make_client()
        await client.close()  # should not raise

    @pytest.mark.asyncio
    async def test_close_closes_open_session(self):
        client = _make_client()
        session = MagicMock()
        session.closed = False
        session.close = AsyncMock()
        client._session = session

        await client.close()
        session.close.assert_awaited_once()
