"""Tests for reconciliation fix — linear category must pass settleCoin=USDT."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trader.exchange.bybit_adapter import BybitAdapter, _default_settle_coin_for_category


class TestDefaultSettleCoin:
    def test_linear_returns_usdt(self):
        assert _default_settle_coin_for_category("linear") == "USDT"

    def test_spot_returns_none(self):
        assert _default_settle_coin_for_category("spot") is None

    def test_inverse_returns_none(self):
        assert _default_settle_coin_for_category("inverse") is None

    def test_unknown_returns_none(self):
        assert _default_settle_coin_for_category("option") is None


class TestReconcileSettleCoin:
    """Verify reconcile() passes settleCoin for linear, fixing error 10001."""

    def _make_adapter(self, category: str = "linear") -> BybitAdapter:
        with (
            patch("trader.exchange.bybit_adapter.BybitRestClient"),
            patch("trader.exchange.bybit_adapter.EndpointSelector"),
            patch("trader.exchange.bybit_adapter.RateLimiter"),
        ):
            adapter = BybitAdapter.__new__(BybitAdapter)
            adapter._default_category = category
            adapter._rest = MagicMock()
            adapter._idempotency = MagicMock()
            adapter._idempotency.pending_count = MagicMock(return_value=0)
            adapter._idempotency.all_states = MagicMock(return_value={})
            return adapter

    @pytest.mark.asyncio
    async def test_linear_reconcile_passes_settle_coin_usdt(self):
        """reconcile() for linear category must pass settle_coin='USDT'."""
        adapter = self._make_adapter("linear")
        adapter._rest.get_open_orders = AsyncMock(return_value={"retCode": 0, "result": {"list": []}})

        await adapter.reconcile()

        adapter._rest.get_open_orders.assert_awaited_once()
        call_kwargs = adapter._rest.get_open_orders.call_args
        assert call_kwargs.kwargs.get("settle_coin") == "USDT"

    @pytest.mark.asyncio
    async def test_spot_reconcile_passes_no_settle_coin(self):
        """reconcile() for spot category must NOT pass a settleCoin."""
        adapter = self._make_adapter("spot")
        adapter._rest.get_open_orders = AsyncMock(return_value={"retCode": 0, "result": {"list": []}})

        await adapter.reconcile()

        adapter._rest.get_open_orders.assert_awaited_once()
        call_kwargs = adapter._rest.get_open_orders.call_args
        assert call_kwargs.kwargs.get("settle_coin") is None

    @pytest.mark.asyncio
    async def test_reconcile_error_returns_failure_result(self):
        """API error during reconcile returns ReconciliationResult with success=False."""
        adapter = self._make_adapter("linear")
        adapter._rest.get_open_orders = AsyncMock(side_effect=RuntimeError("network error"))

        result = await adapter.reconcile()

        assert result.success is False
        assert "network error" in result.summary

    @pytest.mark.asyncio
    async def test_linear_get_open_orders_adapter_passes_settle_coin(self):
        """BybitAdapter.get_open_orders() forwards settle_coin to the REST client."""
        adapter = self._make_adapter("linear")
        adapter._rest.get_open_orders = AsyncMock(return_value={"retCode": 0, "result": {"list": []}})

        await adapter.get_open_orders(category="linear", settle_coin="USDT")

        adapter._rest.get_open_orders.assert_awaited_once_with(category="linear", symbol=None, settle_coin="USDT")
