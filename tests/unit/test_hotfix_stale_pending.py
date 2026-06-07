"""Tests for hotfix: stale pending entry resolution and related fixes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Tests 1-2: mark_entry_resolved fixes
# ---------------------------------------------------------------------------


def _make_engine(**kwargs):
    from trader.execution.engine import ExecutionEngine

    adapter = MagicMock()
    adapter.get_instrument_info = AsyncMock()
    adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("50000"))
    risk_manager = MagicMock()
    risk_manager._limits = MagicMock()
    risk_manager._limits.max_leverage = Decimal("10")
    exposure = MagicMock()
    exposure.update_position = AsyncMock()
    return ExecutionEngine(
        adapter=adapter,
        risk_manager=risk_manager,
        exposure_tracker=exposure,
        shadow_mode=False,
        **kwargs,
    )


def test_terminal_order_update_removes_exact_pending_id():
    """mark_entry_resolved removes the exact ID and decrements count."""
    engine = _make_engine()
    engine.mark_entry_submitted("order-123")
    assert "order-123" in engine._pending_entry_order_link_ids
    assert engine._pending_entry_count == 1
    engine.mark_entry_resolved("order-123")
    assert "order-123" not in engine._pending_entry_order_link_ids
    assert engine._pending_entry_count == 0


def test_unrelated_terminal_order_does_not_release_pending_slot():
    """mark_entry_resolved for a foreign ID must NOT decrement count."""
    engine = _make_engine()
    engine.mark_entry_submitted("order-mine")
    assert engine._pending_entry_count == 1
    # Terminal event for a different order (e.g. from another strategy or manual order)
    engine.mark_entry_resolved("order-unrelated")
    assert engine._pending_entry_count == 1  # unchanged
    assert "order-mine" in engine._pending_entry_order_link_ids


# ---------------------------------------------------------------------------
# Test 3: stale pending TTL
# ---------------------------------------------------------------------------


def test_stale_pending_expires_after_ttl():
    """get_pending_diagnostics reports stale_pending_count > 0 when age > TTL."""
    engine = _make_engine(stale_pending_ttl_seconds=600)
    # Manually inject a "old" timestamp
    engine.mark_entry_submitted("stale-order")
    # Backdate the timestamp
    engine._pending_entry_timestamps["stale-order"] = datetime.now(tz=UTC) - timedelta(seconds=700)
    diag = engine.get_pending_diagnostics()
    assert diag["stale_pending_count"] == 1
    assert diag["pending_entry_count"] == 1
    assert diag["oldest_pending_age_seconds"] > 600


# ---------------------------------------------------------------------------
# Test 4: startup reconcile removes resolved pending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_reconcile_removes_resolved_pending():
    """_reconcile_pending_at_startup returns IDs not found in open orders or history."""
    # We test the logic by mocking app internals
    # Create a minimal app-like object

    class FakeApp:
        _telegram_bot = None
        _trade_journal = None

        async def _reconcile_pending_at_startup(self, pending_ids):
            # Import from the real app module
            from trader.app import TradingApplication

            return await TradingApplication._reconcile_pending_at_startup(self, pending_ids)

    app = FakeApp()

    # Mock bybit adapter and settings
    app._settings = MagicMock()
    app._settings.DEFAULT_MARKET_CATEGORY = "linear"
    app._settings.STALE_PENDING_TTL_SECONDS = 600

    rest_mock = MagicMock()
    rest_mock.get_open_orders = AsyncMock(return_value={"result": {"list": []}})
    # Return history showing order was CANCELLED
    rest_mock.get_order_history = AsyncMock(return_value={"result": {"list": [{"orderStatus": "Cancelled"}]}})

    adapter_mock = MagicMock()
    adapter_mock._rest = rest_mock
    app._bybit_adapter = adapter_mock

    resolved = await app._reconcile_pending_at_startup(["order-stale-1"])
    assert "order-stale-1" in resolved


# ---------------------------------------------------------------------------
# Test 5: LIVE_ARMED gate
# ---------------------------------------------------------------------------


def test_live_requires_live_armed():
    """_active_execution_allowed returns False when LIVE_ARMED=False even with LIVE_MODE=True."""
    from trader.app import TradingApplication

    app = TradingApplication.__new__(TradingApplication)
    app._settings = MagicMock()
    app._settings.TRADING_MODE = MagicMock()
    app._settings.BYBIT_USE_TESTNET = False
    app._settings.LIVE_MODE = True
    app._settings.LIVE_ARMED = False  # NOT armed

    from trader.domain.enums import TradingMode

    app._settings.TRADING_MODE = TradingMode.LIVE

    result = app._active_execution_allowed()
    assert result is False


def test_live_requires_live_armed_true():
    """_active_execution_allowed returns True only when LIVE_ARMED=True."""
    from trader.app import TradingApplication
    from trader.domain.enums import TradingMode

    app = TradingApplication.__new__(TradingApplication)
    app._settings = MagicMock()
    app._settings.BYBIT_USE_TESTNET = False
    app._settings.LIVE_MODE = True
    app._settings.LIVE_ARMED = True
    app._settings.TRADING_MODE = TradingMode.LIVE

    result = app._active_execution_allowed()
    assert result is True


# ---------------------------------------------------------------------------
# Tests 6-7: preflight fixes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uta2_account_type_is_unified():
    """unifiedMarginStatus 3,4,5,6 should all be reported as unified."""
    from trader.exchange.preflight import BybitPreflightChecker

    checker = BybitPreflightChecker.__new__(BybitPreflightChecker)
    checker._rest = MagicMock()

    for status in (3, 4, 5, 6, "3", "4", "5", "6"):
        checker._rest.get_account_info = AsyncMock(return_value={"result": {"unifiedMarginStatus": status}})
        result = await checker._check_account_type()
        assert result.passed, f"Expected unified for status={status}"


@pytest.mark.asyncio
async def test_wallet_transfer_permission_is_not_withdraw_permission():
    """'Wallet' key in permissions without 'Withdraw' value should NOT trigger warning."""
    from trader.exchange.preflight import BybitPreflightChecker

    checker = BybitPreflightChecker.__new__(BybitPreflightChecker)
    checker._rest = MagicMock()
    # Wallet category with only Transfer — no Withdraw
    checker._rest.get_api_key_info = AsyncMock(
        return_value={
            "result": {
                "permissions": {
                    "Wallet": ["Transfer"],
                    "Trade": ["Order"],
                }
            }
        }
    )
    result = await checker._check_api_key_permissions()
    assert result.warning is None, f"Should not warn for Transfer-only Wallet: {result.warning}"


# ---------------------------------------------------------------------------
# Test 8-9: feature pipeline and candle seeding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feature_pipeline_uses_current_screener_symbols():
    """_start_feature_pipeline should use screener.active_symbols, not _SYMBOLS fallback."""
    from unittest.mock import patch

    from trader.app import _SYMBOLS, TradingApplication

    app = TradingApplication.__new__(TradingApplication)
    app._settings = MagicMock()
    app._background_tasks = []
    app._candle_store = MagicMock()  # Must be non-None to pass assert
    app._regime_classifier = None
    app._health_checker = None

    screener = MagicMock()
    screener.active_symbols = ["BTCUSDT", "ETHUSDT"]
    app._screener = screener

    captured_symbols = []

    class FakePipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, symbols, **kwargs):
            captured_symbols.extend(symbols)

            async def _noop():
                pass

            return _noop()

    def fake_create_task(coro, **kw):
        # Don't actually schedule
        coro.close()
        t = MagicMock()
        t.get_name = lambda: "feature-pipeline"
        return t

    with (
        patch("trader.features.pipeline.FeaturePipeline", FakePipeline),
        patch("trader.app.asyncio.create_task", fake_create_task),
    ):
        await app._start_feature_pipeline()

    # captured_symbols should be screener.active_symbols, NOT _SYMBOLS
    assert captured_symbols == ["BTCUSDT", "ETHUSDT"]
    assert captured_symbols != list(_SYMBOLS)


@pytest.mark.asyncio
async def test_seeded_symbol_not_seeded_twice():
    """_seed_candle_store skips symbols already in _seeded_symbols."""
    from trader.app import TradingApplication

    app = TradingApplication.__new__(TradingApplication)
    app._settings = MagicMock()
    app._settings.BYBIT_API_KEY = MagicMock()
    app._settings.BYBIT_API_KEY.get_secret_value = lambda: "key"
    app._candle_store = None

    call_count = {"n": 0}

    rest = MagicMock()

    async def fake_get_kline(**kwargs):
        call_count["n"] += 1
        return {"result": {"list": []}}

    rest.get_kline = fake_get_kline

    adapter = MagicMock()
    adapter._rest = rest
    app._bybit_adapter = adapter
    app._seeded_symbols = set()

    await app._seed_candle_store(symbols=["BTCUSDT"])
    await app._seed_candle_store(symbols=["BTCUSDT"])  # second call must be a no-op

    assert call_count["n"] == 1, f"Expected 1 REST call, got {call_count['n']}"
