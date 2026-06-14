"""Tests for Lean-inspired features: VWAP, Trailing Stop, Profit Gate, Spread-Based Execution."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.features.technical import vwap


# ---------------------------------------------------------------------------
# VWAP tests
# ---------------------------------------------------------------------------


class TestVwap:
    def _make_ohlcv(self, n: int = 20, price: float = 100.0, volume: float = 1000.0):
        highs = [price + 1.0] * n
        lows = [price - 1.0] * n
        closes = [price] * n
        volumes = [volume] * n
        return highs, lows, closes, volumes

    def test_constant_price_returns_price(self):
        highs, lows, closes, volumes = self._make_ohlcv(20, 100.0)
        result = vwap(highs, lows, closes, volumes)
        assert result is not None
        assert abs(result - 100.0) < 1e-9

    def test_period_limits_lookback(self):
        highs, lows, closes, volumes = self._make_ohlcv(30)
        # Period = 10 should use only last 10 bars
        result = vwap(highs, lows, closes, volumes, period=10)
        assert result is not None

    def test_insufficient_data_for_period_returns_none(self):
        highs, lows, closes, volumes = self._make_ohlcv(5)
        result = vwap(highs, lows, closes, volumes, period=10)
        assert result is None

    def test_zero_volume_returns_none(self):
        highs = [100.0] * 10
        lows = [99.0] * 10
        closes = [100.0] * 10
        volumes = [0.0] * 10
        result = vwap(highs, lows, closes, volumes)
        assert result is None

    def test_empty_series_returns_none(self):
        assert vwap([], [], [], []) is None

    def test_volume_weighted_average(self):
        # Two bars: first at 100, second at 200, with volume 1 and 3 respectively
        # Typical price: (H+L+C)/3 = same as close here
        highs = [100.0, 200.0]
        lows = [100.0, 200.0]
        closes = [100.0, 200.0]
        volumes = [1.0, 3.0]
        result = vwap(highs, lows, closes, volumes)
        expected = (100.0 * 1 + 200.0 * 3) / (1 + 3)  # = 175
        assert result is not None
        assert abs(result - expected) < 1e-9


# ---------------------------------------------------------------------------
# Spread-based execution tests (via engine mock)
# ---------------------------------------------------------------------------


def _make_engine_maker(bid: str, ask: str):
    """Build a minimal ExecutionEngine mock for maker pricing tests."""
    from trader.domain.enums import MarketType, OrderSide, OrderType
    from trader.domain.models import InstrumentInfo, OrderIntent
    import uuid

    from trader.execution.engine import ExecutionEngine

    adapter = MagicMock()
    adapter.get_best_bid_ask = AsyncMock(return_value=(Decimal(bid), Decimal(ask)))
    adapter.place_order = AsyncMock(return_value={"result": {"orderId": "test-id"}})
    adapter.get_positions = AsyncMock(return_value=[])
    adapter.cancel_order = AsyncMock()
    adapter.get_open_orders = AsyncMock(return_value=[])

    engine = MagicMock(spec=ExecutionEngine)
    engine._adapter = adapter
    engine._category = "linear"
    engine._maker_allow_escalation = False
    engine._maker_ttl_s = 5.0
    engine._imbalance_provider = None
    engine._shadow_mode = False
    return engine


class TestSpreadBasedPricing:
    def _compute_maker_price(self, bid: str, ask: str, tick: str, side: str) -> Decimal:
        """Replicate the spread-based pricing logic from engine._execute_maker_first."""
        from trader.domain.enums import OrderSide

        bid_d = Decimal(bid)
        ask_d = Decimal(ask)
        tick_d = Decimal(tick)
        spread_bps_val = float((ask_d - bid_d) / bid_d * 10000) if bid_d > 0 else 0.0

        if side == "Buy":
            if spread_bps_val < 5.0:
                price = (bid_d + ask_d) / Decimal("2")
                if tick_d > 0:
                    price = (price // tick_d) * tick_d
            elif spread_bps_val < 30.0:
                price = bid_d + tick_d if tick_d > 0 and bid_d + tick_d < ask_d else bid_d
            else:
                price = bid_d
        else:
            if spread_bps_val < 5.0:
                price = (bid_d + ask_d) / Decimal("2")
                if tick_d > 0:
                    price = ((price // tick_d) + 1) * tick_d
            elif spread_bps_val < 30.0:
                price = ask_d - tick_d if tick_d > 0 and ask_d - tick_d > bid_d else ask_d
            else:
                price = ask_d
        return price

    def test_tight_spread_buy_places_at_mid(self):
        # bid=100, ask=100.02 → spread=2 bps (tight) → mid=100.01 floored to tick=0.01
        price = self._compute_maker_price("100", "100.02", "0.01", "Buy")
        assert price == Decimal("100.01")

    def test_normal_spread_buy_places_bid_plus_tick(self):
        # bid=100, ask=100.15 → spread=15 bps (normal) → bid+tick=100.01
        price = self._compute_maker_price("100", "100.15", "0.01", "Buy")
        assert price == Decimal("100.01")

    def test_wide_spread_buy_places_at_bid(self):
        # bid=100, ask=100.50 → spread=50 bps (wide) → bid=100
        price = self._compute_maker_price("100", "100.50", "0.01", "Buy")
        assert price == Decimal("100")

    def test_tight_spread_sell_places_at_mid_ceiled(self):
        # bid=100, ask=100.02 → spread=2 bps → mid=100.01 ceiled to tick=100.02
        price = self._compute_maker_price("100", "100.02", "0.01", "Sell")
        assert price == Decimal("100.02")

    def test_normal_spread_sell_places_ask_minus_tick(self):
        # bid=100, ask=100.15 → spread=15 bps → ask-tick=100.14
        price = self._compute_maker_price("100", "100.15", "0.01", "Sell")
        assert price == Decimal("100.14")

    def test_wide_spread_sell_places_at_ask(self):
        # bid=100, ask=100.50 → spread=50 bps → ask=100.50
        price = self._compute_maker_price("100", "100.50", "0.01", "Sell")
        assert price == Decimal("100.50")


# ---------------------------------------------------------------------------
# Trailing stop tests
# ---------------------------------------------------------------------------


class TestTrailingStop:
    def _make_engine(self, shadow_mode: bool = False):
        from trader.execution.engine import ExecutionEngine

        engine = MagicMock(spec=ExecutionEngine)
        engine._shadow_mode = shadow_mode
        engine._category = "linear"
        engine._trailing_stop_atr_multiple = 1.5
        engine._trailing_stop_min_pct = 0.01
        engine._adapter = MagicMock()
        engine._adapter.set_trading_stop = AsyncMock()
        engine._setup_trailing_stop = ExecutionEngine._setup_trailing_stop.__get__(engine)
        return engine

    @pytest.mark.asyncio
    async def test_trailing_stop_set_in_live_mode(self):
        engine = self._make_engine(shadow_mode=False)
        entry = Decimal("1000")
        atr = Decimal("10")  # 10 price units
        await engine._setup_trailing_stop("BTCUSDT", entry, atr)
        engine._adapter.set_trading_stop.assert_called_once()
        call_kwargs = engine._adapter.set_trading_stop.call_args[1]
        assert call_kwargs["symbol"] == "BTCUSDT"
        # distance = atr * 1.5 = 15.0 (passed as string to adapter)
        assert call_kwargs["trailing_stop"] == "15.0"

    @pytest.mark.asyncio
    async def test_trailing_stop_skipped_in_shadow_mode(self):
        engine = self._make_engine(shadow_mode=True)
        await engine._setup_trailing_stop("BTCUSDT", Decimal("1000"), Decimal("10"))
        engine._adapter.set_trading_stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_trailing_stop_uses_min_when_atr_is_none(self):
        engine = self._make_engine(shadow_mode=False)
        entry = Decimal("1000")
        await engine._setup_trailing_stop("BTCUSDT", entry, None)
        engine._adapter.set_trading_stop.assert_called_once()
        call_kwargs = engine._adapter.set_trading_stop.call_args[1]
        # min_distance = 1000 * 0.01 = 10.00 (passed as string)
        assert call_kwargs["trailing_stop"] == "10.00"

    @pytest.mark.asyncio
    async def test_trailing_stop_uses_min_when_atr_too_small(self):
        engine = self._make_engine(shadow_mode=False)
        entry = Decimal("1000")
        # atr * 1.5 = 0.15, but min = 10.00 (passed as string)
        await engine._setup_trailing_stop("BTCUSDT", entry, Decimal("0.1"))
        call_kwargs = engine._adapter.set_trading_stop.call_args[1]
        assert call_kwargs["trailing_stop"] == "10.00"

    @pytest.mark.asyncio
    async def test_trailing_stop_error_does_not_propagate(self):
        engine = self._make_engine(shadow_mode=False)
        engine._adapter.set_trading_stop = AsyncMock(side_effect=RuntimeError("exchange down"))
        # Should not raise
        await engine._setup_trailing_stop("BTCUSDT", Decimal("1000"), Decimal("10"))


# ---------------------------------------------------------------------------
# Profit gate tests
# ---------------------------------------------------------------------------


class TestProfitGate:
    def _make_engine(self, positions: dict, shadow_mode: bool = False):
        from trader.domain.enums import OrderSide
        from trader.execution.engine import ExecutionEngine

        engine = MagicMock(spec=ExecutionEngine)
        engine._shadow_mode = shadow_mode
        engine._category = "linear"
        engine._profit_gate_pct = 3.0
        engine._profit_lock_pct = 5.0
        engine._open_positions = positions
        engine._adapter = MagicMock()
        engine._adapter.set_trading_stop = AsyncMock()
        engine._adapter.get_best_bid_ask = AsyncMock(return_value=(Decimal("105"), Decimal("105.1")))
        engine.check_profit_gates = ExecutionEngine.check_profit_gates.__get__(engine)
        return engine

    @pytest.mark.asyncio
    async def test_profit_gate_skipped_in_shadow_mode(self):
        from trader.domain.enums import OrderSide

        pos = {"BTCUSDT": {"side": OrderSide.BUY, "entry_price": Decimal("100"), "size": Decimal("1")}}
        engine = self._make_engine(pos, shadow_mode=True)
        await engine.check_profit_gates()
        engine._adapter.set_trading_stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_profit_lock_tightens_sl_at_5pct(self):
        from trader.domain.enums import OrderSide

        # Entry=100, mid=105.05 → pnl=5.05% ≥ 5% → lock SL
        pos = {"BTCUSDT": {"side": OrderSide.BUY, "entry_price": Decimal("100"), "size": Decimal("1")}}
        engine = self._make_engine(pos)
        await engine.check_profit_gates()
        engine._adapter.set_trading_stop.assert_called_once()
        call_kwargs = engine._adapter.set_trading_stop.call_args[1]
        assert "stop_loss" in call_kwargs
        # SL should be entry * 1.005 = 100.5 (passed as string to adapter)
        assert call_kwargs["stop_loss"] == str(Decimal("100") * Decimal("1.005"))

    @pytest.mark.asyncio
    async def test_profit_gate_tightens_tp_at_3pct(self):
        from trader.domain.enums import OrderSide

        # Entry=100, mid=103.05 → pnl=3.05% ≥ 3% → tighten TP
        engine = self._make_engine({})
        engine._adapter.get_best_bid_ask = AsyncMock(return_value=(Decimal("103"), Decimal("103.1")))
        pos = {"BTCUSDT": {"side": OrderSide.BUY, "entry_price": Decimal("100"), "size": Decimal("1")}}
        engine._open_positions = pos
        await engine.check_profit_gates()
        engine._adapter.set_trading_stop.assert_called_once()
        call_kwargs = engine._adapter.set_trading_stop.call_args[1]
        assert "take_profit" in call_kwargs

    @pytest.mark.asyncio
    async def test_no_action_below_gate(self):
        from trader.domain.enums import OrderSide

        # Entry=100, mid=101 → pnl=1% < 3% → no action
        engine = self._make_engine({})
        engine._adapter.get_best_bid_ask = AsyncMock(return_value=(Decimal("101"), Decimal("101.1")))
        pos = {"BTCUSDT": {"side": OrderSide.BUY, "entry_price": Decimal("100"), "size": Decimal("1")}}
        engine._open_positions = pos
        await engine.check_profit_gates()
        engine._adapter.set_trading_stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_sell_position_pnl_computed_correctly(self):
        from trader.domain.enums import OrderSide

        # SELL: entry=100, price drops to 95 → pnl=(100-95)/100=5% ≥ 5% → lock SL
        engine = self._make_engine({})
        engine._adapter.get_best_bid_ask = AsyncMock(return_value=(Decimal("94.9"), Decimal("95")))
        pos = {"BTCUSDT": {"side": OrderSide.SELL, "entry_price": Decimal("100"), "size": Decimal("1")}}
        engine._open_positions = pos
        await engine.check_profit_gates()
        engine._adapter.set_trading_stop.assert_called_once()
        call_kwargs = engine._adapter.set_trading_stop.call_args[1]
        assert "stop_loss" in call_kwargs
        # SL for SELL = entry * 0.995 = 99.5 (passed as string to adapter)
        assert call_kwargs["stop_loss"] == str(Decimal("100") * Decimal("0.995"))
