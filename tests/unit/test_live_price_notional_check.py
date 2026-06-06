"""Tests for live-price min-notional post-check (step 4b in execution engine).

Covers:
- BUY orders use ask price; SELL orders use bid price
- Passes through when notional is already >= min_notional * 1.03
- Bumps qty by one step when notional is borderline
- Rejects when bump would exceed max_order_qty
- Rejects when bumped notional still below threshold
- Rejects when bumped notional exceeds available_balance
- Shadow mode uses proposal.entry_price; no exchange call is made
- Permissive on get_best_price network failure (returns original decision)
- Permissive when live_price is zero (returns original decision)
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.domain.enums import MarketType, OrderSide, RiskDecisionStatus
from trader.domain.models import InstrumentInfo, RiskDecision, TradeProposal
from trader.execution.engine import _MIN_NOTIONAL_BUFFER, ExecutionEngine

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _instrument(
    min_notional: Decimal = Decimal("5"),
    qty_step: Decimal = Decimal("1"),
    max_order_qty: Decimal = Decimal("1000"),
) -> InstrumentInfo:
    return InstrumentInfo(
        symbol="WLDUSDT",
        market_type=MarketType.LINEAR,
        base_coin="WLD",
        quote_coin="USDT",
        min_order_qty=Decimal("1"),
        max_order_qty=max_order_qty,
        qty_step=qty_step,
        tick_size=Decimal("0.001"),
        min_notional=min_notional,
    )


def _proposal(
    side: OrderSide = OrderSide.BUY,
    entry_price: Decimal = Decimal("2.00"),
) -> TradeProposal:
    return TradeProposal(
        strategy_id="test",
        symbol="WLDUSDT",
        market_type=MarketType.LINEAR,
        side=side,
        requested_qty=Decimal("10"),
        entry_price=entry_price,
        stop_loss=Decimal("1.50") if side == OrderSide.BUY else Decimal("2.50"),
        take_profit=Decimal("2.50") if side == OrderSide.BUY else Decimal("1.50"),
        confidence=0.75,
    )


def _decision(
    approved_qty: Decimal = Decimal("2"),
    rules: list[str] | None = None,
) -> RiskDecision:
    return RiskDecision(
        proposal_id=uuid.uuid4(),
        status=RiskDecisionStatus.APPROVED,
        approved_qty=approved_qty,
        triggered_rules=rules or [],
    )


def _make_engine(
    *, shadow_mode: bool = False, bid: Decimal = Decimal("1.90"), ask: Decimal = Decimal("2.10")
) -> ExecutionEngine:
    adapter = MagicMock()
    adapter.get_best_price = AsyncMock(return_value=(bid, ask))

    risk_manager = MagicMock()
    exposure = MagicMock()

    return ExecutionEngine(
        adapter=adapter,
        risk_manager=risk_manager,
        exposure_tracker=exposure,
        shadow_mode=shadow_mode,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLivePriceNotionalCheck:
    """Unit tests for ExecutionEngine._live_notional_check."""

    @pytest.mark.asyncio
    async def test_buy_uses_ask_price(self):
        """BUY order uses ask price (higher) for notional calculation."""
        # ask=2.10, approved_qty=3 → notional=6.30 >= 5*1.03=5.15  ✓
        engine = _make_engine(bid=Decimal("1.90"), ask=Decimal("2.10"))
        result = await engine._live_notional_check(
            proposal=_proposal(side=OrderSide.BUY),
            decision=_decision(approved_qty=Decimal("3")),
            instrument_info=_instrument(),
            available_balance=Decimal("1000"),
        )
        assert result is not None
        assert result.approved_qty == Decimal("3")
        engine._adapter.get_best_price.assert_awaited_once_with("WLDUSDT")

    @pytest.mark.asyncio
    async def test_sell_uses_bid_price(self):
        """SELL order uses bid price (lower) for notional calculation."""
        # bid=1.90, approved_qty=3 → notional=5.70 >= 5.15  ✓
        engine = _make_engine(bid=Decimal("1.90"), ask=Decimal("2.10"))
        result = await engine._live_notional_check(
            proposal=_proposal(side=OrderSide.SELL),
            decision=_decision(approved_qty=Decimal("3")),
            instrument_info=_instrument(),
            available_balance=Decimal("1000"),
        )
        assert result is not None
        assert result.approved_qty == Decimal("3")

    @pytest.mark.asyncio
    async def test_passes_when_notional_sufficient(self):
        """Returns unchanged decision when notional already exceeds buffer."""
        engine = _make_engine(ask=Decimal("2.00"))
        dec = _decision(approved_qty=Decimal("3"))  # 3*2.00=6.00 >= 5*1.03=5.15
        result = await engine._live_notional_check(
            proposal=_proposal(),
            decision=dec,
            instrument_info=_instrument(),
            available_balance=Decimal("1000"),
        )
        assert result is dec  # exact same object — no copy needed

    @pytest.mark.asyncio
    async def test_bumps_qty_when_borderline(self):
        """Bumps qty by one step when notional is just below threshold."""
        # ask=2.00, qty=2 → 4.00 < 5*1.03=5.15 → bump to qty=3 → 6.00 ✓
        engine = _make_engine(ask=Decimal("2.00"))
        result = await engine._live_notional_check(
            proposal=_proposal(),
            decision=_decision(approved_qty=Decimal("2")),
            instrument_info=_instrument(qty_step=Decimal("1")),
            available_balance=Decimal("1000"),
        )
        assert result is not None
        assert result.approved_qty == Decimal("3")
        assert "live_price_notional_bump" in result.triggered_rules

    @pytest.mark.asyncio
    async def test_preserves_existing_triggered_rules(self):
        """Bumped decision keeps prior triggered_rules and appends new one."""
        engine = _make_engine(ask=Decimal("2.00"))
        result = await engine._live_notional_check(
            proposal=_proposal(),
            decision=_decision(approved_qty=Decimal("2"), rules=["min_notional_floor_applied"]),
            instrument_info=_instrument(qty_step=Decimal("1")),
            available_balance=Decimal("1000"),
        )
        assert result is not None
        assert "min_notional_floor_applied" in result.triggered_rules
        assert "live_price_notional_bump" in result.triggered_rules

    @pytest.mark.asyncio
    async def test_rejects_when_bump_exceeds_max_order_qty(self):
        """Returns None when bumped qty would exceed max_order_qty."""
        # qty=2, qty_step=1 → bumped=3, but max_order_qty=2
        engine = _make_engine(ask=Decimal("2.00"))
        result = await engine._live_notional_check(
            proposal=_proposal(),
            decision=_decision(approved_qty=Decimal("2")),
            instrument_info=_instrument(qty_step=Decimal("1"), max_order_qty=Decimal("2")),
            available_balance=Decimal("1000"),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_rejects_when_bumped_notional_exceeds_balance(self):
        """Returns None when bumped notional would exceed available_balance."""
        # ask=2.00, qty=2 → 4.00 < 5.15, bumped_qty=3, bumped_notional=6.00 > balance=5.50
        engine = _make_engine(ask=Decimal("2.00"))
        result = await engine._live_notional_check(
            proposal=_proposal(),
            decision=_decision(approved_qty=Decimal("2")),
            instrument_info=_instrument(),
            available_balance=Decimal("5.50"),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_rejects_when_bump_still_below_threshold(self):
        """Returns None when even bumped qty doesn't meet min_notional buffer."""
        # ask=0.50, min_notional=5, buffer=5.15; qty=2→1.00, bumped=3→1.50 — both < 5.15
        engine = _make_engine(ask=Decimal("0.50"))
        result = await engine._live_notional_check(
            proposal=_proposal(),
            decision=_decision(approved_qty=Decimal("2")),
            instrument_info=_instrument(),
            available_balance=Decimal("1000"),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_permissive_on_network_error(self):
        """Returns original decision when get_best_price raises an exception."""
        engine = _make_engine()
        engine._adapter.get_best_price = AsyncMock(side_effect=ConnectionError("timeout"))
        dec = _decision(approved_qty=Decimal("2"))
        result = await engine._live_notional_check(
            proposal=_proposal(),
            decision=dec,
            instrument_info=_instrument(),
            available_balance=Decimal("1000"),
        )
        assert result is dec  # pass-through on error

    @pytest.mark.asyncio
    async def test_permissive_when_live_price_is_zero(self):
        """Returns original decision when ask/bid price is zero."""
        engine = _make_engine(bid=Decimal("0"), ask=Decimal("0"))
        dec = _decision(approved_qty=Decimal("2"))
        result = await engine._live_notional_check(
            proposal=_proposal(),
            decision=dec,
            instrument_info=_instrument(),
            available_balance=Decimal("1000"),
        )
        assert result is dec

    @pytest.mark.asyncio
    async def test_shadow_mode_uses_entry_price_not_exchange(self):
        """In shadow mode, entry_price is used; no exchange call is made."""
        engine = _make_engine(shadow_mode=True, ask=Decimal("9999"))
        # entry_price=2.00, qty=3 → 6.00 >= 5.15 ✓
        result = await engine._live_notional_check(
            proposal=_proposal(entry_price=Decimal("2.00")),
            decision=_decision(approved_qty=Decimal("3")),
            instrument_info=_instrument(),
            available_balance=Decimal("1000"),
        )
        assert result is not None
        engine._adapter.get_best_price.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_shadow_mode_bumps_when_entry_price_borderline(self):
        """Shadow mode bumps qty when entry_price gives borderline notional."""
        engine = _make_engine(shadow_mode=True)
        # entry_price=2.00, qty=2 → 4.00 < 5.15 → bump to 3
        result = await engine._live_notional_check(
            proposal=_proposal(entry_price=Decimal("2.00")),
            decision=_decision(approved_qty=Decimal("2")),
            instrument_info=_instrument(qty_step=Decimal("1")),
            available_balance=Decimal("1000"),
        )
        assert result is not None
        assert result.approved_qty == Decimal("3")

    def test_min_notional_buffer_constant(self):
        """_MIN_NOTIONAL_BUFFER is 1.03 (3 % safety margin)."""
        assert _MIN_NOTIONAL_BUFFER == Decimal("1.03")
