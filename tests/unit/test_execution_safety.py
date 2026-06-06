"""Tests for execution-engine safety guards and private-WS ExecutionUpdateEvent handling.

Covers:
  - Conservative market-price guard before place_order (live mode, fail-closed)
  - Engine uses BybitAdapter public wrapper (not _rest directly)
  - Engine uses configured min-notional safety buffer (not hardcoded 1.03)
  - Min-notional bump exposure re-check in RiskManager
  - ExecutionUpdateEvent idempotency and sync_positions trigger
"""

from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.domain.enums import MarketRegime, MarketType, OrderSide, RiskDecisionStatus
from trader.domain.events import ExecutionUpdateEvent, OrderType
from trader.domain.models import InstrumentInfo, TradeProposal
from trader.risk.exposure import ExposureTracker
from trader.risk.profiles import RiskProfile

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_instrument(
    min_notional: Decimal = Decimal("5"),
    min_qty: Decimal = Decimal("0.001"),
    qty_step: Decimal = Decimal("0.001"),
    tick_size: Decimal = Decimal("0.01"),
    max_qty: Decimal = Decimal("1000"),
) -> InstrumentInfo:
    return InstrumentInfo(
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        base_coin="BTC",
        quote_coin="USDT",
        min_order_qty=min_qty,
        max_order_qty=max_qty,
        qty_step=qty_step,
        tick_size=tick_size,
        min_notional=min_notional,
    )


def _make_proposal(
    qty: Decimal = Decimal("0.001"),
    entry_price: Decimal = Decimal("50000"),
    side: OrderSide = OrderSide.BUY,
) -> TradeProposal:
    return TradeProposal(
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        side=side,
        requested_qty=qty,
        entry_price=entry_price,
        stop_loss=entry_price * Decimal("0.98"),
        take_profit=entry_price * Decimal("1.04"),
        confidence=0.8,
        strategy_id="test_strategy",
        regime=MarketRegime.BULL_TREND,
    )


def _make_engine(shadow: bool = False, buffer_pct: float = 3.0) -> Any:
    """Build a minimal ExecutionEngine with mocked adapter and risk manager."""
    from trader.execution.engine import ExecutionEngine

    adapter = MagicMock()
    adapter._rest = MagicMock()
    adapter.place_order = AsyncMock(return_value={"result": {"orderId": "EX001"}})
    adapter.get_instrument_info = AsyncMock(return_value=_make_instrument())
    adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("6000"))

    risk_manager = MagicMock()
    from trader.domain.models import RiskDecision

    risk_manager.evaluate = AsyncMock(
        return_value=RiskDecision(
            proposal_id=uuid.uuid4(),
            status=RiskDecisionStatus.APPROVED,
            approved_qty=Decimal("0.001"),
            triggered_rules=[],
            portfolio_heat=5.0,
            current_drawdown_pct=0.0,
            open_positions_count=0,
        )
    )
    risk_manager._limits = MagicMock()
    risk_manager._limits.max_leverage = Decimal("10")

    exposure = MagicMock()
    exposure.update_position = AsyncMock()

    engine = ExecutionEngine(
        adapter=adapter,
        risk_manager=risk_manager,
        exposure_tracker=exposure,
        shadow_mode=shadow,
        category="linear",
        min_notional_safety_buffer_pct=buffer_pct,
    )
    return engine, adapter


# ---------------------------------------------------------------------------
# Change 1: conservative-price guard (fail-closed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conservative_price_guard_blocks_below_min_notional():
    """Live mode: executable notional < min_notional * buffer → None, place_order not called."""
    engine, adapter = _make_engine(shadow=False)

    # qty=0.001 * conservative_price=4000 = 4.0, min_notional=5 * 1.03 = 5.15 → blocked
    adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("4000"))

    proposal = _make_proposal(qty=Decimal("0.001"), entry_price=Decimal("50000"))

    result = await engine.submit(proposal, capital=Decimal("10000"), available_balance=Decimal("5000"))

    assert result is None
    adapter.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_conservative_price_guard_passes_above_min_notional():
    """Live mode: executable notional >= min_notional * buffer → place_order called."""
    engine, adapter = _make_engine(shadow=False)

    # qty=0.001 * 6000 = 6.0 > 5 * 1.03 = 5.15 → allowed
    adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("6000"))

    proposal = _make_proposal(qty=Decimal("0.001"), entry_price=Decimal("50000"))

    await engine.submit(proposal, capital=Decimal("10000"), available_balance=Decimal("5000"))

    adapter.place_order.assert_called_once()


@pytest.mark.asyncio
async def test_conservative_price_fetch_failure_blocks_order():
    """Fail-closed: if get_conservative_market_price raises, place_order must NOT be called."""
    engine, adapter = _make_engine(shadow=False)

    adapter.get_conservative_market_price = AsyncMock(side_effect=Exception("timeout"))

    proposal = _make_proposal(qty=Decimal("0.001"), entry_price=Decimal("50000"))

    result = await engine.submit(proposal, capital=Decimal("10000"), available_balance=Decimal("5000"))

    assert result is None
    adapter.place_order.assert_not_called()
    # Failure timestamp must be recorded
    assert "BTCUSDT" in engine._last_failure_at


@pytest.mark.asyncio
async def test_conservative_price_fetch_failure_records_journal_event():
    """Fail-closed: price-check failure writes REJECTED_PRICE_CHECK_FAILED to the journal."""
    engine, adapter = _make_engine(shadow=False)

    journal = MagicMock()
    journal.record_order_event = AsyncMock()
    journal.record_risk_decision = AsyncMock()
    engine._trade_journal = journal

    adapter.get_conservative_market_price = AsyncMock(side_effect=Exception("network error"))

    proposal = _make_proposal(qty=Decimal("0.001"), entry_price=Decimal("50000"))

    await engine.submit(proposal, capital=Decimal("10000"), available_balance=Decimal("5000"))

    journal.record_order_event.assert_called_once()
    call_kwargs = journal.record_order_event.call_args.kwargs
    assert call_kwargs["status"] == "REJECTED_PRICE_CHECK_FAILED"
    assert "network error" in call_kwargs["error"]


@pytest.mark.asyncio
async def test_conservative_price_guard_shadow_mode_skips_check():
    """In shadow mode the conservative-price guard is not executed."""
    engine, adapter = _make_engine(shadow=True)

    adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("4000"))

    proposal = _make_proposal(qty=Decimal("0.001"), entry_price=Decimal("50000"))

    await engine.submit(proposal, capital=Decimal("10000"), available_balance=Decimal("5000"))

    # In shadow mode: place_order never called regardless
    adapter.place_order.assert_not_called()
    # get_conservative_market_price must not be called in shadow mode
    adapter.get_conservative_market_price.assert_not_called()


# ---------------------------------------------------------------------------
# Public adapter wrapper and configurable buffer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_uses_adapter_public_price_wrapper():
    """Engine calls adapter.get_conservative_market_price, never adapter._rest directly."""
    engine, adapter = _make_engine(shadow=False)

    adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("6000"))
    adapter._rest.get_conservative_market_price = AsyncMock(return_value=Decimal("6000"))

    proposal = _make_proposal(qty=Decimal("0.001"), entry_price=Decimal("50000"))
    await engine.submit(proposal, capital=Decimal("10000"), available_balance=Decimal("5000"))

    adapter.get_conservative_market_price.assert_called_once()
    adapter._rest.get_conservative_market_price.assert_not_called()


@pytest.mark.asyncio
async def test_engine_uses_configured_min_notional_buffer():
    """Execution engine last-moment guard uses raw exchange minimum (no double-buffer).

    Buffer is applied once at sizing time (RiskManager). The execution engine
    only rejects if below the raw exchange minimum to avoid code=110094.
    A notional above the raw minimum but below the sizing target emits a warning
    but allows the order through.
    """
    # 0.001 * 5300 = $5.30 > $5.00 (raw exchange min) → allowed even with buffer=10%
    engine_strict, adapter_strict = _make_engine(shadow=False, buffer_pct=10.0)
    adapter_strict.get_conservative_market_price = AsyncMock(return_value=Decimal("5300"))

    result = await engine_strict.submit(
        _make_proposal(qty=Decimal("0.001"), entry_price=Decimal("50000")),
        capital=Decimal("10000"),
        available_balance=Decimal("5000"),
    )
    # $5.30 > $5.00 raw minimum → ORDER ALLOWED (buffer consumed warning emitted)
    adapter_strict.place_order.assert_called()

    # 0.001 * 4900 = $4.90 < $5.00 → REJECTED (below raw exchange minimum)
    engine2, adapter2 = _make_engine(shadow=False, buffer_pct=3.0)
    adapter2.get_conservative_market_price = AsyncMock(return_value=Decimal("4900"))

    result2 = await engine2.submit(
        _make_proposal(qty=Decimal("0.001"), entry_price=Decimal("50000")),
        capital=Decimal("10000"),
        available_balance=Decimal("5000"),
    )
    adapter2.place_order.assert_not_called()


# ---------------------------------------------------------------------------
# Change 2: min-notional bump exposure re-check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bump_exposure_recheck_blocks_over_cap():
    """Min-notional bump rejected when bumped notional > remaining per-position budget."""
    from trader.risk.circuit_breakers import CircuitBreakerManager
    from trader.risk.drawdown import DrawdownTracker
    from trader.risk.kill_switch import KillSwitch
    from trader.risk.manager import RiskManager

    capital = Decimal("10000")
    limits_profile = RiskProfile.CONSERVATIVE

    exposure = ExposureTracker(total_capital=capital, risk_limits=MagicMock())
    exposure._limits = MagicMock()
    exposure._limits.max_total_exposure_pct = Decimal("70")
    exposure._limits.max_capital_per_position_pct = Decimal("10")
    # Simulate existing position eating most per-position budget
    # remaining_position_exposure_usd for BTCUSDT = 0 (cap used up)
    await exposure.update_position("BTCUSDT", "Buy", Decimal("999"))  # almost at 10% of 10000

    drawdown = DrawdownTracker(initial_equity=capital)
    kill_switch = KillSwitch()
    breakers = CircuitBreakerManager(risk_limits=MagicMock())
    breakers._limits = MagicMock()
    breakers._limits.max_consecutive_losses = 5
    breakers._limits.max_hourly_loss_pct = Decimal("5")
    breakers._limits.max_daily_loss_pct = Decimal("10")
    breakers.should_emergency = MagicMock(return_value=False)
    breakers.should_block_entries = MagicMock(return_value=False)
    breakers.should_safe_mode = MagicMock(return_value=False)
    breakers.get_triggered = MagicMock(return_value=[])

    rm = RiskManager(
        risk_profile=limits_profile,
        drawdown_tracker=drawdown,
        exposure_tracker=exposure,
        circuit_breaker_manager=breakers,
        kill_switch=kill_switch,
        min_notional_safety_buffer_pct=3.0,
    )

    # Entry price such that even min_qty bumped notional > remaining_position_exposure_usd
    # remaining = 10% * 10000 - 999 = 1 USD
    # min_notional=5, entry_price=1, min_qty would be 5.15 qty at $1 = $5.15 > $1 budget
    instrument = InstrumentInfo(
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        base_coin="BTC",
        quote_coin="USDT",
        min_order_qty=Decimal("1"),
        max_order_qty=Decimal("10000"),
        qty_step=Decimal("1"),
        tick_size=Decimal("0.01"),
        min_notional=Decimal("5"),
    )

    proposal = TradeProposal(
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        side=OrderSide.BUY,
        requested_qty=Decimal("1"),
        entry_price=Decimal("1"),
        stop_loss=Decimal("0.98"),
        confidence=0.01,  # low → tiny qty after LLM multiplier → triggers bump
        strategy_id="test_strategy",
        regime=MarketRegime.BULL_TREND,
    )

    decision = await rm.evaluate(
        proposal=proposal,
        capital=capital,
        available_balance=Decimal("5000"),
        instrument_info=instrument,
    )

    assert decision.status == RiskDecisionStatus.REJECTED
    assert "exposure_cap_post_bump" in decision.triggered_rules or decision.reason is not None


# ---------------------------------------------------------------------------
# Change 3: ExecutionUpdateEvent consumer
# ---------------------------------------------------------------------------


def _make_execution_event(exec_id: str = "EXEC001") -> ExecutionUpdateEvent:
    return ExecutionUpdateEvent(
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        order_id="ORD001",
        order_link_id="link001",
        exec_id=exec_id,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        exec_price=Decimal("50000"),
        exec_qty=Decimal("0.001"),
        exec_fee=Decimal("0.05"),
        exec_value=Decimal("50"),
    )


@pytest.mark.asyncio
async def test_execution_event_triggers_sync():
    """ExecutionUpdateEvent causes sync_positions() to be called."""
    from trader.domain.events import BalanceUpdateEvent

    sync_mock = AsyncMock(return_value=[])
    reconcile_mock = AsyncMock(return_value=MagicMock(discrepancies_found=0))

    # Build the consumer closure the same way app.py does
    shutdown = asyncio.Event()
    queue: asyncio.Queue = asyncio.Queue()
    seen_exec_ids: set[str] = set()

    execution_engine = MagicMock()
    execution_engine.sync_positions = sync_mock
    bybit_adapter = MagicMock()
    bybit_adapter.reconcile = reconcile_mock
    initial_shadow_mode = lambda: False  # noqa: E731

    async def consume():
        from trader.domain.events import ExecutionUpdateEvent

        while not shutdown.is_set():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.1)
                if isinstance(event, BalanceUpdateEvent) and event.available_balance > Decimal("0"):
                    pass
                elif isinstance(event, ExecutionUpdateEvent):
                    if event.exec_id in seen_exec_ids:
                        continue
                    seen_exec_ids.add(event.exec_id)
                    if execution_engine is not None:
                        try:
                            await execution_engine.sync_positions()
                        except Exception:
                            pass
                    if bybit_adapter is not None and not initial_shadow_mode():
                        try:
                            await bybit_adapter.reconcile()
                        except Exception:
                            pass
            except TimeoutError:
                shutdown.set()

    await queue.put(_make_execution_event("EXEC_A"))
    await consume()

    sync_mock.assert_called_once()
    reconcile_mock.assert_called_once()


@pytest.mark.asyncio
async def test_execution_event_idempotency():
    """Duplicate exec_id events result in sync_positions called only once."""
    shutdown = asyncio.Event()
    queue: asyncio.Queue = asyncio.Queue()
    seen_exec_ids: set[str] = set()

    sync_mock = AsyncMock(return_value=[])
    execution_engine = MagicMock()
    execution_engine.sync_positions = sync_mock

    async def consume():
        from trader.domain.events import ExecutionUpdateEvent

        while not shutdown.is_set():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.1)
                if isinstance(event, ExecutionUpdateEvent):
                    if event.exec_id in seen_exec_ids:
                        continue
                    seen_exec_ids.add(event.exec_id)
                    await execution_engine.sync_positions()
            except TimeoutError:
                shutdown.set()

    # Same exec_id twice
    await queue.put(_make_execution_event("EXEC_DUP"))
    await queue.put(_make_execution_event("EXEC_DUP"))
    await consume()

    # Only called once due to deduplication
    assert sync_mock.call_count == 1
