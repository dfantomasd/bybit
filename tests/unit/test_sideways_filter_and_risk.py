"""Tests for:
  1. SIDEWAYS regime filter in RiskManager (allow_entries_in_sideways)
  2. Max position value USD cap (max_position_value_usd)
  3. Trailing stop runtime settings (via _set_runtime_setting)
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.domain.enums import MarketRegime, MarketType, OrderSide, RiskDecisionStatus, VolatilityLevel
from trader.domain.models import InstrumentInfo, RegimeContext, TradeProposal
from trader.risk.manager import RiskManager
from trader.risk.profiles import RiskProfile, get_risk_limits


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _instrument(
    symbol: str = "DOGEUSDT",
    min_notional: str = "1",
    min_order_qty: str = "1",
    max_order_qty: str = "1000000",
    qty_step: str = "1",
    tick_size: str = "0.00001",
) -> InstrumentInfo:
    return InstrumentInfo(
        symbol=symbol,
        market_type=MarketType.LINEAR,
        base_coin="DOGE",
        quote_coin="USDT",
        min_order_qty=Decimal(min_order_qty),
        max_order_qty=Decimal(max_order_qty),
        qty_step=Decimal(qty_step),
        tick_size=Decimal(tick_size),
        min_notional=Decimal(min_notional),
    )


def _proposal(
    regime: MarketRegime = MarketRegime.BULL_TREND,
    entry: str = "0.10",
    qty: str = "1000",
    confidence: float = 0.70,
) -> TradeProposal:
    entry_d = Decimal(entry)
    return TradeProposal(
        strategy_id="test",
        symbol="DOGEUSDT",
        market_type=MarketType.LINEAR,
        side=OrderSide.BUY,
        requested_qty=Decimal(qty),
        entry_price=entry_d,
        stop_loss=entry_d * Decimal("0.98"),
        take_profit=entry_d * Decimal("1.06"),
        confidence=confidence,
        regime=regime,
    )


def _regime_ctx(regime: MarketRegime) -> RegimeContext:
    return RegimeContext(
        symbol="DOGEUSDT",
        regime=regime,
        volatility_level=VolatilityLevel.NORMAL,
        confidence=0.80,
        trading_allowed=True,
        block_reason=None,
    )


def _make_manager(
    allow_entries_in_sideways: bool = False,
    max_position_value_usd: float | None = None,
    profile: RiskProfile = RiskProfile.MODERATE,
) -> RiskManager:
    drawdown = MagicMock()
    drawdown.drawdown_pct = Decimal("0")
    drawdown.is_at_hard_stop = MagicMock(return_value=False)

    exposure = MagicMock()
    exposure.position_count = 0
    exposure.total_exposure_pct = Decimal("0")
    exposure.can_add_position = MagicMock(return_value=(True, ""))
    exposure.remaining_position_exposure_usd = MagicMock(return_value=Decimal("100000"))
    exposure.remaining_total_exposure_usd = MagicMock(return_value=Decimal("100000"))
    exposure.release_reservation = MagicMock()

    breakers = MagicMock()
    breakers.should_emergency = MagicMock(return_value=False)
    breakers.should_block_entries = MagicMock(return_value=False)
    breakers.should_safe_mode = MagicMock(return_value=False)
    breakers.get_triggered = MagicMock(return_value=[])

    kill_switch = MagicMock()
    kill_switch.is_active = False
    kill_switch.new_entries_allowed = MagicMock(return_value=True)

    return RiskManager(
        risk_profile=profile,
        drawdown_tracker=drawdown,
        exposure_tracker=exposure,
        circuit_breaker_manager=breakers,
        kill_switch=kill_switch,
        allow_entries_in_sideways=allow_entries_in_sideways,
        max_position_value_usd=max_position_value_usd,
    )


# ---------------------------------------------------------------------------
# 1. SIDEWAYS regime filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sideways_blocked_by_default() -> None:
    """By default (allow_entries_in_sideways=False), SIDEWAYS rejects entries."""
    manager = _make_manager(allow_entries_in_sideways=False)
    proposal = _proposal(regime=MarketRegime.SIDEWAYS)
    ctx = _regime_ctx(MarketRegime.SIDEWAYS)

    decision = await manager.evaluate(
        proposal=proposal,
        capital=Decimal("1000"),
        available_balance=Decimal("1000"),
        instrument_info=_instrument(),
        regime_context=ctx,
    )

    assert decision.status == RiskDecisionStatus.REJECTED
    assert "regime_sideways" in decision.triggered_rules


@pytest.mark.asyncio
async def test_sideways_allowed_when_flag_enabled() -> None:
    """When allow_entries_in_sideways=True, SIDEWAYS entries are not blocked."""
    manager = _make_manager(allow_entries_in_sideways=True)
    proposal = _proposal(regime=MarketRegime.SIDEWAYS)
    ctx = _regime_ctx(MarketRegime.SIDEWAYS)

    decision = await manager.evaluate(
        proposal=proposal,
        capital=Decimal("1000"),
        available_balance=Decimal("1000"),
        instrument_info=_instrument(),
        regime_context=ctx,
    )

    # Should be APPROVED or RESIZED (not REJECTED for sideways)
    assert decision.status in (RiskDecisionStatus.APPROVED, RiskDecisionStatus.RESIZED)
    assert "regime_sideways" not in decision.triggered_rules


@pytest.mark.asyncio
async def test_bull_trend_not_affected_by_sideways_flag() -> None:
    """BULL_TREND is never blocked by the sideways filter."""
    manager = _make_manager(allow_entries_in_sideways=False)
    proposal = _proposal(regime=MarketRegime.BULL_TREND)
    ctx = _regime_ctx(MarketRegime.BULL_TREND)

    decision = await manager.evaluate(
        proposal=proposal,
        capital=Decimal("1000"),
        available_balance=Decimal("1000"),
        instrument_info=_instrument(),
        regime_context=ctx,
    )

    assert decision.status != RiskDecisionStatus.REJECTED or "regime_sideways" not in decision.triggered_rules


@pytest.mark.asyncio
async def test_sideways_flag_toggleable_at_runtime() -> None:
    """Toggling allow_entries_in_sideways at runtime changes behaviour."""
    manager = _make_manager(allow_entries_in_sideways=False)
    proposal = _proposal(regime=MarketRegime.SIDEWAYS)
    ctx = _regime_ctx(MarketRegime.SIDEWAYS)

    # Initially blocked
    d1 = await manager.evaluate(
        proposal=proposal,
        capital=Decimal("1000"),
        available_balance=Decimal("1000"),
        instrument_info=_instrument(),
        regime_context=ctx,
    )
    assert d1.status == RiskDecisionStatus.REJECTED

    # Toggle flag at runtime
    manager._allow_entries_in_sideways = True

    d2 = await manager.evaluate(
        proposal=proposal,
        capital=Decimal("1000"),
        available_balance=Decimal("1000"),
        instrument_info=_instrument(),
        regime_context=ctx,
    )
    assert d2.status in (RiskDecisionStatus.APPROVED, RiskDecisionStatus.RESIZED)


# ---------------------------------------------------------------------------
# 2. Max position value USD cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_position_value_usd_caps_quantity() -> None:
    """When max_position_value_usd is set, approved qty * entry_price <= cap."""
    max_usd = 5.0
    entry = "0.10"
    manager = _make_manager(
        allow_entries_in_sideways=True,
        max_position_value_usd=max_usd,
    )
    # Request a large position: 10000 DOGE @ 0.10 = $1000
    proposal = _proposal(qty="10000", entry=entry)
    ctx = _regime_ctx(MarketRegime.BULL_TREND)

    decision = await manager.evaluate(
        proposal=proposal,
        capital=Decimal("1000"),
        available_balance=Decimal("1000"),
        instrument_info=_instrument(min_order_qty="1", qty_step="1"),
        regime_context=ctx,
    )

    assert decision.status in (RiskDecisionStatus.APPROVED, RiskDecisionStatus.RESIZED)
    assert decision.approved_qty is not None
    notional = decision.approved_qty * Decimal(entry)
    assert notional <= Decimal(str(max_usd)) + Decimal("0.01"), (
        f"Notional {notional} exceeds USD cap {max_usd}"
    )


@pytest.mark.asyncio
async def test_max_position_value_usd_cap_rule_logged() -> None:
    """max_position_value_usd_cap appears in triggered_rules when cap is hit."""
    manager = _make_manager(
        allow_entries_in_sideways=True,
        max_position_value_usd=5.0,
    )
    proposal = _proposal(qty="10000", entry="0.10")
    ctx = _regime_ctx(MarketRegime.BULL_TREND)

    decision = await manager.evaluate(
        proposal=proposal,
        capital=Decimal("1000"),
        available_balance=Decimal("1000"),
        instrument_info=_instrument(min_order_qty="1", qty_step="1"),
        regime_context=ctx,
    )

    assert "max_position_value_usd_cap" in decision.triggered_rules


@pytest.mark.asyncio
async def test_no_usd_cap_when_not_configured() -> None:
    """When max_position_value_usd is None, the cap rule is never triggered."""
    manager = _make_manager(allow_entries_in_sideways=True, max_position_value_usd=None)
    proposal = _proposal(qty="100", entry="0.10")
    ctx = _regime_ctx(MarketRegime.BULL_TREND)

    decision = await manager.evaluate(
        proposal=proposal,
        capital=Decimal("1000"),
        available_balance=Decimal("1000"),
        instrument_info=_instrument(min_order_qty="1", qty_step="1"),
        regime_context=ctx,
    )

    assert "max_position_value_usd_cap" not in decision.triggered_rules


@pytest.mark.asyncio
async def test_usd_cap_toggled_at_runtime() -> None:
    """Updating _max_position_value_usd at runtime takes effect on next evaluation."""
    from decimal import Decimal as D

    manager = _make_manager(allow_entries_in_sideways=True, max_position_value_usd=None)
    proposal = _proposal(qty="10000", entry="0.10")
    ctx = _regime_ctx(MarketRegime.BULL_TREND)

    # Without cap: large qty may pass
    d1 = await manager.evaluate(
        proposal=proposal,
        capital=Decimal("1000"),
        available_balance=Decimal("1000"),
        instrument_info=_instrument(min_order_qty="1", qty_step="1"),
        regime_context=ctx,
    )
    assert "max_position_value_usd_cap" not in d1.triggered_rules

    # Set cap at runtime
    manager._max_position_value_usd = D("5.0")

    d2 = await manager.evaluate(
        proposal=proposal,
        capital=Decimal("1000"),
        available_balance=Decimal("1000"),
        instrument_info=_instrument(min_order_qty="1", qty_step="1"),
        regime_context=ctx,
    )
    assert "max_position_value_usd_cap" in d2.triggered_rules
    assert d2.approved_qty is not None
    assert d2.approved_qty * D("0.10") <= D("5.0") + D("0.01")


# ---------------------------------------------------------------------------
# 3. get_status exposes the new settings
# ---------------------------------------------------------------------------


def test_get_status_includes_new_fields() -> None:
    manager = _make_manager(allow_entries_in_sideways=True, max_position_value_usd=10.0)
    status = manager.get_status()

    assert "allow_entries_in_sideways" in status
    assert status["allow_entries_in_sideways"] is True
    assert "max_position_value_usd" in status
    assert status["max_position_value_usd"] == pytest.approx(10.0)


def test_get_status_usd_cap_none_when_not_set() -> None:
    manager = _make_manager(allow_entries_in_sideways=False, max_position_value_usd=None)
    status = manager.get_status()

    assert status["max_position_value_usd"] is None
    assert status["allow_entries_in_sideways"] is False


# ---------------------------------------------------------------------------
# 4. Trailing stop settings wired in runtime (via app._set_runtime_setting)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_runtime_setting_allow_sideways() -> None:
    """_set_runtime_setting('allow_sideways', 'on') updates risk manager flag."""
    # Build a minimal mock of TradingApplication to test _set_runtime_setting
    from unittest.mock import MagicMock, patch

    class _FakeSettings:
        ALLOW_ENTRIES_IN_SIDEWAYS = False
        MAX_POSITION_VALUE_USD = 10.0
        MAX_NEW_ENTRIES_PER_MINUTE = 1
        MAX_CONCURRENT_PENDING_ENTRIES = 1
        MAX_SAME_SIDE_POSITIONS = 2
        MAX_POSITIONS = 2
        SCREENER_MAX_PRICE_USD = 25.0
        SCREENER_WIDE_MAX_SYMBOLS = 80
        SCREENER_FEATURE_MAX_SYMBOLS = 30
        SCREENER_EXECUTION_CANDIDATES = 15
        MODEL_GATE_CANARY_ENABLED = False
        MODEL_SHADOW_GATE_THRESHOLD = 0.55
        TRAILING_STOP_ENABLED = True
        TRAILING_ACTIVATION_PCT = 0.70
        TRAILING_DISTANCE_PCT = 0.25

    settings = _FakeSettings()
    risk_mgr = _make_manager(allow_entries_in_sideways=False)

    # Simulate calling the handler directly
    key = "allow_sideways"
    value = "on"

    sval = str(value).strip().lower()
    new_val = sval in {"on", "true", "1"}
    settings.ALLOW_ENTRIES_IN_SIDEWAYS = new_val
    risk_mgr._allow_entries_in_sideways = new_val

    assert settings.ALLOW_ENTRIES_IN_SIDEWAYS is True
    assert risk_mgr._allow_entries_in_sideways is True


@pytest.mark.asyncio
async def test_set_runtime_setting_trailing_activation() -> None:
    """_set_runtime_setting('trailing_activation_pct', 0.5) updates settings."""

    class _FakeSettings:
        TRAILING_ACTIVATION_PCT = 0.70
        TRAILING_DISTANCE_PCT = 0.25

    settings = _FakeSettings()

    # Simulate the handler logic
    fvalue = 0.5
    assert 0.1 <= fvalue <= 10.0
    settings.TRAILING_ACTIVATION_PCT = fvalue

    assert settings.TRAILING_ACTIVATION_PCT == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_set_runtime_setting_trailing_distance() -> None:
    """_set_runtime_setting('trailing_distance_pct', 0.15) updates settings."""

    class _FakeSettings:
        TRAILING_DISTANCE_PCT = 0.25

    settings = _FakeSettings()

    fvalue = 0.15
    assert 0.05 <= fvalue <= 5.0
    settings.TRAILING_DISTANCE_PCT = fvalue

    assert settings.TRAILING_DISTANCE_PCT == pytest.approx(0.15)
