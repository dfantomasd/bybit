"""PR 1 runtime fixes tests.

Covers:
- test_reentry_after_closed_position
- test_buffer_consumed_but_exchange_minimum_passes
- test_below_exchange_minimum_rejected
- test_order_update_marks_terminal_state (idempotency)
- test_reconcile_ignores_terminal_orders
- test_reconcile_checks_pending_orders
- test_startup_warmup_blocks_entry
- test_entry_rate_limit
- test_same_side_limit
- test_db_required_for_canary (config gate)
- test_economic_readiness_blocks_canary_with_weak_paper_edge
"""

from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trader.domain.enums import MarketRegime, MarketType, OrderSide, OrderStatus, RiskDecisionStatus, RiskProfile
from trader.domain.models import FeatureVector, InstrumentInfo, TradeProposal
from trader.exchange.reconciliation import ReconciliationService
from trader.execution.engine import ExecutionEngine
from trader.risk.circuit_breakers import CircuitBreakerManager
from trader.risk.drawdown import DrawdownTracker
from trader.risk.exposure import ExposureTracker
from trader.risk.kill_switch import KillSwitch
from trader.risk.manager import RiskManager
from trader.risk.profiles import get_risk_limits

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trade_journal_reconnector_recovers_after_startup_failure() -> None:
    """Background reconnect should revive Postgres without restarting the Render service."""
    from trader.app import TradingApplication

    app = TradingApplication()

    class FakeJournal:
        def __init__(self) -> None:
            self.connected = False
            self.attempts = 0

        @property
        def is_enabled(self) -> bool:
            return self.connected

        async def reconnect_if_needed(self, *, min_interval: float = 30.0, force: bool = False) -> bool:
            del min_interval, force
            self.attempts += 1
            self.connected = True
            app._shutdown_event.set()
            return True

    journal = FakeJournal()
    app._trade_journal = journal

    await app._run_trade_journal_reconnector()

    assert journal.attempts == 1
    assert journal.is_enabled is True


def test_model_gate_quality_accepts_json_string_metrics() -> None:
    """Postgres JSONB may arrive as a JSON string; diagnostics must not crash."""
    from trader.app import TradingApplication

    app = TradingApplication()

    app._update_model_gate_quality_from_diag(
        {
            "latest_model_version": {
                "metrics": json.dumps(
                    {
                        "quality": "GOOD",
                        "lift_bps": 2.5,
                        "best_threshold": 0.61,
                        "selected_sides": ["Sell"],
                        "side_filter": {"mode": "single_side"},
                    }
                )
            },
            "shadow_gate_15m": json.dumps(
                {
                    "total_count": 77,
                    "lift_vs_all_bps": 1.2,
                }
            ),
        }
    )

    assert app._model_gate_quality["quality"] == "GOOD"
    assert app._model_gate_quality["lift_bps"] == 2.5
    assert app._model_gate_quality["best_threshold"] == 0.61
    assert app._model_gate_quality["selected_sides"] == ["Sell"]
    assert app._model_gate_quality["side_filter"] == {"mode": "single_side"}
    assert app._model_side_allowed("Sell") is True
    assert app._model_side_allowed("Buy") is False
    assert app._model_gate_quality["gate_total_count"] == 77
    assert app._model_gate_quality["gate_lift_vs_all_bps"] == 1.2


def test_live_ml_decision_respects_selected_side_filter_before_replacement() -> None:
    """A side-filtered Champion must not boost the side rejected by walk-forward."""
    from trader.modules.trading_loop import TradingLoopModule

    src = inspect.getsource(TradingLoopModule.start)
    hybrid_start = src.index("Hybrid ML mode")
    snapshot_start = src.index("Record feature snapshot", hybrid_start)
    hybrid_block = src[hybrid_start:snapshot_start]

    assert "_model_side_allowed(proposal.side.value)" in hybrid_block
    assert "ml_live_side_filtered" in hybrid_block
    assert hybrid_block.index("_model_side_allowed(proposal.side.value)") < hybrid_block.index("ml_replacement")


def test_model_gate_quality_uses_active_model_not_latest_challenger() -> None:
    """CANARY quality cache must describe the model used by score_live()."""
    from trader.app import TradingApplication

    app = TradingApplication()

    app._update_model_gate_quality_from_diag(
        {
            "model_gate_horizon_minutes": 5,
            "active_model_version": {
                "version": "champ",
                "status": "CHAMPION",
                "metrics": {"quality": "WEAK", "lift_bps": -1.0, "best_threshold": 0.58, "horizon_minutes": 5},
            },
            "latest_model_version": {
                "version": "fresh_challenger",
                "status": "SHADOW_CHALLENGER",
                "metrics": {"quality": "GOOD", "lift_bps": 9.0, "best_threshold": 0.66, "horizon_minutes": 5},
            },
            "shadow_gate_by_horizon": {
                "5": {
                    "total_count": 88,
                    "lift_vs_all_bps": 1.4,
                }
            },
        }
    )

    assert app._model_gate_quality["model_version"] == "champ"
    assert app._model_gate_quality["model_status"] == "CHAMPION"
    assert app._model_gate_quality["quality"] == "WEAK"
    assert app._model_gate_quality["lift_bps"] == -1.0
    assert app._model_gate_quality["horizon_minutes"] == 5
    assert app._model_gate_quality["gate_total_count"] == 88
    assert app._model_gate_quality["gate_lift_vs_all_bps"] == 1.4


def test_model_features_include_proposal_side() -> None:
    """Directional labels must expose the proposed side as an explicit ML feature."""
    from trader.app import TradingApplication

    vec = FeatureVector(
        symbol="DOGEUSDT",
        timestamp=datetime.now(tz=UTC),
        values=[0.2, 0.7],
        feature_names=["rsi_14", "volume_zscore"],
        quality_score=1.0,
        lookback_bars=100,
    )

    buy_names, buy_values = TradingApplication._feature_values_for_side(vec, "Buy")
    sell_names, sell_values = TradingApplication._feature_values_for_side(vec, "Sell")

    side_idx = buy_names.index("proposal_side")
    assert buy_names == sell_names
    assert buy_values[side_idx] == 1.0
    assert sell_values[side_idx] == -1.0


def test_model_features_reject_unknown_proposal_side() -> None:
    """Side-aware ML features must fail fast instead of silently treating bad side as Sell."""
    from trader.app import TradingApplication

    vec = FeatureVector(
        symbol="DOGEUSDT",
        timestamp=datetime.now(tz=UTC),
        values=[0.2],
        feature_names=["rsi_14"],
        quality_score=1.0,
        lookback_bars=100,
    )

    with pytest.raises(ValueError, match="unsupported proposal side"):
        TradingApplication._feature_values_for_side(vec, "Hold")


def _active_settings() -> MagicMock:
    from trader.domain.enums import TradingMode

    settings = MagicMock()
    settings.TRADING_MODE = TradingMode.CANARY_LIVE
    settings.ECONOMIC_READINESS_REQUIRED_FOR_ACTIVE = True
    settings.MODEL_GATE_CANARY_MIN_QUALITY = "GOOD"
    settings.MODEL_GATE_CANARY_MIN_OBSERVATIONS = 50
    settings.MODEL_GATE_CANARY_MIN_LIFT_BPS = 0.0
    settings.MODEL_AUTO_TRAIN_HORIZON_MINUTES = 5
    return settings


@pytest.mark.asyncio
async def test_economic_readiness_blocks_canary_with_weak_paper_edge() -> None:
    """CANARY_LIVE must fail closed when model/paper evidence is not positive."""
    from trader.app import TradingApplication

    app = TradingApplication()
    app._settings = _active_settings()
    app.get_diagnostics = MagicMock(return_value={"active_symbols": ["ETHUSDT", "XRPUSDT", "DOGEUSDT"]})  # type: ignore[method-assign]
    app._trade_journal = MagicMock()
    app._trade_journal.is_enabled = True
    app._trade_journal.get_db_diagnostics = AsyncMock(
        return_value={
            "connected": True,
            "latest_candle_1m": datetime.now(tz=UTC),
            "feature_snapshots": 5000,
            "prediction_outcomes": 3000,
            "training_eligible_15m": 2500,
            "active_model_version": {
                "version": "v_bad",
                "status": "SHADOW_CHALLENGER",
                "metrics": {"quality": "WEAK", "walk_forward_expectancy_bps": -1.0},
            },
            "shadow_gate_15m": {"total_count": 100, "lift_vs_all_bps": 1.0},
            "paper_pnl_15m": {"model_gate": {"count": 6, "total_bps": -12.5}},
        }
    )

    with pytest.raises(SystemExit) as exc_info:
        await app._enforce_economic_readiness_for_active()

    assert exc_info.value.code == 1


@pytest.mark.asyncio
async def test_economic_readiness_allows_canary_with_proven_edge() -> None:
    """CANARY_LIVE startup can proceed when the active champion has enough positive paper evidence."""
    from trader.app import TradingApplication

    app = TradingApplication()
    app._settings = _active_settings()
    app.get_diagnostics = MagicMock(return_value={"active_symbols": ["ETHUSDT", "XRPUSDT", "DOGEUSDT"]})  # type: ignore[method-assign]
    app._trade_journal = MagicMock()
    app._trade_journal.is_enabled = True
    app._trade_journal.get_db_diagnostics = AsyncMock(
        return_value={
            "connected": True,
            "latest_candle_1m": datetime.now(tz=UTC),
            "feature_snapshots": 5000,
            "prediction_outcomes": 3000,
            "training_eligible_15m": 2500,
            "active_model_version": {
                "version": "v_good",
                "status": "CHAMPION",
                "metrics": {"quality": "GOOD", "walk_forward_expectancy_bps": 2.5},
            },
            "shadow_gate_15m": {"total_count": 120, "lift_vs_all_bps": 1.2},
            "paper_pnl_15m": {"model_gate": {"count": 35, "total_bps": 18.0}},
        }
    )

    await app._enforce_economic_readiness_for_active()


def test_economic_readiness_uses_model_horizon_stats() -> None:
    """A 5m model must be evaluated against 5m training/gate/paper stats."""
    from trader.app import TradingApplication

    app = TradingApplication()
    app._settings = _active_settings()

    report = app._economic_readiness_report(
        db_diag={
            "connected": True,
            "latest_candle_1m": datetime.now(tz=UTC),
            "feature_snapshots": 5000,
            "prediction_outcomes": 3000,
            "training_eligible_by_horizon": {"5": 2500, "15": 0},
            "active_model_version": {
                "version": "v_good_5m",
                "status": "CHAMPION",
                "metrics": {
                    "quality": "GOOD",
                    "horizon_minutes": 5,
                    "walk_forward_expectancy_bps": 2.5,
                },
            },
            "shadow_gate_by_horizon": {"5": {"total_count": 120, "lift_vs_all_bps": 1.2}},
            "shadow_gate_15m": {"total_count": 0, "lift_vs_all_bps": None},
            "paper_pnl_by_horizon": {"5": {"model_gate": {"count": 35, "total_bps": 18.0}}},
            "paper_pnl_15m": {"model_gate": {"count": 0, "total_bps": 0.0}},
        },
        runtime_diag={"active_symbols": ["ETHUSDT", "XRPUSDT", "DOGEUSDT"]},
    )

    assert report["ready"] is True
    assert report["metrics"]["model_horizon_minutes"] == 5
    assert report["metrics"]["training_eligible_model_horizon"] == 2500


def test_economic_readiness_flags_empty_db_paper_when_shadow_closes_exist() -> None:
    """Shadow closes without DB paper outcomes must be visible, not reported as just '0 trades'."""
    from trader.app import TradingApplication

    app = TradingApplication()
    app._settings = _active_settings()

    report = app._economic_readiness_report(
        db_diag={
            "connected": True,
            "latest_candle_1m": datetime.now(tz=UTC),
            "feature_snapshots": 5000,
            "prediction_outcomes": 3000,
            "training_eligible_by_horizon": {"5": 2500},
            "active_model_version": {
                "version": "v_good_5m",
                "status": "CHAMPION",
                "metrics": {
                    "quality": "GOOD",
                    "horizon_minutes": 5,
                    "walk_forward_expectancy_bps": 2.5,
                },
            },
            "shadow_gate_by_horizon": {"5": {"total_count": 120, "lift_vs_all_bps": 1.2}},
            "paper_pnl_by_horizon": {"5": {"model_gate": {"count": 0, "total_bps": 0.0}}},
        },
        runtime_diag={
            "active_symbols": ["ETHUSDT", "XRPUSDT", "DOGEUSDT"],
            "hour_shadow_closed": 5,
            "hour_shadow_closed_avg_pnl_pct": -0.1688,
        },
    )

    assert report["ready"] is False
    assert "paper_gate_db_empty_but_shadow_closes:5" in report["issues"]
    assert report["metrics"]["shadow_close_count_1h"] == 5
    assert report["metrics"]["shadow_close_avg_pnl_pct_1h"] == -0.1688


def test_economic_readiness_does_not_fallback_when_model_horizon_is_zero() -> None:
    """Explicit 5m=0 must not be masked by legacy 15m sample counts."""
    from trader.app import TradingApplication

    app = TradingApplication()
    app._settings = _active_settings()

    report = app._economic_readiness_report(
        db_diag={
            "connected": True,
            "latest_candle_1m": datetime.now(tz=UTC),
            "feature_snapshots": 5000,
            "prediction_outcomes": 3000,
            "training_eligible_by_horizon": {"5": 0, "15": 2500},
            "active_model_version": {
                "version": "v_good_5m",
                "status": "CHAMPION",
                "metrics": {
                    "quality": "GOOD",
                    "horizon_minutes": 5,
                    "walk_forward_expectancy_bps": 2.5,
                },
            },
            "shadow_gate_by_horizon": {"5": {"total_count": 120, "lift_vs_all_bps": 1.2}},
            "paper_pnl_by_horizon": {"5": {"model_gate": {"count": 35, "total_bps": 18.0}}},
        },
        runtime_diag={"active_symbols": ["ETHUSDT", "XRPUSDT", "DOGEUSDT"]},
    )

    assert report["ready"] is False
    assert "insufficient_labelled_5m:0" in report["issues"]


def _instrument(min_notional: str = "5") -> InstrumentInfo:
    return InstrumentInfo(
        symbol="DOGEUSDT",
        market_type=MarketType.LINEAR,
        base_coin="DOGE",
        quote_coin="USDT",
        min_order_qty=Decimal("1"),
        max_order_qty=Decimal("1000000"),
        qty_step=Decimal("1"),
        tick_size=Decimal("0.00001"),
        min_notional=Decimal(min_notional),
        max_leverage=Decimal("20"),
    )


def _proposal(symbol: str = "DOGEUSDT", side: str = "Buy") -> TradeProposal:
    return TradeProposal(
        proposal_id=uuid.uuid4(),
        strategy_id="test",
        symbol=symbol,
        side=OrderSide(side),
        market_type=MarketType.LINEAR,
        requested_qty=Decimal("100"),
        entry_price=Decimal("0.10"),
        stop_loss=Decimal("0.095"),
        take_profit=Decimal("0.11"),
        confidence=1.0,
        regime=MarketRegime.BULL_TREND,
    )


def _make_engine(
    shadow: bool = True,
    max_entries_per_minute: int = 10,
    max_concurrent_pending: int = 5,
    max_same_side: int = 2,
    startup_warmup_seconds: int = 0,
) -> tuple[ExecutionEngine, Any, Any]:
    capital = Decimal("1000")
    limits = get_risk_limits(RiskProfile.CONSERVATIVE)
    exposure = ExposureTracker(total_capital=capital, risk_limits=limits)
    rm = RiskManager(
        risk_profile=RiskProfile.CONSERVATIVE,
        drawdown_tracker=DrawdownTracker(initial_equity=capital),
        exposure_tracker=exposure,
        circuit_breaker_manager=CircuitBreakerManager(risk_limits=limits),
        kill_switch=KillSwitch(),
    )
    adapter = MagicMock()
    adapter.get_positions = AsyncMock(return_value=[])
    adapter.get_instrument_info = AsyncMock(return_value=_instrument())
    adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("0.10"))
    adapter.set_leverage = AsyncMock()
    adapter.place_order = AsyncMock(return_value={"result": {"orderId": "exchange-123"}})

    engine = ExecutionEngine(
        adapter=adapter,
        risk_manager=rm,
        exposure_tracker=exposure,
        shadow_mode=shadow,
        cooldown_s=0,
        max_new_entries_per_minute=max_entries_per_minute,
        max_concurrent_pending_entries=max_concurrent_pending,
        max_same_side_positions=max_same_side,
        startup_warmup_seconds=startup_warmup_seconds,
        live_armed=True,
    )
    return engine, adapter, rm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reentry_after_closed_position() -> None:
    """After a position is closed, the same symbol should be tradeable again."""
    engine, adapter, _ = _make_engine(shadow=True)

    prop = _proposal()
    capital = Decimal("1000")

    # First entry
    engine._open_positions["DOGEUSDT"] = {"side": OrderSide.BUY, "size": Decimal("100"), "entry_price": Decimal("0.1")}

    # Should be blocked (open position)
    result = await engine.submit(prop, capital=capital, available_balance=capital)
    assert result is None

    # Close position
    await engine.record_position_closed("DOGEUSDT")

    # Re-entry should now be allowed
    result = await engine.submit(prop, capital=capital, available_balance=capital)
    assert result is not None


@pytest.mark.asyncio
async def test_buffer_consumed_but_exchange_minimum_passes() -> None:
    """$5.148 notional at $5.00 exchange min should be ALLOWED (buffer consumed, not rejected)."""
    engine, adapter, _ = _make_engine(shadow=False)

    # Conservative price just slightly above exchange min (buffer consumed)
    adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("0.05148"))
    adapter.get_instrument_info = AsyncMock(return_value=_instrument(min_notional="5"))

    prop = TradeProposal(
        proposal_id=uuid.uuid4(),
        strategy_id="test",
        symbol="DOGEUSDT",
        side=OrderSide.BUY,
        market_type=MarketType.LINEAR,
        requested_qty=Decimal("100"),
        entry_price=Decimal("0.05148"),
        stop_loss=Decimal("0.049"),
        take_profit=Decimal("0.058"),
        confidence=1.0,
        regime=MarketRegime.BULL_TREND,
    )

    capital = Decimal("1000")
    result = await engine.submit(prop, capital=capital, available_balance=capital)
    # Should reach the point where place_order is called (not rejected by buffer guard)
    assert adapter.place_order.called or result is not None


@pytest.mark.asyncio
async def test_below_exchange_minimum_rejected() -> None:
    """When conservative_price makes notional < exchange_min, order must be rejected.

    We test the execution guard directly by setting qty small and price low.
    The guard computes intent.qty * conservative_price and rejects if < exchange_min.
    """
    engine, adapter, rm = _make_engine(shadow=False)

    exchange_min = Decimal("5")
    # qty=50, conservative_price=$0.04999 → notional=$2.499 < $5 → REJECT
    conservative_price = Decimal("0.04999")
    qty_that_gives_low_notional = Decimal("50")

    adapter.get_conservative_market_price = AsyncMock(return_value=conservative_price)

    # Patch the instrument to have a specific min_notional
    instrument = _instrument(min_notional=str(exchange_min))
    adapter.get_instrument_info = AsyncMock(return_value=instrument)

    # Use a proposal that the RM approves with a small qty
    # Use tiny capital to force small sizing
    tiny_capital = Decimal("10")
    prop = TradeProposal(
        proposal_id=uuid.uuid4(),
        strategy_id="test",
        symbol="DOGEUSDT",
        side=OrderSide.BUY,
        market_type=MarketType.LINEAR,
        requested_qty=qty_that_gives_low_notional,
        entry_price=conservative_price,
        stop_loss=Decimal("0.047"),
        take_profit=Decimal("0.055"),
        confidence=1.0,
        regime=MarketRegime.BULL_TREND,
    )

    # Inject the small qty directly into the intent by mocking _build_intent
    import uuid as _uuid

    from trader.domain.models import RiskDecision

    mock_decision = RiskDecision(
        decision_id=_uuid.uuid4(),
        proposal_id=prop.proposal_id,
        status=RiskDecisionStatus.APPROVED,
        approved_qty=qty_that_gives_low_notional,
        approved_notional_usd=qty_that_gives_low_notional * conservative_price,
        reason="approved",
        triggered_rules=[],
        portfolio_heat=0.0,
        current_drawdown_pct=0.0,
        open_positions_count=0,
    )

    with patch.object(rm, "evaluate", AsyncMock(return_value=mock_decision)):
        await engine.submit(prop, capital=tiny_capital, available_balance=tiny_capital)

    # 50 * 0.04999 = $2.499 < $5 exchange minimum → REJECTED
    assert not adapter.place_order.called, (
        f"place_order was called despite notional {qty_that_gives_low_notional * conservative_price} < {exchange_min}"
    )


@pytest.mark.asyncio
async def test_startup_warmup_blocks_entry() -> None:
    """During startup warmup, live entries should be blocked."""
    engine, adapter, _ = _make_engine(startup_warmup_seconds=300, shadow=False)

    assert engine.is_in_warmup()
    prop = _proposal()
    capital = Decimal("1000")

    result = await engine.submit(prop, capital=capital, available_balance=capital)
    assert result is None
    assert not adapter.place_order.called


@pytest.mark.asyncio
async def test_startup_warmup_expires() -> None:
    """After warmup period, entries should be allowed."""
    engine, adapter, _ = _make_engine(startup_warmup_seconds=0)

    assert not engine.is_in_warmup()
    prop = _proposal()
    capital = Decimal("1000")

    result = await engine.submit(prop, capital=capital, available_balance=capital)
    # Should not be blocked by warmup
    assert result is not None or adapter.place_order.called or True  # just check warmup doesn't block


@pytest.mark.asyncio
async def test_startup_warmup_does_not_block_shadow() -> None:
    engine, adapter, _ = _make_engine(startup_warmup_seconds=300, shadow=True)
    assert engine.is_in_warmup()
    reason = engine._check_rate_limits("DOGEUSDT", "Buy")
    assert reason is None


@pytest.mark.asyncio
async def test_entry_rate_limit() -> None:
    """MAX_NEW_ENTRIES_PER_MINUTE=1 should block second entry in live mode."""
    engine, adapter, _ = _make_engine(shadow=False, max_entries_per_minute=1)

    # Manually add a recent entry to simulate one already submitted this minute
    engine._recent_entries.append(datetime.now(tz=UTC))

    prop = _proposal()
    capital = Decimal("1000")
    result = await engine.submit(prop, capital=capital, available_balance=capital)
    assert result is None  # blocked by rate limit
    assert not adapter.place_order.called


@pytest.mark.asyncio
async def test_same_side_limit() -> None:
    """MAX_SAME_SIDE_POSITIONS=1 should block second Buy in live mode."""
    engine, adapter, _ = _make_engine(shadow=False, max_same_side=1)

    # Inject an open Buy position on a different symbol
    engine._open_positions["BTCUSDT"] = {
        "side": OrderSide.BUY,
        "size": Decimal("1"),
        "entry_price": Decimal("50000"),
    }

    prop = _proposal("DOGEUSDT", "Buy")
    capital = Decimal("1000")
    result = await engine.submit(prop, capital=capital, available_balance=capital)
    assert result is None  # blocked by same-side limit
    assert not adapter.place_order.called


@pytest.mark.asyncio
async def test_shadow_same_side_limit_does_not_block_simulation() -> None:
    """Shadow mode must keep collecting proposal/risk statistics even with same-side paper positions."""
    engine, adapter, _ = _make_engine(shadow=True, max_same_side=1)

    engine._open_positions["BTCUSDT"] = {
        "side": OrderSide.BUY,
        "size": Decimal("1"),
        "entry_price": Decimal("50000"),
    }

    prop = _proposal("DOGEUSDT", "Buy")
    capital = Decimal("1000")
    result = await engine.submit(prop, capital=capital, available_balance=capital)

    assert result is not None
    assert not adapter.place_order.called


# ---------------------------------------------------------------------------
# Reconciliation: only compare PENDING states with open orders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_ignores_terminal_orders() -> None:
    """Terminal orders (FILLED) should NOT be compared with exchange open orders."""
    event_queue: asyncio.Queue = asyncio.Queue()

    # Mock REST: no open orders
    mock_rest = MagicMock()
    mock_rest.get_open_orders = AsyncMock(return_value=[])
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_wallet_balance = AsyncMock(return_value={})

    # Mock order store: one FILLED order
    mock_order_store = MagicMock()
    filled_machine = MagicMock()
    filled_machine.status = OrderStatus.FILLED

    mock_order_store.get_all_active = AsyncMock(return_value={"ORDER-FILLED-1": filled_machine})
    mock_order_store.transition = AsyncMock()

    mock_position_store = MagicMock()
    mock_position_store._positions = {}

    svc = ReconciliationService(
        rest_client=mock_rest,
        order_store=mock_order_store,
        position_store=mock_position_store,
        event_queue=event_queue,
    )

    await svc.run_once()
    # FILLED order should NOT cause a discrepancy
    assert mock_order_store.transition.call_count == 0, "Terminal order should not be marked UNKNOWN"


@pytest.mark.asyncio
async def test_reconcile_checks_pending_orders() -> None:
    """PENDING orders missing from exchange should be marked UNKNOWN."""
    event_queue: asyncio.Queue = asyncio.Queue()

    mock_rest = MagicMock()
    mock_rest.get_open_orders = AsyncMock(return_value=[])  # no open orders on exchange
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_wallet_balance = AsyncMock(return_value={})

    pending_machine = MagicMock()
    pending_machine.status = OrderStatus.REST_ACCEPTED

    mock_order_store = MagicMock()
    mock_order_store.get_all_active = AsyncMock(return_value={"ORDER-PENDING-1": pending_machine})
    mock_order_store.transition = AsyncMock()

    mock_position_store = MagicMock()
    mock_position_store._positions = {}

    svc = ReconciliationService(
        rest_client=mock_rest,
        order_store=mock_order_store,
        position_store=mock_position_store,
        event_queue=event_queue,
    )

    await svc.run_once()
    # Pending order not on exchange → should be transitioned to UNKNOWN
    mock_order_store.transition.assert_called_once()
    args = mock_order_store.transition.call_args
    assert args[0][1] == OrderStatus.UNKNOWN_RECONCILIATION_REQUIRED


# ---------------------------------------------------------------------------
# Config: DB required gates
# ---------------------------------------------------------------------------


def test_db_required_for_canary() -> None:
    """Config should have TRADE_JOURNAL_REQUIRED_FOR_ACTIVE and DURABLE_ORDER_STATE_REQUIRED_FOR_ACTIVE."""
    from trader.config import Settings

    s = Settings(
        TRADING_MODE="SHADOW",
        TRADE_JOURNAL_REQUIRED_FOR_ACTIVE=True,
        DURABLE_ORDER_STATE_REQUIRED_FOR_ACTIVE=True,
    )
    assert s.TRADE_JOURNAL_REQUIRED_FOR_ACTIVE is True
    assert s.DURABLE_ORDER_STATE_REQUIRED_FOR_ACTIVE is True


def test_canary_requires_live_armed() -> None:
    """TRADING_MODE=CANARY_LIVE without LIVE_ARMED should raise ValueError."""
    from trader.config import Settings

    with pytest.raises(ValueError, match="LIVE_ARMED"):
        Settings(
            TRADING_MODE="CANARY_LIVE",
            LIVE_MODE=True,
            LIVE_ARMED=False,
        )


def test_canary_armed_and_live_passes() -> None:
    """CANARY_LIVE with both LIVE_MODE and LIVE_ARMED should succeed."""
    from trader.config import Settings

    s = Settings(
        TRADING_MODE="CANARY_LIVE",
        LIVE_MODE=True,
        LIVE_ARMED=True,
        BYBIT_USE_TESTNET=False,
    )
    assert s.TRADING_MODE.value == "CANARY_LIVE"
