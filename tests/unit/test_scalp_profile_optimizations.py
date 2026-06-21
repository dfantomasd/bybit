"""SCALP profile production-log driven optimizations."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.app import TradingApplication
from trader.config import Settings
from trader.domain.enums import RiskProfile
from trader.execution.engine import ExecutionEngine


def _make_app(**overrides) -> TradingApplication:
    app = TradingApplication()
    defaults = {
        "TELEGRAM_ALLOWED_CHAT_IDS": [],
        "RISK_PROFILE": RiskProfile.SCALP,
        "TRADING_MODE": "SHADOW",
        "SHADOW_MODE": True,
        "SCALP_STRICT_SHADOW": True,
        "SCALP_DISABLE_TREND_STRATEGY": True,
        "BUCKET_BLOCK_ENABLED": True,
        "BUCKET_MIN_SAMPLES": 30,
        "BUCKET_BLOCK_AVG_BPS": -2.0,
        "SYMBOL_SIDE_BLOCK_ENABLED": True,
        "SYMBOL_SIDE_MIN_SAMPLES": 20,
        "SYMBOL_SIDE_BLOCK_AVG_BPS": -2.0,
    }
    defaults.update(overrides)
    app._settings = Settings(**defaults)
    return app


class TestScalpStrictShadow:
    def test_scalp_strict_shadow_blocks_toxic_symbol_side(self) -> None:
        app = _make_app()
        app._symbol_side_stats = {("ADAUSDT", "Buy"): (-5.0, 25)}
        assert app._symbol_side_blocked("ADAUSDT", "Buy") is True

    def test_non_scalp_shadow_still_skips_symbol_side_gate(self) -> None:
        app = _make_app(RISK_PROFILE=RiskProfile.MODERATE)
        app._symbol_side_stats = {("ADAUSDT", "Buy"): (-5.0, 25)}
        assert app._symbol_side_blocked("ADAUSDT", "Buy") is False

    def test_scalp_strict_shadow_can_be_disabled(self) -> None:
        app = _make_app(SCALP_STRICT_SHADOW=False)
        app._symbol_side_stats = {("ADAUSDT", "Buy"): (-5.0, 25)}
        assert app._symbol_side_blocked("ADAUSDT", "Buy") is False

    def test_scalp_strict_shadow_blocks_toxic_bucket(self) -> None:
        app = _make_app()
        from types import SimpleNamespace

        key = ("BULL_TREND", "NORMAL", datetime.now(tz=UTC).hour)
        app._bucket_stats = {key: (-5.0, 50)}
        regime = SimpleNamespace(
            regime=SimpleNamespace(value="BULL_TREND"),
            volatility_level=SimpleNamespace(value="NORMAL"),
        )
        assert app._bucket_blocked(regime) is True


class TestMicroAccountNotionalBuffer:
    def test_micro_account_uses_reduced_buffer(self) -> None:
        engine = ExecutionEngine(
            adapter=MagicMock(),
            risk_manager=MagicMock(),
            exposure_tracker=MagicMock(),
            min_notional_safety_buffer_pct=3.0,
            micro_account_balance_usd=50.0,
            micro_account_min_notional_buffer_pct=1.0,
        )
        assert engine._min_notional_buffer_for_balance(Decimal("23.5")) == Decimal("1.0")
        assert engine._min_notional_buffer_for_balance(Decimal("100")) == Decimal("3.0")


@pytest.mark.asyncio
async def test_shadow_scalp_applies_net_edge_gate() -> None:
    from trader.domain.enums import MarketType, OrderSide, RiskDecisionStatus
    from trader.domain.models import InstrumentInfo, RiskDecision, TradeProposal

    adapter = MagicMock()
    risk_manager = MagicMock()
    exposure = MagicMock()
    exposure.total_exposure_pct = Decimal("0")

    fee_provider = MagicMock()
    fee_rates = MagicMock()
    fee_rates.taker_fee_rate = 0.00055
    fee_provider.get = AsyncMock(return_value=fee_rates)

    engine = ExecutionEngine(
        adapter=adapter,
        risk_manager=risk_manager,
        exposure_tracker=exposure,
        shadow_mode=True,
        shadow_apply_net_edge_gate=True,
        fee_provider=fee_provider,
        min_net_edge_pct=0.25,
        max_spread_bps=8.0,
        expected_slippage_pct=0.03,
        funding_buffer_pct=0.01,
        net_edge_safety_margin_pct=0.05,
    )

    proposal = MagicMock(spec=TradeProposal)
    proposal.symbol = "ADAUSDT"
    proposal.side = OrderSide.BUY
    proposal.entry_price = Decimal("0.1608")
    proposal.take_profit = Decimal("0.16144621")
    proposal.stop_loss = Decimal("0.16047689")
    proposal.confidence = 0.65
    proposal.requested_qty = Decimal("33")
    proposal.proposal_id = "test-scalp-shadow"
    proposal.strategy_id = "ema_crossover_v1"
    proposal.rationale = "test"

    decision = MagicMock(spec=RiskDecision)
    decision.status = RiskDecisionStatus.APPROVED
    decision.approved_qty = Decimal("33")
    decision.approved_notional_usd = Decimal("5.31")
    decision.reason = "ok"
    decision.decision_id = "decision-scalp"
    decision.proposal_id = "test-scalp-shadow"
    decision.triggered_rules = []
    decision.portfolio_heat = 22.0
    decision.current_drawdown_pct = 0.0
    decision.open_positions_count = 0

    instrument = InstrumentInfo(
        symbol="ADAUSDT",
        market_type=MarketType.LINEAR,
        base_coin="ADA",
        quote_coin="USDT",
        min_order_qty=Decimal("1"),
        max_order_qty=Decimal("100000"),
        qty_step=Decimal("1"),
        tick_size=Decimal("0.0001"),
        min_notional=Decimal("5"),
    )

    engine._open_positions = {}
    engine._pending_entry_order_link_ids = set()
    engine._is_canary = False
    engine._risk_manager.evaluate = AsyncMock(return_value=decision)
    engine._adapter.get_instrument_info = AsyncMock(return_value=instrument)
    engine._adapter.get_conservative_market_price = AsyncMock(return_value=Decimal("0.1608"))
    engine._trade_journal = None

    result = await engine._submit_locked(
        proposal=proposal,
        capital=Decimal("23.5"),
        available_balance=Decimal("23.5"),
    )

    assert result is None
    assert engine._diag_net_edge_rejected == 1
