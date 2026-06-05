"""Shared pytest fixtures for the test suite."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from trader.domain.enums import (
    MarketRegime,
    MarketType,
    OrderSide,
    OrderType,
    RiskDecisionStatus,
    RiskProfile,
    TradingMode,
    VolatilityLevel,
)
from trader.domain.models import (
    FeatureVector,
    MarketEvent,
    OrderIntent,
    RegimeContext,
    RiskDecision,
    TradeProposal,
)


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Domain model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_market_event() -> MarketEvent:
    """A valid MarketEvent for BTCUSDT."""
    return MarketEvent(
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        close=Decimal("65000.00"),
        volume=Decimal("1234.56"),
        bid=Decimal("64999.00"),
        ask=Decimal("65001.00"),
        bid_size=Decimal("1.5"),
        ask_size=Decimal("2.0"),
        mark_price=Decimal("65000.50"),
        latency_ms=12.5,
    )


@pytest.fixture()
def sample_feature_vector() -> FeatureVector:
    """A valid FeatureVector with 10 features."""
    names = [f"feat_{i}" for i in range(10)]
    values = [float(i) * 0.1 for i in range(10)]
    return FeatureVector(
        symbol="BTCUSDT",
        values=values,
        feature_names=names,
        quality_score=0.92,
        lookback_bars=100,
        version="v1",
    )


@pytest.fixture()
def sample_trade_proposal() -> TradeProposal:
    """A valid BUY TradeProposal for BTCUSDT."""
    return TradeProposal(
        strategy_id="ppo-v1",
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        side=OrderSide.BUY,
        requested_qty=Decimal("0.001"),
        requested_notional_usd=Decimal("65.00"),
        entry_price=Decimal("65000.00"),
        stop_loss=Decimal("63000.00"),
        take_profit=Decimal("68000.00"),
        confidence=0.72,
        expected_return=0.046,
        expected_risk=0.031,
        regime=MarketRegime.BULL_TREND,
        rationale="PPO model output: long signal with 72% confidence",
    )


@pytest.fixture()
def sample_risk_decision(sample_trade_proposal: TradeProposal) -> RiskDecision:
    """An APPROVED RiskDecision for the sample trade proposal."""
    return RiskDecision(
        proposal_id=sample_trade_proposal.proposal_id,
        status=RiskDecisionStatus.APPROVED,
        approved_qty=sample_trade_proposal.requested_qty,
        approved_notional_usd=sample_trade_proposal.requested_notional_usd,
        reason="All risk checks passed",
        triggered_rules=[],
        portfolio_heat=2.1,
        current_drawdown_pct=0.5,
        open_positions_count=1,
    )


@pytest.fixture()
def sample_order_intent(
    sample_trade_proposal: TradeProposal,
    sample_risk_decision: RiskDecision,
) -> OrderIntent:
    """A valid MARKET OrderIntent derived from proposal + decision."""
    return OrderIntent(
        decision_id=sample_risk_decision.decision_id,
        proposal_id=sample_trade_proposal.proposal_id,
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=Decimal("0.001"),
        order_link_id=f"ppo-{uuid.uuid4().hex[:8]}",
        stop_loss=Decimal("63000.00"),
        take_profit=Decimal("68000.00"),
    )


@pytest.fixture()
def sample_regime_context() -> RegimeContext:
    """A BULL_TREND regime context for BTCUSDT."""
    return RegimeContext(
        symbol="BTCUSDT",
        regime=MarketRegime.BULL_TREND,
        volatility_level=VolatilityLevel.NORMAL,
        confidence=0.85,
        realized_vol_1h=0.012,
        realized_vol_24h=0.018,
        spread_bps=1.5,
        trading_allowed=True,
    )


# ---------------------------------------------------------------------------
# Config / settings fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_settings() -> MagicMock:
    """A MagicMock with sensible defaults mimicking a Settings instance."""
    settings = MagicMock()
    settings.TRADING_MODE = TradingMode.TESTNET
    settings.RISK_PROFILE = RiskProfile.CONSERVATIVE
    settings.BYBIT_USE_TESTNET = True
    settings.LIVE_MODE = False
    settings.SHADOW_MODE = True
    settings.MAX_POSITIONS = 2
    settings.LOG_LEVEL = "INFO"
    settings.LOG_FORMAT = "console"
    settings.FASTAPI_PORT = 8080
    settings.PROMETHEUS_PORT = 9090

    # SecretStr-like mocks
    settings.BYBIT_API_KEY.get_secret_value.return_value = "test-key"
    settings.BYBIT_API_SECRET.get_secret_value.return_value = "test-secret"
    settings.POSTGRES_DSN.get_secret_value.return_value = (
        "postgresql+asyncpg://trader:trader@localhost:5432/trader_test"
    )
    settings.REDIS_URL.get_secret_value.return_value = "redis://localhost:6379/0"
    settings.TELEGRAM_BOT_TOKEN.get_secret_value.return_value = "test-token"
    settings.TELEGRAM_ALLOWED_CHAT_IDS = [123456]
    return settings
