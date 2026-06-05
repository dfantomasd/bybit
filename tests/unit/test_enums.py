"""Unit tests for all domain enumerations."""
from __future__ import annotations

import pytest

from trader.domain.enums import (
    BybitRegion,
    KillSwitchMode,
    MarketRegime,
    MarketType,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskDecisionStatus,
    RiskProfile,
    SystemStatus,
    TradingMode,
    VolatilityLevel,
)


class TestTradingMode:
    def test_all_values_exist(self) -> None:
        assert TradingMode.TESTNET == "TESTNET"
        assert TradingMode.SHADOW == "SHADOW"
        assert TradingMode.CANARY_LIVE == "CANARY_LIVE"
        assert TradingMode.LIVE == "LIVE"

    def test_count(self) -> None:
        assert len(TradingMode) == 4

    def test_is_str(self) -> None:
        assert isinstance(TradingMode.TESTNET, str)

    def test_default_is_safe(self) -> None:
        """TESTNET is the expected safe default — guard against accidental rename."""
        assert TradingMode.TESTNET.value == "TESTNET"
        assert TradingMode.LIVE.value == "LIVE"


class TestSystemStatus:
    def test_all_values_exist(self) -> None:
        expected = {
            "STARTING", "PREFLIGHT", "RUNNING", "SAFE_MODE",
            "PAUSED", "STOPPING", "STOPPED", "BLOCKED", "ERROR",
        }
        actual = {s.value for s in SystemStatus}
        assert actual == expected

    def test_count(self) -> None:
        assert len(SystemStatus) == 9


class TestRiskProfile:
    def test_values(self) -> None:
        assert RiskProfile.CONSERVATIVE == "CONSERVATIVE"
        assert RiskProfile.MODERATE == "MODERATE"
        assert RiskProfile.AGGRESSIVE == "AGGRESSIVE"

    def test_count(self) -> None:
        assert len(RiskProfile) == 3


class TestMarketRegime:
    def test_all_values_exist(self) -> None:
        expected = {
            "BULL_TREND", "BEAR_TREND", "SIDEWAYS",
            "HIGH_VOLATILITY", "LOW_LIQUIDITY", "EVENT_RISK", "UNCERTAIN",
        }
        actual = {r.value for r in MarketRegime}
        assert actual == expected

    def test_uncertain_is_safe_default(self) -> None:
        assert MarketRegime.UNCERTAIN.value == "UNCERTAIN"


class TestOrderStatus:
    def test_all_values_exist(self) -> None:
        expected = {
            "CREATED_LOCAL", "SUBMITTING", "REST_ACCEPTED", "WS_CONFIRMED",
            "PARTIALLY_FILLED", "FILLED", "CANCEL_REQUESTED", "CANCELLED",
            "REJECTED", "EXPIRED", "UNKNOWN_RECONCILIATION_REQUIRED",
        }
        actual = {s.value for s in OrderStatus}
        assert actual == expected

    def test_count(self) -> None:
        assert len(OrderStatus) == 11

    def test_unknown_state_forces_reconciliation(self) -> None:
        """Ensure the value for unknown state is long and explicit."""
        assert "RECONCILIATION" in OrderStatus.UNKNOWN_RECONCILIATION_REQUIRED.value


class TestRiskDecisionStatus:
    def test_all_values_exist(self) -> None:
        expected = {"APPROVED", "RESIZED", "REJECTED", "SAFE_MODE_ONLY", "PAUSED"}
        actual = {s.value for s in RiskDecisionStatus}
        assert actual == expected


class TestMarketType:
    def test_bybit_api_values(self) -> None:
        """Values must match the Bybit v5 API category strings."""
        assert MarketType.SPOT == "spot"
        assert MarketType.LINEAR == "linear"
        assert MarketType.INVERSE == "inverse"
        assert MarketType.OPTION == "option"


class TestOrderSide:
    def test_bybit_api_values(self) -> None:
        """Values must match the Bybit v5 API side strings."""
        assert OrderSide.BUY == "Buy"
        assert OrderSide.SELL == "Sell"


class TestOrderType:
    def test_bybit_api_values(self) -> None:
        assert OrderType.MARKET == "Market"
        assert OrderType.LIMIT == "Limit"


class TestBybitRegion:
    def test_global_value(self) -> None:
        assert BybitRegion.GLOBAL == "GLOBAL"

    def test_all_regions(self) -> None:
        regions = {r.value for r in BybitRegion}
        assert "GLOBAL" in regions
        assert "EEA" in regions
        assert "NL" in regions


class TestVolatilityLevel:
    def test_ordered_values(self) -> None:
        assert VolatilityLevel.LOW == "LOW"
        assert VolatilityLevel.NORMAL == "NORMAL"
        assert VolatilityLevel.HIGH == "HIGH"
        assert VolatilityLevel.EXTREME == "EXTREME"

    def test_count(self) -> None:
        assert len(VolatilityLevel) == 4


class TestKillSwitchMode:
    def test_all_modes_exist(self) -> None:
        expected = {
            "PAUSE_NEW_ENTRIES",
            "CANCEL_OPEN_ORDERS",
            "REDUCE_RISK",
            "CLOSE_ALL_IF_CONFIGURED",
            "FULL_STOP",
        }
        actual = {m.value for m in KillSwitchMode}
        assert actual == expected

    def test_full_stop_is_most_severe(self) -> None:
        """FULL_STOP must exist as the terminal escalation level."""
        assert KillSwitchMode.FULL_STOP.value == "FULL_STOP"

    def test_enum_membership(self) -> None:
        assert "PAUSE_NEW_ENTRIES" in KillSwitchMode._value2member_map_
