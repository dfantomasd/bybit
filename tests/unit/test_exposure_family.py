"""Tests for ExposureTracker.count_family_positions and correlation gate."""

from __future__ import annotations

from decimal import Decimal

import pytest

from trader.risk.exposure import ExposureTracker
from trader.risk.profiles import RiskLimits


def _limits(max_pos: int = 10) -> RiskLimits:
    return RiskLimits(
        risk_per_trade_min_pct=Decimal("0.25"),
        risk_per_trade_max_pct=Decimal("0.5"),
        risk_per_trade_hard_cap_pct=Decimal("1.0"),
        max_leverage=Decimal("1"),
        daily_loss_limit_pct=Decimal("2"),
        daily_loss_hard_stop_pct=Decimal("5"),
        max_drawdown_pct=Decimal("8"),
        hard_stop_drawdown_pct=Decimal("10"),
        max_simultaneous_positions=max_pos,
        max_capital_per_position_pct=Decimal("10"),
        max_total_exposure_pct=Decimal("30"),
        short_allowed=True,
        derivatives_allowed=True,
        auto_resume_after_hard_stop=False,
    )


def _tracker(capital: Decimal = Decimal("10000")) -> ExposureTracker:
    return ExposureTracker(total_capital=capital, risk_limits=_limits())


class TestCountFamilyPositions:
    @pytest.mark.asyncio
    async def test_empty_tracker_returns_zero(self):
        tracker = _tracker()
        assert tracker.count_family_positions("BTCUSDT") == 0

    @pytest.mark.asyncio
    async def test_counts_same_family_position(self):
        tracker = _tracker()
        await tracker.update_position("BTCUSDT", "Buy", Decimal("1000"))
        # WBTCUSDT starts with "WBTC" which is in the BTC family
        assert tracker.count_family_positions("WBTCUSDT") == 1

    @pytest.mark.asyncio
    async def test_does_not_count_different_family(self):
        tracker = _tracker()
        await tracker.update_position("ETHUSDT", "Buy", Decimal("1000"))
        assert tracker.count_family_positions("BTCUSDT") == 0

    @pytest.mark.asyncio
    async def test_counts_multiple_same_family(self):
        tracker = _tracker()
        await tracker.update_position("BTCUSDT", "Buy", Decimal("1000"))
        await tracker.update_position("WBTCUSDT", "Buy", Decimal("500"))
        assert tracker.count_family_positions("BTCUSDT") == 2

    @pytest.mark.asyncio
    async def test_unknown_symbol_family_returns_zero(self):
        tracker = _tracker()
        await tracker.update_position("BTCUSDT", "Buy", Decimal("1000"))
        # DOGE has no known family in _CRYPTO_FAMILIES
        assert tracker.count_family_positions("DOGEUSDT") == 0

    @pytest.mark.asyncio
    async def test_removed_position_not_counted(self):
        tracker = _tracker()
        await tracker.update_position("BTCUSDT", "Buy", Decimal("1000"))
        await tracker.remove_position("BTCUSDT")
        assert tracker.count_family_positions("WBTCUSDT") == 0

    @pytest.mark.asyncio
    async def test_eth_family_detected(self):
        tracker = _tracker()
        await tracker.update_position("ETHUSDT", "Buy", Decimal("500"))
        assert tracker.count_family_positions("WETHUSDT") == 1

    @pytest.mark.asyncio
    async def test_sol_family_detected(self):
        tracker = _tracker()
        await tracker.update_position("SOLUSDT", "Buy", Decimal("500"))
        assert tracker.count_family_positions("MSOLUSDT") == 1

    @pytest.mark.asyncio
    async def test_counts_include_self(self):
        tracker = _tracker()
        await tracker.update_position("BTCUSDT", "Buy", Decimal("1000"))
        # Querying for BTCUSDT itself should also count it
        assert tracker.count_family_positions("BTCUSDT") == 1


class TestFamilyFalsePositives:
    """startswith used to cause unrelated tokens to land in the wrong family."""

    @pytest.mark.asyncio
    async def test_ethfi_not_in_eth_family(self):
        tracker = _tracker()
        await tracker.update_position("ETHUSDT", "Buy", Decimal("500"))
        # ETHFI is a DeFi protocol unrelated to ETH price
        assert tracker.count_family_positions("ETHFIUSDT") == 0

    @pytest.mark.asyncio
    async def test_btcdom_not_in_btc_family(self):
        tracker = _tracker()
        await tracker.update_position("BTCUSDT", "Buy", Decimal("1000"))
        # BTCDOM is BTC market dominance index, not the BTC token
        assert tracker.count_family_positions("BTCDOMUSDT") == 0

    @pytest.mark.asyncio
    async def test_solv_not_in_sol_family(self):
        tracker = _tracker()
        await tracker.update_position("SOLUSDT", "Buy", Decimal("500"))
        assert tracker.count_family_positions("SOLVUSDT") == 0

    @pytest.mark.asyncio
    async def test_ethusdt_still_in_eth_family(self):
        tracker = _tracker()
        await tracker.update_position("ETHUSDT", "Buy", Decimal("500"))
        assert tracker.count_family_positions("ETHPERP") == 1

    @pytest.mark.asyncio
    async def test_wbtcusdt_in_btc_family(self):
        tracker = _tracker()
        await tracker.update_position("BTCUSDT", "Buy", Decimal("1000"))
        assert tracker.count_family_positions("WBTCUSDT") == 1
