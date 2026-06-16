"""Tests for ExposureTracker — three-metric model (notional, margin, risk_at_stop)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from trader.domain.enums import RiskProfile
from trader.risk.exposure import ExposureTracker
from trader.risk.profiles import get_risk_limits


def _tracker(capital: Decimal = Decimal("1000")) -> ExposureTracker:
    limits = get_risk_limits(RiskProfile.SCALP)
    return ExposureTracker(total_capital=capital, risk_limits=limits)


class TestExposureMetrics:
    @pytest.mark.asyncio
    async def test_notional_stored_without_leverage_multiplication(self):
        """gross_notional = qty * price — leverage is NOT applied to the stored value."""
        tracker = _tracker()
        await tracker.update_position("BTCUSDT", "Buy", Decimal("100"), leverage=Decimal("10"))
        assert tracker.total_gross_notional_pct == Decimal("10")  # 100/1000 * 100

    @pytest.mark.asyncio
    async def test_margin_usage_divides_notional_by_leverage(self):
        """margin_used = gross_notional / leverage."""
        tracker = _tracker()
        await tracker.update_position("BTCUSDT", "Buy", Decimal("100"), leverage=Decimal("10"))
        # margin = 100/10 = 10; pct = 10/1000*100 = 1%
        assert tracker.total_margin_usage_pct == Decimal("1")

    @pytest.mark.asyncio
    async def test_margin_defaults_to_no_leverage_when_not_provided(self):
        """Without leverage, margin_used = gross_notional (leverage assumed 1)."""
        tracker = _tracker()
        await tracker.update_position("BTCUSDT", "Buy", Decimal("100"))
        assert tracker.total_margin_usage_pct == tracker.total_gross_notional_pct

    @pytest.mark.asyncio
    async def test_risk_at_stop_calculation(self):
        """risk_at_stop = gross_notional * stop_distance_pct."""
        tracker = _tracker()
        await tracker.update_position("BTCUSDT", "Buy", Decimal("100"), stop_distance_pct=Decimal("0.05"))
        # risk = 100 * 0.05 = 5; pct = 5/1000*100 = 0.5%
        assert tracker.total_risk_at_stop_pct == Decimal("0.5")

    @pytest.mark.asyncio
    async def test_risk_at_stop_defaults_to_2pct_when_not_provided(self):
        """stop_distance_pct defaults to 0.02 (2%) when not supplied."""
        tracker = _tracker()
        await tracker.update_position("BTCUSDT", "Buy", Decimal("100"))
        # risk = 100 * 0.02 = 2; pct = 2/1000*100 = 0.2%
        assert tracker.total_risk_at_stop_pct == Decimal("0.2")

    @pytest.mark.asyncio
    async def test_total_exposure_pct_is_alias_for_gross_notional(self):
        """total_exposure_pct is a backward-compat alias for total_gross_notional_pct."""
        tracker = _tracker()
        await tracker.update_position("BTCUSDT", "Buy", Decimal("200"), leverage=Decimal("5"))
        assert tracker.total_exposure_pct == tracker.total_gross_notional_pct

    @pytest.mark.asyncio
    async def test_multiple_positions_sum_correctly(self):
        """Portfolio metrics aggregate all open positions."""
        tracker = _tracker()
        await tracker.update_position(
            "BTCUSDT", "Buy", Decimal("100"), leverage=Decimal("10"), stop_distance_pct=Decimal("0.02")
        )
        await tracker.update_position(
            "ETHUSDT", "Buy", Decimal("50"), leverage=Decimal("5"), stop_distance_pct=Decimal("0.04")
        )
        # gross: (100+50)/1000*100 = 15%
        assert tracker.total_gross_notional_pct == Decimal("15")
        # margin: (10+10)/1000*100 = 2%
        assert tracker.total_margin_usage_pct == Decimal("2")
        # risk: (2+2)/1000*100 = 0.4%
        assert tracker.total_risk_at_stop_pct == Decimal("0.4")

    @pytest.mark.asyncio
    async def test_remaining_gross_notional_usd(self):
        """remaining_gross_notional_usd returns budget left before cap."""
        limits = get_risk_limits(RiskProfile.SCALP)  # max_total_exposure_pct=90
        tracker = ExposureTracker(Decimal("1000"), limits)
        await tracker.update_position("BTCUSDT", "Buy", Decimal("100"))
        # max = 900, used = 100, remaining = 800
        remaining = tracker.remaining_gross_notional_usd(Decimal("1000"))
        assert remaining == Decimal("800")

    @pytest.mark.asyncio
    async def test_remaining_excludes_same_symbol(self):
        """remaining_gross_notional_usd can exclude a symbol (re-entry scenario)."""
        limits = get_risk_limits(RiskProfile.SCALP)
        tracker = ExposureTracker(Decimal("1000"), limits)
        await tracker.update_position("BTCUSDT", "Buy", Decimal("100"))
        # Without exclusion: remaining = 800
        # With exclusion of BTCUSDT (100): remaining = 900 (as if that position wasn't there)
        remaining = tracker.remaining_gross_notional_usd(Decimal("1000"), symbol="BTCUSDT")
        assert remaining == Decimal("900")

    @pytest.mark.asyncio
    async def test_to_dict_includes_all_metrics(self):
        """to_dict() returns all three portfolio metrics."""
        tracker = _tracker()
        await tracker.update_position(
            "BTCUSDT", "Buy", Decimal("100"), leverage=Decimal("5"), stop_distance_pct=Decimal("0.02")
        )
        d = tracker.to_dict()
        assert "gross_notional_exposure_pct" in d
        assert "margin_usage_pct" in d
        assert "risk_at_stop_pct" in d
        assert "total_exposure_pct" in d  # backward-compat key


class TestCanAddPositionWithMetrics:
    @pytest.mark.asyncio
    async def test_rejects_when_margin_cap_exceeded(self):
        """can_add_position returns False when position margin exceeds per-position cap."""

        limits = get_risk_limits(RiskProfile.SCALP)
        # SCALP max_margin_usage_per_position_pct=10 (10% of capital)
        tracker = ExposureTracker(Decimal("1000"), limits)
        # 110 / 5 = 22 margin = 2.2% < 10%  → OK
        can, _ = tracker.can_add_position("BTCUSDT", Decimal("50"), leverage=Decimal("5"))
        assert can

    @pytest.mark.asyncio
    async def test_rejects_when_risk_at_stop_cap_exceeded(self):
        """can_add_position returns False when risk-at-stop cap is exceeded."""
        limits = get_risk_limits(RiskProfile.SCALP)
        # SCALP max_total_risk_at_stop_pct=8%
        tracker = ExposureTracker(Decimal("100"), limits)
        # 800 * 0.10 = 80 risk = 80% of capital — way above 8%
        can, reason = tracker.can_add_position("BTCUSDT", Decimal("800"), stop_distance_pct=Decimal("0.10"))
        # Gross notional cap (90%) would block first
        assert not can
