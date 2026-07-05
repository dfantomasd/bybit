"""Tests for the VWAP-Pullback micro-scalping strategy."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from trader.domain.enums import OrderSide
from trader.domain.models import FeatureVector
from trader.strategies.scalp_micro import ScalpMicroStrategy

_SYMBOL = "TESTUSDT"
_INTERVAL = "1"
_PRICE = 100.0


def _vector(
    rsi: float = 0.50,
    adx: float = 0.30,
    atr_pct: float = 0.004,
    ewma: float = 0.005,  # bullish stack (> 0.003 threshold)
    vwap_dist: float = -0.3,  # pulled back into VWAP zone for BUY (-0.6 to +0.2)
    ob_imb: float = 0.20,  # buyers in book
    ob_present: float = 1.0,
    macd_hist: float = 0.0001,  # positive momentum
    vol_z: float = 0.5,
) -> FeatureVector:
    names = [
        "rsi_14",
        "adx_14",
        "atr_14_pct",
        "ewma_tier_signal",
        "vwap_distance_pct",
        "ob_imbalance_l5",
        "ob_data_present",
        "macd_hist",
        "volume_zscore",
    ]
    values = [rsi, adx, atr_pct, ewma, vwap_dist, ob_imb, ob_present, macd_hist, vol_z]
    return FeatureVector(
        feature_id=uuid.uuid4(),
        symbol=_SYMBOL,
        timestamp=datetime.now(tz=UTC),
        values=values,
        feature_names=names,
        quality_score=1.0,
        lookback_bars=60,
    )


def _strategy(
    spread_bps: float | None = 1.0,
    taker_fee_pct: float = 0.055,
    shadow_relaxed: bool = False,
    diag_hook: Callable[[str], None] | None = None,
    **kwargs,
) -> ScalpMicroStrategy:
    return ScalpMicroStrategy(
        candle_store=None,
        interval=_INTERVAL,
        spread_provider=lambda _s: spread_bps,
        taker_fee_pct=taker_fee_pct,
        expected_slippage_pct=0.01,
        min_net_return_pct=0.05,
        max_spread_bps=3.0,
        cooldown_seconds=60,
        max_trades_per_minute=10,
        max_position_notional_usd=100.0,
        shadow_relaxed=shadow_relaxed,
        diag_hook=diag_hook,
        **kwargs,
    )


class TestScalpMicroStrategy:
    def setup_method(self) -> None:
        ScalpMicroStrategy._global_signal_times.clear()

    # ------------------------------------------------------------------
    # Happy-path signal generation
    # ------------------------------------------------------------------

    def test_buy_signal_on_vwap_pullback(self) -> None:
        strat = _strategy()
        proposal = strat.evaluate(_vector(), _PRICE, 1000.0)
        assert proposal is not None
        assert proposal.side == OrderSide.BUY

    def test_sell_signal_on_vwap_pullback(self) -> None:
        strat = _strategy()
        sell_vec = _vector(
            ewma=-0.005,  # bearish stack
            vwap_dist=0.3,  # price 0.3% above VWAP (pullback from above)
            ob_imb=-0.20,  # sellers in book
            macd_hist=-0.0001,  # negative momentum
            rsi=0.50,
        )
        proposal = strat.evaluate(sell_vec, _PRICE, 1000.0)
        assert proposal is not None
        assert proposal.side == OrderSide.SELL

    def test_reward_risk_ratio(self) -> None:
        strat = _strategy()
        proposal = strat.evaluate(_vector(), _PRICE, 1000.0)
        assert proposal is not None
        entry = float(proposal.entry_price)
        tp_dist = float(proposal.take_profit) - entry
        sl_dist = entry - float(proposal.stop_loss)
        assert tp_dist > 0 and sl_dist > 0
        # TP = 1.6 * ATR, SL = 0.65 * ATR → ratio ≈ 2.46
        assert abs(tp_dist / sl_dist - (1.6 / 0.65)) < 0.05
        assert proposal.spread_bps == 1.0

    # ------------------------------------------------------------------
    # Signal filter rejections
    # ------------------------------------------------------------------

    def test_wide_spread_rejected(self) -> None:
        rejections: list[str] = []
        strat = _strategy(spread_bps=5.0, diag_hook=rejections.append)
        assert strat.evaluate(_vector(), _PRICE, 1000.0) is None
        assert "spread_rejected" in rejections

    def test_unknown_spread_fails_closed(self) -> None:
        strat = _strategy(spread_bps=None)
        assert strat.evaluate(_vector(), _PRICE, 1000.0) is None

    def test_flat_market_adx_rejected(self) -> None:
        strat = _strategy()
        assert strat.evaluate(_vector(adx=0.15), _PRICE, 1000.0) is None

    def test_net_edge_rejected_when_costs_exceed_edge(self) -> None:
        rejections: list[str] = []
        # atr=0.4%, TP=1.6×ATR → gross=0.64%; fees 0.4*2=0.80% > gross → net negative
        strat = _strategy(taker_fee_pct=0.40, diag_hook=rejections.append)
        assert strat.evaluate(_vector(), _PRICE, 1000.0) is None
        assert "scalp_net_edge_rejected" in rejections

    def test_ewma_too_weak_no_signal(self) -> None:
        # EWMA below threshold: no strong directional trend
        strat = _strategy()
        assert strat.evaluate(_vector(ewma=0.001), _PRICE, 1000.0) is None

    def test_ewma_neutral_no_signal(self) -> None:
        strat = _strategy()
        assert strat.evaluate(_vector(ewma=0.0), _PRICE, 1000.0) is None

    def test_vwap_too_high_for_buy_no_signal(self) -> None:
        # Price 0.5% above VWAP → not a pullback, skip BUY
        strat = _strategy()
        assert strat.evaluate(_vector(vwap_dist=0.5), _PRICE, 1000.0) is None

    def test_vwap_free_fall_no_buy_signal(self) -> None:
        # Price 0.8% below VWAP → free-fall, skip BUY
        strat = _strategy()
        assert strat.evaluate(_vector(vwap_dist=-0.8), _PRICE, 1000.0) is None

    def test_rsi_overbought_no_buy_signal(self) -> None:
        strat = _strategy()
        assert strat.evaluate(_vector(rsi=0.75), _PRICE, 1000.0) is None

    def test_rsi_oversold_no_buy_signal(self) -> None:
        strat = _strategy()
        assert strat.evaluate(_vector(rsi=0.25), _PRICE, 1000.0) is None

    def test_macd_negative_blocks_buy(self) -> None:
        strat = _strategy()
        assert strat.evaluate(_vector(macd_hist=-0.0001), _PRICE, 1000.0) is None

    def test_dead_market_volume_rejected(self) -> None:
        strat = _strategy()
        assert strat.evaluate(_vector(vol_z=-2.0), _PRICE, 1000.0) is None

    # ------------------------------------------------------------------
    # Orderbook imbalance
    # ------------------------------------------------------------------

    def test_ob_missing_fails_closed(self) -> None:
        rejections: list[str] = []
        strat = _strategy(diag_hook=rejections.append)
        assert strat.evaluate(_vector(ob_present=0.0), _PRICE, 1000.0) is None
        assert "imbalance_missing" in rejections

    def test_ob_imbalance_against_buy_rejected(self) -> None:
        rejections: list[str] = []
        strat = _strategy(diag_hook=rejections.append)
        # Book shows sellers, not buyers
        assert strat.evaluate(_vector(ob_imb=-0.30), _PRICE, 1000.0) is None
        assert "imbalance_rejected" in rejections
        assert f"imbalance_rejected:{_SYMBOL}:Buy" in rejections

    def test_ob_weak_imbalance_rejected(self) -> None:
        rejections: list[str] = []
        strat = _strategy(diag_hook=rejections.append)
        # Imbalance below 0.08 threshold
        assert strat.evaluate(_vector(ob_imb=0.05), _PRICE, 1000.0) is None
        assert "imbalance_rejected" in rejections

    def test_ob_missing_allowed_in_shadow_relaxed(self) -> None:
        strat = _strategy(shadow_relaxed=True)
        assert strat.evaluate(_vector(ob_present=0.0), _PRICE, 1000.0) is not None

    def test_ob_weak_imbalance_allowed_in_shadow_relaxed(self) -> None:
        strat = _strategy(shadow_relaxed=True)
        # 0.03 < shadow threshold (0.05) but allowed in shadow mode
        assert strat.evaluate(_vector(ob_imb=0.03), _PRICE, 1000.0) is not None

    # ------------------------------------------------------------------
    # Shadow-relaxed mode
    # ------------------------------------------------------------------

    def test_shadow_relaxed_rejects_negative_net_edge(self) -> None:
        rejections: list[str] = []
        strat = _strategy(taker_fee_pct=0.40, shadow_relaxed=True, diag_hook=rejections.append)
        assert strat.evaluate(_vector(), _PRICE, 1000.0) is None
        assert "scalp_net_edge_rejected" in rejections

    def test_shadow_relaxed_allows_weaker_ewma(self) -> None:
        strat = _strategy(shadow_relaxed=True)
        # 0.002 < normal threshold 0.003 but > shadow threshold 0.001
        assert strat.evaluate(_vector(ewma=0.002), _PRICE, 1000.0) is not None

    def test_shadow_relaxed_allows_weaker_adx(self) -> None:
        strat = _strategy(shadow_relaxed=True)
        # 0.16 < normal threshold 0.18 but > shadow threshold 0.14
        assert strat.evaluate(_vector(adx=0.16), _PRICE, 1000.0) is not None

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def test_symbol_cooldown(self) -> None:
        strat = _strategy()
        first = strat.evaluate(_vector(), _PRICE, 1000.0)
        assert first is not None
        second = strat.evaluate(_vector(), _PRICE, 1000.0)
        assert second is None  # within cooldown

    def test_global_rate_limit(self) -> None:
        strat = _strategy()
        now = datetime.now(tz=UTC)
        for _ in range(10):
            ScalpMicroStrategy._global_signal_times.append(now)
        assert strat.evaluate(_vector(), _PRICE, 1000.0) is None

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def test_notional_cap_applied(self) -> None:
        strat = _strategy()
        proposal = strat.evaluate(_vector(), _PRICE, 100_000.0)
        assert proposal is not None
        assert float(proposal.requested_notional_usd) <= 100.0
