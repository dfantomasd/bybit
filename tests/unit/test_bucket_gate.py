"""Tests for the regime-bucket expectancy gate."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from trader.app import TradingApplication
from trader.config import Settings
from trader.domain.models import FeatureVector


def _make_app(**overrides) -> TradingApplication:
    app = TradingApplication()
    defaults = {
        "TELEGRAM_ALLOWED_CHAT_IDS": [],
        "BUCKET_BLOCK_ENABLED": True,
        "BUCKET_MIN_SAMPLES": 30,
        "BUCKET_BLOCK_AVG_BPS": -2.0,
        "HOUR_BLOCK_ENABLED": True,
        "HOUR_MIN_SAMPLES": 30,
        "HOUR_BLOCK_AVG_BPS": -10.0,
        "STRATEGY_BLOCK_ENABLED": True,
        "STRATEGY_MIN_SAMPLES": 20,
        "STRATEGY_BLOCK_AVG_BPS": 0.0,
        "SYMBOL_SIDE_BLOCK_ENABLED": True,
        "SYMBOL_SIDE_MIN_SAMPLES": 20,
        "SYMBOL_SIDE_BLOCK_AVG_BPS": -2.0,
        "SHADOW_LOSS_GUARD_ENABLED": True,
        "SHADOW_LOSS_GUARD_MIN_CLOSED": 3,
        "SHADOW_LOSS_GUARD_WINDOW": 5,
        "SHADOW_LOSS_GUARD_MAX_LOSS_RATE": 0.6,
        "SHADOW_LOSS_GUARD_MIN_AVG_PNL_PCT": -0.05,
        "SHADOW_LOSS_GUARD_COOLDOWN_SECONDS": 900,
    }
    defaults.update(overrides)
    app._settings = Settings(**defaults)
    return app


def _make_active_app(**overrides) -> TradingApplication:
    app = _make_app(**overrides)
    app._modules.signal_policy.initial_shadow_mode = lambda: False  # type: ignore[method-assign]
    return app


def _regime_ctx(regime: str = "BULL_TREND", volatility: str = "NORMAL") -> SimpleNamespace:
    return SimpleNamespace(
        regime=SimpleNamespace(value=regime),
        volatility_level=SimpleNamespace(value=volatility),
    )


def _key(regime: str = "BULL_TREND", volatility: str = "NORMAL") -> tuple[str, str, int]:
    return (regime, volatility, datetime.now(tz=UTC).hour)


def _trend_vec(symbol: str, *, bullish: bool = True) -> FeatureVector:
    return FeatureVector(
        symbol=symbol,
        feature_names=["ema_9", "ema_21", "ema_slope_9", "macd_hist", "return_3", "return_5"],
        values=(
            [0.002, 0.001, 0.0002, 0.0002, 0.001, 0.002]
            if bullish
            else [-0.002, -0.001, -0.0002, -0.0002, -0.001, -0.002]
        ),
        quality_score=0.95,
        lookback_bars=100,
    )


class TestBucketGate:
    def test_no_stats_never_blocks(self) -> None:
        app = _make_app()
        assert app._bucket_blocked(_regime_ctx()) is False

    def test_toxic_bucket_blocks(self) -> None:
        app = _make_active_app()
        app._bucket_stats = {_key(): (-5.0, 50)}
        assert app._bucket_blocked(_regime_ctx()) is True

    def test_initial_shadow_mode_never_bucket_blocks(self) -> None:
        app = _make_app()
        app._bucket_stats = {_key(): (-5.0, 50)}
        assert app._bucket_blocked(_regime_ctx()) is False

    def test_small_sample_never_blocks(self) -> None:
        app = _make_app()
        app._bucket_stats = {_key(): (-50.0, 29)}  # terrible but n < 30
        assert app._bucket_blocked(_regime_ctx()) is False

    def test_positive_bucket_not_blocked(self) -> None:
        app = _make_app()
        app._bucket_stats = {_key(): (3.0, 100)}
        assert app._bucket_blocked(_regime_ctx()) is False

    def test_borderline_avg_not_blocked(self) -> None:
        app = _make_app()
        app._bucket_stats = {_key(): (-2.0, 100)}  # equal to threshold, not below
        assert app._bucket_blocked(_regime_ctx()) is False

    def test_other_bucket_does_not_block(self) -> None:
        app = _make_app()
        app._bucket_stats = {_key(regime="BEAR_TREND"): (-10.0, 100)}
        assert app._bucket_blocked(_regime_ctx(regime="BULL_TREND")) is False

    def test_toxic_hour_blocks_when_specific_bucket_is_sparse(self) -> None:
        app = _make_active_app()
        app._hour_stats = {datetime.now(tz=UTC).hour: (-25.0, 61)}

        assert app._bucket_blocked(_regime_ctx()) is True

    def test_hour_fallback_requires_enough_samples(self) -> None:
        app = _make_active_app()
        app._hour_stats = {datetime.now(tz=UTC).hour: (-50.0, 29)}

        assert app._bucket_blocked(_regime_ctx()) is False

    def test_positive_hour_does_not_block(self) -> None:
        app = _make_active_app()
        app._hour_stats = {datetime.now(tz=UTC).hour: (4.0, 61)}

        assert app._bucket_blocked(_regime_ctx()) is False

    def test_disabled_setting_never_blocks(self) -> None:
        app = _make_app(BUCKET_BLOCK_ENABLED=False)
        app._bucket_stats = {_key(): (-10.0, 100)}
        assert app._bucket_blocked(_regime_ctx()) is False

    def test_none_regime_ctx_uses_unknown_bucket(self) -> None:
        app = _make_active_app()
        app._bucket_stats = {("UNKNOWN", "UNKNOWN", datetime.now(tz=UTC).hour): (-10.0, 100)}
        assert app._bucket_blocked(None) is True

    def test_toxic_symbol_side_blocks(self) -> None:
        app = _make_active_app()
        app._symbol_side_stats = {("ADAUSDT", "Buy"): (-5.0, 25)}
        assert app._symbol_side_blocked("ADAUSDT", "Buy") is True

    def test_initial_shadow_mode_never_symbol_side_blocks(self) -> None:
        app = _make_app()
        app._symbol_side_stats = {("ADAUSDT", "Buy"): (-5.0, 25)}
        assert app._symbol_side_blocked("ADAUSDT", "Buy") is False

    def test_symbol_side_small_sample_never_blocks(self) -> None:
        app = _make_app()
        app._symbol_side_stats = {("ADAUSDT", "Buy"): (-50.0, 19)}
        assert app._symbol_side_blocked("ADAUSDT", "Buy") is False

    def test_symbol_side_positive_not_blocked(self) -> None:
        app = _make_app()
        app._symbol_side_stats = {("ADAUSDT", "Buy"): (3.0, 50)}
        assert app._symbol_side_blocked("ADAUSDT", "Buy") is False

    def test_other_symbol_side_does_not_block(self) -> None:
        app = _make_app()
        app._symbol_side_stats = {("ADAUSDT", "Sell"): (-10.0, 100)}
        assert app._symbol_side_blocked("ADAUSDT", "Buy") is False

    def test_symbol_side_disabled_setting_never_blocks(self) -> None:
        app = _make_app(SYMBOL_SIDE_BLOCK_ENABLED=False)
        app._symbol_side_stats = {("ADAUSDT", "Buy"): (-10.0, 100)}
        assert app._symbol_side_blocked("ADAUSDT", "Buy") is False

    def test_negative_strategy_blocks_after_exploration_budget(self) -> None:
        app = _make_active_app()
        app._strategy_stats = {"scalp_micro_v1": (-12.0, 20)}

        assert app._strategy_blocked("scalp_micro_v1") is True

    def test_strategy_remains_exploratory_below_min_samples(self) -> None:
        app = _make_active_app()
        app._strategy_stats = {"scalp_micro_v1": (-50.0, 19)}

        assert app._strategy_blocked("scalp_micro_v1") is False

    def test_positive_strategy_remains_enabled(self) -> None:
        app = _make_active_app()
        app._strategy_stats = {"scalp_micro_v1": (2.0, 50)}

        assert app._strategy_blocked("scalp_micro_v1") is False

    def test_shadow_loss_guard_waits_for_min_closed(self) -> None:
        app = _make_app()

        app._record_shadow_close("XRPUSDT", "SL", -0.3)
        app._record_shadow_close("ADAUSDT", "SL", -0.2)

        assert app._shadow_loss_guard_blocks() is False

    def test_shadow_loss_guard_activates_after_poor_recent_run(self) -> None:
        app = _make_app()

        app._record_shadow_close("XRPUSDT", "SL", -0.3)
        app._record_shadow_close("ADAUSDT", "SL", -0.2)
        app._record_shadow_close("DOGEUSDT", "TP", 0.05)

        assert app._shadow_loss_guard_blocks() is True

    def test_shadow_loss_guard_disabled_setting_never_blocks(self) -> None:
        app = _make_app(SHADOW_LOSS_GUARD_ENABLED=False)

        app._record_shadow_close("XRPUSDT", "SL", -0.3)
        app._record_shadow_close("ADAUSDT", "SL", -0.2)
        app._record_shadow_close("DOGEUSDT", "SL", -0.1)

        assert app._shadow_loss_guard_blocks() is False

    def test_shadow_exit_uses_intrabar_high_for_buy_tp(self) -> None:
        app = _make_app(DEFAULT_LINEAR_TAKER_FEE_RATE=0.0, SCREENER_MAX_SPREAD_BPS=0.0, EXPECTED_SLIPPAGE_PCT=0.0)
        pos = {"side": "Buy", "entry": 100.0, "tp": 102.0, "sl": 99.0}

        hit = TradingApplication._shadow_exit_hit(pos, high=102.5, low=100.5)

        assert hit == ("TP", 102.0)
        assert app._shadow_pnl_pct(pos, hit[1]) == 2.0

    def test_shadow_exit_uses_intrabar_low_for_sell_tp(self) -> None:
        app = _make_app(DEFAULT_LINEAR_TAKER_FEE_RATE=0.0, SCREENER_MAX_SPREAD_BPS=0.0, EXPECTED_SLIPPAGE_PCT=0.0)
        pos = {"side": "Sell", "entry": 100.0, "tp": 98.0, "sl": 101.0}

        hit = TradingApplication._shadow_exit_hit(pos, high=99.5, low=97.5)

        assert hit == ("TP", 98.0)
        assert app._shadow_pnl_pct(pos, hit[1]) == 2.0

    def test_shadow_pnl_deducts_round_trip_costs(self) -> None:
        app = _make_app(DEFAULT_LINEAR_TAKER_FEE_RATE=0.00055, SCREENER_MAX_SPREAD_BPS=8.0, EXPECTED_SLIPPAGE_PCT=0.03)
        pos = {"side": "Buy", "entry": 100.0, "tp": 100.05, "sl": 99.0}

        pnl = app._shadow_pnl_pct(pos, 100.05)

        assert round(pnl, 4) == -0.20

    def test_shadow_exit_is_conservative_when_tp_and_sl_same_candle(self) -> None:
        buy = {"side": "Buy", "entry": 100.0, "tp": 102.0, "sl": 99.0}
        sell = {"side": "Sell", "entry": 100.0, "tp": 98.0, "sl": 101.0}

        assert TradingApplication._shadow_exit_hit(buy, high=103.0, low=98.5) == ("SL", 99.0)
        assert TradingApplication._shadow_exit_hit(sell, high=101.5, low=97.5) == ("SL", 101.0)

    def test_trend_mtf_confirmation_accepts_aligned_buy(self) -> None:
        app = _make_app(TREND_MTF_CONFIRMATION_ENABLED=True, TREND_CONFIRMATION_INTERVALS="5,15")
        vectors = {("XRPUSDT", "5"): _trend_vec("XRPUSDT"), ("XRPUSDT", "15"): _trend_vec("XRPUSDT")}
        app._feature_pipeline = SimpleNamespace(latest=lambda symbol, interval: vectors.get((symbol, interval)))

        assert app._trend_mtf_confirmed("XRPUSDT", "Buy") is True

    def test_trend_mtf_confirmation_accepts_one_aligned_interval(self) -> None:
        app = _make_app(TREND_MTF_CONFIRMATION_ENABLED=True, TREND_CONFIRMATION_INTERVALS="5,15")
        vectors = {
            ("XRPUSDT", "5"): _trend_vec("XRPUSDT"),
            ("XRPUSDT", "15"): _trend_vec("XRPUSDT", bullish=False),
        }
        app._feature_pipeline = SimpleNamespace(latest=lambda symbol, interval: vectors.get((symbol, interval)))

        assert app._trend_mtf_confirmed("XRPUSDT", "Buy") is True

    def test_trend_mtf_confirmation_accepts_one_present_aligned_interval(self) -> None:
        app = _make_app(TREND_MTF_CONFIRMATION_ENABLED=True, TREND_CONFIRMATION_INTERVALS="5,15")
        vectors = {("XRPUSDT", "5"): _trend_vec("XRPUSDT")}
        app._feature_pipeline = SimpleNamespace(latest=lambda symbol, interval: vectors.get((symbol, interval)))

        assert app._trend_mtf_confirmed("XRPUSDT", "Buy") is True

    def test_trend_mtf_confirmation_blocks_when_no_interval_aligns(self) -> None:
        app = _make_app(TREND_MTF_CONFIRMATION_ENABLED=True, TREND_CONFIRMATION_INTERVALS="5,15")
        vectors = {
            ("XRPUSDT", "5"): _trend_vec("XRPUSDT", bullish=False),
            ("XRPUSDT", "15"): _trend_vec("XRPUSDT", bullish=False),
        }
        app._feature_pipeline = SimpleNamespace(latest=lambda symbol, interval: vectors.get((symbol, interval)))

        assert app._trend_mtf_confirmed("XRPUSDT", "Buy") is False

    def test_trend_mtf_confirmation_requires_ema_structure_and_slope_for_buy(self) -> None:
        app = _make_app(TREND_MTF_CONFIRMATION_ENABLED=True, TREND_CONFIRMATION_INTERVALS="5")
        vec = FeatureVector(
            symbol="XRPUSDT",
            feature_names=["ema_9", "ema_21", "ema_slope_9", "macd_hist"],
            values=[-0.002, -0.001, 0.0002, 0.0002],
            quality_score=0.95,
            lookback_bars=100,
        )
        app._feature_pipeline = SimpleNamespace(latest=lambda symbol, interval: vec)

        assert app._trend_mtf_confirmed("XRPUSDT", "Buy") is False

    def test_trend_mtf_confirmation_requires_ema_structure_and_slope_for_sell(self) -> None:
        app = _make_app(TREND_MTF_CONFIRMATION_ENABLED=True, TREND_CONFIRMATION_INTERVALS="5")
        vec = FeatureVector(
            symbol="XRPUSDT",
            feature_names=["ema_9", "ema_21", "ema_slope_9", "macd_hist"],
            values=[0.002, 0.001, -0.0002, -0.0002],
            quality_score=0.95,
            lookback_bars=100,
        )
        app._feature_pipeline = SimpleNamespace(latest=lambda symbol, interval: vec)

        assert app._trend_mtf_confirmed("XRPUSDT", "Sell") is False
