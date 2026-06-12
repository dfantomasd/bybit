"""Tests for the regime-bucket expectancy gate."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from trader.app import TradingApplication
from trader.config import Settings


def _make_app(**overrides) -> TradingApplication:
    app = TradingApplication()
    defaults = {
        "TELEGRAM_ALLOWED_CHAT_IDS": [],
        "BUCKET_BLOCK_ENABLED": True,
        "BUCKET_MIN_SAMPLES": 30,
        "BUCKET_BLOCK_AVG_BPS": -2.0,
    }
    defaults.update(overrides)
    app._settings = Settings(**defaults)
    return app


def _regime_ctx(regime: str = "BULL_TREND", volatility: str = "NORMAL") -> SimpleNamespace:
    return SimpleNamespace(
        regime=SimpleNamespace(value=regime),
        volatility_level=SimpleNamespace(value=volatility),
    )


def _key(regime: str = "BULL_TREND", volatility: str = "NORMAL") -> tuple[str, str, int]:
    return (regime, volatility, datetime.now(tz=UTC).hour)


class TestBucketGate:
    def test_no_stats_never_blocks(self) -> None:
        app = _make_app()
        assert app._bucket_blocked(_regime_ctx()) is False

    def test_toxic_bucket_blocks(self) -> None:
        app = _make_app()
        app._bucket_stats = {_key(): (-5.0, 50)}
        assert app._bucket_blocked(_regime_ctx()) is True

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

    def test_disabled_setting_never_blocks(self) -> None:
        app = _make_app(BUCKET_BLOCK_ENABLED=False)
        app._bucket_stats = {_key(): (-10.0, 100)}
        assert app._bucket_blocked(_regime_ctx()) is False

    def test_none_regime_ctx_uses_unknown_bucket(self) -> None:
        app = _make_app()
        app._bucket_stats = {("UNKNOWN", "UNKNOWN", datetime.now(tz=UTC).hour): (-10.0, 100)}
        assert app._bucket_blocked(None) is True
