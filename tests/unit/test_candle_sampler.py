"""Tests for the per-candle training sampler."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.app import TradingApplication
from trader.config import Settings
from trader.data.candles import Candle, CandleStore
from trader.domain.models import FeatureVector

_SYMBOL = "TESTUSDT"


def _make_app(**overrides) -> TradingApplication:
    app = TradingApplication()
    defaults = {"TELEGRAM_ALLOWED_CHAT_IDS": [], "CANDLE_SAMPLING_ENABLED": True}
    defaults.update(overrides)
    app._settings = Settings(**defaults)
    journal = MagicMock()
    journal.is_enabled = True
    journal.record_feature_snapshot = AsyncMock(return_value=str(uuid.uuid4()))
    journal.record_prediction_event = AsyncMock(return_value=str(uuid.uuid4()))
    app._trade_journal = journal
    store = CandleStore(max_bars=10)
    store.add(
        _SYMBOL,
        "1",
        Candle(
            open_time=datetime(2026, 6, 12, 4, 0, tzinfo=UTC),
            open=1.0,
            high=1.1,
            low=0.9,
            close=1.05,
            volume=10,
            confirm=True,
        ),
    )
    app._candle_store = store
    return app


def _vec(ema9: float = 0.01, ema21: float = -0.01) -> FeatureVector:
    return FeatureVector(
        symbol=_SYMBOL,
        timestamp=datetime.now(tz=UTC),
        values=[ema9, ema21, 0.5],
        feature_names=["ema_9", "ema_21", "rsi_14"],
        quality_score=1.0,
        lookback_bars=60,
    )


class TestCandleSampler:
    @pytest.mark.asyncio
    async def test_records_buy_sample_when_fast_above_slow(self) -> None:
        app = _make_app()
        await app._sample_confirmed_candle(_SYMBOL, "1", _vec(ema9=0.01, ema21=-0.01))
        app._trade_journal.record_feature_snapshot.assert_awaited_once()
        event_kwargs = app._trade_journal.record_prediction_event.await_args.kwargs
        assert event_kwargs["strategy_signal"] == "Buy"
        assert event_kwargs["decision"] == "SHADOW_CANDLE"
        assert event_kwargs["model_version"] == "RULE_BASELINE_V1"

    @pytest.mark.asyncio
    async def test_records_sell_sample_when_fast_below_slow(self) -> None:
        app = _make_app()
        await app._sample_confirmed_candle(_SYMBOL, "1", _vec(ema9=-0.02, ema21=0.0))
        assert app._trade_journal.record_prediction_event.await_args.kwargs["strategy_signal"] == "Sell"

    @pytest.mark.asyncio
    async def test_disabled_setting_skips(self) -> None:
        app = _make_app(CANDLE_SAMPLING_ENABLED=False)
        await app._sample_confirmed_candle(_SYMBOL, "1", _vec())
        app._trade_journal.record_feature_snapshot.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_one_sample_per_candle(self) -> None:
        app = _make_app()
        await app._sample_confirmed_candle(_SYMBOL, "1", _vec())
        await app._sample_confirmed_candle(_SYMBOL, "1", _vec())  # same candle re-confirm
        assert app._trade_journal.record_prediction_event.await_count == 1

    @pytest.mark.asyncio
    async def test_non_1m_interval_skipped(self) -> None:
        app = _make_app()
        await app._sample_confirmed_candle(_SYMBOL, "5", _vec())
        app._trade_journal.record_feature_snapshot.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_ema_features_skipped(self) -> None:
        app = _make_app()
        vec = FeatureVector(
            symbol=_SYMBOL,
            timestamp=datetime.now(tz=UTC),
            values=[0.5],
            feature_names=["rsi_14"],
            quality_score=1.0,
            lookback_bars=60,
        )
        await app._sample_confirmed_candle(_SYMBOL, "1", vec)
        app._trade_journal.record_feature_snapshot.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_journal_failure_does_not_raise(self) -> None:
        app = _make_app()
        app._trade_journal.record_feature_snapshot = AsyncMock(side_effect=RuntimeError("db down"))
        await app._sample_confirmed_candle(_SYMBOL, "1", _vec())  # must not raise

    @pytest.mark.asyncio
    async def test_challenger_shadow_gate_event_recorded(self) -> None:
        from trader.ml.challenger import ModelPrediction

        app = _make_app()
        registry = MagicMock()
        registry.score_shadow = MagicMock(
            return_value=ModelPrediction(
                score=0.9,
                label=1,
                confidence=0.9,
                model_version="v_test_challenger",
            )
        )
        app._model_registry = registry
        await app._sample_confirmed_candle(_SYMBOL, "1", _vec())
        assert app._trade_journal.record_prediction_event.await_count == 2
        shadow_kwargs = app._trade_journal.record_prediction_event.await_args_list[1].kwargs
        assert shadow_kwargs["model_version"] == "v_test_challenger"
        assert shadow_kwargs["decision"] == "GATE_PASS"
        assert shadow_kwargs["metadata"]["source"] == "candle_sampler_shadow"

    @pytest.mark.asyncio
    async def test_challenger_shadow_gate_block_below_threshold(self) -> None:
        from trader.ml.challenger import ModelPrediction

        app = _make_app()
        registry = MagicMock()
        registry.score_shadow = MagicMock(
            return_value=ModelPrediction(
                score=0.10,
                label=0,
                confidence=0.90,
                model_version="v_test_challenger",
            )
        )
        app._model_registry = registry
        await app._sample_confirmed_candle(_SYMBOL, "1", _vec())
        shadow_kwargs = app._trade_journal.record_prediction_event.await_args_list[1].kwargs
        assert shadow_kwargs["decision"] == "GATE_BLOCK"

    def test_candle_sampler_shadow_gate_uses_adaptive_observational_threshold(self) -> None:
        app = _make_app(CANDLE_SAMPLER_SHADOW_GATE_MIN_PASS_RATE_PCT=20.0)

        strict_threshold, strict_source = app._modules.signal_policy.candle_sampler_shadow_gate_threshold(0.42)
        assert strict_threshold == pytest.approx(0.52)
        assert strict_source == "strict"

        app._candle_sampler_shadow_scores.clear()
        for score in [0.31 + (i * 0.005) for i in range(20)]:
            threshold, source = app._modules.signal_policy.candle_sampler_shadow_gate_threshold(score)

        assert source == "adaptive"
        assert threshold < 0.52
        assert threshold >= 0.31

    @pytest.mark.asyncio
    async def test_no_challenger_records_only_baseline(self) -> None:
        app = _make_app()
        registry = MagicMock()
        registry.score_shadow = MagicMock(return_value=None)
        app._model_registry = registry
        await app._sample_confirmed_candle(_SYMBOL, "1", _vec())
        assert app._trade_journal.record_prediction_event.await_count == 1
