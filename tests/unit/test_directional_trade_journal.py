"""Regression tests for the directional candle outcome resolver."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from trader.domain.enums import MarketRegime, MarketType, OrderSide
from trader.domain.models import FeatureVector, TradeProposal
from trader.storage.directional_trade_journal import DirectionalTradeJournal
from trader.training.labels import LABEL_SCHEMA_VERSION_TPSL, CostModelBps


class _FakeDirectionalJournal(DirectionalTradeJournal):
    """Minimal resolver harness: no database connection is required."""

    def __init__(self, responses: list[list[dict[str, Any]]]) -> None:
        self._responses = iter(responses)
        self.queries: list[str] = []
        self.saved: list[dict[str, Any]] = []

    async def _fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        del args
        self.queries.append(query)
        return next(self._responses)

    async def resolve_prediction_outcomes(self, **kwargs: Any) -> None:
        self.saved.append(kwargs)


class _SignalCaptureJournal(DirectionalTradeJournal):
    """Capture record_signal writes without opening a database connection."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    async def _execute(self, query: str, *args: Any) -> None:
        self.executed.append((query, args))


class _RecoveringSignalCaptureJournal(_SignalCaptureJournal):
    def __init__(self) -> None:
        super().__init__()
        self._last_write_error: str | None = None
        self._failed_insert_once = False

    async def _execute(self, query: str, *args: Any) -> None:
        self.executed.append((query, args))
        if query.lstrip().upper().startswith("INSERT INTO TRADE_SIGNALS") and not self._failed_insert_once:
            self._failed_insert_once = True
            self._last_write_error = 'column "model_decision" of relation "trade_signals" does not exist'
            return
        self._last_write_error = None


class _RecoveringOutcomeJournal(DirectionalTradeJournal):
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self._last_write_error: str | None = None
        self._failed_insert_once = False

    async def _execute(self, query: str, *args: Any) -> None:
        self.executed.append((query, args))
        if query.lstrip().upper().startswith("INSERT INTO PREDICTION_OUTCOMES") and not self._failed_insert_once:
            self._failed_insert_once = True
            self._last_write_error = 'column "gross_return_bps" of relation "prediction_outcomes" does not exist'
            return
        self._last_write_error = None


def _prediction(entry_time: datetime, *, side: str = "Sell") -> dict[str, Any]:
    return {
        "prediction_id": "00000000-0000-0000-0000-000000000001",
        "symbol": "BTCUSDT",
        "strategy_signal": side,
        "entry_time": entry_time,
        "feature_names": json.dumps(["atr_14_pct"]),
        "feature_values": json.dumps([0.01]),
    }


def _complete_path(entry_time: datetime, *, final_close: float = 99.0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for minute in range(1, 16):
        close = final_close if minute == 15 else 100.0 - minute * 0.02
        rows.append(
            {
                "open_time": entry_time + timedelta(minutes=minute),
                "close": close,
                "high": 100.1,
                "low": min(close, 98.5),
            }
        )
    return rows


@pytest.mark.asyncio
async def test_record_signal_accepts_and_persists_model_decision_metadata() -> None:
    journal = _SignalCaptureJournal()
    proposal = TradeProposal(
        strategy_id="trend",
        symbol="XRPUSDT",
        market_type=MarketType.LINEAR,
        side=OrderSide.BUY,
        requested_qty=Decimal("10"),
        entry_price=Decimal("0.5"),
        confidence=0.7,
        regime=MarketRegime.BULL_TREND,
        rationale="test",
    )
    feature_vector = FeatureVector(
        symbol="XRPUSDT",
        values=[1.0],
        feature_names=["momentum"],
        quality_score=1.0,
        lookback_bars=60,
    )
    model_decision = {"model_version": "champion-1", "score": 0.81, "threshold": 0.7}

    await journal.record_signal(
        proposal=proposal,
        feature_vector=feature_vector,
        regime_context=None,
        model_decision=model_decision,
    )

    assert len(journal.executed) == 1
    query, args = journal.executed[0]
    assert "model_decision" in query
    assert json.loads(args[-2]) == model_decision  # args[-1] is blocked_reason


@pytest.mark.asyncio
async def test_record_signal_accepts_and_persists_blocked_reason() -> None:
    journal = _SignalCaptureJournal()
    proposal = TradeProposal(
        strategy_id="trend",
        symbol="ADAUSDT",
        market_type=MarketType.LINEAR,
        side=OrderSide.SELL,
        requested_qty=Decimal("10"),
        entry_price=Decimal("0.5"),
        confidence=0.7,
        regime=MarketRegime.BEAR_TREND,
        rationale="blocked test",
    )

    await journal.record_signal(
        proposal=proposal,
        feature_vector=None,
        regime_context=None,
        blocked_reason="model_gate_canary_blocked",
    )

    assert len(journal.executed) == 1
    query, args = journal.executed[0]
    assert "blocked_reason" in query
    assert args[-1] == "model_gate_canary_blocked"


@pytest.mark.asyncio
async def test_record_signal_repairs_missing_metadata_columns_and_retries() -> None:
    journal = _RecoveringSignalCaptureJournal()
    proposal = TradeProposal(
        strategy_id="trend",
        symbol="LINKUSDT",
        market_type=MarketType.LINEAR,
        side=OrderSide.BUY,
        requested_qty=Decimal("10"),
        entry_price=Decimal("7.5"),
        confidence=0.7,
        regime=MarketRegime.BULL_TREND,
        rationale="repair test",
    )

    await journal.record_signal(
        proposal=proposal,
        feature_vector=None,
        regime_context=None,
        model_decision={"score": 0.7},
        blocked_reason="expectancy_stats_ready",
    )

    assert len(journal.executed) == 4
    assert journal.executed[0][0].lstrip().startswith("INSERT INTO trade_signals")
    assert "ALTER TABLE trade_signals ADD COLUMN IF NOT EXISTS model_decision" in journal.executed[1][0]
    assert "ALTER TABLE trade_signals ADD COLUMN IF NOT EXISTS blocked_reason" in journal.executed[2][0]
    assert journal.executed[3][0].lstrip().startswith("INSERT INTO trade_signals")
    assert journal._last_write_error is None


@pytest.mark.asyncio
async def test_directional_resolve_prediction_outcomes_repairs_missing_extended_columns() -> None:
    journal = _RecoveringOutcomeJournal()

    await journal.resolve_prediction_outcomes(
        prediction_id="00000000-0000-0000-0000-000000000001",
        horizon_minutes=5,
        net_return_bps=10.0,
        max_favorable_excursion_bps=12.0,
        max_adverse_excursion_bps=-3.0,
        label=1,
        gross_return_bps=15.0,
        cost_bps=5.0,
        label_threshold_bps=5.0,
        label_schema_version="directional_net_v2",
    )

    queries = [query for query, _args in journal.executed]
    assert queries[0].lstrip().startswith("INSERT INTO prediction_outcomes")
    assert any("ADD COLUMN IF NOT EXISTS gross_return_bps" in query for query in queries)
    assert any("ADD COLUMN IF NOT EXISTS label_threshold_bps" in query for query in queries)
    assert queries[-1].lstrip().startswith("INSERT INTO prediction_outcomes")
    assert journal._last_write_error is None


@pytest.mark.asyncio
async def test_profitable_sell_is_persisted_as_positive_directional_outcome() -> None:
    entry_time = datetime(2026, 6, 8, 12, 0, tzinfo=UTC)
    journal = _FakeDirectionalJournal(
        [
            [_prediction(entry_time)],
            [{"close": 100.0}],
            _complete_path(entry_time),
        ]
    )

    resolved = await journal.resolve_outcomes_from_candles(
        horizon_minutes=15,
        label_bps_threshold=5.0,
        cost_model=CostModelBps(),
    )

    assert resolved == 1
    assert len(journal.saved) == 1
    saved = journal.saved[0]
    assert saved["gross_return_bps"] == pytest.approx(100.0)
    assert saved["net_return_bps"] == pytest.approx(100.0)
    assert saved["max_favorable_excursion_bps"] == pytest.approx(150.0)
    assert saved["label"] == 1
    assert saved["label_schema_version"] == LABEL_SCHEMA_VERSION_TPSL
    assert "pe.decision IN ('GATE_PASS', 'GATE_BLOCK')" in journal.queries[0]
    assert "OR fs.training_eligible = true" in journal.queries[0]


@pytest.mark.asyncio
async def test_missing_exact_entry_candle_is_not_backfilled_from_older_price() -> None:
    entry_time = datetime(2026, 6, 8, 12, 0, tzinfo=UTC)
    journal = _FakeDirectionalJournal(
        [
            [_prediction(entry_time)],
            [],
        ]
    )

    resolved = await journal.resolve_outcomes_from_candles(
        horizon_minutes=15,
        cost_model=CostModelBps(),
    )

    assert resolved == 0
    assert journal.saved == []
    assert "open_time = $2" in journal.queries[1]
    assert "open_time <= $2" not in journal.queries[1]


@pytest.mark.asyncio
async def test_incomplete_horizon_path_is_not_labelled() -> None:
    entry_time = datetime(2026, 6, 8, 12, 0, tzinfo=UTC)
    incomplete_path = _complete_path(entry_time)[:-1]
    journal = _FakeDirectionalJournal(
        [
            [_prediction(entry_time)],
            [{"close": 100.0}],
            incomplete_path,
        ]
    )

    resolved = await journal.resolve_outcomes_from_candles(
        horizon_minutes=15,
        cost_model=CostModelBps(),
    )

    assert resolved == 0
    assert journal.saved == []


@pytest.mark.asyncio
async def test_horizon_path_must_end_at_exact_requested_minute() -> None:
    entry_time = datetime(2026, 6, 8, 12, 0, tzinfo=UTC)
    shifted_path = _complete_path(entry_time)
    shifted_path[-1] = shifted_path[-1] | {"open_time": entry_time + timedelta(minutes=16)}
    journal = _FakeDirectionalJournal(
        [
            [_prediction(entry_time)],
            [{"close": 100.0}],
            shifted_path,
        ]
    )

    resolved = await journal.resolve_outcomes_from_candles(
        horizon_minutes=15,
        cost_model=CostModelBps(),
    )

    assert resolved == 0
    assert journal.saved == []
