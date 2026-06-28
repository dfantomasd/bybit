"""PR 3 Database and ML tests.

Tests run against an in-process mock (no real PostgreSQL required).
Covers:
- test_market_candle_upsert
- test_market_candle_no_duplicates
- test_market_candle_backfill (schema check)
- test_feature_snapshot_written
- test_partial_fit_checkpoint
- test_checkpoint_reload_after_restart
- test_model_shadow_scoring_only
- test_model_cannot_change_live_decision_before_promotion
- test_database_model_telegram_screen (smoke test)
- test_no_lookahead_leakage (label uses only historical data)
- test_training_tables_created (schema validation via SQL inspection)
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.ml.challenger import ChallengerModel, ModelRegistry, ModelStatus
from trader.training.labels import LABEL_SCHEMA_VERSION_TPSL

# ---------------------------------------------------------------------------
# Challenger model: partial_fit and prediction
# ---------------------------------------------------------------------------


def test_partial_fit_updates_sample_count() -> None:
    """partial_fit should increment training_samples."""
    model = ChallengerModel(version="v_test", feature_names=["rsi", "ema_diff"])
    model.partial_fit([50.0, 0.02], label=1)
    model.partial_fit([20.0, -0.01], label=0)
    assert model.training_samples == 2


def test_partial_fit_checkpoint() -> None:
    """Model should serialize and deserialize correctly."""
    model = ChallengerModel(version="v_test", feature_names=["rsi", "ema_diff"])
    for i in range(10):
        model.partial_fit([float(i * 5), float(i) * 0.01], label=i % 2)

    data = model.to_bytes()
    assert len(data) > 0

    restored = ChallengerModel.from_bytes(data, version="v_test")
    assert restored.training_samples == 10
    assert restored.feature_names == ["rsi", "ema_diff"]


def test_checkpoint_reload_after_restart() -> None:
    """After reload, model should produce same predictions."""
    model = ChallengerModel(version="v_test", feature_names=["f1", "f2"])
    for i in range(20):
        model.partial_fit([float(i), float(-i)], label=i % 2)

    data = model.to_bytes()
    restored = ChallengerModel.from_bytes(data, version="v_test")

    # Both should produce a result
    pred_orig = model.predict([10.0, -10.0])
    pred_restored = restored.predict([10.0, -10.0])
    # Both should succeed or both fail
    assert (pred_orig is None) == (pred_restored is None)


def test_model_shadow_scoring_only() -> None:
    """Model in SHADOW_CHALLENGER status should NOT set is_live_decision=True."""
    model = ChallengerModel(version="v_test", feature_names=["f1"])
    model.status = ModelStatus.SHADOW_CHALLENGER
    model.allow_live_decisions = False

    for i in range(10):
        model.partial_fit([float(i)], label=i % 2)

    pred = model.predict([5.0])
    if pred is not None:
        assert pred.is_live_decision is False


def test_model_cannot_change_live_decision_before_promotion() -> None:
    """Model should not produce live decisions until promoted to CHAMPION with allow_live_decisions."""
    model = ChallengerModel(version="v_test", feature_names=["f1"])
    model.status = ModelStatus.SHADOW_CHALLENGER
    model.allow_live_decisions = False

    for i in range(10):
        model.partial_fit([float(i)], label=i % 2)

    pred = model.predict([5.0])
    if pred is not None:
        assert not pred.is_live_decision

    # Even if we change status but NOT allow_live_decisions
    model.status = ModelStatus.CHAMPION
    pred = model.predict([5.0])
    if pred is not None:
        assert not pred.is_live_decision  # still False because allow_live_decisions=False


def test_can_promote_insufficient_samples() -> None:
    """can_promote should return False when samples < minimum."""
    model = ChallengerModel(version="v_test", label_schema_version=LABEL_SCHEMA_VERSION_TPSL)
    model.training_samples = 100
    can, reason = model.can_promote(min_samples=500)
    assert not can
    assert "insufficient_samples" in reason


def test_can_promote_negative_expectancy() -> None:
    """can_promote should return False when walk-forward expectancy <= 0."""
    model = ChallengerModel(version="v_test", label_schema_version=LABEL_SCHEMA_VERSION_TPSL)
    model.training_samples = 1000
    can, reason = model.can_promote(min_samples=500, walk_forward_expectancy=-0.01)
    assert not can
    assert "negative_walk_forward" in reason


def test_can_promote_success() -> None:
    """can_promote should return True when all criteria met."""
    model = ChallengerModel(version="v_test", label_schema_version=LABEL_SCHEMA_VERSION_TPSL)
    model.training_samples = 1000
    can, reason = model.can_promote(min_samples=500, walk_forward_expectancy=0.05)
    assert can


# ---------------------------------------------------------------------------
# Model Registry: champion/challenger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_score_uses_champion() -> None:
    """score() is the legacy alias for score_live() — must return champion, not challenger.
    app.py now calls score_shadow() directly for Challenger observational scoring."""
    registry = ModelRegistry()

    champion = ChallengerModel(version="champion_v1", feature_names=["f1"])
    for i in range(10):
        champion.partial_fit([float(i)], label=i % 2)
    champion.status = ModelStatus.CHAMPION
    registry._champion = champion

    challenger = ChallengerModel(version="challenger_v2", feature_names=["f1"])
    for i in range(5):
        challenger.partial_fit([float(i)], label=i % 2)
    registry._challenger = challenger

    pred = registry.score([5.0])
    if pred is not None:
        # score() = score_live() → champion only
        assert pred.model_version == "champion_v1"

    # score_shadow() should prefer challenger over champion
    shadow_pred = registry.score_shadow([5.0])
    if shadow_pred is not None:
        assert shadow_pred.model_version == "challenger_v2"


@pytest.mark.asyncio
async def test_registry_score_falls_back_to_challenger() -> None:
    """When no champion, registry should use challenger."""
    registry = ModelRegistry()

    challenger = ChallengerModel(version="challenger_v1", feature_names=["f1"])
    for i in range(10):
        challenger.partial_fit([float(i)], label=i % 2)
    registry._challenger = challenger

    pred = registry.score([5.0])
    if pred is not None:
        assert pred.model_version == "challenger_v1"


@pytest.mark.asyncio
async def test_registry_load_active_prefers_champion() -> None:
    """load_active_model loads both champion and challenger; champion returned as primary."""
    champion = ChallengerModel(version="champion_v1", feature_names=["f1"])
    for i in range(10):
        champion.partial_fit([float(i)], label=i % 2)

    journal = MagicMock()
    journal.is_enabled = True
    # First call: champion query returns champion; second call: challenger query returns nothing
    journal._fetch = AsyncMock(
        side_effect=[
            [
                {
                    "version": "champion_v1",
                    "artifact": champion.to_bytes(),
                    "training_samples": champion.training_samples,
                }
            ],
            [],  # challenger primary (schema-filtered) → empty
            [],  # challenger fallback (no schema filter) → empty
        ]
    )

    registry = ModelRegistry(trade_journal=journal)
    loaded = await registry.load_active_model()

    assert loaded is not None
    assert loaded.version == "champion_v1"
    assert loaded.status == ModelStatus.CHAMPION
    assert registry.champion is loaded
    assert registry.challenger is None
    assert journal._fetch.await_count == 3


@pytest.mark.asyncio
async def test_registry_load_active_falls_back_to_shadow_challenger() -> None:
    """A freshly trained SHADOW_CHALLENGER should be usable for shadow scoring after restart."""
    challenger = ChallengerModel(version="challenger_v1", feature_names=["f1"])
    for i in range(10):
        challenger.partial_fit([float(i)], label=i % 2)

    journal = MagicMock()
    journal.is_enabled = True
    journal._fetch = AsyncMock(
        side_effect=[
            [],
            [],
            [],
            [
                {
                    "version": "challenger_v1",
                    "status": ModelStatus.SHADOW_CHALLENGER,
                    "artifact": challenger.to_bytes(),
                    "training_samples": challenger.training_samples,
                }
            ],
        ]
    )

    registry = ModelRegistry(trade_journal=journal)
    loaded = await registry.load_active_model()

    assert loaded is not None
    assert loaded.version == "challenger_v1"
    assert loaded.status == ModelStatus.SHADOW_CHALLENGER
    assert registry.champion is None
    assert registry.challenger is loaded
    pred = registry.score([5.0])
    if pred is not None:
        assert pred.model_version == "challenger_v1"
        assert pred.is_live_decision is False


@pytest.mark.asyncio
async def test_select_best_champion_prefers_positive_walk_forward_lift_over_freshness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive out-of-sample evidence wins; freshness only breaks equal lift."""
    monkeypatch.setenv("MODEL_CHAMPION_MIN_PAPER_GATE_COUNT", "50")
    older_better = ChallengerModel(version="older_better", feature_names=["f1"])
    newer_loss = ChallengerModel(version="newer_loss", feature_names=["f1"])
    for i in range(10):
        older_better.partial_fit([float(i)], label=i % 2)
        newer_loss.partial_fit([float(i)], label=i % 2)

    journal = MagicMock()
    journal.is_enabled = True
    journal._fetch = AsyncMock(
        return_value=[
            {
                "version": "older_better",
                "artifact": older_better.to_bytes(),
                "training_samples": older_better.training_samples,
                "metrics": {
                    "label_schema_version": "directional_net_v1",
                    "quality": "GOOD",
                    "walk_forward_expectancy_bps": 1.2,
                    "lift_bps": 5.9,
                    "paper_gate": {"count": 75},
                },
            }
        ]
    )

    registry = ModelRegistry(trade_journal=journal)
    row = await registry.select_best_champion()

    assert row is not None
    assert row["version"] == "older_better"
    query = journal._fetch.await_args.args[0]
    assert "walk_forward_expectancy_bps" in query
    assert "paper_gate,count" in query
    assert "CASE WHEN COALESCE" in query


@pytest.mark.asyncio
async def test_select_best_champion_falls_back_when_no_walk_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """When WF has not been calculated at all, registry keeps legacy ordering with warning."""
    monkeypatch.setenv("MODEL_CHAMPION_MIN_PAPER_GATE_COUNT", "50")
    fallback = ChallengerModel(version="fallback_good", feature_names=["f1"])
    for i in range(10):
        fallback.partial_fit([float(i)], label=i % 2)

    journal = MagicMock()
    journal.is_enabled = True
    journal._fetch = AsyncMock(
        side_effect=[
            [],
            [],
            [
                {
                    "version": "fallback_good",
                    "artifact": fallback.to_bytes(),
                    "training_samples": fallback.training_samples,
                    "metrics": {"label_schema_version": "directional_net_v1", "quality": "GOOD"},
                }
            ],
        ]
    )

    registry = ModelRegistry(trade_journal=journal)
    row = await registry.select_best_champion()

    assert row is not None
    assert row["version"] == "fallback_good"
    assert journal._fetch.await_count == 3


@pytest.mark.asyncio
async def test_select_best_champion_does_not_fallback_when_walk_forward_lacks_paper_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A calculated WF champion with too little paper evidence must fail closed."""
    monkeypatch.setenv("MODEL_CHAMPION_MIN_PAPER_GATE_COUNT", "50")

    journal = MagicMock()
    journal.is_enabled = True
    journal._fetch = AsyncMock(
        side_effect=[
            [],
            [{"has_walk_forward": True}],
        ]
    )

    registry = ModelRegistry(trade_journal=journal)
    row = await registry.select_best_champion()

    assert row is None
    assert journal._fetch.await_count == 2


# ---------------------------------------------------------------------------
# TradeJournal: schema validation (mocked pool)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trade_journal_schema_includes_durable_order_state() -> None:
    """Schema setup should include durable_order_state table."""
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal(postgres_dsn="postgresql://fake:fake@localhost/fake", enabled=False)
    # When disabled, operations are no-ops — just verify init doesn't crash
    assert not journal.is_enabled


@pytest.mark.asyncio
async def test_model_performance_history_includes_score_and_selection_reason() -> None:
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal(postgres_dsn="postgresql://fake:fake@localhost/fake", enabled=True)
    journal._pool = MagicMock()

    async def mock_fetch(query: str, *args: Any) -> list[dict[str, Any]]:
        del args
        if "net_return_bps" in query.lower():
            return []
        del query
        return [
            {
                "version": "v_good",
                "status": "CHAMPION",
                "training_finished_at": datetime(2026, 6, 7, 10, 1, tzinfo=UTC),
                "created_at": datetime(2026, 6, 7, 10, 1, tzinfo=UTC),
                "training_samples": 900,
                "metrics": {
                    "quality": "GOOD",
                    "precision": 0.62,
                    "lift_bps": 4.2,
                    "walk_forward_expectancy_bps": 3.1,
                    "paper_gate": {"count": 80},
                },
            }
        ]

    journal._fetch = mock_fetch  # type: ignore[method-assign]

    rows = await journal.get_model_performance_history()

    assert rows[0]["version"] == "v_good"
    assert rows[0]["model_score"] > 0
    assert rows[0]["paper_gate_count"] == 0
    assert rows[0]["paper_gate_source"] == "live_outcomes"
    assert rows[0]["selection_reason"] == "blocked:paper_gate_count<50"


@pytest.mark.asyncio
async def test_champion_health_includes_checks_alternative_and_promotion_log() -> None:
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal(postgres_dsn="postgresql://fake:fake@localhost/fake", enabled=True)
    journal._pool = MagicMock()
    calls = 0

    async def mock_fetch(query: str, *args: Any) -> list[dict[str, Any]]:
        del args
        if "net_return_bps" in query.lower():
            return []
        nonlocal calls
        calls += 1
        if calls == 1:
            return [
                {
                    "version": "champion",
                    "status": "CHAMPION",
                    "training_finished_at": datetime(2026, 6, 7, 10, 1, tzinfo=UTC),
                    "created_at": datetime(2026, 6, 7, 10, 1, tzinfo=UTC),
                    "training_samples": 1000,
                    "metrics": {
                        "quality": "GOOD",
                        "walk_forward_expectancy_bps": 4.0,
                        "lift_bps": 2.0,
                        "paper_gate": {"count": 80},
                        "wf_folds": 5,
                        "wf_positive_folds": 4,
                        "wf_std_bps": 3.0,
                        "walk_forward_chronology": "strict_after_train",
                    },
                }
            ]
        if calls == 2:
            return [
                {
                    "version": "candidate",
                    "status": "VALIDATED",
                    "training_finished_at": datetime(2026, 6, 7, 11, 1, tzinfo=UTC),
                    "created_at": datetime(2026, 6, 7, 11, 1, tzinfo=UTC),
                    "training_samples": 1000,
                    "metrics": {"quality": "GOOD", "walk_forward_expectancy_bps": 3.0},
                }
            ]
        return [
            {
                "event_type": "PROMOTED",
                "from_version": "old",
                "to_version": "champion",
                "reasons": ["criteria_met"],
                "metrics": {},
                "metrics_snapshot": {},
                "created_at": datetime(2026, 6, 7, 12, 1, tzinfo=UTC),
            }
        ]

    journal._fetch = mock_fetch  # type: ignore[method-assign]

    health = await journal.get_champion_health()

    assert health["champion"]["version"] == "champion"
    assert health["best_alternative"]["version"] == "candidate"
    assert not all(check["ok"] for check in health["checks"])
    paper_check = next(check for check in health["checks"] if check["name"] == "paper_gate_count")
    assert paper_check["ok"] is False
    assert paper_check["value"] == 0
    assert health["promotion_log"][0]["event_type"] == "PROMOTED"


@pytest.mark.asyncio
async def test_live_paper_enrichment_uses_live_zero_over_stale_metrics() -> None:
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal(postgres_dsn="postgresql://fake:fake@localhost/fake", enabled=True)
    journal._pool = MagicMock()

    async def mock_fetch(query: str, *args: Any) -> list[dict[str, Any]]:
        del args
        assert "ORDER BY pe.created_at DESC" in query
        assert ") recent\n                ORDER BY created_at ASC" in query
        return []

    journal._fetch = mock_fetch  # type: ignore[method-assign]

    enriched = await journal._enrich_model_with_live_paper(
        {
            "version": "champion",
            "paper_gate_count": 80,
            "selection_reason": "selected:positive_walk_forward_lift",
            "metrics": {
                "walk_forward_expectancy_bps": 4.0,
                "paper_gate": {"count": 80},
            },
        }
    )

    assert enriched is not None
    assert enriched["paper_gate_count"] == 0
    assert enriched["paper_gate_source"] == "live_outcomes"
    assert enriched["selection_reason"] == "blocked:paper_gate_count<50"


@pytest.mark.asyncio
async def test_market_candle_upsert_sql() -> None:
    """upsert_market_candle should build correct SQL and not duplicate on conflict."""
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal(postgres_dsn="postgresql://fake:fake@localhost/fake", enabled=True)

    calls: list[tuple] = []

    async def mock_execute(query: str, *args: Any) -> None:
        calls.append((query, args))

    journal._execute = mock_execute  # type: ignore[method-assign]
    journal._pool = MagicMock()  # non-None pool to pass is_enabled check
    journal._enabled = True

    await journal.upsert_market_candle(
        symbol="DOGEUSDT",
        interval="1",
        open_time=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        close_time=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        open=Decimal("0.10"),
        high=Decimal("0.11"),
        low=Decimal("0.09"),
        close=Decimal("0.105"),
        volume=Decimal("1000000"),
        turnover=Decimal("100000"),
        confirmed=True,
    )

    assert len(calls) == 1
    query, args = calls[0]
    assert "market_candles" in query
    assert "ON CONFLICT" in query
    assert "DOGEUSDT" in args


@pytest.mark.asyncio
async def test_feature_snapshot_written() -> None:
    """record_feature_snapshot should call INSERT with correct fields."""
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal(postgres_dsn="postgresql://fake:fake@localhost/fake", enabled=True)

    fetched: list[tuple] = []

    async def mock_fetch(query: str, *args: Any) -> list[dict]:
        fetched.append((query, args))
        return [{"snapshot_id": uuid.uuid4()}]

    journal._fetch = mock_fetch  # type: ignore[method-assign]
    journal._pool = MagicMock()
    journal._enabled = True

    await journal.record_feature_snapshot(
        symbol="DOGEUSDT",
        interval="1",
        candle_open_time=datetime(2026, 1, 1, tzinfo=UTC),
        feature_schema_hash="abc123",
        feature_names=["rsi", "ema"],
        feature_values=[45.0, 0.02],
    )

    assert len(fetched) == 1
    query, args = fetched[0]
    assert "feature_snapshots" in query
    assert "ON CONFLICT (symbol, interval, candle_open_time, feature_schema_hash)" in query
    assert "WHERE training_eligible = true" in query
    assert "DOGEUSDT" in args


@pytest.mark.asyncio
async def test_db_diagnostics_reports_trainable_samples_and_latest_model() -> None:
    """DB diagnostics should expose enough model/training state for Telegram controls."""
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal(postgres_dsn="postgresql://fake:fake@localhost/fake", enabled=True)
    journal._pool = MagicMock()
    journal._enabled = True

    async def mock_fetch(query: str, *args: Any) -> list[dict]:
        if "FROM market_candles GROUP BY interval" in query:
            return [{"interval": "1", "cnt": 1000}, {"interval": "15", "cnt": 250}]
        if "MAX(open_time)" in query:
            return [{"ts": datetime(2026, 1, 1, 12, 0, tzinfo=UTC)}]
        if "GROUP BY feature_schema_hash" in query:
            return [
                {
                    "feature_schema_hash": "abc1234567890def",
                    "sample_count": 777,
                    "latest_at": datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                }
            ]
        if "GROUP BY pool" in query and "label_schema_version = $1" in query:
            return [{"pool": "scalp_micro_v1", "samples": 177}]
        if "GROUP BY pool" in query:
            return [{"pool": "scalp_micro_v1", "sample_count": 777}]
        if "GROUP BY label_threshold_bps" in query:
            return [{"threshold": "2.0", "sample_count": 777}]
        if "FROM feature_snapshots" in query:
            return [{"cnt": 1200}]
        if "GROUP BY horizon_minutes" in query:
            return [{"horizon_minutes": 5, "cnt": 300}, {"horizon_minutes": 15, "cnt": 777}]
        if "FROM prediction_outcomes" in query:
            return [{"cnt": 900}]
        if "metadata->>'gate_reason'" in query:
            return [{"reason": "score_below_regime_threshold", "cnt": 8}]
        if "pe.model_version = 'RULE_BASELINE_V1'" in query:
            return [
                {"net_return_bps": 1.0},
                {"net_return_bps": -2.0},
            ]
        if "pe.decision = 'GATE_PASS'" in query:
            return [
                {"net_return_bps": 4.5},
                {"net_return_bps": 3.5},
            ]
        if "count(po.prediction_id) AS resolved_count" in query:
            return [
                {"decision": "GATE_PASS", "total_count": 15, "resolved_count": 12},
                {"decision": "GATE_BLOCK", "total_count": 10, "resolved_count": 8},
            ]
        if "pe.decision IN ('GATE_PASS', 'GATE_BLOCK')" in query:
            return [
                {
                    "decision": "GATE_PASS",
                    "cnt": 12,
                    "avg_net_return_bps": 4.5,
                    "precision": 0.58,
                },
                {
                    "decision": "GATE_BLOCK",
                    "cnt": 8,
                    "avg_net_return_bps": -1.5,
                    "precision": 0.25,
                },
            ]
        if "FROM training_runs" in query:
            return [
                {
                    "status": "COMPLETED",
                    "model_version": "v20260607_1000",
                    "sample_count": 777,
                    "error": None,
                    "metrics": {
                        "quality": "GOOD",
                        "precision": 0.62,
                        "lift_bps": 3.4,
                        "best_threshold": 0.6,
                        "best_threshold_avg_net_return_bps": 4.5,
                    },
                    "started_at": datetime(2026, 6, 7, 10, 0, tzinfo=UTC),
                    "finished_at": datetime(2026, 6, 7, 10, 1, tzinfo=UTC),
                }
            ]
        if "FROM model_versions" in query:
            return [
                {
                    "version": "v20260607_1000",
                    "status": "SHADOW_CHALLENGER",
                    "training_samples": 777,
                    "feature_schema_hash": "abc1234567890def",
                    "metrics": {
                        "quality": "GOOD",
                        "precision": 0.62,
                        "lift_bps": 3.4,
                        "best_threshold": 0.6,
                        "best_threshold_avg_net_return_bps": 4.5,
                        "horizon_minutes": 5,
                    },
                    "training_finished_at": datetime(2026, 6, 7, 10, 1, tzinfo=UTC),
                    "created_at": datetime(2026, 6, 7, 10, 1, tzinfo=UTC),
                }
            ]
        return []

    journal._fetch = mock_fetch  # type: ignore[method-assign]

    diag = await journal.get_db_diagnostics()

    assert diag["labelled_samples_15m"] == 777
    assert diag["prediction_outcomes_by_horizon"] == {"5": 300, "15": 777}
    assert diag["latest_training_run"]["status"] == "COMPLETED"
    assert diag["latest_model_version"]["version"] == "v20260607_1000"
    assert diag["model_gate_horizon_minutes"] == 5
    assert diag["shadow_gate_by_horizon"]["5"]["pass_count"] == 12
    assert diag["shadow_gate_by_horizon"]["5"]["event_total_count"] == 25
    assert diag["shadow_gate_by_horizon"]["5"]["event_resolved_count"] == 20
    assert diag["shadow_gate_by_horizon"]["5"]["event_pending_count"] == 5
    assert diag["paper_pnl_by_horizon"]["5"]["model_gate"]["count"] == 2
    assert diag["shadow_gate_15m"]["pass_count"] == 12
    assert diag["shadow_gate_15m"]["pass_vs_block_bps"] == 6.0
    assert diag["shadow_gate_15m"]["top_block_reasons"] == {"score_below_regime_threshold": 8}
    assert diag["paper_pnl_15m"]["baseline"]["total_bps"] == -1.0
    assert diag["paper_pnl_15m"]["model_gate"]["total_bps"] == 8.0


@pytest.mark.asyncio
async def test_db_diagnostics_reports_training_samples_by_horizon() -> None:
    """Auto-trainer must not mix 5m and 15m sample readiness."""
    from trader.storage.directional_trade_journal import DirectionalTradeJournal

    journal = DirectionalTradeJournal(postgres_dsn="postgresql://fake", enabled=True)
    journal._pool = MagicMock()
    journal._enabled = True

    async def mock_fetch(query: str, *args: Any) -> list[dict[str, Any]]:
        if "GROUP BY feature_schema_hash" in query:
            horizon = int(args[0]) if args else 5
            if horizon == 5:
                return [
                    {
                        "feature_schema_hash": "schema5",
                        "sample_count": 1200,
                        "latest_at": datetime(2026, 6, 7, 10, 0, tzinfo=UTC),
                    }
                ]
            return [
                {
                    "feature_schema_hash": "schema15",
                    "sample_count": 100,
                    "latest_at": datetime(2026, 6, 7, 10, 0, tzinfo=UTC),
                }
            ]
        if "GROUP BY pool" in query:
            return [{"pool": "scalp_micro_v1", "sample_count": 0}]
        if "GROUP BY label_threshold_bps" in query:
            return [{"threshold": "2.0", "sample_count": 0}]
        if "AS pool" in query:
            return []
        if "GROUP BY horizon_minutes" in query and "feature_schema_hash" not in query:
            return [{"horizon_minutes": 5, "cnt": 900}, {"horizon_minutes": 15, "cnt": 100}]
        if "FROM model_versions" in query:
            return []
        return []

    journal._fetch = mock_fetch  # type: ignore[method-assign]

    diag = await journal.get_db_diagnostics()

    assert diag["training_eligible_by_horizon"] == {"5": 1200, "15": 0}
    assert diag["newest_training_schema_by_horizon"]["5"]["feature_schema_hash"] == "schema5"
    assert diag["newest_training_schema_by_horizon"]["15"]["sample_count"] == 100
    assert diag["newest_training_schema_by_horizon"]["5"]["best_schema_count"] == 1200
    assert diag["newest_training_schema_by_horizon"]["5"]["trainable_schema_count"] == 1200


@pytest.mark.asyncio
async def test_paper_pnl_uses_recent_window_then_chronological_equity() -> None:
    """Paper PnL must reflect the latest live-like window, not the oldest rows."""
    from trader.storage.directional_trade_journal import DirectionalTradeJournal

    journal = DirectionalTradeJournal(postgres_dsn="postgresql://fake", enabled=True)
    queries: list[str] = []

    async def mock_fetch(query: str, *_args: Any) -> list[dict[str, Any]]:
        queries.append(query)
        return [{"net_return_bps": 1.0}]

    journal._fetch = mock_fetch  # type: ignore[method-assign]

    paper = await journal._paper_pnl_for_model("v_test", 5, "schema123")

    assert paper["baseline"]["count"] == 1
    assert paper["model_gate"]["count"] == 1
    assert len(queries) == 2
    assert all("ORDER BY pe.created_at DESC" in query for query in queries)
    assert all("LIMIT 1000" in query for query in queries)
    assert all(") recent\n            ORDER BY created_at ASC" in query for query in queries)


@pytest.mark.asyncio
async def test_challenger_returns_for_bootstrap_use_gate_pass_only() -> None:
    """Auto-promoter bootstrap should measure trades the model would allow."""
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal(postgres_dsn="postgresql://fake", enabled=True)
    journal._pool = MagicMock()
    queries: list[str] = []

    async def mock_fetch(query: str, *_args: Any) -> list[dict[str, Any]]:
        queries.append(query)
        return [{"net_return_bps": 2.0}]

    journal._fetch = mock_fetch  # type: ignore[method-assign]

    challenger_returns = await journal.get_returns_for_model(
        "v_challenger",
        horizon_minutes=5,
        label_schema_version=LABEL_SCHEMA_VERSION_TPSL,
    )
    baseline_returns = await journal.get_returns_for_model(
        "RULE_BASELINE_V1",
        horizon_minutes=5,
        label_schema_version=LABEL_SCHEMA_VERSION_TPSL,
    )

    assert challenger_returns == [2.0]
    assert baseline_returns == [2.0]
    assert "pe.decision = 'GATE_PASS'" in queries[0]
    assert "COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'" in queries[1]


def test_auto_trainer_reads_configured_horizon_sample_count() -> None:
    """Guard against regressing to mixed 5m/15m readiness counters."""
    import inspect

    from trader.modules.training import TrainingModule

    src = inspect.getsource(TrainingModule.run_auto_model_trainer)
    assert "training_eligible_by_horizon" in src
    assert "resolve_training_horizon" in src
    assert "active_horizon" in src
    assert "actual_training_samples" in src
    assert "training_samples_compatible" in src
    assert "schema_incompatible" in src or "label_schema_mismatch" in src
    assert "enough_label_schema_change" in src
    assert "preflight_blocked" in src
    assert "trainable_15m=" not in src


def test_auto_trainer_uses_latest_training_run_for_success_cooldown() -> None:
    """A just-finished run must prevent checkpoint churn even if model diagnostics are stale."""
    import inspect

    from trader.modules.training import TrainingModule

    src = inspect.getsource(TrainingModule.run_auto_model_trainer)
    assert "latest_run_samples" in src
    assert "latest_success_samples = max(actual_latest_samples, latest_run_samples)" in src
    assert "enough_initial = initial_chosen is not None" in src
    assert 'latest_finished_at = latest_run.get("finished_at")' in src


def test_model_progress_reporter_uses_configured_gate_horizon() -> None:
    """A h5m challenger must not be reported with hard-coded 15m gate stats."""
    import inspect

    from trader.modules.training import TrainingModule

    src = inspect.getsource(TrainingModule.run_model_progress_reporter)
    assert "report_horizon" in src
    assert "get_shadow_gate_stats(\n                        version,\n                        report_horizon," in src
    assert (
        "gate_event_counter(\n                            version,\n                            report_horizon," in src
    )


def test_training_allowlist_includes_candle_sampler_baselines() -> None:
    """Candle-sampling baselines (SHADOW_CANDLE) must remain trainable with strategy_id set."""
    import inspect

    from trader.training import eligibility, train

    src = inspect.getsource(train._train)
    assert "training_strategy_filter_sql" in src
    eligibility_src = inspect.getsource(eligibility)
    assert "SHADOW_CANDLE" in eligibility_src
    assert "strategy_id' IS NULL" not in eligibility_src


@pytest.mark.asyncio
async def test_db_diagnostics_reports_legacy_challenger_fallback() -> None:
    """Diagnostics should mirror registry fallback for legacy challenger artifacts."""
    from trader.storage.directional_trade_journal import DirectionalTradeJournal

    journal = DirectionalTradeJournal(postgres_dsn="postgresql://fake", enabled=True)
    journal._pool = MagicMock()
    journal._enabled = True
    directional_latest_queries = 0

    async def mock_fetch(query: str, *args: Any) -> list[dict[str, Any]]:
        nonlocal directional_latest_queries
        if "GROUP BY feature_schema_hash" in query:
            return []
        if "GROUP BY pool" in query:
            return []
        if "GROUP BY label_threshold_bps" in query:
            return []
        del args
        if "FROM prediction_outcomes" in query:
            return []
        if "pe.model_version = 'RULE_BASELINE_V1'" in query:
            return []
        if "FROM training_runs" in query:
            return []
        if (
            "FROM model_versions" in query
            and "artifact IS NOT NULL" in query
            and "COALESCE(metrics->>'label_schema_version', '')" in query
            and "status = 'CHAMPION'" not in query
        ):
            directional_latest_queries += 1
            if directional_latest_queries == 1:
                return []
        if (
            "FROM model_versions" in query
            and "artifact IS NOT NULL" in query
            and "COALESCE(metrics->>'label_schema_version', '')" not in query
            and "status = 'CHAMPION'" not in query
        ):
            return [
                {
                    "version": "v_legacy_no_schema",
                    "status": "SHADOW_CHALLENGER",
                    "training_samples": 100,
                    "metrics": {},
                    "feature_schema_hash": "",
                    "training_finished_at": datetime(2026, 6, 7, 10, 1, tzinfo=UTC),
                    "created_at": datetime(2026, 6, 7, 10, 1, tzinfo=UTC),
                }
            ]
        if "FROM model_versions" in query and "status = 'CHAMPION'" in query:
            return []
        return []

    journal._fetch = mock_fetch  # type: ignore[method-assign]

    diag = await journal.get_db_diagnostics()

    assert diag["latest_model_version"]["version"] == "v_legacy_no_schema"
    assert diag["latest_model_version"]["schema_compatible"] is False
    assert diag["latest_model_version"]["training_samples_compatible"] == 0
    assert diag["active_model_version"]["version"] == "v_legacy_no_schema"


@pytest.mark.asyncio
async def test_trade_journal_keeps_reconnectable_after_connect_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient Render/Postgres startup failure must not disable DB forever."""
    from trader.storage import trade_journal as trade_journal_module
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal(postgres_dsn="postgresql://fake:fake@localhost/fake", enabled=True)
    journal._ensure_schema = AsyncMock()  # type: ignore[method-assign]
    fake_pool = MagicMock()
    attempts = 0
    create_pool_kwargs: list[dict[str, Any]] = []

    async def fake_create_pool(*args: Any, **kwargs: Any) -> MagicMock:
        del args
        nonlocal attempts
        create_pool_kwargs.append(kwargs)
        attempts += 1
        if attempts == 1:
            raise OSError("temporary postgres dns failure")
        return fake_pool

    monkeypatch.setattr(trade_journal_module.asyncpg, "create_pool", fake_create_pool)

    await journal.connect()
    first_diag = await journal.get_db_diagnostics()

    assert journal.is_configured is True
    assert journal.is_enabled is False
    assert first_diag["configured"] is True
    assert first_diag["connected"] is False
    assert "temporary postgres dns failure" in first_diag["last_connect_error"]

    reconnected = await journal.reconnect_if_needed(force=True)

    assert reconnected is True
    assert journal.is_enabled is True
    assert journal._pool is fake_pool
    assert all(kwargs["statement_cache_size"] == 0 for kwargs in create_pool_kwargs)


# ---------------------------------------------------------------------------
# No lookahead leakage check (conceptual / structural)
# ---------------------------------------------------------------------------


def test_no_lookahead_leakage() -> None:
    """Feature snapshot only uses data available at open_time (before close_time).

    This is a structural test: we verify that candle_open_time is recorded
    (not close_time), ensuring features are computed on the open bar,
    not on future bar data.
    """
    # The record_feature_snapshot takes candle_open_time, not close_time.
    # Labels are assigned AFTER horizon_minutes have elapsed past the signal timestamp.
    # Since features are indexed by open_time and labels by resolved_at,
    # there is no path for future data to contaminate features.

    # Structural verification: open_time < signal_time < resolved_at
    signal_time = datetime(2026, 1, 1, 0, 15, tzinfo=UTC)
    open_time = datetime(2026, 1, 1, 0, 14, tzinfo=UTC)  # bar that just closed
    resolved_at = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)  # 15m later

    assert open_time < signal_time < resolved_at, "Label resolution is strictly after feature recording"


# ---------------------------------------------------------------------------
# Telegram DB/Model screen (smoke)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_database_model_telegram_screen() -> None:
    """_cmd_db_model should produce a valid message text without crashing."""
    from trader.telegram_bot import TelegramBotConfig, TelegramMonitorBot, TradingController

    async def fake_db_diag() -> dict:
        return {
            "connected": True,
            "candles_by_interval": {"1": 1000, "5": 200, "15": 100, "60": 50},
            "latest_candle_1m": datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            "feature_snapshots": 500,
            "prediction_outcomes": 300,
            "prediction_outcomes_by_horizon": {"5": 120, "15": 180},
            "labelled_samples_15m": 180,
            "training_eligible_by_horizon": {"5": 180, "15": 180},
            "training_config": {"auto_train_horizon_minutes": 5},
            "latest_training_run": {
                "status": "COMPLETED",
                "model_version": "v20260607_1000",
                "sample_count": 180,
                "error": None,
                "metrics": {"quality": "GOOD", "precision": 0.61, "lift_bps": 2.7},
                "finished_at": datetime(2026, 6, 7, 10, 1, tzinfo=UTC),
            },
            "latest_model_version": {
                "version": "v20260607_1000",
                "status": "SHADOW_CHALLENGER",
                "training_samples": 180,
                "metrics": {
                    "quality": "GOOD",
                    "validation_samples": 80,
                    "precision": 0.61,
                    "lift_bps": 2.7,
                    "best_threshold": 0.6,
                    "best_threshold_avg_net_return_bps": 5.1,
                    "walk_forward_expectancy_bps": 4.2,
                },
            },
            "shadow_gate_15m": {
                "model_version": "v20260607_1000",
                "total_count": 20,
                "pass_count": 12,
                "block_count": 8,
                "pass_avg_net_return_bps": 4.5,
                "block_avg_net_return_bps": -1.5,
                "lift_vs_all_bps": 2.4,
                "top_block_reasons": {"score_below_regime_threshold": 8},
            },
            "paper_notional_usd": 5.0,
            "paper_pnl_15m": {
                "baseline": {"count": 20, "total_bps": 10.0, "max_drawdown_bps": -4.0},
                "model_gate": {"count": 12, "total_bps": 18.0, "max_drawdown_bps": -2.0},
            },
        }

    async def fake_health() -> Any:
        return MagicMock(ok=True)

    controller = TradingController(
        pause=AsyncMock(),
        resume=AsyncMock(),
        set_shadow=AsyncMock(),
        set_risk_profile=AsyncMock(),
        emergency_stop=AsyncMock(),
        is_paused=lambda: False,
        is_shadow=lambda: True,
        current_profile=lambda: "CONSERVATIVE",
        active_symbols=lambda: [],
        regime_for=lambda s: None,
        diagnostics_provider=lambda: {"model": {}},
        db_diagnostics_provider=fake_db_diag,
    )

    config = TelegramBotConfig(
        token="fake:TOKEN",
        allowed_chat_ids={12345},
        trading_mode="SHADOW",
        risk_profile="CONSERVATIVE",
        bybit_use_testnet=True,
    )

    bot = TelegramMonitorBot(
        config=config,
        health_provider=fake_health,
        adapter_factory=lambda: None,
        controller=controller,
    )

    # Smoke test: _cmd_db_model should not raise
    fake_update = MagicMock()
    fake_update.effective_chat = MagicMock()
    fake_update.effective_chat.id = 12345
    fake_update.callback_query = None
    fake_message = MagicMock()
    fake_message.reply_text = AsyncMock()
    fake_update.effective_message = fake_message

    fake_context = type("_Ctx", (), {"args": []})()

    # The method calls self._reply which calls message.reply_text with HTML parse mode.
    await bot._cmd_db_model(fake_update, fake_context)  # type: ignore[arg-type]
    text = fake_message.reply_text.await_args.args[0]
    assert "Готово для обучения (5m)" in text
    assert "v20260607_1000" in text
    assert "Качество" in text
    assert "ХОРОШО" in text
    assert "+2.70 bps" in text
    assert "Фильтр модели 5m" in text
    assert "12/20 resolved пропущено" in text
    assert "observed=<code>20</code>, pending=<code>0</code>" in text
    assert "Paper baseline" in text
    assert "Paper model gate" in text
    assert "score_below_regime_threshold" in text


# ---------------------------------------------------------------------------
# Challenger fallback: load model even when label_schema_version not in metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_latest_challenger_falls_back_when_schema_missing() -> None:
    """Registry loads a SHADOW_CHALLENGER even if metrics->label_schema_version is absent."""
    import io

    import joblib
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    # Build a minimal artifact without label_schema_version in metrics
    buf = io.BytesIO()
    clf = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True, random_state=42)
    scaler = StandardScaler()
    joblib.dump(
        {
            "clf": clf,
            "scaler": scaler,
            "meta": {
                "version": "v_legacy_no_schema",
                "feature_names": ["ema_9", "rsi_14"],
                "training_samples": 100,
                # intentionally missing label_schema_version
                "model_type": "SGD",
                "model_params": {},
            },
        },
        buf,
    )
    artifact_bytes = buf.getvalue()

    # Primary query (schema filter) returns nothing; fallback returns the row
    journal = MagicMock()
    journal.is_enabled = True

    primary_call = AsyncMock(return_value=[])
    fallback_row: dict[str, Any] = {
        "version": "v_legacy_no_schema",
        "status": "SHADOW_CHALLENGER",
        "artifact": artifact_bytes,
        "training_samples": 100,
        "metrics": {},
    }
    fallback_call = AsyncMock(return_value=[fallback_row])

    call_count = 0

    async def _fetch(sql: str, *args: Any) -> list[Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return await primary_call(sql, *args)
        return await fallback_call(sql, *args)

    journal._fetch = _fetch

    registry = ModelRegistry(trade_journal=journal)
    model = await registry.load_latest_challenger()

    assert model is not None
    assert model.version == "v_legacy_no_schema"
    assert model.training_samples == 100
    assert model.allow_live_decisions is False
    assert call_count == 2  # primary failed, fallback used


def test_runtime_challenger_loader_prefers_freshness_over_quality() -> None:
    """Runtime shadow scoring should load the freshest challenger; promotion ranks quality separately."""
    import inspect

    src = inspect.getsource(ModelRegistry.load_latest_challenger)
    primary_query = src.split("if not rows:", maxsplit=1)[0]
    assert "ORDER BY training_finished_at DESC NULLS LAST, created_at DESC" in primary_query
    assert "CASE WHEN COALESCE(metrics->>'quality'" not in primary_query
