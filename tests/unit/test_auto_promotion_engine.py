"""Tests for safe automatic model promotion."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from trader.ml.auto_promotion import AutoPromotionConfig, AutoPromotionEngine
from trader.ml.challenger import ModelStatus
from trader.training.labels import LABEL_SCHEMA_VERSION, LABEL_SCHEMA_VERSION_TPSL


def _model(
    version: str,
    *,
    status: str = ModelStatus.SHADOW_CHALLENGER,
    quality: str = "GOOD",
    wf_bps: float = 5.0,
    lift_bps: float = 5.0,
    samples: int = 1000,
) -> dict[str, Any]:
    return {
        "version": version,
        "status": status,
        "training_samples": samples,
        "feature_schema_hash": "abc123",
        "metrics": {
            "quality": quality,
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "walk_forward_expectancy_bps": wf_bps,
            "wf_mean_bps": wf_bps,
            "best_threshold_avg_net_return_bps": wf_bps,
            "lift_bps": lift_bps,
            "precision": 0.42,
            "total_pass_count": 80,
            "wf_folds": 5,
            "wf_positive_folds": 5,
            "wf_std_bps": 4.0,
            "walk_forward_chronology": "strict_after_train",
        },
    }


class _Conn:
    def __init__(self, journal: _Journal) -> None:
        self.journal = journal
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    async def __aenter__(self) -> _Conn:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def transaction(self) -> _Conn:
        return self

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        del query, args
        for row in self.journal.models.values():
            if row["status"] == ModelStatus.CHAMPION:
                return {"version": row["version"]}
        return None

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append((query, args))
        self.journal.conn_exec.append((query, args))
        if "status = 'ROLLED_BACK'" in query:
            version = str(args[0])
            self.journal.models[version]["status"] = ModelStatus.ROLLED_BACK
            return "UPDATE 1"
        if "status = 'ARCHIVED'" in query and "WHERE status = 'CHAMPION'" in query:
            for row in self.journal.models.values():
                if row["status"] == ModelStatus.CHAMPION:
                    row["status"] = ModelStatus.ARCHIVED
            return "UPDATE 1"
        if "status = 'CHAMPION'" in query and "version = $1" in query:
            version = str(args[0])
            if version in self.journal.models:
                self.journal.models[version]["status"] = ModelStatus.CHAMPION
                return "UPDATE 1"
            return "UPDATE 0"
        if "INSERT INTO model_promotion_log" in query:
            self.journal.promotion_logs.append(args)
            return "INSERT 0 1"
        return "SELECT 1"


class _Acquire:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn

    async def __aenter__(self) -> _Conn:
        return self.conn

    async def __aexit__(self, *args: Any) -> None:
        return None


class _Pool:
    def __init__(self, journal: _Journal) -> None:
        self.conn = _Conn(journal)

    def acquire(self) -> _Acquire:
        return _Acquire(self.conn)


class _Journal:
    def __init__(self, models: list[dict[str, Any]]) -> None:
        self.models = {str(row["version"]): dict(row) for row in models}
        self._pool = _Pool(self)
        self.promotion_logs: list[tuple[Any, ...]] = []
        self.conn_exec: list[tuple[str, tuple[Any, ...]]] = []

    async def _fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        if "WHERE version = $1" in query:
            row = self.models.get(str(args[0]))
            return [dict(row)] if row else []
        if "status = 'CHAMPION'" in query:
            return [dict(row) for row in self.models.values() if row["status"] == ModelStatus.CHAMPION][:1]
        if "status = 'ARCHIVED'" in query:
            return [dict(row) for row in self.models.values() if row["status"] == ModelStatus.ARCHIVED][:1]
        if "status IN ('SHADOW_CHALLENGER', 'VALIDATED')" in query:
            return [
                dict(row)
                for row in self.models.values()
                if row["status"] in {ModelStatus.SHADOW_CHALLENGER, ModelStatus.VALIDATED}
            ]
        return []

    async def _execute(self, query: str, *args: Any) -> None:
        if "INSERT INTO model_promotion_log" in query:
            self.promotion_logs.append(args)

    async def get_shadow_gate_stats(self, model_version: str, horizon_minutes: int, label_schema_version: str) -> dict:
        del horizon_minutes, label_schema_version
        row = self.models[model_version]
        metrics = row["metrics"]
        return {
            "total_count": 120,
            "pass_count": int(metrics.get("total_pass_count", 80)),
            "lift_vs_all_bps": float(metrics.get("lift_bps", 5.0)),
            "pass_avg_net_return_bps": float(metrics.get("walk_forward_expectancy_bps", 5.0)),
            "pass_precision": float(metrics.get("precision", 0.42)),
            "side_filtered_count": int(metrics.get("side_filtered_count", 0)),
            "score_block_count": int(metrics.get("score_block_count", 0)),
            "score_block_avg_net_return_bps": metrics.get("score_block_avg_net_return_bps"),
            "top_block_reasons": metrics.get("top_block_reasons", {}),
            "quality": metrics.get("quality", "GOOD"),
        }

    async def get_returns_for_model(
        self,
        model_version: str,
        limit: int,
        horizon_minutes: int | None = None,
        label_schema_version: str | None = None,
    ) -> list:
        del limit, horizon_minutes, label_schema_version
        if model_version == "RULE_BASELINE_V1":
            return [0.0] * 80
        if model_version in self.models:
            metrics = self.models[model_version].get("metrics") or {}
            if isinstance(metrics, dict) and isinstance(metrics.get("returns_bps"), list):
                return list(metrics["returns_bps"])
            return [8.0] * 80
        return []


def _config() -> AutoPromotionConfig:
    return AutoPromotionConfig(
        enabled=True,
        min_training_samples=500,
        min_shadow_signals=50,
        min_pass_count=20,
        min_lift_bps=1.0,
        min_pass_expectancy_bps=0.0,
        min_wf_bps=0.0,
        bootstrap_iterations=200,
        min_bootstrap_samples=50,
    )


@pytest.mark.asyncio
async def test_best_challenger_uses_best_eligible_model_not_latest_weak() -> None:
    journal = _Journal(
        [
            _model("champion", status=ModelStatus.CHAMPION, wf_bps=1.0),
            _model("good-old", wf_bps=6.0, lift_bps=5.0),
            _model("weak-latest", quality="WEAK", wf_bps=-8.0, lift_bps=-2.0),
        ]
    )
    engine = AutoPromotionEngine(trade_journal=journal, config=_config())

    assert await engine.best_challenger() == "good-old"


@pytest.mark.asyncio
async def test_should_promote_blocks_weak_challenger() -> None:
    journal = _Journal([_model("champion", status=ModelStatus.CHAMPION), _model("weak", quality="WEAK")])
    config = replace(_config(), required_quality="GOOD")
    engine = AutoPromotionEngine(trade_journal=journal, config=config)

    decision = await engine.should_promote(None, "weak")

    assert decision.promote is False
    assert any(reason.startswith("quality_below_GOOD") for reason in decision.reasons)


@pytest.mark.asyncio
async def test_should_promote_allows_weak_when_min_quality_is_weak() -> None:
    journal = _Journal([_model("champion", status=ModelStatus.CHAMPION, wf_bps=1.0), _model("weak", quality="WEAK")])
    config = replace(_config(), required_quality="WEAK")
    engine = AutoPromotionEngine(trade_journal=journal, config=config)

    decision = await engine.should_promote(None, "weak")

    assert decision.promote is True


@pytest.mark.asyncio
async def test_should_promote_uses_configured_tpsl_label_schema() -> None:
    challenger = _model("dnv2")
    challenger["metrics"]["label_schema_version"] = LABEL_SCHEMA_VERSION_TPSL
    journal = _Journal([challenger])
    config = replace(_config(), label_schema_version=LABEL_SCHEMA_VERSION_TPSL)
    engine = AutoPromotionEngine(trade_journal=journal, config=config)

    decision = await engine.should_promote(None, "dnv2")

    assert not any(reason.startswith("incompatible_label_schema") for reason in decision.reasons)


@pytest.mark.asyncio
async def test_should_promote_exposes_side_filter_gate_breakdown() -> None:
    challenger = _model("side-filtered")
    challenger["metrics"]["side_filtered_count"] = 44
    challenger["metrics"]["score_block_count"] = 11
    challenger["metrics"]["score_block_avg_net_return_bps"] = -1.25
    challenger["metrics"]["top_block_reasons"] = {
        "side_not_selected_by_model": 44,
        "score_below_regime_threshold": 11,
    }
    journal = _Journal([_model("champion", status=ModelStatus.CHAMPION, wf_bps=1.0), challenger])
    engine = AutoPromotionEngine(trade_journal=journal, config=_config())

    decision = await engine.should_promote(None, "side-filtered")

    assert decision.metrics["side_filtered_count"] == 44
    assert decision.metrics["score_block_count"] == 11
    assert decision.metrics["score_block_expectancy_bps"] == -1.25
    assert decision.metrics["top_block_reasons"]["side_not_selected_by_model"] == 44
    assert decision.metrics["snapshot"]["challenger_gate"]["side_filtered_count"] == 44


@pytest.mark.asyncio
async def test_promote_archives_champion_promotes_challenger_logs_and_reloads() -> None:
    journal = _Journal([_model("champion", status=ModelStatus.CHAMPION, wf_bps=1.0), _model("challenger")])
    reload_registry = AsyncMock()
    engine = AutoPromotionEngine(trade_journal=journal, config=_config(), reload_registry=reload_registry)

    decision = await engine.promote("challenger")

    assert decision.promote is True
    assert journal.models["champion"]["status"] == ModelStatus.ARCHIVED
    assert journal.models["challenger"]["status"] == ModelStatus.CHAMPION
    assert journal.promotion_logs
    assert json.loads(journal.promotion_logs[-1][2]) == ["criteria_met"]
    snapshot = json.loads(journal.promotion_logs[-1][3])["snapshot"]
    assert snapshot["champion"]["version"] == "champion"
    assert snapshot["challenger"]["version"] == "challenger"
    assert "walk_forward_bps" in snapshot["delta"]
    reload_registry.assert_awaited_once()


@pytest.mark.asyncio
async def test_should_promote_blocks_unstable_walk_forward_folds() -> None:
    challenger = _model("unstable")
    challenger["metrics"]["wf_positive_folds"] = 1
    challenger["metrics"]["wf_std_bps"] = 40.0
    journal = _Journal([_model("champion", status=ModelStatus.CHAMPION, wf_bps=1.0), challenger])
    engine = AutoPromotionEngine(trade_journal=journal, config=_config())

    decision = await engine.should_promote(None, "unstable")

    assert decision.promote is False
    assert any(reason.startswith("unstable_walk_forward_folds") for reason in decision.reasons)
    assert any(reason.startswith("unstable_walk_forward_std") for reason in decision.reasons)


@pytest.mark.asyncio
async def test_should_promote_blocks_excessive_challenger_drawdown() -> None:
    challenger = _model("high-drawdown")
    challenger["metrics"]["returns_bps"] = [20.0] * 60 + [-1000.0] + [20.0] * 19
    journal = _Journal([_model("champion", status=ModelStatus.CHAMPION, wf_bps=1.0), challenger])
    engine = AutoPromotionEngine(
        trade_journal=journal,
        config=replace(_config(), max_challenger_drawdown_bps=500.0),
    )

    decision = await engine.should_promote(None, "high-drawdown")

    assert decision.promote is False
    assert any(reason.startswith("challenger_drawdown") for reason in decision.reasons)
    assert decision.metrics["challenger_drawdown_bps"] > 500.0


@pytest.mark.asyncio
async def test_rollback_restores_previous_archived_champion() -> None:
    champion = _model("bad-champion", status=ModelStatus.CHAMPION, wf_bps=-2.0)
    archived = _model("old-champion", status=ModelStatus.ARCHIVED, wf_bps=4.0)
    journal = _Journal([champion, archived])
    reload_registry = AsyncMock()
    engine = AutoPromotionEngine(trade_journal=journal, config=_config(), reload_registry=reload_registry)

    decision = await engine.rollback_if_needed()

    assert decision.rollback is True
    assert journal.models["bad-champion"]["status"] == ModelStatus.ROLLED_BACK
    assert journal.models["old-champion"]["status"] == ModelStatus.CHAMPION
    reload_registry.assert_awaited_once()
