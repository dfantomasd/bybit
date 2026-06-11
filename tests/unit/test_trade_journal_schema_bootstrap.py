"""Regression tests for trade journal schema bootstrap ordering."""

from __future__ import annotations

from typing import Any, cast

import pytest

from trader.storage.trade_journal import TradeJournal


class _FakeConnection:
    def __init__(self) -> None:
        self.label_schema_column_seen = False
        self.feature_snapshot_eligibility_seen = False
        self.feature_snapshot_duplicates_invalidated = False
        self.executed_sql: list[str] = []

    async def execute(self, sql: str) -> None:
        self.executed_sql.append(sql)
        label_alter_pos = sql.find("ALTER TABLE prediction_outcomes")
        label_index_pos = sql.find("idx_prediction_outcomes_label_schema")
        label_alter = label_alter_pos >= 0 and "label_schema_version" in sql[label_alter_pos:]
        label_index = "idx_prediction_outcomes_label_schema" in sql
        feature_alter_pos = sql.find("ALTER TABLE feature_snapshots")
        duplicate_update_pos = sql.find("duplicate_snapshot_same_candle")
        feature_index_pos = sql.find("idx_feature_snapshots_unique_eligible")
        feature_alter = feature_alter_pos >= 0 and "training_eligible" in sql[feature_alter_pos:]
        duplicate_update = duplicate_update_pos >= 0
        feature_index = feature_index_pos >= 0

        if label_index and not self.label_schema_column_seen:
            if not label_alter or label_alter_pos > label_index_pos:
                raise AssertionError("label_schema_version index was created before the column bootstrap")

        if label_alter:
            self.label_schema_column_seen = True

        if feature_index and not self.feature_snapshot_eligibility_seen:
            if not feature_alter or feature_alter_pos > feature_index_pos:
                raise AssertionError("eligible snapshot index was created before eligibility columns")

        if feature_index and not self.feature_snapshot_duplicates_invalidated:
            if not duplicate_update or duplicate_update_pos > feature_index_pos:
                raise AssertionError("eligible snapshot index was created before duplicate invalidation")

        if feature_alter:
            self.feature_snapshot_eligibility_seen = True

        if duplicate_update:
            self.feature_snapshot_duplicates_invalidated = True


class _AcquireContext:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConnection:
        return self._conn

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class _FakePool:
    def __init__(self) -> None:
        self.conn = _FakeConnection()

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self.conn)


@pytest.mark.asyncio
async def test_prediction_outcomes_label_schema_column_is_bootstrapped_before_index() -> None:
    journal = TradeJournal("postgresql://example/db")
    journal._pool = cast(Any, _FakePool())

    await journal._ensure_schema()


@pytest.mark.asyncio
async def test_feature_snapshot_eligible_unique_index_is_bootstrapped_after_deduplication() -> None:
    journal = TradeJournal("postgresql://example/db")
    journal._pool = cast(Any, _FakePool())

    await journal._ensure_schema()


@pytest.mark.asyncio
async def test_ml_and_pending_state_indexes_are_bootstrapped() -> None:
    pool = _FakePool()
    journal = TradeJournal("postgresql://example/db")
    journal._pool = cast(Any, pool)

    await journal._ensure_schema()

    sql = "\n".join(pool.conn.executed_sql)
    assert "idx_prediction_events_model_time" in sql
    assert "idx_prediction_events_model_decision_time" in sql
    assert "idx_prediction_outcomes_horizon_schema" in sql
    assert "idx_order_pending_state_symbol_unresolved" in sql
