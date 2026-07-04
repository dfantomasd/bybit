"""Regression tests for trade journal schema bootstrap ordering."""

from __future__ import annotations

from typing import Any, cast

import pytest

from trader.storage.trade_journal import TradeJournal


class _FakeConnection:
    def __init__(self) -> None:
        self.label_schema_column_seen = False
        self.feature_snapshot_eligibility_seen = False
        self.executed_sql: list[str] = []
        self.fetchval_sql: list[str] = []
        self.existing_columns: set[tuple[str, str]] = set()
        self.fail_once_on: str | None = None
        self.failed_once = False

    async def execute(self, sql: str) -> None:
        self.executed_sql.append(sql)
        if self.fail_once_on and self.fail_once_on in sql and not self.failed_once:
            self.failed_once = True
            raise AttributeError("'NoneType' object has no attribute 'decode'")
        label_alter_pos = sql.find("ALTER TABLE prediction_outcomes")
        label_index_pos = sql.find("idx_prediction_outcomes_label_schema")
        label_alter = label_alter_pos >= 0 and "label_schema_version" in sql[label_alter_pos:]
        label_index = "idx_prediction_outcomes_label_schema" in sql
        feature_alter_pos = sql.find("ALTER TABLE feature_snapshots")
        feature_index_pos = sql.find("idx_feature_snapshots_unique_eligible")
        feature_alter = feature_alter_pos >= 0 and "training_eligible" in sql[feature_alter_pos:]
        feature_index = feature_index_pos >= 0

        if label_index and not self.label_schema_column_seen:
            if not label_alter or label_alter_pos > label_index_pos:
                raise AssertionError("label_schema_version index was created before the column bootstrap")

        if label_alter:
            self.label_schema_column_seen = True

        if feature_index and not self.feature_snapshot_eligibility_seen:
            if not feature_alter or feature_alter_pos > feature_index_pos:
                raise AssertionError("eligible snapshot index was created before eligibility columns")

        if feature_alter:
            self.feature_snapshot_eligibility_seen = True

    async def fetchval(self, sql: str, *args: object) -> object:
        self.fetchval_sql.append(sql)
        if "information_schema.columns" in sql and len(args) >= 2:
            return 1 if (str(args[0]), str(args[1])) in self.existing_columns else None
        return None

    async def fetch(self, sql: str, *args: object) -> list[dict[str, object]]:
        if "information_schema.columns" in sql:
            tables = {str(item) for item in args[0]} if args else set()
            return [
                {"table_name": table, "column_name": column}
                for table, column in sorted(self.existing_columns)
                if table in tables
            ]
        return []

    def transaction(self) -> "_FakeTransaction":
        return _FakeTransaction()


class _FakeTransaction:
    async def __aenter__(self) -> "_FakeTransaction":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


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
async def test_feature_snapshot_eligible_unique_index_is_bootstrapped_after_columns() -> None:
    journal = TradeJournal("postgresql://example/db")
    journal._pool = cast(Any, _FakePool())

    await journal._ensure_schema()
    await journal._ensure_feature_snapshot_unique_index_deferred()


@pytest.mark.asyncio
async def test_ml_and_pending_state_indexes_are_bootstrapped() -> None:
    pool = _FakePool()
    journal = TradeJournal("postgresql://example/db")
    journal._pool = cast(Any, pool)

    await journal._ensure_schema()
    await journal._ensure_model_registry_indexes_deferred()
    await journal._ensure_feature_snapshot_unique_index_deferred()

    sql = "\n".join(pool.conn.executed_sql)
    assert "idx_prediction_events_model_time" in sql
    assert "idx_prediction_events_model_decision_time" in sql
    assert "idx_prediction_outcomes_horizon_schema" in sql
    assert "idx_order_pending_state_symbol_unresolved" in sql
    assert "uq_model_versions_one_champion" in sql
    assert "idx_feature_snapshots_unique_eligible" in sql
    assert "duplicate_snapshot_same_candle" in sql


@pytest.mark.asyncio
async def test_legacy_journal_tables_get_created_at_backfill() -> None:
    pool = _FakePool()
    journal = TradeJournal("postgresql://example/db")
    journal._pool = cast(Any, pool)

    await journal._ensure_schema()

    sql = "\n".join(pool.conn.executed_sql)
    for table in (
        "trade_signals",
        "risk_decisions",
        "order_events",
        "closed_pnl",
        "execution_events",
        "market_candles",
        "feature_snapshots",
        "prediction_events",
    ):
        assert f"ALTER TABLE {table}" in sql
        assert "ADD COLUMN IF NOT EXISTS created_at" in sql


@pytest.mark.asyncio
async def test_schema_bootstrap_recovers_if_idempotent_add_column_hits_pooler_decode_bug() -> None:
    pool = _FakePool()
    pool.conn.fail_once_on = "ADD COLUMN IF NOT EXISTS training_eligible"
    pool.conn.existing_columns.add(("feature_snapshots", "training_eligible"))
    journal = TradeJournal("postgresql://example/db")
    journal._pool = cast(Any, pool)

    await journal._ensure_schema()

    sql = "\n".join(pool.conn.executed_sql)
    assert pool.conn.failed_once is True
    assert "information_schema.columns" in "\n".join(pool.conn.fetchval_sql)
    assert "ADD COLUMN IF NOT EXISTS label_threshold_bps" in sql


@pytest.mark.asyncio
async def test_schema_health_reports_missing_critical_columns() -> None:
    pool = _FakePool()
    pool.conn.existing_columns.update(
        {
            ("feature_snapshots", "snapshot_id"),
            ("feature_snapshots", "training_eligible"),
            ("prediction_outcomes", "prediction_id"),
            ("prediction_outcomes", "label_schema_version"),
        }
    )
    journal = TradeJournal("postgresql://example/db")
    journal._pool = cast(Any, pool)

    health = await journal.get_schema_health()

    assert health["ok"] is False
    assert "prediction_outcomes.label_threshold_bps" in health["missing_columns"]
    assert health["checked_columns"] >= 10


@pytest.mark.asyncio
async def test_schema_health_reports_ok_when_critical_columns_exist() -> None:
    pool = _FakePool()
    for table, columns in TradeJournal._critical_schema_columns().items():
        for column in columns:
            pool.conn.existing_columns.add((table, column))
    journal = TradeJournal("postgresql://example/db")
    journal._pool = cast(Any, pool)

    health = await journal.get_schema_health()

    assert health == {
        "ok": True,
        "checked_columns": sum(len(columns) for columns in TradeJournal._critical_schema_columns().values()),
        "missing_columns": [],
    }


@pytest.mark.asyncio
async def test_feature_snapshot_duplicates_are_invalidated_before_unique_index() -> None:
    pool = _FakePool()
    journal = TradeJournal("postgresql://example/db")
    journal._pool = cast(Any, pool)

    await journal._ensure_schema()
    await journal._ensure_model_registry_indexes_deferred()
    await journal._ensure_feature_snapshot_unique_index_deferred()

    sql = "\n".join(pool.conn.executed_sql)
    repair_pos = sql.find("duplicate_snapshot_same_candle")
    index_pos = sql.find("idx_feature_snapshots_unique_eligible")
    assert repair_pos >= 0
    assert index_pos >= 0
    assert repair_pos < index_pos


@pytest.mark.asyncio
async def test_model_version_schema_hash_is_repaired_from_source_schema_metric() -> None:
    pool = _FakePool()
    journal = TradeJournal("postgresql://example/db")
    journal._pool = cast(Any, pool)

    await journal._ensure_schema()
    await journal._ensure_model_registry_indexes_deferred()

    sql = "\n".join(pool.conn.executed_sql)
    assert "SET feature_schema_hash = metrics->>'source_feature_schema_hash'" in sql
    assert "feature_schema_hash IS DISTINCT FROM metrics->>'source_feature_schema_hash'" in sql


@pytest.mark.asyncio
async def test_market_candles_schema_defines_low_column_once() -> None:
    pool = _FakePool()
    journal = TradeJournal("postgresql://example/db")
    journal._pool = cast(Any, pool)

    await journal._ensure_schema()

    create_market_candles = next(
        sql for sql in pool.conn.executed_sql if "CREATE TABLE IF NOT EXISTS market_candles" in sql
    )
    assert create_market_candles.count("low numeric NOT NULL") == 1


@pytest.mark.asyncio
async def test_feature_snapshot_unique_index_bootstrap_is_best_effort() -> None:
    class FailingIndexConnection(_FakeConnection):
        async def execute(self, sql: str) -> None:
            if "idx_feature_snapshots_unique_eligible" in sql:
                raise RuntimeError("duplicate key value violates unique constraint")
            await super().execute(sql)

    conn = FailingIndexConnection()
    journal = TradeJournal("postgresql://example/db")

    await journal._ensure_feature_snapshot_unique_index(cast(Any, conn))

    sql = "\n".join(conn.executed_sql)
    assert "duplicate_snapshot_same_candle" in sql
    assert "idx_feature_snapshots_unique_eligible" not in sql
