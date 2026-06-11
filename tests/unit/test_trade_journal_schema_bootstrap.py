"""Regression tests for trade journal schema bootstrap ordering."""

from __future__ import annotations

from typing import Any, cast

import pytest

from trader.storage.trade_journal import TradeJournal


class _FakeConnection:
    def __init__(self) -> None:
        self.label_schema_column_seen = False

    async def execute(self, sql: str) -> None:
        label_alter_pos = sql.find("ALTER TABLE prediction_outcomes")
        label_index_pos = sql.find("idx_prediction_outcomes_label_schema")
        label_alter = label_alter_pos >= 0 and "label_schema_version" in sql[label_alter_pos:]
        label_index = "idx_prediction_outcomes_label_schema" in sql

        if label_index and not self.label_schema_column_seen:
            if not label_alter or label_alter_pos > label_index_pos:
                raise AssertionError("label_schema_version index was created before the column bootstrap")

        if label_alter:
            self.label_schema_column_seen = True


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
