"""Tests for TradeJournal.mark_outcomes_learned UUID validation."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_journal():
    from trader.storage.trade_journal import TradeJournal

    journal = TradeJournal.__new__(TradeJournal)
    journal._enabled = True
    journal._pool = MagicMock()
    journal._consecutive_write_errors = 0
    journal._execute = AsyncMock()
    return journal


class TestMarkOutcomesLearned:
    @pytest.mark.asyncio
    async def test_empty_list_does_nothing(self):
        journal = _make_journal()
        await journal.mark_outcomes_learned([])
        journal._execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_valid_uuid_strings_are_accepted(self):
        journal = _make_journal()
        uid = str(uuid.uuid4())
        await journal.mark_outcomes_learned([uid])
        journal._execute.assert_awaited_once()
        args = journal._execute.call_args[0]
        assert uid in args[1]

    @pytest.mark.asyncio
    async def test_valid_uuid_objects_are_accepted(self):
        journal = _make_journal()
        uid = uuid.uuid4()
        await journal.mark_outcomes_learned([uid])
        journal._execute.assert_awaited_once()
        args = journal._execute.call_args[0]
        assert str(uid) in args[1]

    @pytest.mark.asyncio
    async def test_invalid_uuid_strings_are_skipped(self):
        journal = _make_journal()
        await journal.mark_outcomes_learned(["not-a-uuid", "bad"])
        journal._execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_mixed_valid_and_invalid_filters_invalid(self):
        journal = _make_journal()
        valid = str(uuid.uuid4())
        await journal.mark_outcomes_learned([valid, "garbage", None])
        journal._execute.assert_awaited_once()
        args = journal._execute.call_args[0]
        assert args[1] == [valid]

    @pytest.mark.asyncio
    async def test_all_invalid_does_nothing(self):
        journal = _make_journal()
        await journal.mark_outcomes_learned(["bad", 123, None, {}, []])
        journal._execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_valid_uuids_all_passed(self):
        journal = _make_journal()
        uids = [str(uuid.uuid4()) for _ in range(5)]
        await journal.mark_outcomes_learned(uids)
        journal._execute.assert_awaited_once()
        args = journal._execute.call_args[0]
        assert set(args[1]) == set(uids)
