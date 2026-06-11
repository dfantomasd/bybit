"""Tests for the one-shot startup candle backfill."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.app import TradingApplication
from trader.config import Settings


def _make_app(**settings_overrides) -> TradingApplication:
    app = TradingApplication()
    defaults = {
        "TELEGRAM_ALLOWED_CHAT_IDS": [],
        "STARTUP_BACKFILL_ENABLED": True,
        "STARTUP_BACKFILL_DAYS": 1,
        "STARTUP_BACKFILL_MAX_REQUESTS": 10,
    }
    defaults.update(settings_overrides)
    app._settings = Settings(**defaults)
    return app


def _kline_page(start_ms: int, bar_ms: int, n: int) -> dict:
    """Bybit-style kline response: newest-first rows [ts, o, h, l, c, vol, turnover]."""
    rows = []
    for i in range(n):
        ts = start_ms - i * bar_ms
        rows.append([str(ts), "1.0", "1.1", "0.9", "1.05", "100", "105"])
    return {"result": {"list": rows}}


class TestStartupBackfill:
    @pytest.mark.asyncio
    async def test_disabled_setting_skips(self) -> None:
        app = _make_app(STARTUP_BACKFILL_ENABLED=False)
        app._bybit_adapter = MagicMock()
        await app._run_startup_backfill()
        app._bybit_adapter._rest.get_kline.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_without_db(self) -> None:
        app = _make_app()
        app._bybit_adapter = MagicMock()
        app._trade_journal = MagicMock(is_enabled=False)
        app._shutdown_event.set()  # don't wait 60s for DB
        await app._run_startup_backfill()
        app._bybit_adapter._rest.get_kline.assert_not_called()

    @pytest.mark.asyncio
    async def test_backfills_confirmed_candles_only(self) -> None:
        app = _make_app()
        bar_ms = 60_000
        now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
        # Page contains one still-open candle (current minute) and closed ones
        current_open = (now_ms // bar_ms) * bar_ms

        journal = MagicMock()
        journal.is_enabled = True
        journal.upsert_market_candle = AsyncMock()
        journal.get_candle_counts_per_symbol = AsyncMock(return_value={})

        adapter = MagicMock()
        adapter._rest.get_kline = AsyncMock(
            side_effect=[
                _kline_page(current_open, bar_ms, 5),
                {"result": {"list": []}},  # end of history
            ]
        )

        screener = MagicMock()
        screener.wait_ready = AsyncMock()
        screener.active_symbols = ["TESTUSDT"]

        app._bybit_adapter = adapter
        app._trade_journal = journal
        app._screener = screener
        app._market_data_intervals = lambda: ["1"]

        await app._run_startup_backfill()

        # 5 rows in page, the newest (current minute) is unconfirmed → 4 persisted
        assert journal.upsert_market_candle.await_count == 4
        for call in journal.upsert_market_candle.await_args_list:
            assert call.kwargs["confirmed"] is True
            assert call.kwargs["source"] == "rest_backfill"

    @pytest.mark.asyncio
    async def test_skips_pairs_with_existing_history(self) -> None:
        app = _make_app(STARTUP_BACKFILL_DAYS=1)
        # 1 day of 1m bars = 1440; 1500 stored > 90% threshold → skip entirely
        journal = MagicMock()
        journal.is_enabled = True
        journal.upsert_market_candle = AsyncMock()
        journal.get_candle_counts_per_symbol = AsyncMock(return_value={("TESTUSDT", "1"): 1500})

        adapter = MagicMock()
        adapter._rest.get_kline = AsyncMock()

        screener = MagicMock()
        screener.wait_ready = AsyncMock()
        screener.active_symbols = ["TESTUSDT"]

        app._bybit_adapter = adapter
        app._trade_journal = journal
        app._screener = screener
        app._market_data_intervals = lambda: ["1"]

        await app._run_startup_backfill()
        adapter._rest.get_kline.assert_not_called()

    @pytest.mark.asyncio
    async def test_request_cap_enforced(self) -> None:
        app = _make_app(STARTUP_BACKFILL_MAX_REQUESTS=3)
        bar_ms = 60_000
        now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
        current_open = (now_ms // bar_ms) * bar_ms

        journal = MagicMock()
        journal.is_enabled = True
        journal.upsert_market_candle = AsyncMock()
        journal.get_candle_counts_per_symbol = AsyncMock(return_value={})

        pages = [_kline_page(current_open - i * 10 * bar_ms, bar_ms, 10) for i in range(50)]
        adapter = MagicMock()
        adapter._rest.get_kline = AsyncMock(side_effect=pages)

        screener = MagicMock()
        screener.wait_ready = AsyncMock()
        screener.active_symbols = ["AUSDT", "BUSDT", "CUSDT"]

        app._bybit_adapter = adapter
        app._trade_journal = journal
        app._screener = screener
        app._market_data_intervals = lambda: ["1"]

        await app._run_startup_backfill()
        assert adapter._rest.get_kline.await_count <= 3

    @pytest.mark.asyncio
    async def test_failure_does_not_raise(self) -> None:
        app = _make_app()
        journal = MagicMock()
        journal.is_enabled = True
        journal.get_candle_counts_per_symbol = AsyncMock(side_effect=RuntimeError("db down"))
        journal.upsert_market_candle = AsyncMock()

        adapter = MagicMock()
        adapter._rest.get_kline = AsyncMock(side_effect=RuntimeError("rest down"))

        screener = MagicMock()
        screener.wait_ready = AsyncMock()
        screener.active_symbols = ["TESTUSDT"]

        app._bybit_adapter = adapter
        app._trade_journal = journal
        app._screener = screener
        app._market_data_intervals = lambda: ["1"]

        # Must not raise — supervisor safety
        await app._run_startup_backfill()

    @pytest.mark.asyncio
    async def test_shutdown_aborts_promptly(self) -> None:
        app = _make_app()
        journal = MagicMock()
        journal.is_enabled = True
        journal.upsert_market_candle = AsyncMock()
        journal.get_candle_counts_per_symbol = AsyncMock(return_value={})

        adapter = MagicMock()
        adapter._rest.get_kline = AsyncMock()

        screener = MagicMock()
        screener.wait_ready = AsyncMock()
        screener.active_symbols = ["TESTUSDT"]

        app._bybit_adapter = adapter
        app._trade_journal = journal
        app._screener = screener
        app._market_data_intervals = lambda: ["1"]
        app._shutdown_event.set()

        await asyncio.wait_for(app._run_startup_backfill(), timeout=2)
        adapter._rest.get_kline.assert_not_called()
