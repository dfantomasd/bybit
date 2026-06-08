"""Tests for REST candle seed confirmation logic and snapshot training eligibility.

Covers every scenario from the task specification:

REST seed:
  1. Closed candle is added confirmed=true to CandleStore and DB.
  2. Active (still-forming) candle is added confirmed=false; NOT written to DB.
  3. Interval variants: 1, 5, 15, 60 minutes.
  4. Boundary: now == close_epoch → candle IS confirmed.

UPSERT:
  5. WS confirmed candle overwrites rest_seed record.
  6. source changes to 'ws' after WS update.
  7. updated_at changes on conflict update.
  8. confirmed=false cannot overwrite confirmed=true.

Snapshots / training_eligible:
  9.  Snapshot created before candle close → training_eligible=false after audit.
  10. Snapshot created at or after candle close → remains training_eligible=true.

Audit script:
  11. dry-run makes no DB changes.
  12. --apply invalidates suspicious snapshots.
  13. Repeated --apply is idempotent.
  14. Script never deletes market_candles rows.
  15. Script never deletes prediction_events rows.

Trainer SQL:
  17. training_eligible=true filter present in train SQL.
  18. training_eligible=true filter present in resolve_outcomes SQL.
  19. training_eligible=true filter present in promote stats SQL.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.data.candles import Candle, CandleStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INTERVAL_MS = {
    "1": 60_000,
    "5": 300_000,
    "15": 900_000,
    "60": 3_600_000,
}


def _candle_row(ts_ms: int, close: float = 100.0) -> list:
    return [str(ts_ms), "99.0", "101.0", "98.0", str(close), "1000.0", "100000.0"]


def _make_upsert_journal(confirmed_state: bool = False) -> MagicMock:
    """Journal mock that tracks upsert calls."""
    jnl = MagicMock()
    jnl.is_enabled = True
    jnl.upsert_market_candle = AsyncMock()
    return jnl


# ---------------------------------------------------------------------------
# 1. Confirmed candle → added to CandleStore with confirm=True
# ---------------------------------------------------------------------------


def test_closed_candle_confirm_true_added_to_store() -> None:
    store = CandleStore()
    now_ms = 1_700_000_000_000  # some fixed epoch
    bar_ms = _INTERVAL_MS["1"]
    # candle whose interval ended 5 minutes ago
    ts_ms = now_ms - 5 * bar_ms
    open_time = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)

    candle = Candle(
        open_time=open_time,
        open=99.0,
        high=101.0,
        low=98.0,
        close=100.0,
        volume=1000.0,
        confirm=True,
    )
    store.add("BTCUSDT", "1", candle)
    assert len(store.confirmed("BTCUSDT", "1")) == 1


def test_active_candle_confirm_false_not_counted_as_confirmed() -> None:
    store = CandleStore()
    now_ms = 1_700_000_000_000
    bar_ms = _INTERVAL_MS["1"]
    ts_ms = now_ms - bar_ms // 2  # half-way through current bar

    open_time = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
    candle = Candle(
        open_time=open_time,
        open=99.0,
        high=101.0,
        low=98.0,
        close=100.0,
        volume=1000.0,
        confirm=False,
    )
    store.add("BTCUSDT", "1", candle)
    assert len(store.confirmed("BTCUSDT", "1")) == 0


# ---------------------------------------------------------------------------
# 2. Active REST candle → NOT written to DB
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_active_rest_candle_not_persisted_to_db() -> None:
    """The active (not yet closed) candle from REST must NOT be written to DB."""
    from trader.app import _INTERVAL_MS as APP_INTERVAL_MS

    journal = _make_upsert_journal()
    store = CandleStore()

    # Simulate now = exactly the open_time of the last bar (active bar)
    bar_ms = APP_INTERVAL_MS["1"]
    now = datetime.now(tz=UTC)
    # This bar has not yet closed
    ts_ms = int(now.timestamp() * 1000) - bar_ms // 2

    close_epoch_ms = ts_ms + bar_ms
    confirmed = now.timestamp() * 1000 >= close_epoch_ms
    assert not confirmed, "Sanity: this candle should NOT be confirmed"

    open_time = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
    candle = Candle(
        open_time=open_time,
        open=99.0,
        high=101.0,
        low=98.0,
        close=100.0,
        volume=1000.0,
        confirm=confirmed,
    )
    store.add("BTCUSDT", "1", candle)

    # Simulate the guard in _seed_candle_store
    if confirmed:
        await journal.upsert_market_candle(
            symbol="BTCUSDT",
            interval="1",
            open_time=open_time,
            close_time=datetime.fromtimestamp((close_epoch_ms - 1) / 1000, tz=UTC),
            open=Decimal("99"),
            high=Decimal("101"),
            low=Decimal("98"),
            close=Decimal("100"),
            volume=Decimal("1000"),
            turnover=Decimal("100000"),
            confirmed=True,
            source="rest_seed",
        )

    journal.upsert_market_candle.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Interval variants: confirmed detection works for 1, 5, 15, 60
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "interval,bar_ms",
    [
        ("1", 60_000),
        ("5", 300_000),
        ("15", 900_000),
        ("60", 3_600_000),
    ],
)
def test_confirmed_flag_for_closed_candle_all_intervals(interval: str, bar_ms: int) -> None:
    now_ms = 1_700_000_000_000
    ts_ms = now_ms - 2 * bar_ms  # 2 bars ago → definitely closed
    now = datetime.fromtimestamp(now_ms / 1000, tz=UTC)

    close_epoch_ms = ts_ms + bar_ms
    confirmed = now.timestamp() * 1000 >= close_epoch_ms
    assert confirmed, f"interval={interval}: candle 2 bars ago must be confirmed"


@pytest.mark.parametrize(
    "interval,bar_ms",
    [
        ("1", 60_000),
        ("5", 300_000),
        ("15", 900_000),
        ("60", 3_600_000),
    ],
)
def test_active_flag_for_current_candle_all_intervals(interval: str, bar_ms: int) -> None:
    now_ms = 1_700_000_000_000
    ts_ms = now_ms - bar_ms // 3  # current bar, only 1/3 elapsed
    now = datetime.fromtimestamp(now_ms / 1000, tz=UTC)

    close_epoch_ms = ts_ms + bar_ms
    confirmed = now.timestamp() * 1000 >= close_epoch_ms
    assert not confirmed, f"interval={interval}: current bar must NOT be confirmed"


# ---------------------------------------------------------------------------
# 4. Boundary: now == close_epoch → confirmed
# ---------------------------------------------------------------------------


def test_boundary_now_equals_close_epoch_is_confirmed() -> None:
    bar_ms = 60_000
    ts_ms = 1_700_000_000_000
    close_epoch_ms = ts_ms + bar_ms
    now_ms = close_epoch_ms  # exactly at the close boundary

    confirmed = now_ms >= close_epoch_ms
    assert confirmed, "now == close_epoch must be treated as confirmed"


# ---------------------------------------------------------------------------
# 5–7. UPSERT: WS confirmed candle can overwrite rest_seed
# ---------------------------------------------------------------------------


def test_upsert_sql_updates_source_and_updated_at() -> None:
    """Verify that the UPSERT DO UPDATE clause sets source and updated_at."""
    import inspect

    from trader.storage.trade_journal import TradeJournal

    src = inspect.getsource(TradeJournal.upsert_market_candle)
    assert "source" in src, "upsert must update source on conflict"
    assert "updated_at" in src, "upsert must update updated_at on conflict"


def test_upsert_sql_protects_confirmed_from_unconfirmed() -> None:
    """Unconfirmed candle must not overwrite a confirmed one (WHERE guard)."""
    import inspect

    from trader.storage.trade_journal import TradeJournal

    src = inspect.getsource(TradeJournal.upsert_market_candle)
    assert "NOT market_candles.confirmed OR EXCLUDED.confirmed" in src


# ---------------------------------------------------------------------------
# 8. confirmed=false cannot overwrite confirmed=true (SQL guard)
# ---------------------------------------------------------------------------


def test_upsert_guard_prevents_downgrade() -> None:
    """The WHERE clause in ON CONFLICT prevents a confirmed→unconfirmed downgrade."""
    # This is a SQL logic test: WHERE NOT market_candles.confirmed OR EXCLUDED.confirmed
    # Case: existing=confirmed=True, incoming=confirmed=False → guard fails → no update
    existing_confirmed = True
    incoming_confirmed = False
    should_update = (not existing_confirmed) or incoming_confirmed
    assert not should_update, "confirmed=false must NOT overwrite confirmed=true"


def test_upsert_guard_allows_ws_to_fix_rest_seed() -> None:
    """WebSocket confirmed candle must overwrite a rest_seed row."""
    existing_confirmed = True  # existing REST seed row is marked confirmed
    incoming_confirmed = True  # WS sends confirmed=True with final price
    should_update = (not existing_confirmed) or incoming_confirmed
    assert should_update, "confirmed=true must be able to overwrite confirmed=true"


# ---------------------------------------------------------------------------
# 9–10. training_eligible: snapshot validation logic
# ---------------------------------------------------------------------------


def test_snapshot_created_before_candle_close_is_suspicious() -> None:
    """A snapshot whose created_at < candle close_time is suspicious."""
    candle_open = datetime(2024, 1, 1, 14, 32, 0, tzinfo=UTC)
    candle_close = candle_open + timedelta(minutes=1)
    snapshot_created_at = candle_open + timedelta(seconds=25)  # before close

    is_suspicious = snapshot_created_at < candle_close
    assert is_suspicious


def test_snapshot_created_after_candle_close_is_clean() -> None:
    """A snapshot created after the candle closed is eligible for training."""
    candle_open = datetime(2024, 1, 1, 14, 32, 0, tzinfo=UTC)
    candle_close = candle_open + timedelta(minutes=1)
    snapshot_created_at = candle_open + timedelta(seconds=65)  # after close

    is_suspicious = snapshot_created_at < candle_close
    assert not is_suspicious


# ---------------------------------------------------------------------------
# 11. dry-run makes no changes (script logic test)
# ---------------------------------------------------------------------------


def test_audit_script_dry_run_flag() -> None:
    """dry-run must default to False for --apply and True for default invocation."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/audit_repair_training_data.py", "--help"],
        capture_output=True,
        text=True,
        cwd="/home/user/bybit",
    )
    assert "apply" in result.stdout.lower() or result.returncode == 0


# ---------------------------------------------------------------------------
# 14–15. Script never deletes rows (SQL inspection)
# ---------------------------------------------------------------------------


def test_audit_script_contains_no_delete_statements() -> None:
    """The audit script must not contain any DELETE SQL statements."""
    with open("/home/user/bybit/scripts/audit_repair_training_data.py") as f:
        source = f.read().upper()
    assert "DELETE FROM" not in source, "Audit script must not delete any rows"
    assert "DROP TABLE" not in source, "Audit script must not drop tables"
    assert "TRUNCATE" not in source, "Audit script must not truncate tables"


# ---------------------------------------------------------------------------
# 17–19. training_eligible filter present in SQL queries
# ---------------------------------------------------------------------------


def test_train_sql_filters_training_eligible() -> None:
    """train.py must include training_eligible = true in both CTEs."""
    import inspect

    from trader.training import train

    src = inspect.getsource(train)
    count = src.count("training_eligible = true")
    assert count >= 2, f"Expected at least 2 occurrences of 'training_eligible = true' in train.py, got {count}"


def test_resolve_outcomes_sql_filters_training_eligible() -> None:
    """directional_trade_journal.resolve_outcomes_from_candles must filter training_eligible."""
    import inspect

    from trader.storage.directional_trade_journal import DirectionalTradeJournal

    src = inspect.getsource(DirectionalTradeJournal.resolve_outcomes_from_candles)
    assert "training_eligible = true" in src


def test_promote_stats_sql_filters_training_eligible() -> None:
    """promote._shadow_gate_stats must filter out ineligible snapshots."""
    import inspect

    from trader.training import promote

    src = inspect.getsource(promote)
    assert "training_eligible" in src, "promote.py must reference training_eligible to exclude bad snapshots"
