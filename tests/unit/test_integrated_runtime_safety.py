"""Integration tests for runtime safety: exclude phantom pending IDs from recovery."""

import re

from src.trader.execution.engine import ExecutionEngine
from src.trader.storage.trade_journal import TradeJournal


class TestGetPendingDurableOrdersExcludesUnknown:
    """Test that get_pending_durable_orders() SQL contains the unknown:* filter."""

    def test_sql_contains_unknown_filter(self) -> None:
        import inspect

        src = inspect.getsource(TradeJournal.get_pending_durable_orders)
        # Verify the SQL filter exists
        assert "order_link_id NOT LIKE 'unknown:%'" in src
        # Verify it's in the WHERE clause context
        assert re.search(r"WHERE.*order_link_id NOT LIKE 'unknown:%'", src, re.DOTALL)


class TestRestorePendingEntriesExcludesUnknown:
    """Test that restore_pending_entries() filters out unknown:* IDs."""

    def test_excludes_empty_ids(self) -> None:
        ee = ExecutionEngine(
            adapter=None,
            risk_manager=None,
            exposure_tracker=None,
            shadow_mode=True,
        )
        ee.restore_pending_entries(["", "valid-id-1", "", "valid-id-2"])
        assert ee._pending_entry_order_link_ids == {"valid-id-1", "valid-id-2"}
        assert ee._pending_entry_count == 2

    def test_excludes_unknown_prefix(self) -> None:
        ee = ExecutionEngine(
            adapter=None,
            risk_manager=None,
            exposure_tracker=None,
            shadow_mode=True,
        )
        ee.restore_pending_entries(["unknown:123", "valid-id", "unknown:abc"])
        assert ee._pending_entry_order_link_ids == {"valid-id"}
        assert ee._pending_entry_count == 1

    def test_excludes_both_empty_and_unknown(self) -> None:
        ee = ExecutionEngine(
            adapter=None,
            risk_manager=None,
            exposure_tracker=None,
            shadow_mode=True,
        )
        ee.restore_pending_entries(["", "unknown:foo", "valid-id", "unknown:bar", ""])
        assert ee._pending_entry_order_link_ids == {"valid-id"}
        assert ee._pending_entry_count == 1


class TestRestorePendingEntriesWithSymbolsExcludesUnknown:
    """Test that restore_pending_entries_with_symbols() filters out unknown:* IDs."""

    def test_excludes_empty_ids(self) -> None:
        ee = ExecutionEngine(
            adapter=None,
            risk_manager=None,
            exposure_tracker=None,
            shadow_mode=True,
        )
        ee.restore_pending_entries_with_symbols(
            [
                {"order_link_id": "", "symbol": "BTCUSDT"},
                {"order_link_id": "valid-id-1", "symbol": "BTCUSDT"},
                {"order_link_id": "valid-id-2", "symbol": "ETHUSDT"},
            ]
        )
        assert ee._pending_entry_order_link_ids == {"valid-id-1", "valid-id-2"}
        assert ee._pending_entry_symbols == {"valid-id-1": "BTCUSDT", "valid-id-2": "ETHUSDT"}
        assert ee._pending_entry_count == 2

    def test_excludes_unknown_prefix(self) -> None:
        ee = ExecutionEngine(
            adapter=None,
            risk_manager=None,
            exposure_tracker=None,
            shadow_mode=True,
        )
        ee.restore_pending_entries_with_symbols(
            [
                {"order_link_id": "unknown:123", "symbol": "BTCUSDT"},
                {"order_link_id": "valid-id", "symbol": "ETHUSDT"},
                {"order_link_id": "unknown:abc", "symbol": "SOLUSDT"},
            ]
        )
        assert ee._pending_entry_order_link_ids == {"valid-id"}
        assert ee._pending_entry_symbols == {"valid-id": "ETHUSDT"}
        assert ee._pending_entry_count == 1

    def test_excludes_both_empty_and_unknown(self) -> None:
        ee = ExecutionEngine(
            adapter=None,
            risk_manager=None,
            exposure_tracker=None,
            shadow_mode=True,
        )
        ee.restore_pending_entries_with_symbols(
            [
                {"order_link_id": "", "symbol": "BTCUSDT"},
                {"order_link_id": "unknown:foo", "symbol": "BTCUSDT"},
                {"order_link_id": "valid-id", "symbol": "ETHUSDT"},
                {"order_link_id": "unknown:bar", "symbol": "SOLUSDT"},
                {"order_link_id": "", "symbol": "XRPUSDT"},
            ]
        )
        assert ee._pending_entry_order_link_ids == {"valid-id"}
        assert ee._pending_entry_symbols == {"valid-id": "ETHUSDT"}
        assert ee._pending_entry_count == 1

    def test_deduplicates_same_id(self) -> None:
        ee = ExecutionEngine(
            adapter=None,
            risk_manager=None,
            exposure_tracker=None,
            shadow_mode=True,
        )
        ee.restore_pending_entries_with_symbols(
            [
                {"order_link_id": "valid-id", "symbol": "BTCUSDT"},
                {"order_link_id": "valid-id", "symbol": "ETHUSDT"},
            ]
        )
        assert ee._pending_entry_order_link_ids == {"valid-id"}
        assert ee._pending_entry_count == 1
        # First symbol should win
        assert ee._pending_entry_symbols["valid-id"] == "BTCUSDT"
