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


class TestShadowGateStatsFeatureSchemaFilter:
    """Test that get_shadow_gate_stats() filters by feature_schema_hash."""

    def test_sql_contains_schema_filter(self) -> None:
        import inspect

        src = inspect.getsource(TradeJournal.get_shadow_gate_stats)
        # Verify the feature_schema_hash filter exists in WHERE clause
        assert "fs.feature_schema_hash = $4" in src
        # Verify the parameter is passed
        assert "feature_schema_hash," in src

    def test_returns_empty_when_model_has_no_schema_hash(self) -> None:
        """If model has no feature_schema_hash, stats should be empty to avoid mixing."""
        import asyncio
        from unittest.mock import MagicMock

        journal = TradeJournal(postgres_dsn="", enabled=False)
        journal._enabled = True
        journal._pool = MagicMock()

        # Mock model with no feature_schema_hash
        async def mock_fetch(query, *args):
            if "model_versions" in query:
                return [{"feature_schema_hash": None, "metrics": None}]
            return []

        journal._fetch = mock_fetch

        result = asyncio.run(journal.get_shadow_gate_stats("test-model", 15, "directional_net_v1"))

        assert result == {"model_version": "test-model", "feature_schema_hash": ""}

    def test_includes_schema_hash_in_query_params(self) -> None:
        """The query should use the model's feature_schema_hash as a parameter."""
        import asyncio
        from unittest.mock import MagicMock

        journal = TradeJournal(postgres_dsn="", enabled=False)
        journal._enabled = True
        journal._pool = MagicMock()

        captured_params = {}

        async def mock_fetch(query, *args):
            captured_params["query"] = query
            captured_params["args"] = args
            if "model_versions" in query:
                return [{"feature_schema_hash": "abc123", "metrics": None}]
            return []

        journal._fetch = mock_fetch

        asyncio.run(journal.get_shadow_gate_stats("test-model", 15, "directional_net_v1"))

        # Verify feature_schema_hash is passed as 4th parameter
        assert captured_params["args"][3] == "abc123"
        # Verify the SQL uses $4 for the schema hash
        assert "fs.feature_schema_hash = $4" in captured_params["query"]


class TestReverseLookupExcludesUnknown:
    """Test that find_order_link_id_by_exchange_order_id() rejects unknown:* IDs."""

    def test_durable_rejects_unknown(self) -> None:
        import asyncio
        from unittest.mock import MagicMock

        journal = TradeJournal(postgres_dsn="", enabled=False)
        journal._enabled = True
        journal._pool = MagicMock()

        async def mock_fetch(query, *args):
            if "durable_order_state" in query:
                return [{"order_link_id": "unknown:123"}]
            if "order_events" in query:
                return [{"order_link_id": "valid-fallback-id"}]
            return []

        journal._fetch = mock_fetch

        result = asyncio.run(journal.find_order_link_id_by_exchange_order_id("exch-123"))
        # Should skip unknown and fall back to order_events
        assert result == "valid-fallback-id"

    def test_fallback_rejects_unknown(self) -> None:
        import asyncio
        from unittest.mock import MagicMock

        journal = TradeJournal(postgres_dsn="", enabled=False)
        journal._enabled = True
        journal._pool = MagicMock()

        async def mock_fetch(query, *args):
            if "durable_order_state" in query:
                return []
            if "order_events" in query:
                return [{"order_link_id": "unknown:456"}]
            return []

        journal._fetch = mock_fetch

        result = asyncio.run(journal.find_order_link_id_by_exchange_order_id("exch-456"))
        assert result is None

    def test_returns_real_local_id(self) -> None:
        import asyncio
        from unittest.mock import MagicMock

        journal = TradeJournal(postgres_dsn="", enabled=False)
        journal._enabled = True
        journal._pool = MagicMock()

        async def mock_fetch(query, *args):
            if "durable_order_state" in query:
                return [{"order_link_id": "local-abc-123"}]
            return []

        journal._fetch = mock_fetch

        result = asyncio.run(journal.find_order_link_id_by_exchange_order_id("exch-789"))
        assert result == "local-abc-123"


class TestLoadPendingFromDbExcludesUnknown:
    """Test that load_pending_from_db() SQL contains the unknown:* filter."""

    def test_sql_contains_unknown_filter(self) -> None:
        import inspect

        src = inspect.getsource(TradeJournal.load_pending_from_db)
        assert "order_link_id NOT LIKE 'unknown:%'" in src
        assert re.search(r"WHERE.*order_link_id NOT LIKE 'unknown:%'", src, re.DOTALL)
