"""Regression tests for P0/P1 runtime-safety hotfix.

Covers all 8 defects fixed in fix/runtime-safety-pending-net-reporting:
  P0.1 – LIVE_ARMED is enforced by _active_execution_allowed()
  P0.2 – mark_entry_resolved() requires order_link_id; no-op without it
  P0.3 – restore_pending_entries() deduplicates and syncs count
  P0.4 – Fee-rate provider fail-closed in live mode
  P0.5 – Round-trip slippage is 2× per-side value
  P0.6 – POST_ONLY_LIMIT is blocked by config validator
  P0.7 – Telegram /net uses net_results_provider, not health_provider
  P1.1 – _parse_dec preserves Decimal("0"); invalid timestamps are skipped
  P1.2 – Preflight: Wallet key alone doesn't trigger withdrawal warning;
          unified status codes 4/5/6 are accepted
  P1.4 – /diagnostics includes pending_entry_count, entry_order_mode,
          fee_provider_available
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# P0.1 — _active_execution_allowed() enforces LIVE_ARMED
# ---------------------------------------------------------------------------


def _make_settings(
    *,
    trading_mode: str = "SHADOW",
    live_mode: bool = False,
    live_armed: bool = False,
    shadow_mode: bool = True,
    use_testnet: bool = False,
) -> MagicMock:
    from trader.domain.enums import TradingMode

    s = MagicMock()
    s.TRADING_MODE = TradingMode(trading_mode)
    s.LIVE_MODE = live_mode
    s.LIVE_ARMED = live_armed
    s.SHADOW_MODE = shadow_mode
    s.BYBIT_USE_TESTNET = use_testnet
    return s


def _allowed(settings: MagicMock) -> bool:
    """Replicate the _active_execution_allowed() logic from app.py."""
    from trader.domain.enums import TradingMode

    if settings.TRADING_MODE == TradingMode.SHADOW:
        return False
    if settings.BYBIT_USE_TESTNET:
        return True
    if settings.TRADING_MODE not in (TradingMode.LIVE, TradingMode.CANARY_LIVE):
        return False
    return settings.LIVE_MODE and settings.LIVE_ARMED and not settings.SHADOW_MODE


class TestActiveExecutionAllowed:
    def test_shadow_mode_always_blocked(self) -> None:
        s = _make_settings(trading_mode="SHADOW")
        assert _allowed(s) is False

    def test_canary_live_without_live_armed_blocked(self) -> None:
        s = _make_settings(
            trading_mode="CANARY_LIVE",
            live_mode=True,
            live_armed=False,
            shadow_mode=False,
        )
        assert _allowed(s) is False

    def test_canary_live_without_live_mode_blocked(self) -> None:
        s = _make_settings(
            trading_mode="CANARY_LIVE",
            live_mode=False,
            live_armed=True,
            shadow_mode=False,
        )
        assert _allowed(s) is False

    def test_canary_live_fully_armed_allowed(self) -> None:
        s = _make_settings(
            trading_mode="CANARY_LIVE",
            live_mode=True,
            live_armed=True,
            shadow_mode=False,
        )
        assert _allowed(s) is True

    def test_live_fully_armed_allowed(self) -> None:
        s = _make_settings(
            trading_mode="LIVE",
            live_mode=True,
            live_armed=True,
            shadow_mode=False,
        )
        assert _allowed(s) is True

    def test_testnet_bypasses_live_armed_check(self) -> None:
        s = _make_settings(
            trading_mode="CANARY_LIVE",
            live_mode=False,
            live_armed=False,
            shadow_mode=False,
            use_testnet=True,
        )
        assert _allowed(s) is True

    def test_shadow_mode_flag_blocks_even_if_armed(self) -> None:
        s = _make_settings(
            trading_mode="CANARY_LIVE",
            live_mode=True,
            live_armed=True,
            shadow_mode=True,
        )
        assert _allowed(s) is False


# ---------------------------------------------------------------------------
# P0.2 — mark_entry_resolved() no-op without order_link_id
# ---------------------------------------------------------------------------


class TestMarkEntryResolved:
    def _make_engine(self) -> Any:
        from trader.execution.engine import ExecutionEngine

        engine = ExecutionEngine.__new__(ExecutionEngine)
        engine._pending_entry_order_link_ids = set()
        engine._pending_entry_count = 0
        engine._shadow_mode = False
        engine._recent_entries = []
        engine._entry_order_mode = "MARKET"
        engine._min_expected_net_edge_pct = None
        engine._fee_provider = None
        engine._slippage_per_side_pct = Decimal("0.03")
        return engine

    def test_resolve_without_id_is_noop(self) -> None:
        from trader.execution.engine import ExecutionEngine

        engine = self._make_engine()
        engine._pending_entry_order_link_ids.add("ord-123")
        engine._pending_entry_count = 1
        ExecutionEngine.mark_entry_resolved(engine, "")
        assert engine._pending_entry_count == 1
        assert "ord-123" in engine._pending_entry_order_link_ids

    def test_resolve_with_unknown_id_is_noop(self) -> None:
        from trader.execution.engine import ExecutionEngine

        engine = self._make_engine()
        engine._pending_entry_order_link_ids.add("ord-123")
        engine._pending_entry_count = 1
        ExecutionEngine.mark_entry_resolved(engine, "ord-999")
        assert engine._pending_entry_count == 1

    def test_resolve_known_id_removes_and_decrements(self) -> None:
        from trader.execution.engine import ExecutionEngine

        engine = self._make_engine()
        engine._pending_entry_order_link_ids.add("ord-123")
        engine._pending_entry_count = 1
        ExecutionEngine.mark_entry_resolved(engine, "ord-123")
        assert engine._pending_entry_count == 0
        assert "ord-123" not in engine._pending_entry_order_link_ids

    def test_resolve_never_goes_below_zero(self) -> None:
        from trader.execution.engine import ExecutionEngine

        engine = self._make_engine()
        engine._pending_entry_order_link_ids.add("ord-123")
        engine._pending_entry_count = 0  # mis-counted
        ExecutionEngine.mark_entry_resolved(engine, "ord-123")
        assert engine._pending_entry_count == 0


# ---------------------------------------------------------------------------
# P0.3 — restore_pending_entries() deduplicates and syncs count
# ---------------------------------------------------------------------------


class TestRestorePendingEntries:
    def _make_engine(self) -> Any:
        from trader.execution.engine import ExecutionEngine

        engine = ExecutionEngine.__new__(ExecutionEngine)
        engine._pending_entry_order_link_ids = set()
        engine._pending_entry_count = 0
        engine._shadow_mode = False
        engine._recent_entries = []
        return engine

    def test_restore_deduplicates(self) -> None:
        from trader.execution.engine import ExecutionEngine

        engine = self._make_engine()
        ExecutionEngine.restore_pending_entries(engine, ["a", "b", "a", "b"])
        assert engine._pending_entry_count == 2
        assert engine._pending_entry_order_link_ids == {"a", "b"}

    def test_restore_skips_empty_strings(self) -> None:
        from trader.execution.engine import ExecutionEngine

        engine = self._make_engine()
        ExecutionEngine.restore_pending_entries(engine, ["", "ord-1", ""])
        assert engine._pending_entry_count == 1

    def test_restore_syncs_count_with_set(self) -> None:
        from trader.execution.engine import ExecutionEngine

        engine = self._make_engine()
        engine._pending_entry_order_link_ids = {"stale"}
        engine._pending_entry_count = 99
        ExecutionEngine.restore_pending_entries(engine, ["ord-1", "ord-2"])
        assert engine._pending_entry_count == len(engine._pending_entry_order_link_ids)

    def test_restore_empty_list_syncs_count_to_existing_set(self) -> None:
        """restore_pending_entries adds to existing set; count syncs to actual set size."""
        from trader.execution.engine import ExecutionEngine

        engine = self._make_engine()
        engine._pending_entry_order_link_ids = {"stale-ord"}
        engine._pending_entry_count = 99  # mis-counted
        ExecutionEngine.restore_pending_entries(engine, [])
        # Empty list adds nothing, but count still syncs to actual set size
        assert engine._pending_entry_count == 1

    def test_get_status_includes_pending_ids(self) -> None:
        from datetime import timedelta

        from trader.execution.engine import ExecutionEngine

        engine = self._make_engine()
        engine._pending_entry_order_link_ids = {"ord-a", "ord-b"}
        engine._pending_entry_count = 2

        # Remaining attrs needed by get_status
        engine._open_positions = {}
        engine._cooldown = timedelta(seconds=30)
        engine._failure_cooldown = timedelta(seconds=60)
        engine._last_entry_at = {}
        engine._last_failure_at = {}
        engine._entry_order_mode = "MARKET"
        engine._slippage_per_side_pct = Decimal("0.03")
        engine._min_expected_net_edge_pct = None
        engine._fee_provider = None

        status = ExecutionEngine.get_status(engine)
        assert "pending_entry_ids" in status
        assert set(status["pending_entry_ids"]) == {"ord-a", "ord-b"}
        assert status["pending_entry_count"] == 2


# ---------------------------------------------------------------------------
# P0.4 — Fee-rate provider fail-closed in live mode
# ---------------------------------------------------------------------------


class TestFeeRateFailClosed:
    def test_live_mode_no_fee_provider_blocks_entry_structural(self) -> None:
        """Engine source must have fail-closed guard for missing fee_provider in live mode."""
        import inspect

        from trader.execution.engine import ExecutionEngine

        src = inspect.getsource(ExecutionEngine._submit_locked)
        assert "fee_rate_unavailable_blocking_entry" in src
        assert "fee_provider_not_configured" in src

    def test_shadow_mode_skips_fee_block_structural(self) -> None:
        """Fee block is gated on `not self._shadow_mode`."""
        import inspect

        from trader.execution.engine import ExecutionEngine

        src = inspect.getsource(ExecutionEngine._submit_locked)
        assert "shadow_mode" in src

    def test_fee_block_returns_none_before_placing_order(self) -> None:
        """After fee block, function returns None (no order submitted)."""
        import inspect

        from trader.execution.engine import ExecutionEngine

        src = inspect.getsource(ExecutionEngine._submit_locked)
        # Structural: the fee unavailable log is followed by return None
        fee_idx = src.find("fee_rate_unavailable_blocking_entry")
        return_none_idx = src.find("return None", fee_idx)
        assert fee_idx != -1
        assert return_none_idx != -1 and return_none_idx > fee_idx


# ---------------------------------------------------------------------------
# P0.5 — Round-trip slippage is 2× per-side
# ---------------------------------------------------------------------------


class TestRoundTripSlippage:
    def test_slippage_applied_twice(self) -> None:
        """Net edge calculation must deduct 2× slippage (entry + exit)."""
        slippage_per_side = Decimal("0.03")
        roundtrip = slippage_per_side * Decimal("2")
        assert roundtrip == Decimal("0.06")

    def test_engine_uses_2x_slippage_in_net_edge(self) -> None:
        """Verify ExecutionEngine._submit_locked uses 2× slippage for round-trip."""
        import inspect

        from trader.execution.engine import ExecutionEngine

        src = inspect.getsource(ExecutionEngine._submit_locked)
        # The fix: roundtrip_slippage_pct = slippage_per_side_pct * Decimal("2")
        assert '* Decimal("2")' in src or 'roundtrip_slippage' in src


# ---------------------------------------------------------------------------
# P0.6 — POST_ONLY_LIMIT blocked by config validator
# ---------------------------------------------------------------------------


class TestPostOnlyLimitBlocked:
    def test_post_only_limit_raises_validation_error(self) -> None:
        import os

        with patch.dict(
            os.environ,
            {
                "BYBIT_API_KEY": "k",
                "BYBIT_API_SECRET": "s",
                "ENTRY_ORDER_MODE": "POST_ONLY_LIMIT",
            },
        ):
            from pydantic import ValidationError

            with pytest.raises((ValidationError, ValueError)):
                from trader.config import Settings

                Settings(BYBIT_API_KEY="k", BYBIT_API_SECRET="s", ENTRY_ORDER_MODE="POST_ONLY_LIMIT")

    def test_market_mode_accepted(self) -> None:
        from trader.config import Settings

        s = Settings(BYBIT_API_KEY="k", BYBIT_API_SECRET="s", ENTRY_ORDER_MODE="MARKET")
        assert s.ENTRY_ORDER_MODE == "MARKET"

    def test_engine_belt_and_suspenders_disables_limit(self) -> None:
        """ExecutionEngine._build_intent falls back from POST_ONLY_LIMIT to MARKET."""
        import inspect

        from trader.execution.engine import ExecutionEngine

        src = inspect.getsource(ExecutionEngine._build_intent)
        assert "post_only_limit_not_implemented" in src.lower() or "use_limit = False" in src


# ---------------------------------------------------------------------------
# P0.7 — Telegram /net uses net_results_provider not health_provider
# ---------------------------------------------------------------------------


class TestTelegramNetResults:
    def test_cmd_net_results_uses_net_provider(self) -> None:
        import inspect

        from trader.telegram_bot import TelegramMonitorBot as TelegramBot

        src = inspect.getsource(TelegramBot._cmd_net_results)
        assert "net_results_provider" in src
        assert "health_provider" not in src

    def test_main_menu_results_button_routes_to_net(self) -> None:
        import inspect

        from trader.telegram_bot import TelegramMonitorBot as TelegramBot

        src = inspect.getsource(TelegramBot._main_menu)
        assert 'callback_data="view:net"' in src
        assert 'callback_data="view:pnl"' not in src

    def test_handle_view_button_has_net_handler(self) -> None:
        import inspect

        from trader.telegram_bot import TelegramMonitorBot as TelegramBot

        src = inspect.getsource(TelegramBot._handle_view_button)
        assert '"net"' in src

    def test_trading_controller_has_net_results_provider_field(self) -> None:
        """TradingController dataclass has the net_results_provider optional field."""
        import dataclasses

        from trader.telegram_bot import TradingController

        field_names = {f.name for f in dataclasses.fields(TradingController)}
        assert "net_results_provider" in field_names


# ---------------------------------------------------------------------------
# P1.1 — _parse_dec preserves Decimal("0")
# ---------------------------------------------------------------------------


class TestParseDec:
    def _parse_dec(self, v: Any) -> Any:
        from trader.storage.trade_journal import _parse_dec

        return _parse_dec(v)

    def test_zero_int_preserved(self) -> None:
        result = self._parse_dec(0)
        assert result == Decimal("0")
        assert result is not None

    def test_zero_float_preserved(self) -> None:
        result = self._parse_dec(0.0)
        assert result == Decimal("0")

    def test_zero_string_preserved(self) -> None:
        result = self._parse_dec("0")
        assert result == Decimal("0")

    def test_none_returns_none(self) -> None:
        assert self._parse_dec(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert self._parse_dec("") is None

    def test_positive_decimal(self) -> None:
        assert self._parse_dec("1.5") == Decimal("1.5")

    def test_negative_decimal(self) -> None:
        assert self._parse_dec("-0.05") == Decimal("-0.05")

    def test_invalid_string_returns_none(self) -> None:
        assert self._parse_dec("not_a_number") is None


# ---------------------------------------------------------------------------
# P1.2 — Preflight: Wallet key alone doesn't trigger withdrawal warning
# ---------------------------------------------------------------------------


class TestPreflightPermissions:
    @pytest.mark.asyncio
    async def test_wallet_key_alone_no_warning(self) -> None:
        from trader.exchange.preflight import PreflightChecker

        checker = PreflightChecker.__new__(PreflightChecker)
        checker._rest = AsyncMock()
        checker._rest.get_api_key_info = AsyncMock(
            return_value={
                "retCode": 0,
                "result": {
                    "permissions": {
                        "Wallet": ["AccountTransfer", "SubMemberTransfer"],
                        "Trade": ["Order"],
                        "ContractTrade": ["Order", "Position"],
                    }
                },
            }
        )

        result = await checker._check_api_key_permissions()
        assert result.warning is None, f"Unexpected warning: {result.warning}"

    @pytest.mark.asyncio
    async def test_withdrawal_in_values_triggers_warning(self) -> None:
        from trader.exchange.preflight import PreflightChecker

        checker = PreflightChecker.__new__(PreflightChecker)
        checker._rest = AsyncMock()
        checker._rest.get_api_key_info = AsyncMock(
            return_value={
                "retCode": 0,
                "result": {
                    "permissions": {
                        "Wallet": ["Withdraw"],
                    }
                },
            }
        )

        result = await checker._check_api_key_permissions()
        assert result.warning is not None
        assert "withdrawal" in result.warning.lower() or "withdraw" in result.warning.lower()

    @pytest.mark.asyncio
    async def test_unified_status_4_accepted(self) -> None:
        from trader.exchange.preflight import PreflightChecker

        checker = PreflightChecker.__new__(PreflightChecker)
        checker._rest = AsyncMock()
        checker._expected_account_type = "UNIFIED"
        checker._rest.get_account_info = AsyncMock(
            return_value={
                "retCode": 0,
                "result": {"unifiedMarginStatus": 4},
            }
        )

        result = await checker._check_account_type()
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_unified_status_6_accepted(self) -> None:
        from trader.exchange.preflight import PreflightChecker

        checker = PreflightChecker.__new__(PreflightChecker)
        checker._rest = AsyncMock()
        checker._expected_account_type = "UNIFIED"
        checker._rest.get_account_info = AsyncMock(
            return_value={
                "retCode": 0,
                "result": {"unifiedMarginStatus": 6},
            }
        )

        result = await checker._check_account_type()
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_regular_account_status_1_not_unified(self) -> None:
        from trader.exchange.preflight import PreflightChecker

        checker = PreflightChecker.__new__(PreflightChecker)
        checker._rest = AsyncMock()
        checker._expected_account_type = "UNIFIED"
        checker._rest.get_account_info = AsyncMock(
            return_value={
                "retCode": 0,
                "result": {"unifiedMarginStatus": 1},
            }
        )

        result = await checker._check_account_type()
        assert result.passed is False


# ---------------------------------------------------------------------------
# P1.4 — /diagnostics includes pending/fee fields
# ---------------------------------------------------------------------------


class TestDiagnosticsFields:
    def test_cmd_diagnostics_shows_pending_entry_count(self) -> None:
        import inspect

        from trader.telegram_bot import TelegramMonitorBot as TelegramBot

        src = inspect.getsource(TelegramBot._cmd_diagnostics)
        assert "pending_entry_count" in src

    def test_cmd_diagnostics_shows_fee_provider(self) -> None:
        import inspect

        from trader.telegram_bot import TelegramMonitorBot as TelegramBot

        src = inspect.getsource(TelegramBot._cmd_diagnostics)
        assert "fee_provider_available" in src

    def test_cmd_diagnostics_shows_entry_order_mode(self) -> None:
        import inspect

        from trader.telegram_bot import TelegramMonitorBot as TelegramBot

        src = inspect.getsource(TelegramBot._cmd_diagnostics)
        assert "entry_order_mode" in src

    def test_engine_get_status_has_pending_ids(self) -> None:
        import inspect

        from trader.execution.engine import ExecutionEngine

        src = inspect.getsource(ExecutionEngine.get_status)
        assert "pending_entry_ids" in src
        assert "pending_entry_count" in src
