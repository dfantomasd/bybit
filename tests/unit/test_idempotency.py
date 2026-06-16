"""Tests for IdempotencyManager."""

from __future__ import annotations

import uuid
from datetime import UTC
from decimal import Decimal

import pytest

from trader.domain.enums import MarketType, OrderSide, OrderStatus, OrderType
from trader.domain.errors import OrderRejectedError
from trader.domain.models import OrderIntent
from trader.exchange.idempotency import IdempotencyManager

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_intent(order_link_id: str) -> OrderIntent:
    return OrderIntent(
        decision_id=uuid.uuid4(),
        proposal_id=uuid.uuid4(),
        symbol="BTCUSDT",
        market_type=MarketType.LINEAR,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        qty=Decimal("0.01"),
        price=Decimal("30000"),
        order_link_id=order_link_id,
    )


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------


class TestGenerateOrderLinkId:
    def setup_method(self) -> None:
        self.mgr = IdempotencyManager()

    def test_generated_id_length_le_36(self) -> None:
        oid = self.mgr.generate_order_link_id("TESTNET", "momentum", str(uuid.uuid4()))
        assert len(oid) <= 36, f"ID too long: {oid!r} (len={len(oid)})"

    def test_generated_id_is_unique(self) -> None:
        ids = {self.mgr.generate_order_link_id("TESTNET", "momentum", str(uuid.uuid4())) for _ in range(100)}
        assert len(ids) == 100, "Generated IDs are not unique"

    def test_id_contains_env_short(self) -> None:
        oid = self.mgr.generate_order_link_id("TESTNET", "momentum", str(uuid.uuid4()))
        assert oid.startswith("TN-"), f"Expected TN prefix, got: {oid}"

    def test_id_contains_date(self) -> None:
        from datetime import datetime

        today = datetime.now(tz=UTC).strftime("%y%m%d")
        oid = self.mgr.generate_order_link_id("TESTNET", "momentum", str(uuid.uuid4()))
        assert today in oid

    def test_different_envs_produce_different_prefixes(self) -> None:
        prop_id = str(uuid.uuid4())
        live_id = self.mgr.generate_order_link_id("LIVE", "mom", prop_id)
        test_id = self.mgr.generate_order_link_id("TESTNET", "mom", prop_id)
        assert live_id[:2] != test_id[:2]

    def test_strategy_id_truncated_to_4_chars(self) -> None:
        oid = self.mgr.generate_order_link_id("TESTNET", "momentum_strategy", str(uuid.uuid4()))
        parts = oid.split("-")
        # Format: TN-YYMMDD-SSSS-PPPPPPPP-RRRRRR
        strat_part = parts[2]
        assert len(strat_part) <= 4


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


class TestDuplicateDetection:
    async def test_check_duplicate_false_for_new_id(self) -> None:
        mgr = IdempotencyManager()
        assert await mgr.check_duplicate("fresh-id") is False

    async def test_check_duplicate_true_after_register(self) -> None:
        mgr = IdempotencyManager()
        intent = _make_intent("TN-260605-MOMO-PROP1234-abc123")
        await mgr.register_intent(intent)
        assert await mgr.check_duplicate("TN-260605-MOMO-PROP1234-abc123") is True

    async def test_register_same_id_twice_raises(self) -> None:
        mgr = IdempotencyManager()
        intent = _make_intent("TN-260605-MOMO-PROP1234-abc123")
        await mgr.register_intent(intent)
        with pytest.raises(OrderRejectedError):
            await mgr.register_intent(intent)


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


class TestStateTransitions:
    async def _setup(self) -> tuple[IdempotencyManager, str]:
        mgr = IdempotencyManager()
        oid = "TN-260605-MOMO-PROP1234-abc123"
        intent = _make_intent(oid)
        await mgr.register_intent(intent)
        return mgr, oid

    async def test_initial_state_is_created_local(self) -> None:
        mgr, oid = await self._setup()
        assert await mgr.get_state(oid) == OrderStatus.CREATED_LOCAL

    async def test_mark_submitted_transitions_to_submitting(self) -> None:
        mgr, oid = await self._setup()
        await mgr.mark_submitted(oid)
        assert await mgr.get_state(oid) == OrderStatus.SUBMITTING

    async def test_mark_confirmed_transitions_to_rest_accepted(self) -> None:
        mgr, oid = await self._setup()
        await mgr.mark_submitted(oid)
        await mgr.mark_confirmed(oid, "exchange-order-id-999")
        assert await mgr.get_state(oid) == OrderStatus.REST_ACCEPTED

    async def test_mark_filled_from_ws_confirmed(self) -> None:
        mgr, oid = await self._setup()
        await mgr.mark_submitted(oid)
        await mgr.mark_confirmed(oid, "ex-001")
        # Manually jump to WS_CONFIRMED
        mgr._store[oid]["status"] = OrderStatus.WS_CONFIRMED
        await mgr.mark_filled(oid)
        assert await mgr.get_state(oid) == OrderStatus.FILLED

    async def test_mark_cancelled_from_ws_confirmed(self) -> None:
        mgr, oid = await self._setup()
        await mgr.mark_submitted(oid)
        await mgr.mark_confirmed(oid, "ex-002")
        mgr._store[oid]["status"] = OrderStatus.WS_CONFIRMED
        await mgr.mark_cancelled(oid)
        assert await mgr.get_state(oid) == OrderStatus.CANCELLED

    async def test_invalid_transition_raises(self) -> None:
        mgr, oid = await self._setup()
        # Cannot jump from CREATED_LOCAL to FILLED directly
        with pytest.raises(OrderRejectedError):
            await mgr.mark_filled(oid)

    async def test_get_state_unknown_id_returns_none(self) -> None:
        mgr = IdempotencyManager()
        result = await mgr.get_state("unknown-id")
        assert result is None

    async def test_cannot_resubmit_already_confirmed_intent(self) -> None:
        """Once confirmed, re-registering the same ID should raise."""
        mgr, oid = await self._setup()
        await mgr.mark_submitted(oid)
        await mgr.mark_confirmed(oid, "ex-003")
        # Try to register a new intent with the same ID
        intent2 = _make_intent(oid)
        with pytest.raises(OrderRejectedError):
            await mgr.register_intent(intent2)

    async def test_all_states_returns_dict(self) -> None:
        mgr = IdempotencyManager()
        intent = _make_intent("TN-260605-MOMO-PROP1234-abc123")
        await mgr.register_intent(intent)
        states = mgr.all_states()
        assert "TN-260605-MOMO-PROP1234-abc123" in states
        assert states["TN-260605-MOMO-PROP1234-abc123"] == OrderStatus.CREATED_LOCAL.value

    async def test_pending_count_reflects_non_terminal_states(self) -> None:
        mgr = IdempotencyManager()
        intent1 = _make_intent("TN-260605-MOMO-PROP1234-abc123")
        intent2 = _make_intent("TN-260605-MOMO-PROP5678-def456")
        await mgr.register_intent(intent1)
        await mgr.register_intent(intent2)
        assert mgr.pending_count() == 2

        # Fill one
        await mgr.mark_submitted("TN-260605-MOMO-PROP1234-abc123")
        await mgr.mark_confirmed("TN-260605-MOMO-PROP1234-abc123", "ex-111")
        mgr._store["TN-260605-MOMO-PROP1234-abc123"]["status"] = OrderStatus.WS_CONFIRMED
        await mgr.mark_filled("TN-260605-MOMO-PROP1234-abc123")
        assert mgr.pending_count() == 1

    async def test_terminal_orders_are_pruned_without_dropping_pending(self) -> None:
        mgr = IdempotencyManager(max_terminal_retained=1)
        first = _make_intent("TN-260605-MOMO-PROP0001-abc123")
        second = _make_intent("TN-260605-MOMO-PROP0002-def456")
        pending = _make_intent("TN-260605-MOMO-PROP0003-fedcba")
        for intent in (first, second, pending):
            await mgr.register_intent(intent)
            await mgr.mark_submitted(intent.order_link_id)
            await mgr.mark_confirmed(intent.order_link_id, f"ex-{intent.order_link_id[-6:]}")

        mgr._store[first.order_link_id]["status"] = OrderStatus.WS_CONFIRMED
        await mgr.mark_filled(first.order_link_id)
        mgr._store[second.order_link_id]["status"] = OrderStatus.WS_CONFIRMED
        await mgr.mark_filled(second.order_link_id)

        states = mgr.all_states()
        assert first.order_link_id not in states
        assert states[second.order_link_id] == OrderStatus.FILLED.value
        assert states[pending.order_link_id] == OrderStatus.REST_ACCEPTED.value
        assert mgr.pending_count() == 1
