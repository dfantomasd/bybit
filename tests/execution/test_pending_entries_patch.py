from datetime import UTC, datetime
from types import SimpleNamespace

from trader.execution.pending_entries_patch import (
    _mark_entry_resolved,
    _mark_entry_submitted,
    _restore_pending_entries_with_symbols,
)


def make_engine(*, shadow: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        _shadow_mode=shadow,
        _pending_entry_order_link_ids=set(),
        _pending_entry_symbols={},
        _pending_entry_created_at={},
        _pending_entry_count=0,
        _recent_entries=[],
    )


def test_submit_and_resolve_exact_id_is_idempotent() -> None:
    engine = make_engine()

    _mark_entry_submitted(engine, "order-1", "BTCUSDT")
    _mark_entry_submitted(engine, "order-1", "BTCUSDT")

    assert engine._pending_entry_order_link_ids == {"order-1"}
    assert engine._pending_entry_count == 1
    assert len(engine._recent_entries) == 1

    _mark_entry_resolved(engine, "order-1")
    _mark_entry_resolved(engine, "order-1")

    assert engine._pending_entry_order_link_ids == set()
    assert engine._pending_entry_count == 0


def test_legacy_resolve_without_id_clears_only_single_pending_entry() -> None:
    engine = make_engine()
    _mark_entry_submitted(engine, "order-1", "ETHUSDT")

    _mark_entry_resolved(engine)

    assert engine._pending_entry_order_link_ids == set()
    assert engine._pending_entry_symbols == {}
    assert engine._pending_entry_count == 0


def test_legacy_resolve_without_id_does_not_guess_when_multiple_entries_exist() -> None:
    engine = make_engine()
    _mark_entry_submitted(engine, "order-1", "BTCUSDT")
    _mark_entry_submitted(engine, "order-2", "ETHUSDT")

    _mark_entry_resolved(engine)

    assert engine._pending_entry_order_link_ids == {"order-1", "order-2"}
    assert engine._pending_entry_count == 2


def test_restore_pending_records_resynchronises_legacy_counter() -> None:
    engine = make_engine()
    created_at = datetime.now(tz=UTC)

    _restore_pending_entries_with_symbols(
        engine,
        [
            {"order_link_id": "order-1", "symbol": "BTCUSDT", "created_at": created_at},
            {"order_link_id": "order-2", "symbol": "ETHUSDT", "created_at": created_at},
        ],
    )

    assert engine._pending_entry_order_link_ids == {"order-1", "order-2"}
    assert engine._pending_entry_symbols == {"order-1": "BTCUSDT", "order-2": "ETHUSDT"}
    assert engine._pending_entry_count == 2


def test_shadow_pending_counter_stays_zero() -> None:
    engine = make_engine(shadow=True)

    _mark_entry_submitted(engine, "shadow-1", "BTCUSDT")

    assert engine._pending_entry_order_link_ids == {"shadow-1"}
    assert engine._pending_entry_count == 0
