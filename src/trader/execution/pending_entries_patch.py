"""Compatibility patch for pending-entry bookkeeping.

The runtime currently has a legacy terminal-order callback that may call
``mark_entry_resolved()`` without an order link ID.  This patch keeps pending
state fail-safe while avoiding permanent global entry starvation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog

from trader.execution import engine as _engine

log = structlog.get_logger(__name__)
_PATCH_INSTALLED = False


def _sync_pending_count(self: Any) -> None:
    """Keep the legacy integer counter aligned with the authoritative ID set."""

    self._pending_entry_count = len(self._pending_entry_order_link_ids) if not self._shadow_mode else 0


def _mark_entry_submitted(self: Any, order_link_id: str = "", symbol: str = "") -> None:
    """Register a pending entry once and update rate-limit counters idempotently."""

    if not order_link_id:
        log.warning("execution.pending_submit_missing_id", symbol=symbol)
        return

    is_new = order_link_id not in self._pending_entry_order_link_ids
    self._pending_entry_order_link_ids.add(order_link_id)
    if symbol:
        self._pending_entry_symbols[order_link_id] = symbol
    self._pending_entry_created_at.setdefault(order_link_id, datetime.now(tz=UTC))

    if is_new and not self._shadow_mode:
        self._recent_entries.append(datetime.now(tz=UTC))
    _sync_pending_count(self)


def _mark_entry_resolved(self: Any, order_link_id: str = "") -> None:
    """Resolve exactly one pending entry without corrupting pending counters.

    When a legacy caller omits the ID, resolve the only pending entry if there
    is exactly one.  With multiple candidates, keep all IDs and log a warning:
    guessing would be unsafe.
    """

    resolved_id = order_link_id
    if not resolved_id:
        pending_ids = sorted(self._pending_entry_order_link_ids)
        if len(pending_ids) == 1:
            resolved_id = pending_ids[0]
            log.warning("execution.pending_resolve_inferred_single_id", order_link_id=resolved_id)
        elif not pending_ids:
            _sync_pending_count(self)
            return
        else:
            log.warning(
                "execution.pending_resolve_ambiguous_missing_id",
                pending_ids=pending_ids,
            )
            _sync_pending_count(self)
            return

    existed = resolved_id in self._pending_entry_order_link_ids
    self._pending_entry_order_link_ids.discard(resolved_id)
    self._pending_entry_symbols.pop(resolved_id, None)
    self._pending_entry_created_at.pop(resolved_id, None)
    _sync_pending_count(self)

    if existed:
        log.info("execution.pending_entry_resolved", order_link_id=resolved_id)


def _restore_pending_entries(self: Any, order_link_ids: list[str]) -> None:
    """Restore pending IDs and keep the legacy counter aligned."""

    for order_link_id in order_link_ids:
        if order_link_id:
            self._pending_entry_order_link_ids.add(order_link_id)
            self._pending_entry_created_at.setdefault(order_link_id, datetime.now(tz=UTC))
    _sync_pending_count(self)
    if order_link_ids:
        log.info(
            "execution.pending_entries_restored",
            count=len(self._pending_entry_order_link_ids),
            ids=sorted(self._pending_entry_order_link_ids),
        )


def _restore_pending_entries_with_symbols(self: Any, records: list[dict[str, Any]]) -> None:
    """Restore detailed pending records and synchronise all indexes."""

    for record in records:
        order_link_id = str(record.get("order_link_id", ""))
        symbol = str(record.get("symbol", ""))
        if not order_link_id:
            continue
        self._pending_entry_order_link_ids.add(order_link_id)
        if symbol:
            self._pending_entry_symbols[order_link_id] = symbol
        created_at = record.get("created_at") or record.get("updated_at")
        self._pending_entry_created_at[order_link_id] = (
            created_at if isinstance(created_at, datetime) else datetime.now(tz=UTC)
        )
    _sync_pending_count(self)
    if records:
        log.info(
            "execution.pending_entries_restored_with_symbols",
            count=len(self._pending_entry_order_link_ids),
            ids=sorted(self._pending_entry_order_link_ids),
        )


def install_pending_entries_patch() -> None:
    """Install corrected ExecutionEngine methods once per Python process."""

    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return

    engine_cls = _engine.ExecutionEngine
    engine_cls.mark_entry_submitted = _mark_entry_submitted
    engine_cls.mark_entry_resolved = _mark_entry_resolved
    engine_cls.restore_pending_entries = _restore_pending_entries
    engine_cls.restore_pending_entries_with_symbols = _restore_pending_entries_with_symbols
    _PATCH_INSTALLED = True


__all__ = ("install_pending_entries_patch",)
