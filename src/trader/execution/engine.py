"""Execution Engine — final step in the trade lifecycle.

Flow for each TradeProposal:
  1. Deduplicate: skip if position already open for this symbol
  2. Cooldown: skip if too soon after last signal on same symbol
  3. Fetch InstrumentInfo (cached)
  4. RiskManager.evaluate() — may REJECT, RESIZE, or APPROVE
  5. Build OrderIntent from approved decision
  6. SHADOW mode → log only; LIVE mode → submit via BybitAdapter
  7. Update ExposureTracker and local position registry

SAFETY: In SHADOW mode no order ever reaches the exchange.
Live execution requires LIVE_MODE=true AND TRADING_MODE=LIVE.
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal
from typing import Any, cast

import structlog

from trader.domain.enums import OrderSide, OrderType, RiskDecisionStatus
from trader.domain.models import (
    FeatureVector,
    InstrumentInfo,
    OrderIntent,
    RegimeContext,
    RiskDecision,
    TradeProposal,
)
from trader.risk.safety_ladder import SafetyLevel, SafetyModeLadder

log = structlog.get_logger(__name__)

# A position absent from an exchange snapshot is removed from the registry
# only when its entry is older than this. Covers the window between placing
# an order and the exchange reflecting the position (partial fills, REST lag).
# Keep small: an over-long grace would mask genuinely closed positions and
# could double-enter after a real TP/SL closure.
_SYNC_REMOVAL_GRACE_SECONDS = 3.0

# Default cooldown between successful entries on the same symbol
_DEFAULT_COOLDOWN_S = 300  # 5 minutes
# Separate shorter cooldown after an API-level order failure
_DEFAULT_FAILURE_COOLDOWN_S = 60  # 1 minute
# Instrument info TTL: re-fetch after this many seconds (tick_size can change)
_INSTRUMENT_CACHE_TTL_S = 3600  # 1 hour
# Pending entries older than this (seconds) with no exchange order are considered stale
_STALE_PENDING_THRESHOLD_S = 600  # 10 minutes

# P0.6: Canary mode hard caps — cannot be overridden by any profile
CANARY_MAX_OPEN_POSITIONS: int = 2
CANARY_MAX_TOTAL_EXPOSURE_PCT: Decimal = Decimal("45")

# Maker-first execution
_MAKER_POLL_INTERVAL_S = 0.5
# Escalation guard: abort instead of taker when price moved further than this
# from the maker limit price (percent of price)
_MAKER_MAX_ESCALATION_DRIFT_PCT = 0.1
# Queue-aware escalation: fraction of maker_ttl_s elapsed before we assume the
# queue in front of our order is deep enough to warrant taker escalation even
# when imbalance is adverse. Also enforces a minimum absolute wait to avoid
# false overrides in test/canary environments with short timeouts.
_QUEUE_DEPTH_ESCALATION_PCT: float = 0.75
_QUEUE_DEPTH_MIN_WAIT_S: float = 2.0  # only relevant in production (>= 2 s wait)


class ExecutionEngine:
    """Orchestrates risk checks and order submission for trade proposals.

    Args:
        adapter:          BybitAdapter instance (used for order placement and
                          instrument info fetching).
        risk_manager:     Fully initialised RiskManager.
        exposure_tracker: ExposureTracker shared with the RiskManager.
        shadow_mode:      When True, orders are logged but never submitted.
        cooldown_s:       Minimum seconds between entries on the same symbol.
        category:         Bybit market category (e.g. "linear").
    """

    def __init__(
        self,
        adapter: Any,
        risk_manager: Any,
        exposure_tracker: Any,
        shadow_mode: bool = True,
        cooldown_s: int = _DEFAULT_COOLDOWN_S,
        failure_cooldown_s: int = _DEFAULT_FAILURE_COOLDOWN_S,
        category: str = "linear",
        trade_journal: Any | None = None,
        min_notional_safety_buffer_pct: float = 3.0,
        max_new_entries_per_minute: int = 60,
        max_concurrent_pending_entries: int = 10,
        max_same_side_positions: int = 10,
        startup_warmup_seconds: int = 0,
        is_canary: bool = False,
        fee_provider: Any | None = None,
        max_spread_bps: float = 8.0,
        expected_slippage_pct: float = 0.03,
        funding_buffer_pct: float = 0.01,
        min_net_edge_pct: float = 0.15,
        net_edge_safety_margin_pct: float = 0.05,
        entry_order_mode: str = "MARKET",
        maker_timeout_s: float = 3.0,
        maker_ttl_s: float = 5.0,
        maker_allow_escalation: bool = True,
        imbalance_provider: Any | None = None,
        safety_ladder: SafetyModeLadder | None = None,
        max_open_positions: int = 10,
        trailing_stop_atr_multiple: float = 1.5,
        trailing_stop_min_pct: float = 0.01,
        profit_gate_pct: float = 3.0,
        profit_lock_pct: float = 5.0,
    ) -> None:
        self._adapter = adapter
        self._risk_manager = risk_manager
        self._exposure = exposure_tracker
        self._shadow_mode = shadow_mode
        self._cooldown = timedelta(seconds=cooldown_s)
        self._failure_cooldown = timedelta(seconds=failure_cooldown_s)
        self._category = category
        self._trade_journal = trade_journal
        self._min_notional_buffer = Decimal(str(min_notional_safety_buffer_pct))
        self._is_canary = is_canary
        self._fee_provider = fee_provider
        self._max_spread_bps = max_spread_bps
        self._expected_slippage_pct = expected_slippage_pct
        self._funding_buffer_pct = funding_buffer_pct
        self._min_net_edge_pct = min_net_edge_pct
        self._net_edge_safety_margin_pct = net_edge_safety_margin_pct
        self._entry_order_mode = str(entry_order_mode).strip().upper()
        self._maker_timeout_s = max(0.5, float(maker_timeout_s))
        self._maker_ttl_s = max(self._maker_timeout_s, float(maker_ttl_s))
        self._maker_allow_escalation = maker_allow_escalation
        # Callable(symbol) -> L5 orderbook imbalance in [-1, 1] or None; used by
        # the maker escalation guard. Missing data fails open (escalation allowed).
        self._imbalance_provider = imbalance_provider
        self._safety_ladder: SafetyModeLadder | None = safety_ladder
        self._max_open_positions: int = max(1, int(max_open_positions))
        self._trailing_stop_atr_multiple = max(0.5, float(trailing_stop_atr_multiple))
        self._trailing_stop_min_pct = max(0.001, float(trailing_stop_min_pct))
        self._profit_gate_pct = max(0.0, float(profit_gate_pct))
        self._profit_lock_pct = max(self._profit_gate_pct, float(profit_lock_pct))

        # P0: Hard block unsupported entry modes. MARKET is the default;
        # MAKER_FIRST is the supervised maker-with-escalation flow below.
        if self._entry_order_mode not in ("MARKET", "MAKER_FIRST"):
            raise ValueError("Only MARKET and MAKER_FIRST entry modes are supported")

        # Burst / rate limiting
        self._max_entries_per_minute = max_new_entries_per_minute
        self._max_concurrent_pending = max_concurrent_pending_entries
        self._max_same_side = max_same_side_positions
        self._startup_warmup = timedelta(seconds=startup_warmup_seconds)
        self._started_at: datetime = datetime.now(tz=UTC)
        # Rolling window of entry timestamps for per-minute rate limiting
        self._recent_entries: list[datetime] = []
        # P0.3: Pending entry limiter keyed by order_link_id (not integer).
        self._pending_entry_order_link_ids: set[str] = set()
        # order_link_id → symbol mapping for screener has_pending_order support
        self._pending_entry_symbols: dict[str, str] = {}
        self._pending_entry_created_at: dict[str, datetime] = {}
        # Legacy count — kept for compatibility with rate-limit check
        self._pending_entry_count: int = 0
        # Per-session diagnostic counters (cumulative, not windowed)
        self._diag_skip_pending: int = 0
        self._diag_order_placed: int = 0
        self._diag_shadow_order_would_be_placed: int = 0
        self._diag_order_failed: int = 0
        self._diag_net_edge_rejected: int = 0
        self._diag_no_tp_rejected: int = 0
        self._diag_fee_unavailable_rejected: int = 0
        self._diag_maker_filled: int = 0
        self._diag_maker_escalated: int = 0
        self._diag_maker_aborted: int = 0

        # symbol → last *successful* entry timestamp
        self._last_entry_at: dict[str, datetime] = {}
        # symbol → last API-failure timestamp (separate from entry cooldown)
        self._last_failure_at: dict[str, datetime] = {}
        # symbol → open position metadata (size, entry_price, side)
        self._open_positions: dict[str, dict[str, Any]] = {}
        # symbol → (InstrumentInfo, cached_at) — TTL enforced
        self._instrument_cache: dict[str, tuple[InstrumentInfo, datetime]] = {}
        # symbol → leverage already confirmed on exchange for this session
        self._leverage_confirmed: dict[str, Decimal] = {}
        # Serialises risk evaluation + local exposure updates across symbols.
        self._submit_lock = asyncio.Lock()
        # Keep strong references to trailing-stop tasks to prevent GC before completion.
        self._trailing_stop_tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Startup warmup / burst guards
    # ------------------------------------------------------------------

    def is_in_warmup(self) -> bool:
        """True if still in the post-startup monitoring-only phase."""
        return datetime.now(tz=UTC) - self._started_at < self._startup_warmup

    def warmup_seconds_remaining(self) -> float:
        elapsed = (datetime.now(tz=UTC) - self._started_at).total_seconds()
        return max(0.0, self._startup_warmup.total_seconds() - elapsed)

    def _prune_recent_entries(self) -> None:
        cutoff = datetime.now(tz=UTC) - timedelta(seconds=60)
        self._recent_entries = [t for t in self._recent_entries if t > cutoff]

    def _check_rate_limits(self, symbol: str, side: str) -> str | None:
        """Return a rejection reason string if burst limits are exceeded, else None.

        Rate limits are enforced only in live mode (non-shadow). In shadow mode
        we simulate freely so tests and monitoring remain unaffected.
        """
        if self.is_in_warmup():
            return f"startup_warmup_active ({self.warmup_seconds_remaining():.0f}s remaining)"

        # Rate / burst limits only apply to live execution
        if not self._shadow_mode:
            self._prune_recent_entries()
            if len(self._recent_entries) >= self._max_entries_per_minute:
                return f"rate_limit: {len(self._recent_entries)}/{self._max_entries_per_minute} entries this minute"

            if self._pending_entry_count >= self._max_concurrent_pending:
                return f"pending_limit: {self._pending_entry_count}/{self._max_concurrent_pending} concurrent pending"

            same_side_count = sum(
                1 for p in self._open_positions.values() if str(p.get("side", "")).upper() == side.upper()
            )
            if same_side_count >= self._max_same_side:
                return f"same_side_limit: {same_side_count}/{self._max_same_side} {side} positions"

        return None

    def mark_entry_submitted(self, order_link_id: str = "", symbol: str = "") -> None:
        """Register order_link_id as pending and bump rate-limit counters.

        Idempotent: duplicate IDs are silently ignored without double-counting.
        Empty ID in live mode is rejected with a warning.
        """
        if order_link_id:
            already_pending = order_link_id in self._pending_entry_order_link_ids
            if already_pending:
                log.warning(
                    "execution.submit_duplicate_id",
                    order_link_id=order_link_id,
                    pending_count=len(self._pending_entry_order_link_ids),
                )
                # Do not increment — idempotent, count stays the same
                return
            self._pending_entry_order_link_ids.add(order_link_id)
            if symbol:
                self._pending_entry_symbols[order_link_id] = symbol
            self._pending_entry_created_at[order_link_id] = datetime.now(tz=UTC)
        elif not self._shadow_mode:
            log.warning(
                "execution.submit_empty_id_live_mode",
                pending_count=len(self._pending_entry_order_link_ids),
            )
            # Refuse to track an un-identified live order — would corrupt count
            return
        if not self._shadow_mode:
            self._recent_entries.append(datetime.now(tz=UTC))
        # Always sync count from the authoritative set
        self._pending_entry_count = len(self._pending_entry_order_link_ids)

    def mark_entry_resolved(self, order_link_id: str = "") -> None:
        """Remove order_link_id from pending set (idempotent, fail-closed).

        Rules:
        - Known ID → remove and sync count. Safe to call multiple times.
        - Unknown ID → warn, do NOT change count (fail-closed).
        - Empty ID, 0 pending → no-op.
        - Empty ID, 1 pending → safe backwards-compat fallback, warns.
        - Empty ID, ≥2 pending → ambiguous, log and stay fail-closed.
        """
        if order_link_id:
            was_pending = order_link_id in self._pending_entry_order_link_ids
            if was_pending:
                self._pending_entry_order_link_ids.discard(order_link_id)
                self._pending_entry_symbols.pop(order_link_id, None)
                self._pending_entry_created_at.pop(order_link_id, None)
            else:
                log.warning(
                    "execution.resolve_unknown_id",
                    order_link_id=order_link_id,
                    pending_ids=sorted(self._pending_entry_order_link_ids),
                )
                # Do not change count — the ID was never registered
        else:
            n = len(self._pending_entry_order_link_ids)
            if n == 0:
                log.debug("execution.resolve_empty_id_no_pending")
            elif n == 1:
                # Backwards-compat fallback: release the single known pending slot
                lone_id = next(iter(self._pending_entry_order_link_ids))
                log.warning(
                    "execution.resolve_empty_id_fallback",
                    lone_id=lone_id,
                )
                self._pending_entry_order_link_ids.discard(lone_id)
                self._pending_entry_symbols.pop(lone_id, None)
                self._pending_entry_created_at.pop(lone_id, None)
            else:
                # Multiple pending — ambiguous which slot to release; stay fail-closed
                log.warning(
                    "execution.resolve_empty_id_ambiguous",
                    pending_count=n,
                    pending_ids=sorted(self._pending_entry_order_link_ids),
                )
        # Always sync count from the authoritative set
        self._pending_entry_count = len(self._pending_entry_order_link_ids)

    async def resolve_pending_durable(self, order_link_id: str, symbol: str = "") -> None:
        """Release a pending slot and persist the resolution to order_pending_state."""
        self.mark_entry_resolved(order_link_id)
        if self._trade_journal is not None and self._trade_journal.is_enabled:
            try:
                await self._trade_journal.mark_order_resolved(order_link_id, symbol)
            except Exception as _res_exc:
                log.debug(
                    "execution.mark_order_resolved_failed",
                    order_link_id=order_link_id,
                    error=str(_res_exc),
                )

    def _release_exposure_reservation(self, proposal: TradeProposal) -> None:
        """Release the RiskManager's pre-submit exposure reservation."""
        release = getattr(self._exposure, "release_reservation", None)
        if callable(release):
            release(str(proposal.proposal_id))

    def _reserve_adjusted_exposure(
        self,
        proposal: TradeProposal,
        qty: Decimal,
        instrument_info: InstrumentInfo,
    ) -> tuple[bool, str]:
        """Replace RiskManager's reservation after post-risk execution sizing."""

        if proposal.entry_price is None or proposal.entry_price <= Decimal("0"):
            return True, ""
        notional = qty * proposal.entry_price
        # RiskManager reserved exposure for the pre-signal qty. From this point
        # any outcome must release it: successful re-reservation uses the final
        # qty, and rejection must not leave phantom pending exposure behind.
        self._release_exposure_reservation(proposal)
        if instrument_info.min_notional is not None and instrument_info.min_notional > Decimal("0"):
            required_notional = instrument_info.min_notional * (
                Decimal("1") + self._min_notional_buffer / Decimal("100")
            )
            if notional < required_notional:
                return (
                    False,
                    f"post-signal notional {notional:.4f} < required {required_notional:.4f}",
                )

        # Replace the original reservation with the final qty so boosts cannot
        # bypass caps and reductions do not over-reserve capital.
        can_add = getattr(type(self._exposure), "can_add_position", None)
        if can_add is None:
            return True, ""
        return self._exposure.can_add_position(
            proposal.symbol,
            notional,
            order_id=str(proposal.proposal_id),
        )

    def has_pending_order_for_symbol(self, symbol: str) -> bool:
        """Return True if there is a pending (unresolved) entry for this symbol."""
        return symbol in self._pending_entry_symbols.values()

    def restore_pending_entries(self, order_link_ids: list[str]) -> None:
        """Restore pending entry IDs from durable storage at startup.

        Empty and duplicate IDs are silently discarded. Count is synced
        from the set after restoration.
        IDs starting with 'unknown:' are excluded to prevent phantom recovery.
        """
        valid_ids = [oid for oid in order_link_ids if oid and not str(oid).startswith("unknown:")]
        unique_ids = sorted(set(valid_ids))
        for oid in unique_ids:
            self._pending_entry_order_link_ids.add(oid)
        self._pending_entry_count = len(self._pending_entry_order_link_ids)
        if unique_ids:
            log.info(
                "execution.pending_entries_restored",
                count=len(unique_ids),
                ids=unique_ids,
                total_pending=self._pending_entry_count,
            )

    def restore_pending_entries_with_symbols(self, records: list[dict[str, Any]]) -> None:
        """Restore pending entries from detailed records (includes symbol mapping).

        Empty and duplicate IDs are silently discarded. Count is synced
        from the set after restoration.
        IDs starting with 'unknown:' are excluded to prevent phantom recovery.
        """
        seen: set[str] = set()
        for rec in records:
            oid = str(rec.get("order_link_id", "")).strip()
            if not oid or oid in seen or oid.startswith("unknown:"):
                continue
            seen.add(oid)
            symbol = str(rec.get("symbol", ""))
            self._pending_entry_order_link_ids.add(oid)
            if symbol:
                self._pending_entry_symbols[oid] = symbol
            created_at = rec.get("created_at")
            if isinstance(created_at, datetime):
                self._pending_entry_created_at[oid] = created_at
        self._pending_entry_count = len(self._pending_entry_order_link_ids)

    def has_pending_entries(self) -> bool:
        return bool(self._pending_entry_order_link_ids)

    async def reconcile_restored_pending_entries(self) -> None:
        """Check restored pending entries against exchange state; clear stale ones.

        Fail-safe: if Bybit API is unavailable, all pending entries are preserved.
        Age threshold: entries younger than _STALE_PENDING_THRESHOLD_S are always kept.
        Uses durable_order_state as authoritative source; falls back to order_events.
        """
        if self._trade_journal is None or not self._pending_entry_order_link_ids:
            return

        log.info(
            "execution.pending_reconcile_started",
            pending_count=len(self._pending_entry_order_link_ids),
            pending_ids=sorted(self._pending_entry_order_link_ids),
        )

        # Build merged record dict — durable_order_state takes priority over order_events
        merged: dict[str, dict[str, Any]] = {}
        try:
            for rec in await self._trade_journal.get_pending_order_events():
                oid = str(rec.get("order_link_id", ""))
                if oid:
                    merged[oid] = dict(rec)
        except Exception as exc:
            log.warning("execution.pending_reconcile_failed", reason="order_events_read_error", error=str(exc))

        try:
            for rec in await self._trade_journal.get_pending_durable_orders():
                oid = str(rec.get("order_link_id", ""))
                if oid:
                    merged[oid] = dict(rec)  # durable overrides
        except Exception as exc:
            log.warning("execution.pending_reconcile_failed", reason="durable_read_error", error=str(exc))
            return  # fail-safe: keep all if we can't read durable state

        # Fail-safe: keep all pending if exchange API is unavailable
        try:
            open_orders = await self._adapter.get_open_orders(self._category)
            exchange_link_ids = {str(o.get("orderLinkId")) for o in open_orders if o.get("orderLinkId")}
        except Exception as exc:
            log.warning(
                "execution.pending_reconcile_failed",
                reason="exchange_api_unavailable",
                error=str(exc),
            )
            return

        now = datetime.now(tz=UTC)
        threshold_s = _STALE_PENDING_THRESHOLD_S

        for order_link_id in list(self._pending_entry_order_link_ids):
            record = merged.get(order_link_id, {})
            symbol = str(record.get("symbol", ""))
            created_at = record.get("created_at")
            age_s = (now - created_at).total_seconds() if isinstance(created_at, datetime) else 0.0

            if order_link_id in exchange_link_ids:
                log.info(
                    "execution.pending_kept_exchange_order",
                    order_link_id=order_link_id,
                    symbol=symbol,
                    age_s=int(age_s),
                )
                continue

            if symbol and self.has_open_position(symbol):
                log.info(
                    "execution.pending_kept_position_exists",
                    order_link_id=order_link_id,
                    symbol=symbol,
                    age_s=int(age_s),
                )
                continue

            if age_s < threshold_s:
                log.info(
                    "execution.pending_kept_recent",
                    order_link_id=order_link_id,
                    symbol=symbol,
                    age_s=int(age_s),
                    threshold_s=threshold_s,
                )
                continue

            # Old entry, no exchange order, no position → mark stale (non-destructive)
            stale_reason = f"no_exchange_order_no_position_age_{int(age_s)}s"
            try:
                await self._trade_journal.mark_order_event_stale(order_link_id, stale_reason)
                await self._trade_journal.mark_durable_order_stale(order_link_id, stale_reason)
            except Exception as exc:
                log.warning(
                    "execution.pending_reconcile_failed",
                    reason="db_mark_stale_error",
                    order_link_id=order_link_id,
                    error=str(exc),
                )
                continue

            self._pending_entry_order_link_ids.discard(order_link_id)
            self._pending_entry_symbols.pop(order_link_id, None)
            self._pending_entry_created_at.pop(order_link_id, None)
            log.info(
                "execution.pending_cleared_stale",
                order_link_id=order_link_id,
                symbol=symbol,
                age_s=int(age_s),
            )

        # Sync count with actual set size after cleanup
        self._pending_entry_count = len(self._pending_entry_order_link_ids)

        log.info(
            "execution.pending_reconcile_complete",
            remaining=len(self._pending_entry_order_link_ids),
            remaining_ids=sorted(self._pending_entry_order_link_ids),
        )

    # ------------------------------------------------------------------
    # Position awareness
    # ------------------------------------------------------------------

    def has_open_position(self, symbol: str) -> bool:
        return symbol in self._open_positions

    def open_position_count(self) -> int:
        return len(self._open_positions)

    async def sync_positions(self) -> list[Any] | None:
        """Reconcile the local position registry with the exchange.

        Runs at startup, periodically, and on private-WS fills. The registry
        mutation is serialised with ``submit`` under ``_submit_lock`` so a
        concurrent snapshot cannot wipe an optimistically registered entry;
        the REST fetch itself stays outside the lock to avoid blocking
        trading on network latency.
        """
        positions = await self._fetch_positions()
        if positions is None:
            return None
        async with self._submit_lock:
            await self._apply_position_snapshot(positions)
        return positions

    async def _sync_positions_locked(self) -> list[Any] | None:
        """``sync_positions`` for callers already holding ``_submit_lock``."""
        positions = await self._fetch_positions()
        if positions is None:
            return None
        await self._apply_position_snapshot(positions)
        return positions

    async def _fetch_positions(self) -> list[Any] | None:
        try:
            return cast(list[Any], await self._adapter.get_positions(self._category))
        except Exception as exc:
            log.warning("execution.sync_positions_failed", error=str(exc))
            return None

    async def _apply_position_snapshot(self, positions: list[Any]) -> None:
        """Merge an exchange snapshot into the registry (never wholesale-replace).

        A symbol disappears from the registry only when the exchange snapshot
        does not list it AND its entry is older than the grace period — a
        snapshot fetched milliseconds before a fill lands must not erase the
        freshly opened position (that would release its exposure and allow a
        duplicate entry on the next signal).
        """
        now = datetime.now(tz=UTC)
        exchange_symbols: set[str] = set()
        for pos in positions:
            if pos.size > Decimal("0"):
                exchange_symbols.add(pos.symbol)
                self._open_positions[pos.symbol] = {
                    "side": pos.side,
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                }
                notional = pos.size * pos.entry_price
                await self._exposure.update_position(pos.symbol, pos.side.value, notional)
        closed_symbols: list[str] = []
        kept_recent: list[str] = []
        for symbol in list(self._open_positions):
            if symbol in exchange_symbols:
                continue
            entered_at = self._last_entry_at.get(symbol)
            if entered_at is not None and (now - entered_at).total_seconds() < _SYNC_REMOVAL_GRACE_SECONDS:
                kept_recent.append(symbol)
                continue
            self._open_positions.pop(symbol, None)
            self._last_entry_at.pop(symbol, None)
            await self._exposure.remove_position(symbol)
            closed_symbols.append(symbol)
        log.info(
            "execution.positions_synced",
            count=len(self._open_positions),
            symbols=list(self._open_positions.keys()),
            closed_symbols=sorted(closed_symbols),
            kept_recent=sorted(kept_recent),
        )

    async def record_position_closed(self, symbol: str) -> None:
        """Call when a position is closed (e.g. TP/SL hit)."""
        self._open_positions.pop(symbol, None)
        self._last_entry_at.pop(symbol, None)
        await self._exposure.remove_position(symbol)

    # ------------------------------------------------------------------
    # Instrument info
    # ------------------------------------------------------------------

    def update_ticker_turnover(self, symbol: str, turnover_24h: Decimal) -> None:
        """Patch cached InstrumentInfo with fresh 24h turnover from a WS ticker event."""
        cached = self._instrument_cache.get(symbol)
        if cached is None:
            return
        info, cached_at = cached
        if info.turnover_24h == turnover_24h:
            return
        self._instrument_cache[symbol] = (info.model_copy(update={"turnover_24h": turnover_24h}), cached_at)

    async def get_instrument_info(self, symbol: str) -> InstrumentInfo:
        cached = self._instrument_cache.get(symbol)
        if cached is not None:
            info, cached_at = cached
            age = (datetime.now(tz=UTC) - cached_at).total_seconds()
            if age < _INSTRUMENT_CACHE_TTL_S:
                return info
        raw: InstrumentInfo = await self._adapter.get_instrument_info(self._category, symbol)
        self._instrument_cache[symbol] = (raw, datetime.now(tz=UTC))
        log.debug(
            "execution.instrument_info_cached",
            symbol=symbol,
            min_qty=str(raw.min_order_qty),
            qty_step=str(raw.qty_step),
        )
        return raw

    async def _ensure_leverage(self, symbol: str, max_leverage: Decimal) -> None:
        """Set exchange leverage to match profile max, if not already confirmed."""
        confirmed = self._leverage_confirmed.get(symbol)
        if confirmed is not None and confirmed <= max_leverage:
            return
        try:
            lev_str = str(int(max_leverage)) if max_leverage == max_leverage.to_integral_value() else str(max_leverage)
            leverage_result = self._adapter._rest.set_leverage(
                category=self._category,
                symbol=symbol,
                buy_leverage=lev_str,
                sell_leverage=lev_str,
            )
            if inspect.isawaitable(leverage_result):
                await leverage_result
            self._leverage_confirmed[symbol] = max_leverage
            log.info("execution.leverage_set", symbol=symbol, leverage=lev_str)
        except Exception as exc:
            log.warning("execution.leverage_set_failed", symbol=symbol, error=str(exc))
            raise

    # ------------------------------------------------------------------
    # Signal quality guards (STDEV, pDiv, SOP)
    # ------------------------------------------------------------------

    def _stdev_trend_guard(
        self,
        proposal: TradeProposal,
        feature_vector: FeatureVector | None,
    ) -> str | None:
        """Block or skip if volatility is extreme and entry is contra-trend.

        Returns a non-empty rejection reason string when the guard fires,
        or None when the proposal is allowed through.
        """
        if feature_vector is None:
            return None
        try:
            vol_idx = feature_vector.feature_names.index("realized_vol_20")
            dist_idx = feature_vector.feature_names.index("sma20_dist")
        except ValueError:
            return None

        realized_vol = float(feature_vector.values[vol_idx])
        sma20_dist = float(feature_vector.values[dist_idx])
        if realized_vol <= 0:
            return None

        # Target vol proxy: profile's max_drawdown_pct converted to daily fraction
        target_vol = float(self._risk_manager._limits.max_drawdown_pct) / 100.0
        if realized_vol <= 3.0 * target_vol:
            return None  # Normal vol regime — pass

        # Extreme vol: only allow trend-confirming entries
        # sma20_dist > 0 → price above SMA (bullish); < 0 → below (bearish)
        if proposal.side.value == "Buy" and sma20_dist < 0:
            return "stdev_guard: extreme vol with contra-trend BUY suppressed"
        if proposal.side.value == "Sell" and sma20_dist > 0:
            return "stdev_guard: extreme vol with contra-trend SELL suppressed"
        return None

    def _pdiv_size_multiplier(
        self,
        proposal: TradeProposal,
        feature_vector: FeatureVector | None,
    ) -> Decimal:
        """Price-divergence guard: reduce size for stale signals.

        Estimates how far the market may have drifted since the signal was
        generated using elapsed time × realized volatility as a proxy.
        Returns a multiplier in [0.5, 1.0].
        """
        if feature_vector is None or proposal.entry_price is None:
            return Decimal("1")

        elapsed_s = (datetime.now(UTC) - proposal.timestamp).total_seconds()
        if elapsed_s <= 0:
            return Decimal("1")

        try:
            vol_idx = feature_vector.feature_names.index("realized_vol_20")
            realized_vol = float(feature_vector.values[vol_idx])
        except (ValueError, IndexError):
            return Decimal("1")

        if realized_vol <= 0:
            return Decimal("1")

        # Estimate expected price drift: vol is daily annualised; scale to elapsed
        seconds_per_day = 86_400.0
        drift_estimate = realized_vol * (elapsed_s / seconds_per_day) ** 0.5
        if drift_estimate <= 0.005:  # < 0.5% expected drift → no penalty
            return Decimal("1")
        if drift_estimate >= 0.02:  # >= 2% → max penalty (50%)
            log.info(
                "execution.pdiv_size_reduced",
                symbol=proposal.symbol,
                elapsed_s=round(elapsed_s),
                drift_pct=round(drift_estimate * 100, 2),
                multiplier="0.5",
            )
            return Decimal("0.5")
        # Linear ramp between 0.5% and 2%: multiplier from 1.0 down to 0.5
        scale = (drift_estimate - 0.005) / (0.02 - 0.005)
        multiplier = 1.0 - scale * 0.5
        log.info(
            "execution.pdiv_size_reduced",
            symbol=proposal.symbol,
            elapsed_s=round(elapsed_s),
            drift_pct=round(drift_estimate * 100, 2),
            multiplier=round(multiplier, 3),
        )
        return Decimal(str(round(multiplier, 4)))

    def _sop_size_multiplier(
        self,
        proposal: TradeProposal,
        feature_vector: FeatureVector | None,
        regime_context: RegimeContext | None,
    ) -> Decimal:
        """Super-Opportunity boost: scale up size when orderbook strongly
        favours the trade direction and spread is not excessive.

        Returns a multiplier in [1.0, 1.2]. Only boosts — never penalises.
        """
        if feature_vector is None:
            return Decimal("1")

        try:
            ob_idx = feature_vector.feature_names.index("ob_imbalance_l5")
            ob_imbalance = float(feature_vector.values[ob_idx])
        except (ValueError, IndexError):
            return Decimal("1")

        spread_bps = (
            regime_context.spread_bps if regime_context is not None and regime_context.spread_bps is not None else None
        )
        # Only boost when spread is normal (< 20 bps = 0.20%)
        if spread_bps is not None and spread_bps > 20:
            return Decimal("1")

        # ob_imbalance > 0 = bid-side pressure (bullish); < 0 = ask-side (bearish)
        aligned = (proposal.side.value == "Buy" and ob_imbalance > 0.5) or (
            proposal.side.value == "Sell" and ob_imbalance < -0.5
        )
        if not aligned:
            return Decimal("1")

        # Scale boost from 1.0 to 1.2 proportionally to imbalance strength
        strength = min(abs(ob_imbalance), 1.0)
        boost = 1.0 + 0.2 * (strength - 0.5) / 0.5  # 0 at 0.5, 0.20 at 1.0
        log.info(
            "execution.sop_size_boost",
            symbol=proposal.symbol,
            ob_imbalance=round(ob_imbalance, 3),
            boost=round(boost, 3),
        )
        return Decimal(str(round(boost, 4)))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def submit(
        self,
        proposal: TradeProposal,
        capital: Decimal,
        available_balance: Decimal,
        feature_vector: FeatureVector | None = None,
        regime_context: RegimeContext | None = None,
    ) -> RiskDecision | None:
        """Evaluate and (optionally) execute a trade proposal.

        Returns:
            RiskDecision if the proposal was evaluated, None if it was
            skipped before reaching the RiskManager (dedup / cooldown).
        """
        async with self._submit_lock:
            return await self._submit_locked(
                proposal=proposal,
                capital=capital,
                available_balance=available_balance,
                feature_vector=feature_vector,
                regime_context=regime_context,
            )

    async def _submit_locked(
        self,
        proposal: TradeProposal,
        capital: Decimal,
        available_balance: Decimal,
        feature_vector: FeatureVector | None = None,
        regime_context: RegimeContext | None = None,
    ) -> RiskDecision | None:
        """Submit implementation guarded by ``_submit_lock``."""
        symbol = proposal.symbol
        _t_engine_start = datetime.now(UTC)

        # 1. Deduplication ─────────────────────────────────────────────
        if self.has_open_position(symbol):
            log.debug("execution.skipped_open_position", symbol=symbol)
            return None

        # Global position cap (enforced in both live and shadow mode)
        if len(self._open_positions) >= self._max_open_positions:
            log.debug(
                "execution.skipped_max_positions",
                symbol=symbol,
                open=len(self._open_positions),
                cap=self._max_open_positions,
            )
            return None

        # P0.2/P0.3: Block new entries while pending ones await resolution
        if self.has_pending_entries():
            self._diag_skip_pending += 1
            log.debug(
                "execution.skipped_pending_entries",
                symbol=symbol,
                pending=list(self._pending_entry_order_link_ids),
            )
            return None

        # P0.6: CANARY hard caps — checked before RiskManager (cannot be overridden)
        if self._is_canary:
            if len(self._open_positions) >= CANARY_MAX_OPEN_POSITIONS:
                log.warning(
                    "canary.blocked_max_positions",
                    symbol=symbol,
                    open_positions=len(self._open_positions),
                    cap=CANARY_MAX_OPEN_POSITIONS,
                )
                return None
            if self._exposure.total_exposure_pct >= CANARY_MAX_TOTAL_EXPOSURE_PCT:
                log.warning(
                    "canary.blocked_max_exposure",
                    symbol=symbol,
                    exposure_pct=str(self._exposure.total_exposure_pct),
                    cap=str(CANARY_MAX_TOTAL_EXPOSURE_PCT),
                )
                return None

        # 1b. Startup warmup + burst rate limits ───────────────────────
        reject_reason = self._check_rate_limits(symbol, proposal.side.value)
        if reject_reason:
            log.info("execution.skipped_rate_limit", symbol=symbol, reason=reject_reason)
            return None

        # 1c. Safety Mode Ladder ────────────────────────────────────────
        if self._safety_ladder is not None:
            if self._safety_ladder.blocks_new_entries():
                log.warning(
                    "safety_ladder.blocked_ak47",
                    symbol=symbol,
                    **self._safety_ladder.describe(),
                )
                return None
            ladder_level = self._safety_ladder.current_level()
            if ladder_level >= SafetyLevel.PINGPONG:
                log.info(
                    "safety_ladder.active",
                    symbol=symbol,
                    **self._safety_ladder.describe(),
                )

        # 2a. Entry cooldown (successful entries only) ────────────────
        last_entry = self._last_entry_at.get(symbol)
        if last_entry is not None:
            elapsed = datetime.now(tz=UTC) - last_entry
            if elapsed < self._cooldown:
                remaining = int((self._cooldown - elapsed).total_seconds())
                log.debug(
                    "execution.skipped_entry_cooldown",
                    symbol=symbol,
                    remaining_s=remaining,
                )
                return None

        # 2b. Failure cooldown (API errors) ────────────────────────────
        last_failure = self._last_failure_at.get(symbol)
        if last_failure is not None:
            elapsed = datetime.now(tz=UTC) - last_failure
            if elapsed < self._failure_cooldown:
                remaining = int((self._failure_cooldown - elapsed).total_seconds())
                log.debug(
                    "execution.skipped_failure_cooldown",
                    symbol=symbol,
                    remaining_s=remaining,
                )
                return None

        # 3. Instrument info (cached) ───────────────────────────────────
        try:
            instrument_info = await self.get_instrument_info(symbol)
        except Exception as exc:
            log.warning(
                "execution.instrument_info_error",
                symbol=symbol,
                error=str(exc),
            )
            return None

        # 3b. SL validation (live mode only) ─────────────────────────
        # In shadow mode we allow proposals without explicit SL for simulation.
        # In live mode, we never enter without a validated stop-loss.
        if not self._shadow_mode:
            if proposal.stop_loss is None:
                log.warning(
                    "execution.rejected_no_stop_loss",
                    symbol=symbol,
                    side=proposal.side.value,
                )
                return None
            # Verify SL is on the correct side of entry
            if proposal.entry_price is not None and proposal.entry_price > Decimal("0"):
                sl_valid = (proposal.side == OrderSide.BUY and proposal.stop_loss < proposal.entry_price) or (
                    proposal.side == OrderSide.SELL and proposal.stop_loss > proposal.entry_price
                )
                if not sl_valid:
                    log.warning(
                        "execution.rejected_invalid_stop_loss_side",
                        symbol=symbol,
                        side=proposal.side.value,
                        entry=str(proposal.entry_price),
                        stop_loss=str(proposal.stop_loss),
                    )
                    return None

        # 3c. Leverage enforcement (live mode only) ───────────────────
        if not self._shadow_mode:
            try:
                max_lev = self._risk_manager._limits.max_leverage
                await self._ensure_leverage(symbol, max_lev)
            except Exception as exc:
                log.warning("execution.leverage_check_failed", symbol=symbol, error=str(exc))
                return None

        # 3d. STDEV + Trend Guard ─────────────────────────────────────
        stdev_block = self._stdev_trend_guard(proposal, feature_vector)
        if stdev_block:
            log.info("execution.stdev_guard_blocked", symbol=symbol, reason=stdev_block)
            return None

        # 3e. pDiv size multiplier — stale-signal penalty ─────────────
        pdiv_multiplier = self._pdiv_size_multiplier(proposal, feature_vector)

        # 3f. SOP size boost — strong orderbook alignment ─────────────
        sop_multiplier = self._sop_size_multiplier(proposal, feature_vector, regime_context)

        # Safety ladder size multiplier [0.0, 1.0] — applied before SOP can boost
        ladder_multiplier = Decimal("1")
        if self._safety_ladder is not None:
            ladder_multiplier = Decimal(str(self._safety_ladder.size_multiplier()))

        # Combine signal-quality + ladder multipliers [0.0, 1.2]
        signal_qty_multiplier = pdiv_multiplier * sop_multiplier * ladder_multiplier

        # VWAP distance gate: penalise contra-VWAP entries
        vwap_dist = None
        if feature_vector is not None:
            try:
                vwap_dist = feature_vector.values[feature_vector.feature_names.index("vwap_distance_pct")]
            except (ValueError, AttributeError, IndexError):
                pass
        if vwap_dist is not None:
            # BUY above VWAP or SELL below VWAP is contra-trend — apply penalty
            if (proposal.side == OrderSide.BUY and vwap_dist > 0) or (
                proposal.side == OrderSide.SELL and vwap_dist < 0
            ):
                # penalty: 1.0 at 0% distance, 0.7 at ≥2% distance
                penalty = max(0.7, 1.0 - abs(vwap_dist) / 5.0)
                signal_qty_multiplier *= Decimal(str(round(penalty, 4)))
                log.debug(
                    "vwap_gate.applied", symbol=symbol, vwap_distance_pct=round(vwap_dist, 3), penalty=round(penalty, 4)
                )

        # 4. Risk evaluation ───────────────────────────────────────────
        # Extract spread and ATR for RiskManager sizing
        spread: Decimal | None = None
        atr: Decimal | None = None
        if regime_context is not None and regime_context.spread_bps is not None:
            # PositionSizer expects percent units: 50 bps => 0.5%, not 0.005 fraction.
            spread = Decimal(str(regime_context.spread_bps)) / Decimal("100")
        if feature_vector is not None:
            # ATR is computed as atr_14_pct in feature pipeline (ATR / price as fraction)
            try:
                idx = feature_vector.feature_names.index("atr_14_pct")
                atr = Decimal(str(feature_vector.values[idx]))
            except (ValueError, IndexError):
                pass
        _t_before_risk = datetime.now(UTC)
        try:
            decision = cast(
                RiskDecision,
                await self._risk_manager.evaluate(
                    proposal=proposal,
                    capital=capital,
                    available_balance=available_balance,
                    instrument_info=instrument_info,
                    feature_vector=feature_vector,
                    regime_context=regime_context,
                    spread=spread,
                    atr=atr,
                ),
            )
        except Exception as exc:
            log.error(
                "execution.risk_evaluation_error",
                symbol=symbol,
                error=str(exc),
            )
            return None
        _t_after_risk = datetime.now(UTC)

        approved = decision.status in (
            RiskDecisionStatus.APPROVED,
            RiskDecisionStatus.RESIZED,
        )

        log.info(
            "execution.risk_decision",
            symbol=symbol,
            status=decision.status.value,
            reason=decision.reason or "—",
            approved_qty=str(decision.approved_qty) if decision.approved_qty else None,
            portfolio_heat=decision.portfolio_heat,
        )
        if self._trade_journal is not None:
            await self._trade_journal.record_risk_decision(symbol, decision)

        # P0.8: Write baseline prediction event for every evaluated proposal
        if self._trade_journal is not None:
            try:
                await self._trade_journal.record_prediction_event(
                    symbol=symbol,
                    interval="1",
                    model_version="RULE_BASELINE_V1",
                    score=proposal.confidence,
                    strategy_signal=proposal.side.value,
                    decision="SHADOW_BASELINE" if self._shadow_mode else decision.status.value,
                )
            except Exception as _pred_exc:
                log.debug("execution.prediction_event_failed", error=str(_pred_exc))

        if not approved:
            return cast(RiskDecision, decision)
        exposure_reserved = True

        # 5b. Cost-aware entry gate (LIVE only) ─────────────────────────────
        if not self._shadow_mode:
            # Fail-closed: TP required for LIVE entries
            if proposal.take_profit is None:
                self._diag_no_tp_rejected += 1
                log.warning(
                    "execution.rejected_no_take_profit",
                    symbol=symbol,
                    side=proposal.side.value,
                )
                self._release_exposure_reservation(proposal)
                exposure_reserved = False
                return None
            if proposal.entry_price is None or proposal.entry_price <= Decimal("0"):
                log.warning(
                    "execution.rejected_no_entry_price_for_net_edge",
                    symbol=symbol,
                    side=proposal.side.value,
                )
                self._release_exposure_reservation(proposal)
                exposure_reserved = False
                return None

            # Fetch fee rates; fall back to conservative default if unavailable
            taker_default = Decimal("0.00055")  # 0.055% Bybit taker standard
            if self._fee_provider is not None:
                try:
                    fee_rates_obj = await self._fee_provider.get(symbol)
                    if fee_rates_obj is not None:
                        taker_default = Decimal(str(fee_rates_obj.taker_fee_rate))
                    else:
                        self._diag_fee_unavailable_rejected += 1
                        log.warning("execution.fee_rate_unavailable_using_default", symbol=symbol)
                except Exception as _fee_exc:
                    self._diag_fee_unavailable_rejected += 1
                    log.warning("execution.fee_rate_fetch_failed", symbol=symbol, error=str(_fee_exc))
            else:
                log.warning("execution.fee_provider_none_using_default", symbol=symbol)
            taker = taker_default

            entry_price_d = proposal.entry_price
            tp_d = self._round_exit_price(
                proposal.take_profit,
                instrument_info.tick_size,
                proposal.side,
                is_stop_loss=False,
            )
            if tp_d is None:
                self._release_exposure_reservation(proposal)
                exposure_reserved = False
                return None
            if proposal.side == OrderSide.BUY:
                gross_edge_pct = (tp_d - entry_price_d) / entry_price_d * Decimal("100")
            else:
                gross_edge_pct = (entry_price_d - tp_d) / entry_price_d * Decimal("100")
            entry_fee_pct = taker * Decimal("100")
            exit_fee_pct = taker * Decimal("100")
            round_trip_fee_pct = entry_fee_pct + exit_fee_pct
            spread_pct = Decimal(str(self._max_spread_bps)) / Decimal("100")
            # P1: Round-trip slippage = entry slippage + exit slippage = 2 * EXPECTED_SLIPPAGE_PCT
            entry_slippage_pct = Decimal(str(self._expected_slippage_pct))
            exit_slippage_pct = Decimal(str(self._expected_slippage_pct))
            round_trip_slippage_pct = entry_slippage_pct + exit_slippage_pct
            funding_pct = Decimal(str(self._funding_buffer_pct))
            safety_margin_pct = Decimal(str(self._net_edge_safety_margin_pct))
            net_edge_pct = (
                gross_edge_pct
                - round_trip_fee_pct
                - spread_pct
                - round_trip_slippage_pct
                - funding_pct
                - safety_margin_pct
            )
            min_edge = Decimal(str(self._min_net_edge_pct))

            log.info(
                "execution.net_edge_check",
                symbol=symbol,
                side=proposal.side.value,
                entry_price=float(entry_price_d),
                take_profit=float(tp_d),
                raw_take_profit=float(proposal.take_profit),
                gross_edge_pct=float(round(gross_edge_pct, 4)),
                entry_fee_pct=float(round(entry_fee_pct, 4)),
                exit_fee_pct=float(round(exit_fee_pct, 4)),
                round_trip_fee_pct=float(round(round_trip_fee_pct, 4)),
                spread_bps=float(self._max_spread_bps),
                spread_cost_pct=float(round(spread_pct, 4)),
                entry_slippage_cost_pct=float(round(entry_slippage_pct, 4)),
                exit_slippage_cost_pct=float(round(exit_slippage_pct, 4)),
                round_trip_slippage_cost_pct=float(round(round_trip_slippage_pct, 4)),
                funding_buffer_pct=float(round(funding_pct, 4)),
                safety_margin_pct=float(round(safety_margin_pct, 4)),
                net_edge_pct=float(round(net_edge_pct, 4)),
                required_min_net_edge_pct=float(min_edge),
                decision="allow" if net_edge_pct >= min_edge else "reject",
            )

            if net_edge_pct < min_edge:
                self._diag_net_edge_rejected += 1
                log.warning(
                    "execution.net_edge_too_low",
                    symbol=symbol,
                    net_edge_pct=float(round(net_edge_pct, 4)),
                    required_min_net_edge_pct=float(min_edge),
                )
                self._release_exposure_reservation(proposal)
                exposure_reserved = False
                return None

        # 4b. Apply signal-quality multipliers (pDiv penalty / SOP boost) ──
        assert decision.approved_qty is not None
        if signal_qty_multiplier != Decimal("1"):
            adjusted_qty = decision.approved_qty * signal_qty_multiplier
            # Round down to qty_step; reject if below min_order_qty
            from decimal import ROUND_DOWN as _RD

            qty_step = instrument_info.qty_step
            adjusted_qty = (adjusted_qty / qty_step).to_integral_value(rounding=_RD) * qty_step
            if adjusted_qty < instrument_info.min_order_qty:
                log.warning(
                    "execution.signal_qty_adjustment_below_min_qty",
                    symbol=symbol,
                    multiplier=str(signal_qty_multiplier),
                    adjusted_qty=str(adjusted_qty),
                    min_order_qty=str(instrument_info.min_order_qty),
                )
                self._release_exposure_reservation(proposal)
                return None
            can_reserve, reserve_reason = self._reserve_adjusted_exposure(proposal, adjusted_qty, instrument_info)
            if not can_reserve:
                log.warning(
                    "execution.signal_qty_adjustment_rejected",
                    symbol=symbol,
                    multiplier=str(signal_qty_multiplier),
                    adjusted_qty=str(adjusted_qty),
                    reason=reserve_reason,
                )
                return None
            decision = decision.model_copy(update={"approved_qty": adjusted_qty, "status": RiskDecisionStatus.RESIZED})
            log.info(
                "execution.signal_qty_adjusted",
                symbol=symbol,
                multiplier=str(signal_qty_multiplier),
                adjusted_qty=str(adjusted_qty),
            )

        # 5. Build OrderIntent ─────────────────────────────────────────
        assert decision.approved_qty is not None
        intent = self._build_intent(proposal, decision, instrument_info)

        # 6. Execute or shadow ─────────────────────────────────────────
        if self._shadow_mode:
            self._diag_shadow_order_would_be_placed += 1
            log.info(
                "shadow.order_would_be_placed",
                symbol=symbol,
                side=proposal.side.value,
                qty=str(decision.approved_qty),
                entry=str(proposal.entry_price),
                stop=str(proposal.stop_loss),
                tp=str(proposal.take_profit),
                order_link_id=intent.order_link_id,
                confidence=round(proposal.confidence, 3),
                mode="SHADOW_NO_EXECUTION",
            )
            if self._trade_journal is not None:
                try:
                    await self._trade_journal.record_order_event(
                        order_link_id=intent.order_link_id,
                        proposal_id=intent.proposal_id,
                        decision_id=intent.decision_id,
                        symbol=symbol,
                        side=proposal.side.value,
                        qty=decision.approved_qty,
                        status="SHADOW",
                    )
                except Exception as _shadow_journal_exc:
                    log.debug("execution.shadow_journal_write_failed", error=str(_shadow_journal_exc))
        else:
            # Last-moment exchange guard: only the raw exchange minimum matters here.
            # The buffer is applied once at sizing time (RiskManager). Applying it
            # again here caused valid orders to be blocked when price moved slightly.
            # Rule: reject only if below raw exchange minimum; warn if buffer consumed.
            if instrument_info.min_notional is not None and instrument_info.min_notional > Decimal("0"):
                try:
                    conservative_price = await self._adapter.get_conservative_market_price(
                        self._category, symbol, proposal.side.value
                    )
                    executable_notional = intent.qty * conservative_price
                    exchange_min = instrument_info.min_notional  # raw exchange minimum — NO buffer
                    sizing_target = exchange_min * (Decimal("1") + self._min_notional_buffer / Decimal("100"))

                    if executable_notional < exchange_min:
                        # Below raw exchange minimum → hard reject (would trigger code=110094)
                        log.warning(
                            "execution.below_exchange_minimum_rejected",
                            symbol=symbol,
                            executable_notional=str(executable_notional),
                            exchange_min=str(exchange_min),
                        )
                        self._release_exposure_reservation(proposal)
                        exposure_reserved = False
                        return None

                    if executable_notional < sizing_target:
                        # Buffer consumed by price movement but still above exchange minimum → allow
                        log.warning(
                            "execution.buffer_consumed_before_submit",
                            symbol=symbol,
                            executable_notional=str(executable_notional),
                            sizing_target=str(sizing_target),
                            exchange_min=str(exchange_min),
                        )
                except Exception as _price_exc:
                    self._last_failure_at[symbol] = datetime.now(tz=UTC)
                    log.warning(
                        "execution.conservative_price_check_failed",
                        symbol=symbol,
                        error=str(_price_exc),
                    )
                    if self._trade_journal is not None:
                        try:
                            await self._trade_journal.record_order_event(
                                order_link_id=intent.order_link_id,
                                proposal_id=intent.proposal_id,
                                decision_id=intent.decision_id,
                                symbol=symbol,
                                side=proposal.side.value,
                                qty=decision.approved_qty,
                                status="REJECTED_PRICE_CHECK_FAILED",
                                error=str(_price_exc),
                            )
                        except Exception as _journal_exc:  # noqa: BLE001
                            log.debug("execution.price_check_journal_failed", error=str(_journal_exc))
                    self._release_exposure_reservation(proposal)
                    exposure_reserved = False
                    return None
            # P0.1: Durable write CREATED_LOCAL before any REST call.
            if self._trade_journal is not None and self._trade_journal.is_enabled:
                try:
                    await self._trade_journal.record_order_event_required(
                        order_link_id=intent.order_link_id,
                        proposal_id=intent.proposal_id,
                        decision_id=intent.decision_id,
                        symbol=symbol,
                        side=proposal.side.value,
                        qty=decision.approved_qty,
                        status="CREATED_LOCAL",
                    )
                except Exception as _durable_exc:
                    log.error(
                        "execution.durable_created_local_failed_aborting",
                        symbol=symbol,
                        order_link_id=intent.order_link_id,
                        error=str(_durable_exc),
                    )
                    self._release_exposure_reservation(proposal)
                    exposure_reserved = False
                    return None

            # P0.3: Register pending entry slot (with symbol for screener protection)
            self.mark_entry_submitted(intent.order_link_id, symbol=symbol)
            # Persist pending registration so resolution state survives restarts
            if self._trade_journal is not None and self._trade_journal.is_enabled:
                try:
                    await self._trade_journal.record_order_pending(intent.order_link_id, symbol)
                except Exception as _pend_exc:
                    log.debug(
                        "execution.record_order_pending_failed",
                        order_link_id=intent.order_link_id,
                        error=str(_pend_exc),
                    )

            # P0.6: Second canary gate immediately before REST
            if self._is_canary:
                if len(self._open_positions) >= CANARY_MAX_OPEN_POSITIONS:
                    await self.resolve_pending_durable(intent.order_link_id, symbol)
                    self._release_exposure_reservation(proposal)
                    exposure_reserved = False
                    log.warning("canary.blocked_max_positions_pre_rest", symbol=symbol)
                    return None
                if self._exposure.total_exposure_pct >= CANARY_MAX_TOTAL_EXPOSURE_PCT:
                    await self.resolve_pending_durable(intent.order_link_id, symbol)
                    self._release_exposure_reservation(proposal)
                    exposure_reserved = False
                    log.warning("canary.blocked_max_exposure_pre_rest", symbol=symbol)
                    return None

            # P0.1: Durable write SUBMITTING immediately before REST call.
            if self._trade_journal is not None and self._trade_journal.is_enabled:
                try:
                    await self._trade_journal.record_order_event_required(
                        order_link_id=intent.order_link_id,
                        proposal_id=intent.proposal_id,
                        decision_id=intent.decision_id,
                        symbol=symbol,
                        side=proposal.side.value,
                        qty=decision.approved_qty,
                        status="SUBMITTING",
                    )
                except Exception as _durable_exc:
                    log.error(
                        "execution.durable_submitting_failed_aborting",
                        symbol=symbol,
                        order_link_id=intent.order_link_id,
                        error=str(_durable_exc),
                    )
                    await self.resolve_pending_durable(intent.order_link_id, symbol)
                    self._release_exposure_reservation(proposal)
                    exposure_reserved = False
                    return None

            if self._entry_order_mode == "MAKER_FIRST":
                entered = await self._execute_maker_first(intent, proposal, decision, instrument_info)
                if not entered:
                    self._release_exposure_reservation(proposal)
                    exposure_reserved = False
                    return None
            else:
                try:
                    _t_order_submit = datetime.now(UTC)
                    resp = await self._adapter.place_order(intent)
                    _t_order_confirm = datetime.now(UTC)
                    exchange_order_id = resp.get("result", {}).get("orderId", "?")
                    self._diag_order_placed += 1
                    _latency_proposal_ms = int((_t_order_confirm - proposal.timestamp).total_seconds() * 1000)
                    _latency_engine_ms = int((_t_order_confirm - _t_engine_start).total_seconds() * 1000)
                    _latency_risk_eval_ms = int((_t_after_risk - _t_before_risk).total_seconds() * 1000)
                    _latency_exchange_ms = int((_t_order_confirm - _t_order_submit).total_seconds() * 1000)
                    log.info(
                        "execution.order_placed",
                        symbol=symbol,
                        side=proposal.side.value,
                        qty=str(decision.approved_qty),
                        exchange_order_id=exchange_order_id,
                        order_link_id=intent.order_link_id,
                        latency_proposal_to_confirm_ms=_latency_proposal_ms,
                        latency_engine_ms=_latency_engine_ms,
                        latency_risk_eval_ms=_latency_risk_eval_ms,
                        latency_exchange_ms=_latency_exchange_ms,
                    )
                    if self._trade_journal is not None:
                        await self._trade_journal.record_order_event(
                            order_link_id=intent.order_link_id,
                            proposal_id=intent.proposal_id,
                            decision_id=intent.decision_id,
                            symbol=symbol,
                            side=proposal.side.value,
                            qty=decision.approved_qty,
                            status="PLACED",
                            exchange_order_id=exchange_order_id,
                        )
                except Exception as exc:
                    self._diag_order_failed += 1
                    await self.resolve_pending_durable(intent.order_link_id, symbol)
                    self._release_exposure_reservation(proposal)
                    exposure_reserved = False
                    # Record failure timestamp (NOT an entry cooldown — separate state)
                    self._last_failure_at[symbol] = datetime.now(tz=UTC)
                    log.error(
                        "execution.order_failed",
                        symbol=symbol,
                        error=str(exc),
                    )
                    if self._trade_journal is not None:
                        await self._trade_journal.record_order_event(
                            order_link_id=intent.order_link_id,
                            proposal_id=intent.proposal_id,
                            decision_id=intent.decision_id,
                            symbol=symbol,
                            side=proposal.side.value,
                            qty=decision.approved_qty,
                            status="FAILED",
                            error=str(exc),
                        )
                    return None

        # 7. Update local state ────────────────────────────────────────
        self._last_entry_at[symbol] = datetime.now(tz=UTC)

        entry_price = proposal.entry_price or Decimal("0")
        notional = decision.approved_qty * entry_price
        # Set the position registry entry optimistically for BOTH live and shadow
        # paths so the strategy loop cannot open a duplicate entry while a fill
        # confirmation is still in-flight. For live orders the WS position event
        # (or the next startup sync) will replace this with real exchange data.
        self._open_positions[symbol] = {
            "side": proposal.side,
            "size": decision.approved_qty,
            "entry_price": entry_price,
            "notional": notional,
            "order_link_id": intent.order_link_id,
            "opened_at": datetime.now(tz=UTC),
        }

        if notional > Decimal("0"):
            await self._exposure.update_position(symbol, proposal.side.value, notional)
        if exposure_reserved:
            self._release_exposure_reservation(proposal)

        if not self._shadow_mode:
            log.info(
                "execution.live_order_placed_optimistic_position_set",
                symbol=symbol,
                order_link_id=intent.order_link_id,
            )
            # Set exchange-side trailing stop asynchronously (non-blocking).
            # Store the task reference so it isn't GC'd before completion.
            _ts_task = asyncio.create_task(self._setup_trailing_stop(symbol, entry_price, atr))
            self._trailing_stop_tasks.add(_ts_task)
            _ts_task.add_done_callback(self._trailing_stop_tasks.discard)

        return cast(RiskDecision, decision)

    # ------------------------------------------------------------------
    # Maker-first execution
    # ------------------------------------------------------------------

    async def _execute_maker_first(
        self,
        intent: OrderIntent,
        proposal: TradeProposal,
        decision: RiskDecision,
        instrument_info: InstrumentInfo,
    ) -> bool:
        """POST_ONLY limit at the touch, then escalate to taker or abort.

        Returns True when a position entry was achieved (full or partial fill,
        or successful escalation) — the caller then runs the shared post-entry
        path. Pending/durable state mechanics mirror the market path exactly.
        """
        symbol = intent.symbol

        # 1. Price the maker order off the live touch. No quote → no entry.
        try:
            bid, ask = await self._adapter.get_best_bid_ask(self._category, symbol)
        except Exception as exc:
            await self._abort_maker_entry(intent, decision, reason=f"no_quote: {exc}")
            return False
        tick = instrument_info.tick_size if instrument_info.tick_size > 0 else Decimal("0")
        if bid <= 0 or ask <= bid:
            await self._abort_maker_entry(intent, decision, reason="locked_or_crossed_market")
            return False
        spread_bps_val = float((ask - bid) / bid * 10000)
        if intent.side == OrderSide.BUY:
            if spread_bps_val < 5.0:
                # Tight spread: place at mid
                price = (bid + ask) / Decimal("2")
                if tick > 0:
                    price = (price // tick) * tick  # floor to tick
            elif spread_bps_val < 30.0:
                # Normal spread: bid + one tick
                price = bid + tick if tick > 0 and bid + tick < ask else bid
            else:
                # Wide spread: aggressive maker at bid
                price = bid
        else:
            if spread_bps_val < 5.0:
                price = (bid + ask) / Decimal("2")
                if tick > 0:
                    price = ((price // tick) + 1) * tick  # ceil to tick
            elif spread_bps_val < 30.0:
                price = ask - tick if tick > 0 and ask - tick > bid else ask
            else:
                price = ask

        maker_intent = intent.model_copy(
            update={"order_type": OrderType.LIMIT, "price": price, "time_in_force": "PostOnly"}
        )

        # 2. Submit POST_ONLY limit
        try:
            resp = await self._adapter.place_order(maker_intent)
        except Exception as exc:
            self._diag_order_failed += 1
            await self.resolve_pending_durable(intent.order_link_id, symbol)
            self._last_failure_at[symbol] = datetime.now(tz=UTC)
            log.error("execution.order_failed", symbol=symbol, error=str(exc), mode="maker_first")
            await self._journal_order_event(maker_intent, decision, status="FAILED", error=str(exc))
            return False

        exchange_order_id = resp.get("result", {}).get("orderId", "?")
        self._diag_order_placed += 1
        log.info(
            "maker.order_placed",
            symbol=symbol,
            side=intent.side.value,
            qty=str(intent.qty),
            price=str(price),
            order_link_id=intent.order_link_id,
            exchange_order_id=exchange_order_id,
            timeout_s=self._maker_timeout_s,
            ttl_s=self._maker_ttl_s,
        )
        await self._journal_order_event(maker_intent, decision, status="PLACED", exchange_order_id=exchange_order_id)

        # 3. Wait for the fill. With escalation we decide at the timeout;
        #    without it the order may rest until its full TTL.
        wait_s = self._maker_timeout_s if self._maker_allow_escalation else self._maker_ttl_s
        state = await self._wait_maker_fill(symbol, intent.order_link_id, wait_s)

        if state == "filled":
            self._diag_maker_filled += 1
            log.info("maker.filled", symbol=symbol, order_link_id=intent.order_link_id)
            return True

        if state == "open":
            # 4. Cancel the resting remainder. A cancel error can mean either a
            #    racing fill OR a live order we failed to cancel — verify which.
            cancel_failed = False
            try:
                await self._adapter.cancel_order(self._category, symbol, intent.order_link_id)
            except Exception as exc:
                cancel_failed = True
                log.info(
                    "maker.cancel_failed_checking_fill",
                    symbol=symbol,
                    order_link_id=intent.order_link_id,
                    error=str(exc),
                )
            if cancel_failed:
                # Fail closed: if the order may still be live on the exchange we
                # must NOT escalate (a taker on top of a live limit doubles the
                # entry) and must NOT release the pending slot — the private WS
                # fill/cancel event or stale-pending reconciliation resolves it.
                try:
                    open_orders = await self._adapter.get_open_orders(self._category, symbol)
                    still_live = any(str(o.get("orderLinkId")) == intent.order_link_id for o in open_orders)
                except Exception as verify_exc:
                    log.warning(
                        "maker.cancel_state_unknown_fail_closed",
                        symbol=symbol,
                        order_link_id=intent.order_link_id,
                        error=str(verify_exc),
                    )
                    self._last_failure_at[symbol] = datetime.now(tz=UTC)
                    return False
                if still_live:
                    self._diag_maker_aborted += 1
                    self._last_failure_at[symbol] = datetime.now(tz=UTC)
                    log.warning(
                        "maker.cancel_failed_order_live",
                        symbol=symbol,
                        order_link_id=intent.order_link_id,
                    )
                    return False
            try:
                await self._sync_positions_locked()
            except Exception as exc:
                log.warning("maker.position_sync_failed", symbol=symbol, error=str(exc))
            if self.has_open_position(symbol):
                # Partial (or racing full) fill — TP/SL are attached to the
                # position via tpslMode=Full, the position manager takes over.
                self._diag_maker_filled += 1
                log.info(
                    "maker.filled",
                    symbol=symbol,
                    order_link_id=intent.order_link_id,
                    partial=True,
                )
                return True

        # 5. No fill at all ("gone" = PostOnly rejected/cancelled by exchange).
        allowed, reason = await self._maker_escalation_allowed(intent, price, time_waited_s=wait_s)
        if not allowed:
            self._diag_maker_aborted += 1
            log.info(
                "maker.aborted",
                symbol=symbol,
                order_link_id=intent.order_link_id,
                reason=reason,
            )
            await self.resolve_pending_durable(intent.order_link_id, symbol)
            await self._journal_order_event(maker_intent, decision, status="MAKER_ABORTED", error=reason)
            return False

        # 5b. Last fill re-check: the escalation gate above took a REST round
        # trip — a fill landing in that window would otherwise be doubled by
        # the taker order.
        try:
            await self._sync_positions_locked()
        except Exception as exc:
            log.warning("maker.position_sync_failed", symbol=symbol, error=str(exc))
        if self.has_open_position(symbol):
            self._diag_maker_filled += 1
            log.info("maker.filled", symbol=symbol, order_link_id=intent.order_link_id, late=True)
            return True

        # 6. Escalate the entry to a market (taker) order under a fresh link id.
        escalation_link_id = (intent.order_link_id[:35] + "E")[:36]
        market_intent = intent.model_copy(
            update={
                "intent_id": uuid.uuid4(),
                "order_link_id": escalation_link_id,
                "order_type": OrderType.MARKET,
                "price": None,
                "time_in_force": "GTC",
            }
        )
        self.mark_entry_submitted(escalation_link_id, symbol=symbol)
        if self._trade_journal is not None and self._trade_journal.is_enabled:
            try:
                await self._trade_journal.record_order_pending(escalation_link_id, symbol)
            except Exception as exc:
                log.debug("maker.record_pending_failed", order_link_id=escalation_link_id, error=str(exc))
        # The original maker order is terminally cancelled — release its slot.
        await self.resolve_pending_durable(intent.order_link_id, symbol)

        try:
            resp = await self._adapter.place_order(market_intent)
        except Exception as exc:
            self._diag_order_failed += 1
            await self.resolve_pending_durable(escalation_link_id, symbol)
            self._last_failure_at[symbol] = datetime.now(tz=UTC)
            log.error("execution.order_failed", symbol=symbol, error=str(exc), mode="maker_escalation")
            await self._journal_order_event(market_intent, decision, status="FAILED", error=str(exc))
            return False

        exchange_order_id = resp.get("result", {}).get("orderId", "?")
        self._diag_order_placed += 1
        self._diag_maker_escalated += 1
        log.info(
            "maker.escalated",
            symbol=symbol,
            side=intent.side.value,
            qty=str(intent.qty),
            order_link_id=escalation_link_id,
            exchange_order_id=exchange_order_id,
        )
        await self._journal_order_event(market_intent, decision, status="PLACED", exchange_order_id=exchange_order_id)
        return True

    async def _setup_trailing_stop(
        self,
        symbol: str,
        entry_price: Decimal,
        atr: Decimal | None,
    ) -> None:
        """Set an exchange-side trailing stop after a live entry.

        Only called in live mode. The trailing distance is ``atr * atr_multiple``
        floored to ``trailing_stop_min_pct`` of entry price.
        """
        if self._shadow_mode:
            return
        try:
            if atr is not None and atr > 0:
                distance = atr * Decimal(str(self._trailing_stop_atr_multiple))
            else:
                distance = entry_price * Decimal(str(self._trailing_stop_min_pct))
            min_distance = entry_price * Decimal(str(self._trailing_stop_min_pct))
            distance = max(distance, min_distance)
            await self._adapter.set_trading_stop(
                category=self._category,
                symbol=symbol,
                trailing_stop=str(distance),
            )
            log.info(
                "trailing_stop.set",
                symbol=symbol,
                trailing_stop=str(distance),
                atr=str(atr) if atr else None,
            )
        except Exception as exc:
            log.warning("trailing_stop.set_failed", symbol=symbol, error=str(exc))

    async def check_profit_gates(self) -> None:
        """Tighten TP/SL for positions that have exceeded profit thresholds.

        Live mode only. Designed to be called periodically (e.g. every 10s)
        by the strategy loop.
        """
        if self._shadow_mode:
            return
        for symbol, pos in list(self._open_positions.items()):
            try:
                bid, ask = await self._adapter.get_best_bid_ask(self._category, symbol)
                mid = (bid + ask) / Decimal("2")
                entry = pos.get("entry_price", Decimal("0"))
                side = pos.get("side")
                if not entry or entry <= 0:
                    continue
                if side == OrderSide.BUY:
                    pnl_pct = float((mid - entry) / entry * 100)
                else:
                    pnl_pct = float((entry - mid) / entry * 100)
                if pnl_pct >= self._profit_lock_pct:
                    # Lock in profits: move SL to break-even + small buffer
                    if side == OrderSide.BUY:
                        new_sl = entry * Decimal("1.005")
                    else:
                        new_sl = entry * Decimal("0.995")
                    await self._adapter.set_trading_stop(
                        category=self._category,
                        symbol=symbol,
                        stop_loss=str(new_sl),
                    )
                    log.info(
                        "profit_gate.sl_locked",
                        symbol=symbol,
                        pnl_pct=round(pnl_pct, 2),
                        new_sl=str(new_sl),
                    )
                elif pnl_pct >= self._profit_gate_pct:
                    # Tighten TP by reducing to 60% of remaining move
                    if side == OrderSide.BUY:
                        new_tp = mid + (mid - entry) * Decimal("0.6")
                    else:
                        new_tp = mid - (entry - mid) * Decimal("0.6")
                    await self._adapter.set_trading_stop(
                        category=self._category,
                        symbol=symbol,
                        take_profit=str(new_tp),
                    )
                    log.info(
                        "profit_gate.tp_tightened",
                        symbol=symbol,
                        pnl_pct=round(pnl_pct, 2),
                        new_tp=str(new_tp),
                    )
            except Exception as exc:
                log.warning("profit_gate.check_failed", symbol=symbol, error=str(exc))

    async def _wait_maker_fill(self, symbol: str, order_link_id: str, wait_s: float) -> str:
        """Poll the open-orders list until fill, disappearance or timeout.

        Returns "filled" (position confirmed), "gone" (order vanished without a
        position — e.g. PostOnly rejected) or "open" (still resting at timeout).
        """
        deadline = asyncio.get_event_loop().time() + wait_s
        while True:
            try:
                open_orders = await self._adapter.get_open_orders(self._category, symbol)
                still_open = any(str(o.get("orderLinkId")) == order_link_id for o in open_orders)
                if not still_open:
                    try:
                        await self._sync_positions_locked()
                    except Exception as exc:
                        log.warning("maker.position_sync_failed", symbol=symbol, error=str(exc))
                    return "filled" if self.has_open_position(symbol) else "gone"
            except Exception as exc:
                # Transient API error — keep waiting, the order state is unknown
                log.debug("maker.fill_poll_failed", symbol=symbol, error=str(exc))
            if asyncio.get_event_loop().time() >= deadline:
                return "open"
            await asyncio.sleep(_MAKER_POLL_INTERVAL_S)

    async def _maker_escalation_allowed(
        self,
        intent: OrderIntent,
        maker_price: Decimal,
        time_waited_s: float = 0.0,
    ) -> tuple[bool, str]:
        """Safety gate before paying taker: config, queue depth, book pressure, price drift.

        Queue-aware escalation: if the imbalance is adverse but we have waited
        long enough (> 75% of our maker window), the queue in front of us is
        likely deep and growing — override the imbalance block and allow taker
        escalation to ensure entry.
        """
        if not self._maker_allow_escalation:
            return False, "escalation_disabled"

        # --- Queue-depth heuristic ---
        # Estimate how deep we are in the queue by how long we've been waiting.
        # If > _QUEUE_DEPTH_ESCALATION_PCT of our maker window has elapsed, the
        # queue is likely stale and deep — prefer taker regardless of imbalance.
        queue_override = False
        if self._maker_ttl_s > 0 and time_waited_s >= _QUEUE_DEPTH_MIN_WAIT_S:
            time_fraction = time_waited_s / self._maker_ttl_s
            if time_fraction >= _QUEUE_DEPTH_ESCALATION_PCT:
                queue_override = True
                log.debug(
                    "maker.queue_depth_override",
                    symbol=intent.symbol,
                    time_waited_s=round(time_waited_s, 1),
                    time_fraction=round(time_fraction, 2),
                )

        # Imbalance must not contradict the direction (fail open when unknown).
        # Skip this gate when queue-depth override is active.
        if not queue_override and self._imbalance_provider is not None:
            try:
                imbalance = self._imbalance_provider(intent.symbol)
            except Exception:
                imbalance = None
            if imbalance is not None:
                against = imbalance < 0 if intent.side == OrderSide.BUY else imbalance > 0
                if against:
                    return False, f"imbalance_against:{round(imbalance, 3)}"

        # Price must not have run away from where the maker order was resting
        try:
            current = await self._adapter.get_conservative_market_price(
                self._category, intent.symbol, intent.side.value
            )
        except Exception as exc:
            return False, f"no_price:{exc}"
        if maker_price <= 0:
            return False, "bad_maker_price"
        drift_pct = abs(float((current - maker_price) / maker_price)) * 100.0
        if drift_pct > _MAKER_MAX_ESCALATION_DRIFT_PCT:
            return False, f"price_drifted:{round(drift_pct, 4)}pct"
        reason = "queue_depth_override" if queue_override else "ok"
        return True, reason

    async def _abort_maker_entry(self, intent: OrderIntent, decision: RiskDecision, reason: str) -> None:
        """Release pending state and journal an aborted maker entry."""
        self._diag_maker_aborted += 1
        log.info("maker.aborted", symbol=intent.symbol, order_link_id=intent.order_link_id, reason=reason)
        await self.resolve_pending_durable(intent.order_link_id, intent.symbol)
        await self._journal_order_event(intent, decision, status="MAKER_ABORTED", error=reason)

    async def _journal_order_event(
        self,
        intent: OrderIntent,
        decision: RiskDecision,
        *,
        status: str,
        exchange_order_id: str | None = None,
        error: str | None = None,
    ) -> None:
        if self._trade_journal is None:
            return
        try:
            await self._trade_journal.record_order_event(
                order_link_id=intent.order_link_id,
                proposal_id=intent.proposal_id,
                decision_id=intent.decision_id,
                symbol=intent.symbol,
                side=intent.side.value,
                qty=intent.qty,
                status=status,
                exchange_order_id=exchange_order_id,
                error=error,
            )
        except Exception as exc:
            log.debug("execution.journal_event_failed", status=status, error=str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_intent(
        self,
        proposal: TradeProposal,
        decision: RiskDecision,
        instrument_info: InstrumentInfo,
    ) -> OrderIntent:
        # Compact UUID → alphanumeric ID (max 36 chars for Bybit)
        link_id = str(proposal.proposal_id).replace("-", "")[:36]

        assert decision.approved_qty is not None
        take_profit = self._round_exit_price(
            proposal.take_profit,
            instrument_info.tick_size,
            proposal.side,
            is_stop_loss=False,
        )
        stop_loss = self._round_exit_price(
            proposal.stop_loss,
            instrument_info.tick_size,
            proposal.side,
            is_stop_loss=True,
        )
        use_limit = self._entry_order_mode == "POST_ONLY_LIMIT"
        return OrderIntent(
            decision_id=decision.decision_id,
            proposal_id=proposal.proposal_id,
            symbol=proposal.symbol,
            market_type=proposal.market_type,
            side=proposal.side,
            order_type=OrderType.LIMIT if use_limit else OrderType.MARKET,
            qty=decision.approved_qty,
            price=None,  # Market order — no price needed
            order_link_id=link_id,
            take_profit=take_profit,
            stop_loss=stop_loss,
            tp_order_type=OrderType.LIMIT if use_limit else OrderType.MARKET,
            sl_order_type=OrderType.MARKET,  # SL always market
        )

    def _round_exit_price(
        self,
        price: Decimal | None,
        tick_size: Decimal,
        side: OrderSide,
        is_stop_loss: bool,
    ) -> Decimal | None:
        """Round exit prices without moving stops to the wrong side."""
        if price is None or tick_size <= Decimal("0"):
            return price
        rounding = ROUND_DOWN
        if side == OrderSide.SELL:
            rounding = ROUND_CEILING
        ticks = (price / tick_size).to_integral_value(rounding=rounding)
        return ticks * tick_size

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        return {
            "shadow_mode": self._shadow_mode,
            "open_positions": {
                sym: {
                    "side": pos["side"].value,
                    "size": str(pos["size"]),
                    "entry_price": str(pos["entry_price"]),
                }
                for sym, pos in self._open_positions.items()
            },
            "cooldown_s": int(self._cooldown.total_seconds()),
            "failure_cooldown_s": int(self._failure_cooldown.total_seconds()),
            "last_entries": {sym: ts.isoformat() for sym, ts in self._last_entry_at.items()},
            "last_failures": {sym: ts.isoformat() for sym, ts in self._last_failure_at.items()},
        }

    def get_diag_counts(self) -> dict[str, int]:
        """Return cumulative diagnostic counters since startup."""
        return {
            "skipped_pending_entries": self._diag_skip_pending,
            "order_placed": self._diag_order_placed,
            "shadow_order_would_be_placed": self._diag_shadow_order_would_be_placed,
            "order_failed": self._diag_order_failed,
            "pending_entry_count": len(self._pending_entry_order_link_ids),
            "net_edge_rejected": self._diag_net_edge_rejected,
            "no_tp_rejected": self._diag_no_tp_rejected,
            "fee_unavailable_rejected": self._diag_fee_unavailable_rejected,
            "maker_filled": self._diag_maker_filled,
            "maker_escalated": self._diag_maker_escalated,
            "maker_aborted": self._diag_maker_aborted,
        }

    def pending_entry_diagnostics(self) -> dict[str, Any]:
        """Return pending entry details for diagnostics/heartbeat."""
        now = datetime.now(tz=UTC)
        ids = sorted(self._pending_entry_order_link_ids)
        symbols = [self._pending_entry_symbols.get(oid, "?") for oid in ids]
        oldest_age_s: float | None = None
        if ids and self._pending_entry_created_at:
            oldest_ts = min(
                (self._pending_entry_created_at[oid] for oid in ids if oid in self._pending_entry_created_at),
                default=None,
            )
            if oldest_ts is not None:
                oldest_age_s = (now - oldest_ts).total_seconds()
        return {
            "pending_entry_count": len(ids),
            "pending_entry_ids": ids[:10],
            "pending_entry_symbols": symbols[:10],
            "oldest_pending_age_s": oldest_age_s,
        }
