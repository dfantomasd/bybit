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
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal
from typing import Any

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

log = structlog.get_logger(__name__)

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
        self._entry_order_mode = entry_order_mode

        # P0: Hard block non-MARKET entry modes — only MARKET is supported during current rollout
        normalized_entry_mode = str(self._entry_order_mode).strip().upper()
        if normalized_entry_mode != "MARKET":
            raise ValueError("Only MARKET entry mode is supported during current rollout")

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
        self._diag_order_failed: int = 0
        self._diag_net_edge_rejected: int = 0
        self._diag_no_tp_rejected: int = 0
        self._diag_fee_unavailable_rejected: int = 0

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

    def has_pending_order_for_symbol(self, symbol: str) -> bool:
        """Return True if there is a pending (unresolved) entry for this symbol."""
        return symbol in self._pending_entry_symbols.values()

    def restore_pending_entries(self, order_link_ids: list[str]) -> None:
        """Restore pending entry IDs from durable storage at startup.

        Empty and duplicate IDs are silently discarded. Count is synced
        from the set after restoration.
        """
        valid_ids = [oid for oid in order_link_ids if oid]
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

    def restore_pending_entries_with_symbols(self, records: list[dict]) -> None:
        """Restore pending entries from detailed records (includes symbol mapping).

        Empty and duplicate IDs are silently discarded. Count is synced
        from the set after restoration.
        """
        seen: set[str] = set()
        for rec in records:
            oid = str(rec.get("order_link_id", "")).strip()
            if not oid or oid in seen:
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
        merged: dict[str, dict] = {}
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
        """Sync open positions from the exchange into the local registry.

        Call once at startup (after seeding candles) so the engine doesn't
        open duplicate positions on restart.
        """
        try:
            positions = await self._adapter.get_positions(self._category)
            previous_symbols = set(self._open_positions)
            exchange_symbols: set[str] = set()
            refreshed_positions: dict[str, dict[str, Any]] = {}
            for pos in positions:
                if pos.size > Decimal("0"):
                    exchange_symbols.add(pos.symbol)
                    refreshed_positions[pos.symbol] = {
                        "side": pos.side,
                        "size": pos.size,
                        "entry_price": pos.entry_price,
                    }
                    notional = pos.size * pos.entry_price
                    await self._exposure.update_position(pos.symbol, pos.side.value, notional)
            closed_symbols = previous_symbols - exchange_symbols
            for symbol in closed_symbols:
                await self._exposure.remove_position(symbol)
                self._last_entry_at.pop(symbol, None)
            self._open_positions = refreshed_positions
            log.info(
                "execution.positions_synced",
                count=len(self._open_positions),
                symbols=list(self._open_positions.keys()),
                closed_symbols=sorted(closed_symbols),
            )
            return positions
        except Exception as exc:
            log.warning("execution.sync_positions_failed", error=str(exc))
            return None

    async def record_position_closed(self, symbol: str) -> None:
        """Call when a position is closed (e.g. TP/SL hit)."""
        self._open_positions.pop(symbol, None)
        self._last_entry_at.pop(symbol, None)
        await self._exposure.remove_position(symbol)

    # ------------------------------------------------------------------
    # Instrument info
    # ------------------------------------------------------------------

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
            await self._adapter._rest.set_leverage(
                category=self._category,
                symbol=symbol,
                buy_leverage=lev_str,
                sell_leverage=lev_str,
            )
            self._leverage_confirmed[symbol] = max_leverage
            log.info("execution.leverage_set", symbol=symbol, leverage=lev_str)
        except Exception as exc:
            log.warning("execution.leverage_set_failed", symbol=symbol, error=str(exc))

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

        # 1. Deduplication ─────────────────────────────────────────────
        if self.has_open_position(symbol):
            log.debug("execution.skipped_open_position", symbol=symbol)
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
                from trader.domain.enums import OrderSide

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

        # 4. Risk evaluation ───────────────────────────────────────────
        # Extract spread and ATR for RiskManager sizing
        spread: Decimal | None = None
        atr: Decimal | None = None
        if regime_context is not None and regime_context.spread_bps is not None:
            spread = Decimal(str(regime_context.spread_bps)) / Decimal("10000")
        if feature_vector is not None:
            # ATR is computed as atr_14_pct in feature pipeline (ATR / price as fraction)
            try:
                idx = feature_vector.feature_names.index("atr_14_pct")
                atr = Decimal(str(feature_vector.values[idx]))
            except (ValueError, IndexError):
                pass
        try:
            decision = await self._risk_manager.evaluate(
                proposal=proposal,
                capital=capital,
                available_balance=available_balance,
                instrument_info=instrument_info,
                feature_vector=feature_vector,
                regime_context=regime_context,
                spread=spread,
                atr=atr,
            )
        except Exception as exc:
            log.error(
                "execution.risk_evaluation_error",
                symbol=symbol,
                error=str(exc),
            )
            return None

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
            return decision

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
            tp_d = proposal.take_profit
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
                return None

        # 5. Build OrderIntent ─────────────────────────────────────────
        assert decision.approved_qty is not None
        intent = self._build_intent(proposal, decision, instrument_info)

        # 6. Execute or shadow ─────────────────────────────────────────
        if self._shadow_mode:
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
                    return None

            # P0.3: Register pending entry slot (with symbol for screener protection)
            self.mark_entry_submitted(intent.order_link_id, symbol=symbol)

            # P0.6: Second canary gate immediately before REST
            if self._is_canary:
                if len(self._open_positions) >= CANARY_MAX_OPEN_POSITIONS:
                    self.mark_entry_resolved(intent.order_link_id)
                    log.warning("canary.blocked_max_positions_pre_rest", symbol=symbol)
                    return None
                if self._exposure.total_exposure_pct >= CANARY_MAX_TOTAL_EXPOSURE_PCT:
                    self.mark_entry_resolved(intent.order_link_id)
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
                    self.mark_entry_resolved(intent.order_link_id)
                    return None

            try:
                resp = await self._adapter.place_order(intent)
                exchange_order_id = resp.get("result", {}).get("orderId", "?")
                self._diag_order_placed += 1
                log.info(
                    "execution.order_placed",
                    symbol=symbol,
                    side=proposal.side.value,
                    qty=str(decision.approved_qty),
                    exchange_order_id=exchange_order_id,
                    order_link_id=intent.order_link_id,
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
                self.mark_entry_resolved(intent.order_link_id)
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

        if not self._shadow_mode:
            await self.sync_positions()
            if not self.has_open_position(symbol):
                log.warning(
                    "execution.order_accepted_position_not_confirmed",
                    symbol=symbol,
                    order_link_id=intent.order_link_id,
                )
            return decision

        entry_price = proposal.entry_price or Decimal("0")
        notional = decision.approved_qty * entry_price
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

        return decision

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
        if is_stop_loss and side == OrderSide.SELL:
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
            "order_failed": self._diag_order_failed,
            "pending_entry_count": len(self._pending_entry_order_link_ids),
            "net_edge_rejected": self._diag_net_edge_rejected,
            "no_tp_rejected": self._diag_no_tp_rejected,
            "fee_unavailable_rejected": self._diag_fee_unavailable_rejected,
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
