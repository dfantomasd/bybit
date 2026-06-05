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

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
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

# Default cooldown between entries on the same symbol
_DEFAULT_COOLDOWN_S = 300  # 5 minutes


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
        category: str = "linear",
    ) -> None:
        self._adapter = adapter
        self._risk_manager = risk_manager
        self._exposure = exposure_tracker
        self._shadow_mode = shadow_mode
        self._cooldown = timedelta(seconds=cooldown_s)
        self._category = category

        # symbol → last entry timestamp
        self._last_entry_at: dict[str, datetime] = {}
        # symbol → open position metadata (size, entry_price, side)
        self._open_positions: dict[str, dict[str, Any]] = {}
        # symbol → InstrumentInfo (fetched once, cached forever)
        self._instrument_cache: dict[str, InstrumentInfo] = {}

    # ------------------------------------------------------------------
    # Position awareness
    # ------------------------------------------------------------------

    def has_open_position(self, symbol: str) -> bool:
        return symbol in self._open_positions

    def open_position_count(self) -> int:
        return len(self._open_positions)

    async def sync_positions(self) -> None:
        """Sync open positions from the exchange into the local registry.

        Call once at startup (after seeding candles) so the engine doesn't
        open duplicate positions on restart.
        """
        try:
            positions = await self._adapter.get_positions(self._category)
            self._open_positions.clear()
            for pos in positions:
                if pos.size > Decimal("0"):
                    self._open_positions[pos.symbol] = {
                        "side": pos.side,
                        "size": pos.size,
                        "entry_price": pos.entry_price,
                    }
                    notional = pos.size * pos.entry_price
                    await self._exposure.update_position(
                        pos.symbol, pos.side.value, notional
                    )
            log.info(
                "execution.positions_synced",
                count=len(self._open_positions),
                symbols=list(self._open_positions.keys()),
            )
        except Exception as exc:
            log.warning("execution.sync_positions_failed", error=str(exc))

    def record_position_closed(self, symbol: str) -> None:
        """Call when a position is closed (e.g. TP/SL hit)."""
        self._open_positions.pop(symbol, None)

    # ------------------------------------------------------------------
    # Instrument info
    # ------------------------------------------------------------------

    async def get_instrument_info(self, symbol: str) -> InstrumentInfo:
        if symbol not in self._instrument_cache:
            info = await self._adapter.get_instrument_info(self._category, symbol)
            self._instrument_cache[symbol] = info
            log.debug(
                "execution.instrument_info_cached",
                symbol=symbol,
                min_qty=str(info.min_order_qty),
                qty_step=str(info.qty_step),
            )
        return self._instrument_cache[symbol]

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
        symbol = proposal.symbol

        # 1. Deduplication ─────────────────────────────────────────────
        if self.has_open_position(symbol):
            log.debug("execution.skipped_open_position", symbol=symbol)
            return None

        # 2. Cooldown ──────────────────────────────────────────────────
        last = self._last_entry_at.get(symbol)
        if last is not None:
            elapsed = datetime.now(tz=UTC) - last
            if elapsed < self._cooldown:
                remaining = int((self._cooldown - elapsed).total_seconds())
                log.debug(
                    "execution.skipped_cooldown",
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

        # 4. Risk evaluation ───────────────────────────────────────────
        try:
            decision = await self._risk_manager.evaluate(
                proposal=proposal,
                capital=capital,
                available_balance=available_balance,
                instrument_info=instrument_info,
                feature_vector=feature_vector,
                regime_context=regime_context,
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

        if not approved:
            return decision

        # 5. Build OrderIntent ─────────────────────────────────────────
        assert decision.approved_qty is not None
        intent = self._build_intent(proposal, decision)

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
        else:
            try:
                resp = await self._adapter.place_order(intent)
                exchange_order_id = resp.get("result", {}).get("orderId", "?")
                log.info(
                    "execution.order_placed",
                    symbol=symbol,
                    side=proposal.side.value,
                    qty=str(decision.approved_qty),
                    exchange_order_id=exchange_order_id,
                    order_link_id=intent.order_link_id,
                )
            except Exception as exc:
                log.error(
                    "execution.order_failed",
                    symbol=symbol,
                    error=str(exc),
                )
                return decision

        # 7. Update local state ────────────────────────────────────────
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
        self._last_entry_at[symbol] = datetime.now(tz=UTC)

        if notional > Decimal("0"):
            await self._exposure.update_position(
                symbol, proposal.side.value, notional
            )

        return decision

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_intent(self, proposal: TradeProposal, decision: RiskDecision) -> OrderIntent:
        # Compact UUID → alphanumeric ID (max 36 chars for Bybit)
        link_id = str(proposal.proposal_id).replace("-", "")[:36]

        assert decision.approved_qty is not None
        return OrderIntent(
            decision_id=decision.decision_id,
            proposal_id=proposal.proposal_id,
            symbol=proposal.symbol,
            market_type=proposal.market_type,
            side=proposal.side,
            order_type=OrderType.MARKET,
            qty=decision.approved_qty,
            price=None,  # Market order — no price needed
            order_link_id=link_id,
            take_profit=proposal.take_profit,
            stop_loss=proposal.stop_loss,
            tp_order_type=OrderType.LIMIT,
            sl_order_type=OrderType.MARKET,
        )

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
            "last_entries": {
                sym: ts.isoformat()
                for sym, ts in self._last_entry_at.items()
            },
        }
