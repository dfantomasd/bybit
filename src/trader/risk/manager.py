"""Central Risk Manager — final authority on all trade proposals.

No strategy or model can bypass this. All invariants are enforced in code.

CRITICAL INVARIANTS:
1. auto_resume_after_hard_stop is always False (no auto-resume from hard stop)
2. LLM risk_multiplier is clamped to [0.0, 1.0] — can NEVER increase base risk
3. Qty is always rounded DOWN — never up
4. No float arithmetic — all financial calculations use Decimal
5. Hard cap is the last check after all multipliers
6. SHORT requires explicit short_allowed=True
7. DERIVATIVES require explicit derivatives_allowed=True
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal
from typing import Any

from trader.domain.enums import (
    MarketRegime,
    MarketType,
    OrderSide,
    RiskDecisionStatus,
    RiskProfile,
)
from trader.domain.models import (
    FeatureVector,
    InstrumentInfo,
    RegimeContext,
    RiskDecision,
    TradeProposal,
)
from trader.risk.circuit_breakers import CircuitBreakerManager
from trader.risk.drawdown import DrawdownTracker
from trader.risk.exposure import ExposureTracker
from trader.risk.kelly_adapter import KellyAdapter, KellyAdapterContext
from trader.risk.kill_switch import KillSwitch
from trader.risk.profiles import RiskLimits, get_risk_limits
from trader.risk.sizing import PositionSizer

logger = logging.getLogger(__name__)


def _ceil_to_step(qty: Decimal, step: Decimal) -> Decimal:
    """Round qty UP to the nearest qty_step (used only for min-notional floor)."""
    if step <= Decimal("0"):
        return qty
    steps = (qty / step).to_integral_value(rounding=ROUND_CEILING)
    return steps * step


def _clamp_unit_decimal(value: Any, default: Decimal = Decimal("1")) -> Decimal:
    """Return value clamped to [0, 1], falling back when conversion fails."""
    if value is None:
        return default
    try:
        raw = Decimal(str(value))
    except Exception:
        return default
    return max(Decimal("0"), min(Decimal("1"), raw))


# Regime-based risk multipliers
_REGIME_MULTIPLIERS: dict[MarketRegime, Decimal] = {
    MarketRegime.BULL_TREND: Decimal("1.0"),
    MarketRegime.BEAR_TREND: Decimal("0.75"),
    MarketRegime.SIDEWAYS: Decimal("0.9"),
    MarketRegime.HIGH_VOLATILITY: Decimal("0.5"),
    MarketRegime.LOW_LIQUIDITY: Decimal("0.5"),
    MarketRegime.EVENT_RISK: Decimal("0.3"),
    MarketRegime.UNCERTAIN: Decimal("0.7"),
}

# Regimes that block new entries entirely
_BLOCKING_REGIMES: set[MarketRegime] = {
    MarketRegime.LOW_LIQUIDITY,
    MarketRegime.EVENT_RISK,
}

# Derivatives market types
_DERIVATIVES_TYPES: set[MarketType] = {MarketType.LINEAR, MarketType.INVERSE}


class RiskManager:
    """Central risk authority. Called for every TradeProposal.

    Decision flow (in order):
    1.  Check kill switch / safe mode
    2.  Check circuit breakers
    3.  Check regime (HIGH_VOLATILITY, LOW_LIQUIDITY, EVENT_RISK -> reduce/reject)
    4.  Check daily loss limit
    5.  Check drawdown
    6.  Check short/derivatives permissions
    7.  Check position count
    8.  Preliminary hard blocker: zero exposure budget remaining (skip if any budget left)
    9.  Check leverage
    10. Validate stop distance
    11. Calculate position size via PositionSizer (may resize qty down to budget)
    12. Apply LLM risk_multiplier (clamped [0,1], can only reduce)
    13. Apply regime risk multiplier
    14. Final hard cap check
    15. Final exposure validation with actual approved_qty
    16. Post-multiplier min-notional guard (with safety buffer)
    17. Return RiskDecision
    """

    # CRITICAL: auto_resume_after_hard_stop is ALWAYS False regardless of config.
    _AUTO_RESUME_AFTER_HARD_STOP: bool = False

    def __init__(
        self,
        risk_profile: RiskProfile,
        drawdown_tracker: DrawdownTracker,
        exposure_tracker: ExposureTracker,
        circuit_breaker_manager: CircuitBreakerManager,
        kill_switch: KillSwitch,
        metrics: Any = None,
        event_bus: Any = None,
        log: logging.Logger | None = None,
        min_notional_safety_buffer_pct: float = 3.0,
        require_liquidity_for_sizing: bool = False,
        max_correlated_positions: int = 0,
        kelly_adapter: KellyAdapter | None = None,
        trade_journal: Any | None = None,
        ml_controller: Any | None = None,
    ) -> None:
        self._profile = risk_profile
        self._limits: RiskLimits = get_risk_limits(risk_profile)
        self._drawdown = drawdown_tracker
        self._exposure = exposure_tracker
        self._breakers = circuit_breaker_manager
        self._kill_switch = kill_switch
        self._metrics = metrics
        self._event_bus = event_bus
        self._log = log or logger
        self._require_liquidity_for_sizing = require_liquidity_for_sizing
        self._min_notional_safety_buffer_pct = Decimal(str(min_notional_safety_buffer_pct))
        self._max_correlated_positions = max(0, int(max_correlated_positions))
        self._kelly_adapter = kelly_adapter or KellyAdapter()
        self._trade_journal = trade_journal
        self._ml_controller = ml_controller

        # Initialize ML prediction applier if controller available
        self._ml_applier = None
        if ml_controller is not None:
            try:
                from trader.ml.prediction_applier import PredictionApplier

                self._ml_applier = PredictionApplier(ml_controller)
            except Exception as e:
                self._log.debug(f"ml_applier_init_failed: {e}")

        self._daily_pnl: Decimal = Decimal("0")
        self._paused: bool = False
        self._last_daily_reset: datetime = datetime.now(tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        self._daily_reset_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Primary evaluation
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        proposal: TradeProposal,
        capital: Decimal,
        available_balance: Decimal,
        instrument_info: InstrumentInfo,
        regime_context: RegimeContext | None = None,
        feature_vector: FeatureVector | None = None,
        spread: Decimal | None = None,
        atr: Decimal | None = None,
        shadow_mode: bool = False,
        min_atr_multiple: Decimal | None = None,
    ) -> RiskDecision:
        """Evaluate a trade proposal and return a RiskDecision."""
        triggered_rules: list[str] = []

        # ----------------------------------------------------------------
        # 1. ML FILTERING — skip low confidence signals early
        # ----------------------------------------------------------------
        if self._ml_applier is not None and self._trade_journal is not None:
            try:
                recent_trades = await self._trade_journal.get_recent_closed_trades(limit=20)
                should_take, reason = await self._ml_applier.should_take_trade(
                    proposal=proposal,
                    recent_trades=recent_trades,
                    current_price=proposal.entry_price or Decimal("1"),
                    market_regime=regime_context.regime.value if regime_context else "SIDEWAYS",
                )
                if not should_take:
                    self._log.info(
                        "execution.filtered_by_ml",
                        symbol=proposal.symbol,
                        reason=reason,
                    )
                    return self._reject(
                        proposal,
                        f"ml_filter: {reason}",
                        ["ml_filtered"],
                        capital,
                    )
            except Exception as e:
                self._log.debug(f"ml_filtering_failed: {e}")

        # ----------------------------------------------------------------
        # 2. Kill switch check
        # ----------------------------------------------------------------
        if self._kill_switch.is_active:
            if not self._kill_switch.new_entries_allowed():
                return self._reject(
                    proposal,
                    f"kill switch active: {self._kill_switch.current_mode}",
                    ["kill_switch"],
                    capital,
                )

        # ----------------------------------------------------------------
        # 3. Circuit breakers — safe mode
        # ----------------------------------------------------------------
        if self._breakers.should_emergency():
            return self._reject(
                proposal,
                "emergency circuit breaker active",
                ["circuit_breaker_emergency"],
                capital,
            )

        if self._breakers.should_block_entries():
            triggered = [s.breaker_type.value for s in self._breakers.get_triggered()]
            return self._reject(
                proposal,
                f"circuit breaker blocking entries: {triggered}",
                ["circuit_breaker_stop_entries"],
                capital,
            )

        if self._paused:
            return self._make_decision(
                proposal=proposal,
                status=RiskDecisionStatus.PAUSED,
                reason="risk manager paused",
                triggered_rules=["paused"],
                approved_qty=None,
                capital=capital,
            )

        # ----------------------------------------------------------------
        # 4. Regime check
        # ----------------------------------------------------------------
        regime = proposal.regime
        if regime_context is not None:
            regime = regime_context.regime
            if not regime_context.trading_allowed:
                if not (shadow_mode and regime == MarketRegime.LOW_LIQUIDITY):
                    return self._reject(
                        proposal,
                        f"regime {regime.value} blocked trading: {regime_context.block_reason}",
                        ["regime_block"],
                        capital,
                    )

        if regime in _BLOCKING_REGIMES:
            if shadow_mode and regime == MarketRegime.LOW_LIQUIDITY:
                pass
            else:
                return self._reject(
                    proposal,
                    f"regime {regime.value} blocks new entries",
                    ["regime_block"],
                    capital,
                )

        # ----------------------------------------------------------------
        # 4. Daily loss limit
        # ----------------------------------------------------------------
        if capital > Decimal("0"):
            daily_loss_pct = (
                abs(self._daily_pnl) / capital * Decimal("100") if self._daily_pnl < Decimal("0") else Decimal("0")
            )
            if daily_loss_pct >= self._limits.daily_loss_limit_pct:
                return self._reject(
                    proposal,
                    f"daily loss {daily_loss_pct:.2f}% >= limit {self._limits.daily_loss_limit_pct}%",
                    ["daily_loss_limit"],
                    capital,
                )

        # ----------------------------------------------------------------
        # 5. Drawdown hard stop
        # ----------------------------------------------------------------
        drawdown_pct = self._drawdown.drawdown_pct
        if self._drawdown.is_at_hard_stop(self._limits.hard_stop_drawdown_pct):
            # CRITICAL: auto_resume_after_hard_stop is ALWAYS False
            # Never auto-resume regardless of profile config
            return self._reject(
                proposal,
                f"drawdown {drawdown_pct:.2f}% >= hard stop {self._limits.hard_stop_drawdown_pct}%",
                ["drawdown_hard_stop"],
                capital,
            )

        # ----------------------------------------------------------------
        # 6. Short / derivatives permissions
        # ----------------------------------------------------------------
        if proposal.side == OrderSide.SELL and not self._limits.short_allowed:
            return self._reject(
                proposal,
                f"short positions not allowed in {self._profile.value} profile",
                ["short_not_allowed"],
                capital,
            )

        if proposal.market_type in _DERIVATIVES_TYPES and not self._limits.derivatives_allowed:
            return self._reject(
                proposal,
                f"derivatives ({proposal.market_type.value}) not allowed in {self._profile.value} profile",
                ["derivatives_not_allowed"],
                capital,
            )

        if proposal.market_type not in self._limits.allowed_market_types:
            return self._reject(
                proposal,
                f"market type {proposal.market_type.value} not in allowed types for {self._profile.value}",
                ["market_type_not_allowed"],
                capital,
            )

        # ----------------------------------------------------------------
        # 7. Position count
        # ----------------------------------------------------------------
        if self._exposure.position_count >= self._limits.max_simultaneous_positions:
            return self._reject(
                proposal,
                f"max positions ({self._limits.max_simultaneous_positions}) reached",
                ["max_positions"],
                capital,
            )

        # ----------------------------------------------------------------
        # 7b. Correlated family cap (e.g. multiple BTC-beta alts)
        # ----------------------------------------------------------------
        if self._max_correlated_positions > 0:
            same_family = self._exposure.count_same_family_positions(proposal.symbol)
            if same_family >= self._max_correlated_positions:
                return self._reject(
                    proposal,
                    f"correlated family positions {same_family} >= limit {self._max_correlated_positions}",
                    ["max_correlated_positions"],
                    capital,
                )

        # ----------------------------------------------------------------
        # 8. Preliminary hard blocker: zero budget remaining
        #
        # We only reject here if there is literally NO room left in the
        # portfolio. If there is any remaining budget, PositionSizer (step 11)
        # will size down the qty to fit — we must NOT reject the full
        # requested_qty just because it exceeds the remaining budget.
        # ----------------------------------------------------------------
        remaining_exposure_pct = self._limits.max_total_exposure_pct - self._exposure.total_exposure_pct
        if remaining_exposure_pct <= Decimal("0"):
            return self._reject(
                proposal,
                f"portfolio exposure cap fully reached "
                f"({self._exposure.total_exposure_pct:.2f}% >= {self._limits.max_total_exposure_pct}%)",
                ["exposure_cap_full"],
                capital,
            )

        # ----------------------------------------------------------------
        # 9. Leverage check (for derivatives)
        # ----------------------------------------------------------------
        if instrument_info.max_leverage is not None:
            if instrument_info.max_leverage > self._limits.max_leverage:
                triggered_rules.append("leverage_reduced")

        # ----------------------------------------------------------------
        # 10. Validate stop distance
        # ----------------------------------------------------------------
        stop_distance_pct = Decimal("0")
        if proposal.stop_loss is not None and proposal.entry_price is not None and proposal.entry_price > Decimal("0"):
            stop_distance_pct = abs(proposal.entry_price - proposal.stop_loss) / proposal.entry_price
        elif proposal.entry_price is not None and proposal.entry_price > Decimal("0"):
            # Default stop distance of 2% if no explicit SL
            stop_distance_pct = Decimal("0.02")
        else:
            stop_distance_pct = Decimal("0.02")

        if stop_distance_pct <= Decimal("0"):
            return self._reject(
                proposal,
                "stop distance is zero",
                ["invalid_stop"],
                capital,
            )

        # ----------------------------------------------------------------
        # 11. Calculate position size
        # ----------------------------------------------------------------
        # Use ML-based Kelly sizing via adapter; fallback to profile limits
        # Fetch recent trades for context enrichment
        recent_trades = []
        recent_returns_bps = []
        if self._trade_journal is not None:
            try:
                recent_trades = await self._trade_journal.get_recent_closed_trades(limit=20)
                # Extract returns from trades
                recent_returns_bps = [
                    float(trade.get("net_bps", 0)) for trade in recent_trades if trade.get("net_bps") is not None
                ]
            except Exception as e:
                self._log.debug(f"kelly_sizing.trade_history_fetch_failed: {e}")

        kelly_fraction, fractional_kelly, kelly_reasoning = await self._kelly_adapter.predict_kelly_sizing(
            context=KellyAdapterContext(
                recent_trades=recent_trades,
                current_price=proposal.entry_price or Decimal("1"),
                recent_returns_bps=recent_returns_bps,
                all_returns_bps=recent_returns_bps,  # Use recent as all-time for now
                volatility_regime=self._get_volatility_regime_code(regime_context.regime) if regime_context else 0,
                current_drawdown_pct=float(drawdown_pct),
                max_drawdown_pct=float(drawdown_pct),
                strategy_id=getattr(proposal, "strategy_id", "unknown"),
                symbol=proposal.symbol,
                total_trades=self._exposure.position_count,
            )
        )

        # ================================================================
        # ML POSITION SIZING OPTIMIZATION
        # ================================================================
        position_size_adjustment = Decimal("1.0")
        if self._ml_applier is not None and self._trade_journal is not None:
            try:
                recent_trades = await self._trade_journal.get_recent_closed_trades(limit=20)
                optimized = await self._ml_applier.optimize_entry(
                    proposal=proposal,
                    recent_trades=recent_trades,
                    current_price=proposal.entry_price or Decimal("1"),
                    market_regime=regime_context.regime.value if regime_context else "SIDEWAYS",
                )
                position_size_adjustment = Decimal(str(optimized.position_size_adjustment))
                self._log.debug(
                    "position_size.ml_optimized",
                    adjustment=float(position_size_adjustment),
                    confidence=optimized.entry_confidence,
                )
            except Exception as e:
                self._log.debug(f"position_size_optimization_failed: {e}")

        # Convert Kelly fraction (0.01-0.25) to risk percentage (clamped to profile range)
        desired_risk_pct = min(
            kelly_fraction * Decimal("100") * position_size_adjustment,
            self._limits.risk_per_trade_max_pct,
        )

        data_quality_score = 1.0
        event_risk_score = 0.0

        realized_vol: Decimal | None = None
        if feature_vector is not None:
            data_quality_score = feature_vector.quality_score
            try:
                idx = feature_vector.feature_names.index("realized_vol_20")
                realized_vol = Decimal(str(feature_vector.values[idx]))
            except (ValueError, IndexError):
                pass

        if regime_context is not None:
            if regime_context.regime == MarketRegime.HIGH_VOLATILITY:
                event_risk_score = 0.5
            elif regime_context.regime == MarketRegime.EVENT_RISK:
                event_risk_score = 0.8

        remaining_position_budget_usd = self._exposure.remaining_position_exposure_usd(proposal.symbol)
        sizer = PositionSizer(
            self._limits,
            instrument_info,
            require_liquidity=self._require_liquidity_for_sizing,
        )
        approved_qty, rejection_reason = sizer.calculate(
            capital=capital,
            stop_distance_pct=stop_distance_pct,
            desired_risk_pct=desired_risk_pct,
            current_exposure_pct=self._exposure.total_exposure_pct,
            drawdown_pct=drawdown_pct,
            event_risk_score=event_risk_score,
            data_quality_score=data_quality_score,
            spread=spread,
            atr=atr,
            available_balance=available_balance,
            entry_price=proposal.entry_price,
            remaining_position_budget_usd=remaining_position_budget_usd,
            realized_vol=realized_vol,
            min_atr_multiple=min_atr_multiple,
        )

        if approved_qty <= Decimal("0"):
            return self._reject(
                proposal,
                rejection_reason or "position sizer returned zero",
                ["sizer_rejected"],
                capital,
            )

        # ----------------------------------------------------------------
        # 13. Apply signal confidence and optional LLM risk_multiplier.
        # CRITICAL: both multipliers are clamped to [0.0, 1.0] and can only reduce.
        # ----------------------------------------------------------------
        confidence_multiplier = _clamp_unit_decimal(proposal.confidence)
        llm_multiplier = _clamp_unit_decimal(proposal.expected_risk)
        approved_qty = approved_qty * confidence_multiplier * llm_multiplier

        # ----------------------------------------------------------------
        # 14. Apply regime risk multiplier
        # ----------------------------------------------------------------
        regime_mult = _REGIME_MULTIPLIERS.get(regime, Decimal("1.0"))
        approved_qty = approved_qty * regime_mult

        # ----------------------------------------------------------------
        # 15. Final hard cap check
        # ----------------------------------------------------------------
        if proposal.entry_price is not None and proposal.entry_price > Decimal("0"):
            hard_cap_risk = capital * self._limits.risk_per_trade_hard_cap_pct / Decimal("100")
            max_qty_hard_cap = hard_cap_risk / (stop_distance_pct * proposal.entry_price)
            approved_qty = min(approved_qty, max_qty_hard_cap)

        # Re-round after multipliers (always ROUND_DOWN)
        approved_qty = sizer.round_to_step(approved_qty, instrument_info.qty_step)

        # ----------------------------------------------------------------
        # 15. Final exposure validation with actual approved_qty
        #
        # Now that sizing has determined the actual qty, do a final check
        # that adding this position doesn't breach per-position or total cap.
        # ----------------------------------------------------------------
        if proposal.entry_price is not None and proposal.entry_price > Decimal("0") and approved_qty > Decimal("0"):
            final_notional = approved_qty * proposal.entry_price
            can_add, exposure_reason = self._exposure.can_add_position(proposal.symbol, final_notional)
            if not can_add:
                return self._reject(proposal, exposure_reason, ["exposure_cap"], capital)

        # ----------------------------------------------------------------
        # 16. Post-multiplier min-notional guard (with safety buffer)
        #
        # After confidence + regime multipliers reduce qty, the resulting
        # notional may fall below Bybit's minimum (e.g. $5).  We apply a
        # configurable safety buffer (default +3%) to keep the order above
        # the exchange threshold.  We attempt to bump qty — but ONLY if
        # every risk constraint is still satisfied.
        # ----------------------------------------------------------------
        if (
            instrument_info.min_notional is not None
            and proposal.entry_price is not None
            and proposal.entry_price > Decimal("0")
            and approved_qty > Decimal("0")
        ):
            # Apply safety buffer: e.g. $5 * 1.03 = $5.15
            required_notional = instrument_info.min_notional * (
                Decimal("1") + self._min_notional_safety_buffer_pct / Decimal("100")
            )
            final_notional = approved_qty * proposal.entry_price
            if final_notional < required_notional:
                min_qty = _ceil_to_step(
                    required_notional / proposal.entry_price,
                    instrument_info.qty_step,
                )
                min_notional_value = min_qty * proposal.entry_price
                # Remaining portfolio exposure budget in USD
                remaining_exposure_usd = (
                    capital
                    * max(
                        Decimal("0"),
                        self._limits.max_total_exposure_pct - self._exposure.total_exposure_pct,
                    )
                    / Decimal("100")
                )
                # Hard-cap risk at the bumped qty
                hard_cap_usd = capital * self._limits.risk_per_trade_hard_cap_pct / Decimal("100")
                bumped_risk_usd = min_qty * proposal.entry_price * stop_distance_pct

                # NOTE: removed min_qty <= proposal.requested_qty — the bump
                # may exceed requested_qty when multipliers reduced qty below
                # min-notional and the original request was correctly sized.
                can_bump = (
                    min_qty <= instrument_info.max_order_qty
                    and min_qty >= instrument_info.min_order_qty
                    and min_notional_value <= available_balance
                    and min_notional_value <= remaining_exposure_usd
                    and bumped_risk_usd <= hard_cap_usd
                )
                if can_bump:
                    approved_qty = min_qty
                    # Re-validate: bump must not violate per-position exposure cap
                    bumped_notional = approved_qty * proposal.entry_price
                    remaining_pos_budget = self._exposure.remaining_position_exposure_usd(proposal.symbol)
                    if bumped_notional > remaining_pos_budget:
                        return self._reject(
                            proposal,
                            f"bumped notional {bumped_notional:.4f} exceeds remaining per-position budget {remaining_pos_budget:.4f}",
                            ["exposure_cap_post_bump"],
                            capital,
                        )
                    can_add_bumped, bump_exp_reason = self._exposure.can_add_position(proposal.symbol, bumped_notional)
                    if not can_add_bumped:
                        return self._reject(proposal, bump_exp_reason, ["exposure_cap_post_bump"], capital)
                    triggered_rules.append("min_notional_buffer_applied")
                    self._log.info(
                        "risk.min_notional_buffer_applied symbol=%s bumped_to_qty=%s"
                        " required_notional=%s buffer_pct=%s",
                        proposal.symbol,
                        str(approved_qty),
                        str(required_notional),
                        str(self._min_notional_safety_buffer_pct),
                    )
                else:
                    self._log.info(
                        "risk.min_notional_buffer_rejected symbol=%s final_notional=%s required_notional=%s",
                        proposal.symbol,
                        str(final_notional),
                        str(required_notional),
                    )
                    return self._reject(
                        proposal,
                        (
                            f"post-multiplier notional {final_notional:.4f} < "
                            f"required {required_notional:.4f} (min_notional "
                            f"{instrument_info.min_notional} + {self._min_notional_safety_buffer_pct}% buffer); "
                            "cannot raise without violating risk limits"
                        ),
                        ["post_multiplier_min_notional_rejected"],
                        capital,
                    )

        if approved_qty <= Decimal("0") or approved_qty < instrument_info.min_order_qty:
            return self._reject(
                proposal,
                "approved qty reduced to zero after multipliers",
                ["post_multiplier_zero"],
                capital,
            )

        if proposal.entry_price is not None and proposal.entry_price > Decimal("0"):
            reserved_notional = approved_qty * proposal.entry_price
            reserved, reserve_reason = self._exposure.can_add_position(
                proposal.symbol,
                reserved_notional,
                order_id=str(proposal.proposal_id),
            )
            if not reserved:
                return self._reject(
                    proposal,
                    reserve_reason,
                    ["exposure_reservation"],
                    capital,
                )

        # ----------------------------------------------------------------
        # 17. Determine status: APPROVED or RESIZED
        # ----------------------------------------------------------------
        if approved_qty < proposal.requested_qty:
            status = RiskDecisionStatus.RESIZED
            triggered_rules.append("resized")
        else:
            status = RiskDecisionStatus.APPROVED

        return RiskDecision(
            proposal_id=proposal.proposal_id,
            status=status,
            approved_qty=approved_qty,
            original_qty=proposal.requested_qty if status == RiskDecisionStatus.RESIZED else None,
            reason=", ".join(triggered_rules) if triggered_rules else "",
            triggered_rules=triggered_rules,
            portfolio_heat=float(self._exposure.total_exposure_pct),
            current_drawdown_pct=float(drawdown_pct),
            open_positions_count=self._exposure.position_count,
        )

    # ------------------------------------------------------------------
    # Daily PnL tracking
    # ------------------------------------------------------------------

    async def update_daily_pnl(self, realized_pnl: Decimal) -> None:
        """Accumulate realized PnL for daily loss limit checks."""
        self._daily_pnl += realized_pnl

    async def reset_daily_stats(self) -> None:
        """Reset daily stats at UTC midnight."""
        self._daily_pnl = Decimal("0")
        self._last_daily_reset = datetime.now(tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        self._log.info("Daily risk stats reset")

    def start_daily_reset_scheduler(self) -> None:
        """Start a background task that resets daily stats at UTC midnight."""
        if self._daily_reset_task is not None and not self._daily_reset_task.done():
            return
        self._daily_reset_task = asyncio.create_task(
            self._daily_reset_loop(), name="risk-manager-daily-reset"
        )

    def stop_daily_reset_scheduler(self) -> None:
        if self._daily_reset_task is not None:
            self._daily_reset_task.cancel()
            self._daily_reset_task = None

    async def _daily_reset_loop(self) -> None:
        while True:
            now = datetime.now(tz=UTC)
            next_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if next_midnight <= now:
                next_midnight = next_midnight.replace(day=next_midnight.day + 1)
            wait_seconds = (next_midnight - now).total_seconds()
            await asyncio.sleep(wait_seconds)
            await self.reset_daily_stats()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def daily_pnl(self) -> Decimal:
        return self._daily_pnl

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_safe_mode(self) -> bool:
        return self._breakers.should_safe_mode()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        return {
            "profile": self._profile.value,
            "paused": self._paused,
            "safe_mode": self.is_safe_mode,
            "daily_pnl": str(self._daily_pnl),
            "drawdown_pct": str(self._drawdown.drawdown_pct),
            "position_count": self._exposure.position_count,
            "total_exposure_pct": str(self._exposure.total_exposure_pct),
            "kill_switch": self._kill_switch.to_dict(),
            "circuit_breakers": self._breakers.to_dict(),
            # CRITICAL INVARIANT: always report False
            "auto_resume_after_hard_stop": self._AUTO_RESUME_AFTER_HARD_STOP,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reject(
        self,
        proposal: TradeProposal,
        reason: str,
        triggered_rules: list[str],
        capital: Decimal,
    ) -> RiskDecision:
        return self._make_decision(
            proposal=proposal,
            status=RiskDecisionStatus.REJECTED,
            reason=reason,
            triggered_rules=triggered_rules,
            approved_qty=None,
            capital=capital,
        )

    @staticmethod
    def _get_volatility_regime_code(regime: Any) -> int:
        """Преобразовать MarketRegime в числовой код (0-3)."""
        regime_str = str(regime.value) if hasattr(regime, "value") else str(regime)

        if "HIGH_VOLATILITY" in regime_str or "BULL_TREND" in regime_str or "BEAR_TREND" in regime_str:
            return 2  # high
        elif "SIDEWAYS" in regime_str or "LOW_LIQUIDITY" in regime_str:
            return 1  # medium
        else:  # UNCERTAIN, EVENT_RISK
            return 1  # medium (conservative default)

    def _make_decision(
        self,
        proposal: TradeProposal,
        status: RiskDecisionStatus,
        reason: str,
        triggered_rules: list[str],
        approved_qty: Decimal | None,
        capital: Decimal,
    ) -> RiskDecision:
        return RiskDecision(
            proposal_id=proposal.proposal_id,
            status=status,
            approved_qty=approved_qty,
            reason=reason,
            triggered_rules=triggered_rules,
            portfolio_heat=float(self._exposure.total_exposure_pct),
            current_drawdown_pct=float(self._drawdown.drawdown_pct),
            open_positions_count=self._exposure.position_count,
        )
