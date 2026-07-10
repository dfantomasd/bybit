"""SHADOW-only probe strategy for paper order discovery.

This strategy intentionally runs only in SHADOW. Its job is not to be a live
alpha source, but to create enough conditional paper entries for model-gate
and TP/SL outcome analysis when production strategies are too selective.
"""

from __future__ import annotations

import uuid
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal

from trader.domain.enums import MarketRegime, MarketType, OrderSide
from trader.domain.models import FeatureVector, InstrumentInfo, TradeProposal
from trader.risk.net_edge import NetEdgeParams, net_edge_from_tp_distance, passes_min_net_edge
from trader.strategies.base import BaseStrategy

SHADOW_PROBE_STRATEGY_ID = "shadow_probe_hv_v2"
_PRICE_DECIMALS = Decimal("0.00000001")
# Worst-case qty shrink from confidence/regime/VWAP penalties after risk sizing.
_PROBE_WORST_CASE_QTY_MULTIPLIER = Decimal("0.25")

# Probe entry logic: VWAP pullback in the direction of a real EMA stack, with
# momentum and book confirmation. These are intentionally close to scalp_micro
# but slightly looser so SHADOW can still discover enough outcomes.
_EWMA_MIN = 0.002
_VWAP_BUY_LOW = -0.55
_VWAP_BUY_HIGH = 0.10
_VWAP_SELL_LOW = -0.10
_VWAP_SELL_HIGH = 0.55
_RSI_BUY_MIN = 0.35
_RSI_BUY_MAX = 0.64
_RSI_SELL_MIN = 0.36
_RSI_SELL_MAX = 0.65
_ADX_MIN = 0.18
_VOLUME_ZSCORE_MIN = -0.4


def _price(value: float) -> Decimal:
    return Decimal(str(value)).quantize(_PRICE_DECIMALS)


def is_shadow_probe_strategy(strategy_id: str) -> bool:
    """Return True for versioned SHADOW probe strategy ids."""

    return str(strategy_id).startswith("shadow_probe_")


def _round_qty_down(qty: Decimal, step: Decimal) -> Decimal:
    if step <= Decimal("0"):
        return qty
    steps = (qty / step).to_integral_value(rounding=ROUND_DOWN)
    return steps * step


def probe_notional_viable(
    *,
    price: float,
    notional_usd: float,
    info: InstrumentInfo,
    min_notional_buffer_pct: float,
    worst_case_qty_multiplier: Decimal = _PROBE_WORST_CASE_QTY_MULTIPLIER,
) -> bool:
    """Return True when probe sizing can survive post-risk min_notional checks."""
    if price <= 0 or notional_usd <= 0:
        return False
    entry_price = Decimal(str(price))
    raw_qty = Decimal(str(notional_usd)) / entry_price
    qty = _round_qty_down(raw_qty, info.qty_step)
    if qty <= Decimal("0") or qty < info.min_order_qty:
        return False
    if info.min_notional is None or info.min_notional <= Decimal("0"):
        return True
    required_notional = info.min_notional * (Decimal("1") + Decimal(str(min_notional_buffer_pct)) / Decimal("100"))
    adjusted_notional = qty * entry_price * worst_case_qty_multiplier
    return adjusted_notional >= required_notional


class ShadowProbeStrategy(BaseStrategy):
    """Generate safe paper-only probes from broad microstructure signals."""

    def __init__(
        self,
        *,
        imbalance_provider: Callable[[str], float | None] | None = None,
        instrument_info_provider: Callable[[str], InstrumentInfo | None] | None = None,
        side_blocked: Callable[[str, str], bool] | None = None,
        symbol_allowed: Callable[[str], bool] | None = None,
        regime_allows: Callable[[FeatureVector], bool] | None = None,
        open_positions_count: Callable[[], int] | None = None,
        max_open_positions: int = 2,
        burst_max_signals: int = 3,
        burst_window_seconds: int = 300,
        burst_cooldown_seconds: int = 600,
        min_abs_imbalance: float = 0.05,
        min_quality: float = 0.45,
        cooldown_seconds: int = 300,
        max_notional_usd: float = 8.0,
        risk_pct: float = 0.003,
        tp_atr_mult: float = 1.4,
        sl_atr_mult: float = 0.8,
        min_tp_pct: float = 0.45,
        max_tp_pct: float = 1.50,
        min_sl_pct: float = 0.25,
        min_net_return_pct: float = 0.05,
        min_net_reward_risk: float = 1.10,
        min_notional_buffer_pct: float = 3.0,
        cost_params: NetEdgeParams | None = None,
        sell_enabled: bool = False,
        diag_hook: Callable[[str], None] | None = None,
    ) -> None:
        self._imbalance_provider = imbalance_provider
        self._instrument_info_provider = instrument_info_provider
        self._side_blocked = side_blocked
        self._symbol_allowed = symbol_allowed
        self._regime_allows = regime_allows
        self._open_positions_count = open_positions_count
        self._max_open_positions = max(1, int(max_open_positions))
        self._burst_max_signals = max(1, int(burst_max_signals))
        self._burst_window = timedelta(seconds=max(1, int(burst_window_seconds)))
        self._burst_cooldown = timedelta(seconds=max(1, int(burst_cooldown_seconds)))
        self._min_abs_imbalance = max(0.0, float(min_abs_imbalance))
        self._min_quality = max(0.0, min(1.0, float(min_quality)))
        self._cooldown = timedelta(seconds=max(30, int(cooldown_seconds)))
        self._max_notional_usd = max(5.0, float(max_notional_usd))
        self._risk_pct = max(0.0001, float(risk_pct))
        self._tp_atr_mult = max(0.2, float(tp_atr_mult))
        self._sl_atr_mult = max(0.2, float(sl_atr_mult))
        self._min_tp_pct = max(0.05, float(min_tp_pct))
        self._max_tp_pct = max(self._min_tp_pct, float(max_tp_pct))
        self._min_sl_pct = max(0.05, float(min_sl_pct))
        self._min_net_return_pct = max(0.0, float(min_net_return_pct))
        self._min_net_reward_risk = max(0.0, float(min_net_reward_risk))
        self._min_notional_buffer_pct = max(0.0, float(min_notional_buffer_pct))
        self._cost_params = cost_params
        self._sell_enabled = bool(sell_enabled)
        self._diag_hook = diag_hook
        self._last_signal_at: dict[str, datetime] = {}
        self._signal_times: deque[datetime] = deque()
        self._burst_blocked_until: datetime | None = None

    @property
    def strategy_id(self) -> str:
        return SHADOW_PROBE_STRATEGY_ID

    def evict_symbol(self, symbol: str) -> None:
        self._last_signal_at.pop(symbol, None)

    def _diag(self, reason: str, *, symbol: str | None = None, side: OrderSide | str | None = None) -> None:
        if self._diag_hook is None:
            return
        try:
            self._diag_hook(reason)
            if symbol:
                parts = [reason, symbol]
                if side is not None:
                    parts.append(side.value if isinstance(side, OrderSide) else str(side))
                self._diag_hook(":".join(parts))
        except Exception:
            return

    def _cooldown_active(self, symbol: str) -> bool:
        last = self._last_signal_at.get(symbol)
        return last is not None and datetime.now(tz=UTC) - last < self._cooldown

    def _burst_limited(self, now: datetime) -> bool:
        if self._burst_blocked_until is not None:
            if now < self._burst_blocked_until:
                return True
            self._burst_blocked_until = None
        cutoff = now - self._burst_window
        while self._signal_times and self._signal_times[0] < cutoff:
            self._signal_times.popleft()
        if len(self._signal_times) >= self._burst_max_signals:
            self._burst_blocked_until = now + self._burst_cooldown
            return True
        return False

    @staticmethod
    def _features(vec: FeatureVector) -> dict[str, float]:
        return dict(zip(vec.feature_names, vec.values, strict=True))

    @staticmethod
    def _ema_side(features: dict[str, float]) -> OrderSide | None:
        ewma = features.get("ewma_tier_signal")
        rsi = features.get("rsi_14")
        if ewma is None or rsi is None:
            return None
        if ewma > _EWMA_MIN and _RSI_BUY_MIN < rsi < _RSI_BUY_MAX:
            return OrderSide.BUY
        if ewma < -_EWMA_MIN and _RSI_SELL_MIN < rsi < _RSI_SELL_MAX:
            return OrderSide.SELL
        return None

    def _side_from_features(self, vec: FeatureVector, features: dict[str, float]) -> tuple[OrderSide | None, str]:
        ema_side = self._ema_side(features)
        if ema_side is None:
            return None, "entry setup unavailable"

        vwap_dist = features.get("vwap_distance_pct")
        if vwap_dist is None:
            return None, "VWAP distance unavailable"
        if ema_side == OrderSide.BUY and not (_VWAP_BUY_LOW < vwap_dist < _VWAP_BUY_HIGH):
            return None, f"VWAP pullback rejected vwap={vwap_dist:+.3f}"
        if ema_side == OrderSide.SELL and not (_VWAP_SELL_LOW < vwap_dist < _VWAP_SELL_HIGH):
            return None, f"VWAP pullback rejected vwap={vwap_dist:+.3f}"

        adx = features.get("adx_14")
        if adx is None or adx < _ADX_MIN:
            return None, "ADX below trend threshold"

        macd_hist = features.get("macd_hist")
        if macd_hist is not None:
            if ema_side == OrderSide.BUY and macd_hist <= 0:
                return None, "MACD momentum rejected"
            if ema_side == OrderSide.SELL and macd_hist >= 0:
                return None, "MACD momentum rejected"

        volume_zscore = features.get("volume_zscore")
        if volume_zscore is not None and volume_zscore < _VOLUME_ZSCORE_MIN:
            return None, "volume below threshold"

        imbalance = None
        if self._imbalance_provider is not None:
            try:
                imbalance = self._imbalance_provider(vec.symbol)
            except Exception:
                imbalance = None
        if imbalance is None and features.get("ob_data_present", 0.0) >= 1.0:
            imbalance = features.get("ob_imbalance_l5")
        if imbalance is None:
            return None, "orderbook imbalance unavailable"
        if abs(imbalance) < self._min_abs_imbalance:
            return None, f"orderbook imbalance below threshold {imbalance:+.3f}"
        side = OrderSide.BUY if imbalance > 0 else OrderSide.SELL
        if side != ema_side:
            return None, f"book/EMA conflict imbalance={imbalance:+.3f}"
        return side, f"VWAP pullback + EMA/OB confirmation imbalance={imbalance:+.3f}"

    def evaluate(
        self,
        feature_vector: FeatureVector,
        current_price: float,
        available_balance_usd: float,
    ) -> TradeProposal | None:
        if current_price <= 0 or available_balance_usd <= 0:
            return None
        now = datetime.now(tz=UTC)
        if self._open_positions_count is not None and self._open_positions_count() >= self._max_open_positions:
            self._diag("shadow_probe_open_positions_limit", symbol=feature_vector.symbol)
            return None
        if self._burst_limited(now):
            self._diag("shadow_probe_burst_limited", symbol=feature_vector.symbol)
            return None
        if self._symbol_allowed is not None and not self._symbol_allowed(feature_vector.symbol):
            self._diag("shadow_probe_symbol_not_allowed", symbol=feature_vector.symbol)
            return None
        if self._regime_allows is not None and not self._regime_allows(feature_vector):
            self._diag("shadow_probe_regime_blocked", symbol=feature_vector.symbol)
            return None
        if feature_vector.quality_score < self._min_quality:
            self._diag("shadow_probe_low_quality", symbol=feature_vector.symbol)
            return None
        if self._cooldown_active(feature_vector.symbol):
            self._diag("shadow_probe_cooldown", symbol=feature_vector.symbol)
            return None

        features = self._features(feature_vector)
        atr_pct = features.get("atr_14_pct")
        if atr_pct is None or atr_pct <= 0:
            self._diag("shadow_probe_atr_missing", symbol=feature_vector.symbol)
            return None
        # Keep probes meaningful: avoid dead/noisy extremes but stay much wider
        # than live strategies so SHADOW accumulates paper outcomes.
        if atr_pct < 0.00025 or atr_pct > 0.04:
            self._diag("shadow_probe_atr_out_of_range", symbol=feature_vector.symbol)
            return None

        side, reason = self._side_from_features(feature_vector, features)
        if side is None:
            if "setup" in reason or "VWAP" in reason or "ADX" in reason or "MACD" in reason or "volume" in reason:
                self._diag("shadow_probe_entry_filter", symbol=feature_vector.symbol)
            elif "unavailable" in reason:
                self._diag("shadow_probe_imbalance_missing", symbol=feature_vector.symbol)
            elif "below threshold" in reason:
                self._diag("shadow_probe_imbalance_weak", symbol=feature_vector.symbol)
            elif "book/EMA conflict" in reason:
                self._diag("shadow_probe_book_ema_conflict", symbol=feature_vector.symbol)
            else:
                self._diag("shadow_probe_side_unresolved", symbol=feature_vector.symbol)
            return None
        if side == OrderSide.SELL and not self._sell_enabled:
            self._diag("shadow_probe_sell_disabled", symbol=feature_vector.symbol, side=side)
            return None
        if self._side_blocked is not None and self._side_blocked(feature_vector.symbol, side.value):
            self._diag("shadow_probe_side_blocked", symbol=feature_vector.symbol, side=side)
            return None

        sl_dist = max(float(atr_pct) * self._sl_atr_mult, self._min_sl_pct / 100.0)
        tp_dist = max(float(atr_pct) * self._tp_atr_mult, self._min_tp_pct / 100.0, sl_dist * 1.5)
        net_reward_risk = None
        if self._cost_params is not None and self._min_net_reward_risk > 0:
            gross_sl_pct = sl_dist * 100.0
            net_tp_pct = net_edge_from_tp_distance(tp_dist, self._cost_params)
            round_trip_cost_pct = max(0.0, tp_dist * 100.0 - net_tp_pct)
            net_loss_pct = gross_sl_pct + round_trip_cost_pct
            required_tp_dist = (self._min_net_reward_risk * net_loss_pct + round_trip_cost_pct) / 100.0
            if required_tp_dist > tp_dist:
                tp_dist = required_tp_dist
            if tp_dist > self._max_tp_pct / 100.0:
                self._diag("shadow_probe_net_rr_rejected", symbol=feature_vector.symbol, side=side)
                return None
            net_tp_pct = net_edge_from_tp_distance(tp_dist, self._cost_params)
            if net_loss_pct > 0:
                net_reward_risk = net_tp_pct / net_loss_pct
        if self._cost_params is not None and not passes_min_net_edge(
            tp_dist,
            self._cost_params,
            self._min_net_return_pct,
        ):
            self._diag("shadow_probe_net_edge_rejected", symbol=feature_vector.symbol, side=side)
            return None

        notional = min(self._max_notional_usd, max(5.0, available_balance_usd * 0.25))
        if self._instrument_info_provider is not None:
            info = self._instrument_info_provider(feature_vector.symbol)
            if info is not None and not probe_notional_viable(
                price=current_price,
                notional_usd=notional,
                info=info,
                min_notional_buffer_pct=self._min_notional_buffer_pct,
            ):
                self._diag("shadow_probe_min_notional_rejected", symbol=feature_vector.symbol, side=side)
                return None

        qty = notional / current_price
        if qty <= 0:
            self._diag("shadow_probe_qty_invalid", symbol=feature_vector.symbol, side=side)
            return None

        if side == OrderSide.BUY:
            take_profit = current_price * (1 + tp_dist)
            stop_loss = current_price * (1 - sl_dist)
            regime = MarketRegime.BULL_TREND
        else:
            take_profit = current_price * (1 - tp_dist)
            stop_loss = current_price * (1 + sl_dist)
            regime = MarketRegime.BEAR_TREND

        self._last_signal_at[feature_vector.symbol] = now
        self._signal_times.append(now)
        confidence = min(0.62, 0.50 + min(0.10, abs(float(atr_pct)) * 20.0))
        return TradeProposal(
            proposal_id=uuid.uuid4(),
            strategy_id=SHADOW_PROBE_STRATEGY_ID,
            symbol=feature_vector.symbol,
            market_type=MarketType.LINEAR,
            side=side,
            requested_qty=Decimal(str(round(qty, 6))),
            requested_notional_usd=Decimal(str(round(notional, 2))),
            entry_price=_price(current_price),
            take_profit=_price(take_profit),
            stop_loss=_price(stop_loss),
            confidence=confidence,
            expected_return=tp_dist * 100.0,
            expected_risk=1.0,
            regime=regime,
            feature_id=feature_vector.feature_id,
            rationale=(
                f"SHADOW cost-aware probe: {reason}, tp={tp_dist * 100:.3f}%, sl={sl_dist * 100:.3f}%"
                + (f", net_rr={net_reward_risk:.2f}" if net_reward_risk is not None else "")
            ),
        )
