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
from trader.risk.net_edge import NetEdgeParams, passes_min_net_edge
from trader.strategies.base import BaseStrategy

SHADOW_PROBE_STRATEGY_ID = "shadow_probe_hv_v2"
_PRICE_DECIMALS = Decimal("0.00000001")
# Worst-case qty shrink from confidence/regime/VWAP penalties after risk sizing.
_PROBE_WORST_CASE_QTY_MULTIPLIER = Decimal("0.25")


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
        min_sl_pct: float = 0.25,
        min_net_return_pct: float = 0.05,
        min_notional_buffer_pct: float = 3.0,
        cost_params: NetEdgeParams | None = None,
        sell_enabled: bool = False,
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
        self._min_sl_pct = max(0.05, float(min_sl_pct))
        self._min_net_return_pct = max(0.0, float(min_net_return_pct))
        self._min_notional_buffer_pct = max(0.0, float(min_notional_buffer_pct))
        self._cost_params = cost_params
        self._sell_enabled = bool(sell_enabled)
        self._last_signal_at: dict[str, datetime] = {}
        self._signal_times: deque[datetime] = deque()
        self._burst_blocked_until: datetime | None = None

    @property
    def strategy_id(self) -> str:
        return SHADOW_PROBE_STRATEGY_ID

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
        ema9 = features.get("ema_9")
        ema21 = features.get("ema_21")
        rsi = features.get("rsi_14")
        if ema9 is None or ema21 is None:
            return None
        if ema9 > ema21 and (rsi is None or rsi < 68):
            return OrderSide.BUY
        if ema9 < ema21 and (rsi is None or rsi > 32):
            return OrderSide.SELL
        return None

    def _side_from_features(self, vec: FeatureVector, features: dict[str, float]) -> tuple[OrderSide | None, str]:
        ema_side = self._ema_side(features)
        imbalance = None
        if self._imbalance_provider is not None:
            try:
                imbalance = self._imbalance_provider(vec.symbol)
            except Exception:
                imbalance = None
        if imbalance is None:
            return None, "orderbook imbalance unavailable"
        if abs(imbalance) < self._min_abs_imbalance:
            return None, f"orderbook imbalance below threshold {imbalance:+.3f}"
        side = OrderSide.BUY if imbalance > 0 else OrderSide.SELL
        if ema_side is None:
            return None, f"EMA confirmation unavailable imbalance={imbalance:+.3f}"
        if side != ema_side:
            return None, f"book/EMA conflict imbalance={imbalance:+.3f}"
        return side, f"book imbalance {imbalance:+.3f}"

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
            return None
        if self._burst_limited(now):
            return None
        if self._symbol_allowed is not None and not self._symbol_allowed(feature_vector.symbol):
            return None
        if self._regime_allows is not None and not self._regime_allows(feature_vector):
            return None
        if feature_vector.quality_score < self._min_quality:
            return None
        if self._cooldown_active(feature_vector.symbol):
            return None

        features = self._features(feature_vector)
        atr_pct = features.get("atr_14_pct")
        if atr_pct is None or atr_pct <= 0:
            return None
        # Keep probes meaningful: avoid dead/noisy extremes but stay much wider
        # than live strategies so SHADOW accumulates paper outcomes.
        if atr_pct < 0.00025 or atr_pct > 0.04:
            return None

        side, reason = self._side_from_features(feature_vector, features)
        if side is None:
            return None
        if side == OrderSide.SELL and not self._sell_enabled:
            return None
        if self._side_blocked is not None and self._side_blocked(feature_vector.symbol, side.value):
            return None

        sl_dist = max(float(atr_pct) * self._sl_atr_mult, self._min_sl_pct / 100.0)
        tp_dist = max(float(atr_pct) * self._tp_atr_mult, self._min_tp_pct / 100.0, sl_dist * 1.5)
        if self._cost_params is not None and not passes_min_net_edge(
            tp_dist,
            self._cost_params,
            self._min_net_return_pct,
        ):
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
                return None

        qty = notional / current_price
        if qty <= 0:
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
            rationale=f"SHADOW cost-aware probe: {reason}, tp={tp_dist * 100:.3f}%, sl={sl_dist * 100:.3f}%",
        )
