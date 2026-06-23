"""SHADOW-only probe strategy for paper order discovery.

This strategy intentionally runs only in SHADOW. Its job is not to be a live
alpha source, but to create enough conditional paper entries for model-gate
and TP/SL outcome analysis when production strategies are too selective.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from trader.domain.enums import MarketRegime, MarketType, OrderSide
from trader.domain.models import FeatureVector, TradeProposal
from trader.strategies.base import BaseStrategy

_STRATEGY_ID = "shadow_probe_v1"
_PRICE_DECIMALS = Decimal("0.00000001")


def _price(value: float) -> Decimal:
    return Decimal(str(value)).quantize(_PRICE_DECIMALS)


class ShadowProbeStrategy(BaseStrategy):
    """Generate safe paper-only probes from broad microstructure signals."""

    def __init__(
        self,
        *,
        imbalance_provider: Callable[[str], float | None] | None = None,
        min_abs_imbalance: float = 0.03,
        min_quality: float = 0.45,
        cooldown_seconds: int = 300,
        max_notional_usd: float = 8.0,
        risk_pct: float = 0.003,
        tp_atr_mult: float = 1.0,
        sl_atr_mult: float = 0.6,
    ) -> None:
        self._imbalance_provider = imbalance_provider
        self._min_abs_imbalance = max(0.0, float(min_abs_imbalance))
        self._min_quality = max(0.0, min(1.0, float(min_quality)))
        self._cooldown = timedelta(seconds=max(30, int(cooldown_seconds)))
        self._max_notional_usd = max(5.0, float(max_notional_usd))
        self._risk_pct = max(0.0001, float(risk_pct))
        self._tp_atr_mult = max(0.2, float(tp_atr_mult))
        self._sl_atr_mult = max(0.2, float(sl_atr_mult))
        self._last_signal_at: dict[str, datetime] = {}

    @property
    def strategy_id(self) -> str:
        return _STRATEGY_ID

    def _cooldown_active(self, symbol: str) -> bool:
        last = self._last_signal_at.get(symbol)
        return last is not None and datetime.now(tz=UTC) - last < self._cooldown

    @staticmethod
    def _features(vec: FeatureVector) -> dict[str, float]:
        return dict(zip(vec.feature_names, vec.values, strict=True))

    def _side_from_features(self, vec: FeatureVector, features: dict[str, float]) -> tuple[OrderSide | None, str]:
        imbalance = None
        if self._imbalance_provider is not None:
            try:
                imbalance = self._imbalance_provider(vec.symbol)
            except Exception:
                imbalance = None
        if imbalance is not None and abs(imbalance) >= self._min_abs_imbalance:
            side = OrderSide.BUY if imbalance > 0 else OrderSide.SELL
            return side, f"book imbalance {imbalance:+.3f}"

        ema9 = features.get("ema_9")
        ema21 = features.get("ema_21")
        rsi = features.get("rsi_14")
        if ema9 is None or ema21 is None:
            return None, "missing ema"
        if ema9 > ema21 and (rsi is None or rsi < 72):
            return OrderSide.BUY, "ema9>ema21 probe"
        if ema9 < ema21 and (rsi is None or rsi > 28):
            return OrderSide.SELL, "ema9<ema21 probe"
        return None, "no directional bias"

    def evaluate(
        self,
        feature_vector: FeatureVector,
        current_price: float,
        available_balance_usd: float,
    ) -> TradeProposal | None:
        if current_price <= 0 or available_balance_usd <= 0:
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

        sl_dist = max(float(atr_pct) * self._sl_atr_mult, 0.001)
        tp_dist = max(float(atr_pct) * self._tp_atr_mult, sl_dist * 1.25)
        notional = min(self._max_notional_usd, max(5.0, available_balance_usd * 0.25))
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

        self._last_signal_at[feature_vector.symbol] = datetime.now(tz=UTC)
        confidence = min(0.62, 0.50 + min(0.10, abs(float(atr_pct)) * 20.0))
        return TradeProposal(
            proposal_id=uuid.uuid4(),
            strategy_id=_STRATEGY_ID,
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
            expected_risk=sl_dist * 100.0,
            regime=regime,
            feature_id=feature_vector.feature_id,
            rationale=f"SHADOW probe: {reason}, tp={tp_dist * 100:.3f}%, sl={sl_dist * 100:.3f}%",
        )
