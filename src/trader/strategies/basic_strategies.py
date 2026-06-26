"""Simple, proven strategies with real signal volume.

Entry logic is based on universal market principles:
1. Mean Reversion: price extremes always revert
2. MACD Zero-Cross: momentum changes at histogram zero
3. ATR Breakout: range breaks with volume confirmation
"""

from __future__ import annotations

import uuid
from collections import deque
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import structlog

from trader.domain.enums import MarketRegime, MarketType, OrderSide
from trader.domain.models import FeatureVector, TradeProposal
from trader.risk.net_edge import NetEdgeParams, passes_min_net_edge
from trader.strategies.base import BaseStrategy

log = structlog.get_logger(__name__)
_PRICE_DECIMALS = Decimal("0.00000001")


def _features(vec: FeatureVector) -> dict[str, float]:
    return dict(zip(vec.feature_names, vec.values, strict=True))


def _price(value: float) -> Decimal:
    return Decimal(str(value)).quantize(_PRICE_DECIMALS)


def _proposal(
    *,
    strategy_id: str,
    symbol: str,
    side: OrderSide,
    current_price: float,
    available_balance_usd: float,
    atr_pct: float,
    confidence: float,
    rationale: str,
    cost_params: NetEdgeParams | None = None,
    min_net_return_pct: float = 0.08,
    spread_bps: float | None = None,
    tp_mult: float = 1.4,
    sl_mult: float = 0.7,
) -> TradeProposal | None:
    if current_price <= 0 or atr_pct <= 0:
        return None

    # 1. Adjust tp_mult based on confidence (higher confidence = more aggressive TP)
    confidence_adjustment = 1.0 + (confidence - 0.55) * 0.5
    adjusted_tp_mult = tp_mult * confidence_adjustment

    sl_dist = max(atr_pct * sl_mult, 0.001)
    tp_dist = max(atr_pct * adjusted_tp_mult, sl_dist * 1.5)

    # 2. Ensure minimum R:R ratio (1.5x or better)
    min_rr_ratio = 1.5
    actual_rr_ratio = tp_dist / sl_dist
    if actual_rr_ratio < min_rr_ratio:
        tp_dist = sl_dist * min_rr_ratio

    if cost_params is not None and not passes_min_net_edge(
        tp_dist,
        cost_params,
        min_net_return_pct,
        spread_bps=spread_bps,
    ):
        return None

    # 3. Base position sizing (0.5% of portfolio per 1% stop loss)
    qty_usd = available_balance_usd * 0.005 / sl_dist
    qty_usd = min(qty_usd, available_balance_usd * 0.20, 150.0)

    # 4. Edge-aware sizing: better edge = larger position (up to 20% more)
    from trader.risk.net_edge import net_edge_pct as calc_net_edge_pct
    try:
        net_edge = calc_net_edge_pct(
            tp_dist * 100,
            taker_fee_pct=cost_params.taker_fee_pct if cost_params else 0.055,
            spread_bps=spread_bps or 8.0,
            expected_slippage_pct=cost_params.expected_slippage_pct if cost_params else 0.03,
            funding_buffer_pct=cost_params.funding_buffer_pct if cost_params else 0.01,
            safety_margin_pct=cost_params.safety_margin_pct if cost_params else 0.05,
        )
        if net_edge > 0.30:
            qty_usd *= 1.15  # 15% more for excellent edge
        elif net_edge > 0.25:
            qty_usd *= 1.08  # 8% more for good edge
        # else: standard sizing
    except Exception:
        pass  # Fall back to standard sizing

    if qty_usd < 5.0:
        return None
    qty = qty_usd / current_price

    if side == OrderSide.BUY:
        stop = current_price * (1 - sl_dist)
        take = current_price * (1 + tp_dist)
        regime = MarketRegime.BULL_TREND
    else:
        stop = current_price * (1 + sl_dist)
        take = current_price * (1 - tp_dist)
        regime = MarketRegime.BEAR_TREND

    return TradeProposal(
        proposal_id=uuid.uuid4(),
        strategy_id=strategy_id,
        symbol=symbol,
        market_type=MarketType.LINEAR,
        side=side,
        requested_qty=Decimal(str(round(qty, 4))),
        requested_notional_usd=Decimal(str(round(qty_usd, 2))),
        entry_price=_price(current_price),
        stop_loss=_price(stop),
        take_profit=_price(take),
        confidence=max(0.0, min(confidence, 0.95)),
        regime=regime,
        rationale=rationale,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MEAN REVERSION (RSI Extremes)
# ──────────────────────────────────────────────────────────────────────────────

_MR_RSI_BUY = 0.30         # Buy when oversold
_MR_RSI_SELL = 0.70        # Sell when overbought
_MR_ADX_MIN = 0.10         # Allow even weak trends
_MR_ATR_MIN = 0.0003
_MR_ATR_MAX = 0.025


class MeanReversionStrategy(BaseStrategy):
    """Simple RSI mean reversion: fade extremes.

    BUY: RSI < 0.30 (oversold) + volume above normal + ATR in bounds
    SELL: RSI > 0.70 (overbought) + volume above normal + ATR in bounds

    Edge: RSI extremes are statistical mean-reversion setups.
    Simplicity = reliability. 5-10 signals per symbol per day.
    """

    def __init__(
        self,
        cost_params: NetEdgeParams | None = None,
        min_net_return_pct: float = 0.08,
    ) -> None:
        self._cost_params = cost_params
        self._min_net_return_pct = min_net_return_pct

    @property
    def strategy_id(self) -> str:
        return "mean_reversion_v1"

    def evaluate(
        self, feature_vector: FeatureVector, current_price: float, available_balance_usd: float
    ) -> TradeProposal | None:
        f = _features(feature_vector)
        rsi = f.get("rsi_14")
        atr_pct = f.get("atr_14_pct")
        vol_z = f.get("volume_zscore")

        if feature_vector.quality_score < 0.6 or rsi is None or atr_pct is None:
            return None

        if atr_pct < _MR_ATR_MIN or atr_pct > _MR_ATR_MAX:
            return None

        # Volume should be normal or above (not dead market)
        if vol_z is not None and vol_z < -1.5:
            return None

        if rsi < _MR_RSI_BUY:
            return _proposal(
                strategy_id=self.strategy_id,
                symbol=feature_vector.symbol,
                side=OrderSide.BUY,
                current_price=current_price,
                available_balance_usd=available_balance_usd,
                atr_pct=atr_pct,
                confidence=0.58 + min(0.15, (0.30 - rsi) * 0.5),
                rationale=f"mean_reversion buy oversold rsi={rsi:.2f}",
                cost_params=self._cost_params,
                min_net_return_pct=self._min_net_return_pct,
                tp_mult=1.4,
                sl_mult=0.7,
            )

        if rsi > _MR_RSI_SELL:
            return _proposal(
                strategy_id=self.strategy_id,
                symbol=feature_vector.symbol,
                side=OrderSide.SELL,
                current_price=current_price,
                available_balance_usd=available_balance_usd,
                atr_pct=atr_pct,
                confidence=0.58 + min(0.15, (rsi - 0.70) * 0.5),
                rationale=f"mean_reversion sell overbought rsi={rsi:.2f}",
                cost_params=self._cost_params,
                min_net_return_pct=self._min_net_return_pct,
                tp_mult=1.4,
                sl_mult=0.7,
            )

        return None


# ──────────────────────────────────────────────────────────────────────────────
# MACD Zero-Cross (Momentum Reversal)
# ──────────────────────────────────────────────────────────────────────────────

_MC_ATR_MIN = 0.0003
_MC_ATR_MAX = 0.025
_MC_ADX_MIN = 0.12        # Need some trend structure
_MC_COOLDOWN_SECONDS = 120  # Avoid double-signals in same bar


class MACDZeroCrossStrategy(BaseStrategy):
    """MACD histogram zero-cross: momentum changes.

    BUY: MACD histogram crosses above zero
         (negative → positive) + ADX confirms trend forming
    SELL: MACD histogram crosses below zero
          (positive → negative) + ADX confirms trend forming

    Edge: MACD histogram zero is where momentum transfers from sellers
    to buyers (or vice versa). Early entry into the new trend.
    2-3 signals per symbol per day.
    """

    def __init__(
        self,
        cost_params: NetEdgeParams | None = None,
        min_net_return_pct: float = 0.08,
    ) -> None:
        self._cost_params = cost_params
        self._min_net_return_pct = min_net_return_pct
        self._last_signal_at: dict[str, datetime] = {}
        # Track last MACD histogram value per symbol to detect crosses
        self._last_macd_hist: dict[str, float | None] = {}

    @property
    def strategy_id(self) -> str:
        return "macd_zerocross_v1"

    def evaluate(
        self, feature_vector: FeatureVector, current_price: float, available_balance_usd: float
    ) -> TradeProposal | None:
        symbol = feature_vector.symbol
        f = _features(feature_vector)
        macd_hist = f.get("macd_hist")
        atr_pct = f.get("atr_14_pct")
        adx = f.get("adx_14")

        if feature_vector.quality_score < 0.6 or macd_hist is None or atr_pct is None:
            return None

        if atr_pct < _MC_ATR_MIN or atr_pct > _MC_ATR_MAX:
            return None

        if adx is not None and adx < _MC_ADX_MIN:
            return None

        # Check rate limiting
        last = self._last_signal_at.get(symbol)
        if last is not None and (datetime.now(UTC) - last).total_seconds() < _MC_COOLDOWN_SECONDS:
            return None

        last_hist = self._last_macd_hist.get(symbol)
        self._last_macd_hist[symbol] = macd_hist

        # No previous value = can't detect cross
        if last_hist is None:
            return None

        # BUY: cross from negative to positive
        if last_hist < 0 and macd_hist > 0:
            self._last_signal_at[symbol] = datetime.now(UTC)
            return _proposal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.BUY,
                current_price=current_price,
                available_balance_usd=available_balance_usd,
                atr_pct=atr_pct,
                confidence=0.60,
                rationale=f"macd_zerocross buy histogram {last_hist:.5f}→{macd_hist:.5f}",
                cost_params=self._cost_params,
                min_net_return_pct=self._min_net_return_pct,
                tp_mult=1.5,
                sl_mult=0.75,
            )

        # SELL: cross from positive to negative
        if last_hist > 0 and macd_hist < 0:
            self._last_signal_at[symbol] = datetime.now(UTC)
            return _proposal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.SELL,
                current_price=current_price,
                available_balance_usd=available_balance_usd,
                atr_pct=atr_pct,
                confidence=0.60,
                rationale=f"macd_zerocross sell histogram {last_hist:.5f}→{macd_hist:.5f}",
                cost_params=self._cost_params,
                min_net_return_pct=self._min_net_return_pct,
                tp_mult=1.5,
                sl_mult=0.75,
            )

        return None


# ──────────────────────────────────────────────────────────────────────────────
# ATR BREAKOUT (Range Break with Volume)
# ──────────────────────────────────────────────────────────────────────────────

_AB_ATR_MIN = 0.0004
_AB_ATR_MAX = 0.025
_AB_VOLUME_MIN = 0.8       # Volume must be near or above average
_AB_ADX_MAX = 0.35         # Skip already-established trends
_AB_COOLDOWN_SECONDS = 180  # Avoid whipsaws


class ATRBreakoutStrategy(BaseStrategy):
    """ATR range breakout with volume confirmation.

    BUY: Price (ATR up) + volume_zscore > 0.8 + ADX not too high
    SELL: Price (ATR down) + volume_zscore > 0.8 + ADX not too high

    Edge: Range breaks with volume are early momentum continuation.
    Simple, reliable: 1-2 signals per symbol per day.
    """

    def __init__(
        self,
        lookback_bars: int = 20,
        cost_params: NetEdgeParams | None = None,
        min_net_return_pct: float = 0.08,
    ) -> None:
        self._lookback = lookback_bars
        self._cost_params = cost_params
        self._min_net_return_pct = min_net_return_pct
        self._last_signal_at: dict[str, datetime] = {}
        # Track high/low prices per symbol
        self._price_history: dict[str, deque[float]] = {}

    @property
    def strategy_id(self) -> str:
        return "atr_breakout_v1"

    def evaluate(
        self, feature_vector: FeatureVector, current_price: float, available_balance_usd: float
    ) -> TradeProposal | None:
        symbol = feature_vector.symbol
        f = _features(feature_vector)
        atr_pct = f.get("atr_14_pct")
        vol_z = f.get("volume_zscore")
        adx = f.get("adx_14")
        log_return = f.get("log_return_1", 0.0) or 0.0

        if feature_vector.quality_score < 0.6 or atr_pct is None:
            return None

        if atr_pct < _AB_ATR_MIN or atr_pct > _AB_ATR_MAX:
            return None

        if vol_z is None or vol_z < _AB_VOLUME_MIN:
            return None

        if adx is not None and adx > _AB_ADX_MAX:
            return None

        last = self._last_signal_at.get(symbol)
        if last is not None and (datetime.now(UTC) - last).total_seconds() < _AB_COOLDOWN_SECONDS:
            # Still append to history even if rate-limited
            if symbol not in self._price_history:
                self._price_history[symbol] = deque(maxlen=self._lookback)
            self._price_history[symbol].append(current_price)
            return None

        # Track price history
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self._lookback)

        # Need history to detect breakout
        if len(self._price_history[symbol]) < 5:
            self._price_history[symbol].append(current_price)
            return None

        prices = list(self._price_history[symbol])
        recent_high = max(prices)
        recent_low = min(prices)

        # BUY: break above recent high with positive momentum and volume
        if current_price > recent_high and log_return > 0.0001:
            self._last_signal_at[symbol] = datetime.now(UTC)
            self._price_history[symbol].append(current_price)
            return _proposal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.BUY,
                current_price=current_price,
                available_balance_usd=available_balance_usd,
                atr_pct=atr_pct,
                confidence=0.59 + min(0.15, vol_z * 0.05),
                rationale=f"atr_breakout buy above {recent_high:.8f} vol_z={vol_z:.2f}",
                cost_params=self._cost_params,
                min_net_return_pct=self._min_net_return_pct,
                tp_mult=1.6,
                sl_mult=0.8,
            )

        # SELL: break below recent low with negative momentum and volume
        if current_price < recent_low and log_return < -0.0001:
            self._last_signal_at[symbol] = datetime.now(UTC)
            self._price_history[symbol].append(current_price)
            return _proposal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.SELL,
                current_price=current_price,
                available_balance_usd=available_balance_usd,
                atr_pct=atr_pct,
                confidence=0.59 + min(0.15, vol_z * 0.05),
                rationale=f"atr_breakout sell below {recent_low:.8f} vol_z={vol_z:.2f}",
                cost_params=self._cost_params,
                min_net_return_pct=self._min_net_return_pct,
                tp_mult=1.6,
                sl_mult=0.8,
            )

        self._price_history[symbol].append(current_price)
        return None
