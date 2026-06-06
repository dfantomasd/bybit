"""EMA-crossover trend-following strategy with RSI momentum filter.

Signal logic
------------
Long signal:
  - EMA9 > EMA21 (fast above slow)
  - EMA slope (9) > threshold (uptrend)
  - RSI14 between 45 and 70 (not overbought, has momentum)
  - MACD histogram > 0
  - Volume z-score > -0.5 (not drying up)

Short signal (only if shorting enabled):
  - EMA9 < EMA21
  - EMA slope (9) < -threshold
  - RSI14 between 30 and 55
  - MACD histogram < 0
  - Volume z-score > -0.5

Stop loss: 1.5 × ATR below entry
Take profit: 3 × ATR above entry (2:1 R/R minimum)
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import structlog

from trader.domain.enums import MarketRegime, MarketType, OrderSide
from trader.domain.models import FeatureVector, TradeProposal
from trader.strategies.base import BaseStrategy

log = structlog.get_logger(__name__)

_STRATEGY_ID = "ema_crossover_v1"

# Signal thresholds
_EMA_SLOPE_MIN = 0.0001  # EMA9 must be rising at least this fast (normalised)
_RSI_LONG_MIN = 0.45  # RSI14 in [0,1] scale
_RSI_LONG_MAX = 0.70
_RSI_SHORT_MIN = 0.30
_RSI_SHORT_MAX = 0.55
_VOLUME_ZSCORE_MIN = -0.5  # reject low-volume entries

_ATR_STOP_MULTIPLIER = 1.5
_ATR_TP_MULTIPLIER = 3.0
_MIN_ATR_PCT = 0.001  # skip if ATR is basically zero
_MAX_ATR_PCT = 0.05  # skip if market is too volatile

_BASE_CONFIDENCE = 0.55  # starting confidence if all conditions met
_CONFIDENCE_PER_CONDITION = 0.05  # bonus per extra confirmed condition

_PRICE_DECIMALS = Decimal("0.00000001")


def _price(value: float) -> Decimal:
    """Keep enough precision for cheap symbols; exchange tick rounding happens later."""
    return Decimal(str(value)).quantize(_PRICE_DECIMALS)


class EMAcrossoverStrategy(BaseStrategy):
    """Rule-based EMA crossover strategy with RSI and volume filters."""

    def __init__(
        self,
        symbol: str | None = None,  # None = evaluate any symbol
        allow_short: bool = True,
        min_qty_usd: float = 5.0,
        max_risk_pct: float = 0.01,  # 1% of balance per trade
    ) -> None:
        self._symbol = symbol.upper() if symbol else None
        self._allow_short = allow_short
        self._min_qty_usd = min_qty_usd
        self._max_risk_pct = max_risk_pct

    @property
    def strategy_id(self) -> str:
        return _STRATEGY_ID

    def evaluate(
        self,
        feature_vector: FeatureVector,
        current_price: float,
        available_balance_usd: float,
    ) -> TradeProposal | None:
        # If bound to a specific symbol, skip others
        if self._symbol is not None and feature_vector.symbol != self._symbol:
            return None
        if feature_vector.quality_score < 0.6:
            log.debug(
                "ema_crossover.skip_low_quality",
                symbol=feature_vector.symbol,
                quality=feature_vector.quality_score,
            )
            return None

        symbol = feature_vector.symbol
        f = dict(zip(feature_vector.feature_names, feature_vector.values, strict=True))

        # Extract required features
        ema9_dist = f.get("ema_9")  # normalised distance: ema/price - 1
        ema21_dist = f.get("ema_21")
        ema9_slope = f.get("ema_slope_9")
        rsi14 = f.get("rsi_14")  # [0, 1]
        macd_hist = f.get("macd_hist")
        volume_z = f.get("volume_zscore")
        atr_pct = f.get("atr_14_pct")

        # All must be present
        if any(v is None for v in [ema9_dist, ema21_dist, ema9_slope, rsi14, macd_hist]):
            return None

        assert ema9_dist is not None
        assert ema21_dist is not None
        assert ema9_slope is not None
        assert rsi14 is not None
        assert macd_hist is not None

        # ATR checks
        if atr_pct is not None:
            if atr_pct < _MIN_ATR_PCT or atr_pct > _MAX_ATR_PCT:
                return None
        else:
            return None  # need ATR for stop placement

        # Volume filter
        if volume_z is not None and volume_z < _VOLUME_ZSCORE_MIN:
            return None

        # EMA relationship: ema_dist is (ema/price - 1)
        # ema9_dist > ema21_dist → EMA9 is farther above price than EMA21
        # Actually: dist = ema/price - 1; if ema > price, dist > 0
        # EMA9 > EMA21 means ema9_dist > ema21_dist (both relative to same price)
        ema9_above_ema21 = ema9_dist > ema21_dist

        # --- Long signal ---
        if ema9_above_ema21 and ema9_slope > _EMA_SLOPE_MIN:
            if _RSI_LONG_MIN <= rsi14 <= _RSI_LONG_MAX and macd_hist > 0:
                bonus_conditions = sum(
                    [
                        ema9_slope > _EMA_SLOPE_MIN * 3,
                        rsi14 > 0.50,
                        (volume_z or 0) > 0.5,
                    ]
                )
                confidence = _BASE_CONFIDENCE + bonus_conditions * _CONFIDENCE_PER_CONDITION

                stop_dist = atr_pct * _ATR_STOP_MULTIPLIER
                tp_dist = atr_pct * _ATR_TP_MULTIPLIER
                stop_price = current_price * (1 - stop_dist)
                entry_price = current_price

                qty_usd = available_balance_usd * self._max_risk_pct / stop_dist
                # Cap: never more than 30% of balance (prevents ATR-blown positions)
                qty_usd = min(qty_usd, available_balance_usd * 0.30)
                qty_usd = max(qty_usd, self._min_qty_usd)
                qty = qty_usd / current_price

                log.info(
                    "ema_crossover.long_signal",
                    symbol=symbol,
                    confidence=round(confidence, 3),
                    rsi14=round(rsi14, 3),
                    ema9_slope=round(ema9_slope, 6),
                    macd_hist=round(macd_hist, 6),
                )

                return TradeProposal(
                    proposal_id=uuid.uuid4(),
                    strategy_id=_STRATEGY_ID,
                    symbol=symbol,
                    market_type=MarketType.LINEAR,
                    side=OrderSide.BUY,
                    requested_qty=Decimal(str(round(qty, 4))),
                    requested_notional_usd=Decimal(str(round(qty_usd, 2))),
                    entry_price=_price(entry_price),
                    stop_loss=_price(stop_price),
                    take_profit=_price(entry_price * (1 + tp_dist)),
                    confidence=min(confidence, 0.95),
                    regime=MarketRegime.BULL_TREND,
                    rationale=(f"EMA9>EMA21, slope={ema9_slope:.4f}, RSI14={rsi14:.2f}, MACDhist={macd_hist:.4f}"),
                )

        # --- Short signal ---
        if self._allow_short and not ema9_above_ema21 and ema9_slope < -_EMA_SLOPE_MIN:
            if _RSI_SHORT_MIN <= rsi14 <= _RSI_SHORT_MAX and macd_hist < 0:
                bonus_conditions = sum(
                    [
                        ema9_slope < -_EMA_SLOPE_MIN * 3,
                        rsi14 < 0.45,
                        (volume_z or 0) > 0.5,
                    ]
                )
                confidence = _BASE_CONFIDENCE + bonus_conditions * _CONFIDENCE_PER_CONDITION

                stop_dist = atr_pct * _ATR_STOP_MULTIPLIER
                tp_dist = atr_pct * _ATR_TP_MULTIPLIER
                stop_price = current_price * (1 + stop_dist)
                entry_price = current_price

                qty_usd = available_balance_usd * self._max_risk_pct / stop_dist
                qty_usd = min(qty_usd, available_balance_usd * 0.30)
                qty_usd = max(qty_usd, self._min_qty_usd)
                qty = qty_usd / current_price

                log.info(
                    "ema_crossover.short_signal",
                    symbol=symbol,
                    confidence=round(confidence, 3),
                )

                return TradeProposal(
                    proposal_id=uuid.uuid4(),
                    strategy_id=_STRATEGY_ID,
                    symbol=symbol,
                    market_type=MarketType.LINEAR,
                    side=OrderSide.SELL,
                    requested_qty=Decimal(str(round(qty, 4))),
                    requested_notional_usd=Decimal(str(round(qty_usd, 2))),
                    entry_price=_price(entry_price),
                    stop_loss=_price(stop_price),
                    take_profit=_price(entry_price * (1 - tp_dist)),
                    confidence=min(confidence, 0.95),
                    regime=MarketRegime.BEAR_TREND,
                    rationale=(f"EMA9<EMA21, slope={ema9_slope:.4f}, RSI14={rsi14:.2f}, MACDhist={macd_hist:.4f}"),
                )

        return None
