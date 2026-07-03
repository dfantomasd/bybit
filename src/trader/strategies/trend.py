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

Stop loss: 2.0 × ATR from entry
Take profit: 4 × ATR from entry (2:1 R/R)
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
_EMA_SLOPE_MIN = 0.0002  # EMA9 must be rising — raised to filter weak drift
_EMA_SEPARATION_MIN = 0.0002  # EMA9 and EMA21 must be meaningfully separated
_RSI_LONG_MIN = 0.50  # require momentum confirmation for longs
_RSI_LONG_MAX = 0.70
_RSI_SHORT_MIN = 0.30
_RSI_SHORT_MAX = 0.50  # tighter ceiling — avoid buying dips in bear moves
_VOLUME_ZSCORE_MIN = -0.5  # reject low-volume entries
_MACD_HIST_MIN_ABS = 0.0001  # reject weak MACD noise around zero

_ATR_STOP_MULTIPLIER = 2.0  # widened: 1.5→2.0 to survive normal intrabar noise
_ATR_TP_MULTIPLIER = 4.0  # keep 2:1 R/R at the new stop distance
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
        min_adx: float = 0.30,
        block_negative_funding_oi: bool = True,
        taker_fee_pct: float = 0.055,
        expected_slippage_pct: float = 0.03,
        max_spread_bps: float = 8.0,
        min_net_return_pct: float = 0.05,
    ) -> None:
        self._symbol = symbol.upper() if symbol else None
        self._allow_short = allow_short
        self._min_qty_usd = min_qty_usd
        self._max_risk_pct = max_risk_pct
        self._min_adx = min_adx
        self._block_negative_funding_oi = block_negative_funding_oi
        self._round_trip_fee_pct = max(0.0, taker_fee_pct) * 2.0
        self._expected_slippage_pct = max(0.0, expected_slippage_pct)
        self._max_spread_pct = max(0.0, max_spread_bps) / 100.0
        self._min_net_return_pct = max(0.0, min_net_return_pct)

    def _net_tp_return_pct(self, tp_dist: float) -> float:
        gross_tp_pct = tp_dist * 100.0
        safety_pct = 0.01
        round_trip_slippage_pct = self._expected_slippage_pct * 2.0
        return gross_tp_pct - self._round_trip_fee_pct - self._max_spread_pct - round_trip_slippage_pct - safety_pct

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
        log_return_1 = f.get("log_return_1")
        volume_z = f.get("volume_zscore")
        atr_pct = f.get("atr_14_pct")
        adx14 = f.get("adx_14")
        funding_bps = f.get("funding_rate_bps_clipped", f.get("funding_rate_bps", 0.0)) or 0.0
        oi_change_pct = f.get("oi_change_pct_60m_clipped", f.get("oi_change_pct_60m", 0.0)) or 0.0

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

        if self._min_adx > 0:
            if adx14 is None or adx14 < self._min_adx:
                return None

        # Volume filter
        if volume_z is not None and volume_z < _VOLUME_ZSCORE_MIN:
            return None

        # EMA relationship: ema_dist is (ema/price - 1)
        # ema9_dist > ema21_dist → EMA9 is farther above price than EMA21
        # Actually: dist = ema/price - 1; if ema > price, dist > 0
        # EMA9 > EMA21 means ema9_dist > ema21_dist (both relative to same price)
        ema9_above_ema21 = ema9_dist > ema21_dist
        ema_separation = abs(ema9_dist - ema21_dist)
        if ema_separation < _EMA_SEPARATION_MIN:
            return None

        # --- Long signal ---
        if ema9_above_ema21 and ema9_slope > _EMA_SLOPE_MIN:
            # For a long trend continuation, price should already be above the
            # fast EMA (ema_9 < 0 because feature = ema / price - 1), and very
            # recent returns should agree when available. This avoids buying
            # weak pullbacks that only look bullish because EMAs lag.
            if ema9_dist >= 0:
                return None
            if log_return_1 is not None and log_return_1 <= 0:
                return None
            if self._block_negative_funding_oi and funding_bps < -2.0 and oi_change_pct < -0.5:
                return None
            if _RSI_LONG_MIN <= rsi14 <= _RSI_LONG_MAX and macd_hist > _MACD_HIST_MIN_ABS:
                bonus_conditions = sum(
                    [
                        ema9_slope > _EMA_SLOPE_MIN * 3,  # strong slope momentum
                        rsi14 > 0.55,  # clearer upside bias
                        (volume_z or 0) > 0.5,  # above-average volume
                    ]
                )
                confidence = _BASE_CONFIDENCE + bonus_conditions * _CONFIDENCE_PER_CONDITION

                stop_dist = atr_pct * _ATR_STOP_MULTIPLIER
                tp_dist = atr_pct * _ATR_TP_MULTIPLIER
                if self._net_tp_return_pct(tp_dist) < self._min_net_return_pct:
                    return None
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
            # Mirror of the long filter: price should be below fast EMA
            # (ema_9 > 0), and short-term returns should already point down.
            if ema9_dist <= 0:
                return None
            if log_return_1 is not None and log_return_1 >= 0:
                return None
            if _RSI_SHORT_MIN <= rsi14 <= _RSI_SHORT_MAX and macd_hist < -_MACD_HIST_MIN_ABS:
                bonus_conditions = sum(
                    [
                        ema9_slope < -_EMA_SLOPE_MIN * 3,  # strong downward momentum
                        rsi14 > 0.40,  # bearish but not yet oversold (mirrors long-side rsi14 > 0.55)
                        (volume_z or 0) > 0.5,  # above-average volume
                    ]
                )
                confidence = _BASE_CONFIDENCE + bonus_conditions * _CONFIDENCE_PER_CONDITION

                stop_dist = atr_pct * _ATR_STOP_MULTIPLIER
                tp_dist = atr_pct * _ATR_TP_MULTIPLIER
                if self._net_tp_return_pct(tp_dist) < self._min_net_return_pct:
                    return None
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
