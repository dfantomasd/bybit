"""Market regime classifier.

Classifies the current market regime for a symbol from its FeatureVector.
Uses a rule-based cascade (no ML model needed at this stage):

  ADX + BB Bandwidth → Volatility level
  EMA slopes + RSI   → Trend direction
  Volume Z-score     → Liquidity level

Regime hierarchy (checked in order):
  1. LOW_LIQUIDITY    — volume drying up
  2. HIGH_VOLATILITY  — bandwidth exploding
  3. BULL_TREND       — ADX trending + rising EMA
  4. BEAR_TREND       — ADX trending + falling EMA
  5. SIDEWAYS         — ADX low / ranging
  6. UNCERTAIN        — default fallback

The returned ``RegimeContext`` is consumed by the RiskManager, which applies
per-regime risk multipliers and may block entries entirely.
"""

from __future__ import annotations

import structlog

from trader.domain.enums import MarketRegime, VolatilityLevel
from trader.domain.models import FeatureVector, RegimeContext

log = structlog.get_logger(__name__)

# ------------------------------------------------------------------
# Threshold constants (tuned for 1-min bars, perpetual futures)
# ------------------------------------------------------------------

_ADX_TRENDING = 25.0  # ADX (normalised ÷100) = 0.25 → strong trend
_ADX_RANGING = 20.0  # ADX below this = sideways

_BB_BW_HIGH_VOL = 0.06  # BB bandwidth > 6% → high volatility
_BB_BW_EXTREME = 0.12  # > 12% → extreme (still HIGH_VOLATILITY)

_EMA_SLOPE_BULL = 0.00015  # EMA9 slope (normalised) positive threshold
_EMA_SLOPE_BEAR = -0.00015

_RSI_BULL_MIN = 0.48  # RSI14 [0,1] — must have upward momentum
_RSI_BEAR_MAX = 0.52  # RSI14 below this for bear regime

_VOL_ZSCORE_LOW = -1.5  # Volume Z-score below this = drying up


class RegimeClassifier:
    """Classifies market regime from a FeatureVector.

    Stateless — safe to call concurrently for multiple symbols.
    """

    def classify(self, feature_vector: FeatureVector) -> RegimeContext:
        """Return a RegimeContext derived from the feature vector."""
        symbol = feature_vector.symbol
        f = dict(zip(feature_vector.feature_names, feature_vector.values, strict=False))

        # Extract features (all optional — graceful degradation)
        adx: float = f.get("adx_14", 0.0) * 100  # denormalise [0,1] → [0,100]
        bb_bw: float = f.get("bb_bandwidth", 0.0)
        rsi14: float = f.get("rsi_14", 0.5)  # [0,1]
        ema9_slope: float = f.get("ema_slope_9", 0.0)
        ema21_slope: float = f.get("ema_slope_21", 0.0)
        vol_z: float = f.get("volume_zscore", 0.0)
        realized_vol: float = f.get("realized_vol_20", 0.0)

        # ------------------------------------------------------------------
        # 1. LOW_LIQUIDITY — volume drying up
        # ------------------------------------------------------------------
        if vol_z < _VOL_ZSCORE_LOW:
            confidence = min(0.6 + abs(vol_z + _VOL_ZSCORE_LOW) * 0.05, 0.85)
            regime = MarketRegime.LOW_LIQUIDITY
            return self._make_context(
                symbol=symbol,
                regime=regime,
                volatility=VolatilityLevel.LOW,
                confidence=round(confidence, 3),
                trading_allowed=False,
                block_reason="volume z-score below threshold (low liquidity)",
                feature_vector=feature_vector,
            )

        # ------------------------------------------------------------------
        # 2. HIGH_VOLATILITY — bands expanding rapidly
        # ------------------------------------------------------------------
        if bb_bw > _BB_BW_HIGH_VOL:
            confidence = min(0.55 + (bb_bw - _BB_BW_HIGH_VOL) * 5, 0.90)
            vol_level = VolatilityLevel.EXTREME if bb_bw > _BB_BW_EXTREME else VolatilityLevel.HIGH
            return self._make_context(
                symbol=symbol,
                regime=MarketRegime.HIGH_VOLATILITY,
                volatility=vol_level,
                confidence=round(confidence, 3),
                trading_allowed=True,  # allowed but with reduced size
                realized_vol=realized_vol,
                feature_vector=feature_vector,
            )

        # ------------------------------------------------------------------
        # 3 & 4. Trending — ADX confirms direction
        # ------------------------------------------------------------------
        adx_norm = adx / 100.0  # back to normalised for comparison

        if adx_norm >= _ADX_TRENDING / 100:
            # Strong trend — determine direction from EMA slopes and RSI
            both_slopes_up = ema9_slope > _EMA_SLOPE_BULL and ema21_slope > 0
            both_slopes_down = ema9_slope < _EMA_SLOPE_BEAR and ema21_slope < 0

            if both_slopes_up and rsi14 >= _RSI_BULL_MIN:
                confidence = self._trend_confidence(adx_norm, ema9_slope, rsi14, bull=True)
                return self._make_context(
                    symbol=symbol,
                    regime=MarketRegime.BULL_TREND,
                    volatility=VolatilityLevel.NORMAL,
                    confidence=confidence,
                    trading_allowed=True,
                    realized_vol=realized_vol,
                    feature_vector=feature_vector,
                )

            if both_slopes_down and rsi14 <= _RSI_BEAR_MAX:
                confidence = self._trend_confidence(adx_norm, ema9_slope, rsi14, bull=False)
                return self._make_context(
                    symbol=symbol,
                    regime=MarketRegime.BEAR_TREND,
                    volatility=VolatilityLevel.NORMAL,
                    confidence=confidence,
                    trading_allowed=True,
                    realized_vol=realized_vol,
                    feature_vector=feature_vector,
                )

        # ------------------------------------------------------------------
        # 5. SIDEWAYS — low ADX, no clear direction
        # ------------------------------------------------------------------
        if adx_norm < _ADX_RANGING / 100:
            confidence = min(0.55 + (_ADX_RANGING / 100 - adx_norm) * 3, 0.80)
            return self._make_context(
                symbol=symbol,
                regime=MarketRegime.SIDEWAYS,
                volatility=VolatilityLevel.LOW,
                confidence=round(confidence, 3),
                trading_allowed=True,
                realized_vol=realized_vol,
                feature_vector=feature_vector,
            )

        # ------------------------------------------------------------------
        # 6. UNCERTAIN — mixed signals
        # ------------------------------------------------------------------
        return self._make_context(
            symbol=symbol,
            regime=MarketRegime.UNCERTAIN,
            volatility=VolatilityLevel.NORMAL,
            confidence=0.40,
            trading_allowed=True,
            realized_vol=realized_vol,
            feature_vector=feature_vector,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _trend_confidence(
        self,
        adx_norm: float,
        slope: float,
        rsi: float,
        bull: bool,
    ) -> float:
        """Confidence score [0.5, 0.95] for a trend regime."""
        base = 0.55
        # ADX strength adds up to 0.20
        base += min((adx_norm - _ADX_TRENDING / 100) * 2, 0.20)
        # Slope magnitude adds up to 0.10
        base += min(abs(slope) / (_EMA_SLOPE_BULL * 10) * 0.10, 0.10)
        # RSI conviction: distance from 0.5 adds up to 0.10
        base += min(abs(rsi - 0.5) * 0.2, 0.10)
        return round(min(base, 0.95), 3)

    @staticmethod
    def _make_context(
        symbol: str,
        regime: MarketRegime,
        volatility: VolatilityLevel,
        confidence: float,
        trading_allowed: bool,
        block_reason: str | None = None,
        realized_vol: float | None = None,
        feature_vector: FeatureVector | None = None,
    ) -> RegimeContext:
        ctx = RegimeContext(
            symbol=symbol,
            regime=regime,
            volatility_level=volatility,
            confidence=confidence,
            trading_allowed=trading_allowed,
            block_reason=block_reason,
            realized_vol_1h=realized_vol,
        )
        log.debug(
            "regime.classified",
            symbol=symbol,
            regime=regime.value,
            confidence=confidence,
            trading_allowed=trading_allowed,
        )
        return ctx
