"""Regime-aware strategy adaptation.

Adapts strategy priorities, ML gating, and confidence scoring based on market regime.
Synchronizes LOGREG gating with regime classification to avoid toxic ML predictions
in regimes where the model historically fails.
"""

from __future__ import annotations

import structlog

from trader.domain.enums import MarketRegime
from trader.domain.models import RegimeContext

log = structlog.get_logger(__name__)


class RegimeAwarePrioritizer:
    """Computes strategy priorities based on market regime.

    Each regime has different strategy strengths:
    - TREND: ATR Breakout excels (follows momentum)
    - SIDEWAYS: MeanReversion excels (buys lows, sells highs)
    - VOLATILE: MeanReversion excels (catches extremes)
    - LOW_LIQUIDITY/HIGH_VOLATILITY: All strategies reduced priority
    """

    # Priority mappings: higher number = higher priority
    # Comprehensive regime adaptation for all 11 strategies
    _PRIORITY_TEMPLATES = {
        MarketRegime.BULL_TREND: {
            # Trend-following strategies excel
            "atr_breakout_v1": 6,                    # Best: follows uptrend
            "ema_crossover_v1": 6,                   # Best: EMA slope confirms trend
            "macd_zerocross_v1": 5,                  # Good: momentum confirms trend
            "order_flow_v1": 5,                      # Good: liquidity in trend direction
            "liquidation_hunting_v1": 4,             # Moderate: cascades during trends
            "funding_arbitrage_v1": 3,               # Low: less relevant in strong trend
            "volatility_squeeze_v1": 3,              # Low: squeeze less likely in trend
            "market_making_v1": 2,                   # Low: tight ranges unwelcome
            "statistical_arbitrage_v1": 2,           # Low: reversion trades risky
            "scalp_micro_v1": 4,                     # Moderate: volume high in trends
            "mean_reversion_v1": 2,                  # Weak: counter-trend trades risky
        },
        MarketRegime.BEAR_TREND: {
            # Same as BULL but for downtrend
            "atr_breakout_v1": 6,
            "ema_crossover_v1": 6,
            "macd_zerocross_v1": 5,
            "order_flow_v1": 5,
            "liquidation_hunting_v1": 4,
            "funding_arbitrage_v1": 3,
            "volatility_squeeze_v1": 3,
            "market_making_v1": 2,
            "statistical_arbitrage_v1": 2,
            "scalp_micro_v1": 4,
            "mean_reversion_v1": 2,
        },
        MarketRegime.SIDEWAYS: {
            # Range-trading strategies excel
            "mean_reversion_v1": 6,                  # Best: catches range extremes
            "market_making_v1": 5,                   # Good: tight spreads in ranges
            "statistical_arbitrage_v1": 5,           # Good: mean reversion works
            "macd_zerocross_v1": 4,                  # Moderate: center line bounces
            "scalp_micro_v1": 4,                     # Moderate: volume consistent
            "funding_arbitrage_v1": 4,               # Moderate: funding rates stable
            "volatility_squeeze_v1": 3,              # Low: squeeze less dramatic
            "liquidation_hunting_v1": 2,             # Low: fewer cascades in ranges
            "ema_crossover_v1": 2,                   # Low: EMAs chop in sideways
            "order_flow_v1": 3,                      # Low: flow less directional
            "atr_breakout_v1": 1,                    # Very low: false breakouts common
        },
        MarketRegime.HIGH_VOLATILITY: {
            # Extreme movement strategies excel
            "mean_reversion_v1": 5,                  # Good: catches spikes/dumps
            "liquidation_hunting_v1": 5,             # Good: cascades in volatility
            "atr_breakout_v1": 4,                    # Moderate: large moves help
            "volatility_squeeze_v1": 4,              # Moderate: mean reversion after squeeze
            "scalp_micro_v1": 3,                     # Moderate: volume spikes but risky
            "order_flow_v1": 3,                      # Moderate: panic flow visible
            "macd_zerocross_v1": 2,                  # Low: histogram noise high
            "ema_crossover_v1": 2,                   # Low: whipsaws common
            "market_making_v1": 1,                   # Very low: wide spreads kill profits
            "statistical_arbitrage_v1": 2,           # Low: reversions unpredictable
            "funding_arbitrage_v1": 2,               # Low: funding swings wild
        },
        MarketRegime.LOW_LIQUIDITY: {
            # Conservative strategies only
            "volatility_squeeze_v1": 3,              # Best available
            "ema_crossover_v1": 3,                   # Best available
            "order_flow_v1": 2,                      # Risky: slippage in thin book
            "scalp_micro_v1": 1,                     # Very risky: no volume for scalps
            "mean_reversion_v1": 2,                  # Risky: slippage kills profits
            "atr_breakout_v1": 1,                    # Very risky: ATR too high
            "liquidation_hunting_v1": 1,             # Very risky: cascades kill fills
            "funding_arbitrage_v1": 1,               # Very risky: wide funding jumps
            "market_making_v1": 1,                   # Very risky: no counterparty
            "statistical_arbitrage_v1": 1,           # Very risky: wide spreads
            "macd_zerocross_v1": 2,                  # Risky: few signals
        },
        MarketRegime.UNCERTAIN: {
            # Safe defaults: balanced across all
            "atr_breakout_v1": 4,
            "mean_reversion_v1": 4,
            "macd_zerocross_v1": 4,
            "ema_crossover_v1": 4,
            "order_flow_v1": 3,
            "scalp_micro_v1": 3,
            "volatility_squeeze_v1": 3,
            "liquidation_hunting_v1": 3,
            "funding_arbitrage_v1": 3,
            "market_making_v1": 3,
            "statistical_arbitrage_v1": 3,
        },
    }

    @classmethod
    def compute_priorities(
        cls,
        regime_ctx: RegimeContext | None,
        base_priorities: dict[str, int],
    ) -> dict[str, int]:
        """Adjust base priorities based on regime.

        Args:
            regime_ctx: Current market regime classification
            base_priorities: Default priorities from config

        Returns:
            Modified priorities dict with regime-aware adjustments
        """
        if regime_ctx is None:
            return base_priorities

        regime = regime_ctx.regime
        template = cls._PRIORITY_TEMPLATES.get(regime, cls._PRIORITY_TEMPLATES[MarketRegime.UNCERTAIN])

        # Merge: use template for basic strategies, keep advanced strategies from base
        result = base_priorities.copy()
        result.update(template)

        log.debug(
            "regime.priority_adjusted",
            regime=regime.value,
            regime_confidence=regime_ctx.confidence,
            basic_strategy_priorities=template,
        )

        return result

    @classmethod
    def should_apply_ml_gate(cls, regime_ctx: RegimeContext | None) -> bool:
        """Determine if LOGREG ML gate should be applied.

        LOGREG historically performs poorly in SIDEWAYS and HIGH_VOLATILITY,
        producing many false signals. Disable the gate in these regimes to
        allow strategy signals through without ML filtering.

        Args:
            regime_ctx: Current market regime

        Returns:
            True if LOGREG gate should be active, False to bypass it
        """
        if regime_ctx is None:
            return True  # Apply gate by default

        # Gate works well in trends, struggles in sideways/volatile
        gate_disabled_regimes = {
            MarketRegime.SIDEWAYS,
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.LOW_LIQUIDITY,
        }

        should_apply = regime_ctx.regime not in gate_disabled_regimes

        if not should_apply:
            log.debug(
                "regime.ml_gate_disabled",
                regime=regime_ctx.regime.value,
                reason="logreg_performs_poorly_in_this_regime",
            )

        return should_apply

    @classmethod
    def confidence_adjustment_for_alignment(
        cls,
        proposal_side: str,  # "BUY" or "SELL"
        regime_ctx: RegimeContext | None,
    ) -> float:
        """Get confidence boost/penalty based on signal alignment with regime direction.

        Args:
            proposal_side: "BUY" or "SELL"
            regime_ctx: Current market regime

        Returns:
            Confidence multiplier to apply (e.g., 1.10 = +10%, 0.95 = -5%)
        """
        if regime_ctx is None:
            return 1.0  # No adjustment

        # Buy signals aligned with uptrend = boost
        if proposal_side == "BUY" and regime_ctx.regime == MarketRegime.BULL_TREND:
            return 1.10  # +10% confidence boost

        # Sell signals aligned with downtrend = boost
        if proposal_side == "SELL" and regime_ctx.regime == MarketRegime.BEAR_TREND:
            return 1.10  # +10% confidence boost

        # Counter-trend signals get slight penalty (but not too harsh)
        if proposal_side == "BUY" and regime_ctx.regime == MarketRegime.BEAR_TREND:
            return 0.95  # -5% confidence penalty

        if proposal_side == "SELL" and regime_ctx.regime == MarketRegime.BULL_TREND:
            return 0.95  # -5% confidence penalty

        # In sideways/volatile, alignment matters less
        return 1.0
