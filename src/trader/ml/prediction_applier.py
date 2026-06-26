"""Apply ML predictions to optimize actual trading decisions.

Transforms ML predictions into trading improvements:
1. Entry timing optimization (spread prediction)
2. Position sizing (Kelly prediction)
3. Risk management (stoploss optimization)
4. Strategy selection (regime prediction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class OptimizedTradeParams:
    """Trading parameters optimized by ML."""

    # Entry optimization
    use_limit_order: bool  # True if spread prediction says to wait
    entry_offset_bps: float  # How far from mid to place limit order
    wait_minutes: int  # How long to wait for better spread (0-15)

    # Position sizing
    kelly_fraction: Decimal  # 0.01-0.25
    fractional_kelly: Decimal  # 0.1-1.0 of kelly_fraction
    position_size_adjustment: float  # Multiplier to base size

    # Risk management
    optimal_stop_pct: float  # Primary stop
    emergency_stop_pct: float  # Worst-case stop (CVaR)
    trailing_stop_enabled: bool  # Use trailing stop based on regime

    # Entry confidence
    entry_confidence: float  # 0-1, how confident in entry
    take_signal: bool  # Should we take this signal at all?

    # Metadata
    regime: str
    regime_confidence: float


class PredictionApplier:
    """Apply ML predictions to trading parameters."""

    def __init__(self, ml_controller: Any):
        """Initialize with ML controller.

        Args:
            ml_controller: UnifiedMLController instance
        """
        self.ml_controller = ml_controller

    async def optimize_entry(
        self,
        proposal: Any,
        recent_trades: list[dict[str, Any]],
        current_price: Decimal,
        market_regime: str = "SIDEWAYS",
        current_volatility: float = 1.5,
    ) -> OptimizedTradeParams:
        """Optimize entry parameters using ML predictions.

        Args:
            proposal: TradeProposal
            recent_trades: Recent closed trades for context
            current_price: Current market price
            market_regime: Current market regime
            current_volatility: Current volatility %

        Returns:
            OptimizedTradeParams with ML-optimized settings
        """
        if self.ml_controller is None:
            return self._default_params(Decimal("0.10"), market_regime)

        try:
            from trader.ml.execution_integration import ExecutionMLIntegrator

            integrator = ExecutionMLIntegrator(self.ml_controller)

            # Get ML predictions
            ml_context = await integrator.enrich_execution_context(
                proposal=proposal,
                recent_trades=recent_trades,
                current_price=current_price,
                current_regime=market_regime,
                current_volatility=current_volatility,
            )

            # ==================== ENTRY TIMING ====================
            # Use spread prediction to decide on entry type
            spread_risk = ml_context.spread_risk  # 0-1, 1 = high risk

            if spread_risk > 0.7:
                # High spread risk - use limit order and wait
                use_limit = True
                entry_offset_bps = 5.0  # Place 5 bps inside
                wait_minutes = 5  # Wait up to 5 minutes for better fill
            elif spread_risk > 0.5:
                # Medium risk - limit order with short wait
                use_limit = True
                entry_offset_bps = 2.0
                wait_minutes = 2
            else:
                # Low spread risk - can use market order
                use_limit = False
                entry_offset_bps = 1.0
                wait_minutes = 0

            # ==================== POSITION SIZING ====================
            # Use Kelly predictor for sizing
            kelly_fraction = ml_context.kelly_fraction
            fractional_kelly = ml_context.fractional_kelly

            # Adjust for regime confidence
            regime_adjustment = ml_context.regime_confidence
            position_adjustment = float(regime_adjustment)

            # Further adjust based on signal confidence
            signal_adjustment = ml_context.signal_confidence
            position_adjustment *= signal_adjustment

            # Cap at 0.5 - 1.5x
            position_adjustment = max(0.5, min(1.5, position_adjustment))

            # ==================== RISK MANAGEMENT ====================
            # Use stoploss optimizer for levels
            optimal_stop = ml_context.optimal_stop_pct
            emergency_stop = ml_context.emergency_stop_pct

            # Use trailing stop in trends
            trailing_enabled = ml_context.current_regime in ["TREND_UP", "TREND_DOWN"]

            # ==================== ENTRY DECISION ====================
            # Combine signal confidence and regime confidence
            entry_confidence = (
                ml_context.signal_confidence * 0.6 +
                ml_context.regime_confidence * 0.4
            )

            # Don't take signals below 0.45 confidence
            take_signal = entry_confidence > 0.45

            # Also check if signal is strong enough
            if abs(ml_context.fused_signal) < 0.2 and ml_context.regime == "SIDEWAYS":
                # Sideways market with weak signal = too risky
                take_signal = False

            logger.info(
                "entry.optimized",
                regime=ml_context.current_regime,
                entry_confidence=f"{entry_confidence:.2f}",
                spread_risk=f"{spread_risk:.2f}",
                kelly_fraction=f"{kelly_fraction:.3f}",
                take_signal=take_signal,
            )

            return OptimizedTradeParams(
                use_limit_order=use_limit,
                entry_offset_bps=entry_offset_bps,
                wait_minutes=wait_minutes,
                kelly_fraction=kelly_fraction,
                fractional_kelly=fractional_kelly,
                position_size_adjustment=position_adjustment,
                optimal_stop_pct=optimal_stop,
                emergency_stop_pct=emergency_stop,
                trailing_stop_enabled=trailing_enabled,
                entry_confidence=entry_confidence,
                take_signal=take_signal,
                regime=ml_context.current_regime,
                regime_confidence=ml_context.regime_confidence,
            )

        except Exception as e:
            logger.error(f"entry_optimization_failed: {e}")
            return self._default_params(Decimal("0.10"), market_regime)

    async def should_take_trade(
        self,
        proposal: Any,
        recent_trades: list[dict[str, Any]],
        current_price: Decimal,
        market_regime: str = "SIDEWAYS",
        current_volatility: float = 1.5,
    ) -> tuple[bool, str]:
        """Decide if we should take this trade based on ML analysis.

        Returns:
            (should_take, reason)
        """
        try:
            params = await self.optimize_entry(
                proposal=proposal,
                recent_trades=recent_trades,
                current_price=current_price,
                market_regime=market_regime,
                current_volatility=current_volatility,
            )

            if not params.take_signal:
                return False, f"Low confidence: {params.entry_confidence:.2f}"

            # Additional checks based on regime
            if params.regime == "SIDEWAYS" and params.entry_confidence < 0.55:
                return False, "Sideways market needs higher confidence"

            if params.regime not in ["TREND_UP", "TREND_DOWN"]:
                # Check if signal is really strong in non-trend
                if params.entry_confidence < 0.60:
                    return False, f"Non-trend regime needs >0.60 confidence"

            return True, f"ML optimized: confidence {params.entry_confidence:.2f}"

        except Exception as e:
            logger.error(f"should_take_trade failed: {e}")
            return False, f"ML analysis failed: {e}"

    async def optimize_exit(
        self,
        symbol: str,
        side: str,
        entry_price: Decimal,
        current_price: Decimal,
        current_profit_bps: float,
        market_regime: str = "SIDEWAYS",
        current_volatility: float = 1.5,
    ) -> dict:
        """Optimize take-profit and trailing-stop levels.

        Returns:
            {
                'take_profit_price': Decimal,
                'trailing_stop_enabled': bool,
                'trailing_stop_distance_pct': float,
            }
        """
        if self.ml_controller is None:
            return self._default_exit(entry_price, side)

        try:
            # Get regime prediction for trend strength
            from trader.ml.feature_extractor import FeatureExtractor

            extractor = FeatureExtractor()
            regime_features = extractor.extract_regime_features()

            regime_result = await self.ml_controller.regime.predict(regime_features)

            # Base take-profit target
            base_tp_bps = 50.0  # 0.5% profit target

            # Adjust based on regime
            if regime_result and "TREND" in regime_result.current_regime:
                # In trend, let winners run
                tp_multiplier = 1.5
                trailing_enabled = True
                trailing_distance = 0.7  # Close trailing stop (0.7% behind profit)
            elif regime_result and regime_result.current_regime == "SIDEWAYS":
                # In sideways, take profits quickly
                tp_multiplier = 0.7
                trailing_enabled = False
                trailing_distance = 0.0
            else:
                # Volatile or uncertain - be cautious
                tp_multiplier = 0.9
                trailing_enabled = True
                trailing_distance = 1.0

            # Adjust based on profit already made
            if current_profit_bps > 100:
                # 1%+ profit - should take some profits
                tp_multiplier *= 0.8
                trailing_enabled = True

            # Calculate take-profit price
            tp_bps = base_tp_bps * tp_multiplier
            if side == "LONG":
                tp_price = entry_price * (Decimal("1") + Decimal(str(tp_bps / 10000)))
            else:
                tp_price = entry_price * (Decimal("1") - Decimal(str(tp_bps / 10000)))

            logger.info(
                "exit.optimized",
                regime=regime_result.current_regime if regime_result else "unknown",
                tp_bps=f"{tp_bps:.1f}",
                trailing_enabled=trailing_enabled,
            )

            return {
                'take_profit_price': tp_price,
                'trailing_stop_enabled': trailing_enabled,
                'trailing_stop_distance_pct': trailing_distance,
            }

        except Exception as e:
            logger.error(f"exit_optimization_failed: {e}")
            return self._default_exit(entry_price, side)

    @staticmethod
    def _default_params(kelly: Decimal, regime: str) -> OptimizedTradeParams:
        """Return conservative defaults when ML unavailable."""
        return OptimizedTradeParams(
            use_limit_order=False,
            entry_offset_bps=1.0,
            wait_minutes=0,
            kelly_fraction=kelly,
            fractional_kelly=Decimal("0.25"),
            position_size_adjustment=1.0,
            optimal_stop_pct=2.0,
            emergency_stop_pct=3.0,
            trailing_stop_enabled=False,
            entry_confidence=0.5,
            take_signal=True,
            regime=regime,
            regime_confidence=0.3,
        )

    @staticmethod
    def _default_exit(entry_price: Decimal, side: str) -> dict:
        """Return conservative exit defaults."""
        tp_bps = 50.0
        if side == "LONG":
            tp_price = entry_price * (Decimal("1") + Decimal(str(tp_bps / 10000)))
        else:
            tp_price = entry_price * (Decimal("1") - Decimal(str(tp_bps / 10000)))

        return {
            'take_profit_price': tp_price,
            'trailing_stop_enabled': False,
            'trailing_stop_distance_pct': 0.0,
        }
