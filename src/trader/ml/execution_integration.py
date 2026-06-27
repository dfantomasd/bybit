"""Integration of ML predictions with ExecutionEngine.

Bridges ML controller predictions into the execution workflow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MLEnhancedContext:
    """ML-enhanced context for execution decisions."""

    # Kelly sizing
    kelly_fraction: Decimal
    fractional_kelly: Decimal

    # Regime and confidence
    current_regime: str
    regime_confidence: float

    # Signal and probability
    fused_signal: float
    signal_confidence: float

    # Spread analysis
    predicted_spread_bps: float
    spread_risk: float

    # Risk management
    optimal_stop_pct: float
    emergency_stop_pct: float

    # Entry/Exit optimization
    entry_offset_bps: float
    tp_distance_bps: float

    # Metadata
    all_models_trained: bool


class ExecutionMLIntegrator:
    """Integrates ML predictions into execution workflow."""

    def __init__(self, ml_controller: Any) -> None:
        """Initialize with ML controller.

        Args:
            ml_controller: UnifiedMLController instance
        """
        self.ml_controller = ml_controller

    async def enrich_execution_context(
        self,
        proposal: Any,
        recent_trades: list[dict[str, Any]],
        current_price: Decimal,
        current_regime: str = "SIDEWAYS",
        current_volatility: float = 1.5,
    ) -> MLEnhancedContext:
        """Enrich execution context with ML predictions.

        Args:
            proposal: TradeProposal
            recent_trades: List of recent closed trades
            current_price: Current price
            current_regime: Current market regime
            current_volatility: Current volatility %

        Returns:
            MLEnhancedContext with all ML predictions
        """
        if self.ml_controller is None:
            return self._fallback_context()

        try:
            from trader.ml.feature_extractor import FeatureExtractor

            extractor = FeatureExtractor()

            # Extract features for each model
            kelly_features = extractor.extract_kelly_features(
                recent_trades=recent_trades,
                current_volatility=current_volatility,
            )

            regime_features = extractor.extract_regime_features(
                realized_vol_pct=current_volatility,
            )

            signal_context = extractor.extract_signal_context(
                market_regime=current_regime,
                recent_trades=recent_trades,
                volatility_pct=current_volatility,
            )

            spread_features = extractor.extract_spread_features(
                price_volatility_bps=current_volatility * 100,
            )

            stoploss_context = extractor.extract_stoploss_context(
                realized_volatility_pct=current_volatility,
                market_regime=current_regime,
                recent_trades=recent_trades,
            )

            # Get all predictions from ML controller
            predictions = await self.ml_controller.predict_all(
                kelly_features=kelly_features,
                regime_features=regime_features,
                signal_context=signal_context,
                spread_features=spread_features,
                stoploss_context=stoploss_context,
                current_price=current_price,
            )

            # Build enhanced context
            return MLEnhancedContext(
                kelly_fraction=predictions.kelly_fraction,
                fractional_kelly=predictions.fractional_kelly,
                current_regime=predictions.current_regime,
                regime_confidence=predictions.regime_confidence,
                fused_signal=predictions.fused_signal,
                signal_confidence=predictions.signal_confidence,
                predicted_spread_bps=predictions.predicted_spread_bps,
                spread_risk=predictions.spread_risk,
                optimal_stop_pct=predictions.optimal_stop_pct,
                emergency_stop_pct=predictions.emergency_stop_pct,
                entry_offset_bps=0.5,  # Can be optimized
                tp_distance_bps=50.0,  # Can be optimized
                all_models_trained=predictions.all_models_trained,
            )

        except Exception as e:
            logger.error(f"ml_enrichment_failed: {e}")
            return self._fallback_context()

    async def record_trade_outcome(
        self,
        trade_data: dict[str, Any],
        recent_trades: list[dict[str, Any]],
        current_volatility: float = 1.5,
        current_regime: str = "SIDEWAYS",
    ) -> None:
        """Record trade outcome for model training.

        Args:
            trade_data: Closed trade information
            recent_trades: Context for feature extraction
            current_volatility: Market volatility %
            current_regime: Market regime
        """
        if self.ml_controller is None:
            return

        try:
            from trader.ml.feature_extractor import FeatureExtractor

            extractor = FeatureExtractor()

            # Extract features
            kelly_features = extractor.extract_kelly_features(
                recent_trades=recent_trades,
                current_volatility=current_volatility,
            )

            regime_features = extractor.extract_regime_features(
                realized_vol_pct=current_volatility,
            )

            signal_context = extractor.extract_signal_context(
                market_regime=current_regime,
                recent_trades=recent_trades,
                volatility_pct=current_volatility,
            )

            spread_features = extractor.extract_spread_features(
                price_volatility_bps=current_volatility * 100,
            )

            stoploss_context = extractor.extract_stoploss_context(
                realized_volatility_pct=current_volatility,
                market_regime=current_regime,
                recent_trades=recent_trades,
            )

            # Add to training data
            await self.ml_controller.add_training_sample(
                trade_outcome=trade_data,
                kelly_features=kelly_features,
                regime_features=regime_features,
                signal_context=signal_context,
                spread_features=spread_features,
                stoploss_context=stoploss_context,
            )

            # Try to retrain if enough data accumulated
            await self.ml_controller.retrain_models()

            logger.debug(f"trade_outcome_recorded: {trade_data.get('symbol', '?')}")

        except Exception as e:
            logger.error(f"trade_outcome_recording_failed: {e}")

    @staticmethod
    def _fallback_context() -> MLEnhancedContext:
        """Return conservative fallback context when ML unavailable."""
        return MLEnhancedContext(
            kelly_fraction=Decimal("0.10"),
            fractional_kelly=Decimal("0.25"),
            current_regime="SIDEWAYS",
            regime_confidence=0.3,
            fused_signal=0.0,
            signal_confidence=0.3,
            predicted_spread_bps=25.0,
            spread_risk=0.5,
            optimal_stop_pct=2.0,
            emergency_stop_pct=3.0,
            entry_offset_bps=1.0,
            tp_distance_bps=50.0,
            all_models_trained=False,
        )
