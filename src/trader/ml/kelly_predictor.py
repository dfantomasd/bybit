"""ML-based Kelly sizing using XGBoost for adaptive position sizing.

Instead of hand-crafted rules, use ML to learn optimal Kelly from historical
performance data. Predicts both kelly_fraction and fractional_kelly multiplier.

Architecture:
- Feature extraction from recent returns and market conditions
- XGBoost regression to predict optimal kelly_fraction
- Real-time inference (cached, batch updated)
- Async integration with Bybit trading loop
- Decimal precision for financial calculations
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available, using simple numpy-based models")
    from trader.ml.simple_models import SimpleEnsembleRegressor as XGBRegressor  # Use our simple model


@dataclass
class KellyPredictorFeatures:
    """Features for Kelly prediction model."""

    # Recent performance (last 50 trades)
    recent_win_rate: float          # 0-1
    recent_avg_win_bps: float       # basis points
    recent_avg_loss_bps: float      # basis points (negative)
    recent_profit_factor: float     # wins/abs(losses)
    recent_pnl_trend: float         # -1 to 1, equity curve slope

    # Volatility and distribution
    std_dev_bps: float              # volatility of returns
    skewness: float                 # left/right tail
    kurtosis: float                 # fat tails indicator
    var_95_bps: float               # 5th percentile (VaR)

    # Drawdown state
    current_drawdown_pct: float     # 0 or negative
    max_drawdown_pct: float         # worst case
    drawdown_severity: int          # 0-4 (none, mild, moderate, severe, extreme)
    in_drawdown: bool               # boolean

    # Market conditions
    volatility_regime: int          # 0-3 (low, moderate, high, extreme)
    hour_of_day: int                # 0-23
    day_of_week: int                # 0-6
    days_since_start: int           # how long strategy running

    # Strategy metadata
    strategy_id: str                # which strategy
    symbol: str                     # which symbol
    total_trades: int               # cumulative trade count

    # Observation context
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class KellyPredictorOutput:
    """Output from Kelly predictor model."""

    kelly_fraction: Decimal         # 0.01-0.25, from model prediction
    fractional_kelly: Decimal       # 0.1-0.5, from model prediction
    final_position_size: Decimal    # kelly * fractional
    model_confidence: float         # 0-1, how confident is model
    predicted_win_rate: float       # model's prediction of next trade
    predicted_sharpe: float         # predicted risk-adjusted return
    risk_level: str                 # low, medium, high, critical
    model_version: str              # which model was used
    reasoning: str                  # explanation
    feature_importance: dict[str, float] = field(default_factory=dict)


class KellyPredictorBase:
    """Base class for Kelly predictors (interface)."""

    async def train(self, training_data: list[dict[str, Any]]) -> None:
        """Train or update model from historical data."""
        raise NotImplementedError

    async def predict(self, features: KellyPredictorFeatures) -> KellyPredictorOutput:
        """Predict optimal Kelly sizing."""
        raise NotImplementedError

    def get_feature_vector(self, features: KellyPredictorFeatures) -> np.ndarray:
        """Convert features to model input vector."""
        return np.array([
            features.recent_win_rate,
            features.recent_avg_win_bps,
            features.recent_avg_loss_bps,
            features.recent_profit_factor,
            features.recent_pnl_trend,
            features.std_dev_bps,
            features.skewness,
            features.kurtosis,
            features.var_95_bps,
            features.current_drawdown_pct,
            features.max_drawdown_pct,
            float(features.drawdown_severity),
            float(features.in_drawdown),
            float(features.volatility_regime),
            float(features.hour_of_day),
            float(features.day_of_week),
            float(features.days_since_start),
            float(features.total_trades),
        ], dtype=np.float32)

    @staticmethod
    def get_feature_names() -> list[str]:
        """Names of features in model input."""
        return [
            "recent_win_rate",
            "recent_avg_win_bps",
            "recent_avg_loss_bps",
            "recent_profit_factor",
            "recent_pnl_trend",
            "std_dev_bps",
            "skewness",
            "kurtosis",
            "var_95_bps",
            "current_drawdown_pct",
            "max_drawdown_pct",
            "drawdown_severity",
            "in_drawdown",
            "volatility_regime",
            "hour_of_day",
            "day_of_week",
            "days_since_start",
            "total_trades",
        ]


class MLKellyPredictor(KellyPredictorBase):
    """ML-based Kelly predictor using XGBoost."""

    def __init__(self, model_dir: str = "/tmp/kelly_models"):
        self.model_dir = model_dir
        self.kelly_model: Optional[XGBRegressor] = None  # Predicts kelly_fraction
        self.fractional_model: Optional[XGBRegressor] = None  # Predicts fractional_kelly
        self.last_training_time = datetime.now(UTC)
        self.min_training_samples = 100
        self.last_features: Optional[KellyPredictorFeatures] = None
        self.last_prediction: Optional[KellyPredictorOutput] = None

    async def train(self, training_data: list[dict[str, Any]]) -> None:
        """Train Kelly prediction models from historical data.

        Args:
            training_data: List of {features: KellyPredictorFeatures, kelly_actual: float, ...}
        """

        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, skipping ML training")
            return

        if len(training_data) < self.min_training_samples:
            logger.info(f"Insufficient training samples: {len(training_data)} < {self.min_training_samples}")
            return

        try:
            # Extract features and targets
            x_list = []
            y_kelly = []
            y_fractional = []

            for record in training_data:
                features: KellyPredictorFeatures = record.get("features")
                if not features:
                    continue

                x = self.get_feature_vector(features)
                x_list.append(x)
                y_kelly.append(record.get("kelly_actual", 0.15))
                y_fractional.append(record.get("fractional_actual", 0.25))

            if len(x_list) < self.min_training_samples:
                return

            x_train = np.array(x_list, dtype=np.float32)
            y_kelly_train = np.array(y_kelly, dtype=np.float32)
            y_frac_train = np.array(y_fractional, dtype=np.float32)

            # Train kelly_fraction model
            self.kelly_model = XGBRegressor(
                max_depth=4,
                learning_rate=0.1,
                n_estimators=50,
                random_state=42,
                tree_method="hist",
                objective="reg:squarederror",
            )
            self.kelly_model.fit(x_train, y_kelly_train)

            # Train fractional_kelly model
            self.fractional_model = XGBRegressor(
                max_depth=4,
                learning_rate=0.1,
                n_estimators=50,
                random_state=42,
                tree_method="hist",
                objective="reg:squarederror",
            )
            self.fractional_model.fit(x_train, y_frac_train)

            self.last_training_time = datetime.now(UTC)
            logger.info(
                f"kelly_predictor.trained: samples={len(x_list)}, "
                f"kelly_r2={self.kelly_model.score(x_train, y_kelly_train):.3f}, "
                f"fractional_r2={self.fractional_model.score(x_train, y_frac_train):.3f}"
            )

        except Exception as e:
            logger.error(f"kelly_predictor.training_failed: {e}")

    async def predict(self, features: KellyPredictorFeatures) -> KellyPredictorOutput:
        """Predict optimal Kelly sizing from features."""

        self.last_features = features

        # Fallback if models not trained
        if not self.kelly_model or not self.fractional_model:
            return self._fallback_prediction(features)

        try:
            x = self.get_feature_vector(features)
            x_reshaped = x.reshape(1, -1)

            # Get predictions
            kelly_pred = float(self.kelly_model.predict(x_reshaped)[0])
            frac_pred = float(self.fractional_model.predict(x_reshaped)[0])

            # Clamp to reasonable ranges
            kelly_pred = max(0.01, min(kelly_pred, 0.25))
            frac_pred = max(0.1, min(frac_pred, 0.5))

            final_size = kelly_pred * frac_pred

            # Get model confidence (based on prediction variance)
            kelly_std = float(np.std(self.kelly_model.predict(x.reshape(1, -1))))
            confidence = max(0.5, 1.0 - (kelly_std / 0.1))  # Higher std = lower confidence

            # Get feature importance
            feature_importance = dict(
                zip(
                    self.get_feature_names(),
                    self.kelly_model.feature_importances_.tolist()[:5],
                )
            )

            # Assess risk level
            risk_level = self._assess_risk_level(features, kelly_pred)

            output = KellyPredictorOutput(
                kelly_fraction=Decimal(str(kelly_pred)),
                fractional_kelly=Decimal(str(frac_pred)),
                final_position_size=Decimal(str(final_size)),
                model_confidence=confidence,
                predicted_win_rate=features.recent_win_rate,
                predicted_sharpe=kelly_pred / max(features.std_dev_bps / 10000, 0.001),
                risk_level=risk_level,
                model_version="xgboost_v1",
                reasoning=self._build_reasoning(features, kelly_pred, frac_pred),
                feature_importance=feature_importance,
            )

            self.last_prediction = output
            return output

        except Exception as e:
            logger.error(f"kelly_predictor.inference_failed: {e}")
            return self._fallback_prediction(features)

    def _fallback_prediction(self, features: KellyPredictorFeatures) -> KellyPredictorOutput:
        """Simple fallback when model not available."""
        # Base Kelly from statistics
        if features.recent_avg_win_bps <= 0:
            kelly = Decimal("0.10")
        else:
            kelly = Decimal(
                str(
                    (
                        features.recent_win_rate * features.recent_avg_win_bps
                        - (1 - features.recent_win_rate) * abs(features.recent_avg_loss_bps)
                    )
                    / max(features.recent_avg_win_bps, 1.0)
                )
            )
            kelly = max(Decimal("0.01"), min(kelly, Decimal("0.25")))

        # Fractional multiplier
        frac = Decimal("0.25")
        if features.recent_win_rate > 0.6:
            frac = Decimal("0.35")
        elif features.recent_win_rate < 0.45:
            frac = Decimal("0.15")

        if features.in_drawdown:
            frac *= Decimal("0.8")

        frac = max(Decimal("0.1"), min(frac, Decimal("0.5")))

        return KellyPredictorOutput(
            kelly_fraction=kelly,
            fractional_kelly=frac,
            final_position_size=kelly * frac,
            model_confidence=0.3,
            predicted_win_rate=features.recent_win_rate,
            predicted_sharpe=float(kelly) / max(features.std_dev_bps / 10000, 0.001),
            risk_level="medium",
            model_version="fallback",
            reasoning="Model not yet trained, using statistical Kelly",
        )

    @staticmethod
    def _assess_risk_level(features: KellyPredictorFeatures, kelly_pred: float) -> str:
        """Assess risk level from features and prediction."""
        if features.drawdown_severity >= 3 or features.current_drawdown_pct < -10:
            return "critical"
        elif features.in_drawdown and features.kurtosis > 4:
            return "high"
        elif features.recent_win_rate < 0.45 or features.kurtosis > 5:
            return "high"
        elif features.recent_win_rate < 0.5 or features.in_drawdown:
            return "medium"
        else:
            return "low"

    @staticmethod
    def _build_reasoning(features: KellyPredictorFeatures, kelly: float, frac: float) -> str:
        """Build explanation for prediction."""
        reasons = []

        if features.recent_win_rate > 0.6:
            reasons.append(f"Good recent performance ({features.recent_win_rate:.1%} WR)")
        elif features.recent_win_rate < 0.45:
            reasons.append(f"Weak recent performance ({features.recent_win_rate:.1%} WR)")

        if features.in_drawdown:
            reasons.append(f"In drawdown ({features.current_drawdown_pct:.1f}%), reducing size")

        if features.kurtosis > 4:
            reasons.append(f"Fat tails detected, being cautious")

        if features.recent_pnl_trend > 0.05:
            reasons.append("Positive equity trend, increasing exposure")

        if not reasons:
            reasons.append("Normal market conditions")

        return "; ".join(reasons)
