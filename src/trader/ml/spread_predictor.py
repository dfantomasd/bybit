"""ML-based spread prediction.

Предсказывает когда спреды будут узкие/широкие, помогает входить в лучший момент.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@dataclass
class SpreadPredictorFeatures:
    """Признаки для предсказания спреда."""

    hour_of_day: int  # 0-23
    day_of_week: int  # 0-6
    recent_volatility_bps: float  # Недавняя волатильность
    avg_spread_last_hour_bps: float  # Средний спред последний час
    trend_direction: int  # -1=вниз, 0=боковик, 1=вверх
    volume_ratio: float  # Текущий объём / средний
    time_since_market_open_hours: float  # Часов с открытия
    is_funding_time: bool  # Близко ли время фондинга


class SpreadPredictor:
    """Предсказывает оптимальные моменты для входа по спреду."""

    def __init__(self, model_dir: str = "/tmp/spread_models"):
        self.model_dir = model_dir
        self.spread_model: Optional[XGBRegressor] = None
        self.last_training_time = datetime.now(UTC)
        self.min_training_samples = 100

    async def train(self, training_data: list[dict]) -> None:
        """Обучить модель на исторических данных спредов."""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available for spread prediction")
            return

        if len(training_data) < self.min_training_samples:
            logger.info(f"Not enough spread data: {len(training_data)}")
            return

        try:
            x_list = []
            y_spreads = []

            for record in training_data:
                features = record.get("features")
                if not features:
                    continue

                x = np.array([
                    features.hour_of_day,
                    features.day_of_week,
                    features.recent_volatility_bps,
                    features.avg_spread_last_hour_bps,
                    features.trend_direction,
                    features.volume_ratio,
                    features.time_since_market_open_hours,
                    float(features.is_funding_time),
                ], dtype=np.float32)

                x_list.append(x)
                y_spreads.append(record.get("actual_spread_bps", 20.0))

            if len(x_list) < self.min_training_samples:
                return

            x_train = np.array(x_list, dtype=np.float32)
            y_spreads_train = np.array(y_spreads, dtype=np.float32)

            self.spread_model = XGBRegressor(
                max_depth=4,
                learning_rate=0.1,
                n_estimators=50,
                random_state=42,
                tree_method="hist",
                objective="reg:squarederror",
            )
            self.spread_model.fit(x_train, y_spreads_train)
            self.last_training_time = datetime.now(UTC)

            logger.info(f"spread_predictor.trained: {len(x_list)} samples")

        except Exception as e:
            logger.error(f"spread_predictor.training_failed: {e}")

    async def predict(self, features: SpreadPredictorFeatures) -> tuple[float, str]:
        """Предсказать спред в текущий момент.

        Возвращает: (predicted_spread_bps, recommendation)
        """
        if self.spread_model is None:
            return 25.0, "Model not trained, using estimate"

        try:
            x = np.array([
                features.hour_of_day,
                features.day_of_week,
                features.recent_volatility_bps,
                features.avg_spread_last_hour_bps,
                features.trend_direction,
                features.volume_ratio,
                features.time_since_market_open_hours,
                float(features.is_funding_time),
            ], dtype=np.float32).reshape(1, -1)

            predicted_spread = float(self.spread_model.predict(x)[0])
            predicted_spread = max(1.0, min(100.0, predicted_spread))

            # Рекомендация на основе предсказания
            if predicted_spread < 15:
                recommendation = "GOOD - спред узкий, хороший момент для входа"
            elif predicted_spread < 25:
                recommendation = "OK - спред нормальный, можно входить"
            elif predicted_spread < 40:
                recommendation = "CAUTION - спред расширился, подожди"
            else:
                recommendation = "WAIT - спред очень широкий, не входи"

            return predicted_spread, recommendation

        except Exception as e:
            logger.error(f"spread_predictor.inference_failed: {e}")
            return features.avg_spread_last_hour_bps, "Prediction failed"

    def should_trade_based_on_spread(
        self,
        current_spread_bps: float,
        max_acceptable_spread_bps: float = 30.0,
    ) -> bool:
        """Простое правило: торговать ли при текущем спреде?"""
        return current_spread_bps <= max_acceptable_spread_bps
