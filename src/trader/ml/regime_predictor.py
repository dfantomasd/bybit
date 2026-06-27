"""Предсказание рыночного режима на 1-2 часа вперёд.

Предсказывает: тренд, боковик или волатильность - ДО того как это произойдёт.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@dataclass
class RegimeFeatures:
    """Признаки для предсказания режима."""

    recent_rsi: float  # 0-100
    macd_histogram: float  # Гистограмма MACD
    bb_position: float  # Где цена в Боллинджер Бэндах (0-1)
    volatility_trend: float  # Волатильность растёт/падает
    volume_trend: float  # Объёмы растут/падают
    time_of_day: int  # 0-23
    recent_returns_std: float  # Стандартное отклонение недавних свечей
    trend_strength: float  # Сила текущего тренда (-1 to 1)


class RegimePredictor:
    """Предсказывает будущие рыночные режимы."""

    def __init__(self) -> None:
        self.model: XGBClassifier | None = None
        self.min_training_samples = 100
        # Классы: 0 = TREND_UP, 1 = TREND_DOWN, 2 = SIDEWAYS, 3 = VOLATILE
        self.regime_names = ["TREND_UP", "TREND_DOWN", "SIDEWAYS", "VOLATILE"]

    async def train(self, training_data: list[dict]) -> None:
        """Обучить на исторических режимах."""
        if not XGBOOST_AVAILABLE or len(training_data) < self.min_training_samples:
            return

        try:
            x_list = []
            y_regimes = []

            for record in training_data:
                features = record.get("features")
                if not features:
                    continue

                x = np.array(
                    [
                        features.recent_rsi,
                        features.macd_histogram,
                        features.bb_position,
                        features.volatility_trend,
                        features.volume_trend,
                        features.time_of_day,
                        features.recent_returns_std,
                        features.trend_strength,
                    ],
                    dtype=np.float32,
                )

                x_list.append(x)
                regime_class = record.get("regime_class", 2)  # Default SIDEWAYS
                y_regimes.append(regime_class)

            if len(x_list) < self.min_training_samples:
                return

            x_train = np.array(x_list, dtype=np.float32)
            y_regimes_train = np.array(y_regimes, dtype=np.int32)

            self.model = XGBClassifier(
                max_depth=5,
                learning_rate=0.1,
                n_estimators=50,
                random_state=42,
                objective="multi:softmax",
                num_class=4,
            )
            self.model.fit(x_train, y_regimes_train)
            logger.info(f"regime_predictor.trained: {len(x_list)} samples")

        except Exception as e:
            logger.error(f"regime_predictor.training_failed: {e}")

    async def predict(
        self,
        features: RegimeFeatures,
    ) -> tuple[str, float]:
        """Предсказать режим на ближайший час.

        Возвращает: (regime_name, confidence)
        """
        if self.model is None:
            return "SIDEWAYS", 0.3  # Консервативный дефолт

        try:
            x = np.array(
                [
                    features.recent_rsi,
                    features.macd_histogram,
                    features.bb_position,
                    features.volatility_trend,
                    features.volume_trend,
                    features.time_of_day,
                    features.recent_returns_std,
                    features.trend_strength,
                ],
                dtype=np.float32,
            ).reshape(1, -1)

            regime_class = int(self.model.predict(x)[0])
            regime_name = self.regime_names[regime_class]

            # Confidence из вероятностей
            proba = self.model.predict_proba(x)[0]
            confidence = float(np.max(proba))

            return regime_name, confidence

        except Exception as e:
            logger.error(f"regime_predictor.inference_failed: {e}")
            return "SIDEWAYS", 0.3

    def get_trading_advice(self, regime: str, confidence: float) -> str:
        """Рекомендация на основе предсказания режима."""
        if confidence < 0.5:
            return f"{regime} (низкая уверенность) - торгуй осторожнее"

        advice_map = {
            "TREND_UP": "🟢 ТРЕНД ВВЕРХ - можно торговать смело, увеличь размер",
            "TREND_DOWN": "🔴 ТРЕНД ВНИЗ - учись торговать шорты, или жди",
            "SIDEWAYS": "🟡 БОКОВИК - уменьши позицию, больше риска",
            "VOLATILE": "⚫ ВОЛАТИЛЬНОСТЬ - уменьши размер, жди ясности",
        }
        return advice_map.get(regime, "Неизвестный режим")
