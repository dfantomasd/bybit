"""ML-based умная комбинация сигналов.

Вместо простого голосования, система учится: какой сигнал более надёжен
в разных рыночных условиях.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@dataclass
class SignalContext:
    """Контекст для фьюжена сигналов."""

    signal_ma_crossover: float  # -1 (SELL), 0 (NEUTRAL), 1 (BUY)
    signal_rsi: float  # -1 (SELL), 0 (NEUTRAL), 1 (BUY)
    signal_macd: float  # -1 (SELL), 0 (NEUTRAL), 1 (BUY)
    signal_breakout: float  # -1 (SELL), 0 (NEUTRAL), 1 (BUY)
    signal_volume: float  # -1 (SELL), 0 (NEUTRAL), 1 (BUY)

    confidence_ma: float  # 0-1, насколько уверен MA сигнал
    confidence_rsi: float
    confidence_macd: float
    confidence_breakout: float
    confidence_volume: float

    market_regime: str  # "trend", "sideways", "volatile"
    recent_win_rate: float  # Какой % сделок выигрывал
    volatility_pct: float  # Текущая волатильность


class SignalFusion:
    """Объединяет сигналы умно (не просто голосование)."""

    def __init__(self) -> None:
        self.model: XGBRegressor | None = None
        self.signal_weights = {
            "ma_crossover": 1.0,
            "rsi": 1.0,
            "macd": 1.0,
            "breakout": 1.0,
            "volume": 1.0,
        }
        self.min_training_samples = 100

    async def train(self, training_data: list[dict]) -> None:
        """Обучить модель на исторических сигналах и их результатах."""
        if not XGBOOST_AVAILABLE or len(training_data) < self.min_training_samples:
            return

        try:
            x_list = []
            y_outcomes = []

            for record in training_data:
                context = record.get("context")
                if not context:
                    continue

                # Создать вектор признаков из всех сигналов и их конфиденций
                x = np.array(
                    [
                        context.signal_ma_crossover,
                        context.signal_rsi,
                        context.signal_macd,
                        context.signal_breakout,
                        context.signal_volume,
                        context.confidence_ma,
                        context.confidence_rsi,
                        context.confidence_macd,
                        context.confidence_breakout,
                        context.confidence_volume,
                        1.0 if context.market_regime == "trend" else 0.0,
                        1.0 if context.market_regime == "sideways" else 0.0,
                        1.0 if context.market_regime == "volatile" else 0.0,
                        context.recent_win_rate,
                        context.volatility_pct,
                    ],
                    dtype=np.float32,
                )

                x_list.append(x)
                # Исход: 1 если сделка была прибыльна, 0 если убыточна
                outcome = 1 if record.get("was_profitable", False) else 0
                y_outcomes.append(outcome)

            if len(x_list) < self.min_training_samples:
                return

            x_train = np.array(x_list, dtype=np.float32)
            y_outcomes_train = np.array(y_outcomes, dtype=np.float32)

            self.model = XGBRegressor(
                max_depth=5,
                learning_rate=0.1,
                n_estimators=50,
                random_state=42,
                objective="reg:squarederror",
            )
            self.model.fit(x_train, y_outcomes_train)

            # Извлечь веса признаков (важность)
            self._update_signal_weights()
            logger.info(f"signal_fusion.trained: {len(x_list)} samples")

        except Exception as e:
            logger.error(f"signal_fusion.training_failed: {e}")

    async def fuse_signals(
        self,
        context: SignalContext,
    ) -> tuple[float, float, str]:
        """Объединить все сигналы в итоговую рекомендацию.

        Возвращает: (final_signal, confidence, explanation)
            final_signal: -1 (SELL), 0 (NEUTRAL), 1 (BUY)
            confidence: 0-1
            explanation: почему такое решение
        """
        # Простое голосование (если нет модели)
        if self.model is None:
            return self._simple_voting(context)

        try:
            x = np.array(
                [
                    context.signal_ma_crossover,
                    context.signal_rsi,
                    context.signal_macd,
                    context.signal_breakout,
                    context.signal_volume,
                    context.confidence_ma,
                    context.confidence_rsi,
                    context.confidence_macd,
                    context.confidence_breakout,
                    context.confidence_volume,
                    1.0 if context.market_regime == "trend" else 0.0,
                    1.0 if context.market_regime == "sideways" else 0.0,
                    1.0 if context.market_regime == "volatile" else 0.0,
                    context.recent_win_rate,
                    context.volatility_pct,
                ],
                dtype=np.float32,
            ).reshape(1, -1)

            # Предсказание: какова вероятность успеха?
            success_probability = float(self.model.predict(x)[0])
            success_probability = max(0.0, min(1.0, success_probability))

            # Если вероятность успеха > 0.5, сигнал хороший
            if success_probability > 0.6:
                named_signals = [
                    ("ma_crossover", context.signal_ma_crossover),
                    ("rsi", context.signal_rsi),
                    ("macd", context.signal_macd),
                    ("breakout", context.signal_breakout),
                    ("volume", context.signal_volume),
                ]
                buy_signals = sum(
                    self.signal_weights.get(name, 1.0) for name, s in named_signals if s is not None and s > 0
                )
                sell_signals = sum(
                    self.signal_weights.get(name, 1.0) for name, s in named_signals if s is not None and s < 0
                )

                if buy_signals > sell_signals:
                    final_signal = 1.0
                    explanation = (
                        f"BUY ({buy_signals} сигналов за, {sell_signals} против, уверенность {success_probability:.0%})"
                    )
                elif sell_signals > buy_signals:
                    final_signal = -1.0
                    explanation = f"SELL ({sell_signals} сигналов за, {buy_signals} против, уверенность {success_probability:.0%})"
                else:
                    final_signal = 0.0
                    explanation = "Сигналы противоречивы"
            else:
                final_signal = 0.0
                explanation = f"Вероятность успеха только {success_probability:.0%}, рискованно"

            return final_signal, success_probability, explanation

        except Exception as e:
            logger.error(f"signal_fusion.inference_failed: {e}")
            return self._simple_voting(context)

    def _simple_voting(self, context: SignalContext) -> tuple[float, float, str]:
        """Взвешенное голосование (резервный вариант)."""
        named_signals = [
            ("ma_crossover", context.signal_ma_crossover),
            ("rsi", context.signal_rsi),
            ("macd", context.signal_macd),
            ("breakout", context.signal_breakout),
            ("volume", context.signal_volume),
        ]

        buy_weight = sum(self.signal_weights.get(name, 1.0) for name, s in named_signals if s is not None and s > 0)
        sell_weight = sum(self.signal_weights.get(name, 1.0) for name, s in named_signals if s is not None and s < 0)

        if buy_weight > sell_weight:
            return 1.0, 0.5, "Большинство сигналов BUY (взвешенное голосование)"
        elif sell_weight > buy_weight:
            return -1.0, 0.5, "Большинство сигналов SELL (взвешенное голосование)"
        else:
            return 0.0, 0.3, "Сигналы поделились (взвешенное голосование)"

    def _update_signal_weights(self) -> None:
        """Обновить веса сигналов из важности признаков модели."""
        if self.model is None:
            return

        importances = self.model.feature_importances_
        # Первые 5 признаков - это сами сигналы
        signal_importances = importances[:5]

        total_importance = sum(signal_importances)
        if total_importance > 0:
            self.signal_weights["ma_crossover"] = signal_importances[0] / total_importance
            self.signal_weights["rsi"] = signal_importances[1] / total_importance
            self.signal_weights["macd"] = signal_importances[2] / total_importance
            self.signal_weights["breakout"] = signal_importances[3] / total_importance
            self.signal_weights["volume"] = signal_importances[4] / total_importance

            logger.debug(f"signal_weights.updated: {self.signal_weights}")
