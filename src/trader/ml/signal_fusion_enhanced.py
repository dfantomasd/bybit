"""УСИЛЕННАЯ умная комбинация сигналов.

Вместо простого голосования, система:
1. Использует attention-подобный механизм
2. Учитывает корреляции между сигналами
3. Адаптирует веса по рыночному режиму
4. Анализирует сигнальные комбинации (какие работают вместе)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available, using simple numpy-based models")
    from trader.ml.simple_models import SimpleClassifier, SimpleEnsembleRegressor

    XGBRegressor = SimpleEnsembleRegressor
    XGBClassifier = SimpleClassifier


@dataclass
class SignalContextEnhanced:
    """Расширенный контекст для фьюжена сигналов."""

    # === ПЯТЬ ОСНОВНЫХ СИГНАЛОВ ===
    signal_ma_crossover: float  # -1, 0, +1
    signal_rsi: float
    signal_macd: float
    signal_breakout: float
    signal_volume: float

    # === УВЕРЕННОСТЬ КАЖДОГО СИГНАЛА ===
    confidence_ma: float  # 0-1
    confidence_rsi: float
    confidence_macd: float
    confidence_breakout: float
    confidence_volume: float

    # === КОРРЕЛЯЦИИ МЕЖДУ СИГНАЛАМИ ===
    ma_rsi_agreement: float  # -1 to 1, согласованы ли друг с другом
    ma_macd_agreement: float
    rsi_macd_agreement: float
    breakout_volume_agreement: float

    # === КОНТЕКСТ РЫНКА ===
    market_regime: str  # "trend", "sideways", "volatile"
    volatility_pct: float
    recent_win_rate: float
    recent_consecutive_wins: int
    recent_consecutive_losses: int

    # === ИСТОРИЯ СИГНАЛОВ ===
    ma_recent_accuracy: float  # Насколько часто MA давал правильные сигналы
    rsi_recent_accuracy: float
    macd_recent_accuracy: float
    breakout_recent_accuracy: float
    volume_recent_accuracy: float

    # === КОНФЛИКТЫ ===
    signal_conflict_count: int  # Сколько сигналов противоречат друг другу
    strongest_signal_consensus: float  # Процент сигналов в одну сторону


class SignalFusionEnhanced:
    """Умное объединение сигналов с attention и адаптивностью."""

    def __init__(self) -> None:
        self.outcome_model: XGBRegressor | None = None  # Предсказывает профит
        self.confidence_model: XGBRegressor | None = None  # Предсказывает уверенность
        self.ensemble_weights: np.ndarray | None = None  # Веса сигналов

        self.signal_names = ["MA", "RSI", "MACD", "Breakout", "Volume"]
        self.min_training_samples = 200

        # Режимные веса (заполняются при обучении)
        self.regime_weights = {
            "trend": None,
            "sideways": None,
            "volatile": None,
        }

    async def train(self, training_data: list[dict]) -> None:
        """Обучить ДВЕ модели: исход + уверенность."""
        if not XGBOOST_AVAILABLE or len(training_data) < self.min_training_samples:
            return

        try:
            x_list = []
            y_outcomes = []
            y_confidence = []

            for record in training_data:
                context = record.get("context")
                if not context:
                    continue

                # 22 признака (вместо 15 в базовой версии)
                x = np.array(
                    [
                        # Сигналы (5)
                        context.signal_ma_crossover,
                        context.signal_rsi,
                        context.signal_macd,
                        context.signal_breakout,
                        context.signal_volume,
                        # Уверенности (5)
                        context.confidence_ma,
                        context.confidence_rsi,
                        context.confidence_macd,
                        context.confidence_breakout,
                        context.confidence_volume,
                        # Корреляции (4)
                        context.ma_rsi_agreement,
                        context.ma_macd_agreement,
                        context.rsi_macd_agreement,
                        context.breakout_volume_agreement,
                        # Режим (1 из 3)
                        1.0 if context.market_regime == "trend" else 0.0,
                        1.0 if context.market_regime == "sideways" else 0.0,
                        1.0 if context.market_regime == "volatile" else 0.0,
                        # История сигналов (5)
                        context.ma_recent_accuracy,
                        context.rsi_recent_accuracy,
                        context.macd_recent_accuracy,
                        context.breakout_recent_accuracy,
                        context.volume_recent_accuracy,
                    ],
                    dtype=np.float32,
                )

                x_list.append(x)

                # Целевые переменные
                outcome = 1.0 if record.get("was_profitable", False) else 0.0
                y_outcomes.append(outcome)

                # Уверенность = вероятность выигрыша
                confidence = record.get("expected_confidence", 0.5)
                y_confidence.append(confidence)

            if len(x_list) < self.min_training_samples:
                return

            x_train = np.array(x_list, dtype=np.float32)
            y_outcomes_train = np.array(y_outcomes, dtype=np.float32)
            y_confidence_train = np.array(y_confidence, dtype=np.float32)

            # Модель 1: Предсказание исхода (прибыль/убыток)
            logger.info("💰 Обучаю модель ИСХОДА сигналов...")
            self.outcome_model = XGBRegressor(
                max_depth=7,
                learning_rate=0.1,
                n_estimators=200,
                random_state=42,
                subsample=0.7,
                colsample_bytree=0.7,
            )
            self.outcome_model.fit(x_train, y_outcomes_train)
            outcome_r2 = self.outcome_model.score(x_train, y_outcomes_train)
            logger.info(f"✅ Модель исхода: R² = {outcome_r2:.3f}")

            # Модель 2: Предсказание уверенности
            logger.info("🎯 Обучаю модель УВЕРЕННОСТИ...")
            self.confidence_model = XGBRegressor(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=150,
                random_state=42,
            )
            self.confidence_model.fit(x_train, y_confidence_train)
            confidence_r2 = self.confidence_model.score(x_train, y_confidence_train)
            logger.info(f"✅ Модель уверенности: R² = {confidence_r2:.3f}")

            # Извлечь веса из важности признаков (первые 5 = сигналы)
            self._update_signal_weights(x_train, y_outcomes_train)

            logger.info(f"🎯 Обучение сигналов завершено: {len(x_list)} сэмплов")

        except Exception as e:
            logger.error(f"signal_fusion_training_failed: {e}")

    async def fuse_signals(
        self,
        context: SignalContextEnhanced,
    ) -> dict:
        """Объединить сигналы используя attention и ML.

        Возвращает:
        {
            'final_signal': float,  # -1 to +1
            'confidence': float,  # 0-1
            'expected_profit_bps': float,
            'signal_weights': dict,  # Вес каждого сигнала
            'attention_scores': dict,  # Attention для каждого сигнала
            'recommendation': str,
            'explanation': str,
        }
        """
        if self.outcome_model is None or self.confidence_model is None:
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
                    context.ma_rsi_agreement,
                    context.ma_macd_agreement,
                    context.rsi_macd_agreement,
                    context.breakout_volume_agreement,
                    1.0 if context.market_regime == "trend" else 0.0,
                    1.0 if context.market_regime == "sideways" else 0.0,
                    1.0 if context.market_regime == "volatile" else 0.0,
                    context.ma_recent_accuracy,
                    context.rsi_recent_accuracy,
                    context.macd_recent_accuracy,
                    context.breakout_recent_accuracy,
                    context.volume_recent_accuracy,
                ],
                dtype=np.float32,
            ).reshape(1, -1)

            # 1. ПРЕДСКАЗАТЬ ИСХОД
            expected_outcome = float(self.outcome_model.predict(x)[0])
            expected_outcome = max(0.0, min(1.0, expected_outcome))

            # 2. ПРЕДСКАЗАТЬ УВЕРЕННОСТЬ
            predicted_confidence = float(self.confidence_model.predict(x)[0])
            predicted_confidence = max(0.0, min(1.0, predicted_confidence))

            # 3. ATTENTION: вычислить веса для каждого сигнала
            attention_scores = self._compute_attention(context)

            # 4. ФИНАЛЬНЫЙ СИГНАЛ с attention-взвешиванием
            signals = np.array(
                [
                    context.signal_ma_crossover,
                    context.signal_rsi,
                    context.signal_macd,
                    context.signal_breakout,
                    context.signal_volume,
                ]
            )

            # Применить attention веса
            weighted_signals = signals * np.array(
                [
                    attention_scores["MA"],
                    attention_scores["RSI"],
                    attention_scores["MACD"],
                    attention_scores["Breakout"],
                    attention_scores["Volume"],
                ]
            )

            final_signal = np.sum(weighted_signals) / np.sum(
                [
                    attention_scores["MA"],
                    attention_scores["RSI"],
                    attention_scores["MACD"],
                    attention_scores["Breakout"],
                    attention_scores["Volume"],
                ]
            )

            # 5. ОЖИДАЕМАЯ ПРИБЫЛЬ
            expected_profit_bps = (expected_outcome - 0.5) * 200 + predicted_confidence * 100

            # 6. РЕКОМЕНДАЦИЯ
            recommendation = self._get_trading_recommendation(final_signal, predicted_confidence, context)

            # 7. ОБЪЯСНЕНИЕ
            explanation = self._build_explanation(context, attention_scores, predicted_confidence)

            return {
                "final_signal": final_signal,
                "confidence": predicted_confidence,
                "expected_profit_bps": expected_profit_bps,
                "signal_weights": {
                    "MA": float(attention_scores["MA"]),
                    "RSI": float(attention_scores["RSI"]),
                    "MACD": float(attention_scores["MACD"]),
                    "Breakout": float(attention_scores["Breakout"]),
                    "Volume": float(attention_scores["Volume"]),
                },
                "attention_scores": attention_scores,
                "recommendation": recommendation,
                "explanation": explanation,
            }

        except Exception as e:
            logger.error(f"signal_fusion_inference_failed: {e}")
            return self._simple_voting(context)

    def _compute_attention(self, context: SignalContextEnhanced) -> dict:
        """Вычислить attention веса для каждого сигнала."""
        # Базовые веса на основе уверенности
        base_weights = {
            "MA": context.confidence_ma,
            "RSI": context.confidence_rsi,
            "MACD": context.confidence_macd,
            "Breakout": context.confidence_breakout,
            "Volume": context.confidence_volume,
        }

        # Корректировка на основе исторической точности
        accuracy_weights = {
            "MA": context.ma_recent_accuracy,
            "RSI": context.rsi_recent_accuracy,
            "MACD": context.macd_recent_accuracy,
            "Breakout": context.breakout_recent_accuracy,
            "Volume": context.volume_recent_accuracy,
        }

        # Режимные веса
        regime_factor = {
            "trend": {"MA": 1.2, "RSI": 0.8, "MACD": 1.0, "Breakout": 1.3, "Volume": 0.9},
            "sideways": {"MA": 0.7, "RSI": 1.3, "MACD": 1.0, "Breakout": 0.6, "Volume": 1.0},
            "volatile": {"MA": 0.8, "RSI": 1.1, "MACD": 0.9, "Breakout": 0.7, "Volume": 1.2},
        }.get(context.market_regime, {"MA": 1.0, "RSI": 1.0, "MACD": 1.0, "Breakout": 1.0, "Volume": 1.0})

        # Комбинировать веса
        final_weights = {}
        for signal in ["MA", "RSI", "MACD", "Breakout", "Volume"]:
            weight = (base_weights[signal] + accuracy_weights[signal]) / 2 * regime_factor[signal]
            final_weights[signal] = weight

        # Нормировать
        total = sum(final_weights.values())
        if total > 0:
            final_weights = {k: v / total for k, v in final_weights.items()}

        return final_weights

    @staticmethod
    def _get_trading_recommendation(signal: float, confidence: float, context: SignalContextEnhanced) -> str:
        """Торговая рекомендация."""
        if confidence < 0.45:
            return "❓ Уверенность низкая, жди"

        if context.signal_conflict_count > 2:
            return "⚠️ Сигналы противоречивы, рискованно"

        if signal > 0.6:
            return "🟢 STRONG BUY - все сигналы за"
        elif signal > 0.3:
            return "🟢 BUY - большинство сигналов за"
        elif signal < -0.6:
            return "🔴 STRONG SELL - все сигналы против"
        elif signal < -0.3:
            return "🔴 SELL - большинство против"
        else:
            return "🟡 NEUTRAL - неясно"

    @staticmethod
    def _build_explanation(context: SignalContextEnhanced, weights: dict, confidence: float) -> str:
        """Объяснение решения."""
        strongest = max(weights, key=weights.get)
        strongest_weight = weights[strongest]

        return (
            f"Сильнейший сигнал: {strongest} ({strongest_weight:.0%}). "
            f"Уверенность: {confidence:.0%}. "
            f"Режим: {context.market_regime}. "
            f"Конфликты: {context.signal_conflict_count}"
        )

    def _update_signal_weights(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Извлечь веса из важности признаков модели."""
        if self.outcome_model is None:
            return

        importances = self.outcome_model.feature_importances_
        signal_importances = importances[:5]

        total = sum(signal_importances)
        if total > 0:
            self.ensemble_weights = signal_importances / total

        logger.debug(f"Signal weights: {dict(zip(self.signal_names, self.ensemble_weights, strict=False))}")

    @staticmethod
    def _simple_voting(context: SignalContextEnhanced) -> dict:
        """Простое голосование как резервный вариант."""
        signals = [
            context.signal_ma_crossover,
            context.signal_rsi,
            context.signal_macd,
            context.signal_breakout,
            context.signal_volume,
        ]

        avg_signal = np.mean(signals)
        confidence = 0.4

        return {
            "final_signal": avg_signal,
            "confidence": confidence,
            "expected_profit_bps": 0.0,
            "signal_weights": {"MA": 0.2, "RSI": 0.2, "MACD": 0.2, "Breakout": 0.2, "Volume": 0.2},
            "attention_scores": {"MA": 0.2, "RSI": 0.2, "MACD": 0.2, "Breakout": 0.2, "Volume": 0.2},
            "recommendation": "Модель не обучена, простое голосование",
            "explanation": "Используется резервное голосование",
        }
