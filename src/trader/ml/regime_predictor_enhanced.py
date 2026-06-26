"""УСИЛЕННЫЙ предсказатель рыночного режима.

Вместо просто классификации текущего режима:
1. Предсказывает ПЕРЕХОДЫ между режимами (1-2 часа вперёд)
2. Даёт Multi-step вероятности (что произойдёт через 5, 15, 60 минут)
3. Анализирует энтропию (мера хаоса в рынке)
4. Определяет фазу тренда (начало/развитие/конец)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available, using simple numpy-based models")
    from trader.ml.simple_models import SimpleEnsembleRegressor, SimpleClassifier
    XGBRegressor = SimpleEnsembleRegressor
    XGBClassifier = SimpleClassifier


@dataclass
class RegimeFeaturesEnhanced:
    """Расширенные признаки для предсказания режима."""

    # === ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ ===
    rsi: float  # 0-100
    macd_histogram: float  # Гистограмма MACD
    macd_signal_distance: float  # Расстояние MACD-Signal
    bb_position: float  # Позиция цены в Bollinger (0-1)
    bb_width_pct: float  # Ширина лент

    # === ВОЛАТИЛЬНОСТЬ ===
    realized_vol_pct: float  # Реальная волатильность
    volatility_regime: int  # 0=low, 1=med, 2=high, 3=extreme
    volatility_trend: float  # Растёт или падает (-1 to 1)
    volatility_acceleration: float  # Ускоряет ли волатильность

    # === НАПРАВЛЕНИЕ И СИЛА ===
    trend_direction: float  # -1 to 1
    trend_strength: float  # Сила тренда 0-1
    adx: float  # ADX индекс (0-100)
    di_plus: float  # +DI (сила восходящего тренда)
    di_minus: float  # -DI (сила нисходящего тренда)

    # === ЭНТРОПИЯ И ХАОС ===
    market_entropy: float  # 0-1, мера хаоса (0=упорядоченно, 1=хаос)
    price_acceleration: float  # Цена ускоряет или замедляет
    momentum_strength: float  # Сила момента

    # === ОБЪЁМЫ И ДИСБАЛАНС ===
    volume_profile_concentration: float  # 0-1, сконцентрирован объём или разбросан
    buy_sell_imbalance: float  # -1 to 1, больше покупателей или продавцов
    volume_trend: float  # Объёмы растут или падают

    # === ИСТОРИЯ ===
    recent_returns_std: float  # Стандартное отклонение недавних свечей
    recent_returns_skew: float  # Асимметрия (левый/правый хвост)
    regime_duration_candles: int  # Сколько свечей в текущем режиме


@dataclass
class RegimePredictionMultiStep:
    """Мульти-шаг предсказание режима."""

    current_regime: str  # TREND_UP, TREND_DOWN, SIDEWAYS, VOLATILE
    confidence_current: float  # 0-1

    # Вероятности через 5, 15, 60 минут
    next_5m_regime: str
    prob_next_5m: float
    transition_confidence_5m: float  # Насколько мы уверены в переходе

    next_15m_regime: str
    prob_next_15m: float
    transition_confidence_15m: float

    next_60m_regime: str
    prob_next_60m: float
    transition_confidence_60m: float

    # Анализ
    market_entropy: float  # 0-1, мера хаоса
    trend_phase: str  # "EARLY", "DEVELOPING", "MATURE", "ENDING"
    entropy_trend: str  # "ORGANIZING" (формируется тренд) или "DETERIORATING" (разрушается)

    recommendation: str  # Торговая рекомендация


class RegimePredictorEnhanced:
    """Предсказывает режимы и их переходы."""

    def __init__(self):
        self.regime_model: Optional[XGBClassifier] = None  # Текущий режим
        self.transition_model: Optional[XGBClassifier] = None  # Переходы (5m/15m/60m)
        self.entropy_model: Optional[XGBRegressor] = None  # Предсказывает энтропию

        self.regime_names = ["TREND_UP", "TREND_DOWN", "SIDEWAYS", "VOLATILE"]
        self.min_training_samples = 300  # Больше для качества

        # HMM-подобные матрицы переходов
        self.transition_matrix = None

    async def train(self, training_data: list[dict]) -> None:
        """Обучить ТРИ модели: режим + переходы + энтропия."""
        if not XGBOOST_AVAILABLE or len(training_data) < self.min_training_samples:
            return

        try:
            x_list = []
            y_regimes = []
            y_next_5m = []
            y_next_15m = []
            y_next_60m = []
            y_entropy = []

            for record in training_data:
                features = record.get("features")
                if not features:
                    continue

                # 17 признаков
                x = np.array([
                    features.rsi,
                    features.macd_histogram,
                    features.macd_signal_distance,
                    features.bb_position,
                    features.bb_width_pct,
                    features.realized_vol_pct,
                    features.volatility_trend,
                    features.volatility_acceleration,
                    features.trend_direction,
                    features.trend_strength,
                    features.adx,
                    features.di_plus,
                    features.di_minus,
                    features.market_entropy,
                    features.price_acceleration,
                    features.momentum_strength,
                    features.buy_sell_imbalance,
                ], dtype=np.float32)

                x_list.append(x)

                # Текущий режим
                y_regimes.append(record.get("current_regime_class", 2))

                # Будущие режимы
                y_next_5m.append(record.get("next_5m_regime_class", 2))
                y_next_15m.append(record.get("next_15m_regime_class", 2))
                y_next_60m.append(record.get("next_60m_regime_class", 2))

                # Энтропия
                y_entropy.append(record.get("market_entropy", 0.5))

            if len(x_list) < self.min_training_samples:
                return

            x_train = np.array(x_list, dtype=np.float32)
            y_regimes_train = np.array(y_regimes, dtype=np.int32)
            y_next_5m_train = np.array(y_next_5m, dtype=np.int32)
            y_next_15m_train = np.array(y_next_15m, dtype=np.int32)
            y_next_60m_train = np.array(y_next_60m, dtype=np.int32)
            y_entropy_train = np.array(y_entropy, dtype=np.float32)

            # Модель 1: Текущий режим
            logger.info("🔄 Обучаю модель ТЕКУЩЕГО РЕЖИМА...")
            self.regime_model = XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=150,
                random_state=42,
                objective="multi:softmax",
                num_class=4,
                subsample=0.8,
            )
            self.regime_model.fit(x_train, y_regimes_train)
            regime_acc = self.regime_model.score(x_train, y_regimes_train)
            logger.info(f"✅ Модель режима: accuracy = {regime_acc:.3f}")

            # Модель 2: Переходы (multi-step)
            logger.info("🚀 Обучаю модель ПЕРЕХОДОВ (5m/15m/60m)...")

            # Объединяем все переходы в одну обучающую выборку
            y_all_transitions = np.concatenate([
                y_next_5m_train,
                y_next_15m_train,
                y_next_60m_train,
            ])
            x_transitions_train = np.vstack([x_train, x_train, x_train])

            self.transition_model = XGBClassifier(
                max_depth=7,
                learning_rate=0.1,
                n_estimators=150,
                random_state=42,
                objective="multi:softmax",
                num_class=4,
                subsample=0.75,
                scale_pos_weight=1.2,
            )
            self.transition_model.fit(x_transitions_train, y_all_transitions)
            transition_acc = self.transition_model.score(x_transitions_train, y_all_transitions)
            logger.info(f"✅ Модель переходов: accuracy = {transition_acc:.3f}")

            # Модель 3: Энтропия
            logger.info("🌀 Обучаю модель ЭНТРОПИИ...")
            self.entropy_model = XGBRegressor(
                max_depth=5,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42,
            )
            self.entropy_model.fit(x_train, y_entropy_train)
            entropy_r2 = self.entropy_model.score(x_train, y_entropy_train)
            logger.info(f"✅ Модель энтропии: R² = {entropy_r2:.3f}")

            # Построить матрицу переходов (HMM-подобная)
            self._build_transition_matrix(y_regimes_train, y_next_5m_train)

            logger.info(f"🎯 Обучение режимов завершено: {len(x_list)} сэмплов")

        except Exception as e:
            logger.error(f"regime_training_failed: {e}")

    async def predict(self, features: RegimeFeaturesEnhanced) -> RegimePredictionMultiStep:
        """Предсказать режим и его будущие переходы."""
        if self.regime_model is None:
            return self._get_fallback_prediction()

        try:
            x = np.array([
                features.rsi,
                features.macd_histogram,
                features.macd_signal_distance,
                features.bb_position,
                features.bb_width_pct,
                features.realized_vol_pct,
                features.volatility_trend,
                features.volatility_acceleration,
                features.trend_direction,
                features.trend_strength,
                features.adx,
                features.di_plus,
                features.di_minus,
                features.market_entropy,
                features.price_acceleration,
                features.momentum_strength,
                features.buy_sell_imbalance,
            ], dtype=np.float32).reshape(1, -1)

            # 1. ТЕКУЩИЙ РЕЖИМ
            current_class = int(self.regime_model.predict(x)[0])
            current_proba = float(np.max(self.regime_model.predict_proba(x)))
            current_regime = self.regime_names[current_class]

            # 2. ПЕРЕХОДЫ
            transition_proba = self.transition_model.predict_proba(x)[0]

            next_5m_class = np.argmax(transition_proba[:4])
            next_5m_regime = self.regime_names[next_5m_class]
            prob_next_5m = float(transition_proba[next_5m_class])

            next_15m_class = np.argmax(transition_proba[4:8]) if len(transition_proba) > 4 else next_5m_class
            next_15m_regime = self.regime_names[next_15m_class]
            prob_next_15m = float(transition_proba[next_15m_class] if len(transition_proba) > 4 else transition_proba[next_5m_class])

            next_60m_class = np.argmax(transition_proba[8:12]) if len(transition_proba) > 8 else next_15m_class
            next_60m_regime = self.regime_names[next_60m_class]
            prob_next_60m = float(transition_proba[next_60m_class] if len(transition_proba) > 8 else transition_proba[next_15m_class])

            # 3. ЭНТРОПИЯ
            market_entropy = float(self.entropy_model.predict(x)[0])
            market_entropy = max(0.0, min(1.0, market_entropy))

            # 4. ФАЗОВЫЙ АНАЛИЗ
            trend_phase = self._classify_phase(features, market_entropy)

            # 5. ЭНТРОПИЯ ТРЕНД
            entropy_trend = "ORGANIZING" if features.volatility_trend < 0 else "DETERIORATING"

            # 6. РЕКОМЕНДАЦИЯ
            recommendation = self._get_recommendation(
                current_regime, current_proba, market_entropy, trend_phase
            )

            # Уверенность в переходах
            transition_confidence_5m = abs(prob_next_5m - (1/4))  # Чем дальше от 0.25, тем уверённее
            transition_confidence_15m = abs(prob_next_15m - (1/4))
            transition_confidence_60m = abs(prob_next_60m - (1/4))

            return RegimePredictionMultiStep(
                current_regime=current_regime,
                confidence_current=current_proba,
                next_5m_regime=next_5m_regime,
                prob_next_5m=prob_next_5m,
                transition_confidence_5m=transition_confidence_5m,
                next_15m_regime=next_15m_regime,
                prob_next_15m=prob_next_15m,
                transition_confidence_15m=transition_confidence_15m,
                next_60m_regime=next_60m_regime,
                prob_next_60m=prob_next_60m,
                transition_confidence_60m=transition_confidence_60m,
                market_entropy=market_entropy,
                trend_phase=trend_phase,
                entropy_trend=entropy_trend,
                recommendation=recommendation,
            )

        except Exception as e:
            logger.error(f"regime_inference_failed: {e}")
            return self._get_fallback_prediction()

    @staticmethod
    def _classify_phase(features: RegimeFeaturesEnhanced, entropy: float) -> str:
        """Определить фазу тренда."""
        if entropy > 0.6:
            return "CHAOTIC"  # Хаос, нет тренда

        adx = features.adx
        if adx < 20:
            return "EARLY"  # Начало тренда (слабый ADX)
        elif adx < 40:
            return "DEVELOPING"  # Развивается
        elif adx < 60:
            return "MATURE"  # Зрелый тренд
        else:
            return "ENDING"  # Заканчивается (экстремально высокий ADX)

    @staticmethod
    def _get_recommendation(
        regime: str,
        confidence: float,
        entropy: float,
        phase: str,
    ) -> str:
        """Торговая рекомендация."""
        if confidence < 0.4:
            return "❓ Режим неясен, жди ясности"

        if entropy > 0.6:
            return "🌀 Высокий хаос, не торгуй сейчас"

        if regime == "TREND_UP":
            if phase == "EARLY":
                return "🟢 ТРЕНД НАЧИНАЕТСЯ - входи осторожнее"
            elif phase in ["DEVELOPING", "MATURE"]:
                return "🟢 СИЛЬНЫЙ ТРЕНД ВВЕРХ - торгуй активно!"
            else:
                return "⚠️ ТРЕНД ЗАКАНЧИВАЕТСЯ - готовься к развороту"

        elif regime == "TREND_DOWN":
            if phase == "EARLY":
                return "🔴 Начинается падение - жди подтверждения"
            elif phase in ["DEVELOPING", "MATURE"]:
                return "🔴 СИЛЬНЫЙ ТРЕНД ВНИЗ - учись шортить"
            else:
                return "⚠️ Падение ослабевает - возможен отскок"

        elif regime == "SIDEWAYS":
            return "🟡 БОКОВИК - торгуй с полов/потолков, уменьши размер"

        else:  # VOLATILE
            return "⚫ ВОЛАТИЛЬНОСТЬ - жди решения, уменьши позицию"

    def _build_transition_matrix(self, current: np.ndarray, next_regime: np.ndarray) -> None:
        """Построить матрицу переходов (для HMM-подхода)."""
        try:
            matrix = np.zeros((4, 4))
            for c, n in zip(current, next_regime):
                matrix[int(c)][int(n)] += 1

            # Нормировать по строкам
            row_sums = matrix.sum(axis=1, keepdims=True)
            self.transition_matrix = matrix / (row_sums + 1e-6)

            logger.debug(f"Transition matrix built:\n{self.transition_matrix}")
        except Exception as e:
            logger.error(f"Transition matrix failed: {e}")

    @staticmethod
    def _get_fallback_prediction() -> RegimePredictionMultiStep:
        """Дефолтное предсказание."""
        return RegimePredictionMultiStep(
            current_regime="SIDEWAYS",
            confidence_current=0.3,
            next_5m_regime="SIDEWAYS",
            prob_next_5m=0.25,
            transition_confidence_5m=0.0,
            next_15m_regime="SIDEWAYS",
            prob_next_15m=0.25,
            transition_confidence_15m=0.0,
            next_60m_regime="SIDEWAYS",
            prob_next_60m=0.25,
            transition_confidence_60m=0.0,
            market_entropy=0.5,
            trend_phase="CHAOTIC",
            entropy_trend="UNKNOWN",
            recommendation="Модель не обучена",
        )
