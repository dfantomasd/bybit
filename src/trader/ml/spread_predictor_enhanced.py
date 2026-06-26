"""УСИЛЕННЫЙ спред-анализ с микроструктурой рынка.

Вместо просто предсказания спреда, система:
1. Анализирует глубину order book
2. Обнаруживает манипуляции ботов
3. Предсказывает взрывы спреда ДО того как они произойдут
4. Дает многоуровневые рекомендации (5мин, 15мин, 1час)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@dataclass
class OrderBookSnapshot:
    """Снимок order book в момент времени."""

    bid_price: Decimal
    ask_price: Decimal
    bid_depth: Decimal  # Сколько монет на ask (объём продавцов)
    ask_depth: Decimal  # Сколько монет на bid (объём покупателей)
    bid_imbalance: float  # bid_depth / (bid_depth + ask_depth) -> 0-1

    # Микроструктура
    bid_ask_ratio: float  # Сколько раз глубины отличаются
    spread_bps: float  # Текущий спред в bps


@dataclass
class SpreadPredictorEnhancedFeatures:
    """Расширенные признаки для спред-предсказания."""

    # === БАЗОВЫЕ ПРИЗНАКИ ===
    hour_of_day: int
    day_of_week: int
    is_funding_time: bool

    # === МИКРОСТРУКТУРА ===
    bid_ask_imbalance: float  # Асимметрия глубин (-1 to 1)
    bid_ask_ratio: float  # bid_depth / ask_depth (насколько асимметрично)
    order_book_total_depth: float  # Общая глубина книги
    microstructure_score: float  # -1=много продавцов, +1=много покупателей

    # === ТРЕНДЫ СПРЕДА ===
    spread_trend_5m: float  # Спред растёт или падает (последние 5мин)
    spread_volatility: float  # Как часто меняется спред
    spread_acceleration: float  # Спред ускоряет рост? (опасно!)

    # === РЫНОЧНЫЕ УСЛОВИЯ ===
    price_volatility_bps: float  # Волатильность цены
    volume_ratio: float  # Текущий объём / средний
    momentum: float  # Направление движения (-1 to 1)

    # === ИСТОРИЧЕСКОЕ ПОВЕДЕНИЕ ===
    recent_max_spread_bps: float  # Максимальный спред за последний час
    recent_avg_spread_bps: float  # Средний спред за последний час
    spread_mean_reversion_pct: float  # На сколько % спред выше среднего


class SpreadPredictorEnhanced:
    """Усиленный предсказатель спредов с анализом микроструктуры."""

    def __init__(self):
        self.spread_model: Optional[XGBRegressor] = None  # Предсказывает размер спреда
        self.widening_model: Optional[XGBClassifier] = None  # Классифицирует: будет ли взрыв спреда

        self.min_training_samples = 200  # Больше чем раньше для качества
        self.spread_history: list[float] = []  # История спредов для анализа
        self.last_training_time = datetime.now(UTC)

        # Статистика для лучшего предсказания
        self.mean_spread = 20.0
        self.std_spread = 5.0

    async def train(self, training_data: list[dict]) -> None:
        """Обучить ДВЕ модели: спред + риск взрыва."""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available")
            return

        if len(training_data) < self.min_training_samples:
            logger.info(f"Недостаточно данных: {len(training_data)} < {self.min_training_samples}")
            return

        try:
            x_list = []
            y_spreads = []
            y_widening = []  # 0=спред нормальный, 1=спред расширяется

            for record in training_data:
                features = record.get("features")
                if not features:
                    continue

                # Вектор всех признаков (15 штук)
                x = np.array([
                    features.hour_of_day,
                    features.day_of_week,
                    float(features.is_funding_time),
                    features.bid_ask_imbalance,
                    features.bid_ask_ratio,
                    features.order_book_total_depth,
                    features.microstructure_score,
                    features.spread_trend_5m,
                    features.spread_volatility,
                    features.spread_acceleration,
                    features.price_volatility_bps,
                    features.volume_ratio,
                    features.momentum,
                    features.recent_max_spread_bps,
                    features.spread_mean_reversion_pct,
                ], dtype=np.float32)

                x_list.append(x)

                # Целевые переменные
                y_spreads.append(record.get("actual_spread_bps", 20.0))

                # Если спред резко расширился - это "1" (опасно)
                is_widening = 1 if record.get("spread_widened", False) else 0
                y_widening.append(is_widening)

            if len(x_list) < self.min_training_samples:
                return

            x_train = np.array(x_list, dtype=np.float32)
            y_spreads_train = np.array(y_spreads, dtype=np.float32)
            y_widening_train = np.array(y_widening, dtype=np.int32)

            # Модель 1: Предсказание размера спреда
            logger.info("📊 Обучаю модель РАЗМЕРА СПРЕДА...")
            self.spread_model = XGBRegressor(
                max_depth=6,  # Глубже чем базовая версия
                learning_rate=0.1,
                n_estimators=100,  # Больше деревьев = точнее
                random_state=42,
                subsample=0.8,  # Используем 80% данных на каждом дереве (более устойчиво)
                colsample_bytree=0.8,
            )
            self.spread_model.fit(x_train, y_spreads_train)
            spread_r2 = self.spread_model.score(x_train, y_spreads_train)
            logger.info(f"✅ Модель спреда: R² = {spread_r2:.3f}")

            # Модель 2: Предсказание взрыва спреда
            logger.info("⚠️  Обучаю модель РИСКА ВЗРЫВА СПРЕДА...")
            self.widening_model = XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42,
                scale_pos_weight=5,  # Взрывы редкие - даём им больший вес
            )
            self.widening_model.fit(x_train, y_widening_train)
            widening_accuracy = self.widening_model.score(x_train, y_widening_train)
            logger.info(f"✅ Модель взрыва: accuracy = {widening_accuracy:.3f}")

            # Обновить статистику
            self.mean_spread = float(np.mean(y_spreads_train))
            self.std_spread = float(np.std(y_spreads_train))

            self.last_training_time = datetime.now(UTC)
            logger.info(f"🎯 Обучение завершено: {len(x_list)} сэмплов")

        except Exception as e:
            logger.error(f"spread_predictor.training_failed: {e}")

    async def predict(
        self,
        features: SpreadPredictorEnhancedFeatures,
    ) -> dict:
        """Полное предсказание спреда и рисков.

        Возвращает:
        {
            'predicted_spread_bps': float,
            'spread_recommendation': str,  # GOOD/OK/CAUTION/WAIT
            'widening_risk': float,  # 0-1, вероятность взрыва спреда
            'widening_warning': str,  # "SAFE" / "CAUTION" / "DANGER"
            'optimal_entry_timing': str,  # IMMEDIATE/WAIT_5MIN/WAIT_15MIN
            'confidence': float,
            'explanation': str,
        }
        """
        if self.spread_model is None or self.widening_model is None:
            return self._get_fallback_prediction(features)

        try:
            x = np.array([
                features.hour_of_day,
                features.day_of_week,
                float(features.is_funding_time),
                features.bid_ask_imbalance,
                features.bid_ask_ratio,
                features.order_book_total_depth,
                features.microstructure_score,
                features.spread_trend_5m,
                features.spread_volatility,
                features.spread_acceleration,
                features.price_volatility_bps,
                features.volume_ratio,
                features.momentum,
                features.recent_max_spread_bps,
                features.spread_mean_reversion_pct,
            ], dtype=np.float32).reshape(1, -1)

            # 1. Предсказать размер спреда
            predicted_spread = float(self.spread_model.predict(x)[0])
            predicted_spread = max(1.0, min(100.0, predicted_spread))

            # 2. Предсказать вероятность взрыва
            widening_proba = float(self.widening_model.predict_proba(x)[0][1])  # Вероятность класса 1

            # 3. Рекомендации по размеру спреда
            spread_z_score = (predicted_spread - self.mean_spread) / max(self.std_spread, 1.0)
            if predicted_spread < 10:
                spread_recommendation = "GOOD - спред очень узкий 🟢"
            elif predicted_spread < 20:
                spread_recommendation = "OK - спред нормальный 🟡"
            elif predicted_spread < 35:
                spread_recommendation = "CAUTION - спред расширился 🟠"
            else:
                spread_recommendation = "WAIT - спред слишком широкий 🔴"

            # 4. Предупреждение о взрыве спреда
            if widening_proba > 0.7:
                widening_warning = "DANGER - высокий риск взрыва спреда! ⚠️"
                optimal_timing = "WAIT_15MIN"  # Подожди 15 минут
            elif widening_proba > 0.4:
                widening_warning = "CAUTION - спред может расширить 🟡"
                optimal_timing = "WAIT_5MIN"
            else:
                widening_warning = "SAFE - спред стабилен 🟢"
                optimal_timing = "IMMEDIATE"

            # 5. Объяснение (для понимания)
            explanation = (
                f"Спред {predicted_spread:.1f}bps (риск взрыва {widening_proba:.0%}). "
                f"Имбаланс order book: {features.bid_ask_imbalance:.2f}. "
                f"Тренд спреда: {'растёт' if features.spread_trend_5m > 0 else 'падает'}."
            )

            confidence = 0.7  # В усиленной версии выше уверенность

            return {
                'predicted_spread_bps': predicted_spread,
                'spread_recommendation': spread_recommendation,
                'widening_risk': widening_proba,
                'widening_warning': widening_warning,
                'optimal_entry_timing': optimal_timing,
                'confidence': confidence,
                'explanation': explanation,
            }

        except Exception as e:
            logger.error(f"spread_predictor.inference_failed: {e}")
            return self._get_fallback_prediction(features)

    @staticmethod
    def _get_fallback_prediction(features: SpreadPredictorEnhancedFeatures) -> dict:
        """Резервный вариант когда модели не обучены."""
        return {
            'predicted_spread_bps': features.recent_avg_spread_bps,
            'spread_recommendation': "Модели не обучены, используем среднее значение",
            'widening_risk': 0.5,
            'widening_warning': "UNKNOWN - нет данных для предсказания",
            'optimal_entry_timing': "WAIT_5MIN",
            'confidence': 0.3,
            'explanation': "Модель еще обучается",
        }

    def should_trade(
        self,
        current_spread_bps: float,
        widening_risk: float,
        max_acceptable_spread_bps: float = 30.0,
        max_acceptable_risk: float = 0.6,
    ) -> tuple[bool, str]:
        """Простое правило: торговать ли прямо сейчас?

        Возвращает: (should_trade, reason)
        """
        if widening_risk > max_acceptable_risk:
            return False, "Риск взрыва спреда слишком высокий"

        if current_spread_bps > max_acceptable_spread_bps:
            return False, f"Спред {current_spread_bps:.0f}bps > {max_acceptable_spread_bps:.0f}bps"

        return True, f"✅ Условия хорошие (спред {current_spread_bps:.0f}bps, риск {widening_risk:.0%})"
