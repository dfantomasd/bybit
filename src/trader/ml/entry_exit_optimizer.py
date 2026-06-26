"""Умный выбор момента входа и выхода из позиции.

Предсказывает лучший момент для входа в рамках одной свечи.
Определяет оптимальную зону для take-profit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
class CandleContext:
    """Контекст текущей свечи."""

    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    current_price: Decimal
    time_into_candle_pct: float  # 0-100, где мы сейчас в свече

    rsi: float  # 0-100
    atr_pct: float  # ATR как % от цены
    recent_volatility: float  # Волатильность
    trend_strength: float  # -1 to 1

    momentum: float  # Текущий момент (-1 to 1)
    volume_ratio: float  # Текущий объём / средний


@dataclass
class EntryExitRecommendation:
    """Рекомендация для входа и выхода."""

    best_entry_price: Decimal  # Оптимальная цена входа
    take_profit_price: Decimal  # Зона take-profit
    stop_loss_distance_pct: float  # Расстояние стопа

    entry_timing: str  # "IMMEDIATE", "WAIT_FOR_DIP", "WAIT_FOR_PEAK"
    confidence: float  # 0-1
    expected_profit_bps: float  # Ожидаемая прибыль в bps


class EntryExitOptimizer:
    """Оптимизирует точку входа и выхода для максимизации профита."""

    def __init__(self):
        self.entry_model: Optional[XGBRegressor] = None  # Предсказывает лучший момент входа
        self.tp_model: Optional[XGBRegressor] = None  # Предсказывает оптимальный TP уровень
        self.min_training_samples = 100

    async def train(self, training_data: list[dict]) -> None:
        """Обучить модели на исторических точках входа/выхода."""
        if not XGBOOST_AVAILABLE:
            return

        if len(training_data) < self.min_training_samples:
            return

        try:
            x_list_entry = []
            y_entry_timing = []  # 0=immediate, 1=wait_dip, 2=wait_peak
            y_entry_price_offset = []  # Как много % надо отступить от текущей цены

            x_list_tp = []
            y_tp_distance = []  # На сколько % выше вставить TP

            for record in training_data:
                context = record.get("context")
                if not context:
                    continue

                # Вектор признаков
                x = np.array([
                    float(context.rsi),
                    context.atr_pct,
                    context.recent_volatility,
                    context.trend_strength,
                    context.momentum,
                    context.volume_ratio,
                    context.time_into_candle_pct,
                ], dtype=np.float32)

                # Для модели входа
                x_list_entry.append(x)
                entry_timing = record.get("optimal_entry_timing", 0)
                entry_offset = record.get("best_entry_offset_pct", 0.0)
                y_entry_timing.append(entry_timing)
                y_entry_price_offset.append(entry_offset)

                # Для модели TP
                x_list_tp.append(x)
                tp_distance = record.get("optimal_tp_distance_pct", 1.0)
                y_tp_distance.append(tp_distance)

            if len(x_list_entry) < self.min_training_samples:
                return

            # Обучить модель входа
            x_entry_train = np.array(x_list_entry, dtype=np.float32)
            y_entry_train = np.array(y_entry_price_offset, dtype=np.float32)

            self.entry_model = XGBRegressor(
                max_depth=4,
                learning_rate=0.1,
                n_estimators=50,
                random_state=42,
            )
            self.entry_model.fit(x_entry_train, y_entry_train)

            # Обучить модель TP
            x_tp_train = np.array(x_list_tp, dtype=np.float32)
            y_tp_train = np.array(y_tp_distance, dtype=np.float32)

            self.tp_model = XGBRegressor(
                max_depth=4,
                learning_rate=0.1,
                n_estimators=50,
                random_state=42,
            )
            self.tp_model.fit(x_tp_train, y_tp_train)

            logger.info(f"entry_exit_optimizer.trained: {len(x_list_entry)} samples")

        except Exception as e:
            logger.error(f"entry_exit_optimizer.training_failed: {e}")

    async def get_recommendation(
        self,
        context: CandleContext,
        side: str = "BUY",
    ) -> EntryExitRecommendation:
        """Получить рекомендацию для входа и выхода.

        Args:
            context: Информация о текущей свече
            side: "BUY" или "SELL"

        Возвращает: EntryExitRecommendation с ценами и таймингом
        """
        if self.entry_model is None or self.tp_model is None:
            return self._get_default_recommendation(context, side)

        try:
            x = np.array([
                float(context.rsi),
                context.atr_pct,
                context.recent_volatility,
                context.trend_strength,
                context.momentum,
                context.volume_ratio,
                context.time_into_candle_pct,
            ], dtype=np.float32).reshape(1, -1)

            # Предсказать лучший момент входа
            entry_offset_pct = float(self.entry_model.predict(x)[0])
            entry_offset_pct = max(-2.0, min(2.0, entry_offset_pct))  # -2% до +2%

            if side == "BUY":
                best_entry = context.current_price * (Decimal("1") + Decimal(str(entry_offset_pct / 100)))
            else:  # SELL
                best_entry = context.current_price * (Decimal("1") - Decimal(str(entry_offset_pct / 100)))

            # Определить тайминг
            if abs(entry_offset_pct) < 0.3:
                entry_timing = "IMMEDIATE"
            elif entry_offset_pct < 0:  # Цена упала - ждём падения
                entry_timing = "WAIT_FOR_DIP"
            else:  # Цена выросла - ждём роста
                entry_timing = "WAIT_FOR_PEAK"

            # Предсказать TP
            tp_distance_pct = float(self.tp_model.predict(x)[0])
            tp_distance_pct = max(0.5, min(3.0, tp_distance_pct))  # 0.5% до 3%

            if side == "BUY":
                take_profit = best_entry * (Decimal("1") + Decimal(str(tp_distance_pct / 100)))
            else:  # SELL
                take_profit = best_entry * (Decimal("1") - Decimal(str(tp_distance_pct / 100)))

            # Стоп-лосс на основе ATR
            stop_loss_pct = context.atr_pct * 1.5
            stop_loss_pct = max(0.5, min(3.0, stop_loss_pct))

            # Ожидаемая прибыль
            expected_profit = tp_distance_pct - (stop_loss_pct / 2)
            expected_profit_bps = expected_profit * 100

            confidence = 0.6

            return EntryExitRecommendation(
                best_entry_price=best_entry,
                take_profit_price=take_profit,
                stop_loss_distance_pct=stop_loss_pct,
                entry_timing=entry_timing,
                confidence=confidence,
                expected_profit_bps=expected_profit_bps,
            )

        except Exception as e:
            logger.error(f"entry_exit_optimizer.inference_failed: {e}")
            return self._get_default_recommendation(context, side)

    @staticmethod
    def _get_default_recommendation(
        context: CandleContext,
        side: str = "BUY",
    ) -> EntryExitRecommendation:
        """Стандартная рекомендация когда модели не обучены."""
        atr_based_tp = context.atr_pct * 2
        atr_based_sl = context.atr_pct * 1.0

        if side == "BUY":
            best_entry = context.current_price
            take_profit = best_entry * (Decimal("1") + Decimal(str(atr_based_tp / 100)))
        else:
            best_entry = context.current_price
            take_profit = best_entry * (Decimal("1") - Decimal(str(atr_based_tp / 100)))

        return EntryExitRecommendation(
            best_entry_price=best_entry,
            take_profit_price=take_profit,
            stop_loss_distance_pct=atr_based_sl,
            entry_timing="IMMEDIATE",
            confidence=0.3,
            expected_profit_bps=atr_based_tp * 100 - (atr_based_sl / 2) * 100,
        )
