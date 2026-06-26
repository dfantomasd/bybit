"""Адаптивный стоп-лосс на основе волатильности.

Расчитывает оптимальный стоп в зависимости от текущих рыночных условий.
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
class StopLossContext:
    """Контекст для расчёта стоп-лосса."""

    realized_volatility_pct: float  # Недавняя волатильность в %
    atr_pct: float  # ATR как % от цены
    recent_win_rate: float  # Win rate за последние сделки
    recent_swings_bps: list[float]  # Размеры недавних колебаний
    market_regime: str  # "trend", "sideways", "volatile"
    hour_of_day: int  # 0-23


class StopLossOptimizer:
    """Оптимизирует размер стоп-лосса под текущие условия."""

    def __init__(self):
        self.model: Optional[XGBRegressor] = None
        self.min_training_samples = 100
        self.regime_multipliers = {
            "trend": 1.5,  # В тренде можно более широкий стоп
            "sideways": 0.8,  # В боковике уже стоп
            "volatile": 1.2,  # При волатильности немного шире
        }

    async def train(self, training_data: list[dict]) -> None:
        """Обучить на исторических данных."""
        if not XGBOOST_AVAILABLE or len(training_data) < self.min_training_samples:
            return

        try:
            x_list = []
            y_stops = []

            for record in training_data:
                context = record.get("context")
                if not context:
                    continue

                x = np.array([
                    context.realized_volatility_pct,
                    context.atr_pct,
                    context.recent_win_rate,
                    np.mean(context.recent_swings_bps) if context.recent_swings_bps else 50.0,
                    np.std(context.recent_swings_bps) if len(context.recent_swings_bps) > 1 else 20.0,
                    context.hour_of_day,
                ], dtype=np.float32)

                x_list.append(x)
                y_stops.append(record.get("optimal_stop_pct", 2.0))

            if len(x_list) < self.min_training_samples:
                return

            x_train = np.array(x_list, dtype=np.float32)
            y_stops_train = np.array(y_stops, dtype=np.float32)

            self.model = XGBRegressor(
                max_depth=4,
                learning_rate=0.1,
                n_estimators=50,
                random_state=42,
            )
            self.model.fit(x_train, y_stops_train)
            logger.info(f"stoploss_optimizer.trained: {len(x_list)} samples")

        except Exception as e:
            logger.error(f"stoploss_optimizer.training_failed: {e}")

    def calculate_optimal_stop(
        self,
        context: StopLossContext,
        default_stop_pct: float = 2.0,
        min_stop_pct: float = 0.5,
        max_stop_pct: float = 5.0,
    ) -> float:
        """Расчитать оптимальный стоп-лосс в %.

        Возвращает: stop_distance_pct
        """
        # Базовый расчёт на основе волатильности
        vol_based_stop = context.realized_volatility_pct * 1.5

        if self.model is not None:
            try:
                x = np.array([
                    context.realized_volatility_pct,
                    context.atr_pct,
                    context.recent_win_rate,
                    np.mean(context.recent_swings_bps) if context.recent_swings_bps else 50.0,
                    np.std(context.recent_swings_bps) if len(context.recent_swings_bps) > 1 else 20.0,
                    context.hour_of_day,
                ], dtype=np.float32).reshape(1, -1)

                ml_stop = float(self.model.predict(x)[0])
                optimal_stop = ml_stop
            except Exception as e:
                logger.debug(f"ML prediction failed: {e}, using vol-based")
                optimal_stop = vol_based_stop
        else:
            optimal_stop = vol_based_stop

        # Применить режимный множитель
        regime_mult = self.regime_multipliers.get(context.market_regime, 1.0)
        optimal_stop = optimal_stop * regime_mult

        # Зажать в пределах безопасности
        optimal_stop = max(min_stop_pct, min(max_stop_pct, optimal_stop))

        return optimal_stop

    @staticmethod
    def get_stop_recommendation(
        context: StopLossContext,
        calculated_stop_pct: float,
    ) -> str:
        """Объяснение почему такой стоп."""
        vol = context.realized_volatility_pct
        regime = context.market_regime

        if vol < 1.0:
            vol_desc = "волатильность низкая"
        elif vol < 2.0:
            vol_desc = "волатильность нормальная"
        elif vol < 3.0:
            vol_desc = "волатильность повышена"
        else:
            vol_desc = "волатильность ВЫСОКАЯ"

        return f"Стоп {calculated_stop_pct:.2f}% - {vol_desc}, {regime}"
