"""УСИЛЕННЫЙ адаптивный стоп-лосс.

Улучшения:
1. Анализ реальных уровней поддержки (swing lows)
2. CVaR - учитывание наихудших случаев
3. Динамический стоп - подвигать во время сделки
4. Режимный анализ - разные стопы для разных режимов
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
    logger.warning("XGBoost not available, using simple numpy-based models")
    from trader.ml.simple_models import SimpleEnsembleRegressor

    XGBRegressor = SimpleEnsembleRegressor


@dataclass
class StopLossContextEnhanced:
    """Расширенный контекст для расчёта стопа."""

    # === ВОЛАТИЛЬНОСТЬ ===
    realized_volatility_pct: float
    atr_pct: float
    volatility_trend: float  # Волатильность растёт или падает?

    # === ПОДДЕРЖКА/СОПРОТИВЛЕНИЕ ===
    recent_swing_lows: list[float]  # Последние минимумы цены
    recent_swing_highs: list[float]  # Последние максимумы цены
    nearest_support_pct: float  # % от текущей цены до поддержки
    nearest_resistance_pct: float  # % до сопротивления

    # === ХВОСТОВОЙ РИСК ===
    returns_history_pct: list[float]  # История доходов за 50+ свечей
    var_95_pct: float  # Value at Risk 95% (5-й перцентиль)
    cvar_95_pct: float  # Conditional VaR (средний убыток из худших 5%)

    # === РЫНОЧНЫЕ УСЛОВИЯ ===
    market_regime: str  # "trend", "sideways", "volatile"
    trend_strength: float  # -1 to 1, сила тренда
    recent_win_rate: float  # Процент выигрышных сделок
    hour_of_day: int

    # === ДИНАМИКА ===
    time_in_trade_minutes: int = 0  # Сколько минут мы в сделке


class StopLossOptimizerEnhanced:
    """Умный расчёт стопа с анализом уровней и хвостового риска."""

    def __init__(self) -> None:
        self.model: XGBRegressor | None = None
        self.cvar_model: XGBRegressor | None = None  # Отдельная модель для хвостового риска

        self.min_training_samples = 200
        self.regime_multipliers = {
            "trend": 1.5,  # В тренде стоп шире
            "sideways": 0.7,  # В боковике уже
            "volatile": 1.3,  # При волатильности расширяем
        }

        # История для анализа
        self.historical_stops: list[float] = []
        self.mean_optimal_stop = 2.0

    async def train(self, training_data: list[dict]) -> None:
        """Обучить ДВЕ модели: стоп + хвостовой риск."""
        if not XGBOOST_AVAILABLE or len(training_data) < self.min_training_samples:
            return

        try:
            x_list = []
            y_stops = []
            y_cvar = []

            for record in training_data:
                context = record.get("context")
                if not context:
                    continue

                # 13 признаков
                x = np.array(
                    [
                        context.realized_volatility_pct,
                        context.atr_pct,
                        context.volatility_trend,
                        context.nearest_support_pct,
                        context.nearest_resistance_pct,
                        context.var_95_pct,
                        context.cvar_95_pct,
                        1.0 if context.market_regime == "trend" else 0.0,
                        1.0 if context.market_regime == "volatile" else 0.0,
                        context.trend_strength,
                        context.recent_win_rate,
                        context.hour_of_day,
                        context.time_in_trade_minutes / 60,  # в часах
                    ],
                    dtype=np.float32,
                )

                x_list.append(x)
                y_stops.append(record.get("optimal_stop_pct", 2.0))
                y_cvar.append(record.get("optimal_cvar_pct", 3.0))

            if len(x_list) < self.min_training_samples:
                return

            x_train = np.array(x_list, dtype=np.float32)
            y_stops_train = np.array(y_stops, dtype=np.float32)
            y_cvar_train = np.array(y_cvar, dtype=np.float32)

            # Модель 1: Обычный стоп
            logger.info("📍 Обучаю модель СТОП-ЛОССА...")
            self.model = XGBRegressor(
                max_depth=5,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42,
                subsample=0.8,
            )
            self.model.fit(x_train, y_stops_train)
            r2_stop = self.model.score(x_train, y_stops_train)
            logger.info(f"✅ Модель стопа: R² = {r2_stop:.3f}")

            # Модель 2: Хвостовой риск (CVaR)
            logger.info("⚠️  Обучаю модель ХВОСТОВОГО РИСКА...")
            self.cvar_model = XGBRegressor(
                max_depth=5,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42,
            )
            self.cvar_model.fit(x_train, y_cvar_train)
            r2_cvar = self.cvar_model.score(x_train, y_cvar_train)
            logger.info(f"✅ Модель CVaR: R² = {r2_cvar:.3f}")

            self.mean_optimal_stop = float(np.mean(y_stops_train))
            self.historical_stops = y_stops

            logger.info(f"🎯 Обучение стопа завершено: {len(x_list)} сэмплов")

        except Exception as e:
            logger.error(f"stoploss_training_failed: {e}")

    async def calculate_optimal_stop(
        self,
        context: StopLossContextEnhanced,
        use_support_level: bool = True,
        use_cvar_safety: bool = True,
        min_stop_pct: float = 0.5,
        max_stop_pct: float = 6.0,
    ) -> dict:
        """Расчитать оптимальный стоп с учётом всего.

        Возвращает:
        {
            'stop_distance_pct': float,  # Основной стоп
            'emergency_stop_pct': float,  # Экстренный стоп (CVaR)
            'support_level_stop_pct': float,  # Стоп за поддержкой
            'recommendation': str,
            'explanation': str,
        }
        """
        stop_distance = self.mean_optimal_stop

        # 1. ML-ПРЕДСКАЗАНИЕ
        if self.model is not None:
            try:
                x = np.array(
                    [
                        context.realized_volatility_pct,
                        context.atr_pct,
                        context.volatility_trend,
                        context.nearest_support_pct,
                        context.nearest_resistance_pct,
                        context.var_95_pct,
                        context.cvar_95_pct,
                        1.0 if context.market_regime == "trend" else 0.0,
                        1.0 if context.market_regime == "volatile" else 0.0,
                        context.trend_strength,
                        context.recent_win_rate,
                        context.hour_of_day,
                        context.time_in_trade_minutes / 60,
                    ],
                    dtype=np.float32,
                ).reshape(1, -1)

                ml_stop = float(self.model.predict(x)[0])
                stop_distance = ml_stop
            except Exception as e:
                logger.debug(f"ML prediction failed: {e}")

        # 2. АНАЛИЗ ПОДДЕРЖКИ
        support_stop = max_stop_pct
        if use_support_level and context.nearest_support_pct > 0:
            # Ставим стоп за поддержкой (с небольшим буфером)
            support_stop = min(context.nearest_support_pct * 1.1, max_stop_pct)
            if support_stop < stop_distance:
                # Поддержка ближе чем расчётный стоп - используем её
                stop_distance = support_stop

        # 3. ХВОСТОВОЙ РИСК (CVaR)
        emergency_stop = max_stop_pct
        if use_cvar_safety and self.cvar_model is not None:
            try:
                x = np.array(
                    [
                        context.realized_volatility_pct,
                        context.atr_pct,
                        context.volatility_trend,
                        context.nearest_support_pct,
                        context.nearest_resistance_pct,
                        context.var_95_pct,
                        context.cvar_95_pct,
                        1.0 if context.market_regime == "trend" else 0.0,
                        1.0 if context.market_regime == "volatile" else 0.0,
                        context.trend_strength,
                        context.recent_win_rate,
                        context.hour_of_day,
                        context.time_in_trade_minutes / 60,
                    ],
                    dtype=np.float32,
                ).reshape(1, -1)

                emergency_stop = float(self.cvar_model.predict(x)[0])
                # Экстренный стоп всегда больше чем основной
                emergency_stop = max(stop_distance * 1.5, emergency_stop)
            except Exception as e:
                logger.debug(f"CVaR prediction failed: {e}")

        # 4. РЕЖИМНЫЙ МНОЖИТЕЛЬ
        regime_mult = self.regime_multipliers.get(context.market_regime, 1.0)
        stop_distance = stop_distance * regime_mult
        emergency_stop = emergency_stop * regime_mult

        # 5. ДИНАМИЧЕСКОЕ ДВИЖЕНИЕ СТОПА (если мы уже в сделке)
        if context.time_in_trade_minutes > 5:
            # Можно подвинуть стоп ближе к цене по мере развития сделки
            movement_factor = min(1.0, context.time_in_trade_minutes / 60)
            stop_distance = stop_distance * (1.0 - movement_factor * 0.2)  # До 20% ближе

        # 6. ЗАЖАТЬ В ПРЕДЕЛЫ
        stop_distance = max(min_stop_pct, min(max_stop_pct, stop_distance))
        emergency_stop = max(min_stop_pct, min(max_stop_pct, emergency_stop))

        # 7. ОБЪЯСНЕНИЕ
        explanation = self._build_explanation(context, stop_distance, emergency_stop)

        recommendation = "🟢 GOOD" if stop_distance < 1.5 else ("🟡 OK" if stop_distance < 2.5 else "🔴 WIDE")

        return {
            "stop_distance_pct": stop_distance,
            "emergency_stop_pct": emergency_stop,
            "support_level_stop_pct": support_stop,
            "recommendation": recommendation,
            "explanation": explanation,
        }

    def get_dynamic_stop(
        self,
        initial_stop_pct: float,
        current_profit_bps: float,
        trade_duration_minutes: int,
    ) -> float:
        """Подвинуть стоп во время развития сделки (dynamic stop-loss).

        Если сделка в профите - можно подвинуть стоп ближе.
        """
        if current_profit_bps < 50:  # Убыток или маленький профит
            return initial_stop_pct  # Не трогаем стоп

        # Профит растёт - подвигаем стоп
        if current_profit_bps > 200:  # 2% профита
            return initial_stop_pct * 0.7  # На 30% ближе
        elif current_profit_bps > 100:  # 1% профита
            return initial_stop_pct * 0.85  # На 15% ближе

        return initial_stop_pct

    @staticmethod
    def _build_explanation(
        context: StopLossContextEnhanced,
        stop_pct: float,
        emergency_pct: float,
    ) -> str:
        """Объяснение почему такой стоп."""
        vol = context.realized_volatility_pct
        regime = context.market_regime

        if vol < 1.0:
            vol_desc = "волатильность низкая"
        elif vol < 2.0:
            vol_desc = "волатильность нормальная"
        else:
            vol_desc = "волатильность ВЫСОКАЯ"

        explanation = (
            f"Стоп {stop_pct:.2f}% ({vol_desc}, {regime}). "
            f"Поддержка на {context.nearest_support_pct:.2f}%. "
            f"Экстренный стоп: {emergency_pct:.2f}%"
        )
        return explanation
