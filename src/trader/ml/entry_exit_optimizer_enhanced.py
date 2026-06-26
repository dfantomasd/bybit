"""УСИЛЕННАЯ оптимизация входа/выхода с микроструктурой.

Продвинутая версия:
1. Order Flow Imbalance - анализ реальных покупателей/продавцов
2. VWAP execution - входить где объём, выходить там же
3. Optimal execution algorithm - разбить большой ордер
4. Consolidation zone detection - найти где цена "застревает"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available, using simple numpy-based models")
    from trader.ml.simple_models import SimpleEnsembleRegressor, SimpleClassifier
    XGBRegressor = SimpleEnsembleRegressor
    XGBClassifier = SimpleClassifier


@dataclass
class CandleContextEnhanced:
    """Расширенный контекст свечи с микроструктурой."""

    # === БАЗОВЫЕ ДАННЫЕ ===
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal

    # === ИНДИКАТОРЫ ===
    rsi: float
    atr_pct: float
    volatility_pct: float
    trend_strength: float  # -1 to 1

    # === VWAP АНАЛИЗ ===
    vwap_price: Decimal  # Volume Weighted Average Price
    distance_to_vwap_pct: float  # На сколько % от VWAP текущая цена
    vwap_trend: float  # VWAP растёт или падает

    # === ORDER FLOW ===
    buy_volume: Decimal  # Объём покупателей
    sell_volume: Decimal  # Объём продавцов
    order_flow_imbalance: float  # (buy - sell) / (buy + sell)
    cumulative_delta: float  # Накопленная разница покупок/продаж

    # === КОНСОЛИДАЦИЯ ===
    recent_swing_lows: list[float]  # Где цена отскакивала вверх
    recent_swing_highs: list[float]  # Где цена откатывалась вниз
    consolidation_range_pct: float  # Размер зоны консолидации
    is_in_consolidation: bool  # Находимся ли в зоне

    # === МИКРОСТРУКТУРА ===
    bid_ask_spread_bps: float
    bid_ask_imbalance: float  # Дисбаланс глубины книги
    order_clustering: float  # На сколько сгруппированы ордера (0-1)

    # === ВРЕМЕННОЕ ЗНАЧЕНИЕ ===
    time_into_candle_pct: float  # Где мы в свече (0-100%)
    candle_direction: int  # -1 (вниз), 0 (боковик), +1 (вверх)


@dataclass
class EntryExitOptimizationResult:
    """Результат оптимизации входа/выхода."""

    # === ОСНОВНЫЕ ЦЕНЫ ===
    entry_price: Decimal  # Цена входа
    take_profit_price: Decimal
    stop_loss_price: Decimal
    emergency_exit_price: Decimal  # Экстренный выход

    # === ТАЙМИНГ ===
    entry_timing: str  # IMMEDIATE, ACCUMULATE, WAIT_FOR_VWAP
    execution_strategy: str  # MARKET, LIMIT, ICEBERG
    estimated_entry_slippage_bps: float

    # === ПОРТФЕЛЬ ===
    recommended_position_size_pct: float  # % капитала на позицию
    split_orders: int  # На сколько ордеров разбить

    # === АНАЛИЗ ===
    risk_reward_ratio: float  # Профит / стоп
    expected_profit_bps: float
    probability_of_success: float  # 0-1

    # === РЕКОМЕНДАЦИЯ ===
    recommendation: str
    explanation: str


class EntryExitOptimizerEnhanced:
    """Умный вход/выход с анализом микроструктуры и VWAP."""

    def __init__(self):
        self.entry_model: Optional[XGBRegressor] = None
        self.tp_model: Optional[XGBRegressor] = None
        self.sl_model: Optional[XGBRegressor] = None
        self.execution_model: Optional[XGBClassifier] = None  # Какую стратегию использовать

        self.min_training_samples = 200

    async def train(self, training_data: list[dict]) -> None:
        """Обучить ЧЕТЫРЕв модели: вход + TP + SL + стратегия выполнения."""
        if not XGBOOST_AVAILABLE or len(training_data) < self.min_training_samples:
            return

        try:
            x_list = []
            y_entry = []
            y_tp = []
            y_sl = []
            y_execution = []

            for record in training_data:
                context = record.get("context")
                if not context:
                    continue

                # 21 признак
                x = np.array([
                    float(context.rsi),
                    context.atr_pct,
                    context.volatility_pct,
                    context.trend_strength,
                    context.distance_to_vwap_pct,
                    context.vwap_trend,
                    context.order_flow_imbalance,
                    context.cumulative_delta,
                    context.consolidation_range_pct,
                    float(context.is_in_consolidation),
                    context.bid_ask_spread_bps,
                    context.bid_ask_imbalance,
                    context.order_clustering,
                    context.time_into_candle_pct,
                    float(context.candle_direction),
                    np.mean(context.recent_swing_lows) if context.recent_swing_lows else 0.0,
                    np.mean(context.recent_swing_highs) if context.recent_swing_highs else 0.0,
                    context.order_flow_imbalance ** 2,  # Нелинейный признак
                    context.consolidation_range_pct ** 2,
                    context.bid_ask_imbalance ** 2,
                    context.volatility_pct * context.trend_strength,
                ], dtype=np.float32)

                x_list.append(x)

                # Целевые переменные
                y_entry.append(record.get("optimal_entry_offset_pct", 0.0))
                y_tp.append(record.get("optimal_tp_distance_pct", 1.0))
                y_sl.append(record.get("optimal_sl_distance_pct", 0.7))

                # 0=MARKET, 1=LIMIT, 2=ICEBERG (разбитый)
                execution_class = record.get("best_execution_strategy", 0)
                y_execution.append(execution_class)

            if len(x_list) < self.min_training_samples:
                return

            x_train = np.array(x_list, dtype=np.float32)
            y_entry_train = np.array(y_entry, dtype=np.float32)
            y_tp_train = np.array(y_tp, dtype=np.float32)
            y_sl_train = np.array(y_sl, dtype=np.float32)
            y_execution_train = np.array(y_execution, dtype=np.int32)

            # Модель 1: Оптимальный вход
            logger.info("📍 Обучаю модель ОПТИМАЛЬНОГО ВХОДА...")
            self.entry_model = XGBRegressor(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=120,
                random_state=42,
                subsample=0.8,
            )
            self.entry_model.fit(x_train, y_entry_train)
            logger.info(f"✅ Модель входа готова")

            # Модель 2: Take-Profit
            logger.info("🎯 Обучаю модель TAKE-PROFIT...")
            self.tp_model = XGBRegressor(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=120,
                random_state=42,
            )
            self.tp_model.fit(x_train, y_tp_train)
            logger.info(f"✅ Модель TP готова")

            # Модель 3: Stop-Loss
            logger.info("🛑 Обучаю модель STOP-LOSS...")
            self.sl_model = XGBRegressor(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=120,
                random_state=42,
            )
            self.sl_model.fit(x_train, y_sl_train)
            logger.info(f"✅ Модель SL готова")

            # Модель 4: Стратегия выполнения
            logger.info("⚙️  Обучаю модель СТРАТЕГИИ ВЫПОЛНЕНИЯ...")
            self.execution_model = XGBClassifier(
                max_depth=5,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42,
                objective="multi:softmax",
                num_class=3,
            )
            self.execution_model.fit(x_train, y_execution_train)
            logger.info(f"✅ Модель выполнения готова")

            logger.info(f"🎯 Обучение входа/выхода завершено: {len(x_list)} сэмплов")

        except Exception as e:
            logger.error(f"entry_exit_training_failed: {e}")

    async def get_optimization(
        self,
        context: CandleContextEnhanced,
        side: str = "BUY",
    ) -> EntryExitOptimizationResult:
        """Получить полную оптимизацию входа/выхода."""
        if self.entry_model is None:
            return self._get_default_optimization(context, side)

        try:
            x = np.array([
                float(context.rsi),
                context.atr_pct,
                context.volatility_pct,
                context.trend_strength,
                context.distance_to_vwap_pct,
                context.vwap_trend,
                context.order_flow_imbalance,
                context.cumulative_delta,
                context.consolidation_range_pct,
                float(context.is_in_consolidation),
                context.bid_ask_spread_bps,
                context.bid_ask_imbalance,
                context.order_clustering,
                context.time_into_candle_pct,
                float(context.candle_direction),
                np.mean(context.recent_swing_lows) if context.recent_swing_lows else 0.0,
                np.mean(context.recent_swing_highs) if context.recent_swing_highs else 0.0,
                context.order_flow_imbalance ** 2,
                context.consolidation_range_pct ** 2,
                context.bid_ask_imbalance ** 2,
                context.volatility_pct * context.trend_strength,
            ], dtype=np.float32).reshape(1, -1)

            # Предсказания
            entry_offset = float(self.entry_model.predict(x)[0])
            tp_distance = float(self.tp_model.predict(x)[0])
            sl_distance = float(self.sl_model.predict(x)[0])
            execution_class = int(self.execution_model.predict(x)[0])

            # Зажать значения
            entry_offset = max(-1.0, min(1.0, entry_offset))
            tp_distance = max(0.5, min(3.0, tp_distance))
            sl_distance = max(0.3, min(2.0, sl_distance))

            # Вычислить цены
            current_price = context.close_price
            if side == "BUY":
                entry_price = current_price * (Decimal("1") + Decimal(str(entry_offset / 100)))
                take_profit = entry_price * (Decimal("1") + Decimal(str(tp_distance / 100)))
                stop_loss = entry_price * (Decimal("1") - Decimal(str(sl_distance / 100)))
            else:
                entry_price = current_price * (Decimal("1") - Decimal(str(entry_offset / 100)))
                take_profit = entry_price * (Decimal("1") - Decimal(str(tp_distance / 100)))
                stop_loss = entry_price * (Decimal("1") + Decimal(str(sl_distance / 100)))

            # Стратегия выполнения
            execution_strategies = ["MARKET", "LIMIT", "ICEBERG"]
            execution_strategy = execution_strategies[execution_class]

            # VWAP-based анализ
            if abs(context.distance_to_vwap_pct) < 0.1:
                entry_timing = "IMMEDIATE"  # Рядом с VWAP - хороший вход
            elif context.order_flow_imbalance > 0.3:
                entry_timing = "ACCUMULATE"  # Много покупателей - ждём
            else:
                entry_timing = "WAIT_FOR_VWAP"  # Ждём VWAP

            # Optimal position size
            if context.is_in_consolidation:
                position_size_pct = 1.0  # Консолидация = точный вход, можешь больше
            else:
                position_size_pct = max(0.5, min(2.0, 1.0 / max(context.volatility_pct, 0.5)))

            # Разбить ордер на части?
            if execution_strategy == "ICEBERG":
                split_orders = 3  # Разбиваем на 3 части
            else:
                split_orders = 1

            # Risk/Reward
            tp_profit_bps = abs(float((take_profit - entry_price) / entry_price * 10000))
            sl_loss_bps = abs(float((stop_loss - entry_price) / entry_price * 10000))
            risk_reward = tp_profit_bps / max(sl_loss_bps, 1.0)

            # Вероятность успеха (на основе микроструктуры)
            prob_success = 0.5 + (context.order_flow_imbalance * 0.3) + (context.trend_strength * 0.2)
            prob_success = max(0.3, min(0.9, prob_success))

            # Emergency exit (в худшем случае)
            emergency_distance = sl_distance * 1.5
            if side == "BUY":
                emergency_exit = entry_price * (Decimal("1") - Decimal(str(emergency_distance / 100)))
            else:
                emergency_exit = entry_price * (Decimal("1") + Decimal(str(emergency_distance / 100)))

            # Рекомендация
            recommendation = self._get_recommendation(
                context, entry_timing, risk_reward, prob_success
            )

            explanation = (
                f"Вход {entry_timing} по {execution_strategy}. "
                f"Order flow imbalance: {context.order_flow_imbalance:.1%}. "
                f"Расстояние до VWAP: {context.distance_to_vwap_pct:.2f}%. "
                f"Risk/Reward: {risk_reward:.2f}. "
                f"Успех: {prob_success:.0%}"
            )

            return EntryExitOptimizationResult(
                entry_price=entry_price,
                take_profit_price=take_profit,
                stop_loss_price=stop_loss,
                emergency_exit_price=emergency_exit,
                entry_timing=entry_timing,
                execution_strategy=execution_strategy,
                estimated_entry_slippage_bps=context.bid_ask_spread_bps,
                recommended_position_size_pct=position_size_pct,
                split_orders=split_orders,
                risk_reward_ratio=risk_reward,
                expected_profit_bps=tp_profit_bps - context.bid_ask_spread_bps,
                probability_of_success=prob_success,
                recommendation=recommendation,
                explanation=explanation,
            )

        except Exception as e:
            logger.error(f"entry_exit_inference_failed: {e}")
            return self._get_default_optimization(context, side)

    @staticmethod
    def _get_recommendation(
        context: CandleContextEnhanced,
        timing: str,
        risk_reward: float,
        prob_success: float,
    ) -> str:
        """Рекомендация на основе анализа."""
        if prob_success < 0.4:
            return "🔴 SKIP - слишком рискованно"

        if risk_reward < 1.0:
            return "⚠️ CAUTION - награда меньше риска"

        if timing == "IMMEDIATE" and risk_reward > 2.0:
            return "🟢 EXCELLENT - идеальный момент"

        if context.is_in_consolidation and prob_success > 0.6:
            return "🟢 GOOD - в консолидации, точный вход"

        if context.order_flow_imbalance > 0.3:
            return "🟢 GO - много покупателей, тренд готов"

        return "🟡 OKAY - условия приемлемы"

    @staticmethod
    def _get_default_optimization(
        context: CandleContextEnhanced,
        side: str = "BUY",
    ) -> EntryExitOptimizationResult:
        """Дефолтная оптимизация когда модели не обучены."""
        current_price = context.close_price
        atr_based_tp = context.atr_pct * 2
        atr_based_sl = context.atr_pct * 1.0

        if side == "BUY":
            entry = current_price
            tp = current_price * (Decimal("1") + Decimal(str(atr_based_tp / 100)))
            sl = current_price * (Decimal("1") - Decimal(str(atr_based_sl / 100)))
            emergency = current_price * (Decimal("1") - Decimal(str(atr_based_sl * 1.5 / 100)))
        else:
            entry = current_price
            tp = current_price * (Decimal("1") - Decimal(str(atr_based_tp / 100)))
            sl = current_price * (Decimal("1") + Decimal(str(atr_based_sl / 100)))
            emergency = current_price * (Decimal("1") + Decimal(str(atr_based_sl * 1.5 / 100)))

        return EntryExitOptimizationResult(
            entry_price=entry,
            take_profit_price=tp,
            stop_loss_price=sl,
            emergency_exit_price=emergency,
            entry_timing="IMMEDIATE",
            execution_strategy="MARKET",
            estimated_entry_slippage_bps=context.bid_ask_spread_bps,
            recommended_position_size_pct=1.0,
            split_orders=1,
            risk_reward_ratio=atr_based_tp / atr_based_sl,
            expected_profit_bps=atr_based_tp * 100 - context.bid_ask_spread_bps,
            probability_of_success=0.5,
            recommendation="Модель не обучена, используется ATR",
            explanation="Default optimization based on ATR",
        )
