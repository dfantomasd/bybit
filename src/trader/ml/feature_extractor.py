"""Feature extractor - мост между торговой системой и ML моделями.

Извлекает признаки из торгового контекста для питания ML моделей.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

from trader.domain.enums import OrderSide

logger = logging.getLogger(__name__)


@dataclass
class KellyFeatures:
    """Признаки для Kelly predictor."""
    recent_win_rate: float
    std_dev_bps: float
    kurtosis: float
    drawdown_pct: float
    volatility_regime: str
    hour_of_day: int
    day_of_week: int


@dataclass
class RegimeFeatures:
    """Признаки для Regime predictor."""
    adx: float
    di_plus: float
    di_minus: float
    atr_pct: float
    rsi: float
    macd_line: float
    signal_line: float
    volatility_pct: float
    hour_of_day: int


@dataclass
class SignalContext:
    """Контекст для Signal fusion."""
    signal_ma_crossover: float
    signal_rsi: float
    signal_macd: float
    signal_breakout: float
    signal_volume: float

    confidence_ma: float
    confidence_rsi: float
    confidence_macd: float
    confidence_breakout: float
    confidence_volume: float

    market_regime: str
    recent_win_rate: float


@dataclass
class SpreadFeatures:
    """Признаки для Spread predictor."""
    base_spread_bps: float
    bid_ask_imbalance: float
    order_book_depth: float
    volatility_pct: float
    time_of_day_factor: float


@dataclass
class StopLossContext:
    """Контекст для StopLoss optimizer."""
    realized_volatility_pct: float
    atr_pct: float
    market_regime: str
    trend_strength: float
    recent_win_rate: float


class FeatureExtractor:
    """Извлекает признаки из торгового контекста."""

    def __init__(self):
        self.recent_returns_bps: list[float] = []
        self.recent_signals: dict[str, list[float]] = {
            "ma": [],
            "rsi": [],
            "macd": [],
            "breakout": [],
            "volume": [],
        }
        self.max_history = 100

    def extract_kelly_features(
        self,
        recent_trades: list[dict[str, Any]],
        current_volatility: float,
        hour_of_day: int = 12,
        day_of_week: int = 0,
    ) -> Optional[KellyFeatures]:
        """Извлечь признаки для Kelly predictor из истории сделок."""
        if len(recent_trades) < 2:
            return None

        # Вычислить win rate
        wins = sum(1 for trade in recent_trades if trade.get("pnl_usd", 0) > 0)
        win_rate = wins / len(recent_trades) if recent_trades else 0.5

        # Вычислить стандартное отклонение
        returns = [trade.get("pnl_bps", 0) for trade in recent_trades]
        if not returns or len(returns) < 2:
            std_dev = 50.0
        else:
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            std_dev = variance ** 0.5

        # Простой расчёт куртозиса (больше 3 = тяжёлые хвосты)
        kurtosis = 3.0  # Default normal distribution
        if len(returns) > 4:
            mean_return = sum(returns) / len(returns)
            m4 = sum((r - mean_return) ** 4 for r in returns) / len(returns)
            m2 = variance
            if m2 > 0:
                kurtosis = m4 / (m2 * m2)

        # Drawdown - максимальный убыток от пика
        cumulative = 0
        peak = 0
        max_dd = 0
        for r in returns:
            cumulative += r
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        drawdown_pct = (max_dd / 10000) if returns else 0.0

        # Определить режим волатильности
        if current_volatility < 1.0:
            vol_regime = "low"
        elif current_volatility < 2.0:
            vol_regime = "medium"
        else:
            vol_regime = "high"

        return KellyFeatures(
            recent_win_rate=win_rate,
            std_dev_bps=std_dev,
            kurtosis=kurtosis,
            drawdown_pct=drawdown_pct,
            volatility_regime=vol_regime,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
        )

    def extract_regime_features(
        self,
        adx: float = 25.0,
        di_plus: float = 25.0,
        di_minus: float = 20.0,
        atr_pct: float = 1.5,
        rsi: float = 50.0,
        macd_line: float = 0.0,
        signal_line: float = 0.0,
        volatility_pct: float = 1.5,
        hour_of_day: int = 12,
    ) -> RegimeFeatures:
        """Извлечь признаки для Regime predictor."""
        return RegimeFeatures(
            adx=adx,
            di_plus=di_plus,
            di_minus=di_minus,
            atr_pct=atr_pct,
            rsi=rsi,
            macd_line=macd_line,
            signal_line=signal_line,
            volatility_pct=volatility_pct,
            hour_of_day=hour_of_day,
        )

    def extract_signal_context(
        self,
        signal_ma_crossover: float = 0.0,
        signal_rsi: float = 0.0,
        signal_macd: float = 0.0,
        signal_breakout: float = 0.0,
        signal_volume: float = 0.0,
        market_regime: str = "SIDEWAYS",
        recent_trades: Optional[list[dict]] = None,
    ) -> SignalContext:
        """Извлечь контекст для Signal fusion."""
        # Вычислить confidence на основе волатильности и истории
        trades = recent_trades or []
        win_rate = sum(1 for t in trades if t.get("pnl_usd", 0) > 0) / len(trades) if trades else 0.5

        # Простая уверенность - функция от расстояния от нейтрали
        def signal_to_confidence(sig: float) -> float:
            return min(0.95, 0.5 + abs(sig) * 0.3)

        return SignalContext(
            signal_ma_crossover=signal_ma_crossover,
            signal_rsi=signal_rsi,
            signal_macd=signal_macd,
            signal_breakout=signal_breakout,
            signal_volume=signal_volume,
            confidence_ma=signal_to_confidence(signal_ma_crossover),
            confidence_rsi=signal_to_confidence(signal_rsi),
            confidence_macd=signal_to_confidence(signal_macd),
            confidence_breakout=signal_to_confidence(signal_breakout),
            confidence_volume=signal_to_confidence(signal_volume),
            market_regime=market_regime,
            recent_win_rate=win_rate,
        )

    def extract_spread_features(
        self,
        base_spread_bps: float = 15.0,
        bid_ask_imbalance: float = 0.5,
        order_book_depth: float = 1.0,
        volatility_pct: float = 1.5,
        hour_of_day: int = 12,
    ) -> SpreadFeatures:
        """Извлечь признаки для Spread predictor."""
        # Time of day factor: спреды уже во время низкой ликвидности
        if 2 <= hour_of_day <= 8:  # ночная сессия
            time_factor = 1.2
        elif 8 <= hour_of_day <= 16:  # дневная сессия
            time_factor = 0.9
        else:
            time_factor = 1.0

        return SpreadFeatures(
            base_spread_bps=base_spread_bps,
            bid_ask_imbalance=bid_ask_imbalance,
            order_book_depth=order_book_depth,
            volatility_pct=volatility_pct,
            time_of_day_factor=time_factor,
        )

    def extract_stoploss_context(
        self,
        realized_volatility_pct: float = 1.5,
        atr_pct: float = 1.2,
        market_regime: str = "SIDEWAYS",
        trend_strength: float = 0.5,
        recent_trades: Optional[list[dict]] = None,
    ) -> StopLossContext:
        """Извлечь контекст для StopLoss optimizer."""
        trades = recent_trades or []
        win_rate = sum(1 for t in trades if t.get("pnl_usd", 0) > 0) / len(trades) if trades else 0.5

        return StopLossContext(
            realized_volatility_pct=realized_volatility_pct,
            atr_pct=atr_pct,
            market_regime=market_regime,
            trend_strength=trend_strength,
            recent_win_rate=win_rate,
        )

    def add_trade_return(self, pnl_bps: float) -> None:
        """Добавить результат сделки."""
        self.recent_returns_bps.append(pnl_bps)
        if len(self.recent_returns_bps) > self.max_history:
            self.recent_returns_bps.pop(0)

    def add_signal(self, signal_type: str, value: float) -> None:
        """Добавить значение сигнала."""
        if signal_type in self.recent_signals:
            self.recent_signals[signal_type].append(value)
            if len(self.recent_signals[signal_type]) > self.max_history:
                self.recent_signals[signal_type].pop(0)
