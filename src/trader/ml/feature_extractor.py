"""Feature extractor - правильный мост между торговой системой и ML моделями.

Извлекает полные наборы признаков для каждой ML модели.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Извлекает полные наборы признаков для ML моделей."""

    def __init__(self):
        self.recent_returns_bps: list[float] = []
        self.recent_spreads_bps: list[float] = []
        self.max_history = 100

    def extract_kelly_features(
        self,
        recent_trades: list[dict[str, Any]],
        current_volatility: float = 1.5,
        hour_of_day: int = 12,
        day_of_week: int = 0,
        days_since_start: int = 1,
        strategy_id: str = "default",
        symbol: str = "BTCUSDT",
    ) -> Any:
        """Извлечь KellyPredictorFeatures для Kelly predictor.

        Возвращает готовый объект KellyPredictorFeatures.
        """
        try:
            from trader.ml.kelly_predictor import KellyPredictorFeatures

            # Вычислить win rate и returns
            wins = sum(1 for trade in recent_trades if trade.get("pnl_usd", 0) > 0)
            win_rate = wins / len(recent_trades) if recent_trades else 0.5

            returns = [trade.get("pnl_bps", 0) for trade in recent_trades]
            winning_returns = [r for r in returns if r > 0]
            losing_returns = [r for r in returns if r < 0]

            if not returns or len(returns) < 2:
                std_dev = 50.0
                skewness = 0.0
                var_95 = 0.0
            else:
                mean_return = sum(returns) / len(returns)
                variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
                std_dev = variance ** 0.5 if variance > 0 else 50.0
                skewness = 0.0
                var_95 = min(returns) if returns else 0.0

            avg_win = sum(winning_returns) / len(winning_returns) if winning_returns else 10.0
            avg_loss = sum(losing_returns) / len(losing_returns) if losing_returns else -5.0
            profit_factor = abs(sum(winning_returns) / sum(losing_returns)) if losing_returns else 1.0
            pnl_trend = (returns[-1] - returns[0]) / len(returns) if returns else 0.0

            current_dd, max_dd = self._calculate_drawdown(returns)

            # Determine volatility_regime from current_volatility
            if current_volatility < 0.5:
                volatility_regime = 0  # low
            elif current_volatility < 1.0:
                volatility_regime = 1  # moderate
            elif current_volatility < 2.0:
                volatility_regime = 2  # high
            else:
                volatility_regime = 3  # extreme

            return KellyPredictorFeatures(
                recent_win_rate=float(win_rate),
                recent_avg_win_bps=float(avg_win),
                recent_avg_loss_bps=float(avg_loss),
                recent_profit_factor=float(profit_factor),
                recent_pnl_trend=float(pnl_trend),
                std_dev_bps=float(std_dev),
                skewness=float(skewness),
                kurtosis=3.0,
                var_95_bps=float(var_95),
                current_drawdown_pct=float(current_dd),
                max_drawdown_pct=float(max_dd),
                drawdown_severity=self._get_drawdown_severity(current_dd),
                in_drawdown=current_dd < 0,
                volatility_regime=volatility_regime,
                hour_of_day=hour_of_day,
                day_of_week=day_of_week,
                days_since_start=days_since_start,
                strategy_id=strategy_id,
                symbol=symbol,
                total_trades=len(recent_trades),
            )
        except Exception as e:
            logger.error(f"extract_kelly_features failed: {e}")
            return None

    def extract_regime_features(
        self,
        rsi: float = 50.0,
        macd_histogram: float = 0.0,
        macd_signal_distance: float = 0.0,
        bb_position: float = 0.5,
        bb_width_pct: float = 2.0,
        realized_vol_pct: float = 1.5,
        volatility_trend: float = 0.0,
        volatility_acceleration: float = 0.0,
        trend_direction: float = 0.0,
        trend_strength: float = 0.5,
        adx: float = 25.0,
        di_plus: float = 25.0,
        di_minus: float = 20.0,
        market_entropy: float = 0.5,
        price_acceleration: float = 0.0,
        momentum_strength: float = 0.5,
        volume_concentration: float = 0.5,
        buy_sell_imbalance: float = 0.0,
        volume_trend: float = 0.0,
        recent_returns_std: float = 1.0,
        recent_returns_skew: float = 0.0,
        regime_duration_candles: int = 10,
    ) -> Any:
        """Извлечь RegimeFeaturesEnhanced для Regime predictor."""
        try:
            from trader.ml.regime_predictor_enhanced import RegimeFeaturesEnhanced

            # Определить volatility_regime
            if realized_vol_pct < 0.5:
                vol_regime = 0
            elif realized_vol_pct < 1.0:
                vol_regime = 1
            elif realized_vol_pct < 2.0:
                vol_regime = 2
            else:
                vol_regime = 3

            return RegimeFeaturesEnhanced(
                rsi=float(rsi),
                macd_histogram=float(macd_histogram),
                macd_signal_distance=float(macd_signal_distance),
                bb_position=float(bb_position),
                bb_width_pct=float(bb_width_pct),
                realized_vol_pct=float(realized_vol_pct),
                volatility_regime=vol_regime,
                volatility_trend=float(volatility_trend),
                volatility_acceleration=float(volatility_acceleration),
                trend_direction=float(trend_direction),
                trend_strength=float(trend_strength),
                adx=float(adx),
                di_plus=float(di_plus),
                di_minus=float(di_minus),
                market_entropy=float(market_entropy),
                price_acceleration=float(price_acceleration),
                momentum_strength=float(momentum_strength),
                volume_profile_concentration=float(volume_concentration),
                buy_sell_imbalance=float(buy_sell_imbalance),
                volume_trend=float(volume_trend),
                recent_returns_std=float(recent_returns_std),
                recent_returns_skew=float(recent_returns_skew),
                regime_duration_candles=regime_duration_candles,
            )
        except Exception as e:
            logger.error(f"extract_regime_features failed: {e}")
            return None

    def extract_signal_context(
        self,
        signal_ma_crossover: float = 0.0,
        signal_rsi: float = 0.0,
        signal_macd: float = 0.0,
        signal_breakout: float = 0.0,
        signal_volume: float = 0.0,
        confidence_ma: float = 0.5,
        confidence_rsi: float = 0.5,
        confidence_macd: float = 0.5,
        confidence_breakout: float = 0.5,
        confidence_volume: float = 0.5,
        ma_rsi_agreement: float = 0.0,
        ma_macd_agreement: float = 0.0,
        rsi_macd_agreement: float = 0.0,
        breakout_volume_agreement: float = 0.0,
        market_regime: str = "SIDEWAYS",
        volatility_pct: float = 1.5,
        recent_win_rate: float = 0.5,
        recent_consecutive_wins: int = 0,
        recent_consecutive_losses: int = 0,
        recent_trades: Optional[list[dict]] = None,
    ) -> Any:
        """Извлечь SignalContextEnhanced для Signal fusion."""
        try:
            from trader.ml.signal_fusion_enhanced import SignalContextEnhanced

            trades = recent_trades or []

            # Вычислить recent accuracies - доля правильных предсказаний каждого сигнала
            # Упрощённый подход: используем win_rate как базу
            win_rate = sum(1 for t in trades if t.get("pnl_usd", 0) > 0) / len(trades) if trades else 0.5

            return SignalContextEnhanced(
                signal_ma_crossover=float(signal_ma_crossover),
                signal_rsi=float(signal_rsi),
                signal_macd=float(signal_macd),
                signal_breakout=float(signal_breakout),
                signal_volume=float(signal_volume),
                confidence_ma=float(confidence_ma),
                confidence_rsi=float(confidence_rsi),
                confidence_macd=float(confidence_macd),
                confidence_breakout=float(confidence_breakout),
                confidence_volume=float(confidence_volume),
                ma_rsi_agreement=float(ma_rsi_agreement),
                ma_macd_agreement=float(ma_macd_agreement),
                rsi_macd_agreement=float(rsi_macd_agreement),
                breakout_volume_agreement=float(breakout_volume_agreement),
                market_regime=market_regime,
                volatility_pct=float(volatility_pct),
                recent_win_rate=float(win_rate),
                recent_consecutive_wins=recent_consecutive_wins,
                recent_consecutive_losses=recent_consecutive_losses,
                ma_recent_accuracy=float(win_rate),
                rsi_recent_accuracy=float(win_rate),
                macd_recent_accuracy=float(win_rate),
                breakout_recent_accuracy=float(win_rate),
                volume_recent_accuracy=float(win_rate),
                signal_conflict_count=0,
                strongest_signal_consensus=max(
                    abs(signal_ma_crossover),
                    abs(signal_rsi),
                    abs(signal_macd),
                    abs(signal_breakout),
                    abs(signal_volume),
                ),
            )
        except Exception as e:
            logger.error(f"extract_signal_context failed: {e}")
            return None

    def extract_spread_features(
        self,
        hour_of_day: int = 12,
        day_of_week: int = 0,
        is_funding_time: bool = False,
        bid_ask_imbalance: float = 0.0,
        bid_ask_ratio: float = 1.0,
        order_book_depth: float = 100.0,
        microstructure_score: float = 0.0,
        spread_trend_5m: float = 0.0,
        spread_volatility: float = 0.5,
        spread_acceleration: float = 0.0,
        price_volatility_bps: float = 10.0,
        volume_ratio: float = 1.0,
        momentum: float = 0.0,
        recent_max_spread_bps: float = 30.0,
        recent_avg_spread_bps: float = 20.0,
        spread_mean_reversion_pct: float = 0.0,
    ) -> Any:
        """Извлечь SpreadPredictorEnhancedFeatures для Spread predictor."""
        try:
            from trader.ml.spread_predictor_enhanced import SpreadPredictorEnhancedFeatures

            return SpreadPredictorEnhancedFeatures(
                hour_of_day=hour_of_day,
                day_of_week=day_of_week,
                is_funding_time=is_funding_time,
                bid_ask_imbalance=float(bid_ask_imbalance),
                bid_ask_ratio=float(bid_ask_ratio),
                order_book_total_depth=float(order_book_depth),
                microstructure_score=float(microstructure_score),
                spread_trend_5m=float(spread_trend_5m),
                spread_volatility=float(spread_volatility),
                spread_acceleration=float(spread_acceleration),
                price_volatility_bps=float(price_volatility_bps),
                volume_ratio=float(volume_ratio),
                momentum=float(momentum),
                recent_max_spread_bps=float(recent_max_spread_bps),
                recent_avg_spread_bps=float(recent_avg_spread_bps),
                spread_mean_reversion_pct=float(spread_mean_reversion_pct),
            )
        except Exception as e:
            logger.error(f"extract_spread_features failed: {e}")
            return None

    def extract_stoploss_context(
        self,
        realized_volatility_pct: float = 1.5,
        atr_pct: float = 1.2,
        volatility_trend: float = 0.0,
        recent_swing_lows: Optional[list[float]] = None,
        recent_swing_highs: Optional[list[float]] = None,
        nearest_support_pct: float = 2.0,
        nearest_resistance_pct: float = 2.0,
        returns_history_pct: Optional[list[float]] = None,
        var_95_pct: float = 2.0,
        cvar_95_pct: float = 3.0,
        market_regime: str = "SIDEWAYS",
        trend_strength: float = 0.5,
        recent_win_rate: float = 0.5,
        hour_of_day: int = 12,
        time_in_trade_minutes: int = 0,
        recent_trades: Optional[list[dict]] = None,
    ) -> Any:
        """Извлечь StopLossContextEnhanced для StopLoss optimizer."""
        try:
            from trader.ml.stoploss_optimizer_enhanced import StopLossContextEnhanced

            trades = recent_trades or []
            win_rate = sum(1 for t in trades if t.get("pnl_usd", 0) > 0) / len(trades) if trades else 0.5

            return StopLossContextEnhanced(
                realized_volatility_pct=float(realized_volatility_pct),
                atr_pct=float(atr_pct),
                volatility_trend=float(volatility_trend),
                recent_swing_lows=recent_swing_lows or [],
                recent_swing_highs=recent_swing_highs or [],
                nearest_support_pct=float(nearest_support_pct),
                nearest_resistance_pct=float(nearest_resistance_pct),
                returns_history_pct=returns_history_pct or [],
                var_95_pct=float(var_95_pct),
                cvar_95_pct=float(cvar_95_pct),
                market_regime=market_regime,
                trend_strength=float(trend_strength),
                recent_win_rate=float(win_rate),
                hour_of_day=hour_of_day,
                time_in_trade_minutes=time_in_trade_minutes,
            )
        except Exception as e:
            logger.error(f"extract_stoploss_context failed: {e}")
            return None

    @staticmethod
    def _calculate_drawdown(returns: list[float]) -> tuple[float, float]:
        """Вычислить текущий и максимальный drawdown.

        Возвращает (current_drawdown_pct, max_drawdown_pct)
        """
        if not returns:
            return 0.0, 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        current_dd = 0.0

        for r in returns:
            cumulative += r
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
            current_dd = dd

        # Конвертировать в проценты
        return (-current_dd / 10000) if returns else 0.0, (-max_dd / 10000) if returns else 0.0

    @staticmethod
    def _get_drawdown_severity(drawdown_pct: float) -> int:
        """Определить серьёзность drawdown (0-4)."""
        if drawdown_pct >= 0:
            return 0
        elif drawdown_pct > -1.0:
            return 1
        elif drawdown_pct > -2.5:
            return 2
        elif drawdown_pct > -5.0:
            return 3
        else:
            return 4

    def add_trade_return(self, pnl_bps: float) -> None:
        """Добавить результат сделки."""
        self.recent_returns_bps.append(pnl_bps)
        if len(self.recent_returns_bps) > self.max_history:
            self.recent_returns_bps.pop(0)

    def add_spread_observation(self, spread_bps: float) -> None:
        """Добавить наблюдение спреда."""
        self.recent_spreads_bps.append(spread_bps)
        if len(self.recent_spreads_bps) > self.max_history:
            self.recent_spreads_bps.pop(0)
