"""Adapter integrating MLKellyPredictor with RiskManager.

Bridges the gap between available trading data and ML Kelly predictions.
Extracts features, handles fallback, and provides recommended sizing adjustments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KellyAdapterContext:
    """Context available when evaluating Kelly sizing."""

    # Current trading state
    recent_trades: list[dict[str, Any]]  # Last N trades with outcomes
    current_price: Decimal
    recent_returns_bps: list[float]  # Recent returns in basis points
    all_returns_bps: list[float]  # Full return history for distribution analysis

    # Market conditions
    volatility_regime: int  # 0=low, 1=moderate, 2=high, 3=extreme
    current_drawdown_pct: float  # Current drawdown (negative or zero)
    max_drawdown_pct: float  # Peak drawdown (negative or zero)

    # Strategy metadata
    strategy_id: str
    symbol: str
    total_trades: int

    # Temporal
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class KellyAdapter:
    """Adapter for ML Kelly predictor integration with risk sizing."""

    def __init__(self, ml_kelly_predictor: Any = None):
        """Initialize adapter with optional ML predictor.

        Args:
            ml_kelly_predictor: MLKellyPredictor instance, or None for fallback.
        """
        self._predictor = ml_kelly_predictor
        self._last_context: Optional[KellyAdapterContext] = None
        self._last_prediction: Optional[Any] = None

    async def predict_kelly_sizing(
        self,
        context: KellyAdapterContext,
    ) -> tuple[Decimal, Decimal, Optional[str]]:
        """Predict optimal Kelly sizing from context.

        Args:
            context: KellyAdapterContext with trading state and market data.

        Returns:
            (kelly_fraction, fractional_kelly, reasoning)
            Decimals are clamped to safe ranges [0.01-0.25] and [0.1-0.5].
            Returns fallback values if predictor not available.
        """
        self._last_context = context

        if self._predictor is None:
            return self._fallback_kelly(context)

        try:
            from trader.ml.kelly_predictor import KellyPredictorFeatures

            features = self._extract_features(context)
            prediction = await self._predictor.predict(features)

            self._last_prediction = prediction
            return (
                prediction.kelly_fraction,
                prediction.fractional_kelly,
                prediction.reasoning,
            )
        except Exception as e:
            logger.warning(f"kelly_adapter.prediction_failed: {e}, falling back")
            return self._fallback_kelly(context)

    def _extract_features(self, context: KellyAdapterContext) -> Any:
        """Extract KellyPredictorFeatures from context."""
        from trader.ml.kelly_predictor import KellyPredictorFeatures

        # Recent performance (last 50 trades or fewer)
        recent_trades = context.recent_trades[-50:] if context.recent_trades else []
        recent_wins = sum(1 for t in recent_trades if t.get("pnl_bps", 0) > 0)
        recent_count = len(recent_trades) if recent_trades else 1
        recent_win_rate = recent_wins / recent_count if recent_count > 0 else 0.5

        recent_avg_win_bps = 0.0
        recent_avg_loss_bps = 0.0
        win_amounts: list[float] = []
        loss_amounts: list[float] = []
        if recent_trades:
            win_amounts = [t.get("pnl_bps", 0) for t in recent_trades if t.get("pnl_bps", 0) > 0]
            loss_amounts = [t.get("pnl_bps", 0) for t in recent_trades if t.get("pnl_bps", 0) < 0]
            recent_avg_win_bps = np.mean(win_amounts) if win_amounts else 10.0
            recent_avg_loss_bps = np.mean(loss_amounts) if loss_amounts else -10.0

        recent_profit_factor = (
            abs(sum(win_amounts)) / abs(sum(loss_amounts)) if win_amounts and loss_amounts else 1.0
        )

        recent_pnl_trend = 0.0
        if len(context.recent_returns_bps) >= 5:
            recent_sum = sum(context.recent_returns_bps[-5:])
            recent_pnl_trend = recent_sum / (len(context.recent_returns_bps[-5:]) * 100)
            recent_pnl_trend = max(-1.0, min(1.0, recent_pnl_trend))

        # Volatility and distribution
        all_returns = np.array(context.all_returns_bps) if context.all_returns_bps else np.array([0.0])
        std_dev_bps = float(np.std(all_returns)) if len(all_returns) > 0 else 10.0
        skewness = float(np.mean((all_returns - np.mean(all_returns)) ** 3) / (np.std(all_returns) ** 3)) if std_dev_bps > 0 else 0.0
        kurtosis = float(np.mean((all_returns - np.mean(all_returns)) ** 4) / (np.std(all_returns) ** 4)) if std_dev_bps > 0 else 3.0
        var_95_bps = float(np.percentile(all_returns, 5)) if len(all_returns) > 0 else 0.0

        # Drawdown state
        drawdown_severity = 0
        if context.max_drawdown_pct < -1.0:
            drawdown_severity = 1
        elif context.max_drawdown_pct < -3.0:
            drawdown_severity = 2
        elif context.max_drawdown_pct < -7.0:
            drawdown_severity = 3
        else:
            drawdown_severity = 4

        in_drawdown = context.current_drawdown_pct < -0.1

        # Time features
        now = datetime.now(UTC)
        hour_of_day = now.hour
        day_of_week = now.weekday()
        days_since_start = 1  # Would need start time in context for proper calculation

        return KellyPredictorFeatures(
            recent_win_rate=recent_win_rate,
            recent_avg_win_bps=recent_avg_win_bps,
            recent_avg_loss_bps=recent_avg_loss_bps,
            recent_profit_factor=recent_profit_factor,
            recent_pnl_trend=recent_pnl_trend,
            std_dev_bps=std_dev_bps,
            skewness=skewness,
            kurtosis=kurtosis,
            var_95_bps=var_95_bps,
            current_drawdown_pct=context.current_drawdown_pct,
            max_drawdown_pct=context.max_drawdown_pct,
            drawdown_severity=drawdown_severity,
            in_drawdown=in_drawdown,
            volatility_regime=context.volatility_regime,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            days_since_start=days_since_start,
            strategy_id=context.strategy_id,
            symbol=context.symbol,
            total_trades=context.total_trades,
        )

    @staticmethod
    def _fallback_kelly(context: KellyAdapterContext) -> tuple[Decimal, Decimal, str]:
        """Fallback to statistical Kelly when model unavailable."""
        # Base Kelly from available data
        recent_trades = context.recent_trades[-50:] if context.recent_trades else []

        if not recent_trades:
            return Decimal("0.10"), Decimal("0.25"), "No trade history, using conservative defaults"

        wins = sum(1 for t in recent_trades if t.get("pnl_bps", 0) > 0)
        win_rate = wins / len(recent_trades) if recent_trades else 0.5
        win_amounts = [t.get("pnl_bps", 0) for t in recent_trades if t.get("pnl_bps", 0) > 0]
        loss_amounts = [t.get("pnl_bps", 0) for t in recent_trades if t.get("pnl_bps", 0) < 0]

        if not win_amounts or not loss_amounts:
            kelly = Decimal("0.10")
        else:
            avg_win = np.mean(win_amounts)
            avg_loss = abs(np.mean(loss_amounts))
            kelly_value = (win_rate * avg_win - (1 - win_rate) * avg_loss) / max(avg_win, 1.0)
            kelly = Decimal(str(max(0.01, min(0.25, kelly_value))))

        # Fractional Kelly
        frac = Decimal("0.25")
        if win_rate > 0.6:
            frac = Decimal("0.35")
        elif win_rate < 0.45:
            frac = Decimal("0.15")

        if context.current_drawdown_pct < -0.1:
            frac *= Decimal("0.8")

        frac = max(Decimal("0.1"), min(frac, Decimal("0.5")))

        reasoning = f"Statistical Kelly: {win_rate:.1%} WR, {len(recent_trades)} recent trades"
        return kelly, frac, reasoning
