"""Training pipeline for ML-based Kelly predictor.

Collects historical trades, extracts features, labels outcomes, and trains
XGBoost models to predict optimal kelly_fraction and fractional_kelly.

Walk-forward validation ensures models generalize to unseen data.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Single trade with features and outcome."""

    timestamp: datetime
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    pnl_usd: Decimal
    pnl_bps: float  # Profit in basis points
    win_rate_at_time: float  # Win rate before this trade
    drawdown_pct: float  # Drawdown when trade occurred
    strategy_id: str
    symbol: str


class KellyTrainer:
    """Trains Kelly predictor models from historical trades."""

    def __init__(self, min_training_samples: int = 100, lookback_days: int = 90):
        """Initialize trainer.

        Args:
            min_training_samples: Minimum trades required to train models
            lookback_days: Only use trades from last N days
        """
        self.min_training_samples = min_training_samples
        self.lookback_days = lookback_days
        self._last_training_time: Optional[datetime] = None
        self._training_count = 0

    async def train_from_trades(
        self,
        trades: list[dict[str, Any]],
        kelly_predictor: Any = None,
    ) -> tuple[bool, str]:
        """Train Kelly predictor from trades.

        Args:
            trades: List of trade records with entry/exit info
            kelly_predictor: MLKellyPredictor instance to train

        Returns:
            (success, message)
        """
        if kelly_predictor is None:
            return False, "No Kelly predictor provided"

        if len(trades) < self.min_training_samples:
            return False, f"Insufficient trades: {len(trades)} < {self.min_training_samples}"

        try:
            from trader.ml.kelly_predictor import KellyPredictorFeatures

            # Filter recent trades
            cutoff_time = datetime.now(UTC) - timedelta(days=self.lookback_days)
            recent_trades = [
                t
                for t in trades
                if isinstance(t.get("timestamp"), datetime) and t["timestamp"] > cutoff_time
            ]

            if len(recent_trades) < self.min_training_samples:
                return False, f"Insufficient recent trades: {len(recent_trades)} < {self.min_training_samples}"

            # Extract features and labels
            training_data = []
            for i, trade in enumerate(recent_trades):
                features = self._extract_features_from_trade(trade, recent_trades[:i])
                if features is None:
                    continue

                record = {
                    "features": features,
                    "kelly_actual": float(self._calculate_actual_kelly(trade)),
                    "fractional_actual": self._calculate_actual_fractional(trade),
                    "pnl_bps": trade.get("pnl_bps", 0),
                    "win": 1 if trade.get("pnl_bps", 0) > 0 else 0,
                }
                training_data.append(record)

            if len(training_data) < self.min_training_samples:
                return False, f"Insufficient training records: {len(training_data)} < {self.min_training_samples}"

            # Train the predictor
            await kelly_predictor.train(training_data)
            self._last_training_time = datetime.now(UTC)
            self._training_count += 1

            logger.info(
                f"kelly_training.completed: {len(training_data)} samples, count={self._training_count}"
            )
            return True, f"Trained on {len(training_data)} samples"

        except Exception as e:
            logger.error(f"kelly_training.failed: {e}")
            return False, str(e)

    @staticmethod
    def _extract_features_from_trade(
        trade: dict[str, Any],
        prior_trades: list[dict[str, Any]],
    ) -> Optional[Any]:
        """Extract KellyPredictorFeatures from a trade."""
        try:
            from trader.ml.kelly_predictor import KellyPredictorFeatures

            # Recent performance (last 50 prior trades)
            recent = prior_trades[-50:] if prior_trades else []
            recent_wins = sum(1 for t in recent if t.get("pnl_bps", 0) > 0)
            recent_count = len(recent) if recent else 1
            recent_win_rate = recent_wins / recent_count if recent_count > 0 else 0.5

            recent_avg_win_bps = 0.0
            recent_avg_loss_bps = 0.0
            if recent:
                win_amounts = [t.get("pnl_bps", 0) for t in recent if t.get("pnl_bps", 0) > 0]
                loss_amounts = [t.get("pnl_bps", 0) for t in recent if t.get("pnl_bps", 0) < 0]
                recent_avg_win_bps = float(np.mean(win_amounts)) if win_amounts else 10.0
                recent_avg_loss_bps = float(np.mean(loss_amounts)) if loss_amounts else -10.0

            recent_pnl_trend = 0.0
            if len(recent) >= 5:
                recent_sum = sum(t.get("pnl_bps", 0) for t in recent[-5:])
                recent_pnl_trend = recent_sum / (len(recent[-5:]) * 100)
                recent_pnl_trend = max(-1.0, min(1.0, recent_pnl_trend))

            recent_profit_factor = (
                abs(sum(t.get("pnl_bps", 0) for t in recent if t.get("pnl_bps", 0) > 0))
                / abs(sum(t.get("pnl_bps", 0) for t in recent if t.get("pnl_bps", 0) < 0))
                if recent
                else 1.0
            )

            # Distribution
            all_returns = np.array([t.get("pnl_bps", 0) for t in prior_trades]) if prior_trades else np.array([0.0])
            std_dev_bps = float(np.std(all_returns)) if len(all_returns) > 0 else 10.0
            skewness = (
                float(np.mean((all_returns - np.mean(all_returns)) ** 3) / (np.std(all_returns) ** 3))
                if std_dev_bps > 0
                else 0.0
            )
            kurtosis = (
                float(np.mean((all_returns - np.mean(all_returns)) ** 4) / (np.std(all_returns) ** 4))
                if std_dev_bps > 0
                else 3.0
            )
            var_95_bps = float(np.percentile(all_returns, 5)) if len(all_returns) > 0 else 0.0

            # Drawdown
            current_drawdown_pct = float(trade.get("drawdown_pct", 0.0))
            max_drawdown_pct = float(max([t.get("drawdown_pct", 0.0) for t in prior_trades] or [0.0]))

            drawdown_severity = 0
            if max_drawdown_pct < -1.0:
                drawdown_severity = 1
            elif max_drawdown_pct < -3.0:
                drawdown_severity = 2
            elif max_drawdown_pct < -7.0:
                drawdown_severity = 3
            else:
                drawdown_severity = 4

            in_drawdown = current_drawdown_pct < -0.1

            # Time features
            timestamp = trade.get("timestamp", datetime.now(UTC))
            if isinstance(timestamp, datetime):
                hour_of_day = timestamp.hour
                day_of_week = timestamp.weekday()
            else:
                hour_of_day = 12
                day_of_week = 0

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
                current_drawdown_pct=current_drawdown_pct,
                max_drawdown_pct=max_drawdown_pct,
                drawdown_severity=drawdown_severity,
                in_drawdown=in_drawdown,
                volatility_regime=0,  # Would be set from regime_context
                hour_of_day=hour_of_day,
                day_of_week=day_of_week,
                days_since_start=len(prior_trades) // 10 + 1,  # Rough estimate
                strategy_id=trade.get("strategy_id", "unknown"),
                symbol=trade.get("symbol", ""),
                total_trades=len(prior_trades),
            )
        except Exception as e:
            logger.error(f"feature_extraction_failed: {e}")
            return None

    @staticmethod
    def _calculate_actual_kelly(trade: dict[str, Any]) -> float:
        """Calculate actual Kelly fraction achieved in this trade.

        Based on trade size relative to capital and outcome.
        """
        # Rough approximation: position size as % of capital
        position_size_pct = float(trade.get("position_size_pct", 0.02))
        pnl_bps = float(trade.get("pnl_bps", 0))

        # Kelly is position_size that maximizes risk-adjusted return
        # Approximate as the size that was actually used
        return max(0.01, min(0.25, position_size_pct / 100))

    @staticmethod
    def _calculate_actual_fractional(trade: dict[str, Any]) -> float:
        """Calculate actual fractional Kelly (risk multiplier) for this trade."""
        # Fractional Kelly is typically 0.1 - 0.5 (10-50% of full Kelly)
        # Approximate based on observed position sizing
        position_size_pct = float(trade.get("position_size_pct", 0.02))

        # If we used 2% position size with a 10% kelly, that's 0.2 fractional
        if position_size_pct > 0:
            fractional = position_size_pct / 10  # Assuming 10% kelly baseline
            return max(0.1, min(0.5, fractional))
        return 0.25
