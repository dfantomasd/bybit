"""Advanced Kelly Criterion with state-based risk management.

Architecture:
- AdvancedKellySizer: Single class managing all calculations with state
- Unified metrics: All analysis results in AdvancedKellyMetrics
- Configurable parameters: All thresholds adjustable
- Smoothing: Exponential moving average for kelly changes
- History tracking: Remembers recent decisions for trend analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AdvancedKellyConfig:
    """Configuration for advanced Kelly calculations."""

    # Base Kelly bounds
    kelly_min: float = 0.01
    kelly_max: float = 0.25

    # Fractional Kelly adjustments
    fractional_min: float = 0.1  # Minimum 10% Kelly
    fractional_max: float = 0.5  # Maximum 50% Kelly

    # Drawdown severity thresholds
    dd_mild_pct: float = 1.0
    dd_moderate_pct: float = 3.0
    dd_severe_pct: float = 7.0
    dd_extreme_pct: float = 15.0

    # Drawdown multipliers (how much to reduce Kelly)
    dd_multipliers: dict[str, float] = field(default_factory=lambda: {
        "none": 1.0,
        "mild": 0.95,
        "moderate": 0.85,
        "severe": 0.65,
        "extreme": 0.40,
    })

    # Fat tail detection
    fat_tail_kurtosis_threshold: float = 4.0
    fat_tail_kelly_reduction: float = 0.8

    # VaR safety
    var_confidence: float = 0.95  # 95% VaR
    max_acceptable_loss_pct: float = 2.0

    # Smoothing
    ema_alpha: float = 0.2  # Exponential smoothing factor

    # Recovery mode
    recovery_size_reduction_per_step: float = 0.95  # 5% reduction per recovery step


@dataclass
class AdvancedKellyMetrics:
    """Unified metrics result from advanced Kelly analysis."""

    # Basic Kelly
    kelly_base: float
    kelly_adjusted: float  # After all adjustments

    # Breakdown of adjustments
    kelly_after_drawdown: float
    kelly_after_fat_tails: float
    kelly_after_var: float

    # Dynamic fractional
    fractional_kelly: float
    final_position_size: float  # kelly_adjusted * fractional_kelly

    # Analysis components
    current_drawdown_pct: float
    drawdown_severity: str
    in_drawdown: bool
    has_fat_tails: bool
    kurtosis: float
    var_95_bps: float
    cvar_95_bps: float

    # Trend information
    recent_win_rate: float
    recent_equity_trend: float
    momentum: str  # "strong_up", "up", "neutral", "down", "strong_down"

    # Risk assessment
    risk_level: str  # "low", "medium", "high", "critical"
    warnings: list[str] = field(default_factory=list)
    reasoning: str = ""


class AdvancedKellySizer:
    """Single class managing adaptive Kelly sizing with state.

    Maintains history and smooths Kelly changes over time.
    """

    def __init__(self, config: AdvancedKellyConfig | None = None):
        self.config = config or AdvancedKellyConfig()
        self.kelly_history: list[float] = []
        self.fractional_history: list[float] = []
        self.last_kelly_adjusted: float = 0.15  # Start conservative
        self.last_fractional: float = 0.25

    def calculate(
        self,
        win_rate: float,
        avg_win_bps: float,
        avg_loss_bps: float,
        returns_bps: list[float],
        recent_returns_bps: list[float] | None = None,
        recent_equity_trend: float = 0.0,
    ) -> AdvancedKellyMetrics:
        """Calculate advanced Kelly sizing.

        Args:
            win_rate: Overall win rate
            avg_win_bps: Average win in basis points
            avg_loss_bps: Average loss in basis points
            returns_bps: Full return history
            recent_returns_bps: Recent trades for momentum
            recent_equity_trend: Equity curve trend (-1 to 1)

        Returns:
            AdvancedKellyMetrics with all analysis.
        """

        # 1. Base Kelly calculation
        kelly_base = self._calculate_base_kelly(win_rate, avg_win_bps, avg_loss_bps)

        # 2. Analyze current conditions
        dd_analysis = self._analyze_drawdown(returns_bps)
        dist_analysis = self._analyze_distribution(returns_bps)
        recent_analysis = self._analyze_recent(recent_returns_bps or [])

        # 3. Apply adjustments sequentially
        kelly_after_dd = self._apply_drawdown_adjustment(kelly_base, dd_analysis)
        kelly_after_tails = self._apply_fat_tail_adjustment(kelly_after_dd, dist_analysis)
        kelly_after_var = self._apply_var_adjustment(kelly_after_tails, dist_analysis)

        # 4. Smooth Kelly changes (exponential moving average)
        kelly_smoothed = self._smooth_kelly(kelly_after_var)

        # 5. Calculate dynamic fractional
        fractional = self._calculate_fractional_kelly(
            kelly_smoothed,
            dd_analysis,
            dist_analysis,
            recent_analysis,
            recent_equity_trend,
        )

        # 6. Determine risk level and warnings
        risk_level, warnings = self._assess_risk(
            kelly_smoothed,
            fractional,
            dd_analysis,
            dist_analysis,
            recent_analysis,
        )

        # 7. Build result
        final_size = kelly_smoothed * fractional

        momentum = self._classify_momentum(recent_analysis["win_rate"], recent_equity_trend)

        return AdvancedKellyMetrics(
            kelly_base=kelly_base,
            kelly_adjusted=kelly_smoothed,
            kelly_after_drawdown=kelly_after_dd,
            kelly_after_fat_tails=kelly_after_tails,
            kelly_after_var=kelly_after_var,
            fractional_kelly=fractional,
            final_position_size=final_size,
            current_drawdown_pct=dd_analysis["current_pct"],
            drawdown_severity=dd_analysis["severity"],
            in_drawdown=dd_analysis["in_drawdown"],
            has_fat_tails=dist_analysis["has_fat_tails"],
            kurtosis=dist_analysis["kurtosis"],
            var_95_bps=dist_analysis["var_95"],
            cvar_95_bps=dist_analysis["cvar_95"],
            recent_win_rate=recent_analysis["win_rate"],
            recent_equity_trend=recent_equity_trend,
            momentum=momentum,
            risk_level=risk_level,
            warnings=warnings,
            reasoning=self._build_reasoning(dd_analysis, dist_analysis, recent_analysis),
        )

    # ===== Private helper methods =====

    def _calculate_base_kelly(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate base Kelly from statistics."""
        if avg_win <= 0 or win_rate <= 0:
            return self.config.kelly_min

        kelly = (win_rate * avg_win - (1 - win_rate) * abs(avg_loss)) / avg_win
        return max(self.config.kelly_min, min(kelly, self.config.kelly_max))

    def _analyze_drawdown(self, returns: list[float]) -> dict:
        """Analyze drawdown patterns."""
        if not returns or len(returns) < 10:
            return {
                "current_pct": 0.0,
                "max_pct": 0.0,
                "severity": "none",
                "in_drawdown": False,
            }

        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / 10000.0 * 100  # Convert to pct

        current_dd = float(drawdown[-1])
        max_dd = float(np.max(drawdown))

        # Classify severity
        if max_dd < self.config.dd_mild_pct:
            severity = "none"
        elif max_dd < self.config.dd_moderate_pct:
            severity = "mild"
        elif max_dd < self.config.dd_severe_pct:
            severity = "moderate"
        elif max_dd < self.config.dd_extreme_pct:
            severity = "severe"
        else:
            severity = "extreme"

        return {
            "current_pct": current_dd,
            "max_pct": max_dd,
            "severity": severity,
            "in_drawdown": current_dd > 0.1,
        }

    def _analyze_distribution(self, returns: list[float]) -> dict:
        """Analyze return distribution for fat tails."""
        if not returns or len(returns) < 30:
            return {
                "skewness": 0.0,
                "kurtosis": 3.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "has_fat_tails": False,
            }

        arr = np.array(returns)
        mean = np.mean(arr)
        std = np.std(arr)

        if std <= 0:
            return {
                "skewness": 0.0,
                "kurtosis": 3.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "has_fat_tails": False,
            }

        skew = float((np.mean((arr - mean) ** 3) / (std ** 3)))
        kurt = float((np.mean((arr - mean) ** 4) / (std ** 4)))
        var_95 = float(np.percentile(arr, 5))
        losses = arr[arr < 0]
        cvar_95 = float(np.mean(losses)) if len(losses) > 0 else 0.0

        has_tails = kurt > self.config.fat_tail_kurtosis_threshold

        return {
            "skewness": skew,
            "kurtosis": kurt,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "has_fat_tails": has_tails,
        }

    def _analyze_recent(self, recent_returns: list[float]) -> dict:
        """Analyze recent performance for momentum."""
        if not recent_returns or len(recent_returns) == 0:
            return {"win_rate": 0.5, "trend": 0.0, "momentum": "neutral"}

        wins = sum(1 for r in recent_returns if r > 0)
        wr = wins / len(recent_returns) if recent_returns else 0.5

        # Recent trend
        if len(recent_returns) >= 5:
            recent_sum = sum(recent_returns[-5:])
            trend = recent_sum / (len(recent_returns[-5:]) * 100)
            trend = max(-1.0, min(1.0, trend))
        else:
            trend = 0.0

        return {
            "win_rate": wr,
            "trend": trend,
            "sample_size": len(recent_returns),
        }

    def _apply_drawdown_adjustment(self, kelly: float, dd_analysis: dict) -> float:
        """Adjust Kelly for drawdown severity."""
        multiplier = self.config.dd_multipliers.get(dd_analysis["severity"], 0.8)
        if dd_analysis["in_drawdown"]:
            multiplier *= 0.85  # Additional penalty
        return kelly * multiplier

    def _apply_fat_tail_adjustment(self, kelly: float, dist_analysis: dict) -> float:
        """Adjust Kelly for fat tails."""
        if not dist_analysis["has_fat_tails"]:
            return kelly
        return kelly * self.config.fat_tail_kelly_reduction

    def _apply_var_adjustment(self, kelly: float, dist_analysis: dict) -> float:
        """Adjust Kelly to keep worst-case loss within limit."""
        var_95 = dist_analysis["var_95"]
        if var_95 >= 0:
            return kelly

        worst_loss_pct = (var_95 / 10000.0) * kelly
        max_loss = self.config.max_acceptable_loss_pct / 100.0

        if abs(worst_loss_pct) > max_loss:
            kelly_adjusted = kelly * (max_loss / abs(worst_loss_pct))
            return max(self.config.kelly_min, kelly_adjusted)

        return kelly

    def _smooth_kelly(self, kelly_new: float) -> float:
        """Apply exponential smoothing to Kelly changes."""
        kelly_smoothed = (
            self.config.ema_alpha * kelly_new
            + (1 - self.config.ema_alpha) * self.last_kelly_adjusted
        )
        self.last_kelly_adjusted = kelly_smoothed
        self.kelly_history.append(kelly_smoothed)
        return kelly_smoothed

    def _calculate_fractional_kelly(
        self,
        kelly: float,
        dd_analysis: dict,
        dist_analysis: dict,
        recent_analysis: dict,
        equity_trend: float,
    ) -> float:
        """Calculate dynamic fractional Kelly."""
        # Base: 25% Kelly
        frac = 0.25

        # Adjust for recent performance
        recent_wr = recent_analysis["win_rate"]
        if recent_wr > 0.6:
            frac = 0.35
        elif recent_wr < 0.45:
            frac = 0.15

        # Adjust for equity trend
        if equity_trend < -0.02:
            frac *= 0.7
        elif equity_trend > 0.05:
            frac *= 1.1

        # Adjust for fat tails
        if dist_analysis["has_fat_tails"]:
            frac *= 0.8

        # Adjust for drawdown recovery
        if dd_analysis["in_drawdown"]:
            recovery_pct = dd_analysis["current_pct"] / max(dd_analysis["max_pct"], 0.001)
            frac *= (1 - recovery_pct * 0.5)

        frac = max(self.config.fractional_min, min(frac, self.config.fractional_max))
        self.fractional_history.append(frac)
        return frac

    def _assess_risk(
        self,
        kelly: float,
        fractional: float,
        dd_analysis: dict,
        dist_analysis: dict,
        recent_analysis: dict,
    ) -> tuple[str, list[str]]:
        """Assess overall risk level."""
        warnings = []

        if dd_analysis["severity"] == "extreme":
            risk_level = "critical"
            warnings.append(f"Extreme drawdown: {dd_analysis['current_pct']:.1f}%")
        elif dd_analysis["severity"] == "severe":
            risk_level = "high"
        elif dd_analysis["in_drawdown"] and dist_analysis["has_fat_tails"]:
            risk_level = "high"
        elif recent_analysis["win_rate"] < 0.45:
            risk_level = "high"
            warnings.append(f"Low win rate: {recent_analysis['win_rate']:.1%}")
        elif dist_analysis["has_fat_tails"]:
            risk_level = "medium"
            warnings.append(f"Fat tails detected (kurtosis={dist_analysis['kurtosis']:.1f})")
        else:
            risk_level = "low"

        return risk_level, warnings

    def _classify_momentum(self, recent_wr: float, equity_trend: float) -> str:
        """Classify momentum from recent performance."""
        if equity_trend > 0.05 and recent_wr > 0.6:
            return "strong_up"
        elif equity_trend > 0.0 and recent_wr > 0.55:
            return "up"
        elif equity_trend < -0.05 and recent_wr < 0.45:
            return "strong_down"
        elif equity_trend < 0.0 and recent_wr < 0.50:
            return "down"
        else:
            return "neutral"

    def _build_reasoning(self, dd_analysis: dict, dist_analysis: dict, recent_analysis: dict) -> str:
        """Build human-readable explanation."""
        reasons = []

        if dd_analysis["severity"] != "none":
            reasons.append(f"Drawdown {dd_analysis['severity']}: {dd_analysis['current_pct']:.1f}%")

        if dist_analysis["has_fat_tails"]:
            reasons.append(f"Fat tails (kurtosis={dist_analysis['kurtosis']:.1f})")

        if recent_analysis["win_rate"] < 0.5:
            reasons.append(f"Recent performance weak: {recent_analysis['win_rate']:.1%} WR")

        if not reasons:
            reasons.append("Normal market conditions")

        return "; ".join(reasons)
