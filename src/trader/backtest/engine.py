"""Candle-replay backtest engine for rule-based strategies."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from trader.backtest.metrics import PerformanceMetrics, compute_metrics
from trader.data.candles import Candle, CandleStore
from trader.domain.enums import OrderSide
from trader.domain.models import FeatureVector
from trader.features.technical import adx, atr, ema, macd, rsi, volume_zscore
from trader.risk.net_edge import NetEdgeParams
from trader.strategies.base import BaseStrategy


@dataclass(frozen=True)
class BacktestConfig:
    initial_balance_usd: float = 10_000.0
    risk_pct: float = 0.01
    taker_fee_pct: float = 0.055
    expected_slippage_pct: float = 0.03
    spread_bps: float = 5.0
    funding_buffer_pct: float = 0.01
    safety_margin_pct: float = 0.01
    max_positions: int = 1


@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    gross_pnl_pct: float
    net_pnl_pct: float
    exit_reason: str
    # Trade's actual contribution to account equity, scaled by the position's
    # notional fraction of equity — unlike net_pnl_pct (the raw directional
    # price move), this is what compute_metrics must use to stay consistent
    # with equity_curve.
    equity_pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    trades: list[TradeRecord] = field(default_factory=list)
    metrics: PerformanceMetrics | None = None
    equity_curve: list[float] = field(default_factory=list)


class BacktestEngine:
    """Replay OHLCV candles through a strategy and simulate TP/SL exits."""

    def __init__(self, strategy: BaseStrategy, config: BacktestConfig | None = None) -> None:
        self._strategy = strategy
        self._config = config or BacktestConfig()

    def run(
        self,
        closes: list[float],
        highs: list[float],
        lows: list[float],
        volumes: list[float],
        *,
        symbol: str = "BTCUSDT",
        interval: str = "1",
    ) -> BacktestResult:
        if not (len(closes) == len(highs) == len(lows) == len(volumes)):
            raise ValueError("OHLCV series must have equal length")
        if len(closes) < 60:
            return BacktestResult(metrics=compute_metrics([]))

        store = CandleStore(max_bars=len(closes) + 10)
        base = datetime(2025, 1, 1, tzinfo=UTC)
        for i, (close, high, low, vol) in enumerate(zip(closes, highs, lows, volumes, strict=True)):
            prev = closes[i - 1] if i > 0 else close
            store.add(
                symbol,
                interval,
                Candle(
                    open_time=base + timedelta(minutes=i),
                    open=prev,
                    high=high,
                    low=low,
                    close=close,
                    volume=vol,
                    confirm=True,
                ),
            )

        cost = NetEdgeParams(
            taker_fee_pct=self._config.taker_fee_pct,
            expected_slippage_pct=self._config.expected_slippage_pct,
            max_spread_bps=self._config.spread_bps,
            funding_buffer_pct=self._config.funding_buffer_pct,
            safety_margin_pct=self._config.safety_margin_pct,
        )

        trades: list[TradeRecord] = []
        equity = self._config.initial_balance_usd
        equity_curve = [equity]
        open_trade: dict[str, Any] | None = None

        for idx in range(30, len(closes)):
            price = closes[idx]
            if open_trade is not None:
                exit_info = self._check_exit(open_trade, highs[idx], lows[idx], idx)
                if exit_info is not None:
                    exit_price, reason = exit_info
                    gross = self._directional_pnl_pct(open_trade["side"], open_trade["entry"], exit_price)
                    net = gross - self._round_trip_cost_pct(cost, spread_bps=self._config.spread_bps)
                    notional = equity * self._config.risk_pct / max(open_trade["sl_dist_pct"], 1e-9)
                    notional = min(notional, equity * 0.3)
                    equity_before = equity
                    dollar_pnl = notional * (net / 100.0)
                    equity += dollar_pnl
                    equity_pnl_pct = (dollar_pnl / equity_before * 100.0) if equity_before > 0 else 0.0
                    trades.append(
                        TradeRecord(
                            symbol=symbol,
                            side=open_trade["side"],
                            entry_idx=open_trade["entry_idx"],
                            exit_idx=idx,
                            entry_price=open_trade["entry"],
                            exit_price=exit_price,
                            gross_pnl_pct=gross,
                            net_pnl_pct=net,
                            exit_reason=reason,
                            equity_pnl_pct=equity_pnl_pct,
                        )
                    )
                    equity_curve.append(equity)
                    open_trade = None
                continue

            vec = self._build_feature_vector(
                symbol, closes[: idx + 1], highs[: idx + 1], lows[: idx + 1], volumes[: idx + 1]
            )
            if vec is None:
                continue

            proposal = self._strategy.evaluate(vec, price, equity)
            if proposal is None or proposal.take_profit is None or proposal.stop_loss is None:
                continue

            entry = float(proposal.entry_price)
            tp = float(proposal.take_profit)
            sl = float(proposal.stop_loss)
            side = proposal.side.value
            if side == OrderSide.BUY.value:
                tp_dist = (tp - entry) / entry
                sl_dist = (entry - sl) / entry
            else:
                tp_dist = (entry - tp) / entry
                sl_dist = (sl - entry) / entry

            open_trade = {
                "side": side,
                "entry": entry,
                "tp": tp,
                "sl": sl,
                "entry_idx": idx,
                "sl_dist_pct": sl_dist,
                "tp_dist_pct": tp_dist,
            }

        metrics = compute_metrics([t.equity_pnl_pct for t in trades])
        return BacktestResult(trades=trades, metrics=metrics, equity_curve=equity_curve)

    @staticmethod
    def _round_trip_cost_pct(cost: NetEdgeParams, *, spread_bps: float) -> float:
        return (
            cost.taker_fee_pct * 2.0
            + spread_bps / 100.0
            + cost.expected_slippage_pct * 2.0
            + cost.funding_buffer_pct
            + cost.safety_margin_pct
        )

    @staticmethod
    def _directional_pnl_pct(side: str, entry: float, exit_price: float) -> float:
        if side == OrderSide.BUY.value:
            return (exit_price - entry) / entry * 100.0
        return (entry - exit_price) / entry * 100.0

    @staticmethod
    def _check_exit(
        trade: dict[str, Any],
        high: float,
        low: float,
        idx: int,
    ) -> tuple[float, str] | None:
        del idx
        side = trade["side"]
        tp = trade["tp"]
        sl = trade["sl"]
        if side == OrderSide.BUY.value:
            if low <= sl:
                return sl, "stop_loss"
            if high >= tp:
                return tp, "take_profit"
        else:
            if high >= sl:
                return sl, "stop_loss"
            if low <= tp:
                return tp, "take_profit"
        return None

    @staticmethod
    def _build_feature_vector(
        symbol: str,
        closes: list[float],
        highs: list[float],
        lows: list[float],
        volumes: list[float],
    ) -> FeatureVector | None:
        ema9_series = ema(closes, 9)
        ema21_series = ema(closes, 21)
        if len(ema9_series) < 2 or len(ema21_series) < 2:
            return None

        price = closes[-1]
        if price <= 0:
            return None
        ema9 = ema9_series[-1]
        ema21 = ema21_series[-1]
        ema9_dist = ema9 / price - 1.0
        ema21_dist = ema21 / price - 1.0
        ema9_slope = (ema9_series[-1] - ema9_series[-2]) / price

        rsi14 = rsi(closes, 14)
        macd_vals = macd(closes)
        atr14 = atr(highs, lows, closes, 14)
        adx14 = adx(highs, lows, closes, 14)
        vol_z = volume_zscore(volumes, 20)
        if rsi14 is None or macd_vals is None or atr14 is None or adx14 is None:
            return None

        _, _, macd_hist = macd_vals
        log_return_1 = math.log(closes[-1] / closes[-2]) if closes[-2] > 0 and closes[-1] > 0 else 0.0

        names = [
            "ema_9",
            "ema_21",
            "ema_slope_9",
            "rsi_14",
            "macd_hist",
            "log_return_1",
            "volume_zscore",
            "atr_14_pct",
            "adx_14",
            "funding_rate_bps",
            "oi_change_pct_60m",
        ]
        values = [
            ema9_dist,
            ema21_dist,
            ema9_slope,
            rsi14 / 100.0,
            macd_hist,
            log_return_1,
            vol_z if vol_z is not None else 0.0,
            atr14 / price,
            adx14 / 100.0,
            0.0,
            0.0,
        ]
        return FeatureVector(
            feature_id=uuid.uuid4(),
            symbol=symbol,
            timestamp=datetime.now(tz=UTC),
            values=values,
            feature_names=names,
            quality_score=1.0,
            lookback_bars=len(closes),
        )


def generate_synthetic_trend(
    n: int = 500, *, start: float = 100.0, drift: float = 0.0008, noise: float = 0.002
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Generate synthetic trending OHLCV for backtest smoke tests."""
    closes: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    volumes: list[float] = []
    price = start
    for i in range(n):
        shock = math.sin(i / 17.0) * noise
        price = max(1.0, price * (1.0 + drift + shock))
        spread = price * 0.001
        high = price + spread
        low = price - spread
        closes.append(price)
        highs.append(high)
        lows.append(low)
        volumes.append(100.0 + (i % 7) * 20.0)
    return closes, highs, lows, volumes
