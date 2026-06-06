"""Feature computation pipeline.

Reads from CandleStore, computes technical indicators, and emits FeatureVectors.
Updates the HealthChecker so the health endpoint reflects feature freshness.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import structlog

from trader.data.candles import CandleStore
from trader.domain.models import FeatureVector
from trader.features.technical import (
    adx,
    atr,
    bb_bandwidth,
    bb_percent_b,
    candle_body_ratio,
    ema_slope,
    ema_value,
    log_return,
    macd,
    realized_volatility,
    returns,
    rsi,
    sma,
    volume_zscore,
)

log = structlog.get_logger(__name__)

# Minimum confirmed candles needed before features can be computed
_MIN_BARS = 30


class FeaturePipeline:
    """Computes FeatureVectors from CandleStore on a fixed interval.

    Args:
        candle_store:   Shared OHLCV store.
        health_checker: HealthChecker to notify on each successful computation.
        interval_s:     How often to recompute (seconds).
    """

    def __init__(
        self,
        candle_store: CandleStore,
        health_checker: Any | None = None,
        interval_s: float = 5.0,
    ) -> None:
        self._store = candle_store
        self._health = health_checker
        self._interval = interval_s
        self._stop_event = asyncio.Event()
        # Last computed vector per (symbol, interval)
        self._latest: dict[tuple[str, str], FeatureVector] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def run(
        self,
        symbols: list[str],
        intervals: list[str],
        symbol_source: Any | None = None,
    ) -> None:
        """Compute features in a loop until ``stop()`` is called.

        Args:
            symbols:       Initial symbol list (used when symbol_source is None).
            intervals:     Candle intervals to compute features for.
            symbol_source: Optional object with an ``active_symbols`` property
                           that returns the current dynamic symbol list (e.g.
                           MarketScreener). When provided, the symbol list is
                           refreshed on every iteration.
        """
        log.info("feature_pipeline.started", symbols=symbols, intervals=intervals)
        while not self._stop_event.is_set():
            active = symbol_source.active_symbols if symbol_source is not None else symbols

            # Compute all (symbol, interval) pairs concurrently
            async def _compute_one(symbol: str, interval: str) -> None:
                try:
                    vec = self.compute(symbol, interval)
                    if vec is not None:
                        self._latest[(symbol, interval)] = vec
                        if self._health is not None:
                            self._health.set_feature_computed_at(datetime.now(tz=UTC))
                except Exception as exc:
                    log.warning(
                        "feature_pipeline.compute_error",
                        symbol=symbol,
                        interval=interval,
                        error=str(exc),
                    )

            tasks = [_compute_one(symbol, interval) for symbol in active for interval in intervals]
            if tasks:
                await asyncio.gather(*tasks)

            try:
                await asyncio.wait_for(
                    asyncio.shield(self._stop_event.wait()),
                    timeout=self._interval,
                )
            except TimeoutError:
                pass

    def stop(self) -> None:
        self._stop_event.set()

    def latest(self, symbol: str, interval: str) -> FeatureVector | None:
        return self._latest.get((symbol, interval))

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(self, symbol: str, interval: str) -> FeatureVector | None:
        """Compute all indicators for one (symbol, interval) pair.

        Returns ``None`` if there are insufficient candles.
        """
        if not self._store.is_ready(symbol, interval, _MIN_BARS):
            return None

        closes = self._store.closes(symbol, interval)
        highs = self._store.highs(symbol, interval)
        lows = self._store.lows(symbol, interval)
        volumes = self._store.volumes(symbol, interval)

        features: dict[str, float] = {}
        missing: list[str] = []

        # --- Returns ---
        for period in [1, 3, 5, 15, 30]:
            key = f"return_{period}"
            val = returns(closes, period)
            if val is not None:
                features[key] = val
            else:
                missing.append(key)

        # --- Log return ---
        val = log_return(closes, 1)
        if val is not None:
            features["log_return_1"] = val
        else:
            missing.append("log_return_1")

        # --- Realized volatility ---
        val = realized_volatility(closes, 20)
        if val is not None:
            features["realized_vol_20"] = val
        else:
            missing.append("realized_vol_20")

        # --- RSI ---
        for period in [14, 21]:
            key = f"rsi_{period}"
            val = rsi(closes, period)
            if val is not None:
                features[key] = val / 100.0  # normalise to [0, 1]
            else:
                missing.append(key)

        # --- MACD ---
        result = macd(closes)
        if result is not None:
            m, s, h = result
            # Normalise by last close price
            price = closes[-1] if closes else 1.0
            features["macd_line"] = m / price if price else 0.0
            features["macd_signal"] = s / price if price else 0.0
            features["macd_hist"] = h / price if price else 0.0
        else:
            missing.extend(["macd_line", "macd_signal", "macd_hist"])

        # --- Bollinger Bands ---
        val_bb = bb_percent_b(closes)
        if val_bb is not None:
            features["bb_pct_b"] = val_bb
        else:
            missing.append("bb_pct_b")

        val_bw = bb_bandwidth(closes)
        if val_bw is not None:
            features["bb_bandwidth"] = val_bw
        else:
            missing.append("bb_bandwidth")

        # --- EMA ---
        for period in [9, 21, 50]:
            key = f"ema_{period}"
            val_ema = ema_value(closes, period)
            if val_ema is not None and closes[-1] > 0:
                features[key] = val_ema / closes[-1] - 1.0  # normalised distance
            else:
                missing.append(key)

        # EMA slope
        for period in [9, 21]:
            key = f"ema_slope_{period}"
            val_slope = ema_slope(closes, period)
            if val_slope is not None:
                features[key] = val_slope
            else:
                missing.append(key)

        # --- SMA distance ---
        val_sma = sma(closes, 20)
        if val_sma is not None and closes[-1] > 0:
            features["sma20_dist"] = (closes[-1] - val_sma) / closes[-1]
        else:
            missing.append("sma20_dist")

        # --- ATR ---
        if len(highs) >= 15 and len(lows) >= 15:
            val_atr = atr(highs, lows, closes, 14)
            if val_atr is not None and closes[-1] > 0:
                features["atr_14_pct"] = val_atr / closes[-1]
            else:
                missing.append("atr_14_pct")
        else:
            missing.append("atr_14_pct")

        # --- ADX ---
        if len(highs) >= 30 and len(lows) >= 30:
            val_adx = adx(highs, lows, closes, 14)
            if val_adx is not None:
                features["adx_14"] = val_adx / 100.0
            else:
                missing.append("adx_14")
        else:
            missing.append("adx_14")

        # --- Volume ---
        val_vz = volume_zscore(volumes, 20)
        if val_vz is not None:
            features["volume_zscore"] = val_vz
        else:
            missing.append("volume_zscore")

        # --- Candle pattern (last bar) ---
        candles = self._store.confirmed(symbol, interval)
        if candles:
            last = candles[-1]
            features["candle_body_ratio"] = candle_body_ratio(last.open, last.high, last.low, last.close)

        # Quality score: fraction of features computed
        total = len(features) + len(missing)
        quality = len(features) / total if total > 0 else 0.0

        if not features:
            return None

        names = sorted(features.keys())
        values = [features[k] for k in names]

        return FeatureVector(
            symbol=symbol,
            timestamp=datetime.now(tz=UTC),
            values=values,
            feature_names=names,
            quality_score=quality,
            lookback_bars=len(closes),
        )
