"""Feature computation pipeline.

Reads from CandleStore, computes technical indicators, and emits FeatureVectors.

Primary mode: event-driven via ``on_confirmed_candle(symbol, interval)`` —
called by the public WebSocket consumer whenever a confirmed (closed) kline arrives.

Fallback: ``run()`` acts as a staleness watchdog that re-fires computation for any
(symbol, interval) pair that has not been updated within ``stale_threshold_s``
(default 90 s). This covers gaps when a WS message is missed.
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
    obv,
    realized_volatility,
    returns,
    rsi,
    sma,
    volume_sma_ratio,
    volume_zscore,
)

log = structlog.get_logger(__name__)

# Minimum confirmed candles needed before features can be computed
_MIN_BARS = 30


class FeaturePipeline:
    """Computes FeatureVectors from CandleStore.

    Args:
        candle_store:       Shared OHLCV store.
        health_checker:     HealthChecker to notify on each successful computation.
        interval_s:         Legacy parameter (ignored by the watchdog loop; kept for API compat).
        stale_threshold_s:  Seconds after which a (symbol, interval) pair is considered stale.
        watchdog_interval_s: How often the staleness watchdog fires.
    """

    def __init__(
        self,
        candle_store: CandleStore,
        health_checker: Any | None = None,
        interval_s: float = 5.0,
        stale_threshold_s: float = 90.0,
        watchdog_interval_s: float = 60.0,
        orderbook_tracker: Any | None = None,
    ) -> None:
        self._store = candle_store
        self._health = health_checker
        self._orderbook_tracker = orderbook_tracker
        self._stale_threshold_s = stale_threshold_s
        self._watchdog_interval_s = watchdog_interval_s
        self._stop_event = asyncio.Event()
        # Last computed vector per (symbol, interval)
        self._latest: dict[tuple[str, str], FeatureVector] = {}
        # Timestamp of last successful compute per (symbol, interval)
        self._last_computed_at: dict[tuple[str, str], datetime] = {}
        # Active symbol/interval sets (populated by run())
        self._active_symbols: list[str] = []
        self._active_intervals: list[str] = []
        self._symbol_source: Any | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def on_confirmed_candle(self, symbol: str, interval: str) -> FeatureVector | None:
        """Compute features immediately for one (symbol, interval) pair.

        Called by the WS consumer on every confirmed (closed) kline.
        Returns the FeatureVector if enough data is available, else None.
        """
        try:
            vec = self.compute(symbol, interval)
            if vec is not None:
                self._latest[(symbol, interval)] = vec
                self._last_computed_at[(symbol, interval)] = datetime.now(tz=UTC)
                if self._health is not None:
                    self._health.set_feature_computed_at(datetime.now(tz=UTC))
            return vec
        except Exception as exc:
            log.warning(
                "feature_pipeline.compute_error",
                symbol=symbol,
                interval=interval,
                error=str(exc),
            )
            return None

    async def run(
        self,
        symbols: list[str],
        intervals: list[str],
        symbol_source: Any | None = None,
    ) -> None:
        """Staleness watchdog — re-fires compute for pairs not updated recently.

        Args:
            symbols:       Initial symbol list (used when symbol_source is None).
            intervals:     Candle intervals to monitor.
            symbol_source: Optional object with an ``active_symbols`` property
                           (e.g. MarketScreener). When provided, the symbol list is
                           refreshed on every watchdog iteration.
        """
        self._active_symbols = list(symbols)
        self._active_intervals = list(intervals)
        self._symbol_source = symbol_source

        log.info(
            "feature_pipeline.watchdog_started",
            symbols=list(symbols),
            intervals=intervals,
            stale_threshold_s=self._stale_threshold_s,
            watchdog_interval_s=self._watchdog_interval_s,
        )

        _prev_active: list[str] = list(symbols)

        while not self._stop_event.is_set():
            active = symbol_source.active_symbols if symbol_source is not None else symbols
            now = datetime.now(tz=UTC)

            # Log only when the active symbol list actually changes
            if sorted(active) != sorted(_prev_active):
                log.info(
                    "feature_pipeline.symbols_updated",
                    old_symbols=_prev_active,
                    new_symbols=list(active),
                )
                _prev_active = list(active)

            stale_pairs: list[tuple[str, str]] = []
            for symbol in active:
                for interval in intervals:
                    last = self._last_computed_at.get((symbol, interval))
                    if last is None or (now - last).total_seconds() > self._stale_threshold_s:
                        stale_pairs.append((symbol, interval))

            if stale_pairs:
                log.debug("feature_pipeline.watchdog_recomputing", stale_count=len(stale_pairs))
                tasks = [self.on_confirmed_candle(symbol, interval) for symbol, interval in stale_pairs]
                await asyncio.gather(*tasks)

            try:
                await asyncio.wait_for(
                    asyncio.shield(self._stop_event.wait()),
                    timeout=self._watchdog_interval_s,
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

        # --- OBV (On-Balance Volume) ---
        val_obv = obv(closes, volumes)
        if val_obv is not None:
            features["obv_normalized"] = val_obv
        else:
            missing.append("obv_normalized")

        # --- Volume to SMA(20) ratio ---
        val_vol_ratio = volume_sma_ratio(volumes, 20)
        if val_vol_ratio is not None:
            features["volume_ratio_sma20"] = val_vol_ratio
        else:
            missing.append("volume_ratio_sma20")

        # --- Orderbook microstructure (only for symbols with a live L2 feed) ---
        # Stale/missing books contribute to `missing` rather than fake-neutral
        # values, so the model never trains on fabricated orderbook data.
        if self._orderbook_tracker is not None:
            ob_imb = self._orderbook_tracker.latest_imbalance(symbol)
            if ob_imb is not None:
                features["ob_imbalance_l5"] = ob_imb
            else:
                missing.append("ob_imbalance_l5")

            micro_dev = self._orderbook_tracker.microprice_deviation_bps(symbol)
            if micro_dev is not None:
                features["microprice_deviation_bps"] = micro_dev
            else:
                missing.append("microprice_deviation_bps")

            imb_trend = self._orderbook_tracker.imbalance_trend_10s(symbol)
            if imb_trend is not None:
                features["ob_imbalance_trend_10s"] = imb_trend
            else:
                missing.append("ob_imbalance_trend_10s")

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
