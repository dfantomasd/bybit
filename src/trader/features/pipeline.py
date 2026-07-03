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
import hashlib
import json
import math
from datetime import UTC, datetime
from typing import Any

import structlog

from trader.data.candles import CandleStore
from trader.domain.models import FeatureVector
from trader.features.candle_patterns import (
    MTF_PATTERN_INTERVALS,
    PATTERN_LOOKBACK,
    compute_pattern_features,
    zero_pattern_features,
)
from trader.features.technical import (
    adx,
    atr,
    bb_bandwidth,
    bb_percent_b,
    candle_body_ratio,
    ema_slope,
    ema_value,
    ewma_periods_for_bar_count,
    log_return,
    macd,
    multi_ewma_signal,
    obv,
    realized_volatility,
    returns,
    rsi,
    sma,
    volume_sma_ratio,
    volume_zscore,
    vwap,
)

log = structlog.get_logger(__name__)

# Minimum confirmed candles needed before features can be computed.
# The pipeline emits stable zero/fallback feature values for indicators that
# still need longer lookbacks (for example ema_50), so 40 bars is enough to
# start producing useful vectors and historical seed samples without fragmenting
# the schema.
_MIN_BARS = 40


def _schema_hash(names: list[str]) -> str:
    return hashlib.sha256(json.dumps(names).encode()).hexdigest()[:16]


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
        market_stats_source: Any | None = None,
    ) -> None:
        self._store = candle_store
        self._health = health_checker
        self._orderbook_tracker = orderbook_tracker
        self._market_stats_source = market_stats_source
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
        # Symbols currently being REST-seeded; skip caching until seed completes.
        self._seeding_symbols: set[str] = set()

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
            if vec is not None and symbol not in self._seeding_symbols:
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
                    self._stop_event.wait(),
                    timeout=self._watchdog_interval_s,
                )
            except TimeoutError:
                pass

    def stop(self) -> None:
        self._stop_event.set()

    def begin_symbol_seed(self, symbol: str) -> None:
        """Mark a symbol as mid-REST-seed so cached vectors are not published."""
        self._seeding_symbols.add(symbol)
        self.invalidate_symbol(symbol)

    def end_symbol_seed(self, symbol: str) -> None:
        """Clear the seeding guard after all intervals are loaded and recomputed."""
        self._seeding_symbols.discard(symbol)

    def evict_symbol(self, symbol: str) -> None:
        """Drop all cached vectors and seeding state for a removed symbol."""
        sym = symbol.upper()
        self._seeding_symbols.discard(sym)
        for key in [k for k in self._latest if k[0] == sym]:
            self.evict_cached_vector(key[0], key[1])

    def invalidate_symbol(self, symbol: str) -> None:
        """Remove cached feature vectors for a symbol after its candles are reseeded."""
        from trader.features.source_candle_guard import clear_source_bindings_for_symbol

        keys = [k for k in self._latest if k[0] == symbol]
        for k in keys:
            del self._latest[k]
            self._last_computed_at.pop(k, None)
        clear_source_bindings_for_symbol(symbol)

    def evict_cached_vector(self, symbol: str, interval: str) -> None:
        """Drop one cached vector and its source-candle binding after staleness detection."""
        from trader.features.source_candle_guard import remove_source_binding

        key = (symbol, interval)
        vec = self._latest.pop(key, None)
        self._last_computed_at.pop(key, None)
        if vec is not None:
            remove_source_binding(vec.feature_id)

    def latest(self, symbol: str, interval: str) -> FeatureVector | None:
        if symbol in self._seeding_symbols:
            return None
        return self._latest.get((symbol, interval))

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _apply_candle_pattern_features(self, symbol: str, interval: str, features: dict[str, float]) -> None:
        """Attach pattern scores; primary trading interval gets 5m/15m MTF patterns."""
        min_pat_bars = 3
        if interval in {"1", "1m"}:
            for mtf_interval, prefix in (("5", "pat5_"), ("15", "pat15_")):
                if mtf_interval in MTF_PATTERN_INTERVALS and self._store.is_ready(symbol, mtf_interval, min_pat_bars):
                    mtf_candles = self._store.confirmed(symbol, mtf_interval)
                    features.update(compute_pattern_features(mtf_candles[-PATTERN_LOOKBACK:], prefix=prefix))
                else:
                    features.update(zero_pattern_features(prefix=prefix))
        elif interval in MTF_PATTERN_INTERVALS:
            native = self._store.confirmed(symbol, interval)
            if len(native) >= min_pat_bars:
                features.update(compute_pattern_features(native[-PATTERN_LOOKBACK:], prefix="pat_"))
            else:
                features.update(zero_pattern_features(prefix="pat_"))

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
            price = closes[-1] if closes else 0.0
            if price != 0.0:
                features["macd_line"] = m / price
                features["macd_signal"] = s / price
                features["macd_hist"] = h / price
            else:
                missing.extend(["macd_line", "macd_signal", "macd_hist"])
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

        now = datetime.now(tz=UTC)
        hour = now.hour
        dow = now.weekday()
        features["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        features["hour_cos"] = math.cos(2 * math.pi * hour / 24)
        features["dow_sin"] = math.sin(2 * math.pi * dow / 7)
        features["dow_cos"] = math.cos(2 * math.pi * dow / 7)

        # --- Orderbook microstructure ---
        # Always emitted with a presence flag so every symbol shares ONE feature
        # schema (training requires min_samples within a single schema bucket;
        # conditional features fragment it). 0.0 + ob_data_present=0 lets the
        # model distinguish "no book data" from a genuinely neutral book.
        if self._orderbook_tracker is not None:
            ob_imb = self._orderbook_tracker.latest_imbalance(symbol)
            micro_dev = self._orderbook_tracker.microprice_deviation_bps(symbol)
            imb_trend = self._orderbook_tracker.imbalance_trend_10s(symbol)
            present = ob_imb is not None and micro_dev is not None
            features["ob_data_present"] = 1.0 if present else 0.0
            features["ob_imbalance_l5"] = ob_imb if ob_imb is not None else 0.0
            features["microprice_deviation_bps"] = micro_dev if micro_dev is not None else 0.0
            features["microprice_deviation_bps_clipped"] = max(-20.0, min(20.0, features["microprice_deviation_bps"]))
            features["ob_imbalance_trend_10s"] = imb_trend if imb_trend is not None else 0.0

        # --- Funding rate / open interest (positioning data) ---
        # Same presence-flag pattern as orderbook features: always emitted so
        # every symbol shares ONE feature schema.
        if self._market_stats_source is not None:
            stats = None
            try:
                stats = self._market_stats_source.market_stats(symbol)
            except Exception:  # noqa: BLE001 - cache read must never kill compute
                stats = None
            features["mkt_data_present"] = 1.0 if stats else 0.0
            features["funding_rate_bps"] = stats["funding_rate_bps"] if stats else 0.0
            features["oi_change_pct_60m"] = stats["oi_change_pct_60m"] if stats else 0.0
            features["funding_rate_bps_clipped"] = max(-10.0, min(10.0, features["funding_rate_bps"]))
            features["oi_change_pct_60m_clipped"] = max(-5.0, min(5.0, features["oi_change_pct_60m"]))

        # --- Candle pattern (last bar) ---
        # Always emitted (0.0 fallback) so every symbol shares ONE feature schema.
        candles = self._store.confirmed(symbol, interval)
        if candles:
            last = candles[-1]
            features["candle_body_ratio"] = candle_body_ratio(last.open, last.high, last.low, last.close)
        else:
            features["candle_body_ratio"] = 0.0
            missing.append("candle_body_ratio")

        # --- Multi-tier EWMA directional signal ---
        # Always emitted (0.0 fallback) so every symbol shares ONE feature schema.
        ewma_periods = ewma_periods_for_bar_count(len(closes))
        val_ewma = multi_ewma_signal(closes, periods=ewma_periods)
        features["ewma_tier_signal"] = val_ewma if val_ewma is not None else 0.0
        if val_ewma is None and interval in {"1", "1m", "5", "5m"}:
            missing.append("ewma_tier_signal")

        # --- VWAP distance ---
        # Always emitted (0.0 fallback) so every symbol shares ONE feature schema.
        val_vwap: float | None = None
        if len(highs) >= 14 and len(lows) >= 14 and len(volumes) >= 14:
            _vwap = vwap(highs, lows, closes, volumes, period=14)
            if _vwap is not None and _vwap > 0 and closes:
                val_vwap = (closes[-1] - _vwap) / _vwap * 100.0
        features["vwap_distance_pct"] = val_vwap if val_vwap is not None else 0.0
        if val_vwap is None:
            missing.append("vwap_distance_pct")

        # --- Candlestick pattern scores (5m/15m MTF on 1m; native on 5m/15m) ---
        self._apply_candle_pattern_features(symbol, interval, features)

        # Quality score: fraction of features computed
        total = len(features) + len(missing)
        quality = len(features) / total if total > 0 else 0.0

        if not features:
            return None

        names = sorted(features.keys())
        values = [features[k] for k in names]

        schema_hash = _schema_hash(names)

        if missing:
            log.warning(
                "feature_pipeline.missing_features",
                symbol=symbol,
                interval=interval,
                missing=missing,
                feature_count=len(names),
                schema_hash=schema_hash,
            )
        else:
            log.debug(
                "feature_pipeline.computed",
                symbol=symbol,
                interval=interval,
                feature_count=len(names),
                schema_hash=schema_hash,
            )

        return FeatureVector(
            symbol=symbol,
            timestamp=now,
            values=values,
            feature_names=names,
            quality_score=quality,
            lookback_bars=len(closes),
        )
