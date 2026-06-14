"""Technical indicator computations.

All functions accept plain Python lists of floats and return floats or tuples.
No external dependencies beyond the standard library — numpy is optional.

Design rules
------------
- Every function validates its input length.
- Returns ``None`` when there is insufficient data (caller decides what to do).
- Pure functions: no side effects, no state.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate(series: Sequence[float], min_len: int, name: str) -> None:
    if len(series) < min_len:
        raise ValueError(f"{name}: need at least {min_len} values, got {len(series)}")


# ---------------------------------------------------------------------------
# Returns & Volatility
# ---------------------------------------------------------------------------


def returns(closes: Sequence[float], period: int = 1) -> float | None:
    """Percentage return over ``period`` bars (newest-last).

    Returns ``None`` if there is insufficient data.
    """
    if len(closes) < period + 1:
        return None
    return (closes[-1] - closes[-(period + 1)]) / closes[-(period + 1)]


def log_return(closes: Sequence[float], period: int = 1) -> float | None:
    if len(closes) < period + 1:
        return None
    try:
        return math.log(closes[-1] / closes[-(period + 1)])
    except (ValueError, ZeroDivisionError):
        return None


def realized_volatility(closes: Sequence[float], period: int = 20) -> float | None:
    """Annualised realised volatility from log returns."""
    if len(closes) < period + 1:
        return None
    log_rets = []
    for i in range(1, period + 1):
        idx = -(period + 1 - i)
        prev = closes[idx - 1]
        curr = closes[idx]
        if prev > 0 and curr > 0:
            log_rets.append(math.log(curr / prev))
    if len(log_rets) < 2:
        return None
    std = statistics.stdev(log_rets)
    return std * math.sqrt(365 * 24 * 60)  # annualise for 1-min bars


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------


def ema(series: Sequence[float], period: int) -> list[float]:
    """Exponential moving average (full series, oldest-first)."""
    if len(series) < period:
        return []
    k = 2 / (period + 1)
    result: list[float] = []
    # seed with SMA of first period
    seed = sum(series[:period]) / period
    result.append(seed)
    for val in series[period:]:
        result.append(val * k + result[-1] * (1 - k))
    return result


def ema_value(series: Sequence[float], period: int) -> float | None:
    """Return the latest EMA value."""
    vals = ema(series, period)
    return vals[-1] if vals else None


def sma(series: Sequence[float], period: int) -> float | None:
    """Simple moving average of the last ``period`` values."""
    if len(series) < period:
        return None
    return sum(series[-period:]) / period


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


def rsi(closes: Sequence[float], period: int = 14) -> float | None:
    """Relative Strength Index [0, 100]."""
    if len(closes) < period + 1:
        return None

    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    if len(gains) < period:
        return None

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def macd(
    closes: Sequence[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[float, float, float] | None:
    """Return (macd_line, signal_line, histogram) or None if insufficient data."""
    min_len = slow + signal
    if len(closes) < min_len:
        return None

    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)

    # Align: ema_slow starts at index slow-1 of closes
    # ema_fast starts at index fast-1 of closes
    # We need the difference aligned to ema_slow length
    slow - fast  # ema_slow has `offset` fewer values than ema_fast (via ema)
    # Both ema() results have len(closes) - period + 1 values... wait:
    # ema(closes, period) returns len(closes) - period + 1 values when len >= period
    # ema_fast: len(closes) - fast + 1
    # ema_slow: len(closes) - slow + 1
    # Align to ema_slow length
    diff_count = len(ema_slow)
    ema_fast_aligned = ema_fast[-diff_count:]

    macd_line = [f - s for f, s in zip(ema_fast_aligned, ema_slow, strict=False)]
    if len(macd_line) < signal:
        return None

    signal_ema = ema(macd_line, signal)
    if not signal_ema:
        return None

    macd_val = macd_line[-1]
    sig_val = signal_ema[-1]
    hist = macd_val - sig_val
    return macd_val, sig_val, hist


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


def bollinger_bands(
    closes: Sequence[float],
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[float, float, float] | None:
    """Return (upper, middle, lower) Bollinger Bands or None."""
    if len(closes) < period:
        return None
    window = list(closes[-period:])
    middle = sum(window) / period
    variance = sum((x - middle) ** 2 for x in window) / period
    std = math.sqrt(variance)
    return middle + std_dev * std, middle, middle - std_dev * std


def bb_percent_b(closes: Sequence[float], period: int = 20, std_dev: float = 2.0) -> float | None:
    """Bollinger %B: position of price within the bands [0..1 in-band]."""
    bands = bollinger_bands(closes, period, std_dev)
    if bands is None:
        return None
    upper, middle, lower = bands
    band_width = upper - lower
    if band_width == 0:
        return 0.5
    return (closes[-1] - lower) / band_width


def bb_bandwidth(closes: Sequence[float], period: int = 20, std_dev: float = 2.0) -> float | None:
    """Bandwidth = (upper - lower) / middle."""
    bands = bollinger_bands(closes, period, std_dev)
    if bands is None or bands[1] == 0:
        return None
    upper, middle, lower = bands
    return (upper - lower) / middle


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float | None:
    """Average True Range."""
    n = min(len(highs), len(lows), len(closes))
    if n < period + 1:
        return None

    true_ranges: list[float] = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return None

    # Wilder smoothing
    atr_val = sum(true_ranges[:period]) / period
    for tr in true_ranges[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period
    return atr_val


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------


def adx(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float | None:
    """Average Directional Index [0, 100]."""
    n = min(len(highs), len(lows), len(closes))
    if n < 2 * period + 1:
        return None

    plus_dm: list[float] = []
    minus_dm: list[float] = []
    true_ranges: list[float] = []

    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return None

    def smooth(series: list[float], p: int) -> list[float]:
        result = [sum(series[:p])]
        for v in series[p:]:
            result.append(result[-1] - result[-1] / p + v)
        return result

    atr_s = smooth(true_ranges, period)
    pdm_s = smooth(plus_dm, period)
    mdm_s = smooth(minus_dm, period)

    dx_vals: list[float] = []
    for a, p, m in zip(atr_s, pdm_s, mdm_s, strict=False):
        if a == 0:
            continue
        pdi = 100 * p / a
        mdi = 100 * m / a
        denom = pdi + mdi
        if denom == 0:
            continue
        dx_vals.append(100 * abs(pdi - mdi) / denom)

    if len(dx_vals) < period:
        return None

    adx_val = sum(dx_vals[:period]) / period
    for dx in dx_vals[period:]:
        adx_val = (adx_val * (period - 1) + dx) / period
    return adx_val


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def volume_zscore(volumes: Sequence[float], period: int = 20) -> float | None:
    """Z-score of current volume vs rolling mean/std."""
    if len(volumes) < period:
        return None
    window = list(volumes[-period:])
    mean = sum(window) / period
    variance = sum((v - mean) ** 2 for v in window) / period
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return (volumes[-1] - mean) / std


# ---------------------------------------------------------------------------
# Candle pattern
# ---------------------------------------------------------------------------


def candle_body_ratio(open_: float, high: float, low: float, close: float) -> float:
    """Body size relative to full range [0, 1]."""
    full_range = high - low
    if full_range == 0:
        return 0.0
    return abs(close - open_) / full_range


def upper_wick_ratio(open_: float, high: float, low: float, close: float) -> float:
    full_range = high - low
    if full_range == 0:
        return 0.0
    body_top = max(open_, close)
    return (high - body_top) / full_range


def lower_wick_ratio(open_: float, high: float, low: float, close: float) -> float:
    full_range = high - low
    if full_range == 0:
        return 0.0
    body_bottom = min(open_, close)
    return (body_bottom - low) / full_range


# ---------------------------------------------------------------------------
# EMA slope / trend strength
# ---------------------------------------------------------------------------


def obv(closes: Sequence[float], volumes: Sequence[float]) -> float | None:
    """On-Balance Volume normalised by the mean volume over the window.

    Returns the OBV divided by mean(volumes) so the value is scale-independent
    and comparable across different symbols and time periods.
    """
    n = min(len(closes), len(volumes))
    if n < 2:
        return None
    closes_ = list(closes[-n:])
    volumes_ = list(volumes[-n:])
    running = 0.0
    for i in range(1, n):
        if closes_[i] > closes_[i - 1]:
            running += volumes_[i]
        elif closes_[i] < closes_[i - 1]:
            running -= volumes_[i]
    mean_vol = sum(volumes_) / n
    if mean_vol == 0:
        return 0.0
    return running / mean_vol


def volume_sma_ratio(volumes: Sequence[float], period: int = 20) -> float | None:
    """Ratio of current volume to SMA of volume over ``period`` bars.

    Values > 1 indicate above-average volume; < 1 indicates below-average.
    """
    if len(volumes) < period:
        return None
    window = list(volumes[-period:])
    mean = sum(window) / period
    if mean == 0:
        return 1.0
    return volumes[-1] / mean


def ema_slope(series: Sequence[float], period: int, lookback: int = 3) -> float | None:
    """Slope of the EMA over the last ``lookback`` values (normalised by price)."""
    ema_vals = ema(series, period)
    if len(ema_vals) < lookback + 1:
        return None
    # Simple linear slope over last lookback+1 points
    y = ema_vals[-(lookback + 1) :]
    n = len(y)
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y, strict=False))
    denom = sum((xi - x_mean) ** 2 for xi in x)
    if denom == 0 or y_mean == 0:
        return None
    return (num / denom) / y_mean  # normalise by price level


def multi_ewma_signal(
    closes: Sequence[float],
    periods: tuple[int, ...] = (3, 12, 50, 100, 200),
) -> float | None:
    """Multi-tier EWMA directional signal in [-1, 1].

    Computes the average normalised cross of adjacent EMA tiers.
    Positive = bullish stack, negative = bearish.
    Inspired by Krypto-trading-bot's EWMA_LS strategy.
    """
    if len(closes) < max(periods) + 1:
        return None
    last_price = closes[-1]
    if last_price <= 0:
        return None
    score = 0.0
    count = 0
    for i in range(len(periods) - 1):
        fast = ema_value(closes, periods[i])
        slow = ema_value(closes, periods[i + 1])
        if fast is None or slow is None:
            return None
        score += (fast - slow) / last_price
        count += 1
    if count == 0:
        return None
    raw = score / count
    # Clamp to [-1, 1]: scale by 100 to turn sub-percent differences into
    # meaningful magnitudes, then clamp.
    return max(-1.0, min(1.0, raw * 100))


def vwap(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
    period: int | None = None,
) -> float | None:
    """Volume-Weighted Average Price over the last ``period`` bars.

    Uses typical price: (high + low + close) / 3.
    Returns ``None`` when there is insufficient data or zero total volume.
    """
    n = min(len(highs), len(lows), len(closes), len(volumes))
    if n == 0:
        return None
    if period is not None:
        if n < period:
            return None
        highs = highs[-period:]
        lows = lows[-period:]
        closes = closes[-period:]
        volumes = volumes[-period:]
    total_vol = sum(volumes)
    if total_vol == 0:
        return None
    total_tp_vol = sum(
        ((h + lo + c) / 3.0) * v
        for h, lo, c, v in zip(highs, lows, closes, volumes, strict=False)
    )
    return total_tp_vol / total_vol
