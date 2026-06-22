"""Candlestick pattern scores for ML features (not hard trade rules).

Scores are in [0, 1]: 0 = no match, 1 = strong match. The model decides lift.
Designed for 5m/15m bars; injected into 1m vectors as ``pat5_*`` / ``pat15_*``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

PATTERN_LOOKBACK = 10
"""Maximum bars examined for multi-bar patterns."""

MTF_PATTERN_INTERVALS: tuple[str, ...] = ("5", "15")
"""Higher timeframes used for pattern features on the primary 1m vector."""


class _OHLCBar(Protocol):
    open: float
    high: float
    low: float
    close: float


def _body(open_: float, close: float) -> float:
    return abs(close - open_)


def _range(high: float, low: float) -> float:
    return max(high - low, 0.0)


def _is_bull(bar: _OHLCBar) -> bool:
    return bar.close >= bar.open


def _body_ratio(bar: _OHLCBar) -> float:
    rng = _range(bar.high, bar.low)
    if rng <= 0:
        return 0.0
    return _body(bar.open, bar.close) / rng


def _lower_wick_ratio(bar: _OHLCBar) -> float:
    rng = _range(bar.high, bar.low)
    if rng <= 0:
        return 0.0
    body_bottom = min(bar.open, bar.close)
    return (body_bottom - bar.low) / rng


def _upper_wick_ratio(bar: _OHLCBar) -> float:
    rng = _range(bar.high, bar.low)
    if rng <= 0:
        return 0.0
    body_top = max(bar.open, bar.close)
    return (bar.high - body_top) / rng


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _avg_body(bars: Sequence[_OHLCBar]) -> float:
    if not bars:
        return 0.0
    return sum(_body(b.open, b.close) for b in bars) / len(bars)


def score_doji(bar: _OHLCBar, *, avg_body: float | None = None) -> float:
    """Small body relative to range and recent average body."""
    rng = _range(bar.high, bar.low)
    if rng <= 0:
        return 0.0
    ref = avg_body if avg_body and avg_body > 0 else rng * 0.3
    body = _body(bar.open, bar.close)
    return _clamp01(1.0 - body / max(ref, rng * 0.1))


def score_hammer(bar: _OHLCBar) -> float:
    """Bullish rejection: long lower wick, small upper wick, body in upper third."""
    rng = _range(bar.high, bar.low)
    if rng <= 0:
        return 0.0
    lower = _lower_wick_ratio(bar)
    upper = _upper_wick_ratio(bar)
    body = _body_ratio(bar)
    if lower < 0.55 or upper > 0.25 or body > 0.45:
        return 0.0
    return _clamp01((lower - 0.55) / 0.35 + (0.25 - upper) / 0.25)


def score_hanging_man(bar: _OHLCBar) -> float:
    """Bearish rejection at highs — same shape as hammer after an advance."""
    return score_hammer(bar)


def score_shooting_star(bar: _OHLCBar) -> float:
    """Long upper wick, small lower wick."""
    rng = _range(bar.high, bar.low)
    if rng <= 0:
        return 0.0
    upper = _upper_wick_ratio(bar)
    lower = _lower_wick_ratio(bar)
    body = _body_ratio(bar)
    if upper < 0.55 or lower > 0.25 or body > 0.45:
        return 0.0
    return _clamp01((upper - 0.55) / 0.35 + (0.25 - lower) / 0.25)


def score_engulfing(bars: Sequence[_OHLCBar], *, bullish: bool) -> float:
    """Two-bar engulfing pattern score."""
    if len(bars) < 2:
        return 0.0
    prev, cur = bars[-2], bars[-1]
    prev_body = _body(prev.open, prev.close)
    cur_body = _body(cur.open, cur.close)
    if prev_body <= 0 or cur_body <= 0:
        return 0.0
    if bullish:
        if not (not _is_bull(prev) and _is_bull(cur) and cur.open <= prev.close and cur.close >= prev.open):
            return 0.0
    elif not (_is_bull(prev) and not _is_bull(cur) and cur.open >= prev.close and cur.close <= prev.open):
        return 0.0
    ratio = cur_body / prev_body
    return _clamp01((ratio - 1.0) / 1.5)


def score_morning_star(bars: Sequence[_OHLCBar]) -> float:
    """Three-bar bullish reversal: down, small star, strong up."""
    if len(bars) < 3:
        return 0.0
    a, b, c = bars[-3], bars[-2], bars[-1]
    if _is_bull(a) or not _is_bull(c):
        return 0.0
    star_body = _body(b.open, b.close)
    avg = _avg_body(bars[-10:])
    if star_body > avg * 0.45:
        return 0.0
    gap_down = b.high < a.close
    gap_up = c.open > b.high
    c_body = _body(c.open, c.close)
    if c_body < avg * 0.5:
        return 0.0
    reclaim = c.close > (a.open + a.close) / 2.0
    score = 0.35
    score += 0.25 if gap_down else 0.0
    score += 0.20 if gap_up else 0.0
    score += 0.20 if reclaim else 0.0
    return _clamp01(score)


def score_evening_star(bars: Sequence[_OHLCBar]) -> float:
    """Three-bar bearish reversal: up, small star, strong down."""
    if len(bars) < 3:
        return 0.0
    a, b, c = bars[-3], bars[-2], bars[-1]
    if not _is_bull(a) or _is_bull(c):
        return 0.0
    star_body = _body(b.open, b.close)
    avg = _avg_body(bars[-10:])
    if star_body > avg * 0.45:
        return 0.0
    gap_up = b.low > a.close
    gap_down = c.open < b.low
    c_body = _body(c.open, c.close)
    if c_body < avg * 0.5:
        return 0.0
    reject = c.close < (a.open + a.close) / 2.0
    score = 0.35
    score += 0.25 if gap_up else 0.0
    score += 0.20 if gap_down else 0.0
    score += 0.20 if reject else 0.0
    return _clamp01(score)


def score_three_soldiers(bars: Sequence[_OHLCBar], *, bullish: bool) -> float:
    """Three consecutive strong bodies in one direction."""
    if len(bars) < 3:
        return 0.0
    trio = bars[-3:]
    avg = _avg_body(bars[-10:])
    if avg <= 0:
        return 0.0
    score = 0.0
    for i, bar in enumerate(trio):
        body = _body(bar.open, bar.close)
        if body < avg * 0.55:
            return 0.0
        if bullish and not _is_bull(bar):
            return 0.0
        if not bullish and _is_bull(bar):
            return 0.0
        if i > 0 and bullish and bar.close <= trio[i - 1].close:
            return 0.0
        if i > 0 and not bullish and bar.close >= trio[i - 1].close:
            return 0.0
        score += 0.33
    return _clamp01(score)


def _pattern_keys(prefix: str) -> list[str]:
    p = prefix if prefix.endswith("_") else f"{prefix}_"
    return [
        f"{p}data_present",
        f"{p}lower_wick",
        f"{p}upper_wick",
        f"{p}doji",
        f"{p}hammer",
        f"{p}hanging_man",
        f"{p}shooting_star",
        f"{p}engulfing_bull",
        f"{p}engulfing_bear",
        f"{p}morning_star",
        f"{p}evening_star",
        f"{p}three_white_soldiers",
        f"{p}three_black_crows",
    ]


def zero_pattern_features(*, prefix: str = "pat_") -> dict[str, float]:
    p = prefix if prefix.endswith("_") else f"{prefix}_"
    return dict.fromkeys(_pattern_keys(p), 0.0)


def compute_pattern_features(
    candles: Sequence[_OHLCBar],
    *,
    prefix: str = "pat_",
    min_bars: int = 3,
) -> dict[str, float]:
    """Return pattern scores from the last up to ``PATTERN_LOOKBACK`` bars."""
    out = zero_pattern_features(prefix=prefix)
    p = prefix if prefix.endswith("_") else f"{prefix}_"
    if len(candles) < min_bars:
        return out

    window = list(candles[-PATTERN_LOOKBACK:])
    last = window[-1]
    avg_body = _avg_body(window)

    out[f"{p}data_present"] = 1.0
    out[f"{p}lower_wick"] = _lower_wick_ratio(last)
    out[f"{p}upper_wick"] = _upper_wick_ratio(last)
    out[f"{p}doji"] = score_doji(last, avg_body=avg_body)
    out[f"{p}hammer"] = score_hammer(last)
    out[f"{p}hanging_man"] = score_hanging_man(last)
    out[f"{p}shooting_star"] = score_shooting_star(last)
    out[f"{p}engulfing_bull"] = score_engulfing(window, bullish=True)
    out[f"{p}engulfing_bear"] = score_engulfing(window, bullish=False)
    out[f"{p}morning_star"] = score_morning_star(window)
    out[f"{p}evening_star"] = score_evening_star(window)
    out[f"{p}three_white_soldiers"] = score_three_soldiers(window, bullish=True)
    out[f"{p}three_black_crows"] = score_three_soldiers(window, bullish=False)
    return out
