"""In-memory candle (OHLCV) store with fixed-size rolling window.

Thread-safe for asyncio. Not thread-safe for multi-threading.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Candle:
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    confirm: bool = False  # True when bar is closed


_DEFAULT_MAXLEN = 500


class CandleStore:
    """Fixed-size rolling window of OHLCV candles per (symbol, interval).

    - Stores up to ``max_bars`` confirmed candles per key.
    - The last candle in the deque may be unconfirmed (live bar update).
    - Confirmed candles are appended; unconfirmed overwrite the tail.
    """

    def __init__(self, max_bars: int = _DEFAULT_MAXLEN) -> None:
        self._max_bars = max_bars
        # key: (symbol, interval) → deque[Candle]
        self._data: dict[tuple[str, str], collections.deque[Candle]] = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, symbol: str, interval: str, candle: Candle) -> None:
        """Add or update a candle.

        If the last stored candle has the same ``open_time``, it is
        replaced (live bar update). Otherwise, the candle is appended.

        Asyncio-safety: this method contains no ``await`` and therefore
        no coroutine yield point.  Under asyncio's cooperative scheduling
        the entire body executes atomically between any two event-loop
        ticks, so concurrent callers cannot interleave.  No lock is needed.
        """
        key = (symbol.upper(), interval)
        if key not in self._data:
            self._data[key] = collections.deque(maxlen=self._max_bars)

        buf = self._data[key]
        if buf and buf[-1].open_time == candle.open_time:
            # Update in-progress bar
            buf[-1] = candle
        else:
            buf.append(candle)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def candles(self, symbol: str, interval: str) -> list[Candle]:
        """Return all stored candles (oldest first)."""
        key = (symbol.upper(), interval)
        return list(self._data.get(key, []))

    def confirmed(self, symbol: str, interval: str) -> list[Candle]:
        """Return only confirmed (closed) candles."""
        return [c for c in self.candles(symbol, interval) if c.confirm]

    def latest(self, symbol: str, interval: str, n: int) -> list[Candle]:
        """Return the last *n* confirmed candles, oldest-first."""
        conf = self.confirmed(symbol, interval)
        return conf[-n:] if len(conf) >= n else conf

    def closes(self, symbol: str, interval: str, n: int | None = None) -> list[float]:
        """Closing prices of confirmed candles, newest-last."""
        data = self.confirmed(symbol, interval)
        if n is not None:
            data = data[-n:]
        return [c.close for c in data]

    def highs(self, symbol: str, interval: str, n: int | None = None) -> list[float]:
        data = self.confirmed(symbol, interval)
        if n is not None:
            data = data[-n:]
        return [c.high for c in data]

    def lows(self, symbol: str, interval: str, n: int | None = None) -> list[float]:
        data = self.confirmed(symbol, interval)
        if n is not None:
            data = data[-n:]
        return [c.low for c in data]

    def volumes(self, symbol: str, interval: str, n: int | None = None) -> list[float]:
        data = self.confirmed(symbol, interval)
        if n is not None:
            data = data[-n:]
        return [c.volume for c in data]

    def count(self, symbol: str, interval: str, confirmed_only: bool = True) -> int:
        if confirmed_only:
            return len(self.confirmed(symbol, interval))
        key = (symbol.upper(), interval)
        return len(self._data.get(key, []))

    def is_ready(self, symbol: str, interval: str, min_bars: int) -> bool:
        """True when at least *min_bars* confirmed candles are available."""
        return self.count(symbol, interval, confirmed_only=True) >= min_bars

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def symbols(self) -> list[str]:
        return list({sym for sym, _ in self._data})

    def intervals(self, symbol: str) -> list[str]:
        return [iv for sym, iv in self._data if sym == symbol.upper()]

    def __repr__(self) -> str:
        parts = {f"{s}/{iv}": len(buf) for (s, iv), buf in self._data.items()}
        return f"CandleStore({parts})"


def candle_from_kline_event(event: object) -> Candle:
    """Convert a ``KlineEvent`` domain event to a ``Candle``.

    Works with any object that has the expected attributes.
    """
    return Candle(
        open_time=event.open_time,  # type: ignore[attr-defined]
        open=float(event.open),  # type: ignore[attr-defined]
        high=float(event.high),  # type: ignore[attr-defined]
        low=float(event.low),  # type: ignore[attr-defined]
        close=float(event.close),  # type: ignore[attr-defined]
        volume=float(event.volume),  # type: ignore[attr-defined]
        confirm=event.confirm,  # type: ignore[attr-defined]
    )
