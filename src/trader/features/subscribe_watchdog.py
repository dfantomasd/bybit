"""Watchdog for screener-driven WS subscriptions.

Tracks symbols waiting for their first 1m kline after subscribe and triggers
retry / reconnect when ``SCREENER_SUBSCRIBE_TIMEOUT_SECONDS`` elapses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SubscribeWatchdog:
    """Per-symbol WS subscribe confirmation tracker."""

    timeout_s: float
    max_retries: int = 3
    _pending: dict[str, float] = field(default_factory=dict)
    _retry_counts: dict[str, int] = field(default_factory=dict)
    timeouts_total: int = 0
    retries_total: int = 0
    reconnects_total: int = 0

    def register(self, symbols: list[str], *, now: float | None = None) -> None:
        ts = now if now is not None else time.monotonic()
        for raw in symbols:
            sym = str(raw).upper()
            if sym:
                self._pending[sym] = ts

    def confirm_ws_kline(self, symbol: str, interval: str) -> None:
        if str(interval) != "1":
            return
        sym = str(symbol).upper()
        self._pending.pop(sym, None)
        self._retry_counts.pop(sym, None)

    def expired(self, *, now: float | None = None) -> list[str]:
        ts = now if now is not None else time.monotonic()
        return [sym for sym, started in self._pending.items() if ts - started > self.timeout_s]

    def mark_retry(self, symbol: str, *, now: float | None = None) -> bool:
        """Record a retry; return True when reconnect should be forced."""
        sym = str(symbol).upper()
        self.retries_total += 1
        count = self._retry_counts.get(sym, 0) + 1
        self._retry_counts[sym] = count
        self._pending[sym] = now if now is not None else time.monotonic()
        if count >= self.max_retries:
            self.reconnects_total += 1
            self._retry_counts.pop(sym, None)
            return True
        return False

    def record_timeout(self, symbol: str) -> None:
        self.timeouts_total += 1
        _ = symbol

    def pending_symbols(self) -> list[str]:
        return sorted(self._pending)

    def to_dict(self) -> dict[str, object]:
        return {
            "pending": self.pending_symbols(),
            "timeouts_total": self.timeouts_total,
            "retries_total": self.retries_total,
            "reconnects_total": self.reconnects_total,
            "timeout_s": self.timeout_s,
        }
