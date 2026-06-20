"""Rolling orderbook analytics per symbol.

OrderbookTracker is the CandleStore analogue for L2 orderbook data: it consumes
OrderBookEvent streams (already emitted by BybitPublicWebSocket) and keeps a
small ring buffer of derived metrics per symbol — imbalance, microprice
deviation — so strategies and the feature pipeline can read them without
touching raw book state.

Snapshots are throttled to at most one per second per symbol so the 10-slot
ring buffer spans ~10 seconds, which is what the imbalance trend feature needs.
The latest metrics are still refreshed on every event.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import structlog

from trader.data.orderbook import compute_depth_imbalance, compute_microprice

log = structlog.get_logger(__name__)

# Ring buffer of throttled snapshots (1/s → ~10 s of history)
_HISTORY_SLOTS = 10
_SNAPSHOT_MIN_INTERVAL_S = 1.0
# Data older than this is considered stale and not served to consumers
_STALENESS_S = 30.0
_IMBALANCE_DEPTH = 5
_TREND_WINDOW_S = 10.0


@dataclass(frozen=True)
class ObSnapshot:
    """Derived orderbook metrics at one point in time."""

    ts: datetime
    imbalance_l5: float
    microprice_deviation_bps: float


class OrderbookTracker:
    """Per-symbol ring buffer of derived orderbook metrics.

    Single-event-loop access only (same threading model as LocalOrderBook):
    ``record`` is called from the WS consumer coroutine, readers run in the
    strategy loop on the same loop.
    """

    def __init__(
        self,
        history_slots: int = _HISTORY_SLOTS,
        staleness_s: float = _STALENESS_S,
    ) -> None:
        self._history: dict[str, deque[ObSnapshot]] = {}
        self._latest: dict[str, ObSnapshot] = {}
        self._history_slots = history_slots
        self._staleness_s = staleness_s

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def record(
        self,
        symbol: str,
        bids: list[list[Decimal]],
        asks: list[list[Decimal]],
        now: datetime | None = None,
    ) -> None:
        """Derive metrics from sorted (best-first) bid/ask levels.

        Empty or one-sided books are ignored — previous metrics simply age out
        via the staleness check rather than being overwritten with garbage.
        """
        if not bids or not asks:
            return

        bid_levels = [(b[0], b[1]) for b in bids]
        ask_levels = [(a[0], a[1]) for a in asks]

        imbalance = compute_depth_imbalance(bid_levels, ask_levels, depth=_IMBALANCE_DEPTH)
        microprice = compute_microprice(bid_levels, ask_levels)
        if imbalance is None or microprice is None:
            return

        mid = (bid_levels[0][0] + ask_levels[0][0]) / 2
        if mid <= 0:
            return
        micro_dev_bps = float((microprice - mid) / mid) * 10_000.0

        ts = now or datetime.now(tz=UTC)
        snap = ObSnapshot(ts=ts, imbalance_l5=imbalance, microprice_deviation_bps=micro_dev_bps)
        if symbol not in self._latest:
            log.info(
                "orderbook.updated",
                symbol=symbol,
                imbalance_l5=round(imbalance, 4),
                microprice_deviation_bps=round(micro_dev_bps, 3),
            )
        self._latest[symbol] = snap

        history = self._history.setdefault(symbol, deque(maxlen=self._history_slots))
        if not history or (ts - history[-1].ts).total_seconds() >= _SNAPSHOT_MIN_INTERVAL_S:
            history.append(snap)

    # ------------------------------------------------------------------
    # Readers return None when data is missing or stale; callers decide whether
    # that should block a signal.
    # ------------------------------------------------------------------

    def _fresh_latest(self, symbol: str, now: datetime | None = None) -> ObSnapshot | None:
        snap = self._latest.get(symbol)
        if snap is None:
            return None
        ref = now or datetime.now(tz=UTC)
        if (ref - snap.ts).total_seconds() > self._staleness_s:
            return None
        return snap

    def latest_imbalance(self, symbol: str) -> float | None:
        """Latest L5 imbalance in [-1, 1], or None when missing/stale."""
        snap = self._fresh_latest(symbol)
        return snap.imbalance_l5 if snap is not None else None

    def microprice_deviation_bps(self, symbol: str) -> float | None:
        """Latest microprice deviation from mid in bps, or None."""
        snap = self._fresh_latest(symbol)
        return snap.microprice_deviation_bps if snap is not None else None

    def imbalance_trend_10s(self, symbol: str) -> float | None:
        """Change of imbalance over the last ~10 seconds, or None.

        Computed as latest minus the oldest snapshot within the trend window;
        needs at least two snapshots spanning >= 2 seconds to be meaningful.
        """
        snap = self._fresh_latest(symbol)
        history = self._history.get(symbol)
        if snap is None or not history:
            return None
        baseline: ObSnapshot | None = None
        for old in history:
            age = (snap.ts - old.ts).total_seconds()
            if age <= _TREND_WINDOW_S:
                baseline = old
                break
        if baseline is None:
            return None
        span = (snap.ts - baseline.ts).total_seconds()
        if span < 2.0:
            return None
        return snap.imbalance_l5 - baseline.imbalance_l5

    def tracked_symbols(self) -> list[str]:
        return sorted(self._latest.keys())
