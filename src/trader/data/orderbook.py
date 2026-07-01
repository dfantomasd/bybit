"""Local L2 order book and analytics utilities.

LocalOrderBook maintains a real-time L2 book from Bybit V5 WS snapshots + deltas.
The analytics functions operate on raw bid/ask level lists.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LocalOrderBook
# ---------------------------------------------------------------------------


class LocalOrderBook:
    """Maintains a local L2 order book from Bybit snapshots + deltas.

    Thread-safe for single-event-loop async access (no asyncio.Lock needed;
    updates happen only in the WS coroutine).
    """

    def __init__(self, symbol: str, depth: int = 50) -> None:
        self._symbol = symbol
        self._depth = depth

        # price (Decimal) → qty (Decimal), sorted maintained via sort on access
        self._bids: dict[Decimal, Decimal] = {}
        self._asks: dict[Decimal, Decimal] = {}

        self._last_update_id: int = 0
        self._sequence: int = 0
        self._valid: bool = False

    # ------------------------------------------------------------------
    # Snapshot / delta application
    # ------------------------------------------------------------------

    def apply_snapshot(self, data: dict[str, Any]) -> None:
        """Reset the book from a full snapshot message."""
        self._bids = {}
        self._asks = {}

        for level in data.get("b", []):
            price = Decimal(str(level[0]))
            qty = Decimal(str(level[1]))
            if qty == 0:
                _log.warning("orderbook.snapshot_zero_qty_bid", symbol=self._symbol, price=str(price))
            else:
                self._bids[price] = qty

        for level in data.get("a", []):
            price = Decimal(str(level[0]))
            qty = Decimal(str(level[1]))
            if qty == 0:
                _log.warning("orderbook.snapshot_zero_qty_ask", symbol=self._symbol, price=str(price))
            else:
                self._asks[price] = qty

        self._last_update_id = int(data.get("u", 0))
        self._sequence = int(data.get("seq", 0))
        if self._last_update_id == 0:
            _log.warning(
                "orderbook.snapshot_missing_update_id",
                symbol=self._symbol,
                note="gap detection disabled until first delta sets _last_update_id",
            )
        self._valid = True

    def apply_delta(self, data: dict[str, Any]) -> bool:
        """Apply a delta update.  Returns False if sequence gap detected."""
        if not self._valid:
            # Waiting for snapshot — ignore delta
            return False

        new_update_id = int(data.get("u", 0))
        new_seq = int(data.get("seq", 0))

        # Sequence gap check: update_id must be exactly previous + 1
        # (Bybit V5: u is monotonically increasing; gaps indicate dropped frames)
        if self._last_update_id > 0 and new_update_id != self._last_update_id + 1:
            self._valid = False
            return False

        # Apply bid changes
        for level in data.get("b", []):
            price = Decimal(str(level[0]))
            qty = Decimal(str(level[1]))
            if qty == 0:
                self._bids.pop(price, None)
            else:
                self._bids[price] = qty

        # Apply ask changes
        for level in data.get("a", []):
            price = Decimal(str(level[0]))
            qty = Decimal(str(level[1]))
            if qty == 0:
                self._asks.pop(price, None)
            else:
                self._asks[price] = qty

        self._last_update_id = new_update_id
        self._sequence = new_seq
        return True

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_best_bid(self) -> tuple[Decimal, Decimal] | None:
        """Return (price, qty) of the best bid, or None if empty or invalid."""
        if not self._valid or not self._bids:
            return None
        price = max(self._bids)
        return price, self._bids[price]

    def get_best_ask(self) -> tuple[Decimal, Decimal] | None:
        """Return (price, qty) of the best ask, or None if empty or invalid."""
        if not self._valid or not self._asks:
            return None
        price = min(self._asks)
        return price, self._asks[price]

    def get_mid_price(self) -> Decimal | None:
        """Return (best_bid + best_ask) / 2, or None."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is None or ask is None:
            return None
        return (bid[0] + ask[0]) / 2

    def get_spread(self) -> Decimal | None:
        """Return best_ask - best_bid, or None."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is None or ask is None:
            return None
        return ask[0] - bid[0]

    def get_imbalance(self, depth: int = 5) -> float | None:
        """Return (bid_vol - ask_vol) / total_vol for top *depth* levels.

        Result in [-1, 1]; None when the book is empty or invalid.
        """
        if not self._valid or not self._bids or not self._asks:
            return None

        sorted_bids = sorted(self._bids.items(), reverse=True)[:depth]
        sorted_asks = sorted(self._asks.items())[:depth]

        bid_vol = float(sum(q for _, q in sorted_bids))
        ask_vol = float(sum(q for _, q in sorted_asks))
        total = bid_vol + ask_vol
        if total == 0:
            return None
        return (bid_vol - ask_vol) / total

    def is_valid(self) -> bool:
        return self._valid

    def invalidate(self) -> None:
        """Mark the book as invalid (sequence gap or forced reset).

        Clears bid/ask data so stale levels cannot leak if _valid is ever
        reset without going through apply_snapshot().
        """
        self._valid = False
        self._bids.clear()
        self._asks.clear()
        self._last_update_id = 0

    # ------------------------------------------------------------------
    # Internal access for analytics helpers
    # ------------------------------------------------------------------

    def _sorted_bids(self) -> list[tuple[Decimal, Decimal]]:
        return sorted(self._bids.items(), reverse=True)

    def _sorted_asks(self) -> list[tuple[Decimal, Decimal]]:
        return sorted(self._asks.items())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def last_update_id(self) -> int:
        return self._last_update_id

    @property
    def sequence(self) -> int:
        return self._sequence


# ---------------------------------------------------------------------------
# Analytics helpers (operate on raw level lists)
# ---------------------------------------------------------------------------


def compute_microprice(
    bids: list[tuple[Decimal, Decimal]],
    asks: list[tuple[Decimal, Decimal]],
) -> Decimal | None:
    """Compute microprice: weighted mid using best bid/ask quantities.

    microprice = (best_ask * bid_qty + best_bid * ask_qty) / (bid_qty + ask_qty)
    """
    if not bids or not asks:
        return None
    best_bid_p, best_bid_q = bids[0]
    best_ask_p, best_ask_q = asks[0]
    total_q = best_bid_q + best_ask_q
    if total_q == 0:
        return None
    return (best_ask_p * best_bid_q + best_bid_p * best_ask_q) / total_q


def compute_weighted_midprice(
    bids: list[tuple[Decimal, Decimal]],
    asks: list[tuple[Decimal, Decimal]],
    depth: int = 5,
) -> Decimal | None:
    """Volume-weighted average of mid prices across top *depth* levels."""
    if not bids or not asks:
        return None
    bid_slice = bids[:depth]
    ask_slice = asks[:depth]

    total_weight = Decimal(0)
    weighted_sum = Decimal(0)

    for (bp, bq), (ap, aq) in zip(bid_slice, ask_slice, strict=False):
        mid = (bp + ap) / 2
        weight = bq + aq
        weighted_sum += mid * weight
        total_weight += weight

    if total_weight == 0:
        return None
    return weighted_sum / total_weight


def compute_depth_imbalance(
    bids: list[tuple[Decimal, Decimal]],
    asks: list[tuple[Decimal, Decimal]],
    depth: int = 10,
) -> float | None:
    """Return (bid_vol - ask_vol) / (bid_vol + ask_vol) in [-1, 1]."""
    if not bids or not asks:
        return None
    bid_vol = float(sum(q for _, q in bids[:depth]))
    ask_vol = float(sum(q for _, q in asks[:depth]))
    total = bid_vol + ask_vol
    if total == 0:
        return None
    return (bid_vol - ask_vol) / total


def compute_top_n_imbalance(
    bids: list[tuple[Decimal, Decimal]],
    asks: list[tuple[Decimal, Decimal]],
    n: int = 3,
) -> float | None:
    """Return imbalance for top *n* levels only."""
    return compute_depth_imbalance(bids, asks, depth=n)


def detect_abnormal_spread(
    spread: Decimal,
    mid: Decimal,
    threshold_pct: float = 0.5,
) -> bool:
    """Return True if spread > threshold_pct % of mid price."""
    if mid <= 0:
        return False
    spread_pct = float(spread / mid) * 100
    return spread_pct > threshold_pct
