"""Tests for the rolling orderbook analytics tracker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from trader.data.orderbook_tracker import OrderbookTracker

_SYMBOL = "TESTUSDT"


def _levels(prices_qtys: list[tuple[str, str]]) -> list[list[Decimal]]:
    return [[Decimal(p), Decimal(q)] for p, q in prices_qtys]


def _book(bid_qty: str, ask_qty: str) -> tuple[list[list[Decimal]], list[list[Decimal]]]:
    """5-level book around mid 100 with uniform per-side quantities."""
    bids = _levels([(f"{100 - 0.01 * (i + 1):.2f}", bid_qty) for i in range(5)])
    asks = _levels([(f"{100 + 0.01 * (i + 1):.2f}", ask_qty) for i in range(5)])
    return bids, asks


class TestOrderbookTracker:
    def test_imbalance_balanced_book_is_zero(self) -> None:
        tracker = OrderbookTracker()
        bids, asks = _book("10", "10")
        tracker.record(_SYMBOL, bids, asks)
        assert tracker.latest_imbalance(_SYMBOL) == 0.0

    def test_imbalance_bid_heavy_positive(self) -> None:
        tracker = OrderbookTracker()
        bids, asks = _book("30", "10")
        tracker.record(_SYMBOL, bids, asks)
        imb = tracker.latest_imbalance(_SYMBOL)
        assert imb is not None and abs(imb - 0.5) < 1e-9  # (150-50)/200

    def test_microprice_deviation_sign(self) -> None:
        tracker = OrderbookTracker()
        # Heavy bid qty pushes microprice toward the ask → positive deviation
        bids, asks = _book("30", "10")
        tracker.record(_SYMBOL, bids, asks)
        dev = tracker.microprice_deviation_bps(_SYMBOL)
        assert dev is not None and dev > 0

    def test_unknown_symbol_returns_none(self) -> None:
        tracker = OrderbookTracker()
        assert tracker.latest_imbalance("NOPEUSDT") is None
        assert tracker.microprice_deviation_bps("NOPEUSDT") is None
        assert tracker.imbalance_trend_10s("NOPEUSDT") is None

    def test_stale_data_returns_none(self) -> None:
        tracker = OrderbookTracker(staleness_s=30.0)
        bids, asks = _book("10", "10")
        old = datetime.now(tz=UTC) - timedelta(seconds=60)
        tracker.record(_SYMBOL, bids, asks, now=old)
        assert tracker.latest_imbalance(_SYMBOL) is None

    def test_empty_book_ignored(self) -> None:
        tracker = OrderbookTracker()
        tracker.record(_SYMBOL, [], _book("10", "10")[1])
        assert tracker.latest_imbalance(_SYMBOL) is None

    def test_trend_over_10s(self) -> None:
        tracker = OrderbookTracker()
        t0 = datetime.now(tz=UTC) - timedelta(seconds=8)
        bids, asks = _book("10", "10")  # imbalance 0
        tracker.record(_SYMBOL, bids, asks, now=t0)
        bids, asks = _book("30", "10")  # imbalance 0.5
        tracker.record(_SYMBOL, bids, asks, now=t0 + timedelta(seconds=8))
        trend = tracker.imbalance_trend_10s(_SYMBOL)
        assert trend is not None and abs(trend - 0.5) < 1e-9

    def test_trend_requires_min_span(self) -> None:
        tracker = OrderbookTracker()
        t0 = datetime.now(tz=UTC)
        bids, asks = _book("10", "10")
        tracker.record(_SYMBOL, bids, asks, now=t0)
        bids, asks = _book("30", "10")
        tracker.record(_SYMBOL, bids, asks, now=t0 + timedelta(seconds=1))
        # Only ~1 s of history — trend is not meaningful yet
        assert tracker.imbalance_trend_10s(_SYMBOL) is None

    def test_snapshot_throttling_one_per_second(self) -> None:
        tracker = OrderbookTracker()
        t0 = datetime.now(tz=UTC) - timedelta(seconds=5)
        bids, asks = _book("10", "10")
        # 10 events within the same second → a single history slot
        for i in range(10):
            tracker.record(_SYMBOL, bids, asks, now=t0 + timedelta(milliseconds=50 * i))
        assert len(tracker._history[_SYMBOL]) == 1
