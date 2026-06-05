"""Tests for LocalOrderBook and analytics helpers."""
from __future__ import annotations

from decimal import Decimal

import pytest

from trader.data.orderbook import (
    LocalOrderBook,
    compute_depth_imbalance,
    compute_microprice,
    compute_top_n_imbalance,
    compute_weighted_midprice,
    detect_abnormal_spread,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_snapshot(bids, asks, update_id=1, seq=1):
    """Build a snapshot data dict matching Bybit V5 format."""
    return {
        "b": [[str(p), str(q)] for p, q in bids],
        "a": [[str(p), str(q)] for p, q in asks],
        "u": update_id,
        "seq": seq,
    }


def make_delta(bids, asks, update_id, seq=None):
    return {
        "b": [[str(p), str(q)] for p, q in bids],
        "a": [[str(p), str(q)] for p, q in asks],
        "u": update_id,
        "seq": seq or update_id,
    }


# ---------------------------------------------------------------------------
# Snapshot tests
# ---------------------------------------------------------------------------


def test_snapshot_initializes_book():
    """apply_snapshot populates bids and asks and marks book valid."""
    ob = LocalOrderBook("BTCUSDT")
    ob.apply_snapshot(make_snapshot(
        bids=[(30000, 1.0), (29999, 2.0)],
        asks=[(30001, 0.5), (30002, 1.5)],
    ))
    assert ob.is_valid()
    assert ob.last_update_id == 1
    assert ob.get_best_bid() == (Decimal("30000"), Decimal("1.0"))
    assert ob.get_best_ask() == (Decimal("30001"), Decimal("0.5"))


def test_new_snapshot_resets_book():
    """A second snapshot completely resets the book."""
    ob = LocalOrderBook("BTCUSDT")
    ob.apply_snapshot(make_snapshot(
        bids=[(30000, 1.0)],
        asks=[(30001, 0.5)],
        update_id=1,
    ))
    ob.apply_snapshot(make_snapshot(
        bids=[(29500, 3.0)],
        asks=[(29600, 2.0)],
        update_id=10,
    ))
    assert ob.is_valid()
    # Old levels gone
    assert ob.get_best_bid() == (Decimal("29500"), Decimal("3.0"))
    assert ob.get_best_ask() == (Decimal("29600"), Decimal("2.0"))
    assert ob.last_update_id == 10


# ---------------------------------------------------------------------------
# Delta tests
# ---------------------------------------------------------------------------


def test_delta_updates_price_level():
    """Delta with non-zero qty updates an existing price level."""
    ob = LocalOrderBook("BTCUSDT")
    ob.apply_snapshot(make_snapshot(
        bids=[(30000, 1.0)],
        asks=[(30001, 0.5)],
        update_id=5,
    ))
    result = ob.apply_delta(make_delta(
        bids=[(30000, 2.5)],
        asks=[],
        update_id=6,
    ))
    assert result is True
    bid = ob.get_best_bid()
    assert bid == (Decimal("30000"), Decimal("2.5"))


def test_delta_removes_zero_qty_level():
    """Delta with qty=0 removes the price level."""
    ob = LocalOrderBook("BTCUSDT")
    ob.apply_snapshot(make_snapshot(
        bids=[(30000, 1.0), (29999, 0.5)],
        asks=[(30001, 0.5)],
        update_id=5,
    ))
    # Remove 30000 bid
    result = ob.apply_delta(make_delta(
        bids=[(30000, 0)],
        asks=[],
        update_id=6,
    ))
    assert result is True
    best_bid = ob.get_best_bid()
    # 30000 removed, best bid is now 29999
    assert best_bid == (Decimal("29999"), Decimal("0.5"))


def test_sequence_gap_invalidates_book():
    """A non-sequential update_id (gap) causes the book to be invalidated."""
    ob = LocalOrderBook("BTCUSDT")
    ob.apply_snapshot(make_snapshot(
        bids=[(30000, 1.0)],
        asks=[(30001, 0.5)],
        update_id=10,
    ))
    # Skip update_id 11 — send 12 directly
    result = ob.apply_delta(make_delta(
        bids=[(30000, 1.5)],
        asks=[],
        update_id=12,
    ))
    assert result is False
    assert not ob.is_valid()


def test_delta_ignored_when_book_invalid():
    """Delta is ignored (returns False) when book is invalid."""
    ob = LocalOrderBook("BTCUSDT")
    # Don't apply snapshot — book starts invalid
    result = ob.apply_delta(make_delta(
        bids=[(30000, 1.0)],
        asks=[],
        update_id=1,
    ))
    assert result is False


# ---------------------------------------------------------------------------
# Accessor tests
# ---------------------------------------------------------------------------


def test_best_bid_ask():
    """get_best_bid / get_best_ask return correct top-of-book levels."""
    ob = LocalOrderBook("ETHUSDT")
    ob.apply_snapshot(make_snapshot(
        bids=[(2000, 5), (1999, 10), (1998, 3)],
        asks=[(2001, 2), (2002, 8)],
    ))
    bid = ob.get_best_bid()
    ask = ob.get_best_ask()
    assert bid[0] == Decimal("2000")
    assert ask[0] == Decimal("2001")


def test_mid_price():
    """get_mid_price returns arithmetic mean of best bid and ask."""
    ob = LocalOrderBook("BTCUSDT")
    ob.apply_snapshot(make_snapshot(
        bids=[(30000, 1)],
        asks=[(30002, 1)],
    ))
    mid = ob.get_mid_price()
    assert mid == Decimal("30001")


def test_spread():
    """get_spread returns best_ask - best_bid."""
    ob = LocalOrderBook("BTCUSDT")
    ob.apply_snapshot(make_snapshot(
        bids=[(30000, 1)],
        asks=[(30002, 1)],
    ))
    spread = ob.get_spread()
    assert spread == Decimal("2")


def test_imbalance_positive():
    """When bid volume >> ask volume, imbalance is positive."""
    ob = LocalOrderBook("BTCUSDT")
    ob.apply_snapshot(make_snapshot(
        bids=[(30000, 10), (29999, 5)],
        asks=[(30001, 1), (30002, 1)],
    ))
    imb = ob.get_imbalance(depth=5)
    assert imb is not None
    assert imb > 0


def test_imbalance_negative():
    """When ask volume >> bid volume, imbalance is negative."""
    ob = LocalOrderBook("BTCUSDT")
    ob.apply_snapshot(make_snapshot(
        bids=[(30000, 1), (29999, 1)],
        asks=[(30001, 10), (30002, 5)],
    ))
    imb = ob.get_imbalance(depth=5)
    assert imb is not None
    assert imb < 0


def test_empty_book_returns_none():
    """Accessors return None when the book is empty."""
    ob = LocalOrderBook("BTCUSDT")
    # Apply snapshot with no data
    ob.apply_snapshot({"b": [], "a": [], "u": 1, "seq": 1})
    assert ob.get_best_bid() is None
    assert ob.get_best_ask() is None
    assert ob.get_mid_price() is None
    assert ob.get_spread() is None
    assert ob.get_imbalance() is None


def test_invalidate():
    """invalidate() marks the book as invalid."""
    ob = LocalOrderBook("BTCUSDT")
    ob.apply_snapshot(make_snapshot(bids=[(30000, 1)], asks=[(30001, 1)]))
    assert ob.is_valid()
    ob.invalidate()
    assert not ob.is_valid()


# ---------------------------------------------------------------------------
# Analytics helpers
# ---------------------------------------------------------------------------


def test_microprice():
    """compute_microprice produces weighted mid using top bid/ask quantities."""
    bids = [(Decimal("30000"), Decimal("2")), (Decimal("29999"), Decimal("1"))]
    asks = [(Decimal("30001"), Decimal("1")), (Decimal("30002"), Decimal("2"))]
    mp = compute_microprice(bids, asks)
    assert mp is not None
    # (30001 * 2 + 30000 * 1) / (2 + 1) = (60002 + 30000) / 3 = 30000.666...
    assert mp == (Decimal("30001") * Decimal("2") + Decimal("30000") * Decimal("1")) / Decimal("3")


def test_weighted_midprice():
    """compute_weighted_midprice returns a reasonable midprice."""
    bids = [(Decimal("30000"), Decimal("1")), (Decimal("29999"), Decimal("1"))]
    asks = [(Decimal("30001"), Decimal("1")), (Decimal("30002"), Decimal("1"))]
    wm = compute_weighted_midprice(bids, asks, depth=2)
    assert wm is not None
    # Should be between best bid and best ask
    assert Decimal("29999") < wm < Decimal("30002")


def test_abnormal_spread_detection():
    """detect_abnormal_spread returns True when spread exceeds threshold."""
    spread = Decimal("50")   # 50 USD spread
    mid = Decimal("1000")    # on a 1000 USD asset = 5% spread
    assert detect_abnormal_spread(spread, mid, threshold_pct=0.5) is True
    # Normal spread
    spread_ok = Decimal("0.1")
    assert detect_abnormal_spread(spread_ok, mid, threshold_pct=0.5) is False


def test_compute_depth_imbalance_empty():
    """compute_depth_imbalance returns None on empty inputs."""
    result = compute_depth_imbalance([], [])
    assert result is None


def test_compute_top_n_imbalance():
    """compute_top_n_imbalance with n=1 uses only the top level."""
    bids = [(Decimal("100"), Decimal("5")), (Decimal("99"), Decimal("100"))]
    asks = [(Decimal("101"), Decimal("1")), (Decimal("102"), Decimal("100"))]
    imb = compute_top_n_imbalance(bids, asks, n=1)
    assert imb is not None
    # bid_vol=5, ask_vol=1 → (5-1)/(5+1) = 4/6 ≈ 0.667
    expected = (5 - 1) / (5 + 1)
    assert abs(imb - expected) < 0.001
