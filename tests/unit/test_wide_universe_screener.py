"""Tests for the multi-tier MarketScreener."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.features.screener import MarketScreener, ScreenerMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ticker(
    symbol: str,
    turnover: float = 50_000_000,
    price: float = 1.0,
    spread_bps: float = 4.0,
    bid_sz: float = 50_000,
    ask_sz: float = 50_000,
    price_pct: float = 0.02,
    cur_listing_phase: str = "",
    bid: float | None = None,
    ask: float | None = None,
) -> dict:
    half_spread = price * spread_bps / 2 / 10_000
    b = bid if bid is not None else price - half_spread
    a = ask if ask is not None else price + half_spread
    return {
        "symbol": symbol,
        "lastPrice": str(price),
        "bid1Price": str(b),
        "ask1Price": str(a),
        "bid1Size": str(bid_sz),
        "ask1Size": str(ask_sz),
        "turnover24h": str(turnover),
        "volume24h": str(turnover / price if price > 0 else 0),
        "price24hPcnt": str(price_pct),
        "curPreListingPhase": cur_listing_phase,
    }


def _make_screener(
    tickers: list[dict],
    *,
    wide_max: int = 80,
    feature_max: int = 30,
    exec_candidates: int = 15,
    min_volume: float = 20_000_000,
    max_spread_bps: float = 8.0,
    min_depth_usd: float = 5_000.0,
    denylist: list[str] | None = None,
    has_open_position: object = None,
    has_pending_order: object = None,
) -> MarketScreener:
    rest = MagicMock()
    rest.get_tickers = AsyncMock(return_value={"result": {"list": tickers}})
    return MarketScreener(
        rest_client=rest,
        wide_max_symbols=wide_max,
        feature_max_symbols=feature_max,
        execution_candidates=exec_candidates,
        min_volume_usd=min_volume,
        max_spread_bps=max_spread_bps,
        min_top_book_depth_usd=min_depth_usd,
        denylist=denylist,
        has_open_position=has_open_position,
        has_pending_order=has_pending_order,
    )


# ---------------------------------------------------------------------------
# Wide universe filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wide_universe_filtering_excludes_low_volume():
    """Symbols below min_volume_usd must not appear in the wide universe."""
    tickers = [
        _ticker("BTCUSDT", turnover=100_000_000),
        _ticker("LOWUSDT", turnover=5_000_000),  # below 20M threshold
    ]
    screener = _make_screener(tickers, min_volume=20_000_000)
    await screener._refresh()

    symbols = screener.feature_universe
    assert "BTCUSDT" in symbols
    assert "LOWUSDT" not in symbols


@pytest.mark.asyncio
async def test_wide_universe_filtering_excludes_high_spread():
    """Symbols with spread above max_spread_bps must be excluded."""
    tickers = [
        _ticker("GOODUSDT", spread_bps=3.0),  # 3 bps — well within 8 bps limit
        _ticker("WIDESPREADUSDT", spread_bps=50.0),  # 50 bps — exceeds limit
    ]
    screener = _make_screener(tickers, max_spread_bps=8.0)
    await screener._refresh()

    symbols = screener.feature_universe
    assert "GOODUSDT" in symbols
    assert "WIDESPREADUSDT" not in symbols


@pytest.mark.asyncio
async def test_wide_universe_filtering_excludes_low_depth():
    """Symbols with top-book depth below threshold are excluded."""
    tickers = [
        _ticker("DEEPUSDT", bid_sz=50_000, ask_sz=50_000),  # ~100k USD depth
        _ticker("SHALLOWUSDT", bid_sz=1, ask_sz=1),  # ~2 USD depth
    ]
    screener = _make_screener(tickers, min_depth_usd=5_000.0)
    await screener._refresh()

    symbols = screener.feature_universe
    assert "DEEPUSDT" in symbols
    assert "SHALLOWUSDT" not in symbols


@pytest.mark.asyncio
async def test_wide_universe_filtering_excludes_stablecoins():
    """Stablecoins (USDC, DAI, etc.) must never appear."""
    tickers = [
        _ticker("BTCUSDT"),
        _ticker("USDCUSDT"),  # stablecoin base
        _ticker("DAIUSDT"),
    ]
    screener = _make_screener(tickers)
    await screener._refresh()

    symbols = screener.feature_universe
    assert "USDCUSDT" not in symbols
    assert "DAIUSDT" not in symbols


@pytest.mark.asyncio
async def test_wide_universe_filtering_excludes_premarket():
    """Symbols with curPreListingPhase set are pre-market and must be excluded."""
    tickers = [
        _ticker("NORMALUSDT"),
        _ticker("PREMARKETUSDT", cur_listing_phase="PreLaunch"),
    ]
    screener = _make_screener(tickers)
    await screener._refresh()

    symbols = screener.feature_universe
    assert "NORMALUSDT" in symbols
    assert "PREMARKETUSDT" not in symbols


@pytest.mark.asyncio
async def test_wide_universe_denylist():
    """Symbols in the denylist are excluded regardless of other metrics."""
    tickers = [
        _ticker("GOODUSDT"),
        _ticker("DENIEDUSDT"),
    ]
    screener = _make_screener(tickers, denylist=["DENIEDUSDT"])
    await screener._refresh()

    symbols = screener.feature_universe
    assert "GOODUSDT" in symbols
    assert "DENIEDUSDT" not in symbols


@pytest.mark.asyncio
async def test_wide_universe_excludes_non_usdt():
    """Non-USDT symbols must be ignored."""
    tickers = [
        _ticker("BTCUSDT"),
        {
            "symbol": "BTCUSDC",
            "lastPrice": "50000",
            "bid1Price": "49999",
            "ask1Price": "50001",
            "bid1Size": "10",
            "ask1Size": "10",
            "turnover24h": "100000000",
            "price24hPcnt": "0.01",
            "curPreListingPhase": "",
        },
    ]
    screener = _make_screener(tickers)
    await screener._refresh()

    assert "BTCUSDC" not in screener.feature_universe


# ---------------------------------------------------------------------------
# Feature universe limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feature_universe_limit():
    """Feature universe must not exceed feature_max_symbols."""
    tickers = [_ticker(f"COIN{i:02d}USDT") for i in range(50)]
    screener = _make_screener(tickers, wide_max=50, feature_max=20, exec_candidates=10)
    await screener._refresh()

    assert len(screener.feature_universe) <= 20


@pytest.mark.asyncio
async def test_feature_universe_is_subset_of_wide():
    """Every symbol in the feature universe must be in the wide universe."""
    tickers = [_ticker(f"COIN{i:02d}USDT") for i in range(40)]
    screener = _make_screener(tickers, wide_max=30, feature_max=15, exec_candidates=8)
    await screener._refresh()

    wide_syms = {s.symbol for s in screener.wide_universe}
    for sym in screener.feature_universe:
        assert sym in wide_syms


# ---------------------------------------------------------------------------
# Execution candidates limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_candidates_limit():
    """Execution candidates must not exceed execution_candidates count."""
    tickers = [_ticker(f"COIN{i:02d}USDT") for i in range(40)]
    screener = _make_screener(tickers, wide_max=40, feature_max=20, exec_candidates=8)
    await screener._refresh()

    assert len(screener.execution_candidates) <= 8


@pytest.mark.asyncio
async def test_execution_candidates_subset_of_feature():
    """Every execution candidate must be in the feature universe."""
    tickers = [_ticker(f"COIN{i:02d}USDT") for i in range(40)]
    screener = _make_screener(tickers, wide_max=40, feature_max=20, exec_candidates=10)
    await screener._refresh()

    feature_set = set(screener.feature_universe)
    for sym in screener.execution_candidates:
        assert sym in feature_set


# ---------------------------------------------------------------------------
# Position / order protection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_position_symbol_is_never_removed():
    """A symbol with an open position must not leave the feature universe."""
    tickers_initial = [_ticker("BTCUSDT"), _ticker("ETHUSDT")]

    rest = MagicMock()
    rest.get_tickers = AsyncMock(return_value={"result": {"list": tickers_initial}})

    open_positions = {"ETHUSDT"}

    screener = MarketScreener(
        rest_client=rest,
        wide_max_symbols=10,
        feature_max_symbols=10,
        execution_candidates=5,
        min_volume_usd=0.0,
        max_spread_bps=100.0,
        min_top_book_depth_usd=0.0,
        has_open_position=lambda s: s in open_positions,
    )

    # First refresh — both symbols enter
    await screener._refresh()
    assert "BTCUSDT" in screener.feature_universe
    assert "ETHUSDT" in screener.feature_universe

    # Second refresh — API returns only BTCUSDT; ETHUSDT has open position
    rest.get_tickers = AsyncMock(return_value={"result": {"list": [_ticker("BTCUSDT")]}})
    await screener._refresh()

    assert "ETHUSDT" in screener.feature_universe, "open-position symbol must be protected"


@pytest.mark.asyncio
async def test_pending_order_symbol_is_never_removed():
    """A symbol with a pending order must not leave the feature universe."""
    tickers_initial = [_ticker("XRPUSDT"), _ticker("ADAUSDT")]

    rest = MagicMock()
    rest.get_tickers = AsyncMock(return_value={"result": {"list": tickers_initial}})

    pending_orders = {"ADAUSDT"}

    screener = MarketScreener(
        rest_client=rest,
        wide_max_symbols=10,
        feature_max_symbols=10,
        execution_candidates=5,
        min_volume_usd=0.0,
        max_spread_bps=100.0,
        min_top_book_depth_usd=0.0,
        has_pending_order=lambda s: s in pending_orders,
    )

    await screener._refresh()
    rest.get_tickers = AsyncMock(return_value={"result": {"list": [_ticker("XRPUSDT")]}})
    await screener._refresh()

    assert "ADAUSDT" in screener.feature_universe


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_higher_turnover_ranks_higher():
    """Symbol with higher turnover should receive a better composite score."""
    tickers = [
        _ticker("LOWVOL_USDT", turnover=25_000_000),
        _ticker("HIGHVOL_USDT", turnover=500_000_000),
    ]
    screener = _make_screener(tickers)
    await screener._refresh()

    wide = screener.wide_universe
    symbols_ranked = [s.symbol for s in wide]
    high_idx = symbols_ranked.index("HIGHVOL_USDT")
    low_idx = symbols_ranked.index("LOWVOL_USDT")
    assert high_idx < low_idx


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_screener_metrics_populated():
    """After a refresh, ScreenerMetrics should be populated."""
    tickers = [_ticker(f"COIN{i:02d}USDT") for i in range(10)]
    screener = _make_screener(tickers, wide_max=10, feature_max=5, exec_candidates=3)
    await screener._refresh()

    m = screener.metrics
    assert isinstance(m, ScreenerMetrics)
    assert m.wide_universe_count > 0
    assert m.feature_universe_count > 0
    assert m.execution_candidate_count > 0
    assert m.last_screen_at is not None


# ---------------------------------------------------------------------------
# Backward compat: active_symbols == feature_universe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_active_symbols_equals_feature_universe():
    """active_symbols must equal feature_universe for FeaturePipeline compat."""
    tickers = [_ticker(f"COIN{i:02d}USDT") for i in range(10)]
    screener = _make_screener(tickers)
    await screener._refresh()

    assert screener.active_symbols == screener.feature_universe


# ---------------------------------------------------------------------------
# Backward compat: max_symbols alias
# ---------------------------------------------------------------------------


def test_max_symbols_alias_sets_feature_max():
    """Passing max_symbols=N should set feature_max_symbols=N."""
    rest = MagicMock()
    screener = MarketScreener(rest_client=rest, max_symbols=7)
    assert screener._feature_max == 7
