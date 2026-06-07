"""Market screener — multi-tier dynamic symbol selection from Bybit linear futures.

Three-tier architecture
-----------------------
Wide universe   (SCREENER_WIDE_MAX_SYMBOLS, default 80):
    All USDT linear perpetuals passing volume, spread, depth, and data-quality
    filters. Ranked by a composite liquidity+execution-cost score.

Feature universe  (SCREENER_FEATURE_MAX_SYMBOLS, default 30):
    Top-N from the wide universe. These symbols receive kline.1 and tickers WS
    subscriptions and have features computed every ~5 seconds.

Execution candidates  (SCREENER_EXECUTION_CANDIDATES, default 15):
    Top-M from the feature universe. These are evaluated by the strategy each
    cycle. A symbol must be an execution candidate to receive a trade proposal.

The ``active_symbols`` property returns the *feature universe* for backward
compatibility with FeaturePipeline, which needs the superset.

Expanding the scanner does NOT increase risk. More symbols → more
opportunities → stricter selection — not more simultaneous positions.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# Quote coins we accept (USDT perpetual futures only)
_ACCEPTED_QUOTE = "USDT"

# Stablecoins and wrapped assets that should never appear in the universe
_SKIP_BASE = {
    "USDC",
    "BUSD",
    "DAI",
    "TUSD",
    "USDP",
    "FRAX",
    "USDD",
    "GUSD",
    "USDJ",
    "USDN",
}

# Fallback used when all API calls fail
_FALLBACK_SYMBOLS = ["DOGEUSDT", "XRPUSDT", "ADAUSDT", "WLDUSDT", "NEARUSDT"]

# Score weights (must sum to 1.0)
_W_TURNOVER = 0.35
_W_SPREAD = 0.30
_W_DEPTH = 0.15
_W_VOLATILITY = 0.10
_W_DATA_QUALITY = 0.10


@dataclass(frozen=True)
class ScoredSymbol:
    """Per-symbol scoring result from one screener refresh."""

    symbol: str
    turnover_usd: float
    spread_bps: float
    top_book_depth_usd: float
    volatility_pct: float
    data_quality_score: float
    combined_score: float
    rejection_reason: str | None = None


@dataclass
class ScreenerMetrics:
    """Live metrics exposed to the Telegram dashboard and diagnostics."""

    wide_universe_count: int = 0
    feature_universe_count: int = 0
    execution_candidate_count: int = 0
    blocked_symbol_count: int = 0
    last_screen_at: datetime | None = None
    last_universe_swap_at: datetime | None = None
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    top_candidates: list[tuple[str, float]] = field(default_factory=list)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value or 0)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError):
        return default


def _score_ticker(
    t: dict[str, Any],
    *,
    max_spread_bps: float,
    min_depth_usd: float,
    all_turnovers: list[float],
) -> ScoredSymbol:
    """Compute composite score for a single ticker dict."""
    symbol: str = t.get("symbol", "")

    # --- Required price fields ---
    bid = _safe_float(t.get("bid1Price"))
    ask = _safe_float(t.get("ask1Price"))
    last = _safe_float(t.get("lastPrice"))

    required_present = int(bid > 0) + int(ask > 0) + int(last > 0)
    data_quality = required_present / 3.0

    # --- Spread ---
    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
    spread_bps = ((ask - bid) / mid * 10_000) if mid > 0 else 999.0

    # --- Top-book depth (proxy: bid1Size * bid1Price + ask1Size * ask1Price) ---
    bid_sz = _safe_float(t.get("bid1Size"))
    ask_sz = _safe_float(t.get("ask1Size"))
    top_depth_usd = bid_sz * bid + ask_sz * ask

    # --- Turnover ---
    turnover = _safe_float(t.get("turnover24h"))

    # --- Volatility (24h absolute price change %) ---
    pct_raw = _safe_float(t.get("price24hPcnt"))
    volatility_pct = abs(pct_raw * 100.0)  # convert fraction → %

    # --- Scores (all normalised to [0, 1]) ---

    # Turnover: log-normalised relative to max in universe
    max_turnover = max(all_turnovers) if all_turnovers else 1.0
    log_max = math.log1p(max_turnover)
    turnover_score = math.log1p(turnover) / log_max if log_max > 0 else 0.0

    # Spread: lower is better; 0 bps → 1.0, max_spread_bps → 0.0
    spread_score = max(0.0, 1.0 - spread_bps / max_spread_bps)

    # Depth: sigmoid-like normalisation around min_depth_usd
    depth_score = min(1.0, top_depth_usd / (min_depth_usd * 5)) if min_depth_usd > 0 else 0.5

    # Volatility: moderate volatility is best for scalping (peak at ~3%)
    vol_score = (
        min(1.0, volatility_pct / 3.0) if volatility_pct <= 3.0 else max(0.0, 1.0 - (volatility_pct - 3.0) / 10.0)
    )

    combined = (
        _W_TURNOVER * turnover_score
        + _W_SPREAD * spread_score
        + _W_DEPTH * depth_score
        + _W_VOLATILITY * vol_score
        + _W_DATA_QUALITY * data_quality
    )

    return ScoredSymbol(
        symbol=symbol,
        turnover_usd=turnover,
        spread_bps=spread_bps,
        top_book_depth_usd=top_depth_usd,
        volatility_pct=volatility_pct,
        data_quality_score=data_quality,
        combined_score=combined,
    )


class MarketScreener:
    """Multi-tier market screener for Bybit linear USDT perpetuals.

    Args:
        rest_client:            BybitRestClient (used to fetch tickers).
        wide_max_symbols:       Max symbols in the wide universe (volume filter pass).
        feature_max_symbols:    Max symbols for feature computation. Published
                                 as ``active_symbols`` for FeaturePipeline.
        execution_candidates:   Max candidates passed to the strategy each cycle.
        min_volume_usd:         Minimum 24h turnover (USD) to enter wide universe.
        max_spread_bps:         Maximum bid-ask spread (basis points) allowed.
        min_top_book_depth_usd: Minimum top-of-book depth required.
        min_price_usd:          Optional minimum last price; 0 disables.
        max_price_usd:          Optional maximum last price; 0 disables.
        interval_s:             Screener refresh period (seconds).
        denylist:               Symbols explicitly excluded (special zones, etc.).
        on_symbols_added:       Callback when symbols enter the feature universe.
        on_symbols_removed:     Callback when symbols leave the feature universe.
        has_open_position:      Guard: never remove symbols with open positions.
        has_pending_order:      Guard: never remove symbols with pending orders.

        # Deprecated / backward-compat
        max_symbols:            Maps to feature_max_symbols if provided.
    """

    def __init__(
        self,
        rest_client: Any,
        *,
        wide_max_symbols: int = 80,
        feature_max_symbols: int = 30,
        execution_candidates: int = 15,
        min_volume_usd: float = 20_000_000.0,
        max_spread_bps: float = 8.0,
        min_top_book_depth_usd: float = 5_000.0,
        min_price_usd: float = 0.0,
        max_price_usd: float = 0.0,
        interval_s: int = 900,
        denylist: list[str] | None = None,
        on_symbols_added: Callable[[list[str]], Awaitable[None]] | None = None,
        on_symbols_removed: Callable[[list[str]], Awaitable[None]] | None = None,
        has_open_position: Callable[[str], bool] | None = None,
        has_pending_order: Callable[[str], bool] | None = None,
        # Backward-compat aliases
        max_symbols: int | None = None,
    ) -> None:
        import asyncio

        if max_symbols is not None:
            feature_max_symbols = max_symbols

        self._rest = rest_client
        self._wide_max = wide_max_symbols
        self._feature_max = feature_max_symbols
        self._exec_candidates = execution_candidates
        self._min_volume = min_volume_usd
        self._max_spread_bps = max_spread_bps
        self._min_depth_usd = min_top_book_depth_usd
        self._min_price_usd = min_price_usd
        self._max_price_usd = max_price_usd
        self._interval = interval_s
        self._denylist: set[str] = set(denylist or [])
        self._manual_symbols: set[str] = set()
        self._on_symbols_added = on_symbols_added
        self._on_symbols_removed = on_symbols_removed
        self._has_open_position = has_open_position
        self._has_pending_order = has_pending_order

        self._stop_event = asyncio.Event()
        self._initialized = asyncio.Event()

        # Published state
        self._wide_universe: list[ScoredSymbol] = []
        self._feature_universe: list[str] = list(_FALLBACK_SYMBOLS)
        self._execution_candidates_list: list[str] = list(_FALLBACK_SYMBOLS[:5])
        self._metrics = ScreenerMetrics()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def active_symbols(self) -> list[str]:
        """Feature universe — backward-compatible name used by FeaturePipeline."""
        return list(self._feature_universe)

    @property
    def wide_universe(self) -> list[ScoredSymbol]:
        """Full ranked wide universe from the last refresh."""
        return list(self._wide_universe)

    @property
    def feature_universe(self) -> list[str]:
        """Symbols receiving WS+feature computation."""
        return list(self._feature_universe)

    @property
    def execution_candidates(self) -> list[str]:
        """Top-ranked symbols for strategy evaluation this cycle."""
        return list(self._execution_candidates_list)

    @property
    def metrics(self) -> ScreenerMetrics:
        return self._metrics

    @property
    def manual_symbols(self) -> list[str]:
        """Operator-selected symbols that should stay in the trading universe when eligible."""
        return sorted(self._manual_symbols)

    def set_manual_symbols(self, symbols: list[str]) -> None:
        self._manual_symbols = {symbol.upper() for symbol in symbols if symbol}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Screen the market in a loop until ``stop()`` is called."""
        log.info(
            "screener.started",
            wide_max=self._wide_max,
            feature_max=self._feature_max,
            exec_candidates=self._exec_candidates,
        )
        while not self._stop_event.is_set():
            await self._refresh()
            if not self._initialized.is_set():
                self._initialized.set()
            try:
                import asyncio

                await asyncio.wait_for(
                    asyncio.shield(self._stop_event.wait()),
                    timeout=self._interval,
                )
            except TimeoutError:
                pass

    async def wait_ready(self) -> None:
        await self._initialized.wait()

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    async def _refresh(self) -> None:
        try:
            wide, feature, exec_cands = await self._screen()
            if not feature:
                log.warning("screener.no_symbols_returned", fallback=self._feature_universe)
                return

            prev_feature = set(self._feature_universe)
            new_feature = set(feature)

            # --- Protect symbols with open positions or pending orders ---
            protected: set[str] = set()
            for s in prev_feature:
                if s in new_feature:
                    continue
                if self._has_open_position is not None and self._has_open_position(s):
                    protected.add(s)
                elif self._has_pending_order is not None and self._has_pending_order(s):
                    protected.add(s)

            if protected:
                log.info("screener.symbols_protected", protected=sorted(protected))
                feature = feature + [s for s in sorted(protected) if s not in feature]
                new_feature = set(feature)

            added = sorted(new_feature - prev_feature)
            removed = sorted(prev_feature - new_feature)

            # Publish new universe
            self._wide_universe = wide
            self._feature_universe = feature
            self._execution_candidates_list = exec_cands

            now = datetime.now(tz=UTC)
            self._metrics.wide_universe_count = len(wide)
            self._metrics.feature_universe_count = len(feature)
            self._metrics.execution_candidate_count = len(exec_cands)
            self._metrics.blocked_symbol_count = len(protected)
            self._metrics.last_screen_at = now
            self._metrics.top_candidates = [(s.symbol, round(s.combined_score, 4)) for s in wide[:10]]

            if prev_feature != new_feature:
                self._metrics.last_universe_swap_at = now
                log.info(
                    "screener.universe_updated",
                    wide=len(wide),
                    feature=len(feature),
                    exec_cands=len(exec_cands),
                    added=added,
                    removed=removed,
                    protected=sorted(protected),
                )

            log.info(
                "screener.universe_applied",
                wide=len(wide),
                feature=len(feature),
                exec_cands=len(exec_cands),
                top5=feature[:5],
            )

            if added and self._on_symbols_added is not None:
                await self._on_symbols_added(added)
            if removed and self._on_symbols_removed is not None:
                await self._on_symbols_removed(removed)

        except Exception as exc:
            log.warning("screener.refresh_failed", error=str(exc))

    async def _screen(
        self,
    ) -> tuple[list[ScoredSymbol], list[str], list[str]]:
        """Fetch tickers, filter, score, and return three-tier result."""
        resp = await self._rest.get_tickers(category="linear")
        tickers: list[dict[str, Any]] = resp.get("result", {}).get("list", [])

        rejection_counts: dict[str, int] = {}

        def _reject(reason: str) -> None:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        # Pre-filter: basic structural checks
        pre_filtered: list[dict[str, Any]] = []
        for t in tickers:
            symbol: str = t.get("symbol", "")

            if not symbol.endswith(_ACCEPTED_QUOTE):
                continue
            if symbol in self._denylist:
                _reject("denylist")
                continue

            base = symbol.removesuffix(_ACCEPTED_QUOTE)
            if base in _SKIP_BASE:
                _reject("stablecoin")
                continue

            # Skip pre-market instruments (curPreListingPhase non-empty)
            if t.get("curPreListingPhase"):
                _reject("premarket")
                continue

            try:
                last_price = float(t.get("lastPrice", 0) or 0)
            except (ValueError, TypeError):
                _reject("bad_price")
                continue
            if last_price <= 0.00001:
                _reject("dead_coin")
                continue
            if self._min_price_usd > 0 and last_price < self._min_price_usd:
                _reject("below_min_price")
                continue
            if self._max_price_usd > 0 and last_price > self._max_price_usd:
                _reject("above_max_price")
                continue

            try:
                vol = float(t.get("turnover24h", 0) or 0)
            except (ValueError, TypeError):
                _reject("bad_volume")
                continue
            if vol < self._min_volume:
                _reject("low_volume")
                continue

            pre_filtered.append(t)

        # Collect all turnovers for relative normalisation
        all_turnovers = [_safe_float(t.get("turnover24h")) for t in pre_filtered]

        # Score and filter by spread + depth
        scored: list[ScoredSymbol] = []
        for t in pre_filtered:
            sym_score = _score_ticker(
                t,
                max_spread_bps=self._max_spread_bps,
                min_depth_usd=self._min_depth_usd,
                all_turnovers=all_turnovers,
            )

            if sym_score.spread_bps > self._max_spread_bps:
                _reject("high_spread")
                continue

            if self._min_depth_usd > 0 and sym_score.top_book_depth_usd < self._min_depth_usd:
                _reject("low_depth")
                continue

            scored.append(sym_score)

        # Sort by combined score descending
        scored.sort(key=lambda s: s.combined_score, reverse=True)

        # Tier 1: wide universe
        wide = scored[: self._wide_max]

        ranked_symbols = [s.symbol for s in wide]

        # Tier 2: feature universe. Manual symbols are only honored after they
        # pass the same liquidity/spread/price filters and appear in the wide universe.
        manual_ranked = [symbol for symbol in ranked_symbols if symbol in self._manual_symbols]
        feature_symbols = list(dict.fromkeys([*manual_ranked, *ranked_symbols[: self._feature_max]]))[
            : self._feature_max
        ]

        # Tier 3: execution candidates (top-M of feature)
        exec_symbols = list(dict.fromkeys([*manual_ranked, *feature_symbols]))[: self._exec_candidates]

        self._metrics.rejection_reasons = rejection_counts

        log.debug(
            "screener.screened",
            total_tickers=len(tickers),
            pre_filtered=len(pre_filtered),
            scored=len(scored),
            wide=len(wide),
            feature=len(feature_symbols),
            exec_cands=len(exec_symbols),
            rejection_summary=rejection_counts,
        )

        return wide, feature_symbols, exec_symbols
