"""Market screener — dynamic symbol selection from Bybit linear futures.

Fetches top symbols by 24h USD volume, filters out illiquid and
problematic pairs, and maintains a ranked active list.

Design
------
- Runs as a background async loop every ``interval_s`` seconds.
- First run blocks until the initial list is ready (startup safety).
- Subsequent runs update ``active_symbols`` without blocking the caller.
- Falls back to ``_FALLBACK_SYMBOLS`` if the exchange call fails.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# Used when all API calls fail — cheap, liquid coins suitable for small balance
_FALLBACK_SYMBOLS = ["DOGEUSDT", "XRPUSDT", "ADAUSDT", "WLDUSDT", "NEARUSDT"]

# Quote coins we accept (USDT perpetual futures only)
_ACCEPTED_QUOTE = "USDT"

# Coins we explicitly skip (stablecoins, wrapped assets, etc.)
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

# Always exclude — reserved for venue-specific problem symbols if needed.
_EXCLUDED_SYMBOLS: set[str] = set()


class MarketScreener:
    """Ranks and selects tradeable symbols from Bybit linear futures.

    Args:
        rest_client:    BybitRestClient (used to fetch tickers).
        max_symbols:    Maximum symbols to keep in the active list.
        min_volume_usd: Minimum 24h turnover in USD to pass the filter.
        interval_s:     How often to refresh the list (seconds).
    """

    def __init__(
        self,
        rest_client: Any,
        max_symbols: int = 10,
        min_volume_usd: float = 20_000_000.0,  # 20M USD/day minimum
        max_price_usd: float = 0.0,  # 0 disables price cap
        interval_s: int = 900,  # 15 min
        on_symbols_added: Callable[[list[str]], Awaitable[None]] | None = None,
        on_symbols_removed: Callable[[list[str]], Awaitable[None]] | None = None,
        has_open_position: Callable[[str], bool] | None = None,
    ) -> None:
        self._rest = rest_client
        self._max_symbols = max_symbols
        self._min_volume = min_volume_usd
        self._max_price = max_price_usd
        self._interval = interval_s
        self._on_symbols_added = on_symbols_added
        self._on_symbols_removed = on_symbols_removed
        self._has_open_position = has_open_position
        self._stop_event = asyncio.Event()
        self._active_symbols: list[str] = list(_FALLBACK_SYMBOLS)
        self._initialized = asyncio.Event()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def active_symbols(self) -> list[str]:
        """Current ranked list of symbols to trade."""
        return list(self._active_symbols)

    async def run(self) -> None:
        """Screen the market in a loop until ``stop()`` is called.

        The first iteration completes synchronously (blocks) so that callers
        can await ``wait_ready()`` before starting dependent tasks.
        """
        log.info("screener.started", max_symbols=self._max_symbols)
        while not self._stop_event.is_set():
            await self._refresh()
            if not self._initialized.is_set():
                self._initialized.set()
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._stop_event.wait()),
                    timeout=self._interval,
                )
            except TimeoutError:
                pass

    async def wait_ready(self) -> None:
        """Wait until the first screen has completed."""
        await self._initialized.wait()

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    async def _refresh(self) -> None:
        try:
            symbols = await self._screen()
            if symbols:
                prev = set(self._active_symbols)
                new = set(symbols)

                # Protect symbols that have an open position from being removed
                if self._has_open_position is not None:
                    protected = {s for s in prev if s not in new and self._has_open_position(s)}
                    if protected:
                        log.info("screener.symbols_protected", protected=sorted(protected))
                        # Append protected symbols back; keep new symbols first (by volume rank)
                        symbols = symbols + [s for s in sorted(protected) if s not in symbols]
                        new = set(symbols)

                added = sorted(new - prev)
                removed = sorted(prev - new)

                if prev != new:
                    log.info(
                        "screener.symbols_updated",
                        total=len(symbols),
                        added=added,
                        removed=removed,
                    )

                self._active_symbols = symbols
                log.info("screener.universe_applied", count=len(symbols), symbols=symbols)

                # Notify app of added symbols (seed candles + WS subscribe)
                if added and self._on_symbols_added is not None:
                    for symbol in added:
                        log.info("screener.symbol_seeded", symbol=symbol)
                    await self._on_symbols_added(added)

                # Notify app of removed symbols
                if removed and self._on_symbols_removed is not None:
                    await self._on_symbols_removed(removed)
            else:
                log.warning("screener.no_symbols_returned", fallback=self._active_symbols)
        except Exception as exc:
            log.warning("screener.refresh_failed", error=str(exc))

    async def _screen(self) -> list[str]:
        """Fetch Bybit linear tickers and return top symbols by volume."""
        resp = await self._rest.get_tickers(category="linear")
        tickers: list[dict] = resp.get("result", {}).get("list", [])

        candidates: list[tuple[str, float]] = []
        for t in tickers:
            symbol: str = t.get("symbol", "")

            # Only USDT-quoted perpetuals
            if not symbol.endswith(_ACCEPTED_QUOTE):
                continue

            # Skip explicitly excluded symbols (too expensive for small balance)
            if symbol in _EXCLUDED_SYMBOLS:
                continue

            # Skip stablecoins
            base = symbol.removesuffix(_ACCEPTED_QUOTE)
            if base in _SKIP_BASE:
                continue

            # Price checks
            try:
                last_price = float(t.get("lastPrice", 0) or 0)
            except (ValueError, TypeError):
                continue
            if last_price <= 0.00001:  # dead coin
                continue
            if self._max_price and last_price > self._max_price:
                continue  # too expensive — min order would exceed budget

            # Volume filter (turnover24h = USD volume)
            try:
                vol = float(t.get("turnover24h", 0) or 0)
            except (ValueError, TypeError):
                continue
            if vol < self._min_volume:
                continue

            candidates.append((symbol, vol))

        # Sort descending by USD volume, take top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        ranked = [sym for sym, _ in candidates[: self._max_symbols]]

        log.debug(
            "screener.screened",
            total_passed=len(candidates),
            selected=len(ranked),
            top=ranked[:5],
        )
        return ranked
