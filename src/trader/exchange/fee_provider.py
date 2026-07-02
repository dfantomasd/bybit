"""FeeRateProvider — cached fee rate fetcher for Bybit linear perpetuals."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, NamedTuple

import structlog

log = structlog.get_logger(__name__)

_CACHE_TTL_SECONDS = 3600


class FeeRates(NamedTuple):
    maker_fee_rate: float
    taker_fee_rate: float
    source: str
    fetched_at: datetime


class FeeRateProvider:
    """Fetch and cache per-symbol fee rates from /v5/account/fee-rate.

    Falls back to config defaults in SHADOW mode when the API is unavailable.
    In LIVE mode an API failure is fail-closed: returns None, caller must block entry.
    """

    def __init__(
        self,
        rest: Any,
        category: str = "linear",
        default_maker: float = 0.0002,
        default_taker: float = 0.00055,
        shadow_mode: bool = True,
    ) -> None:
        self._rest = rest
        self._category = category
        self._default_maker = default_maker
        self._default_taker = default_taker
        self._shadow_mode = shadow_mode
        self._cache: dict[str, FeeRates] = {}

    @property
    def shadow_mode(self) -> bool:
        return self._shadow_mode

    @shadow_mode.setter
    def shadow_mode(self, value: bool) -> None:
        self._shadow_mode = bool(value)

    def _is_stale(self, rates: FeeRates) -> bool:
        return (datetime.now(tz=UTC) - rates.fetched_at).total_seconds() > _CACHE_TTL_SECONDS

    async def get(self, symbol: str) -> FeeRates | None:
        """Return fee rates for symbol. Returns None in LIVE if API fails."""
        cached = self._cache.get(symbol)
        if cached and not self._is_stale(cached):
            return cached

        try:
            resp = await self._rest.get_fee_rate(category=self._category, symbol=symbol)
            items = (resp.get("result") or {}).get("list", [])
            if items:
                item = items[0]
                rates = FeeRates(
                    maker_fee_rate=float(item.get("makerFeeRate", self._default_maker)),
                    taker_fee_rate=float(item.get("takerFeeRate", self._default_taker)),
                    source="api",
                    fetched_at=datetime.now(tz=UTC),
                )
                self._cache[symbol] = rates
                return rates
        except Exception as exc:
            log.warning("fee_provider.api_failed", symbol=symbol, error=str(exc))

        if self._shadow_mode:
            fallback = FeeRates(
                maker_fee_rate=self._default_maker,
                taker_fee_rate=self._default_taker,
                source="fallback",
                fetched_at=datetime.now(tz=UTC),
            )
            self._cache[symbol] = fallback
            log.debug("fee_provider.using_fallback", symbol=symbol)
            return fallback

        # LIVE mode — fail closed
        log.error("fee_provider.live_fail_closed", symbol=symbol)
        return None

    def invalidate(self, symbol: str | None = None) -> None:
        if symbol:
            self._cache.pop(symbol, None)
        else:
            self._cache.clear()
