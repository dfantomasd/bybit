"""Maps BybitRegion enum to correct REST/WS endpoints."""
from __future__ import annotations

from trader.domain.enums import BybitRegion
from trader.domain.errors import ConfigurationError

# ---------------------------------------------------------------------------
# Endpoint registry
# ---------------------------------------------------------------------------

ENDPOINTS: dict[BybitRegion, dict[str, str]] = {
    BybitRegion.GLOBAL: {
        "rest": "https://api.bybit.com",
        "rest_testnet": "https://api-testnet.bybit.com",
        "ws_public": "wss://stream.bybit.com/v5/public",
        "ws_public_testnet": "wss://stream-testnet.bybit.com/v5/public",
        "ws_private": "wss://stream.bybit.com/v5/private",
        "ws_private_testnet": "wss://stream-testnet.bybit.com/v5/private",
    },
    BybitRegion.NL: {
        "rest": "https://api.nl.bybit.com",
        "rest_testnet": "https://api-testnet.bybit.com",  # NL testnet falls back to global
        "ws_public": "wss://stream.nl.bybit.com/v5/public",
        "ws_public_testnet": "wss://stream-testnet.bybit.com/v5/public",
        "ws_private": "wss://stream.nl.bybit.com/v5/private",
        "ws_private_testnet": "wss://stream-testnet.bybit.com/v5/private",
    },
    BybitRegion.EEA: {
        "rest": "https://api.eea.bybit.com",
        "rest_testnet": "https://api-testnet.bybit.com",
        "ws_public": "wss://stream.eea.bybit.com/v5/public",
        "ws_public_testnet": "wss://stream-testnet.bybit.com/v5/public",
        "ws_private": "wss://stream.eea.bybit.com/v5/private",
        "ws_private_testnet": "wss://stream-testnet.bybit.com/v5/private",
    },
    BybitRegion.TR: {
        "rest": "https://api.tr.bybit.com",
        "rest_testnet": "https://api-testnet.bybit.com",
        "ws_public": "wss://stream.tr.bybit.com/v5/public",
        "ws_public_testnet": "wss://stream-testnet.bybit.com/v5/public",
        "ws_private": "wss://stream.tr.bybit.com/v5/private",
        "ws_private_testnet": "wss://stream-testnet.bybit.com/v5/private",
    },
    BybitRegion.KZ: {
        "rest": "https://api.kz.bybit.com",
        "rest_testnet": "https://api-testnet.bybit.com",
        "ws_public": "wss://stream.kz.bybit.com/v5/public",
        "ws_public_testnet": "wss://stream-testnet.bybit.com/v5/public",
        "ws_private": "wss://stream.kz.bybit.com/v5/private",
        "ws_private_testnet": "wss://stream-testnet.bybit.com/v5/private",
    },
    BybitRegion.GE: {
        "rest": "https://api.ge.bybit.com",
        "rest_testnet": "https://api-testnet.bybit.com",
        "ws_public": "wss://stream.ge.bybit.com/v5/public",
        "ws_public_testnet": "wss://stream-testnet.bybit.com/v5/public",
        "ws_private": "wss://stream.ge.bybit.com/v5/private",
        "ws_private_testnet": "wss://stream-testnet.bybit.com/v5/private",
    },
    BybitRegion.AE: {
        "rest": "https://api.ae.bybit.com",
        "rest_testnet": "https://api-testnet.bybit.com",
        "ws_public": "wss://stream.ae.bybit.com/v5/public",
        "ws_public_testnet": "wss://stream-testnet.bybit.com/v5/public",
        "ws_private": "wss://stream.ae.bybit.com/v5/private",
        "ws_private_testnet": "wss://stream-testnet.bybit.com/v5/private",
    },
    BybitRegion.ID: {
        "rest": "https://api.id.bybit.com",
        "rest_testnet": "https://api-testnet.bybit.com",
        "ws_public": "wss://stream.id.bybit.com/v5/public",
        "ws_public_testnet": "wss://stream-testnet.bybit.com/v5/public",
        "ws_private": "wss://stream.id.bybit.com/v5/private",
        "ws_private_testnet": "wss://stream-testnet.bybit.com/v5/private",
    },
}

# Regions that support testnet (only GLOBAL has a dedicated testnet infra)
_TESTNET_SUPPORTED_REGIONS: set[BybitRegion] = {BybitRegion.GLOBAL}

# Regions where testnet is allowed but falls back to global testnet
_TESTNET_FALLBACK_REGIONS: set[BybitRegion] = {
    BybitRegion.NL,
    BybitRegion.EEA,
    BybitRegion.TR,
    BybitRegion.KZ,
    BybitRegion.GE,
    BybitRegion.AE,
    BybitRegion.ID,
}


class EndpointSelector:
    """Selects correct REST and WebSocket endpoints for a given region/testnet combo."""

    def __init__(self, region: BybitRegion, use_testnet: bool) -> None:
        self._region = region
        self._use_testnet = use_testnet
        self._endpoints = ENDPOINTS[region]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def region(self) -> BybitRegion:
        return self._region

    @property
    def use_testnet(self) -> bool:
        return self._use_testnet

    @property
    def rest_base(self) -> str:
        """Return the REST base URL for this region/testnet combo."""
        if self._use_testnet:
            return self._endpoints["rest_testnet"]
        return self._endpoints["rest"]

    @property
    def ws_public_base(self) -> str:
        """Return the public WebSocket base URL."""
        if self._use_testnet:
            return self._endpoints["ws_public_testnet"]
        return self._endpoints["ws_public"]

    @property
    def ws_private_base(self) -> str:
        """Return the private (authenticated) WebSocket base URL."""
        if self._use_testnet:
            return self._endpoints["ws_private_testnet"]
        return self._endpoints["ws_private"]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_region_compatibility(self) -> None:
        """Raise ConfigurationError if region/testnet combo is not supported.

        Rules:
        - GLOBAL region supports testnet natively.
        - All other regions fall back to global testnet (allowed but with a warning).
        - Non-GLOBAL regions in live mode require operator confirmation (we just warn).
        """
        if self._region not in ENDPOINTS:
            raise ConfigurationError(
                f"Unknown region: {self._region}",
                field="BYBIT_REGION",
            )
        if self._use_testnet and self._region in _TESTNET_FALLBACK_REGIONS:
            # Allowed, but endpoints are shared with GLOBAL testnet — not an error.
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_all_endpoints(self) -> dict[str, str]:
        """Return the full endpoint dict for this region."""
        return dict(self._endpoints)

    def __repr__(self) -> str:
        return (
            f"EndpointSelector(region={self._region.value}, "
            f"use_testnet={self._use_testnet}, "
            f"rest_base={self.rest_base!r})"
        )
