"""Tests for EndpointSelector."""
from trader.domain.enums import BybitRegion
from trader.exchange.endpoint_selector import ENDPOINTS, EndpointSelector


class TestEndpointRegistry:
    """Verify the ENDPOINTS registry is complete and consistent."""

    def test_all_regions_present(self) -> None:
        for region in BybitRegion:
            assert region in ENDPOINTS, f"Region {region} missing from ENDPOINTS"

    def test_each_region_has_required_keys(self) -> None:
        required = {"rest", "rest_testnet", "ws_public", "ws_public_testnet", "ws_private", "ws_private_testnet"}
        for region, endpoints in ENDPOINTS.items():
            assert required <= set(endpoints.keys()), f"Region {region} is missing keys"

    def test_global_rest_url(self) -> None:
        assert ENDPOINTS[BybitRegion.GLOBAL]["rest"] == "https://api.bybit.com"

    def test_global_testnet_rest_url(self) -> None:
        assert ENDPOINTS[BybitRegion.GLOBAL]["rest_testnet"] == "https://api-testnet.bybit.com"

    def test_global_ws_public(self) -> None:
        assert ENDPOINTS[BybitRegion.GLOBAL]["ws_public"] == "wss://stream.bybit.com/v5/public"

    def test_global_ws_private(self) -> None:
        assert ENDPOINTS[BybitRegion.GLOBAL]["ws_private"] == "wss://stream.bybit.com/v5/private"

    def test_nl_rest_url(self) -> None:
        assert ENDPOINTS[BybitRegion.NL]["rest"] == "https://api.bybit.nl"

    def test_eea_rest_url(self) -> None:
        assert ENDPOINTS[BybitRegion.EEA]["rest"] == "https://api.bybit.eu"

    def test_tr_rest_url(self) -> None:
        assert ENDPOINTS[BybitRegion.TR]["rest"] == "https://api.bybit.tr"

    def test_kz_rest_url(self) -> None:
        assert ENDPOINTS[BybitRegion.KZ]["rest"] == "https://api.bybit.kz"

    def test_ge_rest_url(self) -> None:
        assert ENDPOINTS[BybitRegion.GE]["rest"] == "https://api.bybitgeorgia.ge"

    def test_ae_rest_url(self) -> None:
        assert ENDPOINTS[BybitRegion.AE]["rest"] == "https://api.bybit.ae"

    def test_id_rest_url(self) -> None:
        assert ENDPOINTS[BybitRegion.ID]["rest"] == "https://api.bybit.id"

    def test_all_rest_urls_are_https(self) -> None:
        for region, endpoints in ENDPOINTS.items():
            assert endpoints["rest"].startswith("https://"), f"Region {region} rest URL not HTTPS"

    def test_all_ws_urls_are_wss(self) -> None:
        for region, endpoints in ENDPOINTS.items():
            for key in ("ws_public", "ws_private", "ws_public_testnet", "ws_private_testnet"):
                assert endpoints[key].startswith("wss://"), f"{region}.{key} not WSS"

class TestEndpointSelectorProperties:
    """Test EndpointSelector properties with live vs testnet."""

    def test_global_live_rest_base(self) -> None:
        sel = EndpointSelector(BybitRegion.GLOBAL, use_testnet=False)
        assert sel.rest_base == "https://api.bybit.com"

    def test_global_testnet_rest_base(self) -> None:
        sel = EndpointSelector(BybitRegion.GLOBAL, use_testnet=True)
        assert sel.rest_base == "https://api-testnet.bybit.com"

    def test_global_live_ws_public(self) -> None:
        sel = EndpointSelector(BybitRegion.GLOBAL, use_testnet=False)
        assert sel.ws_public_base == "wss://stream.bybit.com/v5/public"

    def test_global_testnet_ws_public(self) -> None:
        sel = EndpointSelector(BybitRegion.GLOBAL, use_testnet=True)
        assert sel.ws_public_base == "wss://stream-testnet.bybit.com/v5/public"

    def test_global_live_ws_private(self) -> None:
        sel = EndpointSelector(BybitRegion.GLOBAL, use_testnet=False)
        assert sel.ws_private_base == "wss://stream.bybit.com/v5/private"

    def test_global_testnet_ws_private(self) -> None:
        sel = EndpointSelector(BybitRegion.GLOBAL, use_testnet=True)
        assert sel.ws_private_base == "wss://stream-testnet.bybit.com/v5/private"

    def test_nl_live_rest_base(self) -> None:
        sel = EndpointSelector(BybitRegion.NL, use_testnet=False)
        assert sel.rest_base == "https://api.bybit.nl"

    def test_nl_testnet_falls_back_to_global(self) -> None:
        sel = EndpointSelector(BybitRegion.NL, use_testnet=True)
        # NL testnet uses global testnet infra
        assert "testnet" in sel.rest_base

    def test_region_property(self) -> None:
        sel = EndpointSelector(BybitRegion.AE, use_testnet=False)
        assert sel.region == BybitRegion.AE

    def test_use_testnet_property(self) -> None:
        sel = EndpointSelector(BybitRegion.GLOBAL, use_testnet=True)
        assert sel.use_testnet is True

    def test_get_all_endpoints_returns_dict(self) -> None:
        sel = EndpointSelector(BybitRegion.GLOBAL, use_testnet=False)
        endpoints = sel.get_all_endpoints()
        assert isinstance(endpoints, dict)
        assert "rest" in endpoints
        assert "ws_public" in endpoints


class TestEndpointSelectorValidation:
    """Test validate_region_compatibility."""

    def test_global_testnet_valid(self) -> None:
        sel = EndpointSelector(BybitRegion.GLOBAL, use_testnet=True)
        # Should not raise
        sel.validate_region_compatibility()

    def test_global_live_valid(self) -> None:
        sel = EndpointSelector(BybitRegion.GLOBAL, use_testnet=False)
        sel.validate_region_compatibility()

    def test_nl_testnet_allowed_with_fallback(self) -> None:
        sel = EndpointSelector(BybitRegion.NL, use_testnet=True)
        # Should not raise (falls back to global testnet)
        sel.validate_region_compatibility()

    def test_eea_live_valid(self) -> None:
        sel = EndpointSelector(BybitRegion.EEA, use_testnet=False)
        sel.validate_region_compatibility()

    def test_repr_contains_region(self) -> None:
        sel = EndpointSelector(BybitRegion.GLOBAL, use_testnet=False)
        assert "GLOBAL" in repr(sel)
        assert "api.bybit.com" in repr(sel)
