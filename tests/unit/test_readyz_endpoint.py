"""Tests for P1.5: /readyz endpoint returning 503 when unhealthy."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from trader.api.fastapi_app import create_app
from trader.domain.enums import SystemStatus, TradingMode
from trader.domain.models import HealthStatus


def _health_status(overall: str) -> HealthStatus:
    return HealthStatus(
        overall=overall,
        postgres=overall != "unhealthy",
        redis=True,
        bybit_rest=True,
        bybit_ws=overall == "healthy",
        model_fresh=True,
        features_fresh=True,
        system_status=SystemStatus.RUNNING,
        trading_mode=TradingMode.SHADOW,
        messages=["DB down"] if overall == "unhealthy" else [],
    )


class TestReadyzEndpoint:
    def test_readyz_returns_200_when_healthy(self):
        """GET /readyz returns 200 when system is healthy."""
        mock_checker = MagicMock()
        mock_checker.overall_health = AsyncMock(return_value=_health_status("healthy"))
        app = create_app(api_key="key", health_checker=mock_checker)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/readyz")
        assert resp.status_code == 200

    def test_readyz_returns_200_when_degraded(self):
        """GET /readyz returns 200 when system is degraded (not unhealthy)."""
        mock_checker = MagicMock()
        mock_checker.overall_health = AsyncMock(return_value=_health_status("degraded"))
        app = create_app(api_key="key", health_checker=mock_checker)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/readyz")
        assert resp.status_code == 200

    def test_readyz_returns_503_when_unhealthy(self):
        """GET /readyz returns 503 when system is unhealthy."""
        mock_checker = MagicMock()
        mock_checker.overall_health = AsyncMock(return_value=_health_status("unhealthy"))
        app = create_app(api_key="key", health_checker=mock_checker)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/readyz")
        assert resp.status_code == 503
        assert "not_ready" in resp.json().get("status", "")

    def test_readyz_no_auth_required(self):
        """GET /readyz does NOT require authentication (for probe compatibility)."""
        mock_checker = MagicMock()
        mock_checker.overall_health = AsyncMock(return_value=_health_status("healthy"))
        app = create_app(api_key="secret", health_checker=mock_checker)
        client = TestClient(app, raise_server_exceptions=False)
        # No X-API-Key header
        resp = client.get("/readyz")
        # Must NOT be 401
        assert resp.status_code != 401

    def test_readyz_200_when_no_health_checker(self):
        """GET /readyz returns 200 when no health checker configured (graceful fallback)."""
        app = create_app(api_key="key", health_checker=None)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/readyz")
        assert resp.status_code == 200


class TestRuntimeSettingsEndpoint:
    def test_get_settings_requires_auth(self):
        app = create_app(api_key="key", runtime_settings=lambda: {"max_positions": 2})
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/api/settings")

        assert resp.status_code == 401

    def test_get_settings_returns_runtime_settings(self):
        app = create_app(api_key="key", runtime_settings=lambda: {"max_positions": 2})
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/api/settings", headers={"X-API-Key": "key"})

        assert resp.status_code == 200
        assert resp.json()["settings"]["max_positions"] == 2

    def test_post_settings_updates_runtime_setting(self):
        updates: list[tuple[str, object]] = []

        async def set_runtime_setting(key: str, value: object) -> str:
            updates.append((key, value))
            return f"{key} updated"

        app = create_app(
            api_key="key",
            runtime_settings=lambda: {"max_positions": 3},
            set_runtime_setting=set_runtime_setting,
        )
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/api/settings",
            headers={"X-API-Key": "key"},
            json={"key": "max_positions", "value": 3},
        )

        assert resp.status_code == 200
        assert resp.json()["message"] == "max_positions updated"
        assert updates == [("max_positions", 3)]


class TestStateStoreEndpoints:
    """Verify that /status, /positions, and /model use state_store when wired."""

    def _make_state_store(
        self,
        *,
        system_status: SystemStatus = SystemStatus.RUNNING,
        trading_mode: TradingMode = TradingMode.SHADOW,
        is_live: bool = False,
        open_position_count: int = 0,
    ) -> MagicMock:
        store = MagicMock()
        store.system_status = system_status
        store.trading_mode = trading_mode
        store.is_live = is_live
        store.open_position_count = open_position_count
        store.open_positions = []
        store.current_regimes = {}
        store.active_model_metadata = None
        return store

    def test_status_returns_running_when_state_store_wired(self) -> None:
        state_store = self._make_state_store(system_status=SystemStatus.RUNNING)
        app = create_app(api_key="key", state_store=state_store)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/status", headers={"X-API-Key": "key"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["system_status"] == SystemStatus.RUNNING.value
        assert data["trading_mode"] == TradingMode.SHADOW.value

    def test_model_returns_no_model_deployed_when_metadata_none(self) -> None:
        state_store = self._make_state_store()
        state_store.active_model_metadata = None
        app = create_app(api_key="key", state_store=state_store)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/model", headers={"X-API-Key": "key"})

        assert resp.status_code == 200
        assert resp.json() == {"status": "no_model_deployed"}

    def test_positions_returns_list_when_state_store_wired(self) -> None:
        from types import SimpleNamespace

        state_store = self._make_state_store()
        state_store.open_positions = [
            SimpleNamespace(
                symbol="BTCUSDT",
                market_type="LINEAR",
                side="Buy",
                size=0.001,
                entry_price=50000,
                mark_price=None,
                unrealised_pnl=0,
                leverage=1,
            )
        ]
        app = create_app(api_key="key", state_store=state_store)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/positions", headers={"X-API-Key": "key"})

        assert resp.status_code == 200
        positions = resp.json()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "BTCUSDT"
