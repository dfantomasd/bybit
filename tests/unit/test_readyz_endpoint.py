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
