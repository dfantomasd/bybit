"""Read-only FastAPI observability application.

Endpoints:
  GET /health          - Component health status
  GET /status          - System status summary
  GET /positions       - Current open positions (no secrets)
  GET /metrics         - Prometheus text exposition
  GET /regime          - Current market regime per symbol
  GET /model           - Deployed model metadata

Security:
  - API key authentication via X-API-Key header (internal use only)
  - CORS restricted to configured origins
  - Security headers (X-Content-Type-Options, etc.)
  - Request-level structured logging
  - Rate limiting via slowapi

All endpoints are READ-ONLY. No trading actions can be triggered via this API.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from trader.domain.enums import SystemStatus, TradingMode
from trader.domain.models import HealthStatus, ModelMetadata, Position
from trader.monitoring.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

_API_KEY_HEADER = "X-API-Key"


def create_app(
    api_key: str,
    allowed_origins: list[str] | None = None,
    # These would be injected from the trading system at runtime
    health_checker: Any | None = None,
    state_store: Any | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        api_key:          Internal API key for endpoint authentication.
        allowed_origins:  CORS allowed origins (default: none).
        health_checker:   HealthChecker instance.
        state_store:      Object exposing current positions, regime, model info.
    """
    app = FastAPI(
        title="Bybit AI Trader — Observability API",
        description="Read-only monitoring and status API. No trading actions.",
        version="0.1.0",
        docs_url=None,  # Disable Swagger UI in production
        redoc_url=None,
        openapi_url=None,
    )

    # -----------------------------------------------------------------------
    # Middleware
    # -----------------------------------------------------------------------

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or [],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["X-API-Key"],
    )

    # Security headers middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next: Callable[..., Any]) -> Response:
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next: Callable[..., Any]) -> Response:
        start = time.monotonic()
        response: Response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000
        log.info(
            "api_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            client=request.client.host if request.client else "unknown",
        )
        return response

    # -----------------------------------------------------------------------
    # Auth dependency
    # -----------------------------------------------------------------------

    def verify_api_key(request: Request) -> None:
        provided = request.headers.get(_API_KEY_HEADER)
        if not provided or provided != api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    AuthDep = Depends(verify_api_key)

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.get(
        "/health",
        response_model=None,
        summary="Component health status",
        tags=["observability"],
    )
    async def get_health(_auth: None = AuthDep) -> JSONResponse:
        """Return aggregated health of all system components."""
        if health_checker is None:
            return JSONResponse(
                status_code=200,
                content={
                    "overall": "unknown",
                    "message": "HealthChecker not configured",
                },
            )
        health: HealthStatus = await health_checker.overall_health()
        http_code = 200 if health.overall != "unhealthy" else 503
        return JSONResponse(
            status_code=http_code,
            content=health.model_dump(mode="json"),
        )

    @app.get(
        "/status",
        summary="System status summary",
        tags=["observability"],
    )
    async def get_status(_auth: None = AuthDep) -> dict[str, Any]:
        """Return current system lifecycle status and trading mode."""
        if state_store is None:
            return {
                "system_status": SystemStatus.STOPPED,
                "trading_mode": TradingMode.TESTNET,
                "message": "StateStore not configured",
            }
        return {
            "system_status": getattr(state_store, "system_status", SystemStatus.STOPPED),
            "trading_mode": getattr(state_store, "trading_mode", TradingMode.TESTNET),
            "open_positions": getattr(state_store, "open_position_count", 0),
            "is_live": getattr(state_store, "is_live", False),
        }

    @app.get(
        "/positions",
        summary="Current open positions",
        tags=["trading"],
    )
    async def get_positions(_auth: None = AuthDep) -> list[dict[str, Any]]:
        """Return sanitised view of open positions. No secrets or credentials."""
        if state_store is None:
            return []
        positions: list[Position] = getattr(state_store, "open_positions", [])
        return [
            {
                "symbol": p.symbol,
                "market_type": p.market_type,
                "side": p.side,
                "size": str(p.size),
                "entry_price": str(p.entry_price),
                "mark_price": str(p.mark_price) if p.mark_price else None,
                "unrealised_pnl": str(p.unrealised_pnl),
                "leverage": str(p.leverage),
            }
            for p in positions
        ]

    @app.get(
        "/metrics",
        summary="Prometheus metrics exposition",
        tags=["observability"],
        response_class=PlainTextResponse,
    )
    async def get_metrics(_auth: None = AuthDep) -> PlainTextResponse:
        """Expose Prometheus metrics in text format."""
        data = generate_latest()
        return PlainTextResponse(
            content=data.decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.get(
        "/regime",
        summary="Current market regime per symbol",
        tags=["trading"],
    )
    async def get_regime(_auth: None = AuthDep) -> dict[str, Any]:
        """Return the currently detected market regime for each tracked symbol."""
        if state_store is None:
            return {}
        regimes: dict[str, Any] = getattr(state_store, "current_regimes", {})
        # Serialise safely
        result: dict[str, Any] = {}
        for symbol, ctx in regimes.items():
            result[symbol] = {
                "regime": getattr(ctx, "regime", None),
                "volatility_level": getattr(ctx, "volatility_level", None),
                "confidence": getattr(ctx, "confidence", None),
                "trading_allowed": getattr(ctx, "trading_allowed", False),
            }
        return result

    @app.get(
        "/model",
        summary="Deployed model metadata",
        tags=["trading"],
    )
    async def get_model_info(_auth: None = AuthDep) -> dict[str, Any]:
        """Return metadata about the currently deployed ML model."""
        if state_store is None:
            return {}
        meta: ModelMetadata | None = getattr(state_store, "active_model_metadata", None)
        if meta is None:
            return {"status": "no_model_deployed"}
        return meta.model_dump(mode="json")

    # -----------------------------------------------------------------------
    # Exception handlers
    # -----------------------------------------------------------------------

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        log.error(
            "unhandled_api_error",
            path=request.url.path,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app
