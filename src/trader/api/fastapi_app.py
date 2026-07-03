"""FastAPI observability application.

Endpoints:
  GET  /livez           - Unauthenticated process liveness
  GET  /health          - Component health status
  GET  /status          - System status summary
  GET  /positions       - Current open positions (no secrets)
  GET  /metrics         - Prometheus text exposition
  GET  /regime          - Current market regime per symbol
  GET  /model           - Deployed model metadata
  POST /api/settings    - Mutates live runtime risk/execution settings
                           (position limits, entry rate, model-gate
                           threshold) — NOT read-only; see operator_controls.

Security:
  - API key authentication via X-API-Key header (internal use only)
  - CORS restricted to configured origins
  - Security headers (X-Content-Type-Options, etc.)
  - Request-level structured logging

Most endpoints are read-only observability data. POST /api/settings is the
one exception and changes trading behavior in real time — the X-API-Key
header is its only authorization/audit control.
"""

from __future__ import annotations

import hmac
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
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
    trade_journal: Any | None = None,
    runtime_settings: Callable[[], dict[str, Any]] | None = None,
    set_runtime_setting: Callable[[str, Any], Awaitable[str]] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        api_key:          Internal API key for endpoint authentication.
        allowed_origins:  CORS allowed origins (default: none).
        health_checker:   HealthChecker instance.
        state_store:      Object exposing current positions, regime, model info.
        trade_journal:    TradeJournal-like object for read-only dashboards.
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
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
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
        if not provided or not hmac.compare_digest(provided, api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    auth_dep = Depends(verify_api_key)

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.get(
        "/livez",
        summary="Process liveness",
        tags=["observability"],
    )
    async def get_livez() -> dict[str, str]:
        """Return minimal unauthenticated liveness for container probes.

        Intentionally returns only status — deploy/commit metadata would be
        visible to any unauthenticated caller and is available on /status.
        """
        return {"status": "ok"}

    @app.get(
        "/readyz",
        summary="Process readiness",
        tags=["observability"],
    )
    async def get_readyz() -> JSONResponse:
        """Return 200 when the system is healthy/degraded, 503 when unhealthy.

        Use this for Kubernetes/Render readiness probes — the container should
        not receive traffic when unhealthy (e.g. DB is down). Note: a stale/
        disconnected WS feed alone only degrades the overall status, it does
        not fail this probe — see HealthChecker._compute_overall_health.
        """
        if health_checker is None:
            return JSONResponse(status_code=200, content={"status": "ok", "note": "no health checker"})
        health: HealthStatus = await health_checker.overall_health()
        if health.overall == "unhealthy":
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "messages": health.messages},
            )
        return JSONResponse(status_code=200, content={"status": "ready", "overall": health.overall})

    @app.get(
        "/health",
        response_model=None,
        summary="Component health status",
        tags=["observability"],
    )
    async def get_health(_auth: None = auth_dep) -> JSONResponse:
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
    async def get_status(_auth: None = auth_dep) -> dict[str, Any]:
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
    async def get_positions(_auth: None = auth_dep) -> list[dict[str, Any]]:
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
    async def get_metrics(_auth: None = auth_dep) -> PlainTextResponse:
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
    async def get_regime(_auth: None = auth_dep) -> dict[str, Any]:
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
        "/dashboard",
        summary="Strategy diagnostics dashboard",
        tags=["observability"],
        response_class=HTMLResponse,
    )
    async def get_dashboard(_auth: None = auth_dep) -> HTMLResponse:
        """Return a small protected HTML dashboard backed by TradeJournal aggregates."""
        journal = trade_journal
        if state_store is not None:
            journal = getattr(state_store, "trade_journal", None) or journal
        if journal is None or not getattr(journal, "is_enabled", False):
            return HTMLResponse(
                status_code=503,
                content="<html><body><h1>Dashboard unavailable</h1><p>Trade journal is not connected.</p></body></html>",
            )
        data = await journal.get_dashboard_data()
        if data.get("error"):
            return HTMLResponse(
                status_code=500,
                content=f"<html><body><h1>Dashboard error</h1><pre>{data['error']}</pre></body></html>",
            )
        payload = json.dumps(data, default=str)
        html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Bybit AI Trader Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #101418; color: #e7edf3; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }}
    section {{ background: #171d24; border: 1px solid #2c3642; border-radius: 8px; padding: 16px; }}
    canvas {{ width: 100%; max-height: 360px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border: 1px solid #2c3642; padding: 4px; text-align: center; }}
    th {{ background: #202833; }}
    .meta {{ color: #9fb0c0; margin-bottom: 18px; }}
  </style>
</head>
<body>
<main>
  <h1>Bybit AI Trader Dashboard</h1>
  <div class="meta" id="meta"></div>
  <div class="grid">
    <section><h2>Equity curve, net bps</h2><canvas id="equity"></canvas></section>
    <section><h2>Baseline return histogram</h2><canvas id="histogram"></canvas></section>
  </div>
  <section style="margin-top:18px"><h2>Baseline PnL heatmap, UTC</h2><div id="heatmap"></div></section>
</main>
<script type="application/json" id="__data__">{payload}</script>
<script>
const DATA = JSON.parse(document.getElementById('__data__').textContent);
document.getElementById('meta').textContent = `horizon=${{DATA.horizon_minutes}}m, model=${{DATA.model_version || 'none'}}`;
const base = DATA.equity_baseline || [];
const gate = DATA.equity_gate_pass || [];
const labels = base.map((p, i) => p.x ? p.x.slice(0, 16).replace('T', ' ') : String(i + 1));
new Chart(document.getElementById('equity'), {{
  type: 'line',
  data: {{ labels, datasets: [
    {{ label: 'baseline', data: base.map(p => p.y), borderColor: '#5cc8ff', pointRadius: 0, tension: 0.1 }},
    {{ label: 'gate pass', data: gate.map(p => p.y), borderColor: '#ffd166', pointRadius: 0, tension: 0.1 }}
  ] }},
  options: {{ responsive: true, interaction: {{ mode: 'index', intersect: false }}, scales: {{ x: {{ ticks: {{ maxTicksLimit: 8 }} }} }} }}
}});
new Chart(document.getElementById('histogram'), {{
  type: 'bar',
  data: {{ labels: (DATA.histogram || []).map(b => b.bucket), datasets: [{{ label: 'count', data: (DATA.histogram || []).map(b => b.count), backgroundColor: '#7bd88f' }}] }},
  options: {{ responsive: true, scales: {{ x: {{ ticks: {{ maxTicksLimit: 12 }} }} }} }}
}});
const weekdays = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
const heat = new Map((DATA.heatmap || []).map(r => [`${{r.weekday}}:${{r.hour}}`, r]));
const values = (DATA.heatmap || []).map(r => r.total_bps);
const maxAbs = Math.max(1, ...values.map(v => Math.abs(v)));
let table = '<table><thead><tr><th>day/hour</th>';
for (let h = 0; h < 24; h++) table += `<th>${{String(h).padStart(2,'0')}}</th>`;
table += '</tr></thead><tbody>';
for (let d = 1; d <= 7; d++) {{
  table += `<tr><th>${{weekdays[d-1]}}</th>`;
  for (let h = 0; h < 24; h++) {{
    const r = heat.get(`${{d}}:${{h}}`);
    const v = r ? r.total_bps : 0;
    const alpha = Math.min(0.95, Math.abs(v) / maxAbs);
    const color = v >= 0 ? `rgba(70, 170, 100, ${{alpha}})` : `rgba(210, 80, 80, ${{alpha}})`;
    table += `<td title="n=${{r ? r.count : 0}}" style="background:${{color}}">${{v.toFixed(0)}}</td>`;
  }}
  table += '</tr>';
}}
table += '</tbody></table>';
document.getElementById('heatmap').innerHTML = table;
</script>
</body>
</html>
"""
        return HTMLResponse(content=html)

    @app.get(
        "/api/settings",
        summary="Runtime settings",
        tags=["observability"],
    )
    async def get_settings(_auth: None = auth_dep) -> dict[str, Any]:
        """Return safe runtime settings for operator UI controls."""
        if runtime_settings is None:
            raise HTTPException(status_code=503, detail="Runtime settings are unavailable")
        return {"settings": runtime_settings()}

    @app.post(
        "/api/settings",
        summary="Update runtime setting",
        tags=["observability"],
    )
    async def post_settings(request: Request, _auth: None = auth_dep) -> dict[str, Any]:
        """Update one whitelisted runtime setting through the application controller."""
        if runtime_settings is None or set_runtime_setting is None:
            raise HTTPException(status_code=503, detail="Runtime settings are unavailable")
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON object body is required")
        key = str(payload.get("key") or "").strip()
        if not key:
            raise HTTPException(status_code=400, detail="Missing setting key")
        if "value" not in payload:
            raise HTTPException(status_code=400, detail="Missing setting value")
        try:
            message = await set_runtime_setting(key, payload["value"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.warning("api_settings_update_failed", key=key, error=str(exc))
            raise HTTPException(status_code=500, detail="Setting update failed") from exc
        return {"message": message, "settings": runtime_settings()}

    @app.get(
        "/model",
        summary="Deployed model metadata",
        tags=["trading"],
    )
    async def get_model_info(_auth: None = auth_dep) -> dict[str, Any]:
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
