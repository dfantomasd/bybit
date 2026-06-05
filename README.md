# Bybit AI Trader

An autonomous AI-driven trading system for the Bybit cryptocurrency exchange. Built for personal research and education.

> **LEGAL NOTICE**: This software is provided for educational and personal research purposes only. It is not financial advice. Trading cryptocurrency carries extreme risk. You may lose all of your invested capital. See [LEGAL_NOTICE.md](LEGAL_NOTICE.md) for the full disclaimer.

---

## Risk Warning

- This system can lose money. Past performance in backtests does not guarantee future results.
- **Default mode is TESTNET**. The system will NOT trade real money until you explicitly enable LIVE mode following the procedure in [RUNBOOK.md](RUNBOOK.md).
- Never invest more than you can afford to lose entirely.
- Review [THREAT_MODEL.md](THREAT_MODEL.md) before connecting real API keys.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker + Docker Compose v2
- A Bybit account (use testnet for development: https://testnet.bybit.com/)

### 1. Clone and bootstrap

```bash
git clone <your-fork> bybit-trader
cd bybit-trader
bash scripts/bootstrap.sh
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — replace ALL CHANGE_ME values with real credentials
# IMPORTANT: Never commit .env to git
nano .env
```

### 3. Start in testnet mode

```bash
make docker-up
# Or for development with hot-reload:
make docker-up-dev
```

For the current safe monitoring MVP, use the minimal compose file instead:

```bash
docker compose -f docker-compose.mini.yml up -d --build
```

This starts only `trader-core` and Redis by default, and expects `POSTGRES_DSN`
to point at an external PostgreSQL database such as Supabase. Telegram commands
are read-only: `/status`, `/balance`, `/positions`, and `/help`.

To use the bundled local PostgreSQL service instead:

```bash
docker compose -f docker-compose.mini.yml --profile local-db up -d --build
```

### 4. Verify

```bash
# Check health
curl http://localhost:8080/health

# View metrics
curl http://localhost:8080/metrics

# View logs
make docker-logs-trader
```

---

## Architecture Overview

```
Market Data (WebSocket)
        |
        v
  Feature Pipeline  ──────────────────────────────────┐
        |                                              |
        v                                              |
  Regime Detector                               Audit Logger
        |                                              |
        v                                              |
  Strategy / RL Model                                  |
        |                                              |
        v                                              |
  Trade Proposal                                       |
        |                                              |
        v                                              |
  [RISK MANAGER] ← final authority, cannot be bypassed |
        |                                              |
     approved?                                         |
        |                                              |
        v                                              |
  Execution Engine ──── REST API ──── Bybit Exchange   |
        |                                              |
        v                                              |
  Reconciliation Loop ─────────────────────────────────┘
        |
        v
  PostgreSQL (audit trail) + Redis (state cache)
        |
        v
  Prometheus + Grafana + Loki
```

### Hot path (latency-sensitive)
WebSocket feed → Feature pipeline → Model inference → Risk check → Order submission

### Slow path (background)
Reconciliation, model training, reporting, Telegram notifications

---

## Trading Modes

| Mode | Description | Real Money | Requires |
|------|-------------|-----------|---------|
| `TESTNET` | Orders to Bybit testnet | No | `BYBIT_USE_TESTNET=true` |
| `SHADOW` | Compute signals, never submit | No | Default mode |
| `CANARY_LIVE` | Live with severely reduced sizes | Yes | Explicit activation |
| `LIVE` | Full live trading | Yes | `LIVE_MODE=true` + `TRADING_MODE=LIVE` |

**The system defaults to TESTNET. Reaching LIVE mode requires two independent flags.**

---

## Configuration

All configuration is via environment variables (or `.env` file). See `.env.example` for the complete reference.

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADING_MODE` | `TESTNET` | Execution mode |
| `RISK_PROFILE` | `CONSERVATIVE` | Risk parameters |
| `LIVE_MODE` | `false` | Must be `true` for live trading |
| `BYBIT_USE_TESTNET` | `true` | Must be `true` for non-live modes |
| `MAX_POSITIONS` | `2` | Max concurrent open positions |

Risk profile YAML configuration: `config/profiles.yaml`
Feature flags: `config/feature_flags.yaml`

---

## Security Notes

- API keys are stored as `SecretStr` and never appear in logs or `repr()` output.
- The `LIVE_MODE` flag is a deliberate double-gate against accidental live activation.
- The Risk Manager is the final authority on all orders — no model can bypass it.
- All order IDs (`orderLinkId`) are unique and idempotent — duplicate submissions are safe.
- See [SECURITY.md](SECURITY.md) for the full security policy.
- See [THREAT_MODEL.md](THREAT_MODEL.md) for known attack vectors and mitigations.

---

## Development

```bash
# Install dev dependencies
make install

# Run unit tests
make test-unit

# Lint + typecheck
make check

# Security scan
make security

# Run all tests
make test
```

### Project Structure

```
src/trader/
├── app.py              # Application entry point
├── config.py           # Pydantic-settings configuration
├── domain/
│   ├── enums.py        # All enumerations
│   ├── models.py       # Pydantic domain models
│   ├── events.py       # Event bus event types
│   └── errors.py       # Custom exception hierarchy
├── monitoring/
│   ├── metrics.py      # Prometheus metrics registry
│   ├── logging.py      # Structlog configuration
│   └── health.py       # Health check service
└── api/
    └── fastapi_app.py  # Read-only observability API
```

---

## Observability

| Service | URL | Purpose |
|---------|-----|---------|
| FastAPI | http://localhost:8080 | System status, positions |
| Grafana | http://localhost:3000 | Dashboards |
| Prometheus | Internal only | Metrics scrape |
| Loki | Internal only | Log aggregation |

---

## Licence

UNLICENSED — Private use only. See [LEGAL_NOTICE.md](LEGAL_NOTICE.md).
