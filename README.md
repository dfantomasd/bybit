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

For Render Free deployment, see [RENDER.md](RENDER.md). The included
`render.yaml` creates a single Free Web Service and uses Supabase for Postgres.

### 4. Verify

```bash
# Check health
curl -H "X-API-Key: $INTERNAL_API_KEY" http://localhost:8080/health

# View metrics
curl -H "X-API-Key: $INTERNAL_API_KEY" http://localhost:8080/metrics

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

**The system defaults to TESTNET endpoints with SHADOW execution enabled.**
In that state it computes signals and simulates positions, but does not submit
orders until shadow mode is disabled. Reaching LIVE mode requires two independent
flags.

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
| `INTERNAL_API_KEY` | generated | API key for `/health`, `/status`, `/metrics` |
| `TRADE_JOURNAL_ENABLED` | `true` | Store signals, risk decisions, order events, and closed PnL in Postgres/Supabase |
| `PERFORMANCE_FILTER_ENABLED` | `true` | Temporarily skip symbols with weak recent closed PnL |
| `PERFORMANCE_MIN_CLOSED_TRADES` | `5` | Minimum closed trades before a symbol can be performance-blocked |
| `PERFORMANCE_MAX_SYMBOL_LOSS_USD` | `-2.0` | Loss threshold for blocking a symbol over the lookback window |
| `PERFORMANCE_LOOKBACK_DAYS` | `7` | Closed PnL lookback window for symbol performance |
| `PERFORMANCE_MIN_TRADABLE_SYMBOLS` | `2` | Relax performance blocks if too few symbols would remain tradable |
| `CLOSED_PNL_REFRESH_INTERVAL_SECONDS` | `300` | How often recent Bybit closed PnL is imported |
| `PROFIT_MANAGER_ENABLED` | `true` | Manage open positions after entry |
| `TRAILING_STOP_ENABLED` | `true` | Move profitable positions to breakeven and enable Bybit trailing stop |
| `TRAILING_ACTIVATION_PCT` | `0.45` | Unrealised profit percent before trailing stop is enabled |
| `TRAILING_DISTANCE_PCT` | `0.30` | Trailing stop distance as percent of current mark price |
| `BREAKEVEN_STOP_OFFSET_PCT` | `0.03` | Small offset beyond entry when moving SL to breakeven |
| `POSITION_SYNC_INTERVAL_SECONDS` | `30` | How often exchange positions are synced after TP/SL closures |

Risk profiles:

| Profile | Intent |
|---------|--------|
| `CONSERVATIVE` | Small, safer linear trades |
| `MODERATE` | Balanced frequency and risk |
| `AGGRESSIVE` | Larger exposure envelope |
| `SCALP` | More frequent small-risk entries with shorter cooldown |

Autonomous execution presets:

| Goal | Key settings |
|------|--------------|
| Observe only | `TRADING_MODE=SHADOW`, `SHADOW_MODE=true`, `BYBIT_USE_TESTNET=false` |
| Testnet autopilot | `TRADING_MODE=TESTNET`, `SHADOW_MODE=false`, `BYBIT_USE_TESTNET=true`, `RISK_PROFILE=SCALP` |
| Live shadow | `TRADING_MODE=SHADOW`, `SHADOW_MODE=true`, `BYBIT_USE_TESTNET=false` |
| Canary live | `TRADING_MODE=CANARY_LIVE`, `LIVE_MODE=true`, `SHADOW_MODE=false`, `BYBIT_USE_TESTNET=false` |
| Full live | `TRADING_MODE=LIVE`, `LIVE_MODE=true`, `SHADOW_MODE=false`, `BYBIT_USE_TESTNET=false` |

Risk profile YAML configuration: `config/profiles.yaml`
Feature flags: `config/feature_flags.yaml`

### Trade memory and performance filter

When `TRADE_JOURNAL_ENABLED=true`, the bot creates Postgres tables on startup
and records:

- generated trade signals with feature values and regime;
- risk decisions, including rejection reasons;
- order events, including shadow, placed, and failed orders;
- recent closed PnL imported from Bybit.

When `PERFORMANCE_FILTER_ENABLED=true`, the strategy loop uses stored closed PnL
to temporarily skip symbols that have at least
`PERFORMANCE_MIN_CLOSED_TRADES` closed trades and total PnL below
`PERFORMANCE_MAX_SYMBOL_LOSS_USD` during `PERFORMANCE_LOOKBACK_DAYS`.
This is adaptive risk control, not a profit guarantee. If Postgres/Supabase is
unavailable, trading continues without the filter.

### Open position management

The entry order sets full-position TP/SL immediately. Separately, when
`PROFIT_MANAGER_ENABLED=true` and `TRAILING_STOP_ENABLED=true`, the bot monitors
open positions. Once unrealised PnL reaches `TRAILING_ACTIVATION_PCT`, it moves
the stop near breakeven and asks Bybit to manage an exchange-side trailing stop
using `/v5/position/trading-stop`.

The same strategy loop also syncs exchange positions every
`POSITION_SYNC_INTERVAL_SECONDS`. This clears local risk/execution state after
Bybit closes a position by TP/SL, so the bot can open the next valid signal
without requiring a restart.

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

## Telegram Controls

Use `/start` to open the button menu. The bot supports status, balance,
positions, recent signals, active symbols, closed PnL, pause/resume, shadow vs
active execution, risk profile changes, and emergency stop. Risk changes and
active execution require `/confirm`.

`/mode shadow` keeps evaluating signals without submitting orders.
`/mode active` sends orders to the currently configured exchange endpoint after
confirmation. Configure `BYBIT_USE_TESTNET=true` for testnet execution.

---

## Licence

UNLICENSED — Private use only. See [LEGAL_NOTICE.md](LEGAL_NOTICE.md).
