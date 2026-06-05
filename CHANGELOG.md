# Changelog

All notable changes to the Bybit AI Trader project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

---

## [0.1.0] — 2026-06-05

### Phase 1: Project Skeleton

This release establishes the complete project foundation including domain models,
configuration management, observability infrastructure, and CI/CD pipelines.
No live trading capability is included; the system defaults to TESTNET mode.

### Added

#### Domain Layer (`src/trader/domain/`)
- `enums.py`: Complete enumeration set — `TradingMode`, `SystemStatus`, `RiskProfile`,
  `MarketRegime`, `OrderStatus`, `RiskDecisionStatus`, `MarketType`, `OrderSide`,
  `OrderType`, `BybitRegion`, `VolatilityLevel`, `KillSwitchMode`
- `models.py`: Full Pydantic v2 domain models with validation — `MarketEvent`,
  `FeatureVector`, `RegimeContext`, `TradeProposal`, `RiskDecision`, `OrderIntent`,
  `Position`, `Balance`, `InstrumentInfo`, `Fill`, `ReconciliationResult`,
  `HealthStatus`, `PreflightReport`, `AuditEvent`, `ModelMetadata`
- `events.py`: Event bus event hierarchy — market data events, account events,
  strategy/risk pipeline events, system events, alert events
- `errors.py`: Custom exception hierarchy rooted at `TradingSystemError`

#### Configuration (`src/trader/config.py`)
- `Settings` class using pydantic-settings v2 with env file, Docker secrets support
- Safety invariants enforced in `model_post_init`: TESTNET/SHADOW modes require
  `BYBIT_USE_TESTNET=true`; LIVE mode requires explicit `LIVE_MODE=true` opt-in
- `RiskProfileConfig` frozen dataclass for CONSERVATIVE / MODERATE / AGGRESSIVE profiles
- Pre-built profile configs with all risk parameters

#### Monitoring (`src/trader/monitoring/`)
- `metrics.py`: Singleton Prometheus metrics registry covering market data, features,
  regime, proposals, risk, orders, positions, reconciliation, WebSocket, REST, ML models,
  and kill-switch metrics (50+ metrics total)
- `logging.py`: Structlog configuration with JSON/console renderers, secret redaction
  processor, async-safe context binding
- `health.py`: `HealthChecker` service with individual and aggregate health checks
  for PostgreSQL, Redis, Bybit REST, WebSocket, model freshness, feature freshness

#### API (`src/trader/api/fastapi_app.py`)
- Read-only FastAPI application with API key authentication
- Endpoints: `/health`, `/status`, `/positions`, `/metrics`, `/regime`, `/model`
- Security headers middleware, CORS, request logging

#### Application Entry Point (`src/trader/app.py`)
- Async `TradingApplication` lifecycle manager
- Signal handling (SIGTERM, SIGINT) with graceful shutdown
- Ordered startup: config → logging → preflight → HTTP server → main loop

#### Infrastructure
- `pyproject.toml`: uv-compatible project config with all dependencies and tool config
- `Dockerfile`: Multi-stage build (builder + runtime), non-root user `trader:1000`
- `docker-compose.yml`: Production compose with network segmentation, resource limits,
  health checks, secrets management (postgres, grafana passwords)
- `docker-compose.dev.yml`: Dev override with source mounts, exposed ports, hot-reload
- `migrations/env.py`: Async Alembic environment for PostgreSQL migrations
- `migrations/script.py.mako`: Migration template

#### Configuration Files
- `.env.example`: Complete environment variable template with section headers
- `config/profiles.yaml`: Risk profile parameters for all three profiles
- `config/symbols.yaml`: Symbol whitelist/blacklist configuration
- `config/feature_flags.yaml`: Feature flag registry (all safe defaults — disabled)
- `config/logging.yaml`: Logging configuration documentation

#### CI/CD (`.github/workflows/`)
- `ci.yml`: Lint (ruff + mypy), security (bandit + gitleaks + pip-audit),
  test (pytest + coverage, Python 3.11 + 3.12), Docker build + Trivy scan
- `security.yml`: Scheduled daily security scans — gitleaks, bandit, pip-audit, Trivy

#### Tests (`tests/`)
- `conftest.py`: Shared fixtures for all domain models and mock settings
- `test_enums.py`: Comprehensive enum value and count tests
- `test_domain_models.py`: Pydantic model validation, validators, frozen enforcement
- `test_config.py`: Settings defaults, secret redaction, safety gate enforcement

#### Developer Experience
- `Makefile`: 25+ targets for install, lint, test, docker, migrate, clean
- `scripts/bootstrap.sh`: First-run setup script (Python check, uv install, venv, .env, migrations)

#### Documentation
- `README.md`: Project overview, architecture diagram, quick start, trading modes
- `DECISIONS.md`: Architecture Decision Records (ADR-001 through ADR-008)
- `CHANGELOG.md`: This file
- `SECURITY.md`: Security policy and responsible disclosure
- `THREAT_MODEL.md`: Known attack vectors and mitigations
- `RUNBOOK.md`: Operational procedures for production management
- `LEGAL_NOTICE.md`: Full legal disclaimer and terms of use

### Security

- All secret fields use `pydantic.SecretStr` — never appear in `str()` / `repr()`
- Structlog secret redaction processor masks API keys, tokens, passwords in all logs
- Two-gate live trading protection: `TRADING_MODE=LIVE` AND `LIVE_MODE=true` both required
- Non-root container user (`trader:1000`) enforced in Dockerfile
- No secrets in any committed file — all templates use `CHANGE_ME` placeholders

### Known Limitations (Phase 1)

- No strategy implementation (Phase 2)
- No RL model training pipeline (Phase 2)
- No WebSocket feed implementation (Phase 2)
- No Telegram notification integration (Phase 2)
- No actual order execution (Phase 2)
- Database schema migrations pending model definitions (Phase 2)
