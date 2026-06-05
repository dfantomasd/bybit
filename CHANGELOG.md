# Changelog

All notable changes to the Bybit AI Trader project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

---

## [0.3.0] — 2026-06-05

### Phase 3: WebSocket Layer, Order State Machine, Reconciliation & Chaos Tests (Week 3)

This release delivers the complete WebSocket layer, order lifecycle state machine,
reconciliation service, event bus, and orderbook analytics. All new code is async-native.

### Added

#### Exchange Package (`src/trader/exchange/`)

- `bybit_ws_public.py` (new): `BybitPublicWebSocket` — manages Bybit V5 public
  WebSocket connections (orderbook, trades, ticker, kline, liquidations).
  Features: automatic reconnect, heartbeat/ping every 20s with 5s pong timeout,
  watchdog reconnect after 30s silence, local orderbook with snapshot+delta merge,
  sequence number validation (gap detection → invalidation), latency measurement,
  typed event emission to asyncio.Queue, reconnect counter metric.

- `bybit_ws_private.py` (new): `BybitPrivateWebSocket` — manages Bybit V5 private
  WebSocket connections (order, execution, position, wallet).
  Features: HMAC-SHA256 auth on connect, reconnect with re-auth, typed event emission
  (OrderUpdateEvent, PositionUpdateEvent, BalanceUpdateEvent), event deduplication by
  orderId+updateTime, audit logging via structlog.

- `state_machine.py` (new): `OrderStateMachine` + `OrderStateStore` — complete order
  lifecycle state machine with 11 states and validated transition graph.
  `VALID_TRANSITIONS` dict covers all OrderStatus values.
  `OrderStateStore` provides asyncio.Lock-protected in-memory store with `create`,
  `get`, `transition`, `get_all_active`, `get_by_status` operations.
  `InvalidStateTransitionError` raised on illegal transitions.

- `reconciliation.py` (new): `ReconciliationService` — periodic reconciliation of
  local state vs Bybit exchange. Runs every 15-60s (configurable) and after every
  reconnect. Checks: positions, orders, balance, stop-loss presence.
  Triggers safe mode on positions without SL or unknown positions.
  `ReconciliationDiff` dataclass captures structured diff output.

- `reconnect_supervisor.py` (new): `ReconnectSupervisor` — WebSocket reconnect lifecycle
  manager with exponential backoff (1→2→4→…→60s), ±20% jitter, max 20 reconnects/hour,
  10s settle window blocking new trade entries, alert on >3 reconnects in 5 min,
  downtime tracking, Prometheus metrics.

#### Data Package (`src/trader/data/`)

- `__init__.py` (new): Package init.

- `event_bus.py` (new): `EventBus` — in-process async event bus with 5 named queues
  (market_data, execution, risk, persistence, system). Features: bounded queues
  (configurable maxsize), non-blocking publish, dropped event counter,
  dead letter queue for critical events, graceful shutdown with drain, async generator
  consumer (`subscribe`), queue size and drop count introspection.

- `orderbook.py` (new): `LocalOrderBook` + analytics utilities.
  `LocalOrderBook`: snapshot+delta management, sequence gap detection, best bid/ask,
  mid-price, spread, imbalance. Analytics: `compute_microprice`,
  `compute_weighted_midprice`, `compute_depth_imbalance`, `compute_top_n_imbalance`,
  `detect_abnormal_spread`.

#### Tests

- `tests/unit/test_state_machine.py` (new): 22 tests covering all valid transitions,
  terminal states, invalid transition errors, history recording, timing, and
  OrderStateStore CRUD operations.

- `tests/unit/test_orderbook.py` (new): 19 tests covering snapshot init, delta updates,
  zero-qty removal, sequence gap invalidation, snapshot reset, best bid/ask,
  mid-price, spread, imbalance, microprice, weighted midprice, abnormal spread detection.

- `tests/unit/test_event_bus.py` (new): 9 tests covering publish/consume, bounded queue
  dropping, critical events to dead letter queue, drain, queue sizes, drop counters.

- `tests/unit/test_reconnect_supervisor.py` (new): 7 tests covering backoff sequence,
  jitter bounds, max cap, entry blocking, stability, downtime tracking, reconnect count.

- `tests/chaos/test_ws_chaos.py` (new): 10 chaos scenarios (each run as both async and
  sync wrapper = 20 test functions) covering snapshot→delta flow, sequence gap
  invalidation, reconnect rebuild, stale WS reconnect, duplicate event idempotency,
  queue full dropping, reconciliation after reconnect, position-without-SL safe mode,
  unknown order detection.

- `tests/chaos/__init__.py` (new): Package init.

### Changed

- Total test count: 231 → **309** (78 new tests added)

---

## [0.2.0] — 2026-06-05

### Phase 2: Bybit Exchange Adapter (Week 2)

This release delivers the complete exchange adapter layer, providing the
primary interface between the trading system and Bybit V5 API.
All new exchange code is async, uses pybit under the hood (sync-in-threadpool),
and applies Decimal arithmetic for all financial calculations.

### Added

#### Exchange Package (`src/trader/exchange/`)

- `endpoint_selector.py` *(extended)*: `EndpointSelector` class mapping `BybitRegion`
  enum to correct REST/WS endpoints for GLOBAL, NL, EEA, TR, KZ, GE, AE, ID regions.
  Properties: `rest_base`, `ws_public_base`, `ws_private_base`.
  Regional live endpoints follow `api.{region}.bybit.com` pattern;
  testnet always falls back to `api-testnet.bybit.com`.

- `auth.py` *(extended)*:
  - `HMACAuthenticator`: Signs requests with HMAC-SHA256 per Bybit V5 spec.
    Pre-sign string: `{timestamp}{api_key}{recv_window}{params}`.
  - `RSAAuthenticator`: Signs requests with RSA-SHA256 (PKCS#1 v1.5) for API keys
    configured with RSA public keys. Returns base64-encoded signature.
  - `verify_bybit_signature()`: Webhook HMAC-SHA256 signature verification with
    constant-time comparison (timing-attack resistant).

- `rate_limiter.py` *(extended)*: Adaptive token-bucket rate limiter with
  per-endpoint tracking keyed by `{METHOD}:{path}`.
  Reads `X-Bapi-Limit-Status`, `X-Bapi-Limit`, `X-Bapi-Limit-Reset-Timestamp` headers.
  Warns at 70%, 85%, 95% usage. Exponential backoff with ±25% jitter on 429/10006 errors.
  Emits Prometheus gauges for remaining capacity and usage percentage.

- `bybit_rest.py` (new): Async REST client wrapping pybit's synchronous `HTTP` session
  in a `ThreadPoolExecutor`. Complete V5 API coverage:
  - Server: `get_server_time`
  - Account: `get_wallet_balance`, `get_account_info`, `get_api_key_info`
  - Market: `get_instruments_info`, `get_tickers`, `get_kline`, `get_orderbook`,
    `get_recent_trades`, `get_funding_rate_history`, `get_open_interest`,
    `get_long_short_ratio`
  - Orders: `place_order`, `amend_order`, `cancel_order`, `get_open_orders`,
    `get_order_history`
  - Positions: `get_positions`, `set_leverage`, `set_trading_stop`
  - Executions: `get_executions`, `get_closed_pnl`, `get_fee_rate`
  - retCode error mapping: 10003/10004 → `AuthenticationError`, 10006 → `RateLimitError`,
    110007 → `InsufficientFundsError`, 110013/110014/110017/110025 → `OrderRejectedError`

- `order_mapper.py` (new): Bidirectional mapper between domain objects and Bybit API dicts.
  - `intent_to_params()`: `OrderIntent` → pybit kwargs, including TP/SL order types,
    positionIdx (one-way mode), reduceOnly flag, timeInForce
  - `round_price()` / `round_qty()`: Decimal-only rounding to tick_size / qty_step
  - `ws_order_to_event()`: WebSocket order update → normalised event dict
  - `ws_execution_to_fill()`: WebSocket execution → `Fill` domain model
  - `rest_position_to_model()`: REST position dict → `Position` domain model
  - `rest_balance_to_model()`: REST coin balance dict → `Balance` domain model
  - `instruments_info_to_model()`: REST instruments dict → `InstrumentInfo` domain model

- `idempotency.py` (new): In-memory order idempotency manager.
  - `generate_order_link_id()`: Generates unique ≤36-char IDs in format
    `{env_short}-{YYMMDD}-{strat[:4]}-{prop[:8]}-{hex6}`
  - `check_duplicate()`, `register_intent()`: Duplicate prevention before submission
  - State machine: `CREATED_LOCAL → SUBMITTING → REST_ACCEPTED → WS_CONFIRMED → FILLED/CANCELLED`
  - Enforces valid state transitions; raises `OrderRejectedError` on invalid transitions
  - `pending_count()`, `all_states()` for reconciliation introspection

- `preflight.py` (new): `PreflightChecker` service running 10 checks before trading:
  1. REST connectivity
  2. Server time drift (warn >5s, fail >30s)
  3. API key validity
  4. API key permissions (warns if Wallet/withdrawal permission present)
  5. Account type (recommends UNIFIED)
  6. Trading categories accessibility
  7. Balance (warns if <10 USDT equity)
  8. Region compatibility
  9. Testnet vs live consistency (warns in LIVE mode)
  10. Leverage settings
  Returns `PreflightReport`; critical check failures set `passed=False`.

- `bybit_adapter.py` (new): High-level adapter composing all exchange components.
  Domain-typed methods: `initialize()`, `get_balance()`, `get_positions()`,
  `get_open_orders()`, `get_instrument_info()`, `place_order()`, `cancel_order()`,
  `set_trading_stop()`, `reconcile()`, `health_check()`.
  `place_order()` integrates idempotency manager for duplicate prevention.

#### Tests (`tests/unit/`)

- `test_endpoint_selector.py` (new): 20+ tests — all regions have correct URLs,
  HTTPS/WSS enforced, testnet switching, `validate_region_compatibility()`, repr.

- `test_rate_limiter.py` (new): 18+ tests — token bucket allow/block, header parsing
  (case-insensitive), usage percentage calculation, warning thresholds at 70/85/95%,
  exponential backoff with jitter, consecutive 429 tracking.

- `test_auth.py` (new): 15+ tests — HMAC known test vector, signature determinism,
  headers completeness, RSA base64 output, webhook verification pass/reject cases.

- `test_order_mapper.py` (new): 25+ tests — intent_to_params all fields, Decimal rounding
  to tick/step, reduce_only / close_on_trigger flags, WS order event, WS fill, REST position.

- `test_idempotency.py` (new): 15+ tests — ID uniqueness, ≤36 char constraint, date/env
  in ID, duplicate detection, full state transition chain, invalid transition rejection,
  resubmission prevention.

- `test_preflight.py` (new): 20+ tests — full run all-green passes, critical failure blocks,
  non-critical failure passes with warning, time drift >30s fails, API key invalid,
  withdrawal permission warning, all individual checks verified.

### Technical Notes

- pybit is synchronous; all REST calls run in `ThreadPoolExecutor` (4 workers default)
  via `asyncio.get_event_loop().run_in_executor()` — event loop is never blocked.
- All price/quantity arithmetic uses Python `Decimal` (never `float`) to prevent
  floating-point rounding errors in financial calculations.
- API key / API secret never appear in log output — structlog structured events use
  only `order_link_id`, `symbol`, `ret_code` etc.
- retCode 110043 ("set leverage not modified") is treated as a non-fatal info log,
  not an exception.

### Test Results

- Total tests: 231 (85 Phase 1 + 146 Phase 2)
- All 231 passing
- Exchange package coverage: auth 100%, endpoint_selector 97%, rate_limiter 94%,
  idempotency 91%, preflight 88%, order_mapper 76%

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
