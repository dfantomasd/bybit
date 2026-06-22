# AGENTS.md

## Cursor Cloud specific instructions

Bybit AI Trader is a single-process Python app (`trader` console script → `trader.app:main_sync`)
that hosts a FastAPI observability API plus ~18 asyncio background loops (market-data WS,
feature pipeline, strategy/risk/execution loops, trade journal, etc.). Standard dev commands
live in the `Makefile` and `README.md` (`## Development`); the notes below only capture the
non-obvious bits for this VM.

### Environment

- The startup update script installs deps into a project venv at `.venv` via `uv`. Activate it
  for any dev command: `source .venv/bin/activate`. `uv` itself is installed for the user and is
  invokable as `python3 -m uv` (it is not always on `PATH`).
- Python 3.12 is used even though the project targets 3.11 (`requires-python>=3.11`); the app and
  tests run fine on 3.12.
- Reinstalling deps does not affect an already-running `trader` process (it loaded modules at
  start). Restart the process to pick up dependency or source changes. The package is installed
  editable (`-e`), so source edits are picked up on the next process start — but a **non-editable**
  install would silently run stale code from `site-packages`, so always keep the editable install.

### Running the app (local dev, no Docker)

- Create a local `.env` (gitignored). The safe, no-credentials dev configuration is **SHADOW** mode
  (`TRADING_MODE=SHADOW`, `BYBIT_USE_TESTNET=false`): it computes signals but never submits orders
  and needs no Bybit API keys. Do **not** use `TRADING_MODE=TESTNET` without `BYBIT_USE_TESTNET=true`
  — config validation raises `ValueError`. `CANARY_LIVE`/`LIVE` additionally require
  `LIVE_MODE=true` + `LIVE_ARMED=true` and are real-money modes; never enable them here.
- Start with `source .venv/bin/activate && trader`. The only port bound is `FASTAPI_PORT` (8080);
  `PROMETHEUS_PORT` (9090) is defined but no separate server listens there — metrics are served at
  `GET /metrics` on 8080.
- Observability endpoints: `GET /livez` and `GET /readyz` need no auth; `GET /health`, `/status`,
  `/metrics`, `/positions`, `/dashboard` require header `X-API-Key: <INTERNAL_API_KEY>`. If
  `INTERNAL_API_KEY` is unset a random key is generated each boot, so set it in `.env` (e.g.
  `INTERNAL_API_KEY=dev-key`) to call authed endpoints.

### Postgres / Redis (optional services)

- In SHADOW the app boots fine without Postgres/Redis (`PREFLIGHT_POSTGRES_REQUIRED=false`,
  `REDIS_REQUIRED=false` are the defaults). They are only required to boot in `CANARY_LIVE`/`LIVE`.
- To exercise the persistence layer, run local Postgres + Redis (installed via apt in the VM:
  `sudo pg_ctlcluster 16 main start`, `sudo redis-server --daemonize yes`). Create a `trader`
  role/db (password `trader`) and point `POSTGRES_DSN=postgresql+asyncpg://trader:trader@localhost:5432/trader`
  and `REDIS_URL=redis://localhost:6379/0`, with `TRADE_JOURNAL_ENABLED=true`.
- There are **no Alembic version scripts** (`migrations/versions/` is empty, `target_metadata=None`),
  so `alembic upgrade head` is a no-op. The trade journal **auto-creates its ~16 tables on startup**
  when connected — that is the real schema-bootstrap path, not Alembic.

### Network limitation in this VM

- Bybit **REST** endpoints (e.g. `/v5/position/list`, kline backfill, transaction log) are
  geo-blocked by CloudFront (HTTP 403) from the VM region — the app logs warnings and continues.
  Bybit **public WebSocket** (`wss://stream.bybit.com/v5/public`) **does** connect, so live
  orderbook/trade/kline data still flows and features are computed. Expect `/health` to report
  `bybit_rest: false` and `overall: degraded`; this is environmental, not a code bug.

### Tests / lint / typecheck

- `make test-unit` (or `pytest tests/unit -q`) runs fully offline with mocked services; all unit
  tests pass and coverage gate is 55% (`fail_under`).
- `make lint` (`ruff check src/ tests/`) passes.
- Known pre-existing dev-check issues on `main` (not environment-related): `make format-check`
  (`ruff format --check`) flags `src/trader/app.py` and `src/trader/config.py`, and `make typecheck`
  (`mypy`) fails on a numpy stub that uses 3.12-only `type` syntax while mypy targets 3.11.
