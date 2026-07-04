"""Postgres-backed trading memory for signals, decisions, orders, and PnL."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import ssl
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, cast
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import asyncpg
import structlog

from trader.domain.models import FeatureVector, RegimeContext, RiskDecision, TradeProposal
from trader.ml.model_selection import model_selection_metrics, selection_reason
from trader.training.labels import (
    LABEL_SCHEMA_VERSION,
    CostModelBps,
    active_label_schema_version,
    build_directional_outcome,
)

log = structlog.get_logger(__name__)

_MODEL_PROMOTION_ADVISORY_LOCK_ID = 926_202_606
_POOL_CLOSE_TIMEOUT_SECONDS = 5.0
_DEFAULT_FETCH_TIMEOUT_SECONDS = 30.0
_DEFAULT_POOL_MAX_SIZE = 5
_FETCH_TIMEOUT_LOG_INTERVAL_SECONDS = 60.0


def _optional_settings() -> Any | None:
    try:
        from trader.config import Settings

        return Settings()
    except Exception:
        return None


def _parse_command_rowcount(result: str | None) -> int:
    """Parse asyncpg command tag (e.g. ``DELETE 42``, ``INSERT 0 1``)."""
    parts = str(result or "").strip().split()
    if len(parts) >= 2 and parts[-1].isdigit():
        return int(parts[-1])
    return 0


def _decimal_or_none(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _parse_dec(v: Any) -> Decimal | None:
    try:
        return Decimal(str(v)) if v not in (None, "", "0", 0) else None
    except Exception:
        return None


def _query_label_schema_version() -> str:
    settings = _optional_settings()
    if settings is not None:
        return active_label_schema_version(use_tpsl_exit=bool(settings.MODEL_LABEL_USE_TPSL_EXIT))
    return LABEL_SCHEMA_VERSION


def _safe_connection_target(dsn: str) -> dict[str, Any]:
    """Return non-secret connection target details for diagnostics."""
    try:
        parsed = urlparse(dsn)
    except Exception:
        return {"parse_error": "invalid_dsn"}
    database = parsed.path.lstrip("/") or None
    if "?" in str(database):
        database = str(database).split("?", 1)[0]
    username = parsed.username or ""
    username_prefix = username.split(".", 1)[0] if username else None
    return {
        "scheme": parsed.scheme or None,
        "host": parsed.hostname,
        "port": parsed.port,
        "database": database,
        "username_prefix": username_prefix,
        "username_has_project_ref": "." in username if username else False,
    }


def asyncpg_pool_connect_kwargs(dsn: str) -> dict[str, Any]:
    """Normalize application DSNs into explicit asyncpg connection kwargs.

    asyncpg accepts URL query params in many cases, but making SSL explicit avoids
    ambiguity with Supabase/Supavisor pooler strings and keeps all entry points
    (runtime, training, backfill, discovery) consistent.
    """
    normalized = dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
    parsed = urlparse(normalized)
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    kept_query: list[tuple[str, str]] = []
    ssl_arg: bool | ssl.SSLContext | None = None
    for key, value in query_pairs:
        if key.lower() == "sslmode":
            mode = value.lower()
            if mode == "require":
                # libpq sslmode=require encrypts the connection without
                # requiring certificate-chain verification. Supabase pooler can
                # present a chain that is not trusted by Render's CA bundle; a
                # plain ssl=True in asyncpg verifies it and fails. CERT_OPTIONAL
                # would still validate a presented server cert and raise on an
                # untrusted chain (the client always receives a cert in a TLS
                # handshake, so CERT_OPTIONAL != "skip verification" here) —
                # only CERT_NONE actually reproduces libpq's sslmode=require.
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                ssl_arg = context
            elif mode in {"verify-ca", "verify-full"}:
                ssl_arg = True
            elif mode in {"disable", "allow", "prefer"}:
                ssl_arg = False
            continue
        kept_query.append((key, value))
    cleaned = urlunparse(parsed._replace(query=urlencode(kept_query, doseq=True)))
    # Expand DSN into explicit kwargs so the password is never in a loggable
    # string (asyncpg may log connection kwargs on error).
    kwargs: dict[str, Any] = {"dsn": cleaned}
    if parsed.username:
        kwargs["user"] = parsed.username
    if parsed.password:
        kwargs["password"] = parsed.password
        # Remove credentials from the DSN string to avoid leaking them in logs.
        cleaned_no_creds = urlunparse(
            parsed._replace(
                netloc=f"{parsed.hostname}:{parsed.port}" if parsed.port else (parsed.hostname or ""),
                query=urlencode(kept_query, doseq=True),
            )
        )
        kwargs["dsn"] = cleaned_no_creds
    if ssl_arg is not None:
        kwargs["ssl"] = ssl_arg
    return kwargs


def _schema_statement_label(statement: str) -> str:
    for line in statement.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("--"):
            return stripped[:160]
    return statement.strip()[:160]


def _split_sql_statements(script: str) -> list[str]:
    """Split SQL script into individual statements, respecting string literals.

    Handles single-quoted strings, double-quoted identifiers, and PostgreSQL
    dollar-quoted literals ($$...$$ or $tag$...$tag$).
    """
    import re

    statements: list[str] = []
    current: list[str] = []
    in_string = False
    string_char = ""
    dollar_tag: str | None = None  # non-None while inside $tag$...$tag$
    i = 0
    while i < len(script):
        ch = script[i]

        if dollar_tag is not None:
            # Inside a dollar-quoted block — scan for the closing tag
            end = script.find(dollar_tag, i)
            if end == -1:
                # Unterminated dollar-quote: consume the rest as-is
                current.append(script[i:])
                break
            tag_end = end + len(dollar_tag)
            current.append(script[i:tag_end])
            i = tag_end
            dollar_tag = None
            continue

        if in_string:
            current.append(ch)
            if ch == string_char:
                # Handle escaped quote by doubling ('' or "")
                if i + 1 < len(script) and script[i + 1] == string_char:
                    current.append(script[i + 1])
                    i += 2
                    continue
                in_string = False
        elif ch == "$":
            # Look for a dollar-quote opening tag: $identifier$ or $$
            m = re.match(r"\$([A-Za-z_]\w*)?\$", script[i:])
            if m:
                dollar_tag = m.group(0)
                current.append(dollar_tag)
                i += len(dollar_tag)
                continue
            else:
                current.append(ch)
        elif ch in ("'", '"'):
            in_string = True
            string_char = ch
            current.append(ch)
        elif ch == ";":
            stmt = "".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
        else:
            current.append(ch)
        i += 1
    remainder = "".join(current).strip()
    if remainder:
        statements.append(remainder)
    return statements


async def _execute_schema_script(conn: Any, script: str) -> None:
    """Execute DDL one statement at a time.

    Some poolers/proxies are less tolerant of large multi-statement packets during
    startup. Splitting schema bootstrap keeps the connection alive more reliably
    and makes failures point at the exact DDL statement.
    """
    statement_index = 0
    for statement in _split_sql_statements(script):
        if not statement:
            continue
        statement_index += 1
        try:
            await conn.execute(statement)
        except Exception as exc:
            label = _schema_statement_label(statement)
            raise RuntimeError(f"schema statement #{statement_index} failed: {label}: {exc}") from exc


def _paper_stats_from_rows(rows: list[Any]) -> dict[str, Any]:
    import math

    returns = [
        v
        for row in rows
        for v in [float(row.get("net_return_bps") or 0.0)]
        if math.isfinite(v)
    ]
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for ret in returns:
        equity += ret
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
    return {
        "count": len(returns),
        "avg_bps": (sum(returns) / len(returns)) if returns else None,
        "total_bps": sum(returns),
        "max_drawdown_bps": max_drawdown,
    }


def _model_horizon_minutes_from_metrics(metrics: dict[str, Any], default: int = 15) -> int:
    for key in ("horizon_minutes", "model_horizon_minutes"):
        raw_horizon = metrics.get(key)
        if raw_horizon is None:
            continue
        try:
            return int(raw_horizon)
        except (TypeError, ValueError):
            continue
    settings = _optional_settings()
    if settings is not None:
        return int(getattr(settings, "MODEL_AUTO_TRAIN_HORIZON_MINUTES", default) or default)
    return default


def _feature_schema_hash_from_metrics(metrics: dict[str, Any]) -> str:
    for key in ("feature_schema_hash", "source_feature_schema_hash"):
        value = metrics.get(key)
        if value:
            return str(value)
    return ""


def _champion_selection_thresholds() -> tuple[int, int, float]:
    min_paper_gate_count = 50
    min_wf_positive_folds = 3
    max_wf_std_bps = 25.0
    try:
        from trader.config import Settings

        settings = Settings()
        min_paper_gate_count = int(settings.MODEL_CHAMPION_MIN_PAPER_GATE_COUNT)
        min_wf_positive_folds = int(settings.MODEL_AUTO_PROMOTE_MIN_WF_POSITIVE_FOLDS)
        max_wf_std_bps = float(settings.MODEL_AUTO_PROMOTE_MAX_WF_STD_BPS)
    except Exception as exc:
        log.debug("trade_journal.champion_threshold_defaults", error=str(exc))
    return min_paper_gate_count, min_wf_positive_folds, max_wf_std_bps


def _parse_ts(v: Any) -> datetime | None:
    try:
        if v is None:
            return None
        return datetime.fromtimestamp(int(v) / 1000, tz=UTC)
    except Exception:
        return None


def _dt_from_ms(value: Any) -> datetime:
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=UTC)
    except Exception:
        return datetime.now(tz=UTC)


class TradeJournal:
    """Best-effort journal that never blocks trading when storage is unhealthy."""

    def __init__(
        self,
        postgres_dsn: str,
        enabled: bool = True,
        *,
        fetch_timeout_seconds: float | None = None,
        pool_max_size: int | None = None,
        reconnect_max_backoff_seconds: float | None = None,
        auth_circuit_breaker_min_backoff_seconds: float | None = None,
    ) -> None:
        self._dsn = postgres_dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
        self._enabled = enabled and bool(postgres_dsn)
        self._fetch_timeout_seconds = max(
            1.0,
            float(fetch_timeout_seconds if fetch_timeout_seconds is not None else _DEFAULT_FETCH_TIMEOUT_SECONDS),
        )
        self._pool_max_size = max(1, int(pool_max_size if pool_max_size is not None else _DEFAULT_POOL_MAX_SIZE))
        if reconnect_max_backoff_seconds is None or auth_circuit_breaker_min_backoff_seconds is None:
            from trader.config import Settings

            settings = Settings()
            if reconnect_max_backoff_seconds is None:
                reconnect_max_backoff_seconds = float(settings.TRADE_JOURNAL_RECONNECT_MAX_BACKOFF_SECONDS)
            if auth_circuit_breaker_min_backoff_seconds is None:
                auth_circuit_breaker_min_backoff_seconds = float(
                    settings.TRADE_JOURNAL_AUTH_CIRCUIT_BREAKER_MIN_BACKOFF_SECONDS
                )
        self._reconnect_max_backoff_seconds = max(60.0, float(reconnect_max_backoff_seconds))
        self._auth_circuit_breaker_min_backoff_seconds = max(
            60.0,
            float(auth_circuit_breaker_min_backoff_seconds),
        )
        self._pool: asyncpg.Pool | None = None
        self._diag_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._last_fetch_timeout_log_at: datetime | None = None
        self._schema_initialized: bool = False
        self._connect_failures: int = 0
        self._reconnect_blocked_until: datetime | None = None
        self._last_backoff_was_auth: bool = False
        self._last_connect_attempt_at: datetime | None = None
        self._last_connect_error_at: datetime | None = None
        self._last_connect_error: str | None = None
        self._last_read_error_at: datetime | None = None
        self._last_read_error: str | None = None
        self._last_successful_write_at: datetime | None = None
        self._last_write_error_at: datetime | None = None
        self._last_write_error: str | None = None
        self._consecutive_write_errors: int = 0
        self._background_tasks: set[asyncio.Task] = set()

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._pool is not None

    @property
    def is_configured(self) -> bool:
        return self._enabled

    @property
    def durable_state_healthy(self) -> bool:
        """False after 3 consecutive write errors — used to fail-closed in CANARY_LIVE/LIVE."""
        return self._consecutive_write_errors < 3

    @property
    def write_available(self) -> bool:
        """True only when durable storage is configured, connected, and not in write-error fail-close."""
        return bool(getattr(self, "_enabled", False)) and self._pool is not None and self.durable_state_healthy

    def write_health(self) -> dict[str, Any]:
        """Return write-health snapshot for observability and safety gates."""
        last_read_error_at = cast(datetime | None, getattr(self, "_last_read_error_at", None))
        last_connect_error_at = cast(datetime | None, getattr(self, "_last_connect_error_at", None))
        configured = bool(getattr(self, "_enabled", False))
        connected = self._pool is not None
        writable = configured and connected and self.durable_state_healthy
        return {
            "healthy": writable,
            "configured": configured,
            "connected": connected,
            "writable": writable,
            "durable_state_healthy": self.durable_state_healthy,
            "consecutive_write_errors": self._consecutive_write_errors,
            "last_successful_write_at": (
                self._last_successful_write_at.isoformat() if self._last_successful_write_at else None
            ),
            "last_write_error_at": (self._last_write_error_at.isoformat() if self._last_write_error_at else None),
            "last_write_error": getattr(self, "_last_write_error", None),
            "last_read_error_at": last_read_error_at.isoformat() if last_read_error_at else None,
            "last_read_error": getattr(self, "_last_read_error", None),
            "last_connect_error_at": (last_connect_error_at.isoformat() if last_connect_error_at else None),
            "last_connect_error": getattr(self, "_last_connect_error", None),
        }

    def reconnect_blocked_remaining_seconds(self) -> float:
        if self._reconnect_blocked_until is None:
            return 0.0
        remaining = (self._reconnect_blocked_until - datetime.now(tz=UTC)).total_seconds()
        return max(0.0, remaining)

    async def connect(self) -> None:
        async with self._connect_lock:
            await self._connect_locked()

    async def _connect_locked(self) -> None:
        """Body of connect(); caller must hold self._connect_lock.

        Re-checks self._pool under the lock so concurrent connect()/
        reconnect_if_needed() callers can't each pass the "not connected"
        check and race to create (and leak) duplicate connection pools.
        """
        if not self._enabled or self._pool is not None:
            return
        self._last_connect_attempt_at = datetime.now(tz=UTC)
        try:
            self._pool = await asyncpg.create_pool(
                **asyncpg_pool_connect_kwargs(self._dsn),
                min_size=1,
                max_size=self._pool_max_size,
                statement_cache_size=0,
            )
            self._last_connect_error_at = None
            self._last_connect_error = None
        except Exception as exc:
            await self._close_pool_after_failure("failed_pool_close_failed")
            self._pool = None
            self._last_connect_error_at = datetime.now(tz=UTC)
            self._last_connect_error = str(exc)
            self._schedule_reconnect_backoff(str(exc))
            log.warning("trade_journal.unavailable", error=str(exc))
            return

        try:
            if self._schema_initialized:
                await self._ping_pool()
            else:
                last_exc: Exception | None = None
                for attempt in range(3):
                    try:
                        await self._ensure_schema()
                        self._schema_initialized = True
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        error_text = str(exc)
                        if attempt < 2 and self._is_transient_schema_error(error_text):
                            delay = 2.0 * (attempt + 1)
                            log.warning(
                                "trade_journal.schema_bootstrap_retry",
                                attempt=attempt + 1,
                                delay_s=delay,
                                error=str(exc)[:160],
                            )
                            await asyncio.sleep(delay)
                            continue
                        if self._is_auth_circuit_breaker_error(error_text):
                            raise
                        raise
                if last_exc is not None:
                    raise last_exc
            self._last_connect_error_at = None
            self._last_connect_error = None
        except Exception as exc:
            self._last_connect_error_at = datetime.now(tz=UTC)
            self._last_connect_error = f"schema bootstrap degraded: {exc}"
            log.warning("trade_journal.schema_bootstrap_degraded", error=str(exc))
            await self._close_pool_after_failure("schema_bootstrap_failed")
            self._pool = None
            self._schedule_reconnect_backoff(str(exc))
            return

        self._connect_failures = 0
        self._reconnect_blocked_until = None
        self._last_backoff_was_auth = False
        log.info("trade_journal.connected", schema_degraded=False)

    async def _ping_pool(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.fetchval("SELECT 1")

    @staticmethod
    def _is_auth_circuit_breaker_error(error: str) -> bool:
        upper = error.upper()
        return (
            "ECIRCUITBREAKER" in upper
            or "EAUTHQUERY" in upper
            or "AUTHENTICATION" in upper
            or "TOO MANY AUTHENTICATION FAILURES" in upper
        )

    @staticmethod
    def _is_transient_schema_error(error: str) -> bool:
        upper = error.upper()
        if TradeJournal._is_auth_circuit_breaker_error(error):
            return False
        return (
            "CONNECTION WAS CLOSED" in upper
            or "CONNECTION RESET" in upper
            or "CONNECTION DOES NOT EXIST" in upper
            or "SERVER CLOSED THE CONNECTION" in upper
            or "SSL SYSCALL" in upper
            or "TIMEOUT" in upper
        )

    def _schedule_reconnect_backoff(self, error: str) -> None:
        self._connect_failures += 1
        base = 30.0
        max_backoff = self._reconnect_max_backoff_seconds
        if self._is_auth_circuit_breaker_error(error):
            delay = min(
                max_backoff,
                max(
                    self._auth_circuit_breaker_min_backoff_seconds,
                    base * (2 ** min(self._connect_failures - 1, 6)),
                ),
            )
        else:
            delay = min(max_backoff / 2.0, base * (2 ** min(self._connect_failures - 1, 4)))
        self._reconnect_blocked_until = datetime.now(tz=UTC) + timedelta(seconds=delay)
        self._last_backoff_was_auth = self._is_auth_circuit_breaker_error(error)
        log.warning(
            "trade_journal.reconnect_backoff",
            delay_s=delay,
            failures=self._connect_failures,
            error=error[:160],
        )

    async def reconnect_if_needed(self, *, min_interval: float = 30.0, force: bool = False) -> bool:
        """Try to reconnect after transient startup/network failures."""
        if not self._enabled:
            return False
        now = datetime.now(tz=UTC)
        blocked_until = self._reconnect_blocked_until
        if blocked_until is not None and now < blocked_until:
            if self._last_backoff_was_auth or not force:
                return self.is_enabled
        async with self._connect_lock:
            if self._pool is not None:
                if not force and self.durable_state_healthy:
                    return True
                await self._close_pool_after_failure("reconnect_pool_close_failed")
                self._pool = None
            if not force and self._last_connect_attempt_at is not None:
                age = now - self._last_connect_attempt_at
                if age < timedelta(seconds=min_interval):
                    return False
            await self._connect_locked()
        if self.is_enabled:
            self._consecutive_write_errors = 0
            self._last_write_error = None
            self._last_write_error_at = None
        return self.is_enabled

    async def close(self) -> None:
        if self._pool is not None:
            await self._close_pool_after_failure("pool_close_failed")
            self._pool = None

    async def _close_pool_after_failure(self, event: str) -> None:
        if self._pool is None:
            return
        try:
            await asyncio.wait_for(self._pool.close(), timeout=_POOL_CLOSE_TIMEOUT_SECONDS)
        except TimeoutError:
            log.warning("trade_journal.pool_close_timeout", timeout_s=_POOL_CLOSE_TIMEOUT_SECONDS)
            self._pool.terminate()
        except Exception as close_exc:
            log.debug(f"trade_journal.{event}", error=str(close_exc))

    async def _ensure_schema(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute("SET statement_timeout = '60s'")
            # Run the whole bootstrap as one transaction so Supabase's
            # transaction-mode pooler keeps every statement on the same backend
            # connection. Without this, each individual conn.execute() is its
            # own implicit transaction and the pooler is free to hand later
            # statements to a different backend mid-script, which has produced
            # "connection dropped"-style failures (surfacing from asyncpg as a
            # bare "'NoneType' object has no attribute 'decode'") partway
            # through bootstrap. It also makes bootstrap atomic: a failure no
            # longer leaves the first N statements committed and the rest missing.
            async with conn.transaction():
                await self._run_schema_scripts(conn)
        for coro, name in (
            (self._ensure_model_registry_indexes_deferred(), "trade-journal-model-registry-index"),
            (self._ensure_feature_snapshot_unique_index_deferred(), "trade-journal-feature-index"),
        ):
            task = asyncio.create_task(coro, name=name)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _run_schema_scripts(self, conn: Any) -> None:
        await _execute_schema_script(
                conn,
                """
            CREATE TABLE IF NOT EXISTS durable_order_state (
                order_link_id text PRIMARY KEY,
                proposal_id uuid,
                decision_id uuid,
                symbol text NOT NULL,
                side text NOT NULL,
                qty numeric NOT NULL,
                state text NOT NULL,
                exchange_order_id text,
                payload_hash text,
                retry_count integer NOT NULL DEFAULT 0,
                last_error text,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now()
            );
            CREATE INDEX IF NOT EXISTS idx_durable_order_state_state
                ON durable_order_state (state, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_durable_order_state_symbol
                ON durable_order_state (symbol, created_at DESC);

            CREATE TABLE IF NOT EXISTS trade_signals (
                proposal_id uuid PRIMARY KEY,
                created_at timestamptz NOT NULL,
                strategy_id text NOT NULL,
                symbol text NOT NULL,
                side text NOT NULL,
                confidence double precision NOT NULL,
                entry_price numeric,
                take_profit numeric,
                stop_loss numeric,
                requested_qty numeric NOT NULL,
                requested_notional_usd numeric,
                regime text,
                rationale text,
                features jsonb
            );
            CREATE INDEX IF NOT EXISTS idx_trade_signals_symbol_created
                ON trade_signals (symbol, created_at DESC);

            CREATE TABLE IF NOT EXISTS risk_decisions (
                decision_id uuid PRIMARY KEY,
                proposal_id uuid NOT NULL,
                created_at timestamptz NOT NULL,
                symbol text NOT NULL,
                status text NOT NULL,
                approved_qty numeric,
                approved_notional_usd numeric,
                reason text,
                triggered_rules jsonb NOT NULL,
                portfolio_heat double precision,
                current_drawdown_pct double precision,
                open_positions_count integer
            );
            CREATE INDEX IF NOT EXISTS idx_risk_decisions_symbol_created
                ON risk_decisions (symbol, created_at DESC);

            CREATE TABLE IF NOT EXISTS order_events (
                order_link_id text PRIMARY KEY,
                proposal_id uuid NOT NULL,
                decision_id uuid NOT NULL,
                created_at timestamptz NOT NULL,
                symbol text NOT NULL,
                side text NOT NULL,
                qty numeric NOT NULL,
                status text NOT NULL,
                exchange_order_id text,
                error text
            );
            CREATE INDEX IF NOT EXISTS idx_order_events_symbol_created
                ON order_events (symbol, created_at DESC);

            CREATE TABLE IF NOT EXISTS closed_pnl (
                closed_pnl_id text PRIMARY KEY,
                created_at timestamptz NOT NULL,
                symbol text NOT NULL,
                side text,
                qty numeric,
                avg_entry_price numeric,
                avg_exit_price numeric,
                closed_pnl numeric NOT NULL,
                raw jsonb NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_closed_pnl_symbol_created
                ON closed_pnl (symbol, created_at DESC);

            -- Market candles with retention
            CREATE TABLE IF NOT EXISTS market_candles (
                symbol text NOT NULL,
                interval text NOT NULL,
                open_time timestamptz NOT NULL,
                close_time timestamptz NOT NULL,
                open numeric NOT NULL,
                high numeric NOT NULL,
                low numeric NOT NULL,
                close numeric NOT NULL,
                volume numeric NOT NULL,
                turnover numeric NOT NULL,
                confirmed boolean NOT NULL DEFAULT false,
                source text NOT NULL DEFAULT 'ws',
                created_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (symbol, interval, open_time)
            );
            CREATE INDEX IF NOT EXISTS idx_market_candles_interval_time
                ON market_candles (interval, open_time DESC);
            CREATE INDEX IF NOT EXISTS idx_market_candles_symbol_interval_time
                ON market_candles (symbol, interval, open_time DESC);

            -- Feature snapshots (one row per confirmed signal evaluation)
            CREATE TABLE IF NOT EXISTS feature_snapshots (
                snapshot_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                created_at timestamptz NOT NULL DEFAULT now(),
                symbol text NOT NULL,
                interval text NOT NULL,
                candle_open_time timestamptz NOT NULL,
                feature_schema_hash text NOT NULL,
                feature_names jsonb NOT NULL,
                feature_values jsonb NOT NULL,
                training_eligible boolean NOT NULL DEFAULT true,
                invalid_reason text,
                invalidated_at timestamptz
            );
            CREATE INDEX IF NOT EXISTS idx_feature_snapshots_symbol_time
                ON feature_snapshots (symbol, created_at DESC);

            -- ML prediction events (shadow scoring even when live decisions disabled)
            CREATE TABLE IF NOT EXISTS prediction_events (
                prediction_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                created_at timestamptz NOT NULL DEFAULT now(),
                symbol text NOT NULL,
                interval text NOT NULL,
                model_version text NOT NULL,
                feature_snapshot_id uuid,
                score double precision NOT NULL,
                strategy_signal text,
                decision text,
                metadata jsonb,
                order_link_id text
            );
            ALTER TABLE prediction_events
                ADD COLUMN IF NOT EXISTS metadata jsonb,
                ADD COLUMN IF NOT EXISTS order_link_id text;
            CREATE INDEX IF NOT EXISTS idx_prediction_events_symbol_time
                ON prediction_events (symbol, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_prediction_events_model_time
                ON prediction_events (model_version, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_prediction_events_model_decision_time
                ON prediction_events (model_version, decision, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_prediction_events_order_link_id
                ON prediction_events (order_link_id) WHERE order_link_id IS NOT NULL;

            -- Outcome labels for training (resolved after horizon_minutes)
            CREATE TABLE IF NOT EXISTS prediction_outcomes (
                prediction_id uuid NOT NULL REFERENCES prediction_events(prediction_id),
                horizon_minutes integer NOT NULL,
                net_return_bps double precision,
                max_favorable_excursion_bps double precision,
                max_adverse_excursion_bps double precision,
                label integer,
                resolved_at timestamptz,
                label_schema_version text DEFAULT 'directional_net_v1',
                PRIMARY KEY (prediction_id, horizon_minutes)
            );
            ALTER TABLE prediction_outcomes
                ADD COLUMN IF NOT EXISTS label_schema_version text DEFAULT 'directional_net_v1';
            CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_label_schema
                ON prediction_outcomes (label_schema_version);
            CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_horizon_schema
                ON prediction_outcomes (horizon_minutes, label_schema_version)
                WHERE label IS NOT NULL;

            -- ML model registry
            CREATE TABLE IF NOT EXISTS model_versions (
                model_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                version text NOT NULL UNIQUE,
                status text NOT NULL DEFAULT 'SHADOW_CHALLENGER',
                training_started_at timestamptz,
                training_finished_at timestamptz,
                training_samples integer,
                feature_schema_hash text,
                artifact bytea,
                metrics jsonb,
                created_at timestamptz NOT NULL DEFAULT now()
            );
            CREATE INDEX IF NOT EXISTS idx_model_versions_status
                ON model_versions (status, created_at DESC);
            ALTER TABLE model_versions
                ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now();

            -- Model promotion audit log. Columns cover both automatic
            -- DB-backed promotion and older manual/pure-eval audit payloads.
            CREATE TABLE IF NOT EXISTS model_promotion_log (
                promotion_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                event_type text NOT NULL,
                decision text,
                challenger_version text,
                champion_version text,
                new_champion_version text,
                from_version text,
                to_version text,
                reasons jsonb NOT NULL DEFAULT '[]'::jsonb,
                metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
                metrics_snapshot jsonb,
                decided_at timestamptz NOT NULL DEFAULT now(),
                created_at timestamptz NOT NULL DEFAULT now()
            );
            ALTER TABLE model_promotion_log
                ADD COLUMN IF NOT EXISTS decision text,
                ADD COLUMN IF NOT EXISTS challenger_version text,
                ADD COLUMN IF NOT EXISTS champion_version text,
                ADD COLUMN IF NOT EXISTS new_champion_version text,
                ADD COLUMN IF NOT EXISTS from_version text,
                ADD COLUMN IF NOT EXISTS to_version text,
                ADD COLUMN IF NOT EXISTS reasons jsonb DEFAULT '[]'::jsonb,
                ADD COLUMN IF NOT EXISTS metrics jsonb DEFAULT '{}'::jsonb,
                ADD COLUMN IF NOT EXISTS metrics_snapshot jsonb,
                ADD COLUMN IF NOT EXISTS decided_at timestamptz DEFAULT now(),
                ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now();
            CREATE INDEX IF NOT EXISTS idx_model_promotion_log_created
                ON model_promotion_log (created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_model_promotion_log_versions
                ON model_promotion_log (from_version, to_version, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_model_promotion_log_decided_at
                ON model_promotion_log (decided_at DESC);

            -- Training run history
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                model_version text,
                mode text NOT NULL DEFAULT 'offline',
                started_at timestamptz NOT NULL DEFAULT now(),
                finished_at timestamptz,
                status text NOT NULL DEFAULT 'RUNNING',
                sample_count integer,
                error text,
                metrics jsonb
            );

            -- P0.5: execution-level fill events (exec_id is the exchange fill ID)
            CREATE TABLE IF NOT EXISTS execution_events (
                exec_id text PRIMARY KEY,
                order_link_id text,
                exchange_order_id text,
                symbol text NOT NULL,
                side text NOT NULL,
                exec_price numeric NOT NULL,
                exec_qty numeric NOT NULL,
                exec_fee numeric,
                exec_value numeric,
                is_maker boolean,
                closed_size numeric,
                proposal_id uuid,
                decision_id uuid,
                created_at timestamptz NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_execution_events_symbol
                ON execution_events (symbol, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_execution_events_order_link
                ON execution_events (order_link_id);
            """
        )
        await _execute_schema_script(
            conn,
            """
                -- Legacy deployments may have these tables without created_at
                -- because CREATE TABLE IF NOT EXISTS does not backfill columns.
                -- Repair them before journal writes start, otherwise individual
                -- signal/risk/order/PnL inserts can fail with:
                -- "column created_at does not exist".
                ALTER TABLE trade_signals
                    ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();
                ALTER TABLE risk_decisions
                    ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();
                ALTER TABLE order_events
                    ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();
                ALTER TABLE closed_pnl
                    ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();
                ALTER TABLE execution_events
                    ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();
                ALTER TABLE market_candles
                    ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();
                ALTER TABLE feature_snapshots
                    ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();
                ALTER TABLE prediction_events
                    ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();
                CREATE TABLE IF NOT EXISTS account_transaction_events (
                    id bigserial PRIMARY KEY,
                    transaction_time timestamptz NOT NULL,
                    symbol text,
                    side text,
                    funding numeric,
                    fee numeric,
                    fee_rate numeric,
                    cash_flow numeric,
                    change numeric,
                    cash_balance numeric,
                    trade_price numeric,
                    trade_id text,
                    order_id text,
                    order_link_id text,
                    created_at timestamptz NOT NULL DEFAULT now()
                );
                CREATE UNIQUE INDEX IF NOT EXISTS account_transaction_events_trade_id_idx
                    ON account_transaction_events (trade_id) WHERE trade_id IS NOT NULL;
                CREATE UNIQUE INDEX IF NOT EXISTS account_transaction_events_hash_idx
                    ON account_transaction_events (transaction_time, symbol, COALESCE(trade_id,''), COALESCE(fee::text,''))
                    WHERE trade_id IS NULL;
                ALTER TABLE account_transaction_events ADD COLUMN IF NOT EXISTS transaction_type text;
                ALTER TABLE account_transaction_events ADD COLUMN IF NOT EXISTS category text;
                -- Candle audit trail: track the last time a row was written or updated
                ALTER TABLE market_candles
                    ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now();
                -- Training eligibility: snapshots created before candle close may have
                -- stale prices; the audit script marks them training_eligible = false.
                ALTER TABLE feature_snapshots
                    ADD COLUMN IF NOT EXISTS training_eligible boolean NOT NULL DEFAULT true;
                ALTER TABLE feature_snapshots
                    ADD COLUMN IF NOT EXISTS invalid_reason text;
                ALTER TABLE feature_snapshots
                    ADD COLUMN IF NOT EXISTS invalidated_at timestamptz;
                ALTER TABLE prediction_outcomes ADD COLUMN IF NOT EXISTS label_schema_version TEXT;
                ALTER TABLE prediction_outcomes ADD COLUMN IF NOT EXISTS gross_return_bps DOUBLE PRECISION;
                ALTER TABLE prediction_outcomes ADD COLUMN IF NOT EXISTS cost_bps DOUBLE PRECISION;
                ALTER TABLE prediction_outcomes ADD COLUMN IF NOT EXISTS label_threshold_bps DOUBLE PRECISION;
                ALTER TABLE prediction_outcomes ADD COLUMN IF NOT EXISTS online_learned_at timestamptz;
                -- Persistent pending-entry resolution state: survives restarts so a
                -- terminal order status seen before a crash never re-blocks the slot.
                CREATE TABLE IF NOT EXISTS order_pending_state (
                    order_link_id text PRIMARY KEY,
                    symbol text,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    resolved_at timestamptz
                );
                CREATE INDEX IF NOT EXISTS idx_order_pending_state_unresolved
                    ON order_pending_state (created_at DESC) WHERE resolved_at IS NULL;
                CREATE INDEX IF NOT EXISTS idx_order_pending_state_symbol_unresolved
                    ON order_pending_state (symbol, created_at DESC) WHERE resolved_at IS NULL;
                -- Hybrid ML mode: mark signals where the model replaced the rule-based decision
                ALTER TABLE trade_signals ADD COLUMN IF NOT EXISTS model_decision jsonb;
                -- Record why a signal was blocked before reaching the execution engine
                ALTER TABLE trade_signals ADD COLUMN IF NOT EXISTS blocked_reason text;
                -- Telegram push-notification subscriptions (survive restarts)
                CREATE TABLE IF NOT EXISTS telegram_subscriptions (
                    chat_id bigint PRIMARY KEY,
                    subscribed_at timestamptz NOT NULL DEFAULT now()
                );
            """,
            )

    async def _ensure_model_registry_indexes_deferred(self) -> None:
        """Repair model registry metadata and champion uniqueness outside startup."""
        if self._pool is None:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("SET statement_timeout = '120s'")
                async with conn.transaction():
                    await conn.execute(
                        """
                        UPDATE model_versions
                        SET feature_schema_hash = metrics->>'source_feature_schema_hash'
                        WHERE metrics ? 'source_feature_schema_hash'
                          AND COALESCE(metrics->>'source_feature_schema_hash', '') <> ''
                          AND feature_schema_hash IS DISTINCT FROM metrics->>'source_feature_schema_hash'
                        """
                    )
                    await conn.execute(
                        """
                        WITH ranked AS (
                            SELECT model_id,
                                   row_number() OVER (
                                       ORDER BY
                                           CASE WHEN COALESCE(
                                               NULLIF(metrics->>'walk_forward_expectancy_bps', ''),
                                               NULLIF(metrics->>'wf_mean_bps', ''),
                                               NULLIF(metrics->>'best_threshold_avg_net_return_bps', '')
                                           )::double precision > 0 THEN 0 ELSE 1 END,
                                           COALESCE(
                                               NULLIF(metrics->>'walk_forward_expectancy_bps', ''),
                                               NULLIF(metrics->>'wf_mean_bps', ''),
                                               NULLIF(metrics->>'best_threshold_avg_net_return_bps', ''),
                                               '-1000000'
                                           )::double precision DESC,
                                           COALESCE(NULLIF(metrics->>'lift_bps', ''), '0')::double precision DESC,
                                           training_finished_at DESC NULLS LAST,
                                           created_at DESC
                                   ) AS rn
                            FROM model_versions
                            WHERE status = 'CHAMPION'
                        )
                        UPDATE model_versions mv
                        SET status = 'ARCHIVED'
                        FROM ranked
                        WHERE mv.model_id = ranked.model_id
                          AND ranked.rn > 1
                        """
                    )
                    await conn.execute(
                        """
                        CREATE UNIQUE INDEX IF NOT EXISTS uq_model_versions_one_champion
                            ON model_versions ((status))
                            WHERE status = 'CHAMPION'
                        """
                    )
        except Exception as exc:
            log.warning("trade_journal.model_registry_indexes_deferred_failed", error=str(exc))

    async def _ensure_feature_snapshot_unique_index_deferred(self) -> None:
        """Run heavy dedupe/index work outside startup-critical schema bootstrap."""
        if self._pool is None:
            return
        try:
            async with self._pool.acquire() as conn:
                await self._ensure_feature_snapshot_unique_index(conn)
        except Exception as exc:
            log.warning("trade_journal.feature_snapshot_unique_index_deferred_failed", error=str(exc))

    async def _ensure_feature_snapshot_unique_index(self, conn: asyncpg.Connection) -> None:
        """Best-effort index bootstrap; never make the journal unavailable."""
        exists = await conn.fetchval("SELECT to_regclass('idx_feature_snapshots_unique_eligible')")
        if exists:
            return
        try:
            async with conn.transaction():
                await conn.execute(
                    """
                    WITH ranked AS (
                        SELECT
                            snapshot_id,
                            ROW_NUMBER() OVER (
                                PARTITION BY
                                    symbol,
                                    interval,
                                    candle_open_time,
                                    feature_schema_hash
                                ORDER BY created_at ASC, snapshot_id ASC
                            ) AS rn
                        FROM feature_snapshots
                        WHERE training_eligible = true
                    )
                    UPDATE feature_snapshots fs
                    SET
                        training_eligible = false,
                        invalid_reason = 'duplicate_snapshot_same_candle',
                        invalidated_at = now()
                    FROM ranked
                    WHERE fs.snapshot_id = ranked.snapshot_id
                      AND ranked.rn > 1
                    """
                )
                await conn.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_snapshots_unique_eligible
                        ON feature_snapshots (symbol, interval, candle_open_time, feature_schema_hash)
                        WHERE training_eligible = true
                    """
                )
        except Exception as exc:
            log.warning(
                "trade_journal.feature_snapshot_unique_index_deferred",
                error=str(exc),
            )

    async def record_signal(
        self,
        proposal: TradeProposal,
        feature_vector: FeatureVector | None,
        regime_context: RegimeContext | None,
        model_decision: dict[str, Any] | None = None,
        blocked_reason: str | None = None,
    ) -> None:
        features = None
        if feature_vector is not None:
            features = {
                "feature_id": str(feature_vector.feature_id),
                "names": feature_vector.feature_names,
                "values": feature_vector.values,
            }
        regime = regime_context.regime.value if regime_context is not None else None
        requested_notional = None
        if proposal.entry_price is not None:
            requested_notional = Decimal(str(proposal.entry_price)) * Decimal(str(proposal.requested_qty))
        await self._execute(
            """
            INSERT INTO trade_signals (
                proposal_id, created_at, strategy_id, symbol, side, confidence,
                entry_price, take_profit, stop_loss, requested_qty,
                requested_notional_usd, regime, rationale, features, model_decision,
                blocked_reason
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb, $15::jsonb, $16)
            ON CONFLICT (proposal_id) DO NOTHING
            """,
            proposal.proposal_id,
            proposal.timestamp,
            proposal.strategy_id,
            proposal.symbol,
            proposal.side.value,
            proposal.confidence,
            proposal.entry_price,
            proposal.take_profit,
            proposal.stop_loss,
            proposal.requested_qty,
            requested_notional,
            regime,
            proposal.rationale,
            json.dumps(features) if features is not None else None,
            json.dumps(model_decision) if model_decision is not None else None,
            blocked_reason,
        )

    async def record_risk_decision(self, symbol: str, decision: RiskDecision) -> None:
        await self._execute(
            """
            INSERT INTO risk_decisions (
                decision_id, proposal_id, created_at, symbol, status, approved_qty,
                approved_notional_usd, reason, triggered_rules, portfolio_heat,
                current_drawdown_pct, open_positions_count
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10, $11, $12)
            ON CONFLICT (decision_id) DO NOTHING
            """,
            decision.decision_id,
            decision.proposal_id,
            decision.timestamp,
            symbol,
            decision.status.value,
            decision.approved_qty,
            decision.approved_notional_usd,
            decision.reason,
            json.dumps(decision.triggered_rules),
            decision.portfolio_heat,
            decision.current_drawdown_pct,
            decision.open_positions_count,
        )

    async def record_order_event(
        self,
        *,
        order_link_id: str,
        proposal_id: Any,
        decision_id: Any,
        symbol: str,
        side: str,
        qty: Decimal,
        status: str,
        exchange_order_id: str | None = None,
        error: str | None = None,
    ) -> None:
        await self._execute(
            """
            INSERT INTO order_events (
                order_link_id, proposal_id, decision_id, created_at, symbol, side,
                qty, status, exchange_order_id, error
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (order_link_id) DO UPDATE SET
                status = EXCLUDED.status,
                exchange_order_id = EXCLUDED.exchange_order_id,
                error = EXCLUDED.error
            """,
            order_link_id,
            proposal_id,
            decision_id,
            datetime.now(tz=UTC),
            symbol,
            side,
            qty,
            status,
            exchange_order_id,
            error,
        )

    async def record_order_event_required(
        self,
        *,
        order_link_id: str,
        proposal_id: Any,
        decision_id: Any,
        symbol: str,
        side: str,
        qty: Decimal,
        status: str,
        exchange_order_id: str | None = None,
        error: str | None = None,
    ) -> None:
        """Durable (fail-closed) order event write. Raises on DB failure.

        Use for CREATED_LOCAL and SUBMITTING states in active modes.
        Caller must not proceed to REST if this raises.
        """
        await self._execute_required(
            """
            INSERT INTO order_events (
                order_link_id, proposal_id, decision_id, created_at, symbol, side,
                qty, status, exchange_order_id, error
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (order_link_id) DO UPDATE SET
                status = EXCLUDED.status,
                exchange_order_id = EXCLUDED.exchange_order_id,
                error = EXCLUDED.error
            """,
            order_link_id,
            proposal_id,
            decision_id,
            datetime.now(tz=UTC),
            symbol,
            side,
            qty,
            status,
            exchange_order_id,
            error,
        )

    async def record_execution_event(
        self,
        *,
        exec_id: str,
        order_link_id: str | None,
        exchange_order_id: str | None,
        symbol: str,
        side: str,
        exec_price: Decimal,
        exec_qty: Decimal,
        exec_fee: Decimal | None = None,
        exec_value: Decimal | None = None,
        is_maker: bool | None = None,
        closed_size: Decimal | None = None,
        proposal_id: Any = None,
        decision_id: Any = None,
    ) -> None:
        """Persist a fill event. Idempotent on exec_id. Never raises."""
        await self._execute(
            """
            INSERT INTO execution_events (
                exec_id, order_link_id, exchange_order_id, symbol, side,
                exec_price, exec_qty, exec_fee, exec_value, is_maker,
                closed_size, proposal_id, decision_id, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            ON CONFLICT (exec_id) DO NOTHING
            """,
            exec_id,
            order_link_id,
            exchange_order_id,
            symbol,
            side,
            exec_price,
            exec_qty,
            exec_fee,
            exec_value,
            is_maker,
            closed_size,
            proposal_id,
            decision_id,
            datetime.now(tz=UTC),
        )

    async def record_transaction_log_entries(self, entries: list[dict[str, Any]]) -> int:
        """Persist Bybit transaction log entries. Returns count inserted."""
        if not self.is_enabled or not entries:
            return 0
        inserted = 0
        for entry in entries:
            try:
                trade_id = entry.get("tradeId") or None
                result = await self._execute(
                    """
                    INSERT INTO account_transaction_events
                        (transaction_time, symbol, side, funding, fee, fee_rate,
                         cash_flow, change, cash_balance, trade_price,
                         trade_id, order_id, order_link_id, transaction_type, category,
                         created_at)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,now())
                    ON CONFLICT (trade_id) DO NOTHING
                    """,
                    _parse_ts(entry.get("transactionTime")),
                    entry.get("symbol"),
                    entry.get("side"),
                    _decimal_or_none(entry.get("funding")),
                    _decimal_or_none(entry.get("fee")),
                    _decimal_or_none(entry.get("feeRate")),
                    _decimal_or_none(entry.get("cashFlow")),
                    _decimal_or_none(entry.get("change")),
                    _decimal_or_none(entry.get("cashBalance")),
                    _decimal_or_none(entry.get("tradePrice")),
                    trade_id,
                    entry.get("orderId"),
                    entry.get("orderLinkId"),
                    entry.get("type"),
                    entry.get("category"),
                )
                inserted += _parse_command_rowcount(result)
            except Exception as exc:
                log.info("trade_journal.transaction_log_insert_failed", error=str(exc))
        return inserted

    async def load_pending_from_db(self) -> list[str]:
        """Return order_link_ids with non-terminal status (CREATED_LOCAL or SUBMITTING).

        Called at startup to restore in-flight entry slots.
        Excludes technical 'unknown:*' IDs.
        """
        rows = await self._fetch(
            """
            SELECT order_link_id
            FROM order_events
            WHERE status IN ('CREATED_LOCAL', 'SUBMITTING')
              AND order_link_id NOT LIKE 'unknown:%'
            ORDER BY created_at ASC
            """
        )
        ids = [str(row["order_link_id"]) for row in rows]
        if ids:
            log.info("trade_journal.pending_restored", count=len(ids), ids=ids)
        return ids

    async def get_daily_net_results(
        self,
        day_utc: date | None = None,
    ) -> dict[str, Any]:
        """Compute today's net PnL from closed_pnl + account_transaction_events.

        Sign convention: profit positive, loss negative; fees/funding are returned
        as costs (negative = paid). net_pnl_usd equals the closedPnl sum because
        Bybit's closedPnl already includes trading fees and funding.
        Returns an empty dict with zeros when DB is not enabled.
        """

        zero: dict[str, Any] = {
            "closed_trade_count": 0,
            "gross_closed_pnl_usd": 0.0,
            "total_fees_usd": 0.0,
            "total_funding_usd": 0.0,
            "net_cash_flow_usd": 0.0,
            "net_pnl_usd": 0.0,
            "maker_fill_count": 0,
            "taker_fill_count": 0,
            "maker_fill_pct": 0.0,
            "taker_fill_pct": 0.0,
            "estimated_slippage_usd": 0.0,
            "transaction_event_count": 0,
            "latest_transaction_at": None,
        }
        if not self.is_enabled:
            return zero

        if day_utc is None:
            day_utc = datetime.now(tz=UTC).date()  # true UTC day, independent of container TZ

        day_start = datetime(day_utc.year, day_utc.month, day_utc.day, tzinfo=UTC)
        day_end = day_start + timedelta(days=1)

        try:
            # Closed PnL from closed_pnl table
            pnl_rows = await self._fetch(
                """
                SELECT count(*) AS trade_count,
                       COALESCE(sum(closed_pnl), 0) AS gross_pnl
                FROM closed_pnl
                WHERE created_at >= $1 AND created_at < $2
                """,
                day_start,
                day_end,
            )
            trade_count = int(pnl_rows[0]["trade_count"]) if pnl_rows else 0
            gross_pnl = float(pnl_rows[0]["gross_pnl"]) if pnl_rows else 0.0

            # Fees, funding, cash flow from account_transaction_events
            tx_rows = await self._fetch(
                """
                SELECT
                    count(*)                                AS tx_count,
                    COALESCE(sum(fee), 0)                   AS total_fees,
                    COALESCE(sum(funding), 0)               AS total_funding,
                    COALESCE(sum(COALESCE(cash_flow, change, 0)), 0) AS net_cash_flow,
                    max(transaction_time)                   AS latest_tx
                FROM account_transaction_events
                WHERE transaction_time >= $1 AND transaction_time < $2
                """,
                day_start,
                day_end,
            )
            tx_count = int(tx_rows[0]["tx_count"]) if tx_rows else 0
            total_fees = float(tx_rows[0]["total_fees"]) if tx_rows else 0.0
            total_funding = float(tx_rows[0]["total_funding"]) if tx_rows else 0.0
            net_cash_flow = float(tx_rows[0]["net_cash_flow"]) if tx_rows else 0.0
            latest_tx = tx_rows[0]["latest_tx"] if tx_rows else None

            # Maker/taker from execution_events (same day)
            exec_rows = await self._fetch(
                """
                SELECT
                    count(*) FILTER (WHERE is_maker = true)  AS maker_count,
                    count(*) FILTER (WHERE is_maker = false) AS taker_count
                FROM execution_events
                WHERE created_at >= $1 AND created_at < $2
                """,
                day_start,
                day_end,
            )
            maker_count = int(exec_rows[0]["maker_count"]) if exec_rows else 0
            taker_count = int(exec_rows[0]["taker_count"]) if exec_rows else 0
            total_fills = maker_count + taker_count
            maker_pct = (maker_count / total_fills * 100.0) if total_fills > 0 else 0.0
            taker_pct = (taker_count / total_fills * 100.0) if total_fills > 0 else 0.0

            # Bybit closedPnl ALREADY nets out open/close fees and funding
            # (help center: Closed P&L = position P&L - fees - funding), so it
            # IS the realized net result — never add costs to it again.
            # Transaction-log fee/funding are positive-when-paid; normalise to
            # the "cost is negative" convention for display.
            net_pnl = gross_pnl
            total_fees = -total_fees
            total_funding = -total_funding

            return {
                "closed_trade_count": trade_count,
                "gross_closed_pnl_usd": round(gross_pnl, 6),
                "total_fees_usd": round(total_fees, 6),
                "total_funding_usd": round(total_funding, 6),
                "net_cash_flow_usd": round(net_cash_flow, 6),
                "net_pnl_usd": round(net_pnl, 6),
                "maker_fill_count": maker_count,
                "taker_fill_count": taker_count,
                "maker_fill_pct": round(maker_pct, 1),
                "taker_fill_pct": round(taker_pct, 1),
                "estimated_slippage_usd": 0.0,
                "transaction_event_count": tx_count,
                "latest_transaction_at": latest_tx,
            }
        except Exception as exc:
            log.warning("trade_journal.get_daily_net_results_failed", error=str(exc))
            return zero

    async def get_pending_order_events(self) -> list[dict[str, Any]]:
        """Return detailed pending order records for reconciliation.

        Read-only. Used by ExecutionEngine.reconcile_restored_pending_entries().
        """
        rows = await self._fetch(
            """
            SELECT order_link_id, proposal_id, decision_id, symbol, side, qty,
                   status, exchange_order_id, error, created_at
            FROM order_events
            WHERE status IN ('CREATED_LOCAL', 'SUBMITTING')
            ORDER BY created_at ASC
            """
        )
        return [dict(r) for r in rows]

    async def mark_order_event_stale(self, order_link_id: str, reason: str) -> None:
        """Mark a pending order event as FAILED_STALE (non-destructive terminal update).

        Only transitions from CREATED_LOCAL or SUBMITTING; never touches terminal states.
        """
        await self._execute(
            """
            UPDATE order_events
            SET status = 'FAILED_STALE', error = $2
            WHERE order_link_id = $1
              AND status IN ('CREATED_LOCAL', 'SUBMITTING')
            """,
            order_link_id,
            reason,
        )

    async def mark_durable_order_stale(self, order_link_id: str, reason: str) -> None:
        """Mark durable_order_state as FAILED for a stale pending order (non-destructive)."""
        await self._execute(
            """
            UPDATE durable_order_state
            SET state = 'FAILED', last_error = $2, updated_at = now()
            WHERE order_link_id = $1
              AND state NOT IN ('FILLED','CANCELLED','REJECTED','EXPIRED','SHADOW','FAILED')
            """,
            order_link_id,
            reason,
        )

    async def upsert_durable_order_state(
        self,
        *,
        order_link_id: str,
        symbol: str,
        side: str,
        qty: Decimal,
        state: str,
        proposal_id: Any = None,
        decision_id: Any = None,
        exchange_order_id: str | None = None,
        payload_hash: str | None = None,
        retry_count: int = 0,
        last_error: str | None = None,
    ) -> None:
        """Upsert durable order state before/after REST submission."""
        await self._execute(
            """
            INSERT INTO durable_order_state (
                order_link_id, proposal_id, decision_id, symbol, side, qty,
                state, exchange_order_id, payload_hash, retry_count, last_error, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, now())
            ON CONFLICT (order_link_id) DO UPDATE SET
                state = EXCLUDED.state,
                exchange_order_id = COALESCE(EXCLUDED.exchange_order_id, durable_order_state.exchange_order_id),
                retry_count = EXCLUDED.retry_count,
                last_error = EXCLUDED.last_error,
                updated_at = now()
            WHERE durable_order_state.state NOT IN ('FILLED','CANCELLED','REJECTED','EXPIRED','SHADOW','FAILED')
            """,
            order_link_id,
            proposal_id,
            decision_id,
            symbol,
            side,
            qty,
            state,
            exchange_order_id,
            payload_hash,
            retry_count,
            last_error,
        )

    async def get_pending_durable_orders(self) -> list[dict[str, Any]]:
        """Return all non-terminal durable order states (for restart recovery)."""
        rows = await self._fetch(
            """
            SELECT order_link_id, proposal_id, decision_id, symbol, side, qty,
                   state, exchange_order_id, retry_count, last_error, created_at, updated_at
            FROM durable_order_state
            WHERE state NOT IN ('FILLED','CANCELLED','REJECTED','EXPIRED','SHADOW','FAILED')
              AND updated_at > now() - interval '24 hours'
              AND order_link_id NOT LIKE 'unknown:%'
            ORDER BY created_at DESC
            """
        )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Persistent pending-entry resolution state (order_pending_state)
    # ------------------------------------------------------------------

    async def record_order_pending(self, order_link_id: str, symbol: str) -> None:
        """Register a pending entry order so its resolution survives restarts."""
        if not self.is_enabled or not order_link_id:
            return
        await self._execute(
            """
            INSERT INTO order_pending_state (order_link_id, symbol)
            VALUES ($1, $2)
            ON CONFLICT (order_link_id) DO NOTHING
            """,
            order_link_id,
            symbol,
        )

    async def mark_order_resolved(self, order_link_id: str, symbol: str = "") -> None:
        """Persist that a pending entry order reached a terminal state.

        Upserts so resolution is recorded even if record_order_pending was missed.
        """
        if not self.is_enabled or not order_link_id:
            return
        await self._execute(
            """
            INSERT INTO order_pending_state (order_link_id, symbol, resolved_at)
            VALUES ($1, $2, now())
            ON CONFLICT (order_link_id) DO UPDATE
                SET resolved_at = COALESCE(order_pending_state.resolved_at, now())
            """,
            order_link_id,
            symbol or None,
        )

    async def is_order_resolved(self, order_link_id: str) -> bool:
        """True if the order has a persisted terminal resolution."""
        if not self.is_enabled or not order_link_id:
            return False
        rows = await self._fetch(
            "SELECT 1 FROM order_pending_state WHERE order_link_id = $1 AND resolved_at IS NOT NULL",
            order_link_id,
        )
        return bool(rows)

    async def get_unresolved_order_link_ids(self) -> list[str]:
        """Return order_link_ids registered as pending but never resolved (last 24h)."""
        rows = await self._fetch(
            """
            SELECT order_link_id FROM order_pending_state
            WHERE resolved_at IS NULL AND created_at > now() - interval '24 hours'
            ORDER BY created_at DESC
            """
        )
        return [str(r["order_link_id"]) for r in rows]

    # ------------------------------------------------------------------
    # Telegram subscriptions
    # ------------------------------------------------------------------

    async def add_telegram_subscription(self, chat_id: int) -> None:
        """Persist a Telegram chat subscription (idempotent)."""
        if not self.is_enabled:
            return
        await self._execute(
            "INSERT INTO telegram_subscriptions (chat_id) VALUES ($1) ON CONFLICT (chat_id) DO NOTHING",
            int(chat_id),
        )

    async def remove_telegram_subscription(self, chat_id: int) -> None:
        if not self.is_enabled:
            return
        await self._execute(
            "DELETE FROM telegram_subscriptions WHERE chat_id = $1",
            int(chat_id),
        )

    async def get_telegram_subscriptions(self) -> list[int]:
        rows = await self._fetch("SELECT chat_id FROM telegram_subscriptions ORDER BY subscribed_at")
        return [int(r["chat_id"]) for r in rows]

    # ------------------------------------------------------------------
    # Telegram /trades and /healthcheck data
    # ------------------------------------------------------------------

    async def get_recent_closed_trades(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the most recent closed trades for the /trades command."""
        rows = await self._fetch(
            """
            SELECT created_at, symbol, side, qty, avg_entry_price, avg_exit_price, closed_pnl
            FROM closed_pnl
            ORDER BY created_at DESC
            LIMIT $1
            """,
            int(limit),
        )
        result: list[dict[str, Any]] = []
        for r in rows:
            entry = float(r["avg_entry_price"] or 0)
            exit_ = float(r["avg_exit_price"] or 0)
            # Bybit closed-pnl `side` is the side of the CLOSING order: a long
            # position is closed by Sell, a short by Buy.
            closing_side = str(r["side"] or "").upper()
            position = "LONG" if closing_side == "SELL" else ("SHORT" if closing_side == "BUY" else "?")
            net_bps: float | None = None
            qty = float(r["qty"] or 0)
            closed_pnl = float(r["closed_pnl"] or 0)
            # Use Bybit's closedPnl (already net of fees+funding) normalised by position value.
            if entry > 0 and qty > 0:
                net_bps = closed_pnl / (entry * qty) * 10_000
            result.append(
                {
                    "created_at": r["created_at"],
                    "symbol": r["symbol"],
                    "side": position,
                    "qty": qty,
                    "entry": entry,
                    "exit": exit_,
                    "pnl_usdt": closed_pnl,
                    "net_bps": net_bps,
                }
            )
        return result

    async def get_today_avg_net_bps(self) -> float | None:
        """Average resolved net return (bps) for today's baseline signals."""
        rows = await self._fetch(
            """
            SELECT avg(po.net_return_bps) AS avg_bps
            FROM prediction_outcomes po
            JOIN prediction_events pe ON pe.prediction_id = po.prediction_id
            WHERE po.net_return_bps IS NOT NULL
              AND po.label_schema_version = $1
              AND pe.model_version = 'RULE_BASELINE_V1'
              AND COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'
              AND po.resolved_at >= date_trunc('day', now())
            """,
            LABEL_SCHEMA_VERSION,
        )
        if rows and rows[0]["avg_bps"] is not None:
            return float(rows[0]["avg_bps"])
        return None

    async def get_bucket_stats(
        self,
        *,
        horizon_minutes: int = 15,
        lookback_days: int = 30,
    ) -> dict[tuple[str, str, int], tuple[float, int]]:
        """Per-(regime, volatility, UTC hour) expectancy of our own baseline signals.

        Aggregates resolved outcomes of RULE_BASELINE_V1 prediction events whose
        metadata carries the regime context recorded at signal time. Returns
        {(regime, volatility, hour): (avg_return_bps, count)}. Events without
        regime metadata are grouped under "UNKNOWN".
        """
        rows = await self._fetch(
            """
            SELECT
                COALESCE(pe.metadata->>'regime', 'UNKNOWN') AS regime,
                COALESCE(pe.metadata->>'volatility', 'UNKNOWN') AS volatility,
                extract(hour FROM pe.created_at AT TIME ZONE 'UTC')::int AS hour,
                avg(po.net_return_bps) AS avg_bps,
                count(*) AS cnt
            FROM prediction_outcomes po
            JOIN prediction_events pe ON pe.prediction_id = po.prediction_id
            WHERE po.net_return_bps IS NOT NULL
              AND po.horizon_minutes = $1
              AND po.label_schema_version = $3
              AND pe.model_version = 'RULE_BASELINE_V1'
              AND COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'
              AND pe.created_at > now() - ($2::text || ' days')::interval
            GROUP BY 1, 2, 3
            """,
            horizon_minutes,
            str(lookback_days),
            LABEL_SCHEMA_VERSION,
        )
        return {
            (str(r["regime"]), str(r["volatility"]), int(r["hour"])): (
                float(r["avg_bps"]),
                int(r["cnt"]),
            )
            for r in rows
        }

    async def get_symbol_side_stats(
        self,
        *,
        horizon_minutes: int = 15,
        lookback_days: int = 30,
    ) -> dict[tuple[str, str], tuple[float, int]]:
        """Per-(symbol, side) expectancy of resolved baseline signals."""

        rows = await self._fetch(
            """
            SELECT
                pe.symbol,
                pe.strategy_signal AS side,
                avg(po.net_return_bps) AS avg_bps,
                count(*) AS cnt
            FROM prediction_outcomes po
            JOIN prediction_events pe ON pe.prediction_id = po.prediction_id
            WHERE po.net_return_bps IS NOT NULL
              AND po.horizon_minutes = $1
              AND po.label_schema_version = $3
              AND pe.model_version = 'RULE_BASELINE_V1'
              AND pe.strategy_signal IN ('Buy', 'Sell')
              AND COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'
              AND pe.created_at > now() - ($2::text || ' days')::interval
            GROUP BY 1, 2
            """,
            horizon_minutes,
            str(lookback_days),
            LABEL_SCHEMA_VERSION,
        )
        return {
            (str(r["symbol"]), str(r["side"])): (
                float(r["avg_bps"]),
                int(r["cnt"]),
            )
            for r in rows
        }

    async def get_hour_stats(
        self,
        *,
        horizon_minutes: int = 15,
        lookback_days: int = 30,
    ) -> dict[int, tuple[float, int]]:
        """Per-UTC-hour expectancy of resolved baseline strategy signals."""

        rows = await self._fetch(
            """
            SELECT
                extract(hour FROM pe.created_at AT TIME ZONE 'UTC')::int AS hour_utc,
                avg(po.net_return_bps) AS avg_bps,
                count(*) AS cnt
            FROM prediction_outcomes po
            JOIN prediction_events pe ON pe.prediction_id = po.prediction_id
            WHERE po.net_return_bps IS NOT NULL
              AND po.horizon_minutes = $1
              AND po.label_schema_version = $3
              AND pe.model_version = 'RULE_BASELINE_V1'
              AND COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'
              AND pe.created_at > now() - ($2::text || ' days')::interval
            GROUP BY 1
            """,
            horizon_minutes,
            str(lookback_days),
            LABEL_SCHEMA_VERSION,
        )
        return {int(row["hour_utc"]): (float(row["avg_bps"]), int(row["cnt"])) for row in rows}

    async def get_strategy_stats(
        self,
        *,
        horizon_minutes: int = 15,
        lookback_days: int = 30,
    ) -> dict[str, tuple[float, int]]:
        """Per-strategy net expectancy for resolved baseline signals."""

        rows = await self._fetch(
            """
            SELECT
                COALESCE(pe.metadata->>'strategy_id', 'UNKNOWN') AS strategy_id,
                avg(po.net_return_bps) AS avg_bps,
                count(*) AS cnt
            FROM prediction_outcomes po
            JOIN prediction_events pe ON pe.prediction_id = po.prediction_id
            WHERE po.net_return_bps IS NOT NULL
              AND po.horizon_minutes = $1
              AND po.label_schema_version = $3
              AND pe.model_version = 'RULE_BASELINE_V1'
              AND COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'
              AND pe.created_at > now() - ($2::text || ' days')::interval
            GROUP BY 1
            """,
            horizon_minutes,
            str(lookback_days),
            LABEL_SCHEMA_VERSION,
        )
        return {str(row["strategy_id"]): (float(row["avg_bps"]), int(row["cnt"])) for row in rows}

    async def get_shadow_probe_symbol_side_stats(
        self,
        *,
        horizon_minutes: int = 15,
        lookback_days: int = 7,
        strategy_id: str = "shadow_probe_hv_v2",
    ) -> dict[tuple[str, str], tuple[float, int]]:
        """Per-(symbol, side) expectancy for one probe research version."""

        rows = await self._fetch(
            """
            SELECT
                pe.symbol,
                pe.strategy_signal AS side,
                avg(po.net_return_bps) AS avg_bps,
                count(*) AS cnt
            FROM prediction_outcomes po
            JOIN prediction_events pe ON pe.prediction_id = po.prediction_id
            WHERE po.net_return_bps IS NOT NULL
              AND po.horizon_minutes = $1
              AND po.label_schema_version = $3
              AND pe.model_version = 'RULE_BASELINE_V1'
              AND pe.decision = 'SHADOW_BASELINE'
              AND pe.strategy_signal IN ('Buy', 'Sell')
              AND COALESCE(pe.metadata->>'strategy_id', '') = $4
              AND pe.created_at > now() - ($2::text || ' days')::interval
            GROUP BY 1, 2
            """,
            horizon_minutes,
            str(lookback_days),
            LABEL_SCHEMA_VERSION,
            strategy_id,
        )
        return {
            (str(r["symbol"]), str(r["side"])): (
                float(r["avg_bps"]),
                int(r["cnt"]),
            )
            for r in rows
        }

    async def get_shadow_probe_symbol_stats(
        self,
        *,
        horizon_minutes: int = 15,
        lookback_days: int = 7,
        strategy_id: str = "shadow_probe_hv_v2",
    ) -> dict[str, tuple[float, int]]:
        """Per-symbol aggregate expectancy for one probe research version."""

        rows = await self._fetch(
            """
            SELECT
                pe.symbol,
                avg(po.net_return_bps) AS avg_bps,
                count(*) AS cnt
            FROM prediction_outcomes po
            JOIN prediction_events pe ON pe.prediction_id = po.prediction_id
            WHERE po.net_return_bps IS NOT NULL
              AND po.horizon_minutes = $1
              AND po.label_schema_version = $3
              AND pe.model_version = 'RULE_BASELINE_V1'
              AND pe.decision = 'SHADOW_BASELINE'
              AND COALESCE(pe.metadata->>'strategy_id', '') = $4
              AND pe.created_at > now() - ($2::text || ' days')::interval
            GROUP BY 1
            """,
            horizon_minutes,
            str(lookback_days),
            LABEL_SCHEMA_VERSION,
            strategy_id,
        )
        return {
            str(r["symbol"]): (
                float(r["avg_bps"]),
                int(r["cnt"]),
            )
            for r in rows
        }

    async def find_order_link_id_by_exchange_order_id(
        self,
        exchange_order_id: str,
    ) -> str | None:
        """Reverse lookup: find order_link_id by exchange_order_id.

        Searches durable_order_state first (authoritative), then order_events as fallback.
        Excludes technical 'unknown:*' IDs to avoid false-positive correlation.
        Returns None if not found or if only unknown IDs exist.
        """
        if not self.is_enabled or not exchange_order_id:
            return None

        try:
            # Try durable_order_state first (authoritative for in-flight orders)
            rows = await self._fetch(
                """
                SELECT order_link_id
                FROM durable_order_state
                WHERE exchange_order_id = $1
                  AND order_link_id NOT LIKE 'unknown:%'
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                exchange_order_id,
            )
            if rows:
                candidate = str(rows[0]["order_link_id"]).strip()
                if candidate and not candidate.startswith("unknown:"):
                    return candidate

            # Fallback to order_events
            rows = await self._fetch(
                """
                SELECT order_link_id
                FROM order_events
                WHERE exchange_order_id = $1
                  AND order_link_id NOT LIKE 'unknown:%'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                exchange_order_id,
            )
            if rows:
                candidate = str(rows[0]["order_link_id"]).strip()
                if candidate and not candidate.startswith("unknown:"):
                    return candidate

            return None
        except Exception as exc:
            self._last_read_error_at = datetime.now(tz=UTC)
            self._last_read_error = str(exc)
            log.debug(
                "trade_journal.reverse_lookup_failed",
                exchange_order_id=exchange_order_id,
                error=str(exc),
            )
            return None

    async def upsert_market_candle(
        self,
        *,
        symbol: str,
        interval: str,
        open_time: datetime,
        close_time: datetime,
        open: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        volume: Decimal,
        turnover: Decimal,
        confirmed: bool,
        source: str = "ws",
    ) -> None:
        """UPSERT a single confirmed candle (no duplicates)."""
        await self._execute(
            """
            INSERT INTO market_candles (
                symbol, interval, open_time, close_time, open, high, low, close,
                volume, turnover, confirmed, source
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (symbol, interval, open_time) DO UPDATE SET
                close_time  = EXCLUDED.close_time,
                open        = EXCLUDED.open,
                high        = EXCLUDED.high,
                low         = EXCLUDED.low,
                close       = EXCLUDED.close,
                volume      = EXCLUDED.volume,
                turnover    = EXCLUDED.turnover,
                confirmed   = EXCLUDED.confirmed,
                source      = EXCLUDED.source,
                updated_at  = now()
            WHERE NOT market_candles.confirmed OR EXCLUDED.confirmed
            """,
            symbol,
            interval,
            open_time,
            close_time,
            open,
            high,
            low,
            close,
            volume,
            turnover,
            confirmed,
            source,
        )

    async def get_candle_counts(self) -> dict[str, int]:
        """Return {interval: confirmed count} for market_candles diagnostics."""
        rows = await self._fetch(
            """
            SELECT interval, count(*) AS cnt
            FROM market_candles
            WHERE confirmed = true
            GROUP BY interval
            """
        )
        return {str(r["interval"]): int(r["cnt"]) for r in rows}

    async def get_candle_readiness_counts(self) -> dict[str, int]:
        """Return capped candle counts used by readiness checks.

        Exact COUNT(*) over a large candle table is an operator nicety, not a
        trading prerequisite. These bounded probes answer the important
        readiness question without making Telegram diagnostics contend with
        ingestion and training queries.
        """
        targets = {"1": 1000, "5": 200, "15": 200, "60": 100}
        intervals = list(targets.keys())
        target_vals = [targets[i] for i in intervals]

        async def _count_one(interval: str, target: int) -> int:
            rows = await self._fetch(
                """
                SELECT count(*) AS cnt
                FROM (
                    SELECT 1
                    FROM market_candles
                    WHERE interval = $1
                      AND confirmed = true
                    LIMIT $2
                ) capped
                """,
                interval,
                target,
            )
            return int(rows[0]["cnt"]) if rows else 0

        results = await asyncio.gather(*(_count_one(iv, tgt) for iv, tgt in zip(intervals, target_vals, strict=True)))
        return dict(zip(intervals, results, strict=True))

    async def get_candle_counts_per_symbol(self) -> dict[tuple[str, str], int]:
        """Return {(symbol, interval): confirmed count} for backfill gap detection."""
        rows = await self._fetch(
            """
            SELECT symbol, interval, count(*) AS cnt
            FROM market_candles
            WHERE confirmed = true
            GROUP BY symbol, interval
            """
        )
        return {(str(r["symbol"]), str(r["interval"])): int(r["cnt"]) for r in rows}

    async def get_recent_market_candles(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return recent confirmed candles oldest-first for startup CandleStore seeding."""
        rows = await self._fetch(
            """
            SELECT open_time, open, high, low, close, volume
            FROM (
                SELECT open_time, open, high, low, close, volume
                FROM market_candles
                WHERE symbol = $1
                  AND interval = $2
                  AND confirmed = true
                ORDER BY open_time DESC
                LIMIT $3
            ) recent
            ORDER BY open_time ASC
            """,
            symbol.upper(),
            str(interval),
            int(limit),
        )
        return [dict(row) for row in rows]

    async def get_latest_candle_time(self, interval: str = "1") -> datetime | None:
        """Return the most recent confirmed open_time for the given interval."""
        rows = await self._fetch(
            """
            SELECT MAX(open_time) AS ts
            FROM market_candles
            WHERE interval = $1
              AND confirmed = true
            """,
            interval,
        )
        if rows and rows[0]["ts"]:
            return cast(datetime, rows[0]["ts"])
        return None

    async def get_feature_snapshot_readiness_count(self, limit: int = 1000) -> int:
        rows = await self._fetch(
            """
            SELECT count(*) AS cnt
            FROM (
                SELECT 1
                FROM feature_snapshots
                LIMIT $1
            ) capped
            """,
            int(limit),
        )
        return int(rows[0]["cnt"]) if rows else 0

    async def get_prediction_outcome_readiness_count(self, limit: int = 1000) -> int:
        rows = await self._fetch(
            """
            SELECT count(*) AS cnt
            FROM (
                SELECT 1
                FROM prediction_outcomes
                WHERE label IS NOT NULL
                LIMIT $1
            ) capped
            """,
            int(limit),
        )
        return int(rows[0]["cnt"]) if rows else 0

    async def get_labelled_15m_readiness_count(self, limit: int = 1000) -> int:
        rows = await self._fetch(
            """
            SELECT count(*) AS cnt
            FROM (
                SELECT 1
                FROM prediction_outcomes
                WHERE horizon_minutes = 15
                  AND label IS NOT NULL
                LIMIT $1
            ) capped
            """,
            int(limit),
        )
        return int(rows[0]["cnt"]) if rows else 0

    async def apply_candle_retention(self, retention_days: dict[str, int] | None = None) -> int:
        """Delete old candles according to retention policy."""
        from trader.storage.retention import RetentionSettings, run_data_retention

        days = retention_days or {"1": 30, "5": 180, "15": 365, "60": 730}
        report = await run_data_retention(self, RetentionSettings(candle_retention_days=days))
        return report.candles_deleted

    async def get_storage_stats(self) -> dict[str, Any]:
        from trader.storage.retention import get_storage_stats

        return await get_storage_stats(self)

    async def get_pnl_attribution(self, *, days: int = 7) -> list[dict[str, Any]]:
        from trader.storage.retention import get_pnl_attribution

        return await get_pnl_attribution(self, days=days)

    async def run_data_retention_policy(self, settings: Any) -> dict[str, Any]:
        from trader.storage.retention import RetentionSettings, run_data_retention

        cfg = RetentionSettings(
            candle_retention_days={
                "1": int(settings.CANDLE_RETENTION_DAYS_1M),
                "5": int(settings.CANDLE_RETENTION_DAYS_5M),
                "15": int(settings.CANDLE_RETENTION_DAYS_15M),
                "60": int(settings.CANDLE_RETENTION_DAYS_60M),
            },
            feature_snapshot_retention_days=int(settings.FEATURE_SNAPSHOT_RETENTION_DAYS),
            feature_snapshot_invalid_retention_days=int(settings.FEATURE_SNAPSHOT_INVALID_RETENTION_DAYS),
            feature_snapshot_orphan_retention_days=int(settings.FEATURE_SNAPSHOT_ORPHAN_RETENTION_DAYS),
            prediction_event_orphan_retention_days=int(settings.PREDICTION_EVENT_ORPHAN_RETENTION_DAYS),
            prediction_outcome_retention_days=int(settings.PREDICTION_OUTCOME_RETENTION_DAYS),
            shadow_signal_retention_days=int(settings.SHADOW_SIGNAL_RETENTION_DAYS),
            resolved_snapshot_export_before_delete_days=int(settings.RESOLVED_SNAPSHOT_EXPORT_BEFORE_DELETE_DAYS),
            export_enabled=bool(settings.DATA_RETENTION_EXPORT_ENABLED),
            export_dir=str(settings.DATA_RETENTION_EXPORT_DIR),
        )
        report = await run_data_retention(self, cfg)
        return report.to_dict()

    async def fetch_feature_drift_samples(
        self,
        *,
        baseline_days: int = 14,
        current_days: int = 3,
        limit: int = 500,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Return baseline vs recent feature vectors for PSI drift checks."""
        baseline_rows = await self._fetch(
            """
            SELECT feature_values
            FROM feature_snapshots
            WHERE training_eligible = true
              AND created_at < now() - ($1::text || ' days')::interval
              AND created_at >= now() - ($2::text || ' days')::interval
            ORDER BY created_at DESC
            LIMIT $3
            """,
            str(current_days),
            str(baseline_days + current_days),
            int(limit),
        )
        current_rows = await self._fetch(
            """
            SELECT feature_values
            FROM feature_snapshots
            WHERE training_eligible = true
              AND created_at >= now() - ($1::text || ' days')::interval
            ORDER BY created_at DESC
            LIMIT $2
            """,
            str(current_days),
            int(limit),
        )

        def _parse(rows: list[Any]) -> list[list[float]]:
            out: list[list[float]] = []
            for row in rows:
                raw = row["feature_values"]
                if isinstance(raw, str):
                    raw = json.loads(raw)
                if isinstance(raw, list):
                    out.append([float(x) for x in raw])
            return out

        return _parse(baseline_rows), _parse(current_rows)

    async def fetch_online_learning_batch(
        self,
        *,
        limit: int = 50,
        challenger_version: str | None = None,
        label_schema_version: str | None = None,
        sources: tuple[str, ...] = ("shadow_challenger",),
    ) -> list[dict[str, Any]]:
        """Resolved shadow-challenger outcomes with feature vectors for partial_fit."""

        schema = label_schema_version
        if schema is None:
            settings = _optional_settings()
            use_tpsl = bool(settings.MODEL_LABEL_USE_TPSL_EXIT) if settings is not None else True
            schema = active_label_schema_version(use_tpsl_exit=use_tpsl)
        source_list = list(sources) if sources else ["shadow_challenger"]
        rows = await self._fetch(
            """
            SELECT po.prediction_id, po.label, fs.feature_values
            FROM prediction_outcomes po
            JOIN prediction_events pe ON pe.prediction_id = po.prediction_id
            JOIN feature_snapshots fs ON fs.snapshot_id = pe.feature_snapshot_id
            WHERE po.label IS NOT NULL
              AND po.online_learned_at IS NULL
              AND po.label_schema_version = $2
              AND ($3::text IS NULL OR pe.model_version = $3)
              AND COALESCE(pe.metadata->>'source', '') = ANY($4::text[])
            ORDER BY po.resolved_at DESC NULLS LAST
            LIMIT $1
            """,
            int(limit),
            schema,
            challenger_version,
            source_list,
        )
        batch: list[dict[str, Any]] = []
        for row in rows:
            raw = row["feature_values"]
            if isinstance(raw, str):
                raw = json.loads(raw)
            if not isinstance(raw, list):
                continue
            batch.append(
                {
                    "prediction_id": str(row["prediction_id"]),
                    "label": int(row["label"]),
                    "features": [float(x) for x in raw],
                }
            )
        return batch

    async def mark_online_learning_applied(self, prediction_ids: list[str]) -> None:
        if not prediction_ids:
            return
        await self._execute(
            """
            UPDATE prediction_outcomes
            SET online_learned_at = now()
            WHERE prediction_id = ANY($1::uuid[])
            """,
            prediction_ids,
        )

    async def record_feature_snapshot(
        self,
        *,
        symbol: str,
        interval: str,
        candle_open_time: datetime,
        feature_schema_hash: str,
        feature_names: list[str],
        feature_values: list[float],
    ) -> str:
        """Write feature vector snapshot; return snapshot_id."""
        args = (
            symbol,
            interval,
            candle_open_time,
            feature_schema_hash,
            json.dumps(feature_names),
            json.dumps(feature_values),
        )
        rows = await self._fetch(
            """
            INSERT INTO feature_snapshots (
                symbol, interval, candle_open_time,
                feature_schema_hash, feature_names, feature_values
            )
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb)
            ON CONFLICT (symbol, interval, candle_open_time, feature_schema_hash)
            WHERE training_eligible = true
            DO UPDATE SET
                feature_names = EXCLUDED.feature_names,
                feature_values = EXCLUDED.feature_values
            RETURNING snapshot_id
            """,
            *args,
        )
        if not rows and self.is_enabled:
            rows = await self._fetch(
                """
                INSERT INTO feature_snapshots (
                    symbol, interval, candle_open_time,
                    feature_schema_hash, feature_names, feature_values
                )
                VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb)
                ON CONFLICT DO NOTHING
                RETURNING snapshot_id
                """,
                *args,
            )
        return str(rows[0]["snapshot_id"]) if rows else ""

    async def record_prediction_event(
        self,
        *,
        symbol: str,
        interval: str,
        model_version: str,
        score: float,
        strategy_signal: str | None = None,
        decision: str | None = None,
        feature_snapshot_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        order_link_id: str | None = None,
    ) -> str:
        """Write a shadow-mode prediction event; return prediction_id."""
        rows = await self._fetch(
            """
            INSERT INTO prediction_events (
                symbol, interval, model_version, feature_snapshot_id,
                score, strategy_signal, decision, metadata, order_link_id
            )
            VALUES ($1, $2, $3, $4::uuid, $5, $6, $7, $8::jsonb, $9)
            RETURNING prediction_id
            """,
            symbol,
            interval,
            model_version,
            feature_snapshot_id if feature_snapshot_id else None,
            score,
            strategy_signal,
            decision,
            json.dumps(metadata or {}),
            order_link_id,
        )
        return str(rows[0]["prediction_id"]) if rows else ""

    async def link_prediction_to_order(self, prediction_id: str, order_link_id: str) -> None:
        """Link an existing prediction_event to its executed order."""
        await self._execute(
            """
            UPDATE prediction_events
            SET order_link_id = $1
            WHERE prediction_id = $2::uuid
              AND order_link_id IS NULL
            """,
            order_link_id,
            prediction_id,
        )

    async def resolve_prediction_outcomes(
        self,
        *,
        prediction_id: str,
        horizon_minutes: int,
        net_return_bps: float,
        max_favorable_excursion_bps: float,
        max_adverse_excursion_bps: float,
        label: int,
        gross_return_bps: float = 0.0,
        cost_bps: float = 0.0,
        label_threshold_bps: float = 5.0,
        label_schema_version: str = LABEL_SCHEMA_VERSION,
    ) -> None:
        """Write or update outcome label for a prediction."""
        await self._execute(
            """
            INSERT INTO prediction_outcomes (
                prediction_id, horizon_minutes, gross_return_bps, cost_bps,
                label_threshold_bps, net_return_bps,
                max_favorable_excursion_bps, max_adverse_excursion_bps,
                label, resolved_at, label_schema_version
            )
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, now(), $10)
            ON CONFLICT (prediction_id, horizon_minutes) DO UPDATE SET
                gross_return_bps = EXCLUDED.gross_return_bps,
                cost_bps = EXCLUDED.cost_bps,
                label_threshold_bps = EXCLUDED.label_threshold_bps,
                net_return_bps = EXCLUDED.net_return_bps,
                max_favorable_excursion_bps = EXCLUDED.max_favorable_excursion_bps,
                max_adverse_excursion_bps = EXCLUDED.max_adverse_excursion_bps,
                label = EXCLUDED.label,
                resolved_at = now(),
                label_schema_version = EXCLUDED.label_schema_version
            """,
            prediction_id,
            horizon_minutes,
            gross_return_bps,
            cost_bps,
            label_threshold_bps,
            net_return_bps,
            max_favorable_excursion_bps,
            max_adverse_excursion_bps,
            label,
            label_schema_version,
        )

    async def resolve_outcomes_from_candles(
        self,
        *,
        horizon_minutes: int,
        label_bps_threshold: float = 5.0,
        limit: int = 200,
    ) -> int:
        """Resolve prediction outcomes using market_candles data.

        NOTE: This base-class implementation is superseded in production by
        DirectionalTradeJournal.resolve_outcomes_from_candles (installed at
        import time via trader.storage.__init__). The subclass uses full-path
        MFE/MAE, label_schema_version filtering, and a correct cost model.
        This base version is kept for test isolation only.

        For each prediction_event that has no outcome at ``horizon_minutes`` yet,
        and whose candle_open_time + horizon has elapsed, look up entry and horizon
        close prices in market_candles and insert the outcome.

        Returns the number of outcomes resolved.
        """
        rows = await self._fetch(
            """
            SELECT
                pe.prediction_id,
                pe.symbol,
                pe.strategy_signal,
                fs.candle_open_time AS entry_time
            FROM prediction_events pe
            JOIN feature_snapshots fs ON fs.snapshot_id = pe.feature_snapshot_id
            LEFT JOIN prediction_outcomes po
                ON po.prediction_id = pe.prediction_id
                AND po.horizon_minutes = $1
            WHERE po.prediction_id IS NULL
              AND pe.created_at < now() - ($1 * interval '1 minute')
              AND pe.feature_snapshot_id IS NOT NULL
              AND pe.strategy_signal IN ('Buy', 'Sell')
            LIMIT $2
            """,
            horizon_minutes,
            limit,
        )

        resolved = 0
        for row in rows:
            prediction_id = str(row["prediction_id"])
            symbol = row["symbol"]
            side = str(row["strategy_signal"])
            entry_time = row["entry_time"]

            # Entry price: must be the exact confirmed candle that closed at the signal time.
            entry_rows = await self._fetch(
                """
                SELECT close FROM market_candles
                WHERE symbol=$1 AND interval='1' AND open_time = $2 AND confirmed = true
                """,
                symbol,
                entry_time,
            )
            if not entry_rows:
                continue
            entry_close = float(entry_rows[0]["close"])
            if entry_close <= 0:
                continue

            from datetime import timedelta

            horizon_time = entry_time + timedelta(minutes=horizon_minutes)
            # Horizon price: the last confirmed candle with open_time <= horizon_time.
            # Since we only write confirmed candles, and we wait until horizon_time
            # is in the past (pe.created_at < now() - horizon), the candle
            # at horizon_time is almost certainly confirmed. Still we filter confirmed.
            horizon_rows = await self._fetch(
                """
                SELECT close, high, low, open_time FROM market_candles
                WHERE symbol=$1 AND interval='1' AND open_time <= $2 AND confirmed = true
                ORDER BY open_time DESC LIMIT 1
                """,
                symbol,
                horizon_time,
            )
            if not horizon_rows:
                continue
            # Reject if the closest candle is more than 5 minutes stale — a gap
            # in market data would produce systematically biased training labels.
            candle_open_time = horizon_rows[0]["open_time"]
            if candle_open_time.tzinfo is None:
                candle_open_time = candle_open_time.replace(tzinfo=UTC)
            from datetime import timedelta as _td

            if abs((candle_open_time - horizon_time).total_seconds()) > 5 * 60:
                continue
            horizon_close = float(horizon_rows[0]["close"])
            horizon_high = float(horizon_rows[0]["high"])
            horizon_low = float(horizon_rows[0]["low"])

            # Canonical directional math (labels.py): a profitable Sell yields a
            # POSITIVE return when price falls — never label raw price moves.
            outcome = build_directional_outcome(
                side=side,
                entry_price=entry_close,
                exit_price=horizon_close,
                highs=[horizon_high],
                lows=[horizon_low],
                cost_model=CostModelBps(),
                label_threshold_bps=label_bps_threshold,
            )

            await self.resolve_prediction_outcomes(
                prediction_id=prediction_id,
                horizon_minutes=horizon_minutes,
                gross_return_bps=outcome.gross_return_bps,
                cost_bps=0.0,
                label_threshold_bps=label_bps_threshold,
                net_return_bps=outcome.net_return_bps,
                max_favorable_excursion_bps=outcome.max_favorable_excursion_bps,
                max_adverse_excursion_bps=outcome.max_adverse_excursion_bps,
                label=outcome.label,
                label_schema_version=LABEL_SCHEMA_VERSION,
            )
            resolved += 1

        return resolved

    async def record_order_update_event(
        self,
        *,
        order_link_id: str,
        exchange_order_id: str | None,
        symbol: str,
        side: str,
        qty: Decimal,
        state: str,
        error: str | None = None,
    ) -> None:
        """Record an OrderUpdateEvent from private WS (updates durable state)."""
        await self.upsert_durable_order_state(
            order_link_id=order_link_id,
            symbol=symbol,
            side=side,
            qty=qty,
            state=state,
            exchange_order_id=exchange_order_id,
            last_error=error,
        )

    async def get_returns_for_model(
        self,
        model_version: str,
        limit: int = 200,
        horizon_minutes: int | None = None,
        label_schema_version: str | None = None,
    ) -> list[float]:
        """Return recent resolved net returns (bps) for a model version, newest first.

        Used by the auto-promoter's bootstrap significance test. Pass
        model_version='RULE_BASELINE_V1' for the baseline distribution.
        """
        if not self.is_enabled:
            return []
        resolved_label_schema = label_schema_version or active_label_schema_version(use_tpsl_exit=True)
        if horizon_minutes is not None:
            if model_version == "RULE_BASELINE_V1":
                rows = await self._fetch(
                    """
                    SELECT po.net_return_bps
                    FROM prediction_outcomes po
                    JOIN prediction_events pe ON pe.prediction_id = po.prediction_id
                    WHERE pe.model_version = $1
                      AND COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'
                      AND po.net_return_bps IS NOT NULL
                      AND po.label_schema_version = $4
                      AND po.horizon_minutes = $3
                    ORDER BY po.resolved_at DESC NULLS LAST
                    LIMIT $2
                    """,
                    model_version,
                    int(limit),
                    int(horizon_minutes),
                    resolved_label_schema,
                )
            else:
                rows = await self._fetch(
                    """
                    SELECT po.net_return_bps
                    FROM prediction_outcomes po
                    JOIN prediction_events pe ON pe.prediction_id = po.prediction_id
                    WHERE pe.model_version = $1
                      AND pe.decision = 'GATE_PASS'
                      AND po.net_return_bps IS NOT NULL
                      AND po.label_schema_version = $4
                      AND po.horizon_minutes = $3
                    ORDER BY po.resolved_at DESC NULLS LAST
                    LIMIT $2
                    """,
                    model_version,
                    int(limit),
                    int(horizon_minutes),
                    resolved_label_schema,
                )
        else:
            if model_version == "RULE_BASELINE_V1":
                rows = await self._fetch(
                    """
                    SELECT po.net_return_bps
                    FROM prediction_outcomes po
                    JOIN prediction_events pe ON pe.prediction_id = po.prediction_id
                    WHERE pe.model_version = $1
                      AND COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'
                      AND po.net_return_bps IS NOT NULL
                      AND po.label_schema_version = $3
                    ORDER BY po.resolved_at DESC NULLS LAST
                    LIMIT $2
                    """,
                    model_version,
                    int(limit),
                    resolved_label_schema,
                )
            else:
                rows = await self._fetch(
                    """
                    SELECT po.net_return_bps
                    FROM prediction_outcomes po
                    JOIN prediction_events pe ON pe.prediction_id = po.prediction_id
                    WHERE pe.model_version = $1
                      AND pe.decision = 'GATE_PASS'
                      AND po.net_return_bps IS NOT NULL
                      AND po.label_schema_version = $3
                    ORDER BY po.resolved_at DESC NULLS LAST
                    LIMIT $2
                    """,
                    model_version,
                    int(limit),
                    resolved_label_schema,
                )
        return [float(r["net_return_bps"]) for r in rows]

    async def get_shadow_gate_stats(
        self,
        model_version: str,
        horizon_minutes: int,
        label_schema_version: str,
    ) -> dict[str, Any]:
        """Return shadow gate statistics for a specific model version.

        Filters by exact model_version, horizon_minutes, label_schema_version,
        and feature_schema_hash (to exclude incompatible feature snapshots).
        Only includes resolved outcomes with GATE_PASS/GATE_BLOCK decisions.
        """
        if not self.is_enabled:
            return {}

        try:
            # First verify the model exists and get its feature schema
            model_rows = await self._fetch(
                """
                SELECT feature_schema_hash, metrics
                FROM model_versions
                WHERE version = $1
                LIMIT 1
                """,
                model_version,
            )
            feature_schema_hash = ""
            if model_rows:
                feature_schema_hash = model_rows[0].get("feature_schema_hash", "") or ""

            # Fail-safe: if the model has no feature schema hash, don't mix incompatible snapshots
            if not feature_schema_hash:
                log.warning(
                    "trade_journal.shadow_gate_stats_no_schema_hash",
                    model_version=model_version,
                    message="Model has no feature_schema_hash; returning empty stats to avoid mixing incompatible snapshots",
                )
                return {"model_version": model_version, "feature_schema_hash": ""}

            gate_rows = await self._fetch(
                """
                SELECT
                    pe.decision,
                    count(*) AS cnt,
                    avg(po.net_return_bps) AS avg_net_return_bps,
                    avg(po.label::double precision) AS precision
                FROM prediction_events pe
                JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                JOIN feature_snapshots fs ON fs.snapshot_id = pe.feature_snapshot_id
                WHERE pe.model_version = $1
                  AND po.horizon_minutes = $2
                  AND po.label IS NOT NULL
                  AND po.label_schema_version = $3
                  AND pe.decision IN ('GATE_PASS', 'GATE_BLOCK')
                  AND fs.feature_values IS NOT NULL
                  AND fs.feature_schema_hash = $4
                GROUP BY pe.decision
                """,
                model_version,
                horizon_minutes,
                label_schema_version,
                feature_schema_hash,
            )

            gate: dict[str, Any] = {"model_version": model_version}
            total_count = 0
            weighted_return = 0.0
            for row in gate_rows:
                decision = str(row["decision"])
                count = int(row["cnt"])
                avg_return = float(row["avg_net_return_bps"] or 0.0)
                raw_precision = row["precision"]
                precision = float(raw_precision) if raw_precision is not None else None
                total_count += count
                weighted_return += avg_return * count
                key = "pass" if decision == "GATE_PASS" else "block"
                gate[f"{key}_count"] = count
                gate[f"{key}_avg_net_return_bps"] = avg_return
                gate[f"{key}_precision"] = precision

            if total_count:
                all_avg = weighted_return / total_count
                pass_avg = gate.get("pass_avg_net_return_bps")
                block_avg = gate.get("block_avg_net_return_bps")
                gate["total_count"] = total_count
                gate["all_avg_net_return_bps"] = all_avg
                gate["lift_vs_all_bps"] = (float(pass_avg) - all_avg) if pass_avg is not None else None
                gate["pass_vs_block_bps"] = (
                    (float(pass_avg) - float(block_avg)) if pass_avg is not None and block_avg is not None else None
                )
                reason_rows = await self._fetch(
                    """
                    SELECT
                        COALESCE(pe.metadata->>'gate_reason', 'unknown') AS reason,
                        count(*) AS cnt,
                        avg(po.net_return_bps) AS avg_net_return_bps
                    FROM prediction_events pe
                    JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                    JOIN feature_snapshots fs ON fs.snapshot_id = pe.feature_snapshot_id
                    WHERE pe.model_version = $1
                      AND pe.decision = 'GATE_BLOCK'
                      AND po.horizon_minutes = $2
                      AND po.label IS NOT NULL
                      AND po.label_schema_version = $3
                      AND fs.feature_values IS NOT NULL
                      AND fs.feature_schema_hash = $4
                    GROUP BY reason
                    ORDER BY cnt DESC
                    LIMIT 3
                    """,
                    model_version,
                    horizon_minutes,
                    label_schema_version,
                    feature_schema_hash,
                )
                if reason_rows:
                    gate["top_block_reasons"] = {str(row["reason"]): int(row["cnt"]) for row in reason_rows}
                    side_filtered_count = 0
                    score_block_count = 0
                    score_block_weighted_return = 0.0
                    for row in reason_rows:
                        reason = str(row["reason"])
                        count = int(row["cnt"] or 0)
                        avg_return = float(row.get("avg_net_return_bps") or 0.0)
                        if reason == "side_not_selected_by_model":
                            side_filtered_count += count
                        else:
                            score_block_count += count
                            score_block_weighted_return += avg_return * count
                    gate["side_filtered_count"] = side_filtered_count
                    gate["score_block_count"] = score_block_count
                    gate["score_block_avg_net_return_bps"] = (
                        score_block_weighted_return / score_block_count if score_block_count else None
                    )

            # Quality check from model metrics
            quality = "UNKNOWN"
            if model_rows:
                metrics_raw = model_rows[0].get("metrics")
                if isinstance(metrics_raw, str):
                    import json

                    metrics = json.loads(metrics_raw)
                else:
                    metrics = metrics_raw or {}
                quality = metrics.get("quality", "UNKNOWN")

            gate["quality"] = quality
            gate["feature_schema_hash"] = feature_schema_hash

            return gate

        except Exception as exc:
            self._last_read_error_at = datetime.now(tz=UTC)
            self._last_read_error = str(exc)
            log.debug("trade_journal.shadow_gate_stats_failed", error=str(exc))
            return {}

    async def get_shadow_gate_event_counts(
        self,
        model_version: str,
        horizon_minutes: int,
        label_schema_version: str,
    ) -> dict[str, Any]:
        """Return raw/resolved gate-event counts for a model version."""
        if not self.is_enabled:
            return {}

        try:
            model_rows = await self._fetch(
                """
                SELECT feature_schema_hash
                FROM model_versions
                WHERE version = $1
                LIMIT 1
                """,
                model_version,
            )
            feature_schema_hash = str(model_rows[0]["feature_schema_hash"] or "") if model_rows else ""
            if not feature_schema_hash:
                return {"model_version": model_version, "feature_schema_hash": ""}

            rows = await self._fetch(
                """
                SELECT
                    pe.decision,
                    count(*) AS total_count,
                    count(po.prediction_id) AS resolved_count
                FROM prediction_events pe
                JOIN feature_snapshots fs ON fs.snapshot_id = pe.feature_snapshot_id
                LEFT JOIN prediction_outcomes po
                    ON po.prediction_id = pe.prediction_id
                   AND po.horizon_minutes = $2
                   AND po.label_schema_version = $3
                WHERE pe.model_version = $1
                  AND pe.decision IN ('GATE_PASS', 'GATE_BLOCK')
                  AND fs.feature_values IS NOT NULL
                  AND fs.feature_schema_hash = $4
                GROUP BY pe.decision
                """,
                model_version,
                horizon_minutes,
                label_schema_version,
                feature_schema_hash,
            )
            result: dict[str, Any] = {
                "model_version": model_version,
                "feature_schema_hash": feature_schema_hash,
                "total_count": 0,
                "resolved_count": 0,
                "pending_count": 0,
            }
            for row in rows:
                decision = str(row["decision"])
                total = int(row["total_count"] or 0)
                resolved = int(row["resolved_count"] or 0)
                key = "pass" if decision == "GATE_PASS" else "block"
                result[f"{key}_count"] = total
                result[f"{key}_resolved_count"] = resolved
                result["total_count"] += total
                result["resolved_count"] += resolved
            result["pending_count"] = max(0, int(result["total_count"]) - int(result["resolved_count"]))
            return result
        except Exception as exc:
            self._last_read_error_at = datetime.now(tz=UTC)
            self._last_read_error = str(exc)
            log.debug("trade_journal.shadow_gate_event_counts_failed", error=str(exc))
            return {}

    async def get_prediction_event_decision_counts(
        self,
        *,
        horizon_minutes: int,
        label_schema_version: str,
        limit: int = 2000,
    ) -> dict[str, Any]:
        """Return recent prediction-event totals and outcome resolution by decision.

        This is diagnostic-only. It tells operators whether paper-gate is empty
        because events are not being written, or because outcomes are still
        pending/unresolved for the requested horizon/schema.
        """
        if not self.is_enabled:
            return {}

        try:
            rows = await self._fetch(
                """
                SELECT
                    COALESCE(decision, 'NULL') AS decision,
                    count(*) AS total_count,
                    count(feature_snapshot_id) AS with_snapshot_count,
                    count(po.prediction_id) AS resolved_count
                FROM (
                    SELECT prediction_id, decision, feature_snapshot_id, created_at
                    FROM prediction_events
                    WHERE decision IN ('SHADOW_BASELINE', 'GATE_PASS', 'GATE_BLOCK')
                    ORDER BY created_at DESC
                    LIMIT $1
                ) pe
                LEFT JOIN prediction_outcomes po
                    ON po.prediction_id = pe.prediction_id
                   AND po.horizon_minutes = $2
                   AND po.label_schema_version = $3
                GROUP BY decision
                ORDER BY total_count DESC
                """,
                max(1, int(limit)),
                int(horizon_minutes),
                label_schema_version,
            )
            result: dict[str, Any] = {
                "horizon_minutes": int(horizon_minutes),
                "label_schema_version": label_schema_version,
                "total_count": 0,
                "resolved_count": 0,
                "pending_count": 0,
                "with_snapshot_count": 0,
                "by_decision": {},
            }
            by_decision: dict[str, Any] = {}
            for row in rows:
                decision = str(row["decision"] or "NULL")
                total = int(row["total_count"] or 0)
                resolved = int(row["resolved_count"] or 0)
                with_snapshot = int(row["with_snapshot_count"] or 0)
                pending = max(0, total - resolved)
                by_decision[decision] = {
                    "total_count": total,
                    "resolved_count": resolved,
                    "pending_count": pending,
                    "with_snapshot_count": with_snapshot,
                }
                result["total_count"] += total
                result["resolved_count"] += resolved
                result["with_snapshot_count"] += with_snapshot
            result["pending_count"] = max(0, int(result["total_count"]) - int(result["resolved_count"]))
            result["by_decision"] = by_decision
            return result
        except Exception as exc:
            self._last_read_error_at = datetime.now(tz=UTC)
            self._last_read_error = str(exc)
            log.debug("trade_journal.prediction_event_decision_counts_failed", error=str(exc))
            return {}

    @staticmethod
    def _decode_json_field(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    @staticmethod
    def _sample_variance(values: list[float]) -> float | None:
        if len(values) < 2:
            return None
        avg = sum(values) / len(values)
        return sum((value - avg) ** 2 for value in values) / (len(values) - 1)

    @staticmethod
    def _two_sided_normal_pvalue(sample: list[float], population: list[float]) -> float | None:
        if len(sample) < 2 or len(population) < 2:
            return None
        sample_var = TradeJournal._sample_variance(sample)
        population_var = TradeJournal._sample_variance(population)
        if sample_var is None or population_var is None:
            return None
        se = math.sqrt(sample_var / len(sample) + population_var / len(population))
        if se <= 0:
            return None
        z_score = abs((sum(sample) / len(sample)) - (sum(population) / len(population))) / se
        return math.erfc(z_score / math.sqrt(2.0))

    async def _analysis_model_version(self) -> dict[str, Any]:
        rows = await self._fetch(
            """
            SELECT version, status, training_finished_at, created_at
            FROM model_versions
            WHERE artifact IS NOT NULL
            ORDER BY CASE WHEN status = 'CHAMPION' THEN 0 ELSE 1 END,
                     training_finished_at DESC NULLS LAST,
                     created_at DESC
            LIMIT 1
            """
        )
        return dict(rows[0]) if rows else {}

    async def get_strategy_pnl_analysis(
        self,
        horizon_minutes: int = 15,
        label_schema_version: str = LABEL_SCHEMA_VERSION,
    ) -> dict[str, Any]:
        """Aggregate baseline strategy expectancy by common operator slices."""
        if not self.is_enabled:
            return {"connected": False}
        try:
            args = (label_schema_version, int(horizon_minutes))
            symbol_rows = await self._fetch(
                """
                SELECT pe.symbol,
                       count(*) AS count,
                       avg(po.net_return_bps) AS avg_net_return_bps,
                       sum(po.net_return_bps) AS total_net_return_bps
                FROM prediction_events pe
                JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                WHERE pe.model_version = 'RULE_BASELINE_V1'
                  AND COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'
                  AND po.label_schema_version = $1
                  AND po.horizon_minutes = $2
                  AND po.net_return_bps IS NOT NULL
                GROUP BY pe.symbol
                ORDER BY avg_net_return_bps DESC
                """,
                *args,
            )
            hour_rows = await self._fetch(
                """
                SELECT EXTRACT(HOUR FROM pe.created_at AT TIME ZONE 'UTC')::int AS hour_utc,
                       count(*) AS count,
                       avg(po.net_return_bps) AS avg_net_return_bps,
                       sum(po.net_return_bps) AS total_net_return_bps
                FROM prediction_events pe
                JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                WHERE pe.model_version = 'RULE_BASELINE_V1'
                  AND COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'
                  AND po.label_schema_version = $1
                  AND po.horizon_minutes = $2
                  AND po.net_return_bps IS NOT NULL
                GROUP BY hour_utc
                ORDER BY hour_utc
                """,
                *args,
            )
            regime_rows = await self._fetch(
                """
                SELECT COALESCE(pe.metadata->>'regime', 'UNKNOWN') AS regime,
                       count(*) AS count,
                       avg(po.net_return_bps) AS avg_net_return_bps,
                       sum(po.net_return_bps) AS total_net_return_bps
                FROM prediction_events pe
                JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                WHERE pe.model_version = 'RULE_BASELINE_V1'
                  AND COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'
                  AND po.label_schema_version = $1
                  AND po.horizon_minutes = $2
                  AND po.net_return_bps IS NOT NULL
                GROUP BY regime
                ORDER BY avg_net_return_bps ASC
                """,
                *args,
            )
            weekday_rows = await self._fetch(
                """
                SELECT EXTRACT(ISODOW FROM pe.created_at AT TIME ZONE 'UTC')::int AS weekday,
                       count(*) AS count,
                       avg(po.net_return_bps) AS avg_net_return_bps,
                       sum(po.net_return_bps) AS total_net_return_bps
                FROM prediction_events pe
                JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                WHERE pe.model_version = 'RULE_BASELINE_V1'
                  AND COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'
                  AND po.label_schema_version = $1
                  AND po.horizon_minutes = $2
                  AND po.net_return_bps IS NOT NULL
                GROUP BY weekday
                ORDER BY weekday
                """,
                *args,
            )
            strategy_rows = await self._fetch(
                """
                SELECT COALESCE(pe.metadata->>'strategy_id', 'UNKNOWN') AS strategy_id,
                       count(*) AS count,
                       avg(po.gross_return_bps) AS avg_gross_return_bps,
                       avg(po.cost_bps) AS avg_cost_bps,
                       avg(po.net_return_bps) AS avg_net_return_bps,
                       sum(po.net_return_bps) AS total_net_return_bps
                FROM prediction_events pe
                JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                WHERE pe.model_version = 'RULE_BASELINE_V1'
                  AND COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'
                  AND po.label_schema_version = $1
                  AND po.horizon_minutes = $2
                  AND po.net_return_bps IS NOT NULL
                GROUP BY strategy_id
                ORDER BY avg_net_return_bps DESC
                """,
                *args,
            )

            def row_dict(row: Any) -> dict[str, Any]:
                data = dict(row)
                for key in (
                    "avg_gross_return_bps",
                    "avg_cost_bps",
                    "avg_net_return_bps",
                    "total_net_return_bps",
                ):
                    if data.get(key) is not None:
                        data[key] = float(data[key])
                if data.get("count") is not None:
                    data["count"] = int(data["count"])
                return data

            symbols = [row_dict(row) for row in symbol_rows]
            return {
                "connected": True,
                "horizon_minutes": int(horizon_minutes),
                "label_schema_version": label_schema_version,
                "symbols_best": symbols[:5],
                "symbols_worst": list(reversed(symbols[-5:])),
                "top_symbols": symbols[:5],
                "worst_symbols": list(reversed(symbols[-5:])),
                "hours": [row_dict(row) for row in hour_rows],
                "regimes": [row_dict(row) for row in regime_rows],
                "weekdays": [row_dict(row) for row in weekday_rows],
                "strategies": [row_dict(row) for row in strategy_rows],
            }
        except Exception as exc:
            self._last_read_error_at = datetime.now(tz=UTC)
            self._last_read_error = str(exc)
            log.warning("trade_journal.strategy_pnl_analysis_failed", error=str(exc))
            return {"connected": self.is_enabled, "error": str(exc)}

    async def get_model_compare_analysis(
        self,
        horizon_minutes: int = 15,
        label_schema_version: str = LABEL_SCHEMA_VERSION,
    ) -> dict[str, Any]:
        """Compare baseline, model gate pass, and same-size random baseline sample."""
        if not self.is_enabled:
            return {"connected": False}
        try:
            model = await self._analysis_model_version()
            model_version = str(model.get("version") or "")
            baseline_rows = await self._fetch(
                """
                SELECT po.net_return_bps
                FROM prediction_events pe
                JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                WHERE pe.model_version = 'RULE_BASELINE_V1'
                  AND COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'
                  AND po.label_schema_version = $1
                  AND po.horizon_minutes = $2
                  AND po.net_return_bps IS NOT NULL
                ORDER BY pe.created_at ASC
                LIMIT 5000
                """,
                label_schema_version,
                int(horizon_minutes),
            )
            baseline = [float(row["net_return_bps"] or 0.0) for row in baseline_rows]
            gate: list[float] = []
            if model_version:
                gate_rows = await self._fetch(
                    """
                    SELECT po.net_return_bps
                    FROM prediction_events pe
                    JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                    WHERE pe.model_version = $1
                      AND pe.decision = 'GATE_PASS'
                      AND po.label_schema_version = $2
                      AND po.horizon_minutes = $3
                      AND po.net_return_bps IS NOT NULL
                    ORDER BY pe.created_at ASC
                    LIMIT 5000
                    """,
                    model_version,
                    label_schema_version,
                    int(horizon_minutes),
                )
                gate = [float(row["net_return_bps"] or 0.0) for row in gate_rows]
            random_sample: list[float] = []
            if gate:
                random_rows = await self._fetch(
                    """
                    SELECT po.net_return_bps
                    FROM prediction_events pe
                    JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                    WHERE pe.model_version = 'RULE_BASELINE_V1'
                      AND COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'
                      AND po.label_schema_version = $1
                      AND po.horizon_minutes = $2
                      AND po.net_return_bps IS NOT NULL
                      AND random() < 0.20
                    LIMIT $3
                    """,
                    label_schema_version,
                    int(horizon_minutes),
                    len(gate),
                )
                random_sample = [float(row["net_return_bps"] or 0.0) for row in random_rows]

            def stats(values: list[float]) -> dict[str, Any]:
                return {
                    "count": len(values),
                    "total_bps": sum(values),
                    "avg_bps": (sum(values) / len(values)) if values else None,
                }

            return {
                "connected": True,
                "horizon_minutes": int(horizon_minutes),
                "label_schema_version": label_schema_version,
                "model_version": model_version or None,
                "model_status": model.get("status"),
                "baseline": stats(baseline),
                "gate_pass": stats(gate),
                "random_baseline_sample": stats(random_sample),
                "p_value_vs_baseline": self._two_sided_normal_pvalue(gate, baseline),
            }
        except Exception as exc:
            self._last_read_error_at = datetime.now(tz=UTC)
            self._last_read_error = str(exc)
            log.warning("trade_journal.model_compare_analysis_failed", error=str(exc))
            return {"connected": self.is_enabled, "error": str(exc)}

    async def get_worst_prediction_outcomes(self, limit: int = 10, horizon_minutes: int = 15) -> list[dict[str, Any]]:
        """Return worst resolved baseline strategy outcomes with features and model gate metadata."""
        if not self.is_enabled:
            return []
        limit = max(1, min(int(limit), 20))
        try:
            rows = await self._fetch(
                """
                SELECT pe.symbol,
                       pe.strategy_signal,
                       pe.created_at,
                       po.net_return_bps,
                       fs.feature_names,
                       fs.feature_values,
                       mpe.model_version AS gate_model_version,
                       mpe.score AS gate_score,
                       mpe.decision AS gate_decision
                FROM prediction_events pe
                JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                LEFT JOIN feature_snapshots fs ON fs.snapshot_id = pe.feature_snapshot_id
                LEFT JOIN LATERAL (
                    SELECT model_version, score, decision
                    FROM prediction_events model_pe
                    WHERE model_pe.feature_snapshot_id = pe.feature_snapshot_id
                      AND model_pe.model_version <> 'RULE_BASELINE_V1'
                      AND model_pe.decision IN ('GATE_PASS', 'GATE_BLOCK')
                    ORDER BY model_pe.created_at DESC
                    LIMIT 1
                ) mpe ON true
                WHERE pe.model_version = 'RULE_BASELINE_V1'
                  AND COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'
                  AND po.label_schema_version = $2
                  AND po.horizon_minutes = $3
                  AND po.net_return_bps < 0
                ORDER BY po.net_return_bps ASC
                LIMIT $1
                """,
                limit,
                LABEL_SCHEMA_VERSION,
                int(horizon_minutes),
            )
            key_features = ["rsi_14", "atr_14_pct", "ob_imbalance_l5", "microprice_deviation_bps"]
            result: list[dict[str, Any]] = []
            for row in rows:
                data = dict(row)
                names = self._decode_json_field(data.get("feature_names")) or []
                values = self._decode_json_field(data.get("feature_values")) or []
                feature_map: dict[str, float | None] = {}
                if isinstance(names, list) and isinstance(values, list):
                    by_name = {str(name): values[idx] for idx, name in enumerate(names) if idx < len(values)}
                    for feature in key_features:
                        raw_value = by_name.get(feature)
                        try:
                            feature_map[feature] = float(raw_value) if raw_value is not None else None
                        except (TypeError, ValueError):
                            feature_map[feature] = None
                result.append(
                    {
                        "symbol": data.get("symbol"),
                        "side": data.get("strategy_signal"),
                        "created_at": data.get("created_at"),
                        "net_return_bps": float(data.get("net_return_bps") or 0.0),
                        "features": feature_map,
                        "model_version": data.get("gate_model_version"),
                        "score": float(data["gate_score"]) if data.get("gate_score") is not None else None,
                        "decision": data.get("gate_decision"),
                    }
                )
            return result
        except Exception as exc:
            self._last_read_error_at = datetime.now(tz=UTC)
            self._last_read_error = str(exc)
            log.warning("trade_journal.worst_prediction_outcomes_failed", error=str(exc))
            return []

    async def get_detailed_costs(
        self,
        horizon_minutes: int = 15,
        label_schema_version: str = LABEL_SCHEMA_VERSION,
    ) -> dict[str, Any]:
        """Return bps-level gross/net outcome costs plus maker share when available."""
        if not self.is_enabled:
            return {"connected": False}
        try:
            day_start = datetime.now(tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            today_rows = await self._fetch(
                """
                SELECT count(*) AS count,
                       avg(po.gross_return_bps) AS avg_gross_return_bps,
                       avg(po.net_return_bps) AS avg_net_return_bps,
                       avg(po.cost_bps) AS avg_cost_bps
                FROM prediction_events pe
                JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                WHERE pe.model_version = 'RULE_BASELINE_V1'
                  AND COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'
                  AND po.label_schema_version = $1
                  AND po.horizon_minutes = $2
                  AND po.net_return_bps IS NOT NULL
                  AND pe.created_at >= $3
                """,
                label_schema_version,
                int(horizon_minutes),
                day_start,
            )
            all_rows = await self._fetch(
                """
                SELECT count(*) AS count,
                       avg(po.gross_return_bps) AS avg_gross_return_bps,
                       avg(po.net_return_bps) AS avg_net_return_bps,
                       avg(po.cost_bps) AS avg_cost_bps
                FROM prediction_events pe
                JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                WHERE pe.model_version = 'RULE_BASELINE_V1'
                  AND COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'
                  AND po.label_schema_version = $1
                  AND po.horizon_minutes = $2
                  AND po.net_return_bps IS NOT NULL
                """,
                label_schema_version,
                int(horizon_minutes),
            )

            def outcome_stats(rows: list[Any]) -> dict[str, Any]:
                row = dict(rows[0]) if rows else {}
                return {
                    "count": int(row.get("count") or 0),
                    "avg_gross_return_bps": (
                        float(row["avg_gross_return_bps"]) if row.get("avg_gross_return_bps") is not None else None
                    ),
                    "avg_net_return_bps": (
                        float(row["avg_net_return_bps"]) if row.get("avg_net_return_bps") is not None else None
                    ),
                    "avg_cost_bps": float(row["avg_cost_bps"]) if row.get("avg_cost_bps") is not None else None,
                }

            maker_rows = await self._fetch(
                """
                SELECT count(*) FILTER (WHERE is_maker = true) AS maker_count,
                       count(*) FILTER (WHERE is_maker IS NOT NULL) AS known_count
                FROM execution_events
                """
            )
            maker_row = dict(maker_rows[0]) if maker_rows else {}
            maker_count = int(maker_row.get("maker_count") or 0)
            known_count = int(maker_row.get("known_count") or 0)
            return {
                "connected": True,
                "horizon_minutes": int(horizon_minutes),
                "label_schema_version": label_schema_version,
                "today": outcome_stats(today_rows),
                "all": outcome_stats(all_rows),
                "all_time": outcome_stats(all_rows),
                "maker_fill_pct": (maker_count / known_count * 100.0) if known_count else None,
                "maker_fill_count": maker_count,
                "known_fill_count": known_count,
            }
        except Exception as exc:
            self._last_read_error_at = datetime.now(tz=UTC)
            self._last_read_error = str(exc)
            log.warning("trade_journal.detailed_costs_failed", error=str(exc))
            return {"connected": self.is_enabled, "error": str(exc)}

    async def get_live_paper_gate_stats(
        self,
        model_version: str,
        *,
        horizon_minutes: int = 15,
        feature_schema_hash: str = "",
    ) -> dict[str, Any]:
        """Return live paper GATE_PASS PnL from resolved prediction outcomes."""
        if not self.is_enabled or not model_version:
            return {"count": 0, "total_bps": 0.0, "avg_bps": None, "max_drawdown_bps": 0.0}
        label_schema = _query_label_schema_version()
        try:
            gate_rows = await self._fetch(
                """
                SELECT net_return_bps
                FROM (
                    SELECT pe.created_at, po.net_return_bps
                    FROM prediction_events pe
                    JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                    LEFT JOIN feature_snapshots fs ON fs.snapshot_id = pe.feature_snapshot_id
                    WHERE pe.model_version = $1
                      AND pe.decision = 'GATE_PASS'
                      AND po.horizon_minutes = $2
                      AND po.label IS NOT NULL
                      AND po.label_schema_version = $3
                      AND ($4::text = '' OR fs.feature_schema_hash = $4)
                    ORDER BY pe.created_at DESC
                    LIMIT 1000
                ) recent
                ORDER BY created_at ASC
                """,
                model_version,
                int(horizon_minutes),
                label_schema,
                feature_schema_hash,
            )
            return _paper_stats_from_rows(gate_rows)
        except Exception as exc:
            self._last_read_error_at = datetime.now(tz=UTC)
            self._last_read_error = str(exc)
            log.debug("trade_journal.live_paper_gate_stats_failed", error=str(exc))
            return {"count": 0, "total_bps": 0.0, "avg_bps": None, "max_drawdown_bps": 0.0}

    async def _enrich_model_with_live_paper(self, model: dict[str, Any] | None) -> dict[str, Any] | None:
        if not model:
            return model
        try:
            version = str(model.get("version") or "")
            if not version:
                return model
            raw_metrics = self._decode_json_field(model.get("metrics")) or {}
            metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
            horizon = _model_horizon_minutes_from_metrics(metrics)
            schema_hash = _feature_schema_hash_from_metrics(metrics)
            live = await self.get_live_paper_gate_stats(
                version,
                horizon_minutes=horizon,
                feature_schema_hash=schema_hash,
            )
            live_count = int(live.get("count") or 0)
            enriched = dict(model)
            enriched["paper_gate_count"] = live_count
            enriched["paper_gate_total_bps"] = live.get("total_bps")
            enriched["paper_gate_avg_bps"] = live.get("avg_bps")
            enriched["paper_gate_source"] = "live_outcomes"
            enriched_metrics = dict(metrics)
            enriched_metrics["paper_gate"] = {
                "count": live_count,
                "total_bps": live.get("total_bps"),
                "avg_bps": live.get("avg_bps"),
            }
            enriched["selection_reason"] = selection_reason(enriched_metrics)
            return enriched
        except Exception as exc:
            log.debug("trade_journal.live_paper_enrich_failed", error=str(exc))
            return model

    async def get_model_performance_history(self, limit: int = 12) -> list[dict[str, Any]]:
        """Return recent model registry metrics for Telegram diagnostics."""
        if not self.is_enabled:
            return []
        try:
            rows = await self._fetch(
                """
                SELECT version, status, training_finished_at, created_at, training_samples, metrics
                FROM model_versions
                WHERE metrics IS NOT NULL
                ORDER BY training_finished_at DESC NULLS LAST, created_at DESC
                LIMIT $1
                """,
                max(1, min(int(limit), 30)),
            )
            result: list[dict[str, Any]] = []
            for row in rows:
                data = dict(row)
                metrics = self._decode_json_field(data.get("metrics")) or {}
                if not isinstance(metrics, dict):
                    metrics = {}
                normalized = model_selection_metrics(metrics)
                entry = {
                    "version": data.get("version"),
                    "status": data.get("status"),
                    "created_at": data.get("training_finished_at") or data.get("created_at"),
                    "training_samples": data.get("training_samples"),
                    "quality": metrics.get("quality", "n/a"),
                    "precision": metrics.get("precision"),
                    "lift_bps": normalized["lift_bps"],
                    "walk_forward_expectancy_bps": normalized["walk_forward_bps"],
                    "walk_forward_bps": normalized["walk_forward_bps"],
                    "model_score": metrics.get("model_score", normalized["model_score"]),
                    "paper_gate_count": normalized["paper_gate_count"],
                    "walk_forward_pass_count": normalized["walk_forward_pass_count"],
                    "selection_reason": selection_reason(metrics),
                    "metrics": metrics,
                }
                result.append(await self._enrich_model_with_live_paper(entry))
            return result
        except Exception as exc:
            self._last_read_error_at = datetime.now(tz=UTC)
            self._last_read_error = str(exc)
            log.warning("trade_journal.model_performance_history_failed", error=str(exc))
            return []

    async def get_champion_health(self) -> dict[str, Any]:
        """Return operator-facing champion health, candidate, and promotion audit context."""
        if not self.is_enabled:
            return {"connected": False}
        min_paper_gate_count, min_wf_positive_folds, max_wf_std_bps = _champion_selection_thresholds()
        try:
            champion_rows = await self._fetch(
                """
                SELECT version, status, training_finished_at, created_at, training_samples, metrics
                FROM model_versions
                WHERE status = 'CHAMPION'
                  AND artifact IS NOT NULL
                ORDER BY training_finished_at DESC NULLS LAST, created_at DESC
                LIMIT 1
                """
            )
            candidate_rows = await self._fetch(
                """
                SELECT version, status, training_finished_at, created_at, training_samples, metrics
                FROM model_versions
                WHERE status IN ('SHADOW_CHALLENGER', 'VALIDATED', 'ARCHIVED')
                  AND artifact IS NOT NULL
                ORDER BY
                    COALESCE(NULLIF(metrics->>'model_score', ''), '-1000000')::double precision DESC,
                    training_finished_at DESC NULLS LAST,
                    created_at DESC
                LIMIT 1
                """
            )
            promotion_rows = await self._fetch(
                """
                SELECT event_type, from_version, to_version, reasons, metrics, metrics_snapshot, created_at
                FROM model_promotion_log
                ORDER BY created_at DESC
                LIMIT 5
                """
            )

            def _model(row: Any | None) -> dict[str, Any] | None:
                if not row:
                    return None
                data = dict(row)
                metrics = self._decode_json_field(data.get("metrics")) or {}
                if not isinstance(metrics, dict):
                    metrics = {}
                normalized = model_selection_metrics(metrics)
                return {
                    "version": data.get("version"),
                    "status": data.get("status"),
                    "created_at": data.get("training_finished_at") or data.get("created_at"),
                    "training_samples": data.get("training_samples"),
                    "quality": metrics.get("quality", "n/a"),
                    "model_score": metrics.get("model_score", normalized["model_score"]),
                    "walk_forward_bps": normalized["walk_forward_bps"],
                    "wf_positive_folds": metrics.get("wf_positive_folds"),
                    "wf_folds": metrics.get("wf_folds"),
                    "wf_std_bps": metrics.get("wf_std_bps"),
                    "lift_bps": normalized["lift_bps"],
                    "paper_gate_count": normalized["paper_gate_count"],
                    "walk_forward_pass_count": normalized["walk_forward_pass_count"],
                    "selection_reason": selection_reason(metrics),
                    "walk_forward_chronology": metrics.get("walk_forward_chronology"),
                    "metrics": metrics,
                }

            champion = await self._enrich_model_with_live_paper(_model(champion_rows[0] if champion_rows else None))
            candidate = await self._enrich_model_with_live_paper(_model(candidate_rows[0] if candidate_rows else None))
            checks: list[dict[str, Any]] = []
            if champion:
                wf = champion.get("walk_forward_bps")
                paper_count = int(champion.get("paper_gate_count") or 0)
                positive_folds = int(champion.get("wf_positive_folds") or 0)
                wf_folds = int(champion.get("wf_folds") or 0)
                wf_std = champion.get("wf_std_bps")
                checks = [
                    {"name": "walk_forward_positive", "ok": wf is not None and float(wf) > 0, "value": wf},
                    {
                        "name": "paper_gate_count",
                        "ok": paper_count >= min_paper_gate_count,
                        "value": paper_count,
                        "threshold": min_paper_gate_count,
                    },
                    {
                        "name": "wf_fold_stability",
                        "ok": wf_folds == 0 or positive_folds >= min(min_wf_positive_folds, wf_folds),
                        "value": f"{positive_folds}/{wf_folds}",
                        "threshold": min(min_wf_positive_folds, wf_folds),
                    },
                    {
                        "name": "wf_std_bps",
                        "ok": wf_std is None or float(wf_std) <= max_wf_std_bps,
                        "value": wf_std,
                        "threshold": max_wf_std_bps,
                    },
                    {
                        "name": "strict_walk_forward",
                        "ok": champion.get("walk_forward_chronology") in (None, "strict_after_train"),
                        "value": champion.get("walk_forward_chronology") or "legacy",
                    },
                ]

            return {
                "connected": True,
                "champion": champion,
                "best_alternative": candidate,
                "checks": checks,
                "promotion_log": [dict(row) for row in promotion_rows],
            }
        except Exception as exc:
            self._last_read_error_at = datetime.now(tz=UTC)
            self._last_read_error = str(exc)
            log.warning("trade_journal.champion_health_failed", error=str(exc))
            return {"connected": self.is_enabled, "error": str(exc)}

    async def get_dashboard_data(self, horizon_minutes: int = 15) -> dict[str, Any]:
        """Return compact Chart.js-ready diagnostics for /dashboard."""
        if not self.is_enabled:
            return {"connected": False}
        try:
            model = await self._analysis_model_version()
            model_version = str(model.get("version") or "")
            label_schema = _query_label_schema_version()
            baseline_rows = await self._fetch(
                """
                SELECT pe.created_at, po.net_return_bps
                FROM prediction_events pe
                JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                WHERE pe.model_version = 'RULE_BASELINE_V1'
                  AND COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'
                  AND po.label_schema_version = $1
                  AND po.horizon_minutes = $2
                  AND po.net_return_bps IS NOT NULL
                ORDER BY pe.created_at ASC
                LIMIT 3000
                """,
                label_schema,
                int(horizon_minutes),
            )
            gate_rows = []
            if model_version:
                gate_rows = await self._fetch(
                    """
                    SELECT pe.created_at, po.net_return_bps
                    FROM prediction_events pe
                    JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                    WHERE pe.model_version = $1
                      AND pe.decision = 'GATE_PASS'
                      AND po.label_schema_version = $2
                      AND po.horizon_minutes = $3
                      AND po.net_return_bps IS NOT NULL
                    ORDER BY pe.created_at ASC
                    LIMIT 3000
                    """,
                    model_version,
                    label_schema,
                    int(horizon_minutes),
                )

            def equity(rows: list[Any]) -> list[dict[str, Any]]:
                total = 0.0
                points: list[dict[str, Any]] = []
                for row in rows:
                    data = dict(row)
                    total += float(data.get("net_return_bps") or 0.0)
                    ts = data.get("created_at")
                    points.append({"x": ts.isoformat() if isinstance(ts, datetime) else str(ts), "y": total})
                return points

            baseline_returns = [float(row["net_return_bps"] or 0.0) for row in baseline_rows]
            bins: list[int] = []
            if baseline_returns:
                min_ret = math.floor(min(baseline_returns) / 10.0) * 10
                max_ret = math.ceil(max(baseline_returns) / 10.0) * 10
                bins = list(range(int(min_ret), int(max_ret) + 10, 10))
            hist = []
            for start in bins:
                end = start + 10
                hist.append(
                    {
                        "bucket": f"{start:+d}..{end:+d}",
                        "count": sum(start <= value < end for value in baseline_returns),
                    }
                )

            heat_rows = await self._fetch(
                """
                SELECT EXTRACT(ISODOW FROM pe.created_at AT TIME ZONE 'UTC')::int AS weekday,
                       EXTRACT(HOUR FROM pe.created_at AT TIME ZONE 'UTC')::int AS hour_utc,
                       sum(po.net_return_bps) AS total_net_return_bps,
                       count(*) AS count
                FROM prediction_events pe
                JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                WHERE pe.model_version = 'RULE_BASELINE_V1'
                  AND COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'
                  AND po.label_schema_version = $1
                  AND po.horizon_minutes = $2
                  AND po.net_return_bps IS NOT NULL
                GROUP BY weekday, hour_utc
                ORDER BY weekday, hour_utc
                """,
                label_schema,
                int(horizon_minutes),
            )
            return {
                "connected": True,
                "horizon_minutes": int(horizon_minutes),
                "model_version": model_version or None,
                "equity_baseline": equity(baseline_rows),
                "equity_gate_pass": equity(gate_rows),
                "histogram": hist,
                "heatmap": [
                    {
                        "weekday": int(row["weekday"]),
                        "hour": int(row["hour_utc"]),
                        "total_bps": float(row["total_net_return_bps"] or 0.0),
                        "count": int(row["count"] or 0),
                    }
                    for row in heat_rows
                ],
            }
        except Exception as exc:
            self._last_read_error_at = datetime.now(tz=UTC)
            self._last_read_error = str(exc)
            log.warning("trade_journal.dashboard_data_failed", error=str(exc))
            return {"connected": self.is_enabled, "error": str(exc)}

    async def get_db_diagnostics(self, *, lite: bool = False) -> dict[str, Any]:
        """Return read-only diagnostics for Telegram 🗄 screen."""
        async with self._diag_lock:
            return await self._get_db_diagnostics_unlocked(lite=lite)

    async def _get_db_diagnostics_unlocked(self, *, lite: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "connected": self.is_enabled,
            "configured": self.is_configured,
            "connection_target": _safe_connection_target(self._dsn),
            "lite": lite,
            "schema_degraded": bool(
                self._last_connect_error and "schema bootstrap degraded" in str(self._last_connect_error).lower()
            ),
            "last_connect_error": self._last_connect_error,
            "last_connect_error_at": self._last_connect_error_at,
            "last_read_error": self._last_read_error,
            "last_read_error_at": self._last_read_error_at,
            "write_health": self.write_health(),
            "candles_by_interval": {},
            "latest_candle_1m": None,
            "last_confirmed_candle_age_s": None,
            "feature_snapshots": 0,
            "prediction_outcomes": 0,
            "prediction_outcomes_by_horizon": {},
            "labelled_samples_15m": 0,
            "latest_training_run": {},
            "latest_model_version": {},
            "active_model_version": {},
            "shadow_gate_15m": {},
            "paper_pnl_15m": {},
            "storage_stats": {},
        }
        if not self.is_enabled:
            return result

        async def _read_or_default(label: str, default: Any, reader: Any) -> Any:
            try:
                return await reader()
            except Exception as exc:
                self._last_read_error_at = datetime.now(tz=UTC)
                self._last_read_error = f"{label}: {exc}"
                result["last_read_error"] = self._last_read_error
                result["last_read_error_at"] = self._last_read_error_at
                log.debug("trade_journal.diagnostics_section_failed", section=label, error=str(exc))
                return default

        try:
            self._last_read_error = None
            self._last_read_error_at = None

            if lite:
                (
                    latest_candle_1m,
                    candles_by_interval,
                    feature_snapshots,
                    prediction_outcomes,
                    labelled_samples_15m,
                    training_rows,
                    model_rows,
                    champion_rows,
                ) = await asyncio.gather(
                    _read_or_default(
                        "latest_candle_1m",
                        None,
                        lambda: self.get_latest_candle_time("1"),
                    ),
                    _read_or_default(
                        "candle_readiness_counts",
                        {},
                        self.get_candle_readiness_counts,
                    ),
                    _read_or_default(
                        "feature_snapshot_readiness_count",
                        0,
                        self.get_feature_snapshot_readiness_count,
                    ),
                    _read_or_default(
                        "prediction_outcome_readiness_count",
                        0,
                        self.get_prediction_outcome_readiness_count,
                    ),
                    _read_or_default(
                        "labelled_15m_readiness_count",
                        0,
                        self.get_labelled_15m_readiness_count,
                    ),
                    self._fetch(
                        """
                        SELECT status, model_version, sample_count, error, metrics, started_at, finished_at
                        FROM training_runs
                        ORDER BY started_at DESC
                        LIMIT 1
                        """
                    ),
                    self._fetch(
                        """
                        SELECT version, status, training_samples, metrics, training_finished_at, created_at
                        FROM model_versions
                        WHERE artifact IS NOT NULL
                        ORDER BY training_finished_at DESC NULLS LAST, created_at DESC
                        LIMIT 1
                        """
                    ),
                    self._fetch(
                        """
                        SELECT version, status, training_samples, metrics, training_finished_at, created_at
                        FROM model_versions
                        WHERE status = 'CHAMPION'
                          AND artifact IS NOT NULL
                        ORDER BY training_finished_at DESC NULLS LAST, created_at DESC
                        LIMIT 1
                        """
                    ),
                )
                result["latest_candle_1m"] = latest_candle_1m
                if latest_candle_1m is not None:
                    result["last_confirmed_candle_age_s"] = max(
                        0.0, (datetime.now(tz=UTC) - latest_candle_1m).total_seconds()
                    )
                result["candles_by_interval"] = candles_by_interval
                result["feature_snapshots"] = int(feature_snapshots or 0)
                result["prediction_outcomes"] = int(prediction_outcomes or 0)
                result["labelled_samples_15m"] = int(labelled_samples_15m or 0)
                result["training_eligible_15m"] = result["labelled_samples_15m"]
                if training_rows:
                    result["latest_training_run"] = dict(training_rows[0])
                if model_rows:
                    result["latest_model_version"] = dict(model_rows[0])
                if champion_rows:
                    result["active_model_version"] = dict(champion_rows[0])
                else:
                    result["active_model_version"] = result.get("latest_model_version", {})
                result["storage_stats"] = await _read_or_default("storage_stats", {}, self.get_storage_stats)
                return result

            (
                candles_by_interval,
                latest_candle_1m,
                feature_snapshots,
                prediction_outcomes,
                by_horizon_rows,
                labelled_samples_15m,
            ) = await asyncio.gather(
                _read_or_default("candle_readiness_counts", {}, self.get_candle_readiness_counts),
                _read_or_default("latest_candle_1m", None, lambda: self.get_latest_candle_time("1")),
                _read_or_default("feature_snapshot_readiness_count", 0, self.get_feature_snapshot_readiness_count),
                _read_or_default("prediction_outcome_readiness_count", 0, self.get_prediction_outcome_readiness_count),
                _read_or_default(
                    "prediction_outcomes_by_horizon",
                    [],
                    lambda: self._fetch(
                        """
                        SELECT horizon_minutes, count(*) AS cnt
                        FROM (
                            SELECT horizon_minutes
                            FROM prediction_outcomes
                            WHERE label IS NOT NULL
                            LIMIT 4000
                        ) capped
                        GROUP BY horizon_minutes
                        ORDER BY horizon_minutes
                        """
                    ),
                ),
                _read_or_default("labelled_15m_readiness_count", 0, self.get_labelled_15m_readiness_count),
            )
            result["candles_by_interval"] = candles_by_interval
            result["latest_candle_1m"] = latest_candle_1m
            if latest_candle_1m is not None:
                result["last_confirmed_candle_age_s"] = max(
                    0.0, (datetime.now(tz=UTC) - latest_candle_1m).total_seconds()
                )
            result["feature_snapshots"] = int(feature_snapshots or 0)
            result["prediction_outcomes"] = int(prediction_outcomes or 0)
            result["prediction_outcomes_by_horizon"] = {
                str(row["horizon_minutes"]): int(row["cnt"]) for row in by_horizon_rows
            }
            result["labelled_samples_15m"] = int(labelled_samples_15m or 0)
            # P1: training_eligible = samples with label + features (same logic as trainer)
            result["training_eligible_15m"] = result["labelled_samples_15m"]

            # Per-schema breakdown: newest schema first. Wrapped in its own try/except so
            # a query failure here cannot break the rest of get_db_diagnostics().
            try:
                schema_rows = await self._fetch(
                    """
                    SELECT feature_schema_hash, count(*) AS cnt, max(latest_at) AS latest_at
                    FROM (
                        SELECT DISTINCT ON (fs.symbol, fs.interval, fs.candle_open_time, fs.feature_schema_hash)
                               fs.feature_schema_hash,
                               fs.created_at AS latest_at
                        FROM feature_snapshots fs
                        JOIN prediction_events pe ON pe.feature_snapshot_id = fs.snapshot_id
                        JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                        WHERE po.horizon_minutes = 15
                          AND po.label IS NOT NULL
                          AND fs.feature_values IS NOT NULL
                          AND fs.training_eligible = true
                          AND pe.model_version = 'RULE_BASELINE_V1'
                          AND pe.strategy_signal IN ('Buy', 'Sell')
                        ORDER BY fs.symbol, fs.interval, fs.candle_open_time,
                                 fs.feature_schema_hash, fs.created_at DESC
                    ) deduped
                    GROUP BY feature_schema_hash
                    ORDER BY latest_at DESC
                    LIMIT 5
                    """
                )
                if schema_rows:
                    result["training_schema_distribution"] = [
                        {
                            "schema_hash": str(row["feature_schema_hash"]),
                            "sample_count": int(row["cnt"]),
                            "latest_at": row["latest_at"].isoformat() if row["latest_at"] else None,
                        }
                        for row in schema_rows
                    ]
                    result["newest_training_schema_hash"] = str(schema_rows[0]["feature_schema_hash"])
                    result["newest_training_schema_samples"] = int(schema_rows[0]["cnt"])
            except Exception as _schema_exc:
                log.debug("trade_journal.schema_distribution_failed", error=str(_schema_exc))

            rows = await self._fetch(
                """
                SELECT status, model_version, sample_count, error, metrics, started_at, finished_at
                FROM training_runs
                ORDER BY started_at DESC
                LIMIT 1
                """
            )
            if rows:
                result["latest_training_run"] = dict(rows[0])
            rows = await self._fetch(
                """
                SELECT version, status, training_samples, metrics, training_finished_at, created_at
                FROM model_versions
                WHERE artifact IS NOT NULL
                ORDER BY training_finished_at DESC NULLS LAST, created_at DESC
                LIMIT 1
                """
            )
            if rows:
                result["latest_model_version"] = dict(rows[0])

            # Active model = latest CHAMPION; fallback to latest_model_version
            champion_rows = await self._fetch(
                """
                SELECT version, status, training_samples, metrics, training_finished_at, created_at
                FROM model_versions
                WHERE status = 'CHAMPION'
                  AND artifact IS NOT NULL
                ORDER BY training_finished_at DESC NULLS LAST, created_at DESC
                LIMIT 1
                """
            )
            if champion_rows:
                result["active_model_version"] = dict(champion_rows[0])
            else:
                result["active_model_version"] = result.get("latest_model_version", {})

            # Use active_model_version for gate/paper stats (not latest which may be challenger)
            active_model_row = champion_rows[0] if champion_rows else (rows[0] if rows else None)
            latest_model_version = str(active_model_row["version"]) if active_model_row else ""
            if latest_model_version:
                label_schema = _query_label_schema_version()
                gate_rows = await self._fetch(
                    """
                    SELECT
                        pe.decision,
                        count(*) AS cnt,
                        avg(po.net_return_bps) AS avg_net_return_bps,
                        avg(po.label::double precision) AS precision
                    FROM prediction_events pe
                    JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                    WHERE pe.model_version = $1
                      AND pe.decision IN ('GATE_PASS', 'GATE_BLOCK')
                      AND po.horizon_minutes = 15
                      AND po.label IS NOT NULL
                      AND po.label_schema_version = $2
                    GROUP BY pe.decision
                    """,
                    latest_model_version,
                    label_schema,
                )
                gate: dict[str, Any] = {"model_version": latest_model_version}
                total_count = 0
                weighted_return = 0.0
                for row in gate_rows:
                    decision = str(row["decision"])
                    count = int(row["cnt"])
                    avg_return = float(row["avg_net_return_bps"] or 0.0)
                    raw_precision = row["precision"]
                    precision = float(raw_precision) if raw_precision is not None else None
                    total_count += count
                    weighted_return += avg_return * count
                    key = "pass" if decision == "GATE_PASS" else "block"
                    gate[f"{key}_count"] = count
                    gate[f"{key}_avg_net_return_bps"] = avg_return
                    gate[f"{key}_precision"] = precision
                if total_count:
                    all_avg = weighted_return / total_count
                    pass_avg = gate.get("pass_avg_net_return_bps")
                    block_avg = gate.get("block_avg_net_return_bps")
                    gate["total_count"] = total_count
                    gate["all_avg_net_return_bps"] = all_avg
                    gate["lift_vs_all_bps"] = (float(pass_avg) - all_avg) if pass_avg is not None else None
                    gate["pass_vs_block_bps"] = (
                        (float(pass_avg) - float(block_avg)) if pass_avg is not None and block_avg is not None else None
                    )
                result["shadow_gate_15m"] = gate
                reason_rows = await self._fetch(
                    """
                    SELECT
                        COALESCE(pe.metadata->>'gate_reason', 'unknown') AS reason,
                        count(*) AS cnt,
                        avg(po.net_return_bps) AS avg_net_return_bps
                    FROM prediction_events pe
                    JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                    WHERE pe.model_version = $1
                      AND pe.decision = 'GATE_BLOCK'
                      AND po.horizon_minutes = 15
                      AND po.label IS NOT NULL
                    GROUP BY reason
                    ORDER BY cnt DESC
                    LIMIT 3
                    """,
                    latest_model_version,
                )
                if reason_rows:
                    gate["top_block_reasons"] = {str(row["reason"]): int(row["cnt"]) for row in reason_rows}
                    side_filtered_count = 0
                    score_block_count = 0
                    score_block_weighted_return = 0.0
                    for row in reason_rows:
                        reason = str(row["reason"])
                        count = int(row["cnt"] or 0)
                        avg_return = float(row.get("avg_net_return_bps") or 0.0)
                        if reason == "side_not_selected_by_model":
                            side_filtered_count += count
                        else:
                            score_block_count += count
                            score_block_weighted_return += avg_return * count
                    gate["side_filtered_count"] = side_filtered_count
                    gate["score_block_count"] = score_block_count
                    gate["score_block_avg_net_return_bps"] = (
                        score_block_weighted_return / score_block_count if score_block_count else None
                    )
                paper_baseline_rows = await self._fetch(
                    """
                    SELECT po.net_return_bps
                    FROM prediction_events pe
                    JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                    WHERE po.horizon_minutes = 15
                      AND po.label IS NOT NULL
                      AND po.label_schema_version = $1
                      AND pe.model_version = 'RULE_BASELINE_V1'
                    ORDER BY pe.created_at ASC
                    LIMIT 1000
                    """,
                    label_schema,
                )
                paper_gate_rows = await self._fetch(
                    """
                    SELECT po.net_return_bps
                    FROM prediction_events pe
                    JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                    WHERE po.horizon_minutes = 15
                      AND po.label IS NOT NULL
                      AND po.label_schema_version = $1
                      AND pe.model_version = $2
                      AND pe.decision = 'GATE_PASS'
                    ORDER BY pe.created_at ASC
                    LIMIT 1000
                    """,
                    label_schema,
                    latest_model_version,
                )

                result["paper_pnl_15m"] = {
                    "model_version": latest_model_version,
                    "baseline": _paper_stats_from_rows(paper_baseline_rows),
                    "model_gate": _paper_stats_from_rows(paper_gate_rows),
                }
            result["storage_stats"] = await _read_or_default("storage_stats", {}, self.get_storage_stats)
        except Exception as exc:
            self._last_read_error_at = datetime.now(tz=UTC)
            self._last_read_error = str(exc)
            result["last_read_error"] = self._last_read_error
            result["last_read_error_at"] = self._last_read_error_at
            log.debug("trade_journal.diagnostics_failed", error=str(exc))
        return result

    async def record_closed_pnl_records(self, records: Iterable[dict[str, Any]]) -> int:
        inserted = 0
        for record in records:
            symbol = str(record.get("symbol") or "").upper()
            pnl = _decimal_or_none(record.get("closedPnl"))
            if not symbol or pnl is None:
                continue
            raw = json.dumps(record)
            closed_pnl_id = self._closed_pnl_id(record)
            result = await self._execute(
                """
                INSERT INTO closed_pnl (
                    closed_pnl_id, created_at, symbol, side, qty, avg_entry_price,
                    avg_exit_price, closed_pnl, raw
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
                ON CONFLICT (closed_pnl_id) DO NOTHING
                """,
                closed_pnl_id,
                _dt_from_ms(record.get("updatedTime") or record.get("createdTime")),
                symbol,
                record.get("side"),
                _decimal_or_none(record.get("qty") or record.get("closedSize")),
                _decimal_or_none(record.get("avgEntryPrice")),
                _decimal_or_none(record.get("avgExitPrice")),
                pnl,
                raw,
            )
            inserted += _parse_command_rowcount(result)
        return inserted

    async def get_blocked_symbols(
        self,
        *,
        min_closed_trades: int,
        max_loss_usd: Decimal,
        lookback_days: int,
    ) -> set[str]:
        rows = await self._fetch(
            """
            SELECT symbol
            FROM closed_pnl
            WHERE created_at >= now() - ($1 * interval '1 day')
            GROUP BY symbol
            HAVING count(*) >= $2 AND sum(closed_pnl) <= $3
            """,
            lookback_days,
            min_closed_trades,
            max_loss_usd,
        )
        return {str(row["symbol"]) for row in rows}

    async def log_promotion_event(
        self,
        *,
        event_type: str,
        decision: str,
        challenger_version: str | None = None,
        champion_version: str | None = None,
        new_champion_version: str | None = None,
        reasons: list[str] | None = None,
        metrics_snapshot: dict[str, Any] | None = None,
    ) -> None:
        import json as _json

        await self._execute(
            """
            INSERT INTO model_promotion_log
                (event_type, decision, challenger_version, champion_version,
                 new_champion_version, from_version, to_version, reasons, metrics_snapshot)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb)
            """,
            event_type,
            decision,
            challenger_version,
            champion_version,
            new_champion_version,
            champion_version,
            new_champion_version or challenger_version,
            _json.dumps(reasons or []),
            _json.dumps(metrics_snapshot or {}),
        )

    async def promote_challenger_to_champion(
        self,
        version: str,
        *,
        event_data: dict[str, Any] | None = None,
    ) -> None:
        """Atomically archive the current champion and promote the challenger."""
        if not self.is_enabled:
            return
        if self._pool is None:
            raise RuntimeError("promote_challenger_to_champion called with no DB pool")
        import json as _json

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock($1)", _MODEL_PROMOTION_ADVISORY_LOCK_ID)
                prev = await conn.fetchrow("SELECT version FROM model_versions WHERE status = 'CHAMPION' LIMIT 1")
                prev_version = prev["version"] if prev else None
                await conn.execute("UPDATE model_versions SET status = 'ARCHIVED' WHERE status = 'CHAMPION'")
                promoted = await conn.execute(
                    """
                    UPDATE model_versions
                    SET status = 'CHAMPION'
                    WHERE version = $1 AND status IN ('SHADOW_CHALLENGER', 'VALIDATED')
                    """,
                    version,
                )
                if not str(promoted).endswith(" 1"):
                    raise RuntimeError(f"promotion_failed: model {version!r} was not an eligible challenger")
                await conn.execute(
                    """
                    INSERT INTO model_promotion_log
                        (event_type, decision, challenger_version, champion_version,
                         new_champion_version, from_version, to_version, reasons, metrics_snapshot)
                    VALUES ('PROMOTION', 'APPROVED', $1, $2, $1, $2, $1, '[]'::jsonb, $3::jsonb)
                    """,
                    version,
                    prev_version,
                    _json.dumps(event_data or {}),
                )
        log.info(
            "trade_journal.promote_challenger_to_champion",
            version=version,
            prev_champion=prev_version,
        )

    async def rollback_champion(
        self,
        *,
        current_version: str,
        reason: str,
        event_data: dict[str, Any] | None = None,
    ) -> str | None:
        """Find the most recent ARCHIVED model, restore it as CHAMPION, roll back current.

        Returns the restored version string, or None if no candidate exists.
        """
        if not self.is_enabled:
            return None
        assert self._pool is not None
        import json as _json

        min_paper_gate_count, _min_wf_positive_folds, _max_wf_std_bps = _champion_selection_thresholds()
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock($1)", _MODEL_PROMOTION_ADVISORY_LOCK_ID)
                restore = await conn.fetchrow(
                    """
                    SELECT version FROM model_versions
                    WHERE status = 'ARCHIVED'
                      AND artifact IS NOT NULL
                      AND COALESCE(metrics->>'label_schema_version', '') = $1
                      AND COALESCE(
                            NULLIF(metrics->>'walk_forward_expectancy_bps', ''),
                            NULLIF(metrics->>'walk_forward_bps', ''),
                            NULLIF(metrics->>'wf_mean_bps', ''),
                            NULLIF(metrics->>'best_threshold_avg_net_return_bps', '')
                          ) IS NOT NULL
                      AND COALESCE(
                            NULLIF(metrics #>> '{paper_gate,count}', ''),
                            NULLIF(metrics->>'paper_gate_count', ''),
                            NULLIF(metrics->>'total_pass_count', ''),
                            '0'
                          )::integer >= $2
                    ORDER BY
                        CASE
                            WHEN COALESCE(
                                NULLIF(metrics->>'walk_forward_expectancy_bps', ''),
                                NULLIF(metrics->>'walk_forward_bps', ''),
                                NULLIF(metrics->>'wf_mean_bps', ''),
                                NULLIF(metrics->>'best_threshold_avg_net_return_bps', '')
                            )::double precision > 0 THEN 0
                            ELSE 1
                        END ASC,
                        CASE
                            WHEN COALESCE(
                                NULLIF(metrics->>'walk_forward_expectancy_bps', ''),
                                NULLIF(metrics->>'walk_forward_bps', ''),
                                NULLIF(metrics->>'wf_mean_bps', ''),
                                NULLIF(metrics->>'best_threshold_avg_net_return_bps', '')
                            )::double precision > 0 THEN COALESCE(NULLIF(metrics->>'lift_bps', ''), '0')::double precision
                            ELSE NULL
                        END DESC NULLS LAST,
                        COALESCE(
                            NULLIF(metrics->>'walk_forward_expectancy_bps', ''),
                            NULLIF(metrics->>'walk_forward_bps', ''),
                            NULLIF(metrics->>'wf_mean_bps', ''),
                            NULLIF(metrics->>'best_threshold_avg_net_return_bps', '')
                        )::double precision DESC,
                        COALESCE(NULLIF(metrics->>'lift_bps', ''), '0')::double precision DESC,
                        training_finished_at DESC NULLS LAST,
                        created_at DESC
                    LIMIT 1
                    """,
                    LABEL_SCHEMA_VERSION,
                    min_paper_gate_count,
                )
                if restore is None:
                    log.warning(
                        "trade_journal.rollback_no_candidate",
                        current_version=current_version,
                    )
                    return None
                restore_version: str = restore["version"]
                await conn.execute(
                    "UPDATE model_versions SET status = 'ROLLED_BACK' WHERE version = $1",
                    current_version,
                )
                await conn.execute(
                    "UPDATE model_versions SET status = 'CHAMPION' WHERE version = $1",
                    restore_version,
                )
                await conn.execute(
                    """
                    INSERT INTO model_promotion_log
                        (event_type, decision, champion_version, new_champion_version,
                         from_version, to_version, reasons, metrics_snapshot)
                    VALUES ('ROLLBACK', 'AUTO', $1, $2, $1, $2, $3::jsonb, $4::jsonb)
                    """,
                    current_version,
                    restore_version,
                    _json.dumps([reason]),
                    _json.dumps(event_data or {}),
                )
        log.warning(
            "trade_journal.rollback_champion",
            rolled_back=current_version,
            restored=restore_version,
            reason=reason,
        )
        return restore_version

    async def _execute(self, query: str, *args: Any) -> str | None:
        if not self.is_enabled:
            return None
        assert self._pool is not None
        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(query, *args)
            self._last_successful_write_at = datetime.now(tz=UTC)
            self._consecutive_write_errors = 0
            self._last_write_error = None
            self._last_write_error_at = None
            return str(result) if result is not None else None
        except Exception as exc:
            self._last_write_error_at = datetime.now(tz=UTC)
            self._last_write_error = str(exc)
            self._consecutive_write_errors += 1
            # Escalate to WARNING after the first failure so fill/PnL write
            # errors are visible in production logs, not buried at DEBUG.
            _log_fn = log.warning if self._consecutive_write_errors >= 1 else log.debug
            _log_fn(
                "trade_journal.write_failed",
                error=str(exc),
                consecutive_errors=self._consecutive_write_errors,
            )
            return None

    async def _execute_required(self, query: str, *args: Any) -> None:
        """Fail-closed execute — raises on DB error.

        Use only for writes that MUST succeed before a REST call is made.
        If the journal is disabled, this is a no-op.
        """
        if not self.is_enabled:
            return
        assert self._pool is not None
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(query, *args)
            self._last_successful_write_at = datetime.now(tz=UTC)
            self._consecutive_write_errors = 0
            self._last_write_error = None
            self._last_write_error_at = None
        except Exception as exc:
            self._last_write_error_at = datetime.now(tz=UTC)
            self._last_write_error = str(exc)
            self._consecutive_write_errors += 1
            raise

    async def _fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        if not self.is_enabled:
            return []
        assert self._pool is not None
        try:
            async with self._pool.acquire() as conn:
                return list(
                    await asyncio.wait_for(
                        conn.fetch(query, *args),
                        timeout=self._fetch_timeout_seconds,
                    )
                )
        except TimeoutError:
            self._last_read_error_at = datetime.now(tz=UTC)
            self._last_read_error = f"query timeout after {self._fetch_timeout_seconds}s"
            now = datetime.now(tz=UTC)
            if (
                self._last_fetch_timeout_log_at is None
                or (now - self._last_fetch_timeout_log_at).total_seconds() >= _FETCH_TIMEOUT_LOG_INTERVAL_SECONDS
            ):
                self._last_fetch_timeout_log_at = now
                log.warning("trade_journal.fetch_timeout", timeout_s=self._fetch_timeout_seconds)
            return []
        except Exception as exc:
            self._last_read_error_at = datetime.now(tz=UTC)
            self._last_read_error = str(exc)
            log.debug("trade_journal.fetch_failed", error=str(exc))
            return []

    def _closed_pnl_id(self, record: dict[str, Any]) -> str:
        stable = "|".join(
            str(record.get(key, "")) for key in ("symbol", "orderId", "updatedTime", "createdTime", "closedPnl")
        )
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()
