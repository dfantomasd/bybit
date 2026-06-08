"""Postgres-backed trading memory for signals, decisions, orders, and PnL."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import asyncpg
import structlog

from trader.domain.models import FeatureVector, RegimeContext, RiskDecision, TradeProposal

log = structlog.get_logger(__name__)


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

    def __init__(self, postgres_dsn: str, enabled: bool = True) -> None:
        self._dsn = postgres_dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
        self._enabled = enabled and bool(postgres_dsn)
        self._pool: asyncpg.Pool | None = None
        self._last_connect_attempt_at: datetime | None = None
        self._last_connect_error_at: datetime | None = None
        self._last_connect_error: str | None = None
        self._last_read_error_at: datetime | None = None
        self._last_read_error: str | None = None
        self._last_successful_write_at: datetime | None = None
        self._last_write_error_at: datetime | None = None
        self._last_write_error: str | None = None
        self._consecutive_write_errors: int = 0

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

    def write_health(self) -> dict[str, Any]:
        """Return write-health snapshot for observability and safety gates."""
        return {
            "healthy": self.durable_state_healthy,
            "consecutive_write_errors": self._consecutive_write_errors,
            "last_successful_write_at": (
                self._last_successful_write_at.isoformat() if self._last_successful_write_at else None
            ),
            "last_write_error_at": (self._last_write_error_at.isoformat() if self._last_write_error_at else None),
            "last_write_error": getattr(self, "_last_write_error", None),
            "last_read_error_at": (
                self._last_read_error_at.isoformat() if getattr(self, "_last_read_error_at", None) else None
            ),
            "last_read_error": getattr(self, "_last_read_error", None),
            "last_connect_error_at": (
                self._last_connect_error_at.isoformat() if getattr(self, "_last_connect_error_at", None) else None
            ),
            "last_connect_error": getattr(self, "_last_connect_error", None),
        }

    async def connect(self) -> None:
        if not self._enabled or self._pool is not None:
            return
        self._last_connect_attempt_at = datetime.now(tz=UTC)
        try:
            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=1,
                max_size=3,
                statement_cache_size=0,
            )
            await self._ensure_schema()
            self._last_connect_error_at = None
            self._last_connect_error = None
            log.info("trade_journal.connected")
        except Exception as exc:
            if self._pool is not None:
                try:
                    await self._pool.close()
                except Exception as close_exc:
                    log.debug("trade_journal.failed_pool_close_failed", error=str(close_exc))
            self._pool = None
            self._last_connect_error_at = datetime.now(tz=UTC)
            self._last_connect_error = str(exc)
            log.warning("trade_journal.unavailable", error=str(exc))

    async def reconnect_if_needed(self, *, min_interval: float = 30.0, force: bool = False) -> bool:
        """Try to reconnect after transient startup/network failures."""
        if not self._enabled:
            return False
        if self._pool is not None:
            return True
        if not force and self._last_connect_attempt_at is not None:
            age = datetime.now(tz=UTC) - self._last_connect_attempt_at
            if age < timedelta(seconds=min_interval):
                return False
        await self.connect()
        return self.is_enabled

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _ensure_schema(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
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

            CREATE TABLE IF NOT EXISTS feature_snapshots (
                snapshot_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                created_at timestamptz NOT NULL DEFAULT now(),
                symbol text NOT NULL,
                interval text NOT NULL,
                candle_open_time timestamptz NOT NULL,
                feature_schema_hash text NOT NULL,
                feature_names jsonb NOT NULL,
                feature_values jsonb NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_feature_snapshots_symbol_time
                ON feature_snapshots (symbol, created_at DESC);

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
                metadata jsonb
            );
            ALTER TABLE prediction_events
                ADD COLUMN IF NOT EXISTS metadata jsonb;
            CREATE INDEX IF NOT EXISTS idx_prediction_events_symbol_time
                ON prediction_events (symbol, created_at DESC);

            CREATE TABLE IF NOT EXISTS prediction_outcomes (
                prediction_id uuid NOT NULL REFERENCES prediction_events(prediction_id),
                horizon_minutes integer NOT NULL,
                net_return_bps double precision,
                max_favorable_excursion_bps double precision,
                max_adverse_excursion_bps double precision,
                label integer,
                resolved_at timestamptz,
                PRIMARY KEY (prediction_id, horizon_minutes)
            );

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
            await conn.execute("""
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
            """)

    async def get_db_diagnostics(self) -> dict[str, Any]:
        """Return read-only diagnostics for Telegram 🗄 screen.

        IMPORTANT: readiness statistics are computed from the active CHAMPION model.
        The newest trained model may be a SHADOW_CHALLENGER and must not reset
        live/canary readiness after a restart or re-training cycle.
        """
        result: dict[str, Any] = {
            "connected": self.is_enabled,
            "configured": self.is_configured,
            "last_connect_error": self._last_connect_error,
            "last_connect_error_at": self._last_connect_error_at,
            "last_read_error": self._last_read_error,
            "last_read_error_at": self._last_read_error_at,
            "write_health": self.write_health(),
            "candles_by_interval": {},
            "latest_candle_1m": None,
            "feature_snapshots": 0,
            "prediction_outcomes": 0,
            "prediction_outcomes_by_horizon": {},
            "labelled_samples_15m": 0,
            "latest_training_run": {},
            "latest_model_version": {},
            "active_model_version": {},
            "shadow_gate_15m": {},
            "paper_pnl_15m": {},
        }
        if not self.is_enabled:
            await self.reconnect_if_needed()
            result["connected"] = self.is_enabled
            result["last_connect_error"] = self._last_connect_error
            result["last_connect_error_at"] = self._last_connect_error_at
            result["write_health"] = self.write_health()
        if not self.is_enabled:
            return result
        try:
            self._last_read_error = None
            self._last_read_error_at = None
            result["candles_by_interval"] = await self.get_candle_counts()
            result["latest_candle_1m"] = await self.get_latest_candle_time("1")
            rows = await self._fetch("SELECT count(*) AS cnt FROM feature_snapshots")
            result["feature_snapshots"] = int(rows[0]["cnt"]) if rows else 0
            rows = await self._fetch("SELECT count(*) AS cnt FROM prediction_outcomes")
            result["prediction_outcomes"] = int(rows[0]["cnt"]) if rows else 0
            rows = await self._fetch(
                """
                SELECT horizon_minutes, count(*) AS cnt
                FROM prediction_outcomes
                WHERE label IS NOT NULL
                GROUP BY horizon_minutes
                ORDER BY horizon_minutes
                """
            )
            result["prediction_outcomes_by_horizon"] = {str(row["horizon_minutes"]): int(row["cnt"]) for row in rows}
            rows = await self._fetch(
                """
                SELECT count(DISTINCT fs.snapshot_id) AS cnt
                FROM feature_snapshots fs
                JOIN prediction_events pe ON pe.feature_snapshot_id = fs.snapshot_id
                JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                WHERE po.horizon_minutes = 15
                  AND po.label IS NOT NULL
                  AND fs.feature_values IS NOT NULL
                  AND pe.model_version = 'RULE_BASELINE_V1'
                """
            )
            result["labelled_samples_15m"] = int(rows[0]["cnt"]) if rows else 0
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

            latest_rows = await self._fetch(
                """
                SELECT version, status, training_samples, metrics, training_finished_at, created_at
                FROM model_versions
                WHERE artifact IS NOT NULL
                ORDER BY training_finished_at DESC NULLS LAST, created_at DESC
                LIMIT 1
                """
            )
            if latest_rows:
                result["latest_model_version"] = dict(latest_rows[0])

            active_rows = await self._fetch(
                """
                SELECT version, status, training_samples, metrics, training_finished_at, created_at
                FROM model_versions
                WHERE artifact IS NOT NULL AND status = 'CHAMPION'
                ORDER BY training_finished_at DESC NULLS LAST, created_at DESC
                LIMIT 1
                """
            )
            if not active_rows:
                active_rows = latest_rows
            if active_rows:
                result["active_model_version"] = dict(active_rows[0])
                active_model_version = str(active_rows[0]["version"])
                gate_rows = await self._fetch(
                    """
                    SELECT pe.decision, count(*) AS cnt,
                           avg(po.net_return_bps) AS avg_net_return_bps,
                           avg(po.label::double precision) AS precision
                    FROM prediction_events pe
                    JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                    WHERE pe.model_version = $1
                      AND pe.decision IN ('GATE_PASS', 'GATE_BLOCK')
                      AND po.horizon_minutes = 15
                      AND po.label IS NOT NULL
                    GROUP BY pe.decision
                    """,
                    active_model_version,
                )
                gate: dict[str, Any] = {"model_version": active_model_version}
                total_count = 0
                weighted_return = 0.0
                for row in gate_rows:
                    decision = str(row["decision"])
                    count = int(row["cnt"])
                    avg_return = float(row["avg_net_return_bps"] or 0.0)
                    precision = float(row["precision"] or 0.0)
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
                    SELECT COALESCE(pe.metadata->>'gate_reason', 'unknown') AS reason, count(*) AS cnt
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
                    active_model_version,
                )
                if reason_rows:
                    gate["top_block_reasons"] = {str(row["reason"]): int(row["cnt"]) for row in reason_rows}
                paper_baseline_rows = await self._fetch(
                    """
                    SELECT po.net_return_bps
                    FROM prediction_events pe
                    JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                    WHERE po.horizon_minutes = 15
                      AND po.label IS NOT NULL
                      AND pe.model_version = 'RULE_BASELINE_V1'
                    ORDER BY pe.created_at ASC
                    LIMIT 1000
                    """
                )
                paper_gate_rows = await self._fetch(
                    """
                    SELECT po.net_return_bps
                    FROM prediction_events pe
                    JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                    WHERE po.horizon_minutes = 15
                      AND po.label IS NOT NULL
                      AND pe.model_version = $1
                      AND pe.decision = 'GATE_PASS'
                    ORDER BY pe.created_at ASC
                    LIMIT 1000
                    """,
                    active_model_version,
                )

                def _paper_stats(rows: list[Any]) -> dict[str, Any]:
                    returns = [float(row["net_return_bps"] or 0.0) for row in rows]
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

                result["paper_pnl_15m"] = {
                    "model_version": active_model_version,
                    "baseline": _paper_stats(paper_baseline_rows),
                    "model_gate": _paper_stats(paper_gate_rows),
                }
        except Exception as exc:
            self._last_read_error_at = datetime.now(tz=UTC)
            self._last_read_error = str(exc)
            result["last_read_error"] = self._last_read_error
            result["last_read_error_at"] = self._last_read_error_at
            log.debug("trade_journal.diagnostics_failed", error=str(exc))
        return result

    async def _fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        if not self.is_enabled:
            return []
        assert self._pool is not None
        try:
            async with self._pool.acquire() as conn:
                return await conn.fetch(query, *args)
        except Exception as exc:
            self._last_read_error_at = datetime.now(tz=UTC)
            self._last_read_error = str(exc)
            log.debug("trade_journal.fetch_failed", error=str(exc))
            return []

    async def get_candle_counts(self) -> dict[str, int]:
        rows = await self._fetch("SELECT interval, count(*) AS cnt FROM market_candles GROUP BY interval")
        return {str(r["interval"]): int(r["cnt"]) for r in rows}

    async def get_latest_candle_time(self, interval: str = "1") -> datetime | None:
        rows = await self._fetch("SELECT MAX(open_time) AS ts FROM market_candles WHERE interval = $1", interval)
        if rows and rows[0]["ts"]:
            return rows[0]["ts"]
        return None

    def _closed_pnl_id(self, record: dict[str, Any]) -> str:
        stable = "|".join(str(record.get(key, "")) for key in ("symbol", "orderId", "updatedTime", "createdTime", "closedPnl"))
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()
