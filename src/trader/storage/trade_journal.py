"""Postgres-backed trading memory for signals, decisions, orders, and PnL."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
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
                feature_values jsonb NOT NULL
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
                metadata jsonb
            );
            ALTER TABLE prediction_events
                ADD COLUMN IF NOT EXISTS metadata jsonb;
            CREATE INDEX IF NOT EXISTS idx_prediction_events_symbol_time
                ON prediction_events (symbol, created_at DESC);

            -- Outcome labels for training (resolved after horizon_minutes)
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
            """)

    async def record_signal(
        self,
        proposal: TradeProposal,
        feature_vector: FeatureVector | None,
        regime_context: RegimeContext | None,
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
            requested_notional = proposal.entry_price * proposal.requested_qty
        await self._execute(
            """
            INSERT INTO trade_signals (
                proposal_id, created_at, strategy_id, symbol, side, confidence,
                entry_price, take_profit, stop_loss, requested_qty,
                requested_notional_usd, regime, rationale, features
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb)
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
                error = EXCLUDED.error,
                created_at = EXCLUDED.created_at
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
                error = EXCLUDED.error,
                created_at = EXCLUDED.created_at
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

    async def record_transaction_log_entries(self, entries: list[dict]) -> int:
        """Persist Bybit transaction log entries. Returns count inserted."""
        if not self.is_enabled or not entries:
            return 0
        inserted = 0
        for entry in entries:
            try:
                trade_id = entry.get("tradeId") or None
                await self._execute(
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
                inserted += 1
            except Exception as exc:
                log.info("trade_journal.transaction_log_insert_failed", error=str(exc))
        return inserted

    async def load_pending_from_db(self) -> list[str]:
        """Return order_link_ids with non-terminal status (CREATED_LOCAL or SUBMITTING).

        Called at startup to restore in-flight entry slots.
        """
        rows = await self._fetch(
            """
            SELECT order_link_id
            FROM order_events
            WHERE status IN ('CREATED_LOCAL', 'SUBMITTING')
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

        Sign convention: profit positive, loss negative, fees negative, funding as-is.
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
            day_utc = date.today()  # UTC

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
            taker_pct = (taker_count / total_fills * 100.0) if total_fills > 0 else 100.0

            # Net PnL = gross + fees (fees are negative from Bybit) + funding
            net_pnl = gross_pnl + total_fees + total_funding

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
            ORDER BY created_at DESC
            """
        )
        return [dict(r) for r in rows]

    async def find_order_link_id_by_exchange_order_id(
        self,
        exchange_order_id: str,
    ) -> str | None:
        """Reverse lookup: find order_link_id by exchange_order_id.

        Searches durable_order_state first (authoritative), then order_events as fallback.
        Returns None if not found.
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
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                exchange_order_id,
            )
            if rows:
                return str(rows[0]["order_link_id"])

            # Fallback to order_events
            rows = await self._fetch(
                """
                SELECT order_link_id
                FROM order_events
                WHERE exchange_order_id = $1
                ORDER BY created_at DESC
                LIMIT 1
                """,
                exchange_order_id,
            )
            if rows:
                return str(rows[0]["order_link_id"])

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
        """Return {interval: count} for market_candles (diagnostics)."""
        rows = await self._fetch("SELECT interval, count(*) AS cnt FROM market_candles GROUP BY interval")
        return {str(r["interval"]): int(r["cnt"]) for r in rows}

    async def get_latest_candle_time(self, interval: str = "1") -> datetime | None:
        """Return the most recent open_time for the given interval."""
        rows = await self._fetch(
            "SELECT MAX(open_time) AS ts FROM market_candles WHERE interval = $1",
            interval,
        )
        if rows and rows[0]["ts"]:
            return rows[0]["ts"]
        return None

    async def apply_candle_retention(self) -> None:
        """Delete old candles according to retention policy."""
        retention = {"1": 30, "5": 180, "15": 365, "60": 730}
        for interval, days in retention.items():
            await self._execute(
                "DELETE FROM market_candles WHERE interval = $1 AND open_time < now() - ($2::text || ' days')::interval",
                interval,
                str(days),
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
        rows = await self._fetch(
            """
            INSERT INTO feature_snapshots (
                symbol, interval, candle_open_time,
                feature_schema_hash, feature_names, feature_values
            )
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb)
            RETURNING snapshot_id
            """,
            symbol,
            interval,
            candle_open_time,
            feature_schema_hash,
            json.dumps(feature_names),
            json.dumps(feature_values),
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
    ) -> str:
        """Write a shadow-mode prediction event; return prediction_id."""
        rows = await self._fetch(
            """
            INSERT INTO prediction_events (
                symbol, interval, model_version, feature_snapshot_id,
                score, strategy_signal, decision, metadata
            )
            VALUES ($1, $2, $3, $4::uuid, $5, $6, $7, $8::jsonb)
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
        )
        return str(rows[0]["prediction_id"]) if rows else ""

    async def resolve_prediction_outcomes(
        self,
        *,
        prediction_id: str,
        horizon_minutes: int,
        net_return_bps: float,
        max_favorable_excursion_bps: float,
        max_adverse_excursion_bps: float,
        label: int,
    ) -> None:
        """Write or update outcome label for a prediction."""
        await self._execute(
            """
            INSERT INTO prediction_outcomes (
                prediction_id, horizon_minutes, net_return_bps,
                max_favorable_excursion_bps, max_adverse_excursion_bps,
                label, resolved_at
            )
            VALUES ($1::uuid, $2, $3, $4, $5, $6, now())
            ON CONFLICT (prediction_id, horizon_minutes) DO UPDATE SET
                net_return_bps = EXCLUDED.net_return_bps,
                label = EXCLUDED.label,
                resolved_at = now()
            """,
            prediction_id,
            horizon_minutes,
            net_return_bps,
            max_favorable_excursion_bps,
            max_adverse_excursion_bps,
            label,
        )

    async def resolve_outcomes_from_candles(
        self,
        *,
        horizon_minutes: int,
        label_bps_threshold: float = 5.0,
        limit: int = 200,
    ) -> int:
        """Resolve prediction outcomes using market_candles data.

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
                fs.candle_open_time AS entry_time
            FROM prediction_events pe
            JOIN feature_snapshots fs ON fs.snapshot_id = pe.feature_snapshot_id
            LEFT JOIN prediction_outcomes po
                ON po.prediction_id = pe.prediction_id
                AND po.horizon_minutes = $1
            WHERE po.prediction_id IS NULL
              AND pe.created_at < now() - ($1 * interval '1 minute')
              AND pe.feature_snapshot_id IS NOT NULL
            LIMIT $2
            """,
            horizon_minutes,
            limit,
        )

        resolved = 0
        for row in rows:
            prediction_id = str(row["prediction_id"])
            symbol = row["symbol"]
            entry_time = row["entry_time"]

            entry_rows = await self._fetch(
                """
                SELECT close FROM market_candles
                WHERE symbol=$1 AND interval='1' AND open_time <= $2
                ORDER BY open_time DESC LIMIT 1
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
            horizon_rows = await self._fetch(
                """
                SELECT close, high, low FROM market_candles
                WHERE symbol=$1 AND interval='1' AND open_time <= $2
                ORDER BY open_time DESC LIMIT 1
                """,
                symbol,
                horizon_time,
            )
            if not horizon_rows:
                continue
            horizon_close = float(horizon_rows[0]["close"])
            horizon_high = float(horizon_rows[0]["high"])
            horizon_low = float(horizon_rows[0]["low"])

            net_return_bps = (horizon_close - entry_close) / entry_close * 10_000
            max_fav = (horizon_high - entry_close) / entry_close * 10_000
            max_adv = (horizon_low - entry_close) / entry_close * 10_000
            label = 1 if net_return_bps > label_bps_threshold else 0

            await self.resolve_prediction_outcomes(
                prediction_id=prediction_id,
                horizon_minutes=horizon_minutes,
                net_return_bps=net_return_bps,
                max_favorable_excursion_bps=max_fav,
                max_adverse_excursion_bps=max_adv,
                label=label,
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

    async def get_shadow_gate_stats(
        self,
        model_version: str,
        horizon_minutes: int,
        label_schema_version: str,
    ) -> dict[str, Any]:
        """Return shadow gate statistics for a specific model version.

        Filters by exact model_version, horizon_minutes, and label_schema_version.
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
                  AND pe.decision IN ('GATE_PASS', 'GATE_BLOCK')
                  AND fs.feature_values IS NOT NULL
                GROUP BY pe.decision
                """,
                model_version,
                horizon_minutes,
            )

            gate: dict[str, Any] = {"model_version": model_version}
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

    async def get_db_diagnostics(self) -> dict[str, Any]:
        """Return read-only diagnostics for Telegram 🗄 screen."""
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
                  AND fs.training_eligible = true
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
                    GROUP BY pe.decision
                    """,
                    latest_model_version,
                )
                gate: dict[str, Any] = {"model_version": latest_model_version}
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
                    latest_model_version,
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
                    """,
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
                    latest_model_version,
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
                    "model_version": latest_model_version,
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

    async def record_closed_pnl_records(self, records: Iterable[dict[str, Any]]) -> int:
        inserted = 0
        for record in records:
            symbol = str(record.get("symbol") or "").upper()
            pnl = _decimal_or_none(record.get("closedPnl"))
            if not symbol or pnl is None:
                continue
            raw = json.dumps(record)
            closed_pnl_id = self._closed_pnl_id(record)
            await self._execute(
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
            inserted += 1
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

    async def _execute(self, query: str, *args: Any) -> None:
        if not self.is_enabled:
            return
        assert self._pool is not None
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(query, *args)
            self._last_successful_write_at = datetime.now(tz=UTC)
            self._consecutive_write_errors = 0
        except Exception as exc:
            self._last_write_error_at = datetime.now(tz=UTC)
            self._last_write_error = str(exc)
            self._consecutive_write_errors += 1
            log.debug(
                "trade_journal.write_failed",
                error=str(exc),
                consecutive_errors=self._consecutive_write_errors,
            )

    async def _execute_required(self, query: str, *args: Any) -> None:
        """Fail-closed execute — raises on DB error.

        Use only for writes that MUST succeed before a REST call is made.
        If the journal is disabled, this is a no-op.
        """
        if not self.is_enabled:
            return
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(query, *args)

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

    def _closed_pnl_id(self, record: dict[str, Any]) -> str:
        stable = "|".join(
            str(record.get(key, "")) for key in ("symbol", "orderId", "updatedTime", "createdTime", "closedPnl")
        )
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()
