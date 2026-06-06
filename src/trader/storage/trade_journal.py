"""Postgres-backed trading memory for signals, decisions, orders, and PnL."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
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

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._pool is not None

    async def connect(self) -> None:
        if not self._enabled:
            return
        try:
            self._pool = await asyncpg.create_pool(dsn=self._dsn, min_size=1, max_size=3)
            await self._ensure_schema()
            log.info("trade_journal.connected")
        except Exception as exc:
            self._enabled = False
            self._pool = None
            log.warning("trade_journal.disabled", error=str(exc))

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _ensure_schema(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
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
            """
            )

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
            WHERE created_at >= now() - ($1::text || ' days')::interval
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
        except Exception as exc:
            log.debug("trade_journal.write_failed", error=str(exc))

    async def _fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        if not self.is_enabled:
            return []
        assert self._pool is not None
        try:
            async with self._pool.acquire() as conn:
                return await conn.fetch(query, *args)
        except Exception as exc:
            log.debug("trade_journal.fetch_failed", error=str(exc))
            return []

    def _closed_pnl_id(self, record: dict[str, Any]) -> str:
        stable = "|".join(
            str(record.get(key, "")) for key in ("symbol", "orderId", "updatedTime", "createdTime", "closedPnl")
        )
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()
