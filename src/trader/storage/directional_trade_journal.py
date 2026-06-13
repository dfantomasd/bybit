"""Directional outcome resolver extension for :mod:`trade_journal`.

The base journal is intentionally left intact. This module overrides only the
ML-outcome parts that must change together: schema migration, outcome writes,
and candle-based resolution. Installation is explicit from ``storage`` package
initialisation so callers importing ``trader.storage.trade_journal.TradeJournal``
receive the extended implementation without a risky full-file rewrite.
"""

from __future__ import annotations

import traceback
from contextvars import ContextVar
from datetime import datetime, timedelta
from typing import Any

import structlog

from trader.domain.models import FeatureVector, RegimeContext, TradeProposal
from trader.features.source_candle_guard import source_candle_for_feature
from trader.storage import trade_journal as _base_module
from trader.storage.trade_journal import TradeJournal as _BaseTradeJournal
from trader.training.labels import LABEL_SCHEMA_VERSION, CostModelBps, build_directional_outcome

log = structlog.get_logger(__name__)

DEFAULT_TAKER_FEE_BPS = 5.5  # 0.055% taker x 2 legs = 11 bps round trip
DEFAULT_SPREAD_BPS = 8.0  # max_spread_bps from config
DEFAULT_SLIPPAGE_PER_SIDE_BPS = 3.0  # expected_slippage_pct x 2 legs = 6 bps
DEFAULT_FUNDING_BPS = 1.0  # funding_buffer_pct
DEFAULT_SAFETY_MARGIN_BPS = 5.0  # mirrors net_edge_safety_margin_pct in engine

_SourceBinding = tuple[str, str, datetime]
_CURRENT_SOURCE_BINDING: ContextVar[_SourceBinding | None] = ContextVar(
    "current_training_snapshot_source_candle",
    default=None,
)


def _normalise_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def default_cost_model() -> CostModelBps:
    """Return the conservative round-trip cost model used by the resolver.

    Values match the current production defaults: two taker fees, full spread,
    per-side slippage, and a small funding buffer. A later configuration wiring
    step may inject symbol-specific values without changing the label formula.
    """
    return CostModelBps(
        entry_fee_bps=DEFAULT_TAKER_FEE_BPS,
        exit_fee_bps=DEFAULT_TAKER_FEE_BPS,
        spread_bps=DEFAULT_SPREAD_BPS,
        entry_slippage_bps=DEFAULT_SLIPPAGE_PER_SIDE_BPS,
        exit_slippage_bps=DEFAULT_SLIPPAGE_PER_SIDE_BPS,
        funding_bps=DEFAULT_FUNDING_BPS,
        safety_margin_bps=DEFAULT_SAFETY_MARGIN_BPS,
    )


class DirectionalTradeJournal(_BaseTradeJournal):
    """Trade journal with directional, cost-aware ML outcomes."""

    async def record_signal(
        self,
        proposal: TradeProposal,
        feature_vector: FeatureVector | None,
        regime_context: RegimeContext | None,
        model_decision: dict[str, Any] | None = None,
        blocked_reason: str | None = None,
    ) -> None:
        """Persist a signal and bind its vector to the current async task."""

        binding = source_candle_for_feature(feature_vector.feature_id) if feature_vector is not None else None
        _CURRENT_SOURCE_BINDING.set(binding)
        try:
            await super().record_signal(
                proposal,
                feature_vector,
                regime_context,
                model_decision=model_decision,
                blocked_reason=blocked_reason,
            )
        except Exception:
            _CURRENT_SOURCE_BINDING.set(None)
            raise

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
        """Reject stale vector-to-candle associations before snapshot persistence."""

        binding = _CURRENT_SOURCE_BINDING.get()
        try:
            if binding is not None:
                expected_symbol, expected_interval, expected_open_time = binding
                actual = (_normalise_symbol(symbol), str(interval), candle_open_time)
                if actual != (expected_symbol, expected_interval, expected_open_time):
                    log.warning(
                        "trade_journal.feature_snapshot_source_mismatch",
                        symbol=symbol,
                        interval=interval,
                        requested_candle_open_time=candle_open_time,
                        expected_binding=binding,
                    )
                    return ""

            return await super().record_feature_snapshot(
                symbol=symbol,
                interval=interval,
                candle_open_time=candle_open_time,
                feature_schema_hash=feature_schema_hash,
                feature_names=feature_names,
                feature_values=feature_values,
            )
        finally:
            _CURRENT_SOURCE_BINDING.set(None)

    async def _ensure_schema(self) -> None:
        """Create the base schema, then migrate directional-label columns."""

        await super()._ensure_schema()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                ALTER TABLE prediction_outcomes
                    ADD COLUMN IF NOT EXISTS gross_return_bps double precision,
                    ADD COLUMN IF NOT EXISTS cost_bps double precision,
                    ADD COLUMN IF NOT EXISTS label_threshold_bps double precision,
                    ADD COLUMN IF NOT EXISTS label_schema_version text;
                CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_schema_horizon
                    ON prediction_outcomes (label_schema_version, horizon_minutes);
                """
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
        """Write or replace one versioned directional outcome."""

        await self._execute(
            """
            INSERT INTO prediction_outcomes (
                prediction_id, horizon_minutes, gross_return_bps, cost_bps,
                label_threshold_bps, net_return_bps,
                max_favorable_excursion_bps, max_adverse_excursion_bps,
                label, label_schema_version, resolved_at
            )
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, now())
            ON CONFLICT (prediction_id, horizon_minutes) DO UPDATE SET
                gross_return_bps = EXCLUDED.gross_return_bps,
                cost_bps = EXCLUDED.cost_bps,
                label_threshold_bps = EXCLUDED.label_threshold_bps,
                net_return_bps = EXCLUDED.net_return_bps,
                max_favorable_excursion_bps = EXCLUDED.max_favorable_excursion_bps,
                max_adverse_excursion_bps = EXCLUDED.max_adverse_excursion_bps,
                label = EXCLUDED.label,
                label_schema_version = EXCLUDED.label_schema_version,
                resolved_at = now()
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
        cost_model: CostModelBps | None = None,
    ) -> int:
        """Resolve Buy and Sell outcomes from complete confirmed 1-minute paths.

        Legacy long-only outcomes are recalculated because the lookup joins only
        rows carrying the current ``LABEL_SCHEMA_VERSION``. The existing primary
        key then updates the old row in place. MFE and MAE use every confirmed
        candle inside the full horizon rather than only the final bar.
        """

        costs = cost_model or default_cost_model()
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
                AND po.label_schema_version = $3
                AND po.label_threshold_bps = $4
            WHERE po.prediction_id IS NULL
              AND pe.created_at < now() - ($1 * interval '1 minute')
              AND pe.feature_snapshot_id IS NOT NULL
              AND pe.strategy_signal IN ('Buy', 'Sell')
              AND fs.training_eligible = true
            ORDER BY pe.created_at ASC
            LIMIT $2
            """,
            horizon_minutes,
            limit,
            LABEL_SCHEMA_VERSION,
            label_bps_threshold,
        )

        resolved = 0
        for row in rows:
            prediction_id = str(row["prediction_id"])
            symbol = str(row["symbol"])
            side = str(row["strategy_signal"])
            entry_time = row["entry_time"]

            entry_rows = await self._fetch(
                """
                SELECT close
                FROM market_candles
                WHERE symbol = $1
                  AND interval = '1'
                  AND open_time = $2
                  AND confirmed = true
                LIMIT 1
                """,
                symbol,
                entry_time,
            )
            if not entry_rows:
                continue

            entry_close = float(entry_rows[0]["close"])
            if entry_close <= 0:
                continue

            horizon_time = entry_time + timedelta(minutes=horizon_minutes)
            path_rows = await self._fetch(
                """
                SELECT open_time, close, high, low
                FROM market_candles
                WHERE symbol = $1
                  AND interval = '1'
                  AND open_time > $2
                  AND open_time <= $3
                  AND confirmed = true
                ORDER BY open_time ASC
                """,
                symbol,
                entry_time,
                horizon_time,
            )
            if len(path_rows) != horizon_minutes:
                continue
            if path_rows[-1]["open_time"] != horizon_time:
                continue

            outcome = build_directional_outcome(
                side=side,
                entry_price=entry_close,
                exit_price=float(path_rows[-1]["close"]),
                highs=[float(item["high"]) for item in path_rows],
                lows=[float(item["low"]) for item in path_rows],
                cost_model=costs,
                label_threshold_bps=label_bps_threshold,
            )
            await self.resolve_prediction_outcomes(
                prediction_id=prediction_id,
                horizon_minutes=horizon_minutes,
                gross_return_bps=outcome.gross_return_bps,
                cost_bps=costs.total_bps,
                label_threshold_bps=label_bps_threshold,
                net_return_bps=outcome.net_return_bps,
                max_favorable_excursion_bps=outcome.max_favorable_excursion_bps,
                max_adverse_excursion_bps=outcome.max_adverse_excursion_bps,
                label=outcome.label,
                label_schema_version=outcome.label_schema_version,
            )
            resolved += 1

        return resolved

    async def get_db_diagnostics(self) -> dict[str, Any]:
        """Expose only current-schema labels and models to diagnostics."""

        result = await super().get_db_diagnostics()
        result["label_schema_version"] = LABEL_SCHEMA_VERSION
        if not self.is_enabled:
            return result

        try:
            return await self._get_db_diagnostics_directional(result)
        except Exception as exc:
            log.warning(
                "directional_trade_journal.diagnostics_failed",
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            result["last_read_error"] = str(exc)
            return result

    async def _get_db_diagnostics_directional(self, result: dict[str, Any]) -> dict[str, Any]:
        rows = await self._fetch(
            """
            SELECT horizon_minutes, count(*) AS cnt
            FROM prediction_outcomes
            WHERE label IS NOT NULL
              AND label_schema_version = $1
            GROUP BY horizon_minutes
            ORDER BY horizon_minutes
            """,
            LABEL_SCHEMA_VERSION,
        )
        result["prediction_outcomes_by_horizon"] = {str(row["horizon_minutes"]): int(row["cnt"]) for row in rows}
        result["prediction_outcomes"] = sum(result["prediction_outcomes_by_horizon"].values())

        # Mirror the training query's eligibility exactly (threshold, signal,
        # training_eligible, one sample per candle). Training requires the
        # minimum WITHIN ONE feature_schema_hash, so report the largest single
        # schema bucket; summing across schemas overstates progress and makes
        # the auto-trainer fire prematurely.
        rows = await self._fetch(
            """
            SELECT feature_schema_hash, count(*) AS cnt
            FROM (
                SELECT DISTINCT ON (fs.symbol, fs.interval, fs.candle_open_time, fs.feature_schema_hash)
                       fs.snapshot_id, fs.feature_schema_hash
                FROM feature_snapshots fs
                JOIN prediction_events pe ON pe.feature_snapshot_id = fs.snapshot_id
                JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                WHERE po.horizon_minutes = 15
                  AND po.label IS NOT NULL
                  AND po.label_schema_version = $1
                  AND po.label_threshold_bps = 5.0
                  AND fs.feature_values IS NOT NULL
                  AND fs.training_eligible = true
                  AND pe.model_version = 'RULE_BASELINE_V1'
                  AND pe.strategy_signal IN ('Buy', 'Sell')
            ) deduped
            GROUP BY feature_schema_hash
            ORDER BY cnt DESC
            """,
            LABEL_SCHEMA_VERSION,
        )
        result["labelled_samples_15m"] = int(rows[0]["cnt"]) if rows else 0
        current_feature_schema_hash = str(dict(rows[0]).get("feature_schema_hash") or "") if rows else ""
        result["training_eligible_schema_15m"] = {
            "feature_schema_hash": current_feature_schema_hash,
            "sample_count": result["labelled_samples_15m"],
            "horizon_minutes": 15,
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "label_threshold_bps": 5.0,
        }
        result["labelled_samples_15m_by_schema"] = {
            str(dict(row).get("feature_schema_hash", "?"))[:8]: int(row["cnt"]) for row in rows
        }
        # The auto-trainer gate reads training_eligible_15m; keep it in sync
        # with the per-schema maximum, not the base class's cross-schema union.
        result["training_eligible_15m"] = result["labelled_samples_15m"]

        rows = await self._fetch(
            """
            SELECT
                version,
                status,
                training_samples,
                metrics,
                COALESCE(metrics->>'feature_schema_hash', '') AS feature_schema_hash,
                training_finished_at,
                created_at
            FROM model_versions
            WHERE artifact IS NOT NULL
              AND COALESCE(metrics->>'label_schema_version', '') = $1
            ORDER BY training_finished_at DESC NULLS LAST, created_at DESC
            LIMIT 1
            """,
            LABEL_SCHEMA_VERSION,
        )
        if not rows:
            rows = await self._fetch(
                """
                SELECT
                    version,
                    status,
                    training_samples,
                    metrics,
                    COALESCE(metrics->>'feature_schema_hash', '') AS feature_schema_hash,
                    training_finished_at,
                    created_at
                FROM model_versions
                WHERE artifact IS NOT NULL
                ORDER BY training_finished_at DESC NULLS LAST, created_at DESC
                LIMIT 1
                """
            )
        latest_model = dict(rows[0]) if rows else {}
        latest_model_schema_hash = str(latest_model.get("feature_schema_hash") or "")
        if latest_model and current_feature_schema_hash and latest_model_schema_hash != current_feature_schema_hash:
            latest_model["actual_training_samples"] = int(latest_model.get("training_samples", 0) or 0)
            latest_model["training_samples"] = 0
            latest_model["training_samples_compatible"] = 0
            latest_model["schema_compatible"] = False
        elif latest_model:
            latest_model["actual_training_samples"] = int(latest_model.get("training_samples", 0) or 0)
            _metrics_raw = latest_model.get("metrics")
            if isinstance(_metrics_raw, str):
                try:
                    import json as _json

                    _metrics_raw = _json.loads(_metrics_raw)
                except Exception:
                    _metrics_raw = {}
            _metrics_dict = _metrics_raw if isinstance(_metrics_raw, dict) else {}
            latest_model_schema = str(_metrics_dict.get("label_schema_version") or "")
            latest_model["schema_compatible"] = latest_model_schema == LABEL_SCHEMA_VERSION
            latest_model["training_samples_compatible"] = (
                latest_model["actual_training_samples"] if latest_model["schema_compatible"] else 0
            )
        result["latest_model_version"] = latest_model

        rows = await self._fetch(
            """
            SELECT
                version,
                status,
                training_samples,
                metrics,
                COALESCE(metrics->>'feature_schema_hash', '') AS feature_schema_hash,
                training_finished_at,
                created_at
            FROM model_versions
            WHERE status = 'CHAMPION'
              AND artifact IS NOT NULL
              AND COALESCE(metrics->>'label_schema_version', '') = $1
            ORDER BY training_finished_at DESC NULLS LAST, created_at DESC
            LIMIT 1
            """,
            LABEL_SCHEMA_VERSION,
        )
        # When no CHAMPION exists, fall back to latest_model_version so that
        # diagnostics and the heartbeat can still show which model is loaded.
        result["active_model_version"] = dict(rows[0]) if rows else result.get("latest_model_version", {})

        return result


def install_directional_trade_journal() -> None:
    """Install the directional implementation for existing import paths."""

    _base_module.TradeJournal = DirectionalTradeJournal  # type: ignore[misc]
