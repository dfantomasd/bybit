"""Directional outcome resolver extension for :mod:`trade_journal`.

The base journal is intentionally left intact. This module overrides only the
ML-outcome parts that must change together: schema migration, outcome writes,
and candle-based resolution. Installation is explicit from ``storage`` package
initialisation so callers importing ``trader.storage.trade_journal.TradeJournal``
receive the extended implementation without a risky full-file rewrite.
"""

from __future__ import annotations

import json
import traceback
from contextvars import ContextVar
from datetime import datetime, timedelta
from typing import Any

import structlog

from trader.domain.models import FeatureVector, RegimeContext, TradeProposal
from trader.features.source_candle_guard import source_candle_for_feature
from trader.storage import trade_journal as _base_module
from trader.storage.trade_journal import TradeJournal as _BaseTradeJournal
from trader.training.labels import (
    LABEL_SCHEMA_VERSION,
    CostModelBps,
    active_label_schema_version,
    atr_pct_from_feature_payload,
    build_directional_outcome,
)
from trader.training.sample_counts import fetch_training_snapshots_for_horizons

log = structlog.get_logger(__name__)

DEFAULT_TAKER_FEE_BPS = 5.5  # 0.055% taker x 2 legs = 11 bps round trip
DEFAULT_SPREAD_BPS = 4.0  # aligned with scalp max spread (~5 bps), not screener wide max
DEFAULT_SLIPPAGE_PER_SIDE_BPS = 3.0  # expected_slippage_pct x 2 legs = 6 bps
DEFAULT_FUNDING_BPS = 1.0  # funding_buffer_pct
DEFAULT_SAFETY_MARGIN_BPS = 5.0  # mirrors net_edge_safety_margin_pct in engine


def _optional_settings() -> Any | None:
    try:
        from trader.config import Settings

        return Settings()
    except Exception as exc:
        log.debug("directional_trade_journal.settings_unavailable", error=str(exc))
        return None


def default_cost_model() -> CostModelBps:
    """Return the round-trip cost model used by the outcome resolver.

    Spread defaults to ``TRAIN_LABEL_SPREAD_BPS`` when settings are available so
    training labels match scalp execution conditions more closely than the legacy
    screener-wide 8 bps assumption.
    """
    spread_bps = DEFAULT_SPREAD_BPS
    settings = _optional_settings()
    if settings is not None:
        spread_bps = float(settings.TRAIN_LABEL_SPREAD_BPS)
    return CostModelBps(
        entry_fee_bps=DEFAULT_TAKER_FEE_BPS,
        exit_fee_bps=DEFAULT_TAKER_FEE_BPS,
        spread_bps=spread_bps,
        entry_slippage_bps=DEFAULT_SLIPPAGE_PER_SIDE_BPS,
        exit_slippage_bps=DEFAULT_SLIPPAGE_PER_SIDE_BPS,
        funding_bps=DEFAULT_FUNDING_BPS,
        safety_margin_bps=DEFAULT_SAFETY_MARGIN_BPS,
    )


_SourceBinding = tuple[str, str, datetime]
_CURRENT_SOURCE_BINDING: ContextVar[_SourceBinding | None] = ContextVar(
    "current_training_snapshot_source_candle",
    default=None,
)


def _normalise_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def _label_resolution_settings() -> tuple[bool, float, float, float, str]:
    """Return TP/SL label settings and the active label schema version."""
    use_tpsl = True
    tp_mult = 1.0
    sl_mult = 0.5
    label_threshold = 2.0
    settings = _optional_settings()
    if settings is not None:
        use_tpsl = bool(settings.MODEL_LABEL_USE_TPSL_EXIT)
        tp_mult = float(settings.MODEL_LABEL_TP_ATR_MULT)
        sl_mult = float(settings.MODEL_LABEL_SL_ATR_MULT)
        label_threshold = float(settings.MODEL_AUTO_TRAIN_LABEL_BPS)
    return use_tpsl, tp_mult, sl_mult, label_threshold, active_label_schema_version(use_tpsl_exit=use_tpsl)


def _training_eligibility_params() -> tuple[list[str] | None, bool, str, float]:
    """Return strategy allowlist, candle flag, label schema, and label threshold."""
    allowlist: list[str] | None = None
    include_candle = False
    label_schema = LABEL_SCHEMA_VERSION
    label_threshold = 2.0
    settings = _optional_settings()
    if settings is not None:
        parsed = [item.strip() for item in settings.TRAIN_STRATEGY_ALLOWLIST.split(",") if item.strip()]
        allowlist = parsed or None
        include_candle = bool(settings.TRAIN_INCLUDE_CANDLE_BASELINE)
        label_schema = active_label_schema_version(use_tpsl_exit=bool(settings.MODEL_LABEL_USE_TPSL_EXIT))
        label_threshold = float(settings.MODEL_AUTO_TRAIN_LABEL_BPS)
    return allowlist, include_candle, label_schema, label_threshold


class DirectionalTradeJournal(_BaseTradeJournal):
    """Trade journal with directional, cost-aware ML outcomes."""

    @staticmethod
    def _model_metrics_dict(model_row: dict[str, Any]) -> dict[str, Any]:
        metrics = model_row.get("metrics") or {}
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except json.JSONDecodeError:
                metrics = {}
        return metrics if isinstance(metrics, dict) else {}

    @classmethod
    def _model_horizon_minutes(cls, model_row: dict[str, Any], default: int | None = None) -> int:
        metrics = cls._model_metrics_dict(model_row)
        for key in ("horizon_minutes", "model_horizon_minutes"):
            raw_horizon = metrics.get(key)
            if raw_horizon is None:
                continue
            try:
                horizon = int(raw_horizon)
            except (TypeError, ValueError):
                continue
            if horizon > 0:
                return horizon
        if default is not None:
            return default
        settings = _optional_settings()
        if settings is not None:
            return int(getattr(settings, "MODEL_AUTO_TRAIN_HORIZON_MINUTES", 5) or 5)
        return 5

    async def _paper_pnl_for_model(
        self,
        model_version: str,
        horizon_minutes: int,
        feature_schema_hash: str,
    ) -> dict[str, Any]:
        _allowlist, _include_candle, label_schema, _label_threshold = _training_eligibility_params()
        baseline_rows = await self._fetch(
            """
            SELECT po.net_return_bps
            FROM prediction_events pe
            JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
            WHERE pe.model_version = 'RULE_BASELINE_V1'
              AND COALESCE(pe.decision, '') <> 'SHADOW_CANDLE'
              AND po.horizon_minutes = $1
              AND po.label IS NOT NULL
              AND po.label_schema_version = $2
              AND pe.strategy_signal IN ('Buy', 'Sell')
            ORDER BY pe.created_at ASC
            LIMIT 1000
            """,
            int(horizon_minutes),
            label_schema,
        )
        gate_rows = await self._fetch(
            """
            SELECT po.net_return_bps
            FROM prediction_events pe
            JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
            JOIN feature_snapshots fs ON fs.snapshot_id = pe.feature_snapshot_id
            WHERE pe.model_version = $1
              AND pe.decision = 'GATE_PASS'
              AND po.horizon_minutes = $2
              AND po.label IS NOT NULL
              AND po.label_schema_version = $3
              AND ($4::text = '' OR fs.feature_schema_hash = $4)
            ORDER BY pe.created_at ASC
            LIMIT 1000
            """,
            model_version,
            int(horizon_minutes),
            label_schema,
            feature_schema_hash,
        )

        def stats(rows: list[Any]) -> dict[str, Any]:
            returns = [float(row.get("net_return_bps") or 0.0) for row in rows]
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

        return {
            "model_version": model_version,
            "horizon_minutes": int(horizon_minutes),
            "baseline": stats(baseline_rows),
            "model_gate": stats(gate_rows),
        }

    async def get_live_paper_gate_stats(
        self,
        model_version: str,
        *,
        horizon_minutes: int = 15,
        feature_schema_hash: str = "",
    ) -> dict[str, Any]:
        try:
            paper = await self._paper_pnl_for_model(model_version, int(horizon_minutes), feature_schema_hash)
            return dict(paper.get("model_gate") or {})
        except Exception as exc:
            log.debug("directional_trade_journal.live_paper_gate_stats_failed", error=str(exc))
            return await super().get_live_paper_gate_stats(
                model_version,
                horizon_minutes=horizon_minutes,
                feature_schema_hash=feature_schema_hash,
            )

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
        label_bps_threshold: float | None = None,
        limit: int = 200,
        cost_model: CostModelBps | None = None,
    ) -> int:
        """Resolve Buy and Sell outcomes from complete confirmed 1-minute paths.

        Legacy long-only outcomes are recalculated because the lookup joins only
        rows carrying the current label schema version. The existing primary
        key then updates the old row in place. MFE and MAE use every confirmed
        candle inside the full horizon rather than only the final bar.

        When ``MODEL_LABEL_USE_TPSL_EXIT`` is enabled, exit price is the first
        TP/SL touch using ATR from the feature snapshot (scalp_micro aligned).
        """

        costs = cost_model or default_cost_model()
        use_tpsl, tp_mult, sl_mult, default_label_bps, label_schema = _label_resolution_settings()
        if label_bps_threshold is None:
            label_bps_threshold = default_label_bps
        rows = await self._fetch(
            """
            SELECT
                pe.prediction_id,
                pe.symbol,
                pe.strategy_signal,
                fs.candle_open_time AS entry_time,
                fs.feature_names,
                fs.feature_values
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
              AND (
                  pe.decision IN ('GATE_PASS', 'GATE_BLOCK')
                  OR fs.training_eligible = true
              )
            ORDER BY
                CASE WHEN pe.decision IN ('GATE_PASS', 'GATE_BLOCK') THEN 0 ELSE 1 END,
                pe.created_at ASC
            LIMIT $2
            """,
            horizon_minutes,
            limit,
            label_schema,
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

            atr_pct = atr_pct_from_feature_payload(row.get("feature_names"), row.get("feature_values"))
            outcome = build_directional_outcome(
                side=side,
                entry_price=entry_close,
                exit_price=float(path_rows[-1]["close"]),
                highs=[float(item["high"]) for item in path_rows],
                lows=[float(item["low"]) for item in path_rows],
                cost_model=costs,
                label_threshold_bps=label_bps_threshold,
                atr_pct=atr_pct,
                tp_atr_mult=tp_mult,
                sl_atr_mult=sl_mult,
                use_tpsl_exit=use_tpsl and atr_pct is not None,
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
                label_schema_version=label_schema,
            )
            resolved += 1

        return resolved

    async def get_db_diagnostics(self, *, lite: bool = False) -> dict[str, Any]:
        """Expose only current-schema labels and models to diagnostics."""

        result = await super().get_db_diagnostics(lite=lite)
        allowlist, include_candle, label_schema, label_threshold = _training_eligibility_params()
        settings = _optional_settings()
        result["label_schema_version"] = label_schema
        result["training_label_threshold_bps"] = label_threshold
        result["training_config"] = {
            "strategy_allowlist": allowlist or [],
            "runtime_strategy_allowlist": allowlist or [],
            "include_candle_baseline": include_candle,
            "label_schema_version": label_schema,
            "label_threshold_bps": label_threshold,
            "auto_train_horizon_minutes": (
                int(settings.MODEL_AUTO_TRAIN_HORIZON_MINUTES) if settings is not None else 5
            ),
        }
        if lite:
            latest_model = result.get("latest_model_version") or {}
            metrics = self._model_metrics_dict(latest_model)
            model_label_schema = str(metrics.get("label_schema_version") or "")
            if latest_model:
                actual_samples = int(latest_model.get("training_samples", 0) or 0)
                schema_compatible = model_label_schema == label_schema
                latest_model["actual_training_samples"] = actual_samples
                latest_model["schema_compatible"] = schema_compatible
                latest_model["training_samples_compatible"] = actual_samples if schema_compatible else 0
        if not self.is_enabled or lite:
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
        allowlist, include_candle, label_schema, label_threshold = _training_eligibility_params()
        settings = _optional_settings()
        result["label_schema_version"] = label_schema
        result["training_label_threshold_bps"] = label_threshold
        result["training_config"] = {
            "strategy_allowlist": allowlist or [],
            "runtime_strategy_allowlist": allowlist or [],
            "include_candle_baseline": include_candle,
            "label_schema_version": label_schema,
            "label_threshold_bps": label_threshold,
            "auto_train_horizon_minutes": (
                int(settings.MODEL_AUTO_TRAIN_HORIZON_MINUTES) if settings is not None else 5
            ),
        }
        rows = await self._fetch(
            """
            SELECT horizon_minutes, count(*) AS cnt
            FROM (
                SELECT horizon_minutes
                FROM prediction_outcomes
                WHERE label IS NOT NULL
                  AND label_schema_version = $1
                LIMIT 4000
            ) capped
            GROUP BY horizon_minutes
            ORDER BY horizon_minutes
            """,
            label_schema,
        )
        if rows:
            result["prediction_outcomes_by_horizon"] = {str(row["horizon_minutes"]): int(row["cnt"]) for row in rows}
            result["prediction_outcomes"] = min(1000, sum(result["prediction_outcomes_by_horizon"].values()))

        min_train_samples = 1000
        settings = _optional_settings()
        if settings is not None:
            min_train_samples = max(50, int(settings.MODEL_AUTO_TRAIN_MIN_SAMPLES))

        snapshots = await fetch_training_snapshots_for_horizons(
            self._fetch,
            horizons=(5, 15),
            label_schema_version=label_schema,
            label_threshold_bps=label_threshold,
            strategy_allowlist=allowlist,
            include_candle_baseline=include_candle,
            min_samples=min_train_samples,
        )
        snapshot_5 = snapshots.get("5")
        snapshot_15 = snapshots.get("15")

        training_by_horizon: dict[str, int] = {}
        filtered_total_by_horizon: dict[str, int] = {}
        newest_schema_by_horizon: dict[str, dict[str, Any]] = {}
        label_thresholds_by_horizon: dict[str, dict[str, int]] = {}
        for horizon_key, snapshot in snapshots.items():
            # Auto-train and train.py require one feature schema to reach min_samples.
            training_by_horizon[horizon_key] = snapshot.trainable_schema_count
            filtered_total_by_horizon[horizon_key] = snapshot.filtered_distinct_candles
            label_thresholds_by_horizon[horizon_key] = snapshot.by_label_threshold
            newest_schema_by_horizon[horizon_key] = {
                "feature_schema_hash": snapshot.newest_schema_hash,
                "sample_count": snapshot.newest_schema_count,
                "best_schema_count": snapshot.best_schema_count,
                "best_schema_hash": snapshot.best_schema_hash,
                "trainable_schema_count": snapshot.trainable_schema_count,
                "trainable_schema_hash": snapshot.trainable_schema_hash,
                "horizon_minutes": horizon_key,
                "label_schema_version": label_schema,
                "label_threshold_bps": label_threshold,
            }

        newest_feature_schema_hash = snapshot_15.newest_schema_hash if snapshot_15 else ""
        newest_feature_schema_samples = snapshot_15.newest_schema_count if snapshot_15 else 0
        if snapshot_5 and snapshot_5.newest_schema_hash:
            newest_feature_schema_hash = snapshot_5.newest_schema_hash
            newest_feature_schema_samples = snapshot_5.newest_schema_count

        dominant_schema_hash = ""
        dominant_schema_samples = 0
        if snapshot_5 and snapshot_5.by_schema:
            dominant_schema_hash = snapshot_5.best_schema_hash
            dominant_schema_samples = snapshot_5.best_schema_count

        result["labelled_samples_15m"] = newest_feature_schema_samples
        result["training_eligible_schema_15m"] = {
            "feature_schema_hash": newest_feature_schema_hash,
            "sample_count": newest_feature_schema_samples,
            "best_schema_count": snapshot_15.best_schema_count if snapshot_15 else 0,
            "horizon_minutes": "5_or_15",
            "label_schema_version": label_schema,
            "label_threshold_bps": label_threshold,
        }
        result["labelled_samples_15m_by_schema"] = {
            key[:8]: value for key, value in (snapshot_15.by_schema if snapshot_15 else {}).items()
        } or result.get("labelled_samples_15m_by_schema", {})
        result["newest_training_schema_hash"] = newest_feature_schema_hash
        result["newest_training_schema_samples"] = newest_feature_schema_samples
        result["training_eligible_15m"] = training_by_horizon.get("15", newest_feature_schema_samples)
        result["training_eligible_by_horizon"] = training_by_horizon
        result["training_filtered_total_by_horizon"] = filtered_total_by_horizon
        result["training_label_thresholds_by_horizon"] = label_thresholds_by_horizon
        result["newest_training_schema_by_horizon"] = newest_schema_by_horizon
        result["dominant_training_schema_hash"] = dominant_schema_hash
        result["dominant_training_schema_samples"] = dominant_schema_samples

        active_pool = snapshot_5.by_strategy_pool if snapshot_5 else {}
        legacy_breakdown_rows = await self._fetch(
            """
            SELECT
                CASE
                    WHEN metadata->>'strategy_id' = 'scalp_micro_v1' THEN 'scalp_micro_v1'
                    WHEN metadata->>'strategy_id' IS NULL
                         AND COALESCE(decision, '') IN ('SHADOW_CANDLE', 'HISTORICAL_REAL')
                        THEN 'candle_baseline'
                    ELSE COALESCE(metadata->>'strategy_id', 'other')
                END AS pool,
                count(*) AS samples
            FROM (
                SELECT DISTINCT ON (
                           fs.symbol,
                           fs.interval,
                           fs.candle_open_time,
                           fs.feature_schema_hash
                       )
                       pe.metadata,
                       pe.decision
                FROM feature_snapshots fs
                JOIN prediction_events pe ON pe.feature_snapshot_id = fs.snapshot_id
                JOIN prediction_outcomes po ON po.prediction_id = pe.prediction_id
                WHERE po.horizon_minutes = 5
                  AND po.label IS NOT NULL
                  AND po.label_schema_version = $1
                  AND po.label_threshold_bps = $2
                  AND fs.feature_values IS NOT NULL
                  AND fs.training_eligible = true
                  AND pe.model_version = 'RULE_BASELINE_V1'
                  AND pe.strategy_signal IN ('Buy', 'Sell')
                ORDER BY fs.symbol, fs.interval, fs.candle_open_time,
                         fs.feature_schema_hash, fs.created_at DESC, pe.created_at DESC
            ) deduped
            GROUP BY pool
            ORDER BY pool
            """,
            label_schema,
            label_threshold,
        )
        legacy_v1 = {
            str(row["pool"]): int(row.get("samples", row.get("sample_count", 0)) or 0) for row in legacy_breakdown_rows
        }
        other_active = sum(
            count
            for pool_name, count in active_pool.items()
            if pool_name
            not in {
                "scalp_micro_v1",
                "shadow_probe_hv_v2",
                "candle_baseline",
                "candle_sampler_v1",
            }
        )
        result["training_pool_breakdown"] = {
            "active_schema": label_schema,
            "eligible_filtered_5m": int(training_by_horizon.get("5", 0) or 0),
            "filtered_total_5m": int(filtered_total_by_horizon.get("5", 0) or 0),
            "scalp_micro_v1_active_schema": int(active_pool.get("scalp_micro_v1", 0)),
            "shadow_probe_hv_v2_active_schema": int(active_pool.get("shadow_probe_hv_v2", 0)),
            "candle_baseline_active_schema": int(active_pool.get("candle_baseline", 0)),
            "candle_sampler_v1_active_schema": int(active_pool.get("candle_sampler_v1", 0)),
            "other_active_schema": int(other_active),
            "legacy_v1_candle_baseline": int(legacy_v1.get("candle_baseline", 0)),
            "legacy_v1_scalp_micro_v1": int(legacy_v1.get("scalp_micro_v1", 0)),
            "active_pools": active_pool,
            "legacy_v1_pools": legacy_v1,
        }

        rows = await self._fetch(
            """
            SELECT
                version,
                status,
                training_samples,
                metrics,
                COALESCE(
                    NULLIF(feature_schema_hash, ''),
                    NULLIF(metrics->>'source_feature_schema_hash', ''),
                    metrics->>'feature_schema_hash',
                    ''
                ) AS feature_schema_hash,
                training_finished_at,
                created_at
            FROM model_versions
            WHERE artifact IS NOT NULL
              AND COALESCE(metrics->>'label_schema_version', '') = $1
            ORDER BY training_finished_at DESC NULLS LAST, created_at DESC
            LIMIT 1
            """,
            label_schema,
        )
        if not rows:
            rows = await self._fetch(
                """
                SELECT
                    version,
                    status,
                    training_samples,
                    metrics,
                    COALESCE(
                        NULLIF(feature_schema_hash, ''),
                        NULLIF(metrics->>'source_feature_schema_hash', ''),
                        metrics->>'feature_schema_hash',
                        ''
                    ) AS feature_schema_hash,
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
        # Compare against the NEWEST schema (what the pipeline is currently producing),
        # not the dominant schema (which may still be the old schema with more historical samples).
        _mismatch_reference = newest_feature_schema_hash or dominant_schema_hash
        if latest_model and _mismatch_reference and latest_model_schema_hash != _mismatch_reference:
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
            latest_model["schema_compatible"] = latest_model_schema == label_schema
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
                COALESCE(
                    NULLIF(feature_schema_hash, ''),
                    NULLIF(metrics->>'source_feature_schema_hash', ''),
                    metrics->>'feature_schema_hash',
                    ''
                ) AS feature_schema_hash,
                training_finished_at,
                created_at
            FROM model_versions
            WHERE status = 'CHAMPION'
              AND artifact IS NOT NULL
              AND COALESCE(metrics->>'label_schema_version', '') = $1
            ORDER BY training_finished_at DESC NULLS LAST, created_at DESC
            LIMIT 1
            """,
            label_schema,
        )
        # When no CHAMPION exists, fall back to latest_model_version so that
        # diagnostics and the heartbeat can still show which model is loaded.
        result["active_model_version"] = dict(rows[0]) if rows else result.get("latest_model_version", {})

        active_model = result.get("active_model_version", {}) or latest_model
        active_version = str(active_model.get("version") or "")
        active_schema_hash = str(
            active_model.get("feature_schema_hash") or latest_model.get("feature_schema_hash") or ""
        )
        analysis_horizon = self._model_horizon_minutes(active_model or latest_model)
        result["model_gate_horizon_minutes"] = analysis_horizon
        if active_version:
            gate = await self.get_shadow_gate_stats(active_version, analysis_horizon, label_schema)
            gate["horizon_minutes"] = analysis_horizon
            paper = await self._paper_pnl_for_model(active_version, analysis_horizon, active_schema_hash)
            if active_schema_hash or gate.get("total_count"):
                result["shadow_gate_by_horizon"] = {str(analysis_horizon): gate}
                result["paper_pnl_by_horizon"] = {str(analysis_horizon): paper}
                result[f"shadow_gate_{analysis_horizon}m"] = gate
                result[f"paper_pnl_{analysis_horizon}m"] = paper
                # Compatibility keys for existing Telegram/dashboard callers. The
                # payload includes horizon_minutes so consumers can display the
                # actual horizon instead of trusting the legacy key name.
                result["shadow_gate_15m"] = gate
                result["paper_pnl_15m"] = paper

        return result


def install_directional_trade_journal() -> None:
    """Install the directional implementation for existing import paths."""

    _base_module.TradeJournal = DirectionalTradeJournal  # type: ignore[misc]
