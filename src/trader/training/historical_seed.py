"""Historical training sample generator from real market_candles.

Replays confirmed OHLCV rows from PostgreSQL through the same FeaturePipeline
used in production, writes feature_snapshots + RULE_BASELINE_V1 prediction
events, and resolves outcomes from forward real candles.

Usage:
    # 1. Load candles from Bybit REST into Postgres
    python -m trader.training.backfill --symbols BTCUSDT,ETHUSDT --intervals 1 --days 14

    # 2. Generate labelled training samples from stored candles
    python -m trader.training.historical_seed --symbols BTCUSDT,ETHUSDT --interval 1

    # 3. Train the challenger model on resolved labels
    python -m trader.training.train --min-samples 500 --horizon 5

NEVER runs inside the trading process.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import click

from trader.data.candles import Candle, CandleStore
from trader.features.pipeline import _MIN_BARS, FeaturePipeline
from trader.storage.directional_trade_journal import default_cost_model
from trader.training.feature_side import feature_values_for_side
from trader.training.labels import (
    active_label_schema_version,
    build_directional_outcome,
)

_INTERVAL_MINUTES = {
    "1": 1,
    "3": 3,
    "5": 5,
    "15": 15,
    "30": 30,
    "60": 60,
}

_HISTORICAL_DECISION = "HISTORICAL_REAL"
_MODEL_VERSION = "RULE_BASELINE_V1"


@dataclass(frozen=True)
class DbCandle:
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class SeedStats:
    symbol: str
    interval: str
    candles_loaded: int
    samples_written: int
    samples_skipped: int
    outcomes_resolved: int


def _interval_minutes(interval: str) -> int:
    minutes = _INTERVAL_MINUTES.get(str(interval).strip())
    if minutes is None:
        raise ValueError(f"unsupported interval for historical seed: {interval!r}")
    return minutes


def _candle_close_time(open_time: datetime, interval: str) -> datetime:
    return open_time + timedelta(minutes=_interval_minutes(interval))


def _rule_side_from_features(feature_names: list[str], feature_values: list[float]) -> str | None:
    features = dict(zip(feature_names, feature_values, strict=True))
    ema9 = features.get("ema_9")
    ema21 = features.get("ema_21")
    if ema9 is None or ema21 is None:
        return None
    return "Buy" if ema9 > ema21 else "Sell"


def _forward_path(candles: list[DbCandle], entry_idx: int, horizon_minutes: int) -> list[DbCandle] | None:
    if horizon_minutes <= 0:
        return None
    end_idx = entry_idx + horizon_minutes
    if end_idx >= len(candles):
        return None
    entry_time = candles[entry_idx].open_time
    horizon_time = entry_time + timedelta(minutes=horizon_minutes)
    path = candles[entry_idx + 1 : end_idx + 1]
    if len(path) != horizon_minutes:
        return None
    if path[-1].open_time != horizon_time:
        return None
    return path


def _bucket_start(open_time: datetime, bucket_minutes: int) -> datetime:
    """Align open_time to the start of a fixed-width candle bucket."""
    ts = open_time.astimezone(UTC) if open_time.tzinfo else open_time.replace(tzinfo=UTC)
    epoch_minutes = int(ts.timestamp() // 60)
    aligned = (epoch_minutes // bucket_minutes) * bucket_minutes
    return datetime.fromtimestamp(aligned * 60, tz=UTC)


def _is_full_bucket(group: list[DbCandle], bucket_minutes: int) -> bool:
    if len(group) != bucket_minutes:
        return False
    start = group[0].open_time
    for idx, candle in enumerate(group):
        expected = start + timedelta(minutes=idx)
        if candle.open_time != expected:
            return False
    return True


def aggregate_candles(candles_1m: list[DbCandle], *, bucket_minutes: int) -> list[DbCandle]:
    """Build higher-TF OHLCV bars from consecutive 1m history (offline replay fallback)."""
    if bucket_minutes <= 1 or not candles_1m:
        return []
    buckets: dict[datetime, list[DbCandle]] = {}
    for candle in candles_1m:
        start = _bucket_start(candle.open_time, bucket_minutes)
        buckets.setdefault(start, []).append(candle)

    aggregated: list[DbCandle] = []
    for start in sorted(buckets):
        group = sorted(buckets[start], key=lambda c: c.open_time)
        if not _is_full_bucket(group, bucket_minutes):
            continue
        aggregated.append(
            DbCandle(
                open_time=start,
                open=group[0].open,
                high=max(item.high for item in group),
                low=min(item.low for item in group),
                close=group[-1].close,
                volume=sum(item.volume for item in group),
            )
        )
    return aggregated


def _db_candle_to_store(candle: DbCandle) -> Candle:
    return Candle(
        open_time=candle.open_time,
        open=candle.open,
        high=candle.high,
        low=candle.low,
        close=candle.close,
        volume=candle.volume,
        confirm=True,
    )


def _sync_mtf_candles(
    store: CandleStore,
    *,
    symbol: str,
    mtf_candles: dict[str, list[DbCandle]],
    mtf_ptr: dict[str, int],
    as_of_close: datetime,
) -> None:
    """Add confirmed higher-TF bars whose close time is <= ``as_of_close``."""
    for interval, candles in mtf_candles.items():
        ptr = mtf_ptr.get(interval, 0)
        while ptr < len(candles):
            bar = candles[ptr]
            if _candle_close_time(bar.open_time, interval) > as_of_close:
                break
            store.add(symbol, interval, _db_candle_to_store(bar))
            ptr += 1
        mtf_ptr[interval] = ptr


def seed_candles_for_symbol(
    *,
    symbol: str,
    interval: str,
    candles: list[DbCandle],
    horizons: list[int],
    label_bps_threshold: float,
    skip_existing: bool,
    store_max_bars: int = 500,
    use_tpsl_exit: bool = True,
    tp_atr_mult: float = 1.0,
    sl_atr_mult: float = 0.5,
    label_schema_version: str | None = None,
    mtf_candles: dict[str, list[DbCandle]] | None = None,
) -> tuple[list[dict[str, Any]], SeedStats]:
    """Pure replay: compute features and pending DB rows without I/O."""
    if interval != "1":
        raise ValueError("historical seed currently supports only 1-minute candles for outcome resolution")

    store = CandleStore(max_bars=store_max_bars)
    pipeline = FeaturePipeline(store)
    costs = default_cost_model()
    schema_version = label_schema_version or active_label_schema_version(use_tpsl_exit=use_tpsl_exit)
    max_horizon = max(horizons)
    pending_rows: list[dict[str, Any]] = []
    samples_written = 0
    samples_skipped = 0
    outcomes_resolved = 0
    mtf_by_interval = {k: list(v) for k, v in (mtf_candles or {}).items()}
    mtf_ptr = dict.fromkeys(mtf_by_interval, 0)

    for idx, row in enumerate(candles):
        store.add(
            symbol,
            interval,
            Candle(
                open_time=row.open_time,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
                confirm=True,
            ),
        )
        current_close = _candle_close_time(row.open_time, interval)
        if mtf_by_interval:
            _sync_mtf_candles(
                store,
                symbol=symbol,
                mtf_candles=mtf_by_interval,
                mtf_ptr=mtf_ptr,
                as_of_close=current_close,
            )
        if idx < _MIN_BARS - 1:
            continue
        if idx + max_horizon >= len(candles):
            break

        vec = pipeline.compute(symbol, interval)
        if vec is None:
            continue

        side = _rule_side_from_features(vec.feature_names, vec.values)
        if side is None:
            continue

        model_feature_names, model_feature_values = feature_values_for_side(vec, side)
        candle_open_time = candles[idx].open_time
        created_at = _candle_close_time(candle_open_time, interval)
        schema_hash = hashlib.sha256(json.dumps(model_feature_names).encode()).hexdigest()[:16]

        sample_key = (symbol, interval, candle_open_time.isoformat(), schema_hash)
        if skip_existing and any(item["sample_key"] == sample_key for item in pending_rows):
            samples_skipped += 1
            continue

        outcomes: list[dict[str, Any]] = []
        entry_close = candles[idx].close
        feature_map = dict(zip(vec.feature_names, vec.values, strict=True))
        atr_pct = feature_map.get("atr_14_pct")
        atr_value = float(atr_pct) if atr_pct is not None else None
        for horizon in horizons:
            path = _forward_path(candles, idx, horizon)
            if path is None or entry_close <= 0:
                continue
            outcome = build_directional_outcome(
                side=side,
                entry_price=entry_close,
                exit_price=path[-1].close,
                highs=[item.high for item in path],
                lows=[item.low for item in path],
                cost_model=costs,
                label_threshold_bps=label_bps_threshold,
                atr_pct=atr_value,
                tp_atr_mult=tp_atr_mult,
                sl_atr_mult=sl_atr_mult,
                use_tpsl_exit=use_tpsl_exit and atr_value is not None,
            )
            outcomes.append(
                {
                    "horizon_minutes": horizon,
                    "gross_return_bps": outcome.gross_return_bps,
                    "cost_bps": costs.total_bps,
                    "label_threshold_bps": label_bps_threshold,
                    "net_return_bps": outcome.net_return_bps,
                    "max_favorable_excursion_bps": outcome.max_favorable_excursion_bps,
                    "max_adverse_excursion_bps": outcome.max_adverse_excursion_bps,
                    "label": outcome.label,
                    "label_schema_version": schema_version,
                }
            )
            outcomes_resolved += 1

        if not outcomes:
            continue

        pending_rows.append(
            {
                "sample_key": sample_key,
                "created_at": created_at,
                "symbol": symbol,
                "interval": interval,
                "candle_open_time": candle_open_time,
                "feature_schema_hash": schema_hash,
                "feature_names": model_feature_names,
                "feature_values": model_feature_values,
                "side": side,
                "outcomes": outcomes,
            }
        )
        samples_written += 1

    return pending_rows, SeedStats(
        symbol=symbol,
        interval=interval,
        candles_loaded=len(candles),
        samples_written=samples_written,
        samples_skipped=samples_skipped,
        outcomes_resolved=outcomes_resolved,
    )


async def _load_candles(pool: Any, *, symbol: str, interval: str) -> list[DbCandle]:
    rows = await pool.fetch(
        """
        SELECT open_time, open, high, low, close, volume
        FROM market_candles
        WHERE symbol = $1
          AND interval = $2
          AND confirmed = true
        ORDER BY open_time ASC
        """,
        symbol,
        interval,
    )
    return [
        DbCandle(
            open_time=row["open_time"],
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        for row in rows
    ]


async def _existing_sample_keys(pool: Any, *, symbol: str, interval: str) -> set[tuple[str, str, str, str]]:
    rows = await pool.fetch(
        """
        SELECT fs.symbol, fs.interval, fs.candle_open_time, fs.feature_schema_hash
        FROM feature_snapshots fs
        JOIN prediction_events pe ON pe.feature_snapshot_id = fs.snapshot_id
        WHERE fs.symbol = $1
          AND fs.interval = $2
          AND fs.training_eligible = true
          AND pe.model_version = $3
          AND pe.decision = $4
        """,
        symbol,
        interval,
        _MODEL_VERSION,
        _HISTORICAL_DECISION,
    )
    return {
        (
            str(row["symbol"]),
            str(row["interval"]),
            row["candle_open_time"].isoformat(),
            str(row["feature_schema_hash"]),
        )
        for row in rows
    }


async def _persist_sample(pool: Any, sample: dict[str, Any]) -> bool:
    async with pool.acquire() as conn:
        async with conn.transaction():
            snapshot_id = await conn.fetchval(
                """
                INSERT INTO feature_snapshots (
                    created_at, symbol, interval, candle_open_time,
                    feature_schema_hash, feature_names, feature_values, training_eligible
                )
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, true)
                ON CONFLICT (symbol, interval, candle_open_time, feature_schema_hash)
                WHERE training_eligible = true
                DO UPDATE SET
                    feature_names = EXCLUDED.feature_names,
                    feature_values = EXCLUDED.feature_values,
                    created_at = EXCLUDED.created_at
                RETURNING snapshot_id
                """,
                sample["created_at"],
                sample["symbol"],
                sample["interval"],
                sample["candle_open_time"],
                sample["feature_schema_hash"],
                json.dumps(sample["feature_names"]),
                json.dumps(sample["feature_values"]),
            )
            if snapshot_id is None:
                return False

            prediction_id = await conn.fetchval(
                """
                INSERT INTO prediction_events (
                    created_at, symbol, interval, model_version, feature_snapshot_id,
                    score, strategy_signal, decision, metadata
                )
                VALUES ($1, $2, $3, $4, $5::uuid, $6, $7, $8, $9::jsonb)
                RETURNING prediction_id
                """,
                sample["created_at"],
                sample["symbol"],
                sample["interval"],
                _MODEL_VERSION,
                snapshot_id,
                0.5,
                sample["side"],
                _HISTORICAL_DECISION,
                json.dumps({"source": "historical_seed", "candle_open_time": sample["candle_open_time"].isoformat()}),
            )
            if prediction_id is None:
                return False

            for outcome in sample["outcomes"]:
                await conn.execute(
                    """
                    INSERT INTO prediction_outcomes (
                        prediction_id, horizon_minutes, gross_return_bps, cost_bps,
                        label_threshold_bps, net_return_bps,
                        max_favorable_excursion_bps, max_adverse_excursion_bps,
                        label, resolved_at, label_schema_version
                    )
                    VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (prediction_id, horizon_minutes) DO UPDATE SET
                        gross_return_bps = EXCLUDED.gross_return_bps,
                        cost_bps = EXCLUDED.cost_bps,
                        label_threshold_bps = EXCLUDED.label_threshold_bps,
                        net_return_bps = EXCLUDED.net_return_bps,
                        max_favorable_excursion_bps = EXCLUDED.max_favorable_excursion_bps,
                        max_adverse_excursion_bps = EXCLUDED.max_adverse_excursion_bps,
                        label = EXCLUDED.label,
                        resolved_at = EXCLUDED.resolved_at,
                        label_schema_version = EXCLUDED.label_schema_version
                    """,
                    prediction_id,
                    outcome["horizon_minutes"],
                    outcome["gross_return_bps"],
                    outcome["cost_bps"],
                    outcome["label_threshold_bps"],
                    outcome["net_return_bps"],
                    outcome["max_favorable_excursion_bps"],
                    outcome["max_adverse_excursion_bps"],
                    outcome["label"],
                    sample["created_at"] + timedelta(minutes=int(outcome["horizon_minutes"])),
                    outcome["label_schema_version"],
                )
    return True


async def _seed_symbol(
    pool: Any,
    *,
    symbol: str,
    interval: str,
    horizons: list[int],
    label_bps_threshold: float,
    skip_existing: bool,
) -> SeedStats:
    from trader.config import Settings

    settings = Settings()
    use_tpsl = bool(settings.MODEL_LABEL_USE_TPSL_EXIT)
    tp_mult = float(settings.MODEL_LABEL_TP_ATR_MULT)
    sl_mult = float(settings.MODEL_LABEL_SL_ATR_MULT)
    label_schema = active_label_schema_version(use_tpsl_exit=use_tpsl)
    candles = await _load_candles(pool, symbol=symbol, interval=interval)
    mtf_candles: dict[str, list[DbCandle]] = {}
    for mtf_interval in ("5", "15"):
        loaded = await _load_candles(pool, symbol=symbol, interval=mtf_interval)
        if not loaded and interval == "1":
            loaded = aggregate_candles(candles, bucket_minutes=_interval_minutes(mtf_interval))
        if loaded:
            mtf_candles[mtf_interval] = loaded

    if len(candles) < _MIN_BARS + max(horizons):
        return SeedStats(
            symbol=symbol,
            interval=interval,
            candles_loaded=len(candles),
            samples_written=0,
            samples_skipped=0,
            outcomes_resolved=0,
        )

    existing_keys: set[tuple[str, str, str, str]] = set()
    if skip_existing:
        existing_keys = await _existing_sample_keys(pool, symbol=symbol, interval=interval)

    pending_rows, _stats = seed_candles_for_symbol(
        symbol=symbol,
        interval=interval,
        candles=candles,
        horizons=horizons,
        label_bps_threshold=label_bps_threshold,
        skip_existing=False,
        use_tpsl_exit=use_tpsl,
        tp_atr_mult=tp_mult,
        sl_atr_mult=sl_mult,
        label_schema_version=label_schema,
        mtf_candles=mtf_candles or None,
    )

    written = 0
    skipped = 0
    outcomes = 0
    for sample in pending_rows:
        key = (
            sample["symbol"],
            sample["interval"],
            sample["candle_open_time"].isoformat(),
            sample["feature_schema_hash"],
        )
        if skip_existing and key in existing_keys:
            skipped += 1
            continue
        if await _persist_sample(pool, sample):
            written += 1
            outcomes += len(sample["outcomes"])
        else:
            skipped += 1

    return SeedStats(
        symbol=symbol,
        interval=interval,
        candles_loaded=len(candles),
        samples_written=written,
        samples_skipped=skipped,
        outcomes_resolved=outcomes,
    )


def _parse_horizons(raw: str, default_horizon: int) -> list[int]:
    values = [int(item.strip()) for item in str(raw or "").split(",") if item.strip()]
    return sorted(set(values or [default_horizon]))


def _settings_horizon() -> int:
    try:
        from trader.config import Settings

        return int(getattr(Settings(), "MODEL_LABEL_HORIZON", 15))
    except Exception as exc:
        import logging
        import os

        logging.getLogger(__name__).warning("historical_seed.settings_load_failed_using_default_horizon", exc_info=exc)
        try:
            return int(os.environ.get("MODEL_LABEL_HORIZON", "15"))
        except (TypeError, ValueError):
            return 15


async def _historical_seed(
    symbols: list[str],
    interval: str,
    horizons: list[int],
    label_bps_threshold: float,
    skip_existing: bool,
) -> int:
    import asyncpg

    from trader.config import Settings
    from trader.storage.trade_journal import asyncpg_pool_connect_kwargs

    settings = Settings()
    pool = await asyncpg.create_pool(
        **asyncpg_pool_connect_kwargs(settings.POSTGRES_DSN.get_secret_value()),
        min_size=1,
        max_size=2,
        statement_cache_size=0,
    )

    total_written = 0
    total_outcomes = 0
    try:
        for symbol in symbols:
            stats = await _seed_symbol(
                pool,
                symbol=symbol,
                interval=interval,
                horizons=horizons,
                label_bps_threshold=label_bps_threshold,
                skip_existing=skip_existing,
            )
            click.echo(
                f"{stats.symbol}/{stats.interval}: candles={stats.candles_loaded}, "
                f"samples={stats.samples_written}, skipped={stats.samples_skipped}, "
                f"outcomes={stats.outcomes_resolved}"
            )
            total_written += stats.samples_written
            total_outcomes += stats.outcomes_resolved
        click.echo(
            f"Historical seed complete: {total_written} samples, {total_outcomes} outcomes across {len(symbols)} symbols"
        )
        return 0
    finally:
        await pool.close()


@click.command()
@click.option("--symbols", "--symbol", default="BTCUSDT", help="Comma-separated trading symbols")
@click.option("--interval", default="1", show_default=True, help="Candle interval (only 1m supported)")
@click.option(
    "--horizons",
    default="",
    help="Comma-separated outcome horizons in minutes (default: MODEL_LABEL_HORIZON)",
)
@click.option("--label-bps", default=5.0, type=float, show_default=True, help="Label threshold in bps")
@click.option("--skip-existing/--no-skip-existing", default=True, show_default=True)
def main(symbols: str, interval: str, horizons: str, label_bps: float, skip_existing: bool) -> None:
    """Generate labelled training samples by replaying real market_candles."""
    symbol_list = [item.strip().upper() for item in symbols.split(",") if item.strip()]
    horizon_list = _parse_horizons(horizons, _settings_horizon())
    raise SystemExit(
        asyncio.run(
            _historical_seed(
                symbol_list,
                interval,
                horizon_list,
                label_bps,
                skip_existing,
            )
        )
    )


if __name__ == "__main__":
    main()
