"""Bind cached feature vectors to the confirmed candle used for computation.

The strategy loop reads cached vectors.  A new confirmed candle can arrive between
vector computation and snapshot persistence, so the cache must expose only vectors
whose source candle still matches the latest confirmed bar.  The registry is also
used by the storage guard to validate the snapshot write after awaited journal I/O.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from typing import Any, cast

import structlog

from trader.domain.models import FeatureVector
from trader.features import pipeline as _pipeline_module
from trader.features.pipeline import FeaturePipeline as _BaseFeaturePipeline

log = structlog.get_logger(__name__)

_MAX_SOURCE_BINDINGS = 20_000
_SourceBinding = tuple[str, str, datetime]
_SOURCE_CANDLE_BY_FEATURE_ID: OrderedDict[str, _SourceBinding] = OrderedDict()


def _normalise_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def register_source_candle(
    *,
    feature_id: Any,
    symbol: str,
    interval: str,
    candle_open_time: datetime,
) -> None:
    """Remember the confirmed source candle for one computed feature vector."""

    key = str(feature_id)
    _SOURCE_CANDLE_BY_FEATURE_ID[key] = (
        _normalise_symbol(symbol),
        str(interval),
        candle_open_time,
    )
    _SOURCE_CANDLE_BY_FEATURE_ID.move_to_end(key)
    while len(_SOURCE_CANDLE_BY_FEATURE_ID) > _MAX_SOURCE_BINDINGS:
        _SOURCE_CANDLE_BY_FEATURE_ID.popitem(last=False)


def source_candle_for_feature(feature_id: Any) -> _SourceBinding | None:
    """Return ``(symbol, interval, open_time)`` for a previously computed vector."""

    binding = _SOURCE_CANDLE_BY_FEATURE_ID.get(str(feature_id))
    if binding is not None:
        _SOURCE_CANDLE_BY_FEATURE_ID.move_to_end(str(feature_id))
    return binding


class SourceCandleFeaturePipeline(_BaseFeaturePipeline):
    """Feature pipeline that rejects stale cached vectors fail-closed."""

    def compute(self, symbol: str, interval: str) -> FeatureVector | None:
        vec = super().compute(symbol, interval)
        if vec is None:
            return None

        candles = self._store.latest(symbol, interval, 1)
        if not candles:
            log.warning(
                "feature_pipeline.source_candle_missing",
                symbol=symbol,
                interval=interval,
            )
            return None

        register_source_candle(
            feature_id=vec.feature_id,
            symbol=symbol,
            interval=interval,
            candle_open_time=candles[-1].open_time,
        )
        return vec

    def latest(self, symbol: str, interval: str) -> FeatureVector | None:
        vec = super().latest(symbol, interval)
        if vec is None:
            return None

        binding = source_candle_for_feature(vec.feature_id)
        candles = self._store.latest(symbol, interval, 1)
        latest_open_time = candles[-1].open_time if candles else None
        expected = (_normalise_symbol(symbol), str(interval), latest_open_time)

        if binding is None or latest_open_time is None or binding != expected:
            log.warning(
                "feature_pipeline.stale_cached_vector_rejected",
                symbol=symbol,
                interval=interval,
                feature_id=str(vec.feature_id),
                source_binding=binding,
                latest_candle_open_time=latest_open_time,
            )
            return None

        return vec


def install_source_candle_guard() -> None:
    """Install guarded pipeline for existing import paths."""

    pipeline_module = cast(Any, _pipeline_module)
    pipeline_module.FeaturePipeline = SourceCandleFeaturePipeline
