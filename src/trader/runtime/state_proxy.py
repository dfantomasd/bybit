"""Read-only FastAPI view of ``TradingApplication`` state."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from trader.domain.enums import SystemStatus, TradingMode

if TYPE_CHECKING:
    from trader.app import TradingApplication


class AppStateProxy:
    """Thin read-only view of TradingApplication state for the FastAPI layer.

    Uses ``getattr`` with safe defaults throughout so it never raises
    even when sub-components haven't been initialised yet.
    """

    def __init__(self, app: TradingApplication) -> None:
        self._app = app

    @property
    def system_status(self) -> Any:
        return getattr(self._app, "_status", SystemStatus.STOPPED)

    @property
    def trading_mode(self) -> Any:
        s = getattr(self._app, "_settings", None)
        return s.TRADING_MODE if s is not None else TradingMode.TESTNET

    @property
    def open_position_count(self) -> int:
        eng = getattr(self._app, "_execution_engine", None)
        if eng is None:
            return 0
        return len(getattr(eng, "_open_positions", {}))

    @property
    def is_live(self) -> bool:
        s = getattr(self._app, "_settings", None)
        return bool(s and getattr(s, "LIVE_MODE", False))

    @property
    def open_positions(self) -> list[Any]:
        """Return Position-like SimpleNamespace objects for /positions endpoint."""
        from types import SimpleNamespace

        eng = getattr(self._app, "_execution_engine", None)
        if eng is None:
            return []
        raw: dict[str, Any] = getattr(eng, "_open_positions", {})
        result = []
        for symbol, pos in raw.items():
            result.append(
                SimpleNamespace(
                    symbol=symbol,
                    market_type="LINEAR",
                    side=pos.get("side"),
                    size=pos.get("size", 0),
                    entry_price=pos.get("entry_price", 0),
                    mark_price=None,
                    unrealised_pnl=0,
                    leverage=1,
                )
            )
        return result

    @property
    def current_regimes(self) -> dict[str, Any]:
        return {}

    @property
    def active_model_metadata(self) -> Any:
        """Return a ModelMetadata built from the in-memory champion or challenger."""
        from trader.domain.models import ModelMetadata

        registry = getattr(self._app, "_model_registry", None)
        if registry is None:
            return None
        model = getattr(registry, "champion", None) or getattr(registry, "challenger", None)
        if model is None:
            return None
        try:
            return ModelMetadata(
                model_id=model.version,
                version=model.version,
                algorithm=getattr(model, "model_type", "SGD"),
                strategy_id="ema_crossover_v1",
                trained_at=getattr(model, "created_at", datetime.now(tz=UTC)),
                train_episodes=getattr(model, "training_samples", None),
                feature_version=getattr(model, "label_schema_version", "v1"),
            )
        except Exception:
            return None


# Backward-compatible alias
_AppStateProxy = AppStateProxy
