"""Base class for application-bound runtime modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trader.app import TradingApplication


class AppBoundModule:
    """Mixin providing typed access to the parent ``TradingApplication``."""

    def __init__(self, app: TradingApplication) -> None:
        self._app = app

    @property
    def app(self) -> TradingApplication:
        return self._app

    def _initial_shadow_mode(self) -> bool:
        return self._app._modules.signal_policy.initial_shadow_mode()

    def _record_diag(self, event: str) -> None:
        self._app._modules.diagnostics.record(event)

    async def _restore_execution_pending_entries(self) -> None:
        await self._app._restore_execution_pending_entries()

    async def _maybe_apply_online_learning(self) -> None:
        await self._app._modules.training.maybe_apply_online_learning()

    def _active_symbols(self) -> list[str]:
        return self._app._active_symbols()

    def _market_data_intervals(self) -> list[str]:
        return self._app._market_data_intervals()

    def _should_persist_candle_interval(self, interval: str) -> bool:
        return self._app._should_persist_candle_interval(interval)

    def _new_candle_store(self) -> Any:
        return self._app._new_candle_store()

    def _ws_topics_for_symbol(self, symbol: str) -> list[str]:
        return self._app._ws_topics_for_symbol(symbol)

    def _update_model_gate_quality_from_diag(self, diag: dict[str, Any]) -> None:
        self._app._modules.signal_policy.update_model_gate_quality_from_diag(diag)

    async def _evaluate_feature_drift(self) -> dict[str, Any]:
        return await self._app._modules.training.evaluate_feature_drift()

    async def _run_model_training(self, min_samples: int, horizon: int, label_bps: float) -> None:
        await self._app._modules.training.run_model_training(min_samples, horizon, label_bps)

    async def _get_champion_walk_forward_bps(self) -> float:
        return await self._app._get_champion_walk_forward_bps()
