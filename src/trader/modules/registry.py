"""Register and spawn pluggable runtime modules."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from trader.modules.market_data import MarketDataModule
from trader.modules.ops import OpsModule
from trader.modules.training import TrainingModule
from trader.runtime.supervisor import RuntimeSupervisor

if TYPE_CHECKING:
    from trader.app import TradingApplication


class ModuleRegistry:
    """Wires background loops from focused runtime modules."""

    def __init__(self, app: TradingApplication) -> None:
        self._app = app
        self.ops = OpsModule(app)
        self.market_data = MarketDataModule(app)
        self.training = TrainingModule(app)
        self.supervisor = RuntimeSupervisor(app)

    def spawn_background_tasks(self, tasks: list[asyncio.Task[object]]) -> None:
        self.ops.spawn_background_tasks(tasks)
        self.market_data.spawn_background_tasks(tasks)
        self.training.spawn_background_tasks(tasks)
        tasks.append(asyncio.create_task(self.supervisor.run(), name="supervisor"))
