"""Runtime module protocol and background-task helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, Protocol, runtime_checkable

from trader.modules.base import AppBoundModule


@runtime_checkable
class RuntimeModule(Protocol):
    """Pluggable background subsystem started by ``TradingApplication``."""

    @property
    def name(self) -> str: ...

    def spawn_background_tasks(self, tasks: list[asyncio.Task[object]]) -> None: ...


class ModuleTaskMixin(AppBoundModule):
    """Helper for modules that register asyncio background loops."""

    def _spawn(
        self,
        tasks: list[asyncio.Task[object]],
        coro: Coroutine[Any, Any, object],
        task_name: str,
    ) -> None:
        tasks.append(asyncio.create_task(coro, name=task_name))
