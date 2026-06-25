#!/usr/bin/env python3
"""Final extraction: execution runtime (private WS, risk stack, positions)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

APP = Path("src/trader/app.py")

EXECUTION = {
    "_init_risk_manager": "init_risk_manager",
    "_refresh_balance": "refresh_balance",
    "_init_execution_engine": "init_execution_engine",
    "_start_private_ws": "start_private_ws",
    "_run_risk_monitor": "run_risk_monitor",
    "_maybe_recover_stale_ws": "maybe_recover_stale_ws",
    "_refresh_closed_pnl_memory": "refresh_closed_pnl_memory",
    "_manage_open_positions": "manage_open_positions",
    "_sync_execution_positions": "sync_execution_positions",
    "_cache_exchange_positions": "cache_exchange_positions",
    "_cache_exchange_position_update": "cache_exchange_position_update",
    "_recent_exchange_positions": "recent_exchange_positions",
    "_effective_performance_blocks": "effective_performance_blocks",
    "_activation_price": "activation_price",
    "_breakeven_stop": "breakeven_stop",
    "_round_to_tick": "round_to_tick",
}

DELEGATES = {
    "_init_risk_manager": "await self._modules.execution.init_risk_manager(initial_capital)",
    "_refresh_balance": "return await self._modules.execution.refresh_balance()",
    "_init_execution_engine": "await self._modules.execution.init_execution_engine()",
    "_start_private_ws": "await self._modules.execution.start_private_ws()",
    "_run_risk_monitor": "await self._modules.execution.run_risk_monitor()",
    "_maybe_recover_stale_ws": "await self._modules.execution.maybe_recover_stale_ws(market_data_age_s)",
    "_refresh_closed_pnl_memory": "await self._modules.execution.refresh_closed_pnl_memory()",
    "_manage_open_positions": "await self._modules.execution.manage_open_positions()",
    "_sync_execution_positions": "await self._modules.execution.sync_execution_positions()",
    "_cache_exchange_positions": "self._modules.execution.cache_exchange_positions(positions)",
    "_cache_exchange_position_update": "self._modules.execution.cache_exchange_position_update(position)",
    "_recent_exchange_positions": "return self._modules.execution.recent_exchange_positions()",
    "_effective_performance_blocks": "return self._modules.execution.effective_performance_blocks(active_symbols)",
    "_activation_price": "return self._modules.execution.activation_price(entry_price, side)",
    "_breakeven_stop": "return self._modules.execution.breakeven_stop(entry_price, side, fee_rates)",
    "_round_to_tick": "return self._modules.execution.round_to_tick(price, tick_size, round_up)",
}


def method_ranges(tree: ast.Module) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TradingApplication":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[item.name] = (item.lineno, item.end_lineno or item.lineno)
    return out


def slice_lines(all_lines: list[str], start: int, end: int) -> str:
    return "".join(all_lines[start - 1 : end])


def transform(body: str) -> str:
    body = re.sub(r"\bself\.(?!_app\.)", "self._app.", body)
    replacements = {
        "self._app._activation_price(": "self.activation_price(",
        "self._app._breakeven_stop(": "self.breakeven_stop(",
        "self._app._round_to_tick(": "self.round_to_tick(",
        "self._app._cache_exchange_position_update(": "self.cache_exchange_position_update(",
        "self._app._recent_exchange_positions(": "self.recent_exchange_positions(",
        "self._app._maybe_recover_stale_ws(": "self.maybe_recover_stale_ws(",
    }
    for old, new in replacements.items():
        body = body.replace(old, new)
    return body


def rename_def(body: str, old: str, new: str) -> str:
    return re.sub(rf"^(    (?:async )?def ){old}\b", rf"\1{new}", body, count=1, flags=re.M)


def build_module(ranges: dict[str, tuple[int, int]], all_lines: list[str]) -> str:
    header = '''"""Execution runtime: risk stack, private WS, position management."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal
from typing import Any
from uuid import UUID

from trader.domain.enums import TradingMode
from trader.modules.base import AppBoundModule
from trader.monitoring.logging import get_logger
from trader.runtime.constants import _BALANCE_REFRESH_INTERVAL, _FALLBACK_BALANCE_USD, _JOURNAL_FALLBACK_UUID

log = get_logger(__name__)


class ExecutionRuntimeModule(AppBoundModule):
    name = "execution"

'''
    out = header
    for old, new in EXECUTION.items():
        start, end = ranges[old]
        chunk = slice_lines(all_lines, start, end)
        chunk = transform(chunk)
        chunk = rename_def(chunk, old, new)
        out += chunk + "\n"
    return out


def patch_app(all_lines: list[str], tree: ast.Module) -> list[str]:
    replacements: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "TradingApplication":
            continue
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if item.name not in DELEGATES:
                continue
            if not item.body:
                continue
            replacements.append(
                (item.body[0].lineno - 1, item.end_lineno or item.lineno, f"        {DELEGATES[item.name]}\n")
            )
    for body_start, end, new_body in sorted(replacements, key=lambda x: x[0], reverse=True):
        all_lines[body_start:end] = [new_body]
    return all_lines


def patch_registry() -> None:
    reg_path = Path("src/trader/modules/registry.py")
    reg = reg_path.read_text()
    if "ExecutionRuntimeModule" in reg:
        return
    reg = reg.replace(
        "from trader.modules.diagnostics import DiagnosticsModule\n",
        "from trader.modules.diagnostics import DiagnosticsModule\nfrom trader.modules.execution_runtime import ExecutionRuntimeModule\n",
    )
    reg = reg.replace(
        "        self.diagnostics = DiagnosticsModule(app)\n",
        "        self.diagnostics = DiagnosticsModule(app)\n        self.execution = ExecutionRuntimeModule(app)\n",
    )
    reg_path.write_text(reg)

    init = Path("src/trader/modules/__init__.py").read_text()
    if "ExecutionRuntimeModule" not in init:
        init = init.replace(
            "from trader.modules.diagnostics import DiagnosticsModule\n",
            "from trader.modules.diagnostics import DiagnosticsModule\nfrom trader.modules.execution_runtime import ExecutionRuntimeModule\n",
        )
        init = init.replace(
            '"DiagnosticsModule",\n',
            '"DiagnosticsModule",\n    "ExecutionRuntimeModule",\n',
        )
        Path("src/trader/modules/__init__.py").write_text(init)


def main() -> None:
    text = APP.read_text()
    all_lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    ranges = method_ranges(tree)

    Path("src/trader/modules/execution_runtime.py").write_text(build_module(ranges, all_lines))
    patched = patch_app(all_lines, tree)
    APP.write_text("".join(patched))
    patch_registry()
    print("done")


if __name__ == "__main__":
    main()
