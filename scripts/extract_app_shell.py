#!/usr/bin/env python3
"""Single-pass extraction: lifecycle boot/shutdown + operator controls."""

from __future__ import annotations

import ast
import re
from pathlib import Path

APP = Path("src/trader/app.py")

LIFECYCLE = {
    "_load_settings": "load_settings",
    "_configure_observability": "configure_observability",
    "_run_preflight": "run_preflight",
    "_start_trade_journal": "start_trade_journal",
    "_restore_execution_pending_entries": "restore_execution_pending_entries",
    "_start_http_server": "start_http_server",
    "_start_bybit_adapter": "start_bybit_adapter",
    "_start_feature_pipeline": "start_feature_pipeline",
    "_graceful_shutdown": "graceful_shutdown",
}

OPERATOR = {
    "_pause_trading": "pause_trading",
    "_resume_trading": "resume_trading",
    "_set_shadow_mode": "set_shadow_mode",
    "_change_risk_profile": "change_risk_profile",
    "_emergency_stop": "emergency_stop",
    "_start_model_training": "start_model_training",
    "_start_model_training_all": "start_model_training_all",
    "_start_model_promote": "start_model_promote",
    "_runtime_settings": "runtime_settings",
    "_set_runtime_setting": "set_runtime_setting",
    "_symbol_candidates": "symbol_candidates",
    "_selected_symbols": "selected_symbols",
    "_toggle_manual_symbol": "toggle_manual_symbol",
}

LIFECYCLE_DELEGATES = {
    "_load_settings": "await self._modules.lifecycle.load_settings()",
    "_configure_observability": "await self._modules.lifecycle.configure_observability()",
    "_run_preflight": "await self._modules.lifecycle.run_preflight()",
    "_start_trade_journal": "await self._modules.lifecycle.start_trade_journal()",
    "_restore_execution_pending_entries": "await self._modules.lifecycle.restore_execution_pending_entries()",
    "_start_http_server": "return await self._modules.lifecycle.start_http_server()",
    "_start_bybit_adapter": "await self._modules.lifecycle.start_bybit_adapter()",
    "_start_feature_pipeline": "await self._modules.lifecycle.start_feature_pipeline()",
    "_graceful_shutdown": "await self._modules.lifecycle.graceful_shutdown()",
}

OPERATOR_DELEGATES = {
    "_pause_trading": "await self._modules.operator.pause_trading()",
    "_resume_trading": "await self._modules.operator.resume_trading()",
    "_set_shadow_mode": "await self._modules.operator.set_shadow_mode(enabled)",
    "_change_risk_profile": "await self._modules.operator.change_risk_profile(profile)",
    "_emergency_stop": "await self._modules.operator.emergency_stop()",
    "_start_model_training": "return await self._modules.operator.start_model_training(min_samples, horizon, label_bps)",
    "_start_model_training_all": "return await self._modules.operator.start_model_training_all()",
    "_start_model_promote": "return await self._modules.operator.start_model_promote(version)",
    "_runtime_settings": "return self._modules.operator.runtime_settings()",
    "_set_runtime_setting": "return await self._modules.operator.set_runtime_setting(key, value)",
    "_symbol_candidates": "return self._modules.operator.symbol_candidates()",
    "_selected_symbols": "return self._modules.operator.selected_symbols()",
    "_toggle_manual_symbol": "return await self._modules.operator.toggle_manual_symbol(symbol)",
}


def method_ranges(tree: ast.Module) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TradingApplication":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[item.name] = (item.lineno, item.end_lineno or item.lineno)
    return out


def slice_lines(lines: list[str], start: int, end: int) -> str:
    return "".join(lines[start - 1 : end])


def transform(body: str, *, module: str) -> str:
    body = re.sub(r"\bself\.(?!_app\.)", "self._app.", body)
    replacements = {
        "self._app._initial_shadow_mode()": "self._initial_shadow_mode()",
        "self._app._active_execution_allowed()": "self._active_execution_allowed()",
        "self._app._refresh_balance()": "await self._refresh_balance()",
        "self._app._init_risk_manager(": "await self._init_risk_manager(",
        "self._app._run_model_training(": "await self._run_model_training(",
        "self._app._run_model_training_all()": "await self._run_model_training_all()",
        "self._app._maybe_run_startup_retention()": "await self._maybe_run_startup_retention()",
        "self._app._run_trade_journal_reconnector()": "await self._run_trade_journal_reconnector()",
        "self._app._on_screener_symbols_added(": "await self._on_screener_symbols_added(",
        "self._app._active_symbols()": "self._active_symbols()",
        "self._app._runtime_settings": "self.runtime_settings",
        "self._app._set_runtime_setting": "self.set_runtime_setting",
        "self._app._selected_symbols()": "self.selected_symbols()",
        "self._app._symbol_candidates()": "self.symbol_candidates()",
    }
    for old, new in replacements.items():
        body = body.replace(old, new)
    if module == "operator":
        body = body.replace("self._app._runtime_settings()", "self.runtime_settings()")
        body = body.replace("self._app._set_runtime_setting(", "await self.set_runtime_setting(")
    if module == "lifecycle":
        body = body.replace(
            "runtime_settings=self._runtime_settings,\n            set_runtime_setting=self._set_runtime_setting,",
            "runtime_settings=self._app._modules.operator.runtime_settings,\n            set_runtime_setting=self._app._modules.operator.set_runtime_setting,",
        )
    return body


def rename_def(body: str, old: str, new: str) -> str:
    return re.sub(rf"def {re.escape(old)}\(", f"def {new}(", body, count=1)


def build_module(
    class_name: str,
    mapping: dict[str, str],
    ranges: dict[str, tuple[int, int]],
    lines: list[str],
    module: str,
    header: str,
) -> str:
    chunks: list[str] = []
    for old, new in mapping.items():
        start, end = ranges[old]
        chunk = slice_lines(lines, start, end)
        chunk = transform(chunk, module=module)
        chunk = rename_def(chunk, old, new)
        chunks.append(chunk.rstrip() + "\n")
    return header + "\n".join(chunks)


def delegate_body(old: str, delegate: str, sig_line: str) -> str:
    is_async = "async def" in sig_line
    params = sig_line.split("def ", 1)[1]
    if is_async:
        return f"    async def {params}\n        {delegate}\n"
    return f"    def {params}\n        {delegate}\n"


def patch_app(
    lines: list[str], ranges: dict[str, tuple[int, int]], mapping: dict[str, str], delegates: dict[str, str]
) -> None:
    for old in sorted(mapping, key=lambda k: ranges[k][0], reverse=True):
        start, end = ranges[old]
        sig_line = lines[start - 1]
        body = delegate_body(old, delegates[old], sig_line)
        lines[start - 1 : end] = [body]


def main() -> None:
    text = APP.read_text()
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    ranges = method_ranges(tree)

    lifecycle_header = '''"""Application lifecycle: settings, preflight, HTTP, adapters, shutdown."""

from __future__ import annotations

import asyncio
import os
import secrets
import signal
from typing import Any

import uvicorn

from trader.domain.enums import SystemStatus, TradingMode
from trader.modules.base import AppBoundModule
from trader.monitoring.logging import configure_logging, get_logger
from trader.runtime.constants import _WS_INTERVAL
from trader.runtime.state_proxy import _AppStateProxy

log = get_logger(__name__)


class LifecycleModule(AppBoundModule):
    name = "lifecycle"

    def _initial_shadow_mode(self) -> bool:
        return self._app._modules.signal_policy.initial_shadow_mode()

'''
    operator_header = '''"""Operator controls: pause/shadow/risk, runtime limits, symbol selection, train/promote."""

from __future__ import annotations

import asyncio
import html
import json
from decimal import Decimal
from typing import Any, cast

from trader.domain.enums import TradingMode
from trader.modules.base import AppBoundModule
from trader.monitoring.logging import get_logger
from trader.runtime.constants import _SYMBOLS

log = get_logger(__name__)


class OperatorControlsModule(AppBoundModule):
    name = "operator"

    async def _refresh_balance(self) -> Decimal:
        return await self._app._modules.execution.refresh_balance()

    async def _init_risk_manager(self, initial_capital: Decimal) -> None:
        await self._app._modules.execution.init_risk_manager(initial_capital)

    async def _run_model_training(self, min_samples: int, horizon: int, label_bps: float) -> None:
        await self._app._modules.training.run_model_training(min_samples, horizon, label_bps)

    async def _run_model_training_all(self) -> None:
        await self._app._modules.training.run_model_training_all()

    def _active_execution_allowed(self) -> bool:
        return self._app._modules.signal_policy.active_execution_allowed()

    async def _on_screener_symbols_added(self, symbols: list[str]) -> None:
        await self._app._modules.market_data.on_screener_symbols_added(symbols)

'''
    lifecycle_path = Path("src/trader/modules/lifecycle.py")
    operator_path = Path("src/trader/modules/operator_controls.py")
    lifecycle_path.write_text(build_module("LifecycleModule", LIFECYCLE, ranges, lines, "lifecycle", lifecycle_header))
    operator_path.write_text(
        build_module("OperatorControlsModule", OPERATOR, ranges, lines, "operator", operator_header)
    )

    all_mapping = {**LIFECYCLE, **OPERATOR}
    all_delegates = {**LIFECYCLE_DELEGATES, **OPERATOR_DELEGATES}
    patch_app(lines, ranges, all_mapping, all_delegates)
    new_app = "".join(lines)

    if "LifecycleModule" not in new_app:
        pass  # delegates only use registry

    APP.write_text(new_app)
    print(f"Wrote {lifecycle_path}, {operator_path}, patched {APP}")


if __name__ == "__main__":
    main()
