#!/usr/bin/env python3
"""Extract diagnostics/telegram modules from app.py."""

from __future__ import annotations

import ast
import re
from pathlib import Path

APP = Path("src/trader/app.py")

DIAGNOSTICS = {
    "_dict_or_empty": ("dict_or_empty", True),
    "_float_or_none": ("float_or_none", True),
    "_utc_age_seconds": ("utc_age_seconds", True),
    "_economic_readiness_report": ("economic_readiness_report", False),
    "_enforce_economic_readiness_for_active": ("enforce_economic_readiness_for_active", False),
    "_record_diag": ("record", False),
    "_top_blocker_from_diag": ("top_blocker_from_diag", False),
    "_check_zero_trading": ("check_zero_trading", False),
    "_runtime_candle_readiness_counts": ("runtime_candle_readiness_counts", False),
    "_merge_runtime_db_diag_fallbacks": ("merge_db_fallbacks", False),
    "get_diagnostics": ("get_snapshot", False),
}

TELEGRAM = {
    "_resolve_telegram_delivery": ("resolve_delivery", False),
    "_start_telegram_bot": ("start", False),
}

DELEGATES = {
    "_dict_or_empty": "return DiagnosticsModule.dict_or_empty(value)",
    "_float_or_none": "return DiagnosticsModule.float_or_none(value)",
    "_utc_age_seconds": "return DiagnosticsModule.utc_age_seconds(value)",
    "_economic_readiness_report": (
        "return self._modules.diagnostics.economic_readiness_report(db_diag=db_diag, runtime_diag=runtime_diag)"
    ),
    "_enforce_economic_readiness_for_active": "await self._modules.diagnostics.enforce_economic_readiness_for_active()",
    "_resolve_telegram_delivery": "return self._modules.telegram.resolve_delivery()",
    "_start_telegram_bot": "await self._modules.telegram.start()",
    "_record_diag": "self._modules.diagnostics.record(event)",
    "_top_blocker_from_diag": "return self._modules.diagnostics.top_blocker_from_diag(diag, default=default)",
    "_check_zero_trading": "self._modules.diagnostics.check_zero_trading()",
    "_runtime_candle_readiness_counts": "return self._modules.diagnostics.runtime_candle_readiness_counts()",
    "_merge_runtime_db_diag_fallbacks": "self._modules.diagnostics.merge_db_fallbacks(diag)",
    "get_diagnostics": "return self._modules.diagnostics.get_snapshot()",
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


def transform(body: str, *, kind: str) -> str:
    body = re.sub(r"\bself\.(?!_app\.)", "self._app.", body)
    if kind == "diag":
        body = body.replace("self._app.get_diagnostics()", "self.get_snapshot()")
        body = body.replace("self._app._merge_runtime_db_diag_fallbacks(", "self.merge_db_fallbacks(")
        body = body.replace("self._app._top_blocker_from_diag(", "self.top_blocker_from_diag(")
        body = body.replace("self._app._economic_readiness_report(", "self.economic_readiness_report(")
        body = body.replace("self._app._dict_or_empty(", "DiagnosticsModule.dict_or_empty(")
        body = body.replace("self._app._float_or_none(", "DiagnosticsModule.float_or_none(")
        body = body.replace("self._app._utc_age_seconds(", "DiagnosticsModule.utc_age_seconds(")
        body = body.replace("self._app._runtime_candle_readiness_counts(", "self.runtime_candle_readiness_counts(")
        body = body.replace("self._dict_or_empty(", "DiagnosticsModule.dict_or_empty(")
        body = body.replace("self._float_or_none(", "DiagnosticsModule.float_or_none(")
        body = body.replace("self._utc_age_seconds(", "DiagnosticsModule.utc_age_seconds(")
    else:
        body = body.replace("self._app.get_diagnostics", "self._app._modules.diagnostics.get_snapshot")
        body = body.replace(
            "self._app._merge_runtime_db_diag_fallbacks",
            "self._app._modules.diagnostics.merge_db_fallbacks",
        )
        body = body.replace(
            "self._app._top_blocker_from_diag",
            "self._app._modules.diagnostics.top_blocker_from_diag",
        )
        body = body.replace("self._app._resolve_telegram_delivery()", "self.resolve_delivery()")
        body = body.replace(
            "enrich_db_diag_fallbacks=self._app._merge_runtime_db_diag_fallbacks",
            "enrich_db_diag_fallbacks=self._app._modules.diagnostics.merge_db_fallbacks",
        )
        body = body.replace(
            "diagnostics_provider=self._app.get_diagnostics",
            "diagnostics_provider=self._app._modules.diagnostics.get_snapshot",
        )
    return body


def rename_def(body: str, old: str, new: str) -> str:
    return re.sub(rf"^(    (?:async )?def ){old}\b", rf"\1{new}", body, count=1, flags=re.M)


def add_static_decorators(body: str) -> str:
    return re.sub(
        r"^    def (dict_or_empty|float_or_none|utc_age_seconds)\b",
        r"    @staticmethod\n    def \1",
        body,
        flags=re.M,
    )


def build_module(class_name: str, header: str, mapping: dict[str, tuple[str, bool]], ranges: dict[str, tuple[int, int]], all_lines: list[str], kind: str) -> str:
    out = header
    for old, (new, is_static) in mapping.items():
        start, end = ranges[old]
        chunk = slice_lines(all_lines, start, end)
        chunk = transform(chunk, kind=kind)
        chunk = rename_def(chunk, old, new)
        if is_static:
            chunk = re.sub(r"^    @staticmethod\n", "", chunk, flags=re.M)
            chunk = add_static_decorators(chunk)
        out += chunk + "\n"
    return out


def patch_app_delegates(all_lines: list[str], tree: ast.Module) -> list[str]:
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
            body_start = item.body[0].lineno - 1
            end = item.end_lineno or item.lineno
            stmt = DELEGATES[item.name]
            replacements.append((body_start, end, f"        {stmt}\n"))
    for body_start, end, new_body in sorted(replacements, key=lambda x: x[0], reverse=True):
        all_lines[body_start:end] = [new_body]
    return all_lines


def main() -> None:
    text = APP.read_text()
    all_lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    ranges = method_ranges(tree)

    diag_header = '''"""Operator diagnostics and readiness reporting."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any

from trader.domain.enums import TradingMode
from trader.modules.base import AppBoundModule
from trader.monitoring.logging import get_logger
from trader.runtime.constants import _DIAG_WINDOW, _SYMBOLS

log = get_logger(__name__)


class DiagnosticsModule(AppBoundModule):
    name = "diagnostics"

'''
    tg_header = '''"""Telegram operator bridge: bot startup and provider wiring."""

from __future__ import annotations

import asyncio
import html
import os
import time
from datetime import UTC, datetime
from typing import Any, cast

from trader.modules.base import AppBoundModule
from trader.monitoring.logging import get_logger
from trader.runtime.constants import _SYMBOLS, _WS_INTERVAL

log = get_logger(__name__)


class TelegramBridgeModule(AppBoundModule):
    name = "telegram"

'''
    Path("src/trader/modules/diagnostics.py").write_text(
        build_module("DiagnosticsModule", diag_header, DIAGNOSTICS, ranges, all_lines, "diag")
    )
    Path("src/trader/modules/telegram_bridge.py").write_text(
        build_module("TelegramBridgeModule", tg_header, TELEGRAM, ranges, all_lines, "tg")
    )

    patched = patch_app_delegates(all_lines, tree)
    patched_text = "".join(patched)
    if "from trader.modules.diagnostics import DiagnosticsModule" not in patched_text:
        patched_text = patched_text.replace(
            "from trader.modules.registry import ModuleRegistry\n",
            "from trader.modules.diagnostics import DiagnosticsModule\nfrom trader.modules.registry import ModuleRegistry\n",
        )
    APP.write_text(patched_text)

    reg_path = Path("src/trader/modules/registry.py")
    reg = reg_path.read_text()
    if "DiagnosticsModule" not in reg:
        reg = reg.replace(
            "from trader.modules.market_data import MarketDataModule\n",
            "from trader.modules.diagnostics import DiagnosticsModule\nfrom trader.modules.market_data import MarketDataModule\n",
        )
        reg = reg.replace(
            "from trader.modules.training import TrainingModule\n",
            "from trader.modules.telegram_bridge import TelegramBridgeModule\nfrom trader.modules.training import TrainingModule\n",
        )
        reg = reg.replace(
            "        self.training = TrainingModule(app)\n",
            "        self.training = TrainingModule(app)\n        self.diagnostics = DiagnosticsModule(app)\n        self.telegram = TelegramBridgeModule(app)\n",
        )
        reg_path.write_text(reg)

    init_path = Path("src/trader/modules/__init__.py")
    init = init_path.read_text()
    if "DiagnosticsModule" not in init:
        init = init.replace(
            "from trader.modules.market_data import MarketDataModule\n",
            "from trader.modules.diagnostics import DiagnosticsModule\nfrom trader.modules.market_data import MarketDataModule\n",
        )
        init = init.replace(
            "from trader.modules.training import TrainingModule\n",
            "from trader.modules.telegram_bridge import TelegramBridgeModule\nfrom trader.modules.training import TrainingModule\n",
        )
        init = init.replace(
            '"MarketDataModule",\n',
            '"DiagnosticsModule",\n    "MarketDataModule",\n',
        )
        init = init.replace(
            '"TrainingModule",\n',
            '"TelegramBridgeModule",\n    "TrainingModule",\n',
        )
        init_path.write_text(init)

    base_path = Path("src/trader/modules/base.py")
    base = base_path.read_text()
    base = base.replace(
        "        self._app._record_diag(event)",
        "        self._app._modules.diagnostics.record(event)",
    )
    base_path.write_text(base)
    print("done")


if __name__ == "__main__":
    main()
