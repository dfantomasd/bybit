#!/usr/bin/env python3
"""Replace extracted TradingApplication method bodies with module delegates."""

from __future__ import annotations

import ast
import re
from pathlib import Path

APP_PATH = Path("src/trader/app.py")

DELEGATE_TARGETS = {
    "_maybe_run_startup_retention": "self._modules.ops.maybe_run_startup_retention",
    "_run_trade_journal_reconnector": "self._modules.ops.run_trade_journal_reconnector",
    "_run_data_retention": "self._modules.ops.run_data_retention",
    "_run_outcome_resolver": "self._modules.ops.run_outcome_resolver",
    "_run_reconciliation": "self._modules.ops.run_reconciliation",
    "_run_transaction_log_sync": "self._modules.ops.run_transaction_log_sync",
    "_sync_transaction_log": "self._modules.ops.sync_transaction_log",
    "_on_screener_symbols_added": "self._modules.market_data.on_screener_symbols_added",
    "_on_screener_symbols_removed": "self._modules.market_data.on_screener_symbols_removed",
    "_start_screener": "self._modules.market_data.start_screener",
    "_seed_candle_store": "self._modules.market_data.seed_candle_store",
    "_reconcile_unconfirmed_candles": "self._modules.market_data.reconcile_unconfirmed_candles",
    "_run_startup_backfill": "self._modules.market_data.run_startup_backfill",
    "_startup_backfill": "self._modules.market_data.startup_backfill",
    "_start_public_ws": "self._modules.market_data.start_public_ws",
    "_run_load_governor": "self._modules.market_data.run_load_governor",
    "_run_symbol_subscribe_watchdog": "self._modules.market_data.run_symbol_subscribe_watchdog",
    "_run_auto_model_trainer": "self._modules.training.run_auto_model_trainer",
    "_run_auto_model_promoter": "self._modules.training.run_auto_model_promoter",
    "_run_model_progress_reporter": "self._modules.training.run_model_progress_reporter",
    "_run_bucket_stats_refresher": "self._modules.training.run_bucket_stats_refresher",
    "_evaluate_feature_drift": "self._modules.training.evaluate_feature_drift",
    "_maybe_apply_online_learning": "self._modules.training.maybe_apply_online_learning",
    "_run_model_training": "self._modules.training.run_model_training",
    "_run_model_training_all": "self._modules.training.run_model_training_all",
    "_run_supervisor": "self._modules.supervisor.run",
    "_start_strategy_loop": "self._trading_loop.start",
}

CONSTANTS_IMPORT = """from trader.runtime.constants import (
    BALANCE_REFRESH_INTERVAL,
    CRITICAL_TASK_NAMES,
    DIAG_WINDOW,
    FALLBACK_BALANCE_USD,
    FEATURE_INTERVAL,
    INTERVAL_MS,
    JOURNAL_FALLBACK_UUID,
    ML_REPLACEMENT_COUNTER,
    MIN_SEED_BARS,
    STRATEGY_LOOP_INTERVAL,
    SUPERVISOR_CHECK_INTERVAL,
    SUPERVISOR_HEARTBEAT_INTERVAL,
    SYMBOLS,
    TRADE_JOURNAL_RECONNECT_INTERVAL,
    TRAINING_HEARTBEAT_SECONDS,
    TRAINING_TIMEOUT_SECONDS,
    WS_INTERVAL,
    _BALANCE_REFRESH_INTERVAL,
    _CRITICAL_TASK_NAMES,
    _DIAG_WINDOW,
    _FALLBACK_BALANCE_USD,
    _FEATURE_INTERVAL,
    _INTERVAL_MS,
    _JOURNAL_FALLBACK_UUID,
    _ML_REPLACEMENT_COUNTER,
    _MIN_SEED_BARS,
    _STRATEGY_LOOP_INTERVAL,
    _SUPERVISOR_CHECK_INTERVAL,
    _SUPERVISOR_HEARTBEAT_INTERVAL,
    _SYMBOLS,
    _TRADE_JOURNAL_RECONNECT_INTERVAL,
    _TRAINING_HEARTBEAT_SECONDS,
    _TRAINING_TIMEOUT_SECONDS,
    _WS_INTERVAL,
)
from trader.runtime.state_proxy import AppStateProxy, _AppStateProxy
from trader.modules.registry import ModuleRegistry
from trader.modules.trading_loop import TradingLoopModule
"""

MODULE_INIT = """        self._modules = ModuleRegistry(self)
        self._trading_loop = TradingLoopModule(self)
"""

RUN_REPLACEMENT = """            # Supervisor + background loops via pluggable modules
            self._modules.spawn_background_tasks(self._background_tasks)

            # Risk monitor: updates equity/drawdown, checks WS staleness
            risk_monitor_task = asyncio.create_task(self._run_risk_monitor(), name="risk-monitor")
            self._background_tasks.append(risk_monitor_task)
"""


def _call_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = [a.arg for a in node.args.args if a.arg != "self"]
    parts: list[str] = []
    defaults = node.args.defaults
    num_defaults = len(defaults)
    num_args = len(node.args.args) - 1  # exclude self
    first_default_idx = num_args - num_defaults
    for i, name in enumerate(args):
        if i >= first_default_idx and num_defaults:
            parts.append(name)
        else:
            parts.append(name)
    return ", ".join(parts)


def _delegate_body(node: ast.FunctionDef | ast.AsyncFunctionDef, target: str) -> str:
    call_args = _call_args(node)
    call = f"{target}({call_args})" if call_args else f"{target}()"
    indent = "        "
    if isinstance(node, ast.AsyncFunctionDef):
        if node.name == "_start_screener":
            return f"{indent}return await {call}\n"
        if node.name == "_evaluate_feature_drift":
            return f"{indent}return await {call}\n"
        return f"{indent}await {call}\n"
    return f"{indent}return {call}\n"


def replace_method_bodies(lines: list[str], tree: ast.Module) -> list[str]:
    replacements: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "TradingApplication":
            continue
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if item.name not in DELEGATE_TARGETS:
                continue
            # Keep def line and optional docstring; replace body
            start = item.lineno - 1
            body_start = start + 1
            # Skip docstring
            if (
                item.body
                and isinstance(item.body[0], ast.Expr)
                and isinstance(item.body[0].value, ast.Constant)
                and isinstance(item.body[0].value.value, str)
            ):
                body_start = item.body[0].end_lineno
            end = item.end_lineno
            header = lines[start:body_start]
            new_body = _delegate_body(item, DELEGATE_TARGETS[item.name])
            replacements.append((body_start, end, header, new_body))

    # Apply from bottom to top
    for body_start, end, header, new_body in sorted(replacements, key=lambda x: x[0], reverse=True):
        lines[body_start:end] = [new_body]

    return lines


def main() -> None:
    text = APP_PATH.read_text()
    lines = text.splitlines(keepends=True)

    # Remove constants block (lines 47-96 approx) - match from _SYMBOLS to _CRITICAL_TASK_NAMES block
    const_start = None
    const_end = None
    for i, line in enumerate(lines):
        if line.startswith("_SYMBOLS = "):
            const_start = i
        if const_start is not None and line.startswith("_CRITICAL_TASK_NAMES = frozenset"):
            # find closing paren
            j = i
            while j < len(lines) and ")" not in lines[j]:
                j += 1
            const_end = j + 1
            break

    if const_start is None or const_end is None:
        raise SystemExit("Could not find constants block")

    # Insert imports after log = get_logger line
    log_idx = next(i for i, l in enumerate(lines) if l.startswith("log = get_logger"))
    lines[const_start:const_end] = []
    # adjust log_idx after deletion
    if log_idx > const_end:
        log_idx -= const_end - const_start
    elif log_idx > const_start:
        log_idx = const_start

    lines.insert(log_idx + 1, "\n" + CONSTANTS_IMPORT + "\n")

    # Add module init after _online_learning_updates_since_checkpoint line
    init_marker = "        self._online_learning_updates_since_checkpoint: int = 0\n"
    text = "".join(lines)
    if "self._modules = ModuleRegistry" not in text:
        text = text.replace(init_marker, init_marker + MODULE_INIT)

    # Replace _make_state_proxy body
    text = re.sub(
        r"(    def _make_state_proxy\(self\) -> _AppStateProxy:\n)(        return _AppStateProxy\(self\)\n)",
        r"\1        return AppStateProxy(self)\n",
        text,
    )

    # Remove _AppStateProxy class at end (keep main functions)
    proxy_start = text.find("\nclass _AppStateProxy:")
    main_start = text.find("\nasync def main() -> None:")
    if proxy_start != -1 and main_start != -1 and proxy_start < main_start:
        text = text[:proxy_start] + text[main_start:]

    lines = text.splitlines(keepends=True)
    tree = ast.parse("".join(lines))
    lines = replace_method_bodies(lines, tree)

    text = "".join(lines)

    # Replace run() background task block
    old_run_block = re.search(
        r"            await self\._start_strategy_loop\(\)\n\n"
        r"            # Supervisor monitors critical tasks.*?"
        r"            self\._background_tasks\.append\(subscribe_watchdog_task\)\n\n",
        text,
        re.DOTALL,
    )
    if not old_run_block:
        raise SystemExit("Could not find run() background tasks block")
    text = text[: old_run_block.start()] + (
        "            await self._trading_loop.start()\n\n" + RUN_REPLACEMENT + "\n"
    ) + text[old_run_block.end() :]

    # Update startup calls in run to use modules (optional - delegates work via app methods)
    APP_PATH.write_text(text)
    print(f"Updated {APP_PATH}")


if __name__ == "__main__":
    main()
