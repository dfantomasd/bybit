#!/usr/bin/env python3
"""Extract signal policy + shadow gates from app.py into signal_policy module."""

from __future__ import annotations

import ast
import re
from pathlib import Path

APP = Path("src/trader/app.py")

SIGNAL_POLICY = {
    "_active_execution_allowed": "active_execution_allowed",
    "_initial_shadow_mode": "initial_shadow_mode",
    "_is_scalp_profile": "is_scalp_profile",
    "_scalp_strict_shadow": "scalp_strict_shadow",
    "_expectancy_gates_apply": "expectancy_gates_apply",
    "_model_gate_threshold": "model_gate_threshold",
    "_update_model_gate_quality_from_diag": "update_model_gate_quality_from_diag",
    "_model_gate_quality_allows_canary": "model_gate_quality_allows_canary",
    "_model_gate_canary_blocks": "model_gate_canary_blocks",
    "_feature_values_for_side": "feature_values_for_side",
    "_sample_confirmed_candle": "sample_confirmed_candle",
    "_bucket_blocked": "bucket_blocked",
    "_symbol_side_blocked": "symbol_side_blocked",
    "_record_shadow_close": "record_shadow_close",
    "_shadow_exit_hit": "shadow_exit_hit",
    "_shadow_pnl_pct": "shadow_pnl_pct",
    "_shadow_loss_guard_blocks": "shadow_loss_guard_blocks",
    "_trend_confirmation_intervals": "trend_confirmation_intervals",
    "_trend_mtf_confirmed": "trend_mtf_confirmed",
}

DELEGATES = {
    "_active_execution_allowed": "return self._modules.signal_policy.active_execution_allowed()",
    "_initial_shadow_mode": "return self._modules.signal_policy.initial_shadow_mode()",
    "_is_scalp_profile": "return self._modules.signal_policy.is_scalp_profile()",
    "_scalp_strict_shadow": "return self._modules.signal_policy.scalp_strict_shadow()",
    "_expectancy_gates_apply": "return self._modules.signal_policy.expectancy_gates_apply()",
    "_model_gate_threshold": "return self._modules.signal_policy.model_gate_threshold(regime_context)",
    "_update_model_gate_quality_from_diag": "self._modules.signal_policy.update_model_gate_quality_from_diag(diag)",
    "_model_gate_quality_allows_canary": "return self._modules.signal_policy.model_gate_quality_allows_canary()",
    "_model_gate_canary_blocks": "return self._modules.signal_policy.model_gate_canary_blocks(gate_decision, threshold, score)",
    "_feature_values_for_side": "return SignalPolicyModule.feature_values_for_side(vec, side)",
    "_sample_confirmed_candle": "await self._modules.signal_policy.sample_confirmed_candle(symbol, interval, vec)",
    "_bucket_blocked": "return self._modules.signal_policy.bucket_blocked(regime_ctx)",
    "_symbol_side_blocked": "return self._modules.signal_policy.symbol_side_blocked(symbol, side)",
    "_record_shadow_close": "self._modules.signal_policy.record_shadow_close(symbol, reason, pnl_pct)",
    "_shadow_exit_hit": "return SignalPolicyModule.shadow_exit_hit(position, high=high, low=low)",
    "_shadow_pnl_pct": "return self._modules.signal_policy.shadow_pnl_pct(position, exit_price)",
    "_shadow_loss_guard_blocks": "return self._modules.signal_policy.shadow_loss_guard_blocks()",
    "_trend_confirmation_intervals": "return self._modules.signal_policy.trend_confirmation_intervals()",
    "_trend_mtf_confirmed": "return self._modules.signal_policy.trend_mtf_confirmed(symbol, side)",
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
    body = body.replace("self._app._feature_values_for_side(", "self.feature_values_for_side(")
    body = body.replace("self._app._model_gate_threshold(", "self.model_gate_threshold(")
    return body


def rename_def(body: str, old: str, new: str) -> str:
    return re.sub(rf"def {re.escape(old)}\(", f"def {new}(", body, count=1)


def build_module(ranges: dict[str, tuple[int, int]], lines: list[str]) -> str:
    chunks: list[str] = []
    for old, new in SIGNAL_POLICY.items():
        start, end = ranges[old]
        chunk = slice_lines(lines, start, end)
        chunk = transform(chunk)
        if old.startswith("_") and not old.startswith("__"):
            chunk = rename_def(chunk, old, new)
        chunks.append(chunk.rstrip() + "\n")

    header = '''"""Signal policy: ML gates, expectancy filters, shadow helpers, candle sampler."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any

from trader.domain.models import FeatureVector
from trader.modules.base import AppBoundModule
from trader.modules.diagnostics import DiagnosticsModule
from trader.monitoring.logging import get_logger
from trader.runtime.constants import _WS_INTERVAL

log = get_logger(__name__)


class SignalPolicyModule(AppBoundModule):
    name = "signal_policy"

'''
    return header + "\n".join(chunks)


def patch_app(lines: list[str], ranges: dict[str, tuple[int, int]]) -> list[str]:
    # Patch from bottom to top
    ordered = sorted(SIGNAL_POLICY.keys(), key=lambda k: ranges[k][0], reverse=True)
    for old in ordered:
        start, end = ranges[old]
        sig_line = lines[start - 1]
        is_static = "@staticmethod" in lines[start - 2] if start >= 2 else False
        is_async = "async def" in sig_line

        delegate = DELEGATES[old]
        if is_static:
            # handle static separately below
            indent = "    "
            if is_async:
                body = f"{indent}@staticmethod\n{indent}async def {old}(...) -> None:\n{indent}    ...\n"
            else:
                params = sig_line.split("def ")[1].split(")")[0] + ")"
                body_lines = [f"{indent}@staticmethod", f"{indent}def {params}:"]
                body_lines.append(f"{indent}    {delegate}")
                body = "\n".join(body_lines) + "\n"
        else:
            params = sig_line.split("def ")[1].split(")")[0] + ")"
            if is_async:
                body_lines = [f"    async def {params}:"]
            else:
                body_lines = [f"    def {params}:"]
            body_lines.append(f"        {delegate}")
            body = "\n".join(body_lines) + "\n"

        if old == "_feature_values_for_side":
            body = (
                "    @staticmethod\n"
                "    def _feature_values_for_side(vec: FeatureVector, side: str) -> tuple[list[str], list[float]]:\n"
                "        return SignalPolicyModule.feature_values_for_side(vec, side)\n"
            )
        elif old == "_shadow_exit_hit":
            body = (
                "    @staticmethod\n"
                "    def _shadow_exit_hit(position: dict[str, Any], *, high: float, low: float) -> tuple[str, float] | None:\n"
                "        return SignalPolicyModule.shadow_exit_hit(position, high=high, low=low)\n"
            )
        elif old == "_update_model_gate_quality_from_diag":
            body = (
                "    def _update_model_gate_quality_from_diag(self, diag: dict[str, Any]) -> None:\n"
                "        self._modules.signal_policy.update_model_gate_quality_from_diag(diag)\n"
            )
        elif old == "_sample_confirmed_candle":
            body = (
                "    async def _sample_confirmed_candle(self, symbol: str, interval: str, vec: Any) -> None:\n"
                "        await self._modules.signal_policy.sample_confirmed_candle(symbol, interval, vec)\n"
            )
        elif old == "_record_shadow_close":
            body = (
                "    def _record_shadow_close(self, symbol: str, reason: str, pnl_pct: float) -> None:\n"
                "        self._modules.signal_policy.record_shadow_close(symbol, reason, pnl_pct)\n"
            )
        elif old == "_model_gate_canary_blocks":
            body = (
                "    def _model_gate_canary_blocks(self, gate_decision: str, threshold: float, score: float) -> tuple[bool, str]:\n"
                "        return self._modules.signal_policy.model_gate_canary_blocks(gate_decision, threshold, score)\n"
            )
        elif old == "_model_gate_quality_allows_canary":
            body = (
                "    def _model_gate_quality_allows_canary(self) -> tuple[bool, str]:\n"
                "        return self._modules.signal_policy.model_gate_quality_allows_canary()\n"
            )
        elif old == "_model_gate_threshold":
            body = (
                "    def _model_gate_threshold(self, regime_context: Any | None) -> float:\n"
                "        return self._modules.signal_policy.model_gate_threshold(regime_context)\n"
            )
        elif old == "_active_execution_allowed":
            body = (
                "    def _active_execution_allowed(self) -> bool:\n"
                "        return self._modules.signal_policy.active_execution_allowed()\n"
            )
        elif old == "_initial_shadow_mode":
            body = (
                "    def _initial_shadow_mode(self) -> bool:\n"
                "        return self._modules.signal_policy.initial_shadow_mode()\n"
            )
        elif old == "_is_scalp_profile":
            body = (
                "    def _is_scalp_profile(self) -> bool:\n"
                "        return self._modules.signal_policy.is_scalp_profile()\n"
            )
        elif old == "_scalp_strict_shadow":
            body = (
                "    def _scalp_strict_shadow(self) -> bool:\n"
                "        return self._modules.signal_policy.scalp_strict_shadow()\n"
            )
        elif old == "_expectancy_gates_apply":
            body = (
                "    def _expectancy_gates_apply(self) -> bool:\n"
                "        return self._modules.signal_policy.expectancy_gates_apply()\n"
            )
        elif old == "_bucket_blocked":
            body = (
                "    def _bucket_blocked(self, regime_ctx: Any) -> bool:\n"
                "        return self._modules.signal_policy.bucket_blocked(regime_ctx)\n"
            )
        elif old == "_symbol_side_blocked":
            body = (
                "    def _symbol_side_blocked(self, symbol: str, side: str) -> bool:\n"
                "        return self._modules.signal_policy.symbol_side_blocked(symbol, side)\n"
            )
        elif old == "_shadow_pnl_pct":
            body = (
                "    def _shadow_pnl_pct(self, position: dict[str, Any], exit_price: float) -> float:\n"
                "        return self._modules.signal_policy.shadow_pnl_pct(position, exit_price)\n"
            )
        elif old == "_shadow_loss_guard_blocks":
            body = (
                "    def _shadow_loss_guard_blocks(self) -> bool:\n"
                "        return self._modules.signal_policy.shadow_loss_guard_blocks()\n"
            )
        elif old == "_trend_confirmation_intervals":
            body = (
                "    def _trend_confirmation_intervals(self) -> list[str]:\n"
                "        return self._modules.signal_policy.trend_confirmation_intervals()\n"
            )
        elif old == "_trend_mtf_confirmed":
            body = (
                "    def _trend_mtf_confirmed(self, symbol: str, side: str) -> bool:\n"
                "        return self._modules.signal_policy.trend_mtf_confirmed(symbol, side)\n"
            )

        lines[start - 1 : end] = [body]

    return lines


def main() -> None:
    text = APP.read_text()
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    ranges = method_ranges(tree)

    module_path = Path("src/trader/modules/signal_policy.py")
    module_path.write_text(build_module(ranges, lines))

    patched = patch_app(lines, ranges)
    new_app = "".join(patched)

    # Add import for SignalPolicyModule in app if delegates use it
    if "SignalPolicyModule" in new_app and "from trader.modules.signal_policy import SignalPolicyModule" not in new_app:
        new_app = new_app.replace(
            "from trader.modules.registry import ModuleRegistry",
            "from trader.modules.registry import ModuleRegistry\nfrom trader.modules.signal_policy import SignalPolicyModule",
        )

    APP.write_text(new_app)
    print(f"Wrote {module_path} and patched {APP}")


if __name__ == "__main__":
    main()
