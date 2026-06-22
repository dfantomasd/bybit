"""Shared runtime constants for the trading application."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

# Fallback symbols (used only if screener fails); prefer cheap coins for small balance
SYMBOLS = ["DOGEUSDT", "XRPUSDT", "ADAUSDT", "WLDUSDT", "NEARUSDT"]
WS_INTERVAL = "1"  # 1-minute klines over WS
# Sentinel UUID for WS fill journal rows when proposal/decision IDs are unknown.
JOURNAL_FALLBACK_UUID = UUID(int=0)
MIN_SEED_BARS = 250  # bars to fetch from REST at startup (multi_ewma_signal needs 201)
STRATEGY_LOOP_INTERVAL = 10.0  # seconds between strategy evaluations
FEATURE_INTERVAL = 5.0  # seconds between feature recomputation
TRAINING_HEARTBEAT_SECONDS = 30.0
TRAINING_TIMEOUT_SECONDS = 1800.0
TRADE_JOURNAL_RECONNECT_INTERVAL = 30.0
BALANCE_REFRESH_INTERVAL = 60.0  # seconds between balance refreshes
FALLBACK_BALANCE_USD = Decimal("1000")  # used when API key not configured
SUPERVISOR_CHECK_INTERVAL = 5.0  # seconds between supervisor task health checks
SUPERVISOR_HEARTBEAT_INTERVAL = 60.0  # seconds between heartbeat log lines
DIAG_WINDOW = timedelta(hours=1)  # sliding window for per-hour diagnostics
INTERVAL_MS = {
    "1": 60_000,
    "3": 180_000,
    "5": 300_000,
    "15": 900_000,
    "30": 1_800_000,
    "60": 3_600_000,
}

try:
    from prometheus_client import Counter as _PromCounter

    ML_REPLACEMENT_COUNTER: Any | None
    ML_REPLACEMENT_COUNTER = _PromCounter(
        "trader_ml_replacement_total",
        "Signals where the ML champion replaced the rule-based decision",
    )
except Exception:  # pragma: no cover - prometheus optional at import time
    ML_REPLACEMENT_COUNTER = None

CRITICAL_TASK_NAMES = frozenset(
    {
        "screener",
        "ws-public",
        "ws-consumer",
        "ws-private",
        "ws-private-consumer",
        "feature-pipeline",
        "strategy-loop",
        "risk-monitor",
        "reconciliation",
        "outcome-resolver",
        "load-governor",
    }
)

# Backward-compatible private aliases used by app.py and tests
_SYMBOLS = SYMBOLS
_WS_INTERVAL = WS_INTERVAL
_JOURNAL_FALLBACK_UUID = JOURNAL_FALLBACK_UUID
_MIN_SEED_BARS = MIN_SEED_BARS
_STRATEGY_LOOP_INTERVAL = STRATEGY_LOOP_INTERVAL
_FEATURE_INTERVAL = FEATURE_INTERVAL
_TRAINING_HEARTBEAT_SECONDS = TRAINING_HEARTBEAT_SECONDS
_TRAINING_TIMEOUT_SECONDS = TRAINING_TIMEOUT_SECONDS
_TRADE_JOURNAL_RECONNECT_INTERVAL = TRADE_JOURNAL_RECONNECT_INTERVAL
_BALANCE_REFRESH_INTERVAL = BALANCE_REFRESH_INTERVAL
_FALLBACK_BALANCE_USD = FALLBACK_BALANCE_USD
_SUPERVISOR_CHECK_INTERVAL = SUPERVISOR_CHECK_INTERVAL
_SUPERVISOR_HEARTBEAT_INTERVAL = SUPERVISOR_HEARTBEAT_INTERVAL
_DIAG_WINDOW = DIAG_WINDOW
_INTERVAL_MS = INTERVAL_MS
_ML_REPLACEMENT_COUNTER = ML_REPLACEMENT_COUNTER
_CRITICAL_TASK_NAMES = CRITICAL_TASK_NAMES
