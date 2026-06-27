"""Repository-root import shim for the ``src`` layout.

This keeps commands like ``python -m trader.training.historical_seed`` working
from a source checkout even before the package is installed editable.
"""

from __future__ import annotations

from pathlib import Path

_SRC_TRADER = Path(__file__).resolve().parent.parent / "src" / "trader"
if _SRC_TRADER.is_dir():
    _src_trader_path = str(_SRC_TRADER)
    if _src_trader_path not in __path__:
        __path__.append(_src_trader_path)
