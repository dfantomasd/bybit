"""Bybit AI Trader — autonomous trading system."""

from __future__ import annotations

import os

# Fail closed during ML recalibration. An operator may still opt in explicitly
# through the environment after the runtime Champion-only gate is reviewed.
os.environ.setdefault("MODEL_AUTO_PROMOTE_ENABLED", "false")

__version__ = "0.1.0"
