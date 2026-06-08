"""Temporary safety adapters for the legacy application ML loop.

``TradingApplication`` currently performs one model-scoring call and reuses its
prediction for shadow logging and optional Canary filtering. During the
transition to explicit Challenger shadow scoring and Champion-only live gating,
keep the legacy scoring call observational and reject every Canary activation.
"""

from __future__ import annotations

import sys
from typing import Any

from trader.ml.challenger import ModelPrediction, ModelRegistry

_TRANSITION_REASON = "directional_runtime_split_pending"


def _score_observational(self: ModelRegistry, features: list[float]) -> ModelPrediction | None:
    """Route the legacy application call to Challenger-first shadow scoring."""

    return self.score_shadow(features)


def _canary_disabled(self: Any) -> tuple[bool, str]:
    """Reject Canary activation until the application loop is explicitly split."""

    del self
    return False, _TRANSITION_REASON


def install_observational_score_alias() -> None:
    """Install temporary observational scoring and a fail-closed Canary guard."""

    ModelRegistry.score = _score_observational

    app_module = sys.modules.get("trader.app")
    app_cls = getattr(app_module, "TradingApplication", None) if app_module is not None else None
    if app_cls is not None:
        app_cls._model_gate_quality_allows_canary = _canary_disabled
