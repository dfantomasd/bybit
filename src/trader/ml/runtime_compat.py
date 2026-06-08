"""Temporary compatibility adapter for the legacy runtime scoring call.

``TradingApplication`` currently calls ``ModelRegistry.score()`` once and uses
that prediction for shadow logging. Until the application loop is split into a
separate shadow scorer and Champion-only live gate, keep that legacy call
observational. The directional journal deliberately exposes no gate-quality
statistics, so Canary decisions remain fail-closed during this transition.
"""

from __future__ import annotations

from trader.ml.challenger import ModelPrediction, ModelRegistry


def _score_observational(self: ModelRegistry, features: list[float]) -> ModelPrediction | None:
    """Route the legacy runtime call to Challenger-first shadow scoring."""

    return self.score_shadow(features)


def install_observational_score_alias() -> None:
    """Install the temporary observational alias for the legacy app loop."""

    ModelRegistry.score = _score_observational
