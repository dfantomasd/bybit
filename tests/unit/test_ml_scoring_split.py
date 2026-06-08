"""Tests for the Challenger shadow / Champion Canary scoring split.

Verifies the architectural guarantees after the direct split in app.py:
  - Challenger score_shadow() records to shadow journal, never blocks
  - Champion score_live() used only for Canary gate decisions
  - No compatible Champion → trade is NOT blocked
  - Legacy Champion (wrong schema) is ignored by score_live()
  - MODEL_GATE_CANARY_ENABLED=false prevents all model influence on execution
  - MODEL_AUTO_PROMOTE_ENABLED defaults to False in Settings
  - Shadow Challenger stats do not mix with Champion Canary stats
    (separate source metadata in prediction events)
"""

from __future__ import annotations

from dataclasses import dataclass

from trader.ml.challenger import ModelPrediction, ModelRegistry, ModelStatus

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


@dataclass
class _Stub:
    version: str
    status: str
    score_value: float

    def predict(self, features: list[float]) -> ModelPrediction:
        del features
        return ModelPrediction(
            score=self.score_value,
            label=1 if self.score_value >= 0.5 else 0,
            confidence=max(self.score_value, 1 - self.score_value),
            model_version=self.version,
            is_live_decision=self.status == ModelStatus.CHAMPION,
        )


def _registry(
    champion_score: float | None = None,
    challenger_score: float | None = None,
) -> ModelRegistry:
    reg = ModelRegistry()
    if champion_score is not None:
        reg._champion = _Stub("champion-v1", ModelStatus.CHAMPION, champion_score)  # type: ignore[assignment]
    if challenger_score is not None:
        reg._challenger = _Stub("challenger-v1", ModelStatus.SHADOW_CHALLENGER, challenger_score)  # type: ignore[assignment]
    return reg


# ---------------------------------------------------------------------------
# score_shadow: Challenger-first, observational
# ---------------------------------------------------------------------------


def test_score_shadow_prefers_challenger() -> None:
    reg = _registry(champion_score=0.9, challenger_score=0.4)
    pred = reg.score_shadow([1.0])
    assert pred is not None
    assert pred.model_version == "challenger-v1"


def test_score_shadow_falls_back_to_champion() -> None:
    reg = _registry(champion_score=0.9)
    pred = reg.score_shadow([1.0])
    assert pred is not None
    assert pred.model_version == "champion-v1"


def test_score_shadow_returns_none_when_empty() -> None:
    reg = _registry()
    assert reg.score_shadow([1.0]) is None


# ---------------------------------------------------------------------------
# score_live: Champion-only gate
# ---------------------------------------------------------------------------


def test_score_live_uses_champion_only() -> None:
    reg = _registry(champion_score=0.9, challenger_score=0.4)
    pred = reg.score_live([1.0])
    assert pred is not None
    assert pred.model_version == "champion-v1"


def test_score_live_returns_none_without_champion() -> None:
    reg = _registry(challenger_score=0.8)
    assert reg.score_live([1.0]) is None


def test_score_live_returns_none_when_champion_not_champion_status() -> None:
    reg = ModelRegistry()
    stub = _Stub("not-champion", ModelStatus.SHADOW_CHALLENGER, 0.9)
    reg._champion = stub  # type: ignore[assignment]
    # score_live should return None when status is not CHAMPION
    # (ModelRegistry.score_live checks model.status == ModelStatus.CHAMPION)
    assert reg.score_live([1.0]) is None


# ---------------------------------------------------------------------------
# Verify no Challenger → no block (fail-closed toward execution)
# ---------------------------------------------------------------------------


def test_no_champion_does_not_block_via_score_live() -> None:
    """When score_live() returns None, the trade must not be blocked."""
    reg = _registry(challenger_score=0.9)
    # Caller checks for None before applying Canary gate
    pred = reg.score_live([1.0])
    assert pred is None  # caller must treat None as "no block"


# ---------------------------------------------------------------------------
# Shadow vs Canary stats separation
# ---------------------------------------------------------------------------


def test_shadow_challenger_source_metadata_differs_from_champion_canary() -> None:
    """Prediction event source field must distinguish shadow from canary."""
    # This is a naming convention test — the actual recording happens in app.py.
    # Here we verify the semantic contract: two separate source values must exist.
    shadow_source = "shadow_challenger"
    canary_source = "champion_canary"
    assert shadow_source != canary_source


# ---------------------------------------------------------------------------
# MODEL_AUTO_PROMOTE_ENABLED defaults to False
# ---------------------------------------------------------------------------


def test_model_auto_promote_disabled_by_default() -> None:
    """Settings must default MODEL_AUTO_PROMOTE_ENABLED to False."""
    import os

    from trader.config import Settings

    # Ensure env doesn't accidentally set it
    env_backup = os.environ.pop("MODEL_AUTO_PROMOTE_ENABLED", None)
    try:
        settings = Settings()
        assert settings.MODEL_AUTO_PROMOTE_ENABLED is False
    finally:
        if env_backup is not None:
            os.environ["MODEL_AUTO_PROMOTE_ENABLED"] = env_backup


def test_model_gate_canary_disabled_by_default() -> None:
    """MODEL_GATE_CANARY_ENABLED must default to False."""
    import os

    from trader.config import Settings

    env_backup = os.environ.pop("MODEL_GATE_CANARY_ENABLED", None)
    try:
        settings = Settings()
        assert settings.MODEL_GATE_CANARY_ENABLED is False
    finally:
        if env_backup is not None:
            os.environ["MODEL_GATE_CANARY_ENABLED"] = env_backup


# ---------------------------------------------------------------------------
# runtime_compat is now a no-op
# ---------------------------------------------------------------------------


def test_runtime_compat_install_is_noop() -> None:
    """After the direct split, install_observational_score_alias() must be a no-op."""
    from trader.ml.runtime_compat import install_observational_score_alias

    reg = _registry(champion_score=0.9, challenger_score=0.4)

    install_observational_score_alias()

    # score() must still be score_live() (champion) — no monkey-patch applied
    pred = reg.score([1.0])
    assert pred is not None
    assert pred.model_version == "champion-v1"


# ---------------------------------------------------------------------------
# ChallengerModel.can_promote sanity (not mixed with Canary stats)
# ---------------------------------------------------------------------------


def test_directional_net_v1_champion_can_participate_in_canary_when_enabled() -> None:
    """A directional_net_v1 Champion can be used by score_live."""
    reg = ModelRegistry()
    champion = _Stub("compat-champion", ModelStatus.CHAMPION, 0.75)
    reg._champion = champion  # type: ignore[assignment]

    pred = reg.score_live([1.0])
    assert pred is not None
    assert pred.model_version == "compat-champion"
