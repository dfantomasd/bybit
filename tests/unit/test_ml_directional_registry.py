"""Regression tests for directional ML registry safety guards."""

from __future__ import annotations

from dataclasses import dataclass

from trader.ml.challenger import ChallengerModel, ModelPrediction, ModelRegistry, ModelStatus
from trader.training.labels import LABEL_SCHEMA_VERSION


@dataclass
class _StubModel:
    version: str
    status: str
    score_value: float

    def predict(self, features: list[float]) -> ModelPrediction:
        del features
        return ModelPrediction(
            score=self.score_value,
            label=1 if self.score_value >= 0.5 else 0,
            confidence=max(self.score_value, 1.0 - self.score_value),
            model_version=self.version,
            is_live_decision=self.status == ModelStatus.CHAMPION,
        )


def test_shadow_scoring_prefers_challenger() -> None:
    registry = ModelRegistry()
    registry._champion = _StubModel("champion", ModelStatus.CHAMPION, 0.80)  # type: ignore[assignment]
    registry._challenger = _StubModel("challenger", ModelStatus.SHADOW_CHALLENGER, 0.60)  # type: ignore[assignment]

    prediction = registry.score_shadow([1.0])

    assert prediction is not None
    assert prediction.model_version == "challenger"


def test_live_scoring_uses_champion_only() -> None:
    registry = ModelRegistry()
    registry._champion = _StubModel("champion", ModelStatus.CHAMPION, 0.80)  # type: ignore[assignment]
    registry._challenger = _StubModel("challenger", ModelStatus.SHADOW_CHALLENGER, 0.60)  # type: ignore[assignment]

    prediction = registry.score_live([1.0])

    assert prediction is not None
    assert prediction.model_version == "champion"


def test_live_scoring_fails_closed_without_champion() -> None:
    registry = ModelRegistry()
    registry._challenger = _StubModel("challenger", ModelStatus.SHADOW_CHALLENGER, 0.60)  # type: ignore[assignment]

    assert registry.score_live([1.0]) is None


def test_legacy_score_alias_uses_champion_after_split() -> None:
    """After the direct score_shadow()/score_live() split in app.py, the legacy
    score() alias is champion-only (score_live). app.py now calls score_shadow()
    explicitly for Challenger observation."""

    registry = ModelRegistry()
    registry._champion = _StubModel("champion", ModelStatus.CHAMPION, 0.80)  # type: ignore[assignment]
    registry._challenger = _StubModel("challenger", ModelStatus.SHADOW_CHALLENGER, 0.60)  # type: ignore[assignment]

    prediction = registry.score([1.0])

    # score() is score_live() — must return champion, not challenger
    assert prediction is not None
    assert prediction.model_version == "champion"


def test_promotion_rejects_legacy_schema() -> None:
    model = ChallengerModel(training_samples=1000, label_schema_version="legacy_unknown")

    allowed, reason = model.can_promote(
        min_samples=500,
        min_resolved_observations=50,
        resolved_observations=100,
        walk_forward_expectancy=2.0,
        quality="GOOD",
        required_quality="GOOD",
    )

    assert allowed is False
    assert reason.startswith("incompatible_label_schema")


def test_promotion_requires_resolved_shadow_observations() -> None:
    model = ChallengerModel(training_samples=1000, label_schema_version=LABEL_SCHEMA_VERSION)

    allowed, reason = model.can_promote(
        min_samples=500,
        min_resolved_observations=50,
        resolved_observations=49,
        walk_forward_expectancy=2.0,
        quality="GOOD",
        required_quality="GOOD",
    )

    assert allowed is False
    assert reason == "insufficient_resolved_observations: 49 < 50"


def test_promotion_requires_good_quality_and_positive_expectancy() -> None:
    model = ChallengerModel(training_samples=1000, label_schema_version=LABEL_SCHEMA_VERSION)

    weak_allowed, weak_reason = model.can_promote(
        min_samples=500,
        min_resolved_observations=50,
        resolved_observations=100,
        walk_forward_expectancy=2.0,
        quality="WEAK",
        required_quality="GOOD",
    )
    negative_allowed, negative_reason = model.can_promote(
        min_samples=500,
        min_resolved_observations=50,
        resolved_observations=100,
        walk_forward_expectancy=-0.1,
        quality="GOOD",
        required_quality="GOOD",
    )

    assert weak_allowed is False
    assert weak_reason == "quality_not_good: WEAK"
    assert negative_allowed is False
    assert negative_reason == "negative_walk_forward: -0.1000"


def test_promotion_accepts_compatible_good_model() -> None:
    model = ChallengerModel(training_samples=1000, label_schema_version=LABEL_SCHEMA_VERSION)

    allowed, reason = model.can_promote(
        min_samples=500,
        min_resolved_observations=50,
        resolved_observations=100,
        walk_forward_expectancy=2.0,
        quality="GOOD",
        required_quality="GOOD",
    )

    assert allowed is True
    assert reason == "criteria_met"
