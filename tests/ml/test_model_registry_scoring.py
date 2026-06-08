from trader.ml.challenger import ModelPrediction, ModelRegistry


class StubModel:
    def __init__(self, version: str) -> None:
        self.version = version

    def predict(self, features: list[float]) -> ModelPrediction:
        return ModelPrediction(
            score=0.75,
            label=1,
            confidence=0.75,
            model_version=self.version,
            is_live_decision=self.version == "champion",
        )


def test_shadow_scoring_prefers_challenger() -> None:
    registry = ModelRegistry()
    registry._champion = StubModel("champion")  # type: ignore[assignment]
    registry._challenger = StubModel("challenger")  # type: ignore[assignment]

    prediction = registry.score_shadow([1.0])

    assert prediction is not None
    assert prediction.model_version == "challenger"


def test_live_scoring_uses_champion_only() -> None:
    registry = ModelRegistry()
    registry._champion = StubModel("champion")  # type: ignore[assignment]
    registry._challenger = StubModel("challenger")  # type: ignore[assignment]

    prediction = registry.score_live([1.0])

    assert prediction is not None
    assert prediction.model_version == "champion"


def test_legacy_score_is_safe_champion_only_alias() -> None:
    registry = ModelRegistry()
    registry._champion = StubModel("champion")  # type: ignore[assignment]
    registry._challenger = StubModel("challenger")  # type: ignore[assignment]

    prediction = registry.score([1.0])

    assert prediction is not None
    assert prediction.model_version == "champion"


def test_live_scoring_does_not_fall_back_to_unpromoted_challenger() -> None:
    registry = ModelRegistry()
    registry._challenger = StubModel("challenger")  # type: ignore[assignment]

    assert registry.score_live([1.0]) is None
    assert registry.score([1.0]) is None
    assert registry.score_shadow([1.0]) is not None
