"""Lightweight supervised challenger model.

Uses scikit-learn SGDClassifier (log_loss) for online learning.
No torch, no RL, no LLM — stays within Render Starter memory limits.

The model operates in shadow-scoring mode by default.
MODEL_ALLOW_LIVE_DECISIONS=false (default) → rule-based strategy remains
the authoritative signal source; model only provides model_score metadata.

Champion/Challenger lifecycle:
  SHADOW_CHALLENGER → VALIDATED → CHAMPION
  CHAMPION → ROLLED_BACK (on regression)
  Any → REJECTED (on failure criteria)
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np

log = logging.getLogger(__name__)

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    log.warning("scikit-learn not available; ChallengerModel disabled")


@dataclass
class ModelPrediction:
    score: float
    label: int  # 0 or 1
    confidence: float
    model_version: str
    is_live_decision: bool = False  # True only after promotion


class ModelStatus:
    SHADOW_CHALLENGER = "SHADOW_CHALLENGER"
    VALIDATED = "VALIDATED"
    CHAMPION = "CHAMPION"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass
class ChallengerModel:
    """Online-updateable binary classifier for trade outcome prediction.

    Features: normalized float vector (RSI, EMA diff, volume ratio, etc.)
    Labels: 1 if net_return_bps > threshold after horizon_minutes
    """

    version: str = "v0.0"
    status: str = ModelStatus.SHADOW_CHALLENGER
    feature_names: list[str] = field(default_factory=list)
    training_samples: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    allow_live_decisions: bool = False

    _clf: Any = field(default=None, repr=False)
    _scaler: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if _SKLEARN_AVAILABLE and self._clf is None:
            # class_weight='balanced' is incompatible with partial_fit; use manual weights instead
            self._clf = SGDClassifier(
                loss="log_loss",
                max_iter=1,
                warm_start=True,
                random_state=42,
            )
            self._scaler = StandardScaler()

    @property
    def feature_schema_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.feature_names).encode()).hexdigest()[:16]

    def predict(self, features: list[float]) -> ModelPrediction | None:
        """Score a feature vector. Returns None if model not fitted."""
        if not _SKLEARN_AVAILABLE or self._clf is None:
            return None
        try:
            x = np.array(features, dtype=np.float32).reshape(1, -1)
            if self.training_samples > 0:
                x = self._scaler.transform(x)
            proba = self._clf.predict_proba(x)[0]
            label = int(np.argmax(proba))
            confidence = float(proba[label])
            return ModelPrediction(
                score=float(proba[1]),  # probability of class 1 (positive outcome)
                label=label,
                confidence=confidence,
                model_version=self.version,
                is_live_decision=self.allow_live_decisions and self.status == ModelStatus.CHAMPION,
            )
        except Exception as exc:
            log.debug("challenger.predict_failed", exc_info=exc)
            return None

    def partial_fit(self, features: list[float], label: int) -> None:
        """Online update with a single labelled sample."""
        if not _SKLEARN_AVAILABLE or self._clf is None:
            return
        x = np.array(features, dtype=np.float32).reshape(1, -1)
        y = np.array([label], dtype=np.int32)
        try:
            self._scaler.partial_fit(x)
            x_scaled = self._scaler.transform(x)
            self._clf.partial_fit(x_scaled, y, classes=[0, 1])
            self.training_samples += 1
        except Exception as exc:
            log.debug("challenger.partial_fit_failed", exc_info=exc)

    def to_bytes(self) -> bytes:
        """Serialize model to bytes for PostgreSQL storage."""
        buf = io.BytesIO()
        joblib.dump(
            {
                "clf": self._clf,
                "scaler": self._scaler,
                "meta": {
                    "version": self.version,
                    "feature_names": self.feature_names,
                    "training_samples": self.training_samples,
                },
            },
            buf,
        )
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes, version: str) -> ChallengerModel:
        """Deserialize model from bytes."""
        buf = io.BytesIO(data)
        payload = joblib.load(buf)
        meta = payload.get("meta", {})
        model = cls(
            version=version,
            feature_names=meta.get("feature_names", []),
            training_samples=meta.get("training_samples", 0),
        )
        model._clf = payload.get("clf")
        model._scaler = payload.get("scaler")
        return model

    def can_promote(
        self,
        min_samples: int = 500,
        min_closed_trades: int = 50,
        walk_forward_expectancy: float = 0.0,
    ) -> tuple[bool, str]:
        """Check promotion criteria."""
        if self.training_samples < min_samples:
            return False, f"insufficient_samples: {self.training_samples} < {min_samples}"
        if walk_forward_expectancy <= 0:
            return False, f"negative_walk_forward: {walk_forward_expectancy:.4f}"
        return True, "criteria_met"


class ModelRegistry:
    """Manages champion/challenger lifecycle in memory + PostgreSQL."""

    def __init__(self, trade_journal: Any | None = None) -> None:
        self._journal = trade_journal
        self._champion: ChallengerModel | None = None
        self._challenger: ChallengerModel | None = None

    @property
    def champion(self) -> ChallengerModel | None:
        return self._champion

    @property
    def challenger(self) -> ChallengerModel | None:
        return self._challenger

    def score(self, features: list[float]) -> ModelPrediction | None:
        """Score with champion if available, else challenger."""
        model = self._champion or self._challenger
        if model is None:
            return None
        return model.predict(features)

    def partial_fit_challenger(self, features: list[float], label: int) -> None:
        if self._challenger is not None:
            self._challenger.partial_fit(features, label)

    async def save_checkpoint(self, model: ChallengerModel) -> None:
        """Persist model checkpoint to PostgreSQL."""
        if self._journal is None or not self._journal.is_enabled:
            return
        try:
            artifact = model.to_bytes()
            await self._journal._execute(
                """
                INSERT INTO model_versions (version, status, training_samples, feature_schema_hash, artifact, metrics)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                ON CONFLICT (version) DO UPDATE SET
                    status = EXCLUDED.status,
                    training_samples = EXCLUDED.training_samples,
                    artifact = EXCLUDED.artifact,
                    metrics = EXCLUDED.metrics,
                    training_finished_at = now()
                """,
                model.version,
                model.status,
                model.training_samples,
                model.feature_schema_hash,
                artifact,
                json.dumps({"samples": model.training_samples}),
            )
        except Exception as exc:
            log.debug("model_registry.save_checkpoint_failed", exc_info=exc)

    async def load_champion(self) -> ChallengerModel | None:
        """Load the latest CHAMPION model from PostgreSQL."""
        if self._journal is None or not self._journal.is_enabled:
            return None
        try:
            rows = await self._journal._fetch(
                """
                SELECT version, artifact, training_samples
                FROM model_versions
                WHERE status = 'CHAMPION' AND artifact IS NOT NULL
                ORDER BY training_finished_at DESC NULLS LAST
                LIMIT 1
                """
            )
            if not rows:
                return None
            row = rows[0]
            model = ChallengerModel.from_bytes(bytes(row["artifact"]), version=str(row["version"]))
            model.status = ModelStatus.CHAMPION
            log.info("model_registry.champion_loaded", version=model.version, samples=model.training_samples)
            return model
        except Exception as exc:
            log.debug("model_registry.load_champion_failed", exc_info=exc)
            return None
