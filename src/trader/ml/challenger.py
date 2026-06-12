"""Lightweight supervised challenger model.

Uses scikit-learn SGDClassifier (log_loss) for online learning.
No torch, no RL, no LLM — stays within Render Starter memory limits.

The model operates in shadow-scoring mode by default. A challenger may be used
for observational scoring only. Live gate decisions are sourced exclusively
from a compatible CHAMPION model trained with the current directional label
schema.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import structlog

from trader.training.labels import LABEL_SCHEMA_VERSION

log = structlog.get_logger(__name__)

LEGACY_LABEL_SCHEMA_VERSION = "legacy_unknown"

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    log.warning("scikit-learn not available; ChallengerModel disabled")


def _parse_metrics(raw: Any) -> dict[str, Any]:
    """Return JSON metrics as a plain dictionary."""

    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    """Read a mapping-like database row while tolerating legacy test fixtures."""

    getter = getattr(row, "get", None)
    if callable(getter):
        return getter(key, default)
    try:
        return row[key]
    except (KeyError, TypeError):
        return default


@dataclass
class ModelPrediction:
    score: float
    label: int  # 0 or 1
    confidence: float
    model_version: str
    is_live_decision: bool = False


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
    Labels: 1 if directional net_return_bps > threshold after horizon_minutes
    """

    version: str = "v0.0"
    status: str = ModelStatus.SHADOW_CHALLENGER
    feature_names: list[str] = field(default_factory=list)
    training_samples: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    allow_live_decisions: bool = False
    label_schema_version: str = LABEL_SCHEMA_VERSION
    model_type: str = "SGD"
    """"SGD" (linear, supports online partial_fit) or "GBDT" (gradient-boosted
    trees via HistGradientBoostingClassifier — stronger on non-linear feature
    interactions, batch-only)."""

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
                score=float(proba[1]),
                label=label,
                confidence=confidence,
                model_version=self.version,
                is_live_decision=self.allow_live_decisions and self.status == ModelStatus.CHAMPION,
            )
        except Exception as exc:
            log.debug("challenger.predict_failed", exc_info=exc)
            return None

    def fit_batch(self, features: Any, labels: Any, *, epochs: int = 5) -> None:
        """Train from scratch on a full labelled batch.

        Unlike per-sample ``partial_fit``, the scaler is fitted on the whole
        batch BEFORE any gradient step (an online scaler feeds the first
        hundreds of samples through near-random scaling, which a single-pass
        SGD never recovers from), several shuffled epochs are run, and class
        imbalance is countered with balanced sample weights.
        """

        if not _SKLEARN_AVAILABLE:
            return
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int32)
        if x.ndim != 2 or len(x) == 0 or len(x) != len(y):
            return
        # Batch training replaces any previous estimator state.
        self._scaler = StandardScaler()
        x_scaled = self._scaler.fit_transform(x)

        if self.model_type.upper() == "GBDT":
            from sklearn.ensemble import HistGradientBoostingClassifier

            self._clf = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                early_stopping=True,
                validation_fraction=0.15,
                class_weight="balanced",
                random_state=42,
            )
            self._clf.fit(x_scaled, y)
            self.training_samples = int(len(x_scaled))
            return

        self._clf = SGDClassifier(
            loss="log_loss",
            max_iter=1,
            warm_start=True,
            random_state=42,
        )
        classes = np.array([0, 1], dtype=np.int32)
        counts = np.bincount(y, minlength=2).astype(np.float64)
        weight_by_class = np.where(counts > 0, counts.sum() / (2.0 * np.maximum(counts, 1.0)), 1.0)
        sample_weight = weight_by_class[y]
        rng = np.random.default_rng(42)
        for _ in range(max(1, int(epochs))):
            order = rng.permutation(len(x_scaled))
            self._clf.partial_fit(
                x_scaled[order],
                y[order],
                classes=classes,
                sample_weight=sample_weight[order],
            )
        self.training_samples = int(len(x_scaled))

    def partial_fit(self, features: list[float], label: int) -> None:
        """Online update with a single labelled sample."""

        if not _SKLEARN_AVAILABLE or self._clf is None:
            return
        if self.model_type.upper() == "GBDT":
            # Gradient-boosted trees cannot be updated online; the periodic
            # batch retrain covers new data instead.
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
                    "label_schema_version": self.label_schema_version,
                    "model_type": self.model_type,
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
            label_schema_version=meta.get("label_schema_version", LEGACY_LABEL_SCHEMA_VERSION),
            model_type=str(meta.get("model_type", "SGD")),
        )
        model._clf = payload.get("clf")
        model._scaler = payload.get("scaler")
        return model

    def can_promote(
        self,
        *,
        min_samples: int = 500,
        min_resolved_observations: int = 0,
        resolved_observations: int = 0,
        walk_forward_expectancy: float = 0.0,
        quality: str = "",
        required_quality: str = "",
    ) -> tuple[bool, str]:
        """Check conservative offline and shadow-observation promotion criteria."""

        if self.label_schema_version != LABEL_SCHEMA_VERSION:
            return (
                False,
                f"incompatible_label_schema: {self.label_schema_version!r} != {LABEL_SCHEMA_VERSION!r}",
            )
        if self.training_samples < min_samples:
            return (
                False,
                f"insufficient_samples: {self.training_samples} < {min_samples}",
            )
        if resolved_observations < min_resolved_observations:
            return (
                False,
                f"insufficient_resolved_observations: {resolved_observations} < {min_resolved_observations}",
            )
        if required_quality and quality.upper() != required_quality.upper():
            return False, f"quality_not_{required_quality.lower()}: {quality or 'none'}"
        if walk_forward_expectancy <= 0:
            return False, f"negative_walk_forward: {walk_forward_expectancy:.4f}"
        return True, "criteria_met"


class ModelRegistry:
    """Manage compatible champion/challenger lifecycle in memory + PostgreSQL."""

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

    def score_shadow(self, features: list[float]) -> ModelPrediction | None:
        """Score observationally with challenger first, then champion fallback."""

        model = self._challenger or self._champion
        return model.predict(features) if model is not None else None

    def score_live(self, features: list[float]) -> ModelPrediction | None:
        """Score authoritatively with the compatible champion only."""

        model = self._champion
        if model is None or model.status != ModelStatus.CHAMPION:
            return None
        return model.predict(features)

    def score(self, features: list[float]) -> ModelPrediction | None:
        """Backward-compatible runtime alias: champion-only, fail-closed."""

        return self.score_live(features)

    def partial_fit_challenger(self, features: list[float], label: int) -> None:
        if self._challenger is not None:
            self._challenger.partial_fit(features, label)

    async def load_active_model(self) -> ChallengerModel | None:
        """Load compatible champion and challenger; return champion when present."""

        champion = await self.load_champion()
        await self.load_latest_challenger()
        return champion or self._challenger

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
                json.dumps(
                    {
                        "samples": model.training_samples,
                        "label_schema_version": model.label_schema_version,
                    }
                ),
            )
        except Exception as exc:
            log.debug("model_registry.save_checkpoint_failed", exc_info=exc)

    async def load_champion(self) -> ChallengerModel | None:
        """Load the latest compatible CHAMPION model."""

        if self._journal is None or not self._journal.is_enabled:
            return None
        try:
            rows = await self._journal._fetch(
                """
                SELECT version, artifact, training_samples, metrics
                FROM model_versions
                WHERE status = 'CHAMPION'
                  AND artifact IS NOT NULL
                  AND COALESCE(metrics->>'label_schema_version', '') = $1
                ORDER BY training_finished_at DESC NULLS LAST
                LIMIT 1
                """,
                LABEL_SCHEMA_VERSION,
            )
            if not rows:
                self._champion = None
                log.warning(
                    "model_registry.no_compatible_champion required_schema=%s",
                    LABEL_SCHEMA_VERSION,
                )
                return None
            row = rows[0]
            metrics = _parse_metrics(_row_get(row, "metrics", {}))
            model = ChallengerModel.from_bytes(bytes(row["artifact"]), version=str(row["version"]))
            model.status = ModelStatus.CHAMPION
            model.training_samples = int(
                _row_get(row, "training_samples", model.training_samples) or model.training_samples
            )
            model.allow_live_decisions = True
            model.label_schema_version = str(metrics.get("label_schema_version") or model.label_schema_version)
            self._champion = model
            log.info(
                "model_registry.champion_loaded version=%s samples=%s",
                model.version,
                model.training_samples,
            )
            return model
        except Exception as exc:
            log.debug("model_registry.load_champion_failed", exc_info=exc)
            return None

    async def load_latest_challenger(self) -> ChallengerModel | None:
        """Load the latest compatible challenger for non-authoritative scoring."""

        if self._journal is None or not self._journal.is_enabled:
            return None
        try:
            rows = await self._journal._fetch(
                """
                SELECT version, status, artifact, training_samples, metrics
                FROM model_versions
                WHERE status IN ('VALIDATED', 'SHADOW_CHALLENGER')
                  AND artifact IS NOT NULL
                  AND COALESCE(metrics->>'label_schema_version', '') = $1
                ORDER BY training_finished_at DESC NULLS LAST, created_at DESC
                LIMIT 1
                """,
                LABEL_SCHEMA_VERSION,
            )
            if not rows:
                self._challenger = None
                return None
            row = rows[0]
            metrics = _parse_metrics(_row_get(row, "metrics", {}))
            model = ChallengerModel.from_bytes(bytes(row["artifact"]), version=str(row["version"]))
            model.status = str(_row_get(row, "status", ModelStatus.SHADOW_CHALLENGER))
            model.training_samples = int(
                _row_get(row, "training_samples", model.training_samples) or model.training_samples
            )
            model.allow_live_decisions = False
            model.label_schema_version = str(metrics.get("label_schema_version") or model.label_schema_version)
            self._challenger = model
            log.info(
                "model_registry.challenger_loaded version=%s samples=%s",
                model.version,
                model.training_samples,
            )
            return model
        except Exception as exc:
            log.debug("model_registry.load_challenger_failed", exc_info=exc)
            return None
