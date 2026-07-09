"""Unified ML Controller - управляет всеми 5 ML моделями.

Координирует обучение, предсказание и сохранение всех моделей.
Является единой точкой входа для торговой системы.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MLPredictions:
    """Все предсказания от всех 5 моделей."""

    # 1. Kelly Predictor
    kelly_fraction: Decimal
    fractional_kelly: Decimal
    kelly_confidence: float
    kelly_reasoning: str

    # 2. Regime Predictor
    current_regime: str
    regime_confidence: float
    next_5m_regime: str
    next_15m_regime: str
    trend_phase: str

    # 3. Signal Fusion
    fused_signal: float  # -1 to +1
    signal_confidence: float
    signal_recommendation: str

    # 4. Spread Predictor
    predicted_spread_bps: float
    spread_risk: float
    spread_recommendation: str

    # 5. StopLoss Optimizer
    optimal_stop_pct: float
    emergency_stop_pct: float
    sl_recommendation: str

    # 6. Entry/Exit Optimizer (bonus)
    entry_price: Decimal
    take_profit_price: Decimal
    stop_loss_price: Decimal
    entry_confidence: float

    # Meta
    timestamp: datetime
    all_models_trained: bool


class UnifiedMLController:
    """Управляет всеми 5 ML моделями как единой системой."""

    def __init__(
        self,
        kelly_predictor: Any,
        regime_predictor: Any,
        signal_fusion: Any,
        spread_predictor: Any,
        stoploss_optimizer: Any,
        entry_exit_optimizer: Any = None,
        model_dir: str = "data/ml_unified_models",
        auto_save: bool = True,
    ) -> None:
        """Инициализация контроллера.

        Args:
            kelly_predictor: MLKellyPredictor instance
            regime_predictor: RegimePredictor instance
            signal_fusion: SignalFusion instance
            spread_predictor: SpreadPredictor instance
            stoploss_optimizer: StopLossOptimizer instance
            entry_exit_optimizer: EntryExitOptimizer instance (optional)
            model_dir: Директория для сохранения моделей
            auto_save: Автоматически сохранять модели после обучения
        """
        self.kelly = kelly_predictor
        self.regime = regime_predictor
        self.signals = signal_fusion
        self.spread = spread_predictor
        self.stoploss = stoploss_optimizer
        self.entry_exit = entry_exit_optimizer

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save

        # История для обучения
        self.kelly_training_data: list[dict] = []
        self.regime_training_data: list[dict] = []
        self.signal_training_data: list[dict] = []
        self.spread_training_data: list[dict] = []
        self.stoploss_training_data: list[dict] = []
        self.entry_exit_training_data: list[dict] = []

        # Статистика
        self.training_count = 0
        self.last_training_time: datetime | None = None
        self.prediction_count = 0
        self.accuracy_stats = {
            "kelly": [],
            "regime": [],
            "signals": [],
            "spread": [],
            "stoploss": [],
        }

        logger.info("🤖 UnifiedMLController initialized")

    async def predict_all(
        self,
        kelly_features: Any,
        regime_features: Any,
        signal_context: Any,
        spread_features: Any,
        stoploss_context: Any,
        candle_context: Any = None,
        current_price: Decimal = Decimal("1"),
    ) -> MLPredictions:
        """Получить предсказания от всех моделей.

        Вызывает все 5 моделей параллельно (async).
        """
        try:
            # Helper async functions to handle None cases
            async def call_kelly() -> Any:
                if kelly_features is not None:
                    return await self.kelly.predict(kelly_features)
                return None

            async def call_regime() -> Any:
                if regime_features is not None:
                    return await self.regime.predict(regime_features)
                return None

            async def call_signals() -> Any:
                if signal_context is not None:
                    return await self.signals.fuse_signals(signal_context)
                return None

            async def call_spread() -> Any:
                if spread_features is not None:
                    return await self.spread.predict(spread_features)
                return None

            async def call_stoploss() -> Any:
                if stoploss_context is not None:
                    return await self.stoploss.calculate_optimal_stop(stoploss_context)
                return None

            async def call_entry_exit() -> Any:
                if self.entry_exit and candle_context:
                    return await self.entry_exit.get_optimization(candle_context)
                return None

            # Запустить все модели параллельно
            results = await asyncio.gather(
                call_kelly(),
                call_regime(),
                call_signals(),
                call_spread(),
                call_stoploss(),
                call_entry_exit(),
                return_exceptions=True,
            )

            kelly_result = results[0] if not isinstance(results[0], Exception) else None
            regime_result = results[1] if not isinstance(results[1], Exception) else None
            signal_result = results[2] if not isinstance(results[2], Exception) else None
            spread_result = results[3] if not isinstance(results[3], Exception) else None
            stoploss_result = results[4] if not isinstance(results[4], Exception) else None
            entry_exit_result = results[5] if not isinstance(results[5], Exception) else None

            # Обработать ошибки
            if isinstance(results[0], Exception):
                logger.error(f"Kelly prediction failed: {results[0]}")
            if isinstance(results[1], Exception):
                logger.error(f"Regime prediction failed: {results[1]}")
            if isinstance(results[2], Exception):
                logger.error(f"Signal fusion failed: {results[2]}")
            if isinstance(results[3], Exception):
                logger.error(f"Spread prediction failed: {results[3]}")
            if isinstance(results[4], Exception):
                logger.error(f"Stoploss optimization failed: {results[4]}")

            # Построить результат
            predictions = MLPredictions(
                # Kelly
                kelly_fraction=kelly_result.kelly_fraction if kelly_result else Decimal("0.10"),
                fractional_kelly=kelly_result.fractional_kelly if kelly_result else Decimal("0.25"),
                kelly_confidence=kelly_result.model_confidence if kelly_result else 0.3,
                kelly_reasoning=kelly_result.reasoning if kelly_result else "Model not trained",
                # Regime
                current_regime=regime_result.current_regime if regime_result else "SIDEWAYS",
                regime_confidence=regime_result.confidence_current if regime_result else 0.3,
                next_5m_regime=regime_result.next_5m_regime if regime_result else "SIDEWAYS",
                next_15m_regime=regime_result.next_15m_regime if regime_result else "SIDEWAYS",
                trend_phase=regime_result.trend_phase if regime_result else "CHAOTIC",
                # Signals — SignalFusionEnhanced.fuse_signals() returns a dict
                # {"final_signal", "confidence", "recommendation", ...}; the
                # legacy SignalFusion.fuse_signals() returns a 3-tuple. Handle
                # both so predict_all() doesn't silently fall back on every call.
                fused_signal=(
                    (signal_result.get("final_signal", 0.0) if isinstance(signal_result, dict) else signal_result[0])
                    if signal_result
                    else 0.0
                ),
                signal_confidence=(
                    (signal_result.get("confidence", 0.3) if isinstance(signal_result, dict) else signal_result[1])
                    if signal_result
                    else 0.3
                ),
                signal_recommendation=(
                    (
                        signal_result.get("recommendation", "NEUTRAL")
                        if isinstance(signal_result, dict)
                        else signal_result[2]
                    )
                    if signal_result
                    else "NEUTRAL"
                ),
                # Spread
                predicted_spread_bps=spread_result.get("predicted_spread_bps", 25.0) if spread_result else 25.0,
                spread_risk=spread_result.get("widening_risk", 0.5) if spread_result else 0.5,
                spread_recommendation=spread_result.get("spread_recommendation", "OK") if spread_result else "OK",
                # StopLoss
                optimal_stop_pct=stoploss_result.get("stop_distance_pct", 2.0) if stoploss_result else 2.0,
                emergency_stop_pct=stoploss_result.get("emergency_stop_pct", 3.0) if stoploss_result else 3.0,
                sl_recommendation=stoploss_result.get("recommendation", "OK") if stoploss_result else "OK",
                # Entry/Exit
                entry_price=entry_exit_result.entry_price if entry_exit_result else current_price,
                take_profit_price=entry_exit_result.take_profit_price if entry_exit_result else current_price,
                stop_loss_price=entry_exit_result.stop_loss_price if entry_exit_result else current_price,
                entry_confidence=entry_exit_result.probability_of_success if entry_exit_result else 0.5,
                # Meta
                timestamp=datetime.now(UTC),
                all_models_trained=(
                    self.kelly.kelly_model is not None
                    and self.regime.regime_model is not None
                    and self.signals.outcome_model is not None
                    and self.spread.spread_model is not None
                    and self.stoploss.model is not None
                ),
            )

            self.prediction_count += 1
            logger.debug(f"✅ All predictions ready (count: {self.prediction_count})")
            return predictions

        except Exception as e:
            logger.error(f"predict_all failed: {e}")
            # Вернуть дефолтные значения
            return self._get_fallback_predictions(current_price)

    async def add_training_sample(
        self,
        trade_outcome: dict[str, Any],
        kelly_features: Any = None,
        regime_features: Any = None,
        signal_context: Any = None,
        spread_features: Any = None,
        stoploss_context: Any = None,
        entry_exit_context: Any = None,
    ) -> None:
        """Добавить сэмпл в историю обучения.

        Вызывается после каждой закрытой сделки.
        """
        try:
            # Добавить в истории обучения
            if kelly_features:
                self.kelly_training_data.append(
                    {
                        "features": kelly_features,
                        "kelly_actual": trade_outcome.get("kelly_used", 0.10),
                        "fractional_actual": trade_outcome.get("fractional_used", 0.25),
                        "was_profitable": trade_outcome.get("pnl_usd", 0) > 0,
                    }
                )

            if regime_features:
                self.regime_training_data.append(
                    {
                        "features": regime_features,
                        "current_regime_class": trade_outcome.get("regime_class", 2),
                        "next_5m_regime_class": trade_outcome.get("next_regime_5m", 2),
                    }
                )

            if signal_context:
                self.signal_training_data.append(
                    {
                        "context": signal_context,
                        "was_profitable": trade_outcome.get("pnl_usd", 0) > 0,
                        "expected_confidence": trade_outcome.get("signal_strength", 0.5),
                    }
                )

            if spread_features:
                self.spread_training_data.append(
                    {
                        "features": spread_features,
                        "actual_spread_bps": trade_outcome.get("actual_spread_bps", 20.0),
                        "spread_widened": trade_outcome.get("spread_widened", False),
                    }
                )

            if stoploss_context:
                self.stoploss_training_data.append(
                    {
                        "context": stoploss_context,
                        "optimal_stop_pct": trade_outcome.get("optimal_stop_pct", 2.0),
                        "optimal_cvar_pct": trade_outcome.get("optimal_cvar_pct", 3.0),
                    }
                )

            if entry_exit_context:
                self.entry_exit_training_data.append(
                    {
                        "context": entry_exit_context,
                        "optimal_entry_offset_pct": trade_outcome.get("entry_offset_pct", 0.0),
                        "optimal_tp_distance_pct": trade_outcome.get("tp_distance_pct", 1.0),
                        "optimal_sl_distance_pct": trade_outcome.get("sl_distance_pct", 0.7),
                    }
                )

            logger.debug("✅ Training sample added")

        except Exception as e:
            logger.error(f"add_training_sample failed: {e}")

    async def retrain_models(self, force: bool = False) -> dict[str, bool]:
        """Переобучить модели на накопленных данных.

        Args:
            force: Принудительно переобучить, даже если мало данных

        Returns:
            Dict[model_name, success]
        """
        results = {}

        try:
            # Проверить количество данных
            min_samples_kelly = 100
            if len(self.kelly_training_data) >= min_samples_kelly or force:
                logger.info(f"🔄 Retraining Kelly ({len(self.kelly_training_data)} samples)...")
                await self.kelly.train(self.kelly_training_data)
                results["kelly"] = True
                self.kelly_training_data = []  # Очистить после обучения
            else:
                results["kelly"] = False

            if len(self.regime_training_data) >= min_samples_kelly or force:
                logger.info(f"🔄 Retraining Regime ({len(self.regime_training_data)} samples)...")
                await self.regime.train(self.regime_training_data)
                results["regime"] = True
                self.regime_training_data = []
            else:
                results["regime"] = False

            if len(self.signal_training_data) >= 100 or force:
                logger.info(f"🔄 Retraining Signals ({len(self.signal_training_data)} samples)...")
                await self.signals.train(self.signal_training_data)
                results["signals"] = True
                self.signal_training_data = []
            else:
                results["signals"] = False

            if len(self.spread_training_data) >= 100 or force:
                logger.info(f"🔄 Retraining Spread ({len(self.spread_training_data)} samples)...")
                await self.spread.train(self.spread_training_data)
                results["spread"] = True
                self.spread_training_data = []
            else:
                results["spread"] = False

            if len(self.stoploss_training_data) >= 100 or force:
                logger.info(f"🔄 Retraining StopLoss ({len(self.stoploss_training_data)} samples)...")
                await self.stoploss.train(self.stoploss_training_data)
                results["stoploss"] = True
                self.stoploss_training_data = []
            else:
                results["stoploss"] = False

            # Сохранить модели
            if any(results.values()):
                if self.auto_save:
                    await self.save_models()
                self.last_training_time = datetime.now(UTC)
                self.training_count += 1
                logger.info(f"✅ Retraining complete: {results}")

            return results

        except Exception as e:
            logger.error(f"retrain_models failed: {e}")
            return dict.fromkeys(["kelly", "regime", "signals", "spread", "stoploss"], False)

    @staticmethod
    def _save_pkl(path: Path, obj: object) -> None:
        """Pickle obj to path and write a SHA-256 checksum sidecar."""
        data = pickle.dumps(obj, protocol=5)
        digest = hashlib.sha256(data).hexdigest()
        path.write_bytes(data)
        path.with_suffix(".sha256").write_text(digest + "\n", encoding="ascii")

    @staticmethod
    def _load_pkl(path: Path) -> object:
        """Load and verify a pickle file written by _save_pkl."""
        data = path.read_bytes()
        sha_path = path.with_suffix(".sha256")
        if sha_path.exists():
            expected = sha_path.read_text(encoding="ascii").strip()
            actual = hashlib.sha256(data).hexdigest()
            if not hmac.compare_digest(actual, expected):
                raise ValueError(f"checksum mismatch for {path}: file may have been tampered with")
        else:
            logger.warning("pkl_no_checksum: %s", path)
        return pickle.loads(data)  # noqa: S301

    async def save_models(self) -> None:
        """Сохранить все модели на диск."""
        try:
            timestamp = datetime.now(UTC).isoformat()

            # Сохранить каждую модель
            if self.kelly.kelly_model:
                path = self.model_dir / "kelly_model.pkl"
                self._save_pkl(path, self.kelly.kelly_model)
                logger.debug(f"💾 Saved kelly_model to {path}")

            if self.regime.regime_model:
                path = self.model_dir / "regime_model.pkl"
                self._save_pkl(path, self.regime.regime_model)
                logger.debug(f"💾 Saved regime_model to {path}")

            if self.signals.outcome_model:
                path = self.model_dir / "signals_model.pkl"
                self._save_pkl(path, self.signals.outcome_model)
                logger.debug(f"💾 Saved signals_model to {path}")

            if self.spread.spread_model:
                path = self.model_dir / "spread_model.pkl"
                self._save_pkl(path, self.spread.spread_model)
                logger.debug(f"💾 Saved spread_model to {path}")

            if self.stoploss.model:
                path = self.model_dir / "stoploss_model.pkl"
                self._save_pkl(path, self.stoploss.model)
                logger.debug(f"💾 Saved stoploss_model to {path}")

            # Сохранить метаданные
            metadata = {
                "timestamp": timestamp,
                "training_count": self.training_count,
                "prediction_count": self.prediction_count,
                "models_trained": {
                    "kelly": self.kelly.kelly_model is not None,
                    "regime": self.regime.regime_model is not None,
                    "signals": self.signals.outcome_model is not None,
                    "spread": self.spread.spread_model is not None,
                    "stoploss": self.stoploss.model is not None,
                },
            }
            with open(self.model_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"✅ All models saved to {self.model_dir}")

        except Exception as e:
            logger.error(f"save_models failed: {e}")

    async def load_models(self) -> None:
        """Загрузить модели с диска."""
        _models: list[tuple[str, Any]] = [
            ("kelly_model", lambda p: setattr(self.kelly, "kelly_model", self._load_pkl(p))),
            ("regime_model", lambda p: setattr(self.regime, "regime_model", self._load_pkl(p))),
            ("signals_model", lambda p: setattr(self.signals, "outcome_model", self._load_pkl(p))),
            ("spread_model", lambda p: setattr(self.spread, "spread_model", self._load_pkl(p))),
            ("stoploss_model", lambda p: setattr(self.stoploss, "model", self._load_pkl(p))),
        ]
        loaded = 0
        for name, loader in _models:
            path = self.model_dir / f"{name}.pkl"
            if not path.exists():
                continue
            try:
                loader(path)
                logger.info(f"Loaded {name}")
                loaded += 1
            except Exception as e:
                # Log per-model failure without aborting the remaining loads.
                logger.error(f"load_models.{name}_failed: {e}")
        logger.info(f"load_models completed: {loaded}/{len(_models)} models loaded")

    @staticmethod
    def _get_fallback_predictions(current_price: Decimal) -> MLPredictions:
        """Дефолтные предсказания когда всё падает."""
        return MLPredictions(
            kelly_fraction=Decimal("0.10"),
            fractional_kelly=Decimal("0.25"),
            kelly_confidence=0.3,
            kelly_reasoning="Fallback - no trained models",
            current_regime="SIDEWAYS",
            regime_confidence=0.3,
            next_5m_regime="SIDEWAYS",
            next_15m_regime="SIDEWAYS",
            trend_phase="CHAOTIC",
            fused_signal=0.0,
            signal_confidence=0.3,
            signal_recommendation="NEUTRAL",
            predicted_spread_bps=25.0,
            spread_risk=0.5,
            spread_recommendation="OK",
            optimal_stop_pct=2.0,
            emergency_stop_pct=3.0,
            sl_recommendation="OK",
            entry_price=current_price,
            take_profit_price=current_price,
            stop_loss_price=current_price,
            entry_confidence=0.5,
            timestamp=datetime.now(UTC),
            all_models_trained=False,
        )

    def get_status(self) -> dict[str, Any]:
        """Получить статус всех моделей."""
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "training_count": self.training_count,
            "prediction_count": self.prediction_count,
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "models_trained": {
                "kelly": self.kelly.kelly_model is not None,
                "regime": self.regime.regime_model is not None,
                "signals": self.signals.outcome_model is not None,
                "spread": self.spread.spread_model is not None,
                "stoploss": self.stoploss.model is not None,
            },
            "training_data_queued": {
                "kelly": len(self.kelly_training_data),
                "regime": len(self.regime_training_data),
                "signals": len(self.signal_training_data),
                "spread": len(self.spread_training_data),
                "stoploss": len(self.stoploss_training_data),
            },
        }
