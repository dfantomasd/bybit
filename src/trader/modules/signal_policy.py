"""Signal policy: ML gates, expectancy filters, shadow helpers, candle sampler."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from trader.domain.enums import TradingMode
from trader.domain.models import FeatureVector
from trader.modules.base import AppBoundModule
from trader.modules.diagnostics import DiagnosticsModule
from trader.monitoring.logging import get_logger
from trader.runtime.constants import _WS_INTERVAL

log = get_logger(__name__)


class SignalPolicyModule(AppBoundModule):
    name = "signal_policy"

    def active_execution_allowed(self) -> bool:
        """Return True when orders may be submitted to the configured endpoint."""
        assert self._app._settings is not None
        if self._app._settings.TRADING_MODE == TradingMode.SHADOW:
            return False
        if self._app._settings.BYBIT_USE_TESTNET:
            return True
        return self._app._settings.LIVE_MODE and self._app._settings.TRADING_MODE in (
            TradingMode.LIVE,
            TradingMode.CANARY_LIVE,
        )

    def initial_shadow_mode(self) -> bool:
        """Return current shadow/paper execution mode (runtime-aware)."""
        if self._app._execution_engine is not None:
            return bool(self._app._execution_engine._shadow_mode)
        assert self._app._settings is not None
        if self._app._settings.SHADOW_MODE:
            return True
        return not self.active_execution_allowed()

    def is_scalp_profile(self) -> bool:
        from trader.domain.enums import RiskProfile

        assert self._app._settings is not None
        return self._app._settings.RISK_PROFILE == RiskProfile.SCALP

    def scalp_strict_shadow(self) -> bool:
        """SCALP paper-trading should mirror LIVE quality gates."""
        if self._app._settings is None:
            return False
        return self.is_scalp_profile() and self._app._settings.SCALP_STRICT_SHADOW and self.initial_shadow_mode()

    def expectancy_gates_apply(self) -> bool:
        """True when bucket/symbol-side gates should block entries."""
        if self.scalp_strict_shadow():
            return True
        return not self.initial_shadow_mode()

    def model_gate_threshold(self, regime_context: Any | None) -> float:
        """Return a conservative threshold adjusted by market regime."""
        assert self._app._settings is not None
        best_threshold = self._app._model_gate_quality.get("best_threshold")
        threshold = (
            float(best_threshold)
            if best_threshold is not None
            else float(self._app._settings.MODEL_SHADOW_GATE_THRESHOLD)
        )
        if regime_context is None:
            return threshold + 0.02

        regime = getattr(
            getattr(regime_context, "regime", None),
            "value",
            str(getattr(regime_context, "regime", "")),
        )
        volatility = getattr(
            getattr(regime_context, "volatility_level", None),
            "value",
            str(getattr(regime_context, "volatility_level", "")),
        )
        if regime in {"BULL_TREND", "BEAR_TREND"}:
            threshold -= 0.02
        elif regime in {"SIDEWAYS", "UNCERTAIN"}:
            threshold += 0.03
        elif regime in {"HIGH_VOLATILITY", "LOW_LIQUIDITY"}:
            threshold += 0.05
        if volatility in {"HIGH", "EXTREME"}:
            threshold += 0.03
        elif volatility == "LOW":
            threshold += 0.01
        return min(0.80, max(0.50, threshold))

    def update_model_gate_quality_from_diag(self, diag: dict[str, Any]) -> None:
        latest_model = DiagnosticsModule.dict_or_empty(diag.get("latest_model_version"))
        metrics = DiagnosticsModule.dict_or_empty(latest_model.get("metrics"))
        gate = DiagnosticsModule.dict_or_empty(diag.get("shadow_gate_15m"))
        self._app._model_gate_quality = {
            "quality": metrics.get("quality"),
            "lift_bps": metrics.get("lift_bps"),
            "best_threshold": metrics.get("best_threshold"),
            "gate_total_count": gate.get("total_count", 0) or 0,
            "gate_lift_vs_all_bps": gate.get("lift_vs_all_bps"),
        }
        self._app._model_gate_quality_checked_at = datetime.now(tz=UTC)

    def model_gate_quality_allows_canary(self) -> tuple[bool, str]:
        assert self._app._settings is not None
        if not self._app._model_gate_quality:
            return False, "quality_unknown"
        expected_quality = str(self._app._settings.MODEL_GATE_CANARY_MIN_QUALITY).upper()
        quality = str(self._app._model_gate_quality.get("quality") or "").upper()
        if expected_quality and quality != expected_quality:
            return False, f"quality_not_{expected_quality.lower()}:{quality or 'none'}"
        gate_total = int(self._app._model_gate_quality.get("gate_total_count") or 0)
        if gate_total < int(self._app._settings.MODEL_GATE_CANARY_MIN_OBSERVATIONS):
            return False, f"insufficient_gate_observations:{gate_total}"
        lift = self._app._model_gate_quality.get("gate_lift_vs_all_bps")
        if lift is None or float(lift) < float(self._app._settings.MODEL_GATE_CANARY_MIN_LIFT_BPS):
            return False, f"insufficient_gate_lift:{lift}"
        return True, "quality_ok"

    def model_gate_canary_blocks(self, gate_decision: str, threshold: float, score: float) -> tuple[bool, str]:
        """Decide whether observational gate may block execution without starving trades."""
        assert self._app._settings is not None
        if not self._app._settings.MODEL_GATE_CANARY_ENABLED:
            return False, "canary_disabled"
        if gate_decision != "GATE_BLOCK":
            self._app._model_gate_recent_blocks.append(False)
            return False, "gate_pass"
        quality_ok, quality_reason = self.model_gate_quality_allows_canary()
        if not quality_ok:
            self._app._model_gate_recent_blocks.append(False)
            return False, quality_reason

        recent = list(self._app._model_gate_recent_blocks)
        block_rate = (sum(recent) / len(recent) * 100.0) if recent else 0.0
        if len(recent) >= self._app._settings.MODEL_GATE_CANARY_MIN_OBSERVATIONS:
            if block_rate >= self._app._settings.MODEL_GATE_CANARY_MAX_BLOCK_RATE_PCT:
                self._app._model_gate_recent_blocks.append(False)
                return False, f"max_block_rate_guard:{block_rate:.1f}%"

        self._app._model_gate_block_counter += 1
        every_n = max(1, int(self._app._settings.MODEL_GATE_CANARY_ALLOW_EVERY_NTH_BLOCKED))
        if self._app._model_gate_block_counter % every_n == 0:
            self._app._model_gate_recent_blocks.append(False)
            return False, f"sample_through_every_{every_n}"

        self._app._model_gate_recent_blocks.append(True)
        return True, f"score_below_threshold:{score:.3f}<{threshold:.3f}"

    @staticmethod
    def feature_values_for_side(vec: FeatureVector, side: str) -> tuple[list[str], list[float]]:
        from trader.training.feature_side import feature_values_for_side

        return feature_values_for_side(vec, side)

    async def sample_confirmed_candle(self, symbol: str, interval: str, vec: Any) -> None:
        """Record a training sample on every confirmed 1m candle.

        Writes a feature snapshot plus a RULE_BASELINE_V1 prediction event whose
        direction is the rule trend (EMA9 vs EMA21) and decision=SHADOW_CANDLE.
        The outcome resolver labels these like any other event, multiplying
        training-sample accumulation ~100x versus signal-only sampling.
        SHADOW_CANDLE events are excluded from signal statistics.
        """
        assert self._app._settings is not None
        if (
            not self._app._settings.CANDLE_SAMPLING_ENABLED
            or interval != _WS_INTERVAL
            or self._app._trade_journal is None
            or not self._app._trade_journal.is_enabled
        ):
            return
        try:
            f = dict(zip(vec.feature_names, vec.values, strict=True))
            ema9 = f.get("ema_9")
            ema21 = f.get("ema_21")
            if ema9 is None or ema21 is None:
                return
            # ema_* features are normalised distances to close; their ordering
            # matches the raw EMA ordering, so this is the rule trend direction.
            side = "Buy" if ema9 > ema21 else "Sell"
            model_feature_names, model_feature_values = self.feature_values_for_side(vec, side)

            candles = self._app._candle_store.confirmed(symbol, interval) if self._app._candle_store else []
            if not candles:
                return
            candle_open_time = candles[-1].open_time
            # One sample per candle per symbol (Bybit can re-send confirms)
            if self._app._last_candle_sample_at.get(symbol) == candle_open_time:
                return
            self._app._last_candle_sample_at[symbol] = candle_open_time

            schema_hash = hashlib.sha256(json.dumps(model_feature_names).encode()).hexdigest()[:16]
            snapshot_id = await self._app._trade_journal.record_feature_snapshot(
                symbol=symbol,
                interval=interval,
                candle_open_time=candle_open_time,
                feature_schema_hash=schema_hash,
                feature_names=model_feature_names,
                feature_values=model_feature_values,
            )
            if not snapshot_id:
                return
            await self._app._trade_journal.record_prediction_event(
                symbol=symbol,
                interval=interval,
                model_version="RULE_BASELINE_V1",
                score=0.5,
                strategy_signal=side,
                decision="SHADOW_CANDLE",
                feature_snapshot_id=snapshot_id,
                metadata={"source": "candle_sampler", "strategy_id": "candle_sampler_v1"},
            )

            self._app._candle_sampler_total += 1

            # Challenger shadow gate on every sampled candle. Signal-only shadow
            # scoring accumulates GATE_PASS/GATE_BLOCK observations slower than
            # the auto-trainer rotates model versions, so per-version gate stats
            # (lift, paper gate) would otherwise stay at zero forever.
            if self._app._settings.MODEL_SHADOW_SCORING_ENABLED and self._app._model_registry is not None:
                shadow_prediction = self._app._model_registry.score_shadow(model_feature_values, model_feature_names)
                if shadow_prediction is not None:
                    self._app._candle_sampler_scored += 1
                    threshold = self.model_gate_threshold(None)
                    gate_decision = None
                    gate_reason = "shadow_gate_disabled"
                    if self._app._settings.MODEL_SHADOW_GATE_ENABLED:
                        gate_decision = "GATE_PASS" if shadow_prediction.score >= threshold else "GATE_BLOCK"
                        gate_reason = (
                            "score_meets_threshold" if gate_decision == "GATE_PASS" else "score_below_threshold"
                        )
                        if gate_decision == "GATE_PASS":
                            self._app._candle_sampler_gate_pass += 1
                        else:
                            self._app._candle_sampler_gate_block += 1
                    await self._app._trade_journal.record_prediction_event(
                        symbol=symbol,
                        interval=interval,
                        model_version=shadow_prediction.model_version,
                        score=shadow_prediction.score,
                        strategy_signal=side,
                        decision=gate_decision,
                        feature_snapshot_id=snapshot_id,
                        metadata={
                            "source": "candle_sampler_shadow",
                            "confidence": shadow_prediction.confidence,
                            "gate_reason": gate_reason,
                            "threshold": threshold,
                        },
                    )
                else:
                    self._app._candle_sampler_no_model += 1
                    # Only warn once per 50 misses to avoid log spam
                    if self._app._candle_sampler_no_model % 50 == 1:
                        challenger = (
                            self._app._model_registry.challenger if self._app._model_registry is not None else None
                        )
                        log.warning(
                            "candle_sampler.shadow_score_unavailable",
                            symbol=symbol,
                            feature_count=len(vec.feature_names),
                            challenger_version=(challenger.version if challenger is not None else None),
                            challenger_feature_count=(
                                len(challenger.feature_names) if challenger is not None else None
                            ),
                            no_model_count=self._app._candle_sampler_no_model,
                        )

            # Periodic health summary — emitted every 200 candles (~30 min at 7 symbols)
            if self._app._candle_sampler_total % 200 == 0:
                score_rate = (
                    round(self._app._candle_sampler_scored / self._app._candle_sampler_total, 3)
                    if self._app._candle_sampler_total
                    else 0.0
                )
                log.info(
                    "candle_sampler.health",
                    total=self._app._candle_sampler_total,
                    scored=self._app._candle_sampler_scored,
                    no_model=self._app._candle_sampler_no_model,
                    gate_pass=self._app._candle_sampler_gate_pass,
                    gate_block=self._app._candle_sampler_gate_block,
                    score_rate=score_rate,
                )

        except Exception as exc:
            log.warning("candle_sampler.failed", symbol=symbol, error=str(exc))

    def bucket_blocked(self, regime_ctx: Any) -> bool:
        """True when the current (regime, volatility, UTC hour) bucket is toxic.

        A bucket blocks only with >= BUCKET_MIN_SAMPLES resolved outcomes and an
        average net return below BUCKET_BLOCK_AVG_BPS — small samples never block.
        In shadow mode the gate is skipped so virtual orders can accumulate training data.
        SCALP strict shadow keeps the gate enabled to avoid paper-trading toxic pairs.
        """
        assert self._app._settings is not None
        if not self.expectancy_gates_apply():
            return False
        if not self._app._settings.BUCKET_BLOCK_ENABLED:
            return False
        regime = (
            regime_ctx.regime.value
            if regime_ctx is not None and getattr(regime_ctx, "regime", None) is not None
            else "UNKNOWN"
        )
        volatility = (
            regime_ctx.volatility_level.value
            if regime_ctx is not None and getattr(regime_ctx, "volatility_level", None) is not None
            else "UNKNOWN"
        )
        hour = datetime.now(tz=UTC).hour
        stats = self._app._bucket_stats.get((regime, volatility, hour))
        if stats is not None:
            avg_bps, count = stats
            if count >= self._app._settings.BUCKET_MIN_SAMPLES and avg_bps < self._app._settings.BUCKET_BLOCK_AVG_BPS:
                return True
        if not self._app._settings.HOUR_BLOCK_ENABLED:
            return False
        hour_stats = self._app._hour_stats.get(hour)
        if hour_stats is None:
            return False
        hour_avg_bps, hour_count = hour_stats
        return bool(
            hour_count >= self._app._settings.HOUR_MIN_SAMPLES and hour_avg_bps < self._app._settings.HOUR_BLOCK_AVG_BPS
        )

    def symbol_side_blocked(self, symbol: str, side: str) -> bool:
        """True when a symbol+side pair has proven negative expectancy.

        In shadow mode the gate is skipped so virtual orders can accumulate training data.
        SCALP strict shadow keeps the gate enabled to avoid paper-trading toxic pairs.
        """

        assert self._app._settings is not None
        if not self.expectancy_gates_apply():
            return False
        if not self._app._settings.SYMBOL_SIDE_BLOCK_ENABLED or not self._app._symbol_side_stats:
            return False
        stats = self._app._symbol_side_stats.get((symbol, side))
        if stats is None:
            return False
        avg_bps, count = stats
        return bool(
            count >= self._app._settings.SYMBOL_SIDE_MIN_SAMPLES
            and avg_bps < self._app._settings.SYMBOL_SIDE_BLOCK_AVG_BPS
        )

    def strategy_blocked(self, strategy_id: str) -> bool:
        """Block a strategy after its exploration sample proves net-negative."""

        assert self._app._settings is not None
        if not self.expectancy_gates_apply():
            return False
        if not self._app._settings.STRATEGY_BLOCK_ENABLED:
            return False
        stats = self._app._strategy_stats.get(strategy_id)
        if stats is None:
            return False
        avg_bps, count = stats
        return bool(
            count >= self._app._settings.STRATEGY_MIN_SAMPLES and avg_bps < self._app._settings.STRATEGY_BLOCK_AVG_BPS
        )

    def shadow_probe_side_blocked(self, symbol: str, side: str) -> bool:
        """Block probe entries on symbol+side pairs with negative paper baseline."""

        assert self._app._settings is not None
        if not self._app._settings.SHADOW_PROBE_SIDE_BLOCK_ENABLED:
            return False
        stats = self._app._shadow_probe_side_stats.get((symbol, side))
        if stats is None:
            return False
        avg_bps, count = stats
        return bool(
            count >= self._app._settings.SHADOW_PROBE_SIDE_MIN_SAMPLES
            and avg_bps < self._app._settings.SHADOW_PROBE_SIDE_BLOCK_AVG_BPS
        )

    def shadow_probe_quality_allows(self, symbol: str, side: str) -> bool:
        """Require non-negative recent probe baseline when enough samples exist."""

        assert self._app._settings is not None
        if not self._app._settings.SHADOW_PROBE_QUALITY_FILTER_ENABLED:
            return True
        stats = self._app._shadow_probe_side_stats.get((symbol, side))
        if stats is None:
            return True
        avg_bps, count = stats
        if count < self._app._settings.SHADOW_PROBE_BASELINE_MIN_SAMPLES:
            return True
        return avg_bps >= self._app._settings.SHADOW_PROBE_BASELINE_MIN_AVG_BPS

    def shadow_probe_symbol_allowed(self, symbol: str) -> bool:
        """Restrict probes to top-performing symbols when configured."""

        assert self._app._settings is not None
        top_n = int(self._app._settings.SHADOW_PROBE_SYMBOL_TOP_N)
        if top_n <= 0:
            return True
        eligible = self._app._shadow_probe_eligible_symbols
        if eligible is None:
            return True
        return symbol in eligible

    def record_shadow_probe_symbol_subscribed(self, symbols: list[str]) -> None:
        """Mark screener-added symbols so probe warmup can skip unstable first minutes."""
        now = datetime.now(tz=UTC)
        for symbol in symbols:
            if symbol:
                self._app._shadow_probe_symbol_subscribed_at[str(symbol)] = now

    def shadow_probe_symbol_warmed_up(self, symbol: str) -> bool:
        """Return False while a newly subscribed symbol is still in probe warmup."""

        assert self._app._settings is not None
        warmup_s = int(self._app._settings.SHADOW_PROBE_SYMBOL_WARMUP_SECONDS)
        if warmup_s <= 0:
            return True
        subscribed_at = self._app._shadow_probe_symbol_subscribed_at.get(symbol)
        if subscribed_at is None:
            return False
        return datetime.now(tz=UTC) - subscribed_at >= timedelta(seconds=warmup_s)

    def shadow_probe_regime_allows(self, regime_ctx: Any | None) -> bool:
        """Block probes in choppy/uncertain regimes where OBI mean-reversion loses."""

        assert self._app._settings is not None
        if bool(getattr(self._app._settings, "SHADOW_PROBE_RESEARCH_PROFILE_V2", False)):
            if bool(getattr(self._app._settings, "SHADOW_PROBE_PAPER_COLLECTION_MODE", False)):
                allowed = {
                    part.strip()
                    for part in str(
                        getattr(self._app._settings, "SHADOW_PROBE_PAPER_REGIMES", "")
                        or "SIDEWAYS,HIGH_VOLATILITY,UNCERTAIN"
                    ).split(",")
                    if part.strip()
                }
            else:
                allowed = {"HIGH_VOLATILITY"}
        else:
            allowed = {
                part.strip()
                for part in str(self._app._settings.SHADOW_PROBE_ALLOWED_REGIMES or "").split(",")
                if part.strip()
            }
        if not allowed:
            return True
        if regime_ctx is None or getattr(regime_ctx, "regime", None) is None:
            return False
        regime = getattr(regime_ctx.regime, "value", str(regime_ctx.regime))
        return str(regime) in allowed

    @staticmethod
    def compute_shadow_probe_eligible_symbols(
        symbol_stats: dict[str, tuple[float, int]],
        *,
        top_n: int,
        min_samples: int,
        min_avg_bps: float,
    ) -> set[str] | None:
        """Return top-N probe-eligible symbols, or None during warmup."""

        if top_n <= 0:
            return None
        ranked = [
            symbol
            for symbol, (avg_bps, count) in symbol_stats.items()
            if count >= min_samples and avg_bps >= min_avg_bps
        ]
        if not ranked:
            return None
        ranked.sort(key=lambda symbol: symbol_stats[symbol][0], reverse=True)
        return set(ranked[:top_n])

    def record_shadow_close(self, symbol: str, reason: str, pnl_pct: float) -> None:
        """Track shadow TP/SL results and arm a cooldown after poor recent outcomes."""

        assert self._app._settings is not None
        if not self._app._settings.SHADOW_LOSS_GUARD_ENABLED:
            return
        now = datetime.now(tz=UTC)
        self._app._shadow_closed_results.append((now, reason, float(pnl_pct)))
        window_size = max(1, int(self._app._settings.SHADOW_LOSS_GUARD_WINDOW))
        recent = list(self._app._shadow_closed_results)[-window_size:]
        min_closed = max(1, int(self._app._settings.SHADOW_LOSS_GUARD_MIN_CLOSED))
        if len(recent) < min_closed:
            return
        losses = [value for _, _, value in recent if value < 0]
        loss_rate = len(losses) / len(recent)
        avg_pnl = sum(value for _, _, value in recent) / len(recent)
        if loss_rate >= float(self._app._settings.SHADOW_LOSS_GUARD_MAX_LOSS_RATE) and avg_pnl <= float(
            self._app._settings.SHADOW_LOSS_GUARD_MIN_AVG_PNL_PCT
        ):
            cooldown_s = max(0, int(self._app._settings.SHADOW_LOSS_GUARD_COOLDOWN_SECONDS))
            self._app._shadow_loss_guard_until = now + timedelta(seconds=cooldown_s)
            log.warning(
                "shadow_loss_guard.activated",
                symbol=symbol,
                reason=reason,
                recent_count=len(recent),
                loss_rate=round(loss_rate, 3),
                avg_pnl_pct=round(avg_pnl, 4),
                cooldown_seconds=cooldown_s,
            )

    @staticmethod
    def shadow_exit_hit(
        position: dict[str, Any],
        *,
        high: float,
        low: float,
        current_price: float | None = None,
        now: datetime | None = None,
        max_hold_seconds: int | None = None,
    ) -> tuple[str, float] | None:
        """Return the first conservative shadow exit for one candle.

        TP/SL wins first. If neither is touched, an optional time exit frees
        SHADOW research slots at the model horizon instead of letting paper
        positions occupy all slots indefinitely in a sideways market.
        """

        side = str(position.get("side") or "")
        tp = float(position["tp"])
        sl = float(position["sl"])
        if side == "Buy":
            tp_hit = high >= tp
            sl_hit = low <= sl
            if tp_hit and sl_hit:
                return "SL", sl
            if tp_hit:
                return "TP", tp
            if sl_hit:
                return "SL", sl
            return SignalPolicyModule.shadow_time_exit_hit(
                position,
                current_price=current_price,
                now=now,
                max_hold_seconds=max_hold_seconds,
            )
        if side == "Sell":
            tp_hit = low <= tp
            sl_hit = high >= sl
            if tp_hit and sl_hit:
                return "SL", sl
            if tp_hit:
                return "TP", tp
            if sl_hit:
                return "SL", sl
            return SignalPolicyModule.shadow_time_exit_hit(
                position,
                current_price=current_price,
                now=now,
                max_hold_seconds=max_hold_seconds,
            )
        return None

    @staticmethod
    def shadow_time_exit_hit(
        position: dict[str, Any],
        *,
        current_price: float | None,
        now: datetime | None,
        max_hold_seconds: int | None,
    ) -> tuple[str, float] | None:
        """Close a SHADOW paper position at horizon when TP/SL did not hit."""

        if current_price is None or current_price <= 0 or not max_hold_seconds or max_hold_seconds <= 0:
            return None
        opened_at = position.get("opened_at")
        if not isinstance(opened_at, datetime):
            return None
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=UTC)
        current_time = now or datetime.now(tz=UTC)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=UTC)
        if (current_time - opened_at).total_seconds() < max_hold_seconds:
            return None
        return "TIME", float(current_price)

    @staticmethod
    def shadow_gross_pnl_pct(position: dict[str, Any], exit_price: float) -> float:
        """Return direction-aware gross shadow PnL percent before cost model."""

        entry = float(position["entry"])
        if entry <= 0:
            raise ValueError("shadow entry must be positive")
        if str(position.get("side") or "") == "Sell":
            return (entry - exit_price) / entry * 100.0
        return (exit_price - entry) / entry * 100.0

    def shadow_pnl_pct(self, position: dict[str, Any], exit_price: float) -> float:
        """Return direction-aware net shadow PnL percent after estimated costs."""

        gross = self.shadow_gross_pnl_pct(position, exit_price)
        if self._app._settings is None:
            return gross
        taker_fee_pct = float(self._app._settings.DEFAULT_LINEAR_TAKER_FEE_RATE) * 100.0
        round_trip_fee_pct = taker_fee_pct * 2.0
        spread_pct = float(self._app._settings.SCREENER_MAX_SPREAD_BPS) / 100.0
        slippage_pct = float(self._app._settings.EXPECTED_SLIPPAGE_PCT) * 2.0
        return gross - round_trip_fee_pct - spread_pct - slippage_pct

    def shadow_loss_guard_blocks(self) -> bool:
        """Return true while recent shadow losses should suppress new entries."""

        assert self._app._settings is not None
        if not self._app._settings.SHADOW_LOSS_GUARD_ENABLED or self._app._shadow_loss_guard_until is None:
            return False
        now = datetime.now(tz=UTC)
        if now >= self._app._shadow_loss_guard_until:
            self._app._shadow_loss_guard_until = None
            return False
        return True

    def trend_confirmation_intervals(self) -> list[str]:
        assert self._app._settings is not None
        raw = str(getattr(self._app._settings, "TREND_CONFIRMATION_INTERVALS", "") or "")
        return [part.strip() for part in raw.split(",") if part.strip() and part.strip() != _WS_INTERVAL]

    def trend_mtf_confirmed(self, symbol: str, side: str) -> bool:
        """Confirm a 1m trend signal with higher-timeframe features."""

        assert self._app._settings is not None
        if not self._app._settings.TREND_MTF_CONFIRMATION_ENABLED:
            return True
        if self._app._feature_pipeline is None:
            return False
        intervals = self._app._trend_confirmation_intervals()
        if not intervals:
            return True
        confirmations = 0
        for interval in intervals:
            vec = self._app._feature_pipeline.latest(symbol, interval)
            if vec is None:
                continue
            f = dict(zip(vec.feature_names, vec.values, strict=True))
            ema9 = f.get("ema_9")
            ema21 = f.get("ema_21")
            slope9 = f.get("ema_slope_9")
            macd_hist = f.get("macd_hist")
            if any(value is None for value in (ema9, ema21, slope9, macd_hist)):
                continue
            assert ema9 is not None
            assert ema21 is not None
            assert slope9 is not None
            assert macd_hist is not None
            if side == "Buy":
                confirmed = ema9 > ema21 and slope9 > 0 and macd_hist > 0
            else:
                confirmed = ema9 < ema21 and slope9 < 0 and macd_hist < 0
            if confirmed:
                confirmations += 1
        return confirmations > 0
