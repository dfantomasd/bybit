"""Strategy ensemble loop and execution wiring."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from trader.domain.enums import TradingMode
from trader.domain.models import FeatureVector
from trader.modules.base import AppBoundModule
from trader.monitoring.logging import get_logger
from trader.runtime.constants import (
    _BALANCE_REFRESH_INTERVAL,
    _FALLBACK_BALANCE_USD,
    _ML_REPLACEMENT_COUNTER,
    _STRATEGY_LOOP_INTERVAL,
    _SYMBOLS,
    _WS_INTERVAL,
)

log = get_logger(__name__)


class TradingLoopModule(AppBoundModule):
    name = "trading"

    async def start(self) -> None:
        """Run strategy ensemble → RiskManager → ExecutionEngine."""
        from trader.strategies.ensemble import StrategyEnsemble
        from trader.strategies.trend import EMAcrossoverStrategy

        assert self._app._settings is not None

        # Fetch initial balance to seed RiskManager
        from trader.config import get_risk_profile_config

        profile_cfg = get_risk_profile_config(self._app._settings.RISK_PROFILE)
        initial_capital = await self._app._refresh_balance()
        if initial_capital <= Decimal("0"):
            initial_capital = _FALLBACK_BALANCE_USD
            log.warning(
                "strategy_loop.using_fallback_capital",
                capital=str(initial_capital),
            )

        # Build risk + execution stack
        await self._app._init_risk_manager(initial_capital)
        await self._app._init_execution_engine()
        await self._app._modules.market_data.prefetch_ticker_turnover(self._app._active_symbols())

        # One symbol-agnostic strategy instance handles ALL screener symbols
        strategies: list[Any] = []
        is_scalp = self._app._is_scalp_profile()
        include_trend = self._app._settings.TREND_STRATEGY_ENABLED and not (
            is_scalp and self._app._settings.SCALP_DISABLE_TREND_STRATEGY
        )
        if include_trend:
            strategies.append(
                EMAcrossoverStrategy(
                    symbol=None,  # None = evaluate any symbol passed in
                    allow_short=True,
                    min_qty_usd=5.0,  # Bybit minimum notional is $5
                    max_risk_pct=0.01,  # 1% of balance per trade
                    min_adx=self._app._settings.TREND_MIN_ADX,
                    block_negative_funding_oi=self._app._settings.TREND_BLOCK_NEGATIVE_FUNDING_OI,
                    taker_fee_pct=self._app._settings.DEFAULT_LINEAR_TAKER_FEE_RATE * 100,
                    expected_slippage_pct=self._app._settings.EXPECTED_SLIPPAGE_PCT,
                    max_spread_bps=self._app._settings.SCREENER_MAX_SPREAD_BPS,
                    min_net_return_pct=self._app._settings.MIN_NET_TREND_RETURN_PCT,
                )
            )
        elif is_scalp:
            log.info("scalp_profile.trend_strategy_disabled")

        def _spread_for(symbol: str) -> float | None:
            """Latest screener spread for the symbol; None when unknown."""
            if self._app._screener is None:
                return None
            for scored in self._app._screener.wide_universe:
                if scored.symbol == symbol:
                    return float(scored.spread_bps)
            return None

        if is_scalp:
            priority_order = [
                sid.strip() for sid in self._app._settings.SCALP_STRATEGY_PRIORITY_ORDER.split(",") if sid.strip()
            ]
        else:
            priority_order = [
                sid.strip() for sid in self._app._settings.STRATEGY_PRIORITY_ORDER.split(",") if sid.strip()
            ]
        strategy_priorities = {
            strategy_id: len(priority_order) - index for index, strategy_id in enumerate(priority_order)
        }

        if self._app._settings.SCALP_STRATEGY_ENABLED and self._app._candle_store is not None:
            from trader.strategies.scalp_micro import ScalpMicroStrategy

            strategies.append(
                ScalpMicroStrategy(
                    candle_store=self._app._candle_store,
                    interval=_WS_INTERVAL,
                    spread_provider=_spread_for,
                    taker_fee_pct=self._app._settings.DEFAULT_LINEAR_TAKER_FEE_RATE * 100,
                    expected_slippage_pct=self._app._settings.EXPECTED_SLIPPAGE_PCT,
                    min_net_return_pct=self._app._settings.MIN_NET_SCALP_RETURN_PCT,
                    max_spread_bps=self._app._settings.MAX_SPREAD_BPS_SCALP,
                    cooldown_seconds=self._app._settings.SCALP_COOLDOWN_SECONDS,
                    max_trades_per_minute=self._app._settings.SCALP_MAX_TRADES_PER_MINUTE,
                    risk_pct=0.01,
                    max_position_notional_usd=self._app._settings.SCALP_MAX_POSITION_NOTIONAL_USD,
                    min_qty_usd=5.0,
                    diag_hook=self._app._record_diag,
                    imbalance_provider=(
                        self._app._orderbook_tracker.latest_imbalance
                        if self._app._orderbook_tracker is not None
                        else None
                    ),
                    min_imbalance=self._app._settings.SCALP_MIN_OB_IMBALANCE,
                    shadow_relaxed=(
                        self._app._initial_shadow_mode()
                        and self._app._settings.SHADOW_RELAX_SCALP_FILTERS
                        and not self._app._scalp_strict_shadow()
                    ),
                )
            )
            log.info(
                "scalp_micro.enabled",
                max_spread_bps=self._app._settings.MAX_SPREAD_BPS_SCALP,
                min_net_return_pct=self._app._settings.MIN_NET_SCALP_RETURN_PCT,
                max_trades_per_minute=self._app._settings.SCALP_MAX_TRADES_PER_MINUTE,
            )

        if self._app._initial_shadow_mode() and self._app._settings.SHADOW_PROBE_ENABLED:
            from trader.domain.models import InstrumentInfo
            from trader.risk.net_edge import NetEdgeParams
            from trader.strategies.shadow_probe import ShadowProbeStrategy

            def _probe_instrument_info(symbol: str) -> InstrumentInfo | None:
                if self._app._execution_engine is None:
                    return None
                cached = self._app._execution_engine._instrument_cache.get(symbol)
                return cached[0] if cached else None

            probe_cost_params = NetEdgeParams(
                taker_fee_pct=self._app._settings.DEFAULT_LINEAR_TAKER_FEE_RATE * 100,
                expected_slippage_pct=self._app._settings.EXPECTED_SLIPPAGE_PCT,
                max_spread_bps=self._app._settings.SCREENER_MAX_SPREAD_BPS,
                funding_buffer_pct=self._app._settings.FUNDING_BUFFER_PCT,
                safety_margin_pct=self._app._settings.NET_EDGE_SAFETY_MARGIN_PCT,
            )
            probe_min_net_return_pct = self._app._settings.SHADOW_PROBE_MIN_NET_RETURN_PCT

            def _probe_symbol_allowed(symbol: str) -> bool:
                return self._app._shadow_probe_symbol_warmed_up(symbol) and self._app._shadow_probe_symbol_allowed(
                    symbol
                )

            def _probe_regime_allows(feature_vector: FeatureVector) -> bool:
                if self._app._regime_classifier is None:
                    return False
                try:
                    regime_ctx = self._app._regime_classifier.classify(feature_vector)
                except Exception:
                    return False
                return self._app._shadow_probe_regime_allows(regime_ctx)

            def _probe_side_blocked(symbol: str, side: str) -> bool:
                if self._app._shadow_probe_side_blocked(symbol, side):
                    return True
                if not self._app._shadow_probe_quality_allows(symbol, side):
                    self._app._record_diag("shadow_probe_quality_blocked")
                    self._app._record_diag(f"shadow_probe_quality_blocked:{symbol}:{side}")
                    return True
                return False

            research_v2 = bool(self._app._settings.SHADOW_PROBE_RESEARCH_PROFILE_V2)
            probe_max_open_positions = 4 if research_v2 else self._app._settings.SHADOW_PROBE_MAX_OPEN_POSITIONS
            probe_burst_max_signals = 6 if research_v2 else self._app._settings.SHADOW_PROBE_BURST_MAX_SIGNALS
            probe_burst_cooldown_seconds = (
                300 if research_v2 else self._app._settings.SHADOW_PROBE_BURST_COOLDOWN_SECONDS
            )
            probe_cooldown_seconds = 180 if research_v2 else self._app._settings.SHADOW_PROBE_COOLDOWN_SECONDS

            strategies.append(
                ShadowProbeStrategy(
                    imbalance_provider=(
                        self._app._orderbook_tracker.latest_imbalance
                        if self._app._orderbook_tracker is not None
                        else None
                    ),
                    instrument_info_provider=_probe_instrument_info,
                    side_blocked=_probe_side_blocked,
                    symbol_allowed=_probe_symbol_allowed,
                    regime_allows=_probe_regime_allows,
                    open_positions_count=(
                        self._app._execution_engine.open_position_count
                        if self._app._execution_engine is not None
                        else None
                    ),
                    max_open_positions=probe_max_open_positions,
                    burst_max_signals=probe_burst_max_signals,
                    burst_window_seconds=self._app._settings.SHADOW_PROBE_BURST_WINDOW_SECONDS,
                    burst_cooldown_seconds=probe_burst_cooldown_seconds,
                    min_abs_imbalance=self._app._settings.SHADOW_PROBE_MIN_ABS_IMBALANCE,
                    cooldown_seconds=probe_cooldown_seconds,
                    max_notional_usd=self._app._settings.SHADOW_PROBE_MAX_NOTIONAL_USD,
                    min_tp_pct=self._app._settings.SHADOW_PROBE_MIN_TP_PCT,
                    max_tp_pct=self._app._settings.SHADOW_PROBE_MAX_TP_PCT,
                    min_sl_pct=self._app._settings.SHADOW_PROBE_MIN_SL_PCT,
                    min_net_return_pct=probe_min_net_return_pct,
                    min_net_reward_risk=self._app._settings.SHADOW_PROBE_MIN_NET_REWARD_RISK,
                    min_notional_buffer_pct=self._app._settings.SHADOW_PROBE_MIN_NOTIONAL_BUFFER_PCT,
                    cost_params=probe_cost_params,
                    sell_enabled=self._app._settings.SHADOW_PROBE_SELL_ENABLED,
                    diag_hook=self._app._record_diag,
                )
            )
            log.info(
                "shadow_probe.enabled",
                research_profile_v2=research_v2,
                paper_collection_mode=bool(self._app._settings.SHADOW_PROBE_PAPER_COLLECTION_MODE),
                paper_regimes=(
                    self._app._settings.SHADOW_PROBE_PAPER_REGIMES
                    if self._app._settings.SHADOW_PROBE_PAPER_COLLECTION_MODE
                    else None
                ),
                min_abs_imbalance=self._app._settings.SHADOW_PROBE_MIN_ABS_IMBALANCE,
                cooldown_seconds=probe_cooldown_seconds,
                max_notional_usd=self._app._settings.SHADOW_PROBE_MAX_NOTIONAL_USD,
                min_tp_pct=self._app._settings.SHADOW_PROBE_MIN_TP_PCT,
                min_sl_pct=self._app._settings.SHADOW_PROBE_MIN_SL_PCT,
                min_net_return_pct=probe_min_net_return_pct,
                symbol_top_n=self._app._settings.SHADOW_PROBE_SYMBOL_TOP_N,
                symbol_warmup_seconds=self._app._settings.SHADOW_PROBE_SYMBOL_WARMUP_SECONDS,
                max_open_positions=probe_max_open_positions,
                burst_max_signals=probe_burst_max_signals,
                burst_cooldown_seconds=probe_burst_cooldown_seconds,
                sell_enabled=self._app._settings.SHADOW_PROBE_SELL_ENABLED,
            )

        if self._app._initial_shadow_mode() and self._app._settings.DISCOVERED_RULE_STRATEGY_ENABLED:
            from trader.strategies.discovered_rule import (
                DiscoveredRuleStrategy,
                auto_generate_discovered_rules_file,
                load_discovered_rules,
                writable_discovered_rules_path,
                write_discovered_rules_failure_report,
            )

            rules_path = writable_discovered_rules_path(self._app._settings.DISCOVERED_RULES_PATH)
            configured_rules_path = self._app._settings.DISCOVERED_RULES_PATH
            discovered_rules = load_discovered_rules(
                rules_path,
                min_validation_count=self._app._settings.DISCOVERED_RULE_MIN_VALIDATION_COUNT,
                min_validation_net_bps=self._app._settings.DISCOVERED_RULE_MIN_VALIDATION_NET_BPS,
                max_rules=self._app._settings.DISCOVERED_RULE_MAX_RULES,
            )
            if not discovered_rules and self._app._settings.DISCOVERED_RULE_AUTO_GENERATE:
                try:
                    generated_path, report = await auto_generate_discovered_rules_file(
                        configured_rules_path,
                        horizon=self._app._settings.MODEL_AUTO_TRAIN_HORIZON_MINUTES,
                        min_samples=self._app._settings.DISCOVERED_RULE_AUTO_GENERATE_MIN_SAMPLES,
                        min_train_count=self._app._settings.DISCOVERED_RULE_AUTO_GENERATE_MIN_TRAIN_COUNT,
                        min_validation_count=self._app._settings.DISCOVERED_RULE_MIN_VALIDATION_COUNT,
                        min_validation_net_bps=self._app._settings.DISCOVERED_RULE_MIN_VALIDATION_NET_BPS,
                        top_n=self._app._settings.DISCOVERED_RULE_MAX_RULES,
                        timeout_seconds=self._app._settings.DISCOVERED_RULE_AUTO_GENERATE_TIMEOUT_SECONDS,
                    )
                    self._app._record_diag("discovered_rule_auto_generated")
                    log.info(
                        "discovered_rule.auto_generated",
                        rules_path=str(generated_path),
                        status=report.get("status"),
                        sample_count=report.get("sample_count"),
                        rule_count=len(list(report.get("rules") or [])),
                    )
                    rules_path = generated_path
                    discovered_rules = load_discovered_rules(
                        rules_path,
                        min_validation_count=self._app._settings.DISCOVERED_RULE_MIN_VALIDATION_COUNT,
                        min_validation_net_bps=self._app._settings.DISCOVERED_RULE_MIN_VALIDATION_NET_BPS,
                        max_rules=self._app._settings.DISCOVERED_RULE_MAX_RULES,
                    )
                except Exception as exc:
                    failure_path = write_discovered_rules_failure_report(
                        configured_rules_path,
                        error=exc,
                        stage="auto_generate",
                    )
                    self._app._record_diag("discovered_rule_auto_generate_failed")
                    self._app._record_diag(f"discovered_rule_auto_generate_failed:{type(exc).__name__}")
                    log.warning(
                        "discovered_rule.auto_generate_failed",
                        rules_path=str(failure_path),
                        error=str(exc),
                    )
            if discovered_rules:
                strategies.append(
                    DiscoveredRuleStrategy(
                        rules=discovered_rules,
                        max_notional_usd=self._app._settings.DISCOVERED_RULE_MAX_NOTIONAL_USD,
                        tp_pct=self._app._settings.DISCOVERED_RULE_TP_PCT,
                        sl_pct=self._app._settings.DISCOVERED_RULE_SL_PCT,
                        min_confidence=self._app._settings.DISCOVERED_RULE_MIN_CONFIDENCE,
                        diag_hook=self._app._record_diag,
                    )
                )
                log.info(
                    "discovered_rule.enabled",
                    rules_path=str(rules_path),
                    rule_count=len(discovered_rules),
                    best_rule=discovered_rules[0].rule_id,
                    best_validation_bps=round(discovered_rules[0].validation_avg_net_bps, 3),
                )
            else:
                self._app._record_diag("discovered_rule_no_rules_loaded")
                log.info(
                    "discovered_rule.disabled_no_rules",
                    rules_path=str(rules_path),
                )

        if (
            self._app._settings.ORDER_FLOW_STRATEGY_ENABLED
            or self._app._settings.FUNDING_ARB_STRATEGY_ENABLED
            or self._app._settings.LIQUIDATION_HUNTING_STRATEGY_ENABLED
            or self._app._settings.MARKET_MAKING_STRATEGY_ENABLED
            or self._app._settings.STAT_ARB_STRATEGY_ENABLED
            or self._app._settings.MEAN_REVERSION_STRATEGY_ENABLED
            or self._app._settings.MACD_ZEROCROSS_STRATEGY_ENABLED
            or self._app._settings.ATR_BREAKOUT_STRATEGY_ENABLED
            or self._app._settings.VOLATILITY_SQUEEZE_STRATEGY_ENABLED
        ):
            from trader.risk.net_edge import NetEdgeParams
            from trader.strategies.advanced_alpha import (
                FundingArbitrageStrategy,
                LiquidationHuntingStrategy,
                MarketMakingStrategy,
                OrderFlowStrategy,
                StatisticalArbitrageStrategy,
                VolatilitySqueezeBreakoutStrategy,
            )
            from trader.strategies.basic_strategies import (
                ATRBreakoutStrategy,
                MACDZeroCrossStrategy,
                MeanReversionStrategy,
            )

            alpha_cost_params = NetEdgeParams(
                taker_fee_pct=self._app._settings.DEFAULT_LINEAR_TAKER_FEE_RATE * 100,
                expected_slippage_pct=self._app._settings.EXPECTED_SLIPPAGE_PCT,
                max_spread_bps=self._app._settings.SCREENER_MAX_SPREAD_BPS,
                funding_buffer_pct=self._app._settings.FUNDING_BUFFER_PCT,
                safety_margin_pct=self._app._settings.NET_EDGE_SAFETY_MARGIN_PCT,
            )
            alpha_min_net = self._app._settings.MIN_NET_ALPHA_RETURN_PCT

            if self._app._settings.ORDER_FLOW_STRATEGY_ENABLED and self._app._flow_tracker is not None:
                strategies.append(
                    OrderFlowStrategy(
                        flow_tracker=self._app._flow_tracker,
                        orderbook_tracker=self._app._orderbook_tracker,
                        min_flow_imbalance=self._app._settings.ORDER_FLOW_MIN_IMBALANCE,
                        min_book_imbalance=self._app._settings.ORDER_FLOW_MIN_BOOK_IMBALANCE,
                        max_spread_bps=self._app._settings.MAX_SPREAD_BPS_SCALP,
                        cost_params=alpha_cost_params,
                        min_net_return_pct=alpha_min_net,
                    )
                )
                log.info("advanced_alpha.strategy_active", strategy_id="order_flow_v1")
            elif self._app._settings.ORDER_FLOW_STRATEGY_ENABLED:
                log.warning(
                    "advanced_alpha.strategy_inactive",
                    strategy_id="order_flow_v1",
                    reason="flow_tracker_missing",
                )
            if self._app._settings.FUNDING_ARB_STRATEGY_ENABLED:
                strategies.append(
                    FundingArbitrageStrategy(
                        min_abs_funding_bps=self._app._settings.FUNDING_ARB_MIN_ABS_BPS,
                        cost_params=alpha_cost_params,
                        min_net_return_pct=alpha_min_net,
                    )
                )
                log.info("advanced_alpha.strategy_active", strategy_id="funding_arbitrage_v1")
            else:
                log.info("advanced_alpha.strategy_disabled", strategy_id="funding_arbitrage_v1")
            if self._app._settings.LIQUIDATION_HUNTING_STRATEGY_ENABLED and self._app._flow_tracker is not None:
                strategies.append(
                    LiquidationHuntingStrategy(
                        flow_tracker=self._app._flow_tracker,
                        min_liq_notional_usd=self._app._settings.LIQUIDATION_HUNTING_MIN_NOTIONAL_USD,
                        min_liq_imbalance=self._app._settings.LIQUIDATION_HUNTING_MIN_IMBALANCE,
                        cost_params=alpha_cost_params,
                        min_net_return_pct=alpha_min_net,
                    )
                )
                log.info("advanced_alpha.strategy_active", strategy_id="liquidation_hunting_v1")
            elif self._app._settings.LIQUIDATION_HUNTING_STRATEGY_ENABLED:
                log.warning(
                    "advanced_alpha.strategy_inactive",
                    strategy_id="liquidation_hunting_v1",
                    reason="flow_tracker_missing",
                )
            if self._app._settings.VOLATILITY_SQUEEZE_STRATEGY_ENABLED:
                strategies.append(
                    VolatilitySqueezeBreakoutStrategy(
                        squeeze_bw_threshold=self._app._settings.VOLATILITY_SQUEEZE_BB_BANDWIDTH,
                        cooldown_seconds=self._app._settings.VOLATILITY_SQUEEZE_COOLDOWN_SECONDS,
                        cost_params=alpha_cost_params,
                        min_net_return_pct=alpha_min_net,
                    )
                )
                log.info("advanced_alpha.strategy_active", strategy_id="volatility_squeeze_v1")
            else:
                log.info("advanced_alpha.strategy_disabled", strategy_id="volatility_squeeze_v1")

            # === Basic proven strategies ===
            if self._app._settings.MEAN_REVERSION_STRATEGY_ENABLED:
                strategies.append(
                    MeanReversionStrategy(
                        cost_params=alpha_cost_params,
                        min_net_return_pct=alpha_min_net,
                    )
                )
                log.info("basic.strategy_active", strategy_id="mean_reversion_v1")
            else:
                log.info("basic.strategy_disabled", strategy_id="mean_reversion_v1")
            if self._app._settings.MACD_ZEROCROSS_STRATEGY_ENABLED:
                strategies.append(
                    MACDZeroCrossStrategy(
                        cost_params=alpha_cost_params,
                        min_net_return_pct=alpha_min_net,
                    )
                )
                log.info("basic.strategy_active", strategy_id="macd_zerocross_v1")
            else:
                log.info("basic.strategy_disabled", strategy_id="macd_zerocross_v1")
            if self._app._settings.ATR_BREAKOUT_STRATEGY_ENABLED:
                strategies.append(
                    ATRBreakoutStrategy(
                        cost_params=alpha_cost_params,
                        min_net_return_pct=alpha_min_net,
                    )
                )
                log.info("basic.strategy_active", strategy_id="atr_breakout_v1")
            else:
                log.info("basic.strategy_disabled", strategy_id="atr_breakout_v1")

            if self._app._settings.MARKET_MAKING_STRATEGY_ENABLED:
                strategies.append(
                    MarketMakingStrategy(
                        spread_provider=_spread_for,
                        min_spread_bps=self._app._settings.MARKET_MAKING_MIN_SPREAD_BPS,
                        max_spread_bps=self._app._settings.MARKET_MAKING_MAX_SPREAD_BPS,
                        cost_params=alpha_cost_params,
                        min_net_return_pct=self._app._settings.MIN_NET_MARKET_MAKING_PCT,
                    )
                )
                log.info("advanced_alpha.strategy_active", strategy_id="market_making_v1")
            else:
                log.info("advanced_alpha.strategy_disabled", strategy_id="market_making_v1")
            if self._app._settings.STAT_ARB_STRATEGY_ENABLED:
                strategies.append(
                    StatisticalArbitrageStrategy(
                        min_zscore=self._app._settings.STAT_ARB_MIN_ZSCORE,
                        cost_params=alpha_cost_params,
                        min_net_return_pct=self._app._settings.MIN_NET_STAT_ARB_PCT,
                    )
                )
                log.info("advanced_alpha.strategy_active", strategy_id="statistical_arbitrage_v1")
            else:
                log.info("advanced_alpha.strategy_disabled", strategy_id="statistical_arbitrage_v1")
            log.info(
                "advanced_alpha.enabled",
                strategies=[s.strategy_id for s in strategies],
                priority_order=priority_order,
            )

        # Configure confluence signals: allow basic strategies to pass alone,
        # but commodity strategies (ema_crossover) require confirmation
        basic_strategy_ids = {"mean_reversion_v1", "macd_zerocross_v1", "atr_breakout_v1"}
        confirmation_required_for = {"ema_crossover_v1"}
        confirmation_sources = basic_strategy_ids | {
            "funding_arbitrage_v1",
            "volatility_squeeze_v1",
            "order_flow_v1",
            "liquidation_hunting_v1",
            "shadow_probe_hv_v2",
            "discovered_rule_v1",
        }

        self._app._strategy_ensemble = StrategyEnsemble(
            strategies=strategies,
            health_checker=self._app._health_checker,
            min_confidence=profile_cfg.min_confidence,
            strategy_priorities=strategy_priorities,
            confirmation_required_for=confirmation_required_for,
            confirmation_sources=confirmation_sources,
            min_confirmation_sources=1,
            diag_hook=self._app._record_diag,
        )
        log.info(
            "ensemble.configured",
            strategies=[s.strategy_id for s in strategies],
            confirmation_required_for=confirmation_required_for,
            confirmation_sources=confirmation_sources,
        )
        await self._app._refresh_closed_pnl_memory()

        # Initialise ML registry when shadow scoring, canary gate, or live decisions need it.
        needs_model_registry = (
            self._app._settings.MODEL_SHADOW_SCORING_ENABLED
            or self._app._settings.MODEL_GATE_CANARY_ENABLED
            or (self._app._settings.MODEL_ENABLED and self._app._settings.MODEL_ALLOW_LIVE_DECISIONS)
        )
        if needs_model_registry:
            try:
                from trader.ml.challenger import ModelRegistry

                self._app._model_registry = ModelRegistry(trade_journal=self._app._trade_journal)
                if self._app._trade_journal is not None and self._app._trade_journal.is_enabled:
                    await self._app._model_registry.load_active_model()
                    try:
                        self._app._update_model_gate_quality_from_diag(
                            await self._app._trade_journal.get_db_diagnostics()
                        )
                    except Exception as _qe:
                        log.debug("model_gate.startup_quality_refresh_failed", error=str(_qe))
                log.info("model_registry.initialized")
            except Exception as _mr_exc:
                log.warning("model_registry.init_failed", error=str(_mr_exc))

        _balance_tick: int = 0
        _effective_blocked_symbols: set[str] = set()
        # Shadow TP/SL tracker: symbol → {entry, tp, sl, side, opened_at}
        _shadow_positions: dict[str, dict[str, Any]] = {}

        async def _check_shadow_exits(symbol: str, current_price: float, high: float, low: float) -> None:
            """Close shadow positions that hit TP or SL."""
            pos = _shadow_positions.get(symbol)
            if pos is None:
                return
            max_hold_seconds = max(60, int(self._app._settings.MODEL_AUTO_TRAIN_HORIZON_MINUTES) * 60)
            hit_info = self._app._shadow_exit_hit(
                pos,
                high=high,
                low=low,
                current_price=current_price,
                max_hold_seconds=max_hold_seconds,
            )
            if hit_info:
                hit, exit_price = hit_info
                pnl_pct = self._app._shadow_pnl_pct(pos, exit_price)
                gross_pnl_pct = self._app._shadow_gross_pnl_pct(pos, exit_price)
                log.info(
                    "shadow.position_closed",
                    symbol=symbol,
                    reason=hit,
                    entry=pos["entry"],
                    exit=exit_price,
                    close=current_price,
                    high=high,
                    low=low,
                    pnl_pct=round(pnl_pct, 3),
                    gross_pnl_pct=round(gross_pnl_pct, 3),
                )
                self._app._record_shadow_close(symbol, hit, pnl_pct)
                del _shadow_positions[symbol]
                if self._app._execution_engine is not None:
                    await self._app._execution_engine.record_position_closed(symbol)
                self._app._trailing_stop_keys.discard(symbol)
                if self._app._telegram_bot is not None:
                    try:
                        label = "✅ TP" if hit == "TP" else ("⏱ TIME" if hit == "TIME" else "🛑 SL")
                        net_sign = "+" if pnl_pct >= 0 else ""
                        gross_sign = "+" if gross_pnl_pct >= 0 else ""
                        await self._app._telegram_bot.notify(
                            f"{label} {symbol} {pos['side']} closed\n"
                            f"Entry: {pos['entry']:.4f} → Exit: {exit_price:.4f}\n"
                            f"PnL: gross {gross_sign}{gross_pnl_pct:.2f}% | net {net_sign}{pnl_pct:.2f}% [SHADOW]"
                        )
                    except Exception as exc:
                        log.debug("telegram.shadow_exit_notify_failed", error=str(exc))

        async def process_symbol(symbol: str, balance: Decimal, capital: Decimal) -> None:
            """Evaluate one symbol: features → regime → ensemble → execution."""
            if symbol in _effective_blocked_symbols:
                log.debug("performance_filter.symbol_blocked", symbol=symbol)
                return

            if self._app._feature_pipeline is None:
                return

            vec = self._app._feature_pipeline.latest(symbol, _WS_INTERVAL)
            if vec is None:
                return

            candles = self._app._candle_store.latest(symbol, _WS_INTERVAL, 1) if self._app._candle_store else []
            if not candles:
                return
            last_candle = candles[-1]
            current_price = last_candle.close

            # Check shadow TP/SL exits first
            await _check_shadow_exits(symbol, current_price, last_candle.high, last_candle.low)
            if self._app._shadow_loss_guard_blocks():
                self._app._record_diag("shadow_loss_guard_blocked")
                log.info(
                    "strategy_loop.shadow_loss_guard_blocked",
                    symbol=symbol,
                    blocked_until=(
                        self._app._shadow_loss_guard_until.isoformat() if self._app._shadow_loss_guard_until else None
                    ),
                )
                return

            # Classify regime
            regime_ctx = None
            if self._app._regime_classifier is not None:
                try:
                    regime_ctx = self._app._regime_classifier.classify(vec)
                except Exception as exc:
                    log.warning("strategy_loop.regime_error", symbol=symbol, error=str(exc))

            # Skip symbols with an existing open position — the engine would
            # reject them anyway, but avoiding the ensemble call prevents
            # no_decision entries from polluting the training dataset.
            if self._app._execution_engine is not None and self._app._execution_engine.has_open_position(symbol):
                return

            # Strategy ensemble
            settings = self._app._settings
            if settings is None or self._app._strategy_ensemble is None:
                log.warning("strategy_loop.not_ready", symbol=symbol)
                return
            try:
                proposal = self._app._strategy_ensemble.evaluate_all(
                    feature_vector=vec,
                    current_price=current_price,
                    available_balance_usd=float(balance),
                    regime_ctx=regime_ctx,
                )
            except Exception as exc:
                log.warning("strategy_loop.ensemble_error", symbol=symbol, error=str(exc))
                return

            if proposal is None:
                return
            model_decision_meta: dict[str, Any] | None = None
            model_feature_names, model_feature_values = self._app._feature_values_for_side(vec, proposal.side.value)

            async def _record_signal(blocked: str | None = None) -> None:
                if self._app._trade_journal is not None:
                    await self._app._trade_journal.record_signal(
                        proposal=proposal,
                        feature_vector=vec,
                        regime_context=regime_ctx,
                        model_decision=model_decision_meta,
                        blocked_reason=blocked,
                    )

            stats_ready, stats_block_reason = self._app._expectancy_stats_ready()
            if not stats_ready:
                reason = stats_block_reason or "expectancy_stats_not_ready"
                self._app._record_diag(reason)
                log.warning(
                    "strategy_loop.expectancy_stats_not_ready",
                    symbol=symbol,
                    reason=reason,
                    bucket_stats_refreshed_at=(
                        self._app._bucket_stats_refreshed_at.isoformat()
                        if self._app._bucket_stats_refreshed_at is not None
                        else None
                    ),
                )
                await _record_signal(reason)
                return

            # Regime-bucket gate: skip execution when this (regime, volatility,
            # UTC hour) bucket has a proven negative expectancy on our own signals.
            # The proposal is intentionally evaluated first so the blocked reason
            # is persisted in trade_signals and visible in Telegram diagnostics.
            if self._app._bucket_blocked(regime_ctx):
                self._app._record_diag("bucket_blocked")
                log.debug(
                    "strategy_loop.bucket_blocked",
                    symbol=proposal.symbol,
                    side=proposal.side.value,
                    strategy_id=proposal.strategy_id,
                )
                await _record_signal("bucket_blocked")
                return

            if self._app._strategy_blocked(proposal.strategy_id):
                self._app._record_diag("strategy_expectancy_blocked")
                log.info(
                    "strategy_loop.strategy_expectancy_blocked",
                    symbol=proposal.symbol,
                    side=proposal.side.value,
                    strategy_id=proposal.strategy_id,
                    stats=self._app._strategy_stats.get(proposal.strategy_id),
                )
                await _record_signal("strategy_expectancy_blocked")
                return
            if self._app._strategy_side_blocked(proposal.strategy_id, proposal.side.value):
                self._app._record_diag("strategy_side_expectancy_blocked")
                log.info(
                    "strategy_loop.strategy_side_expectancy_blocked",
                    symbol=proposal.symbol,
                    side=proposal.side.value,
                    strategy_id=proposal.strategy_id,
                    stats=self._app._strategy_side_stats.get((proposal.strategy_id, proposal.side.value)),
                )
                await _record_signal("strategy_side_expectancy_blocked")
                return
            side_confidence_floor = self._app._strategy_side_confidence_floor(
                proposal.strategy_id,
                proposal.side.value,
            )
            if side_confidence_floor is not None and float(proposal.confidence) < side_confidence_floor:
                self._app._record_diag("strategy_side_confidence_blocked")
                log.info(
                    "strategy_loop.strategy_side_confidence_blocked",
                    symbol=proposal.symbol,
                    side=proposal.side.value,
                    strategy_id=proposal.strategy_id,
                    confidence=round(float(proposal.confidence), 3),
                    required_confidence=round(side_confidence_floor, 3),
                    stats=self._app._strategy_side_stats.get((proposal.strategy_id, proposal.side.value)),
                )
                await _record_signal("strategy_side_confidence_blocked")
                return
            if self._app._strategy_regime_blocked(proposal.strategy_id, regime_ctx):
                self._app._record_diag("strategy_regime_expectancy_blocked")
                regime = (
                    regime_ctx.regime.value
                    if regime_ctx is not None and getattr(regime_ctx, "regime", None) is not None
                    else "UNKNOWN"
                )
                log.info(
                    "strategy_loop.strategy_regime_expectancy_blocked",
                    symbol=proposal.symbol,
                    side=proposal.side.value,
                    strategy_id=proposal.strategy_id,
                    regime=regime,
                    stats=self._app._strategy_regime_stats.get((proposal.strategy_id, regime)),
                )
                await _record_signal("strategy_regime_expectancy_blocked")
                return
            confidence_floor = self._app._strategy_regime_confidence_floor(proposal.strategy_id, regime_ctx)
            if confidence_floor is not None and float(proposal.confidence) < confidence_floor:
                self._app._record_diag("strategy_regime_confidence_blocked")
                regime = (
                    regime_ctx.regime.value
                    if regime_ctx is not None and getattr(regime_ctx, "regime", None) is not None
                    else "UNKNOWN"
                )
                log.info(
                    "strategy_loop.strategy_regime_confidence_blocked",
                    symbol=proposal.symbol,
                    side=proposal.side.value,
                    strategy_id=proposal.strategy_id,
                    regime=regime,
                    confidence=round(float(proposal.confidence), 3),
                    required_confidence=round(confidence_floor, 3),
                    stats=self._app._strategy_regime_stats.get((proposal.strategy_id, regime)),
                )
                await _record_signal("strategy_regime_confidence_blocked")
                return

            # Cooldown: suppress duplicate proposals for the same symbol within one candle
            # period. The strategy loop runs every ~10s but features update every ~60s, so
            # without this the same signal fires 5-6× per candle, flooding training data with
            # correlated duplicates and spamming execution with skipped proposals.
            now_ts = datetime.now(tz=UTC)
            last_sig = self._app._last_signal_at.get(symbol)
            if last_sig is not None and (now_ts - last_sig).total_seconds() < self._app._signal_cooldown_s:
                return
            self._app._last_signal_at[symbol] = now_ts

            self._app._record_diag("signals_emitted")

            if not self._app._initial_shadow_mode() and not self._app._trend_mtf_confirmed(
                proposal.symbol, proposal.side.value
            ):
                self._app._record_diag("trend_confirmation_blocked")
                log.info(
                    "strategy_loop.trend_confirmation_blocked",
                    symbol=proposal.symbol,
                    side=proposal.side.value,
                )
                await _record_signal("trend_confirmation_blocked")
                return

            if self._app._symbol_side_blocked(proposal.symbol, proposal.side.value):
                self._app._record_diag("symbol_side_blocked")
                log.info(
                    "strategy_loop.symbol_side_blocked",
                    symbol=proposal.symbol,
                    side=proposal.side.value,
                )
                await _record_signal("symbol_side_blocked")
                return

            # --- Hybrid ML mode: a compatible CHAMPION may take over the decision ---
            # The model scores P(net-positive outcome) for the proposed direction.
            # When the score clears the gate threshold the signal becomes a model
            # decision: confidence = model score, rationale = "ML model decision".
            # The side is kept — the directional_net label schema scores the
            # proposal's own direction, so a high score IS the model's directional view.
            if settings.MODEL_ENABLED and settings.MODEL_ALLOW_LIVE_DECISIONS and self._app._model_registry is not None:
                try:
                    if not self._app._model_side_allowed(proposal.side.value):
                        self._app._record_diag("ml_live_side_filtered")
                        log.info(
                            "ml_live.side_filtered",
                            symbol=proposal.symbol,
                            side=proposal.side.value,
                        )
                        ml_pred = None
                    else:
                        ml_pred = self._app._model_registry.score_live(model_feature_values, model_feature_names)
                    ml_threshold = self._app._model_gate_threshold(regime_ctx)
                    if (
                        ml_pred is not None
                        and ml_pred.score < ml_threshold
                        and settings.FALLBACK_TO_RULE_WHEN_MODEL_UNSURE
                    ):
                        # Model exists but is unsure — keep the rule-based proposal.
                        # Counted so /healthcheck can show how often the fallback fires.
                        self._app._record_diag("rule_fallback_signal")
                    if ml_pred is not None and ml_pred.score >= ml_threshold:
                        model_decision_meta = {
                            "model_version": ml_pred.model_version,
                            "score": ml_pred.score,
                            "threshold": ml_threshold,
                            "original_confidence": proposal.confidence,
                            "original_rationale": proposal.rationale,
                            "side": proposal.side.value,
                        }
                        proposal = proposal.model_copy(
                            update={
                                "confidence": min(1.0, max(0.0, ml_pred.score)),
                                "rationale": "ML model decision",
                            }
                        )
                        self._app._record_diag("ml_replacement")
                        if _ML_REPLACEMENT_COUNTER is not None:
                            _ML_REPLACEMENT_COUNTER.inc()
                        log.info(
                            "ml_live.decision_replaced",
                            symbol=proposal.symbol,
                            side=proposal.side.value,
                            model_version=ml_pred.model_version,
                            score=ml_pred.score,
                            threshold=ml_threshold,
                        )
                except Exception as _ml_live_exc:
                    log.debug("ml_live.replace_failed", symbol=symbol, error=str(_ml_live_exc))

            # Record feature snapshot for ML training (no lookahead — uses candle open_time)
            snapshot_id = ""
            if self._app._trade_journal is not None and self._app._trade_journal.is_enabled and vec.feature_names:
                try:
                    _schema_hash = hashlib.sha256(json.dumps(model_feature_names).encode()).hexdigest()[:16]
                    _candles = (
                        self._app._candle_store.confirmed(proposal.symbol, _WS_INTERVAL)
                        if self._app._candle_store
                        else []
                    )
                    _candle_open_time = _candles[-1].open_time if _candles else vec.timestamp
                    snapshot_id = await self._app._trade_journal.record_feature_snapshot(
                        symbol=proposal.symbol,
                        interval=_WS_INTERVAL,
                        candle_open_time=_candle_open_time,
                        feature_schema_hash=_schema_hash,
                        feature_names=model_feature_names,
                        feature_values=model_feature_values,
                    )
                except Exception as _snap_exc:
                    log.debug("strategy_loop.feature_snapshot_failed", error=str(_snap_exc))

            # ML shadow scoring — only records metadata, never influences trade decisions
            if self._app._trade_journal is not None and self._app._trade_journal.is_enabled and snapshot_id:
                try:
                    # Regime context in metadata feeds get_bucket_stats (idea: regime-
                    # bucketed expectancy gating) — keep keys stable.
                    await self._app._trade_journal.record_prediction_event(
                        symbol=proposal.symbol,
                        interval=_WS_INTERVAL,
                        model_version="RULE_BASELINE_V1",
                        score=proposal.confidence,
                        strategy_signal=proposal.side.value,
                        decision="SHADOW_BASELINE",
                        feature_snapshot_id=snapshot_id,
                        metadata={
                            "strategy_id": proposal.strategy_id,
                            "strategy_rationale": proposal.rationale or "",
                            "regime": (
                                regime_ctx.regime.value
                                if regime_ctx is not None and getattr(regime_ctx, "regime", None) is not None
                                else "UNKNOWN"
                            ),
                            "volatility": (
                                regime_ctx.volatility_level.value
                                if regime_ctx is not None and getattr(regime_ctx, "volatility_level", None) is not None
                                else "UNKNOWN"
                            ),
                        },
                    )
                except Exception as _baseline_exc:
                    log.debug(
                        "strategy_loop.baseline_prediction_failed",
                        symbol=proposal.symbol,
                        error=str(_baseline_exc),
                    )

            if settings.MODEL_SHADOW_SCORING_ENABLED and self._app._model_registry is not None and snapshot_id:
                # --- Challenger shadow scoring: observational only, never blocks ---
                try:
                    shadow_prediction = self._app._model_registry.score_shadow(
                        model_feature_values, model_feature_names
                    )
                    if shadow_prediction is not None:
                        threshold = self._app._model_gate_threshold(regime_ctx)
                        shadow_gate_decision = None
                        shadow_gate_reason = "shadow_gate_disabled"
                        regime_name = (
                            regime_ctx.regime.value
                            if regime_ctx is not None and getattr(regime_ctx, "regime", None) is not None
                            else "UNKNOWN"
                        )
                        volatility_name = (
                            regime_ctx.volatility_level.value
                            if regime_ctx is not None and getattr(regime_ctx, "volatility_level", None) is not None
                            else "UNKNOWN"
                        )
                        if settings.MODEL_SHADOW_GATE_ENABLED:
                            if not self._app._model_side_allowed(proposal.side.value):
                                shadow_gate_decision = "GATE_BLOCK"
                                shadow_gate_reason = "side_not_selected_by_model"
                            else:
                                shadow_gate_decision = (
                                    "GATE_PASS" if shadow_prediction.score >= threshold else "GATE_BLOCK"
                                )
                                shadow_gate_reason = (
                                    "score_meets_threshold"
                                    if shadow_gate_decision == "GATE_PASS"
                                    else "score_below_regime_threshold"
                                )
                        if self._app._trade_journal is not None and self._app._trade_journal.is_enabled:
                            await self._app._trade_journal.record_prediction_event(
                                symbol=proposal.symbol,
                                interval=_WS_INTERVAL,
                                model_version=shadow_prediction.model_version,
                                score=shadow_prediction.score,
                                strategy_signal=proposal.side.value,
                                decision=shadow_gate_decision,
                                feature_snapshot_id=snapshot_id,
                                metadata={
                                    "source": "shadow_challenger",
                                    "confidence": shadow_prediction.confidence,
                                    "gate_reason": shadow_gate_reason,
                                    "regime": regime_name,
                                    "score": shadow_prediction.score,
                                    "threshold": threshold,
                                    "volatility": volatility_name,
                                },
                            )
                        # Challenger NEVER blocks a live trade — observational only
                    else:
                        log.debug("ml_shadow.no_challenger", symbol=proposal.symbol)
                except Exception as _ml_exc:
                    log.debug(
                        "ml_shadow.scoring_failed",
                        symbol=proposal.symbol,
                        error=str(_ml_exc),
                    )

            # --- Champion Canary gate: independent of shadow scoring ---
            # score_live() returns None when no compatible directional_net Champion exists.
            # In active execution, an enabled gate must fail closed.
            if settings.MODEL_GATE_CANARY_ENABLED and self._app._model_registry is not None:
                if not self._app._initial_shadow_mode() and not snapshot_id:
                    self._app._record_diag("model_gate_canary_blocked")
                    log.warning(
                        "ml_canary.snapshot_missing_fail_closed",
                        symbol=proposal.symbol,
                    )
                    await _record_signal("feature_snapshot_missing")
                    return
                try:
                    live_prediction = self._app._model_registry.score_live(model_feature_values, model_feature_names)
                    if live_prediction is not None:
                        canary_threshold = self._app._model_gate_threshold(regime_ctx)
                        canary_side_allowed = self._app._model_side_allowed(proposal.side.value)
                        canary_gate_decision = (
                            "GATE_PASS"
                            if canary_side_allowed and live_prediction.score >= canary_threshold
                            else "GATE_BLOCK"
                        )
                        canary_blocked, canary_reason = self._app._model_gate_canary_blocks(
                            canary_gate_decision,
                            canary_threshold,
                            live_prediction.score,
                        )
                        if not canary_side_allowed:
                            canary_blocked = True
                            canary_reason = "side_not_selected_by_model"
                        if snapshot_id and self._app._trade_journal is not None and self._app._trade_journal.is_enabled:
                            await self._app._trade_journal.record_prediction_event(
                                symbol=proposal.symbol,
                                interval=_WS_INTERVAL,
                                model_version=live_prediction.model_version,
                                score=live_prediction.score,
                                strategy_signal=proposal.side.value,
                                decision=canary_gate_decision,
                                feature_snapshot_id=snapshot_id,
                                metadata={
                                    "source": "champion_canary",
                                    "canary_blocked": canary_blocked,
                                    "canary_reason": canary_reason,
                                    "confidence": live_prediction.confidence,
                                    "gate_reason": canary_reason if not canary_side_allowed else "canary_gate",
                                    "threshold": canary_threshold,
                                },
                            )
                        if canary_blocked:
                            self._app._record_diag("model_gate_canary_blocked")
                            log.info(
                                "model_gate.canary_blocked",
                                symbol=proposal.symbol,
                                model_version=live_prediction.model_version,
                                score=live_prediction.score,
                                threshold=canary_threshold,
                                reason=canary_reason,
                            )
                            await _record_signal("model_gate_canary_blocked")
                            return
                    else:
                        log.warning(
                            "ml_canary.no_compatible_champion",
                            symbol=proposal.symbol,
                            active_execution=not self._app._initial_shadow_mode(),
                        )
                        if not self._app._initial_shadow_mode():
                            self._app._record_diag("model_gate_canary_blocked")
                            await _record_signal("model_gate_no_compatible_champion")
                            return
                except Exception as _canary_exc:
                    log.warning(
                        "ml_canary.scoring_failed",
                        symbol=proposal.symbol,
                        error=str(_canary_exc),
                    )
                    if not self._app._initial_shadow_mode():
                        self._app._record_diag("model_gate_canary_blocked")
                        await _record_signal("model_gate_scoring_failed")
                        return

            # Skip execution if operator paused trading
            if self._app._trading_paused:
                log.debug("strategy_loop.paused", symbol=symbol)
                await _record_signal("trading_paused")
                return

            # DB availability guard for CANARY_LIVE / LIVE
            if settings.TRADING_MODE in (
                TradingMode.CANARY_LIVE,
                TradingMode.LIVE,
            ):
                if settings.TRADE_JOURNAL_REQUIRED_FOR_ACTIVE:
                    if self._app._trade_journal is None or not self._app._trade_journal.is_enabled:
                        log.warning(
                            "strategy_loop.blocked_no_journal",
                            symbol=symbol,
                            mode=settings.TRADING_MODE,
                        )
                        await _record_signal("no_trade_journal")
                        return
                if settings.DURABLE_ORDER_STATE_REQUIRED_FOR_ACTIVE:
                    if self._app._trade_journal is None or not self._app._trade_journal.durable_state_healthy:
                        log.warning(
                            "strategy_loop.blocked_durable_store_unhealthy",
                            symbol=symbol,
                            mode=settings.TRADING_MODE,
                            write_health=(
                                self._app._trade_journal.write_health() if self._app._trade_journal is not None else {}
                            ),
                        )
                        await _record_signal("durable_store_unhealthy")
                        return

            # ExecutionEngine: dedup/cooldown/risk → order (or shadow log)
            # Notification fires only when execution engine actually approves
            if self._app._execution_engine is None:
                return

            try:
                # Re-read balance right before submit rather than using the
                # value captured once at the top of this cycle: symbols are
                # evaluated concurrently via gather(), and submit() itself is
                # serialized by execution_engine's lock, so an earlier
                # symbol's fill in this same cycle can leave the captured
                # balance stale by the time a later symbol reaches here.
                fresh_balance = self._app._cached_balance
                decision = await self._app._execution_engine.submit(
                    proposal=proposal,
                    capital=fresh_balance,
                    available_balance=fresh_balance,
                    feature_vector=vec,
                    regime_context=regime_ctx,
                )
            except Exception as exc:
                log.warning("strategy_loop.execution_error", symbol=symbol, error=str(exc))
                await _record_signal("execution_error")
                return

            from trader.domain.enums import RiskDecisionStatus

            if decision is None:
                pre_risk_reason = None
                if hasattr(self._app._execution_engine, "consume_last_pre_risk_rejection_reason"):
                    pre_risk_reason = self._app._execution_engine.consume_last_pre_risk_rejection_reason()
                await _record_signal(str(pre_risk_reason or "no_decision"))
                return
            if decision.status == RiskDecisionStatus.REJECTED:
                self._app._record_diag("risk_rejected")
                # Track specific rejection reasons
                for rule in decision.triggered_rules or []:
                    self._app._record_diag(f"risk_rule:{rule}")
                    if rule == "post_multiplier_min_notional_rejected":
                        self._app._record_diag("post_multiplier_min_notional_rejected")
                    if rule in (
                        "exposure_cap",
                        "exposure_cap_full",
                        "exposure_cap_post_bump",
                        "exposure_reservation",
                    ):
                        self._app._record_diag("risk_exposure_rejected")
                    if rule in ("sizer_rejected", "post_multiplier_zero"):
                        reason = (decision.reason or "").lower()
                        if "balance" in reason or "capital" in reason:
                            self._app._record_diag("risk_balance_rejected")
                        elif "spread" in reason or "atr" in reason or "stop distance" in reason:
                            self._app._record_diag("risk_market_filter_rejected")
                        elif "min_notional" in reason or "notional" in reason:
                            self._app._record_diag("post_multiplier_min_notional_rejected")
                        else:
                            self._app._record_diag("risk_sizer_rejected")
                rejected_reason = "risk_rejected"
                if decision.triggered_rules:
                    rejected_reason = f"risk_rejected:{decision.triggered_rules[0]}"
                await _record_signal(rejected_reason)
            if decision.status not in (
                RiskDecisionStatus.APPROVED,
                RiskDecisionStatus.RESIZED,
            ):
                return

            await _record_signal()

            # Trade approved — notify Telegram once and log to signal deque
            is_shadow = self._app._initial_shadow_mode()
            regime_str = regime_ctx.regime.value if regime_ctx is not None else "UNKNOWN"
            from trader.telegram_bot import SignalEntry

            entry = SignalEntry(
                timestamp=datetime.now(tz=UTC),
                symbol=proposal.symbol,
                side=proposal.side.value,
                confidence=proposal.confidence,
                regime=regime_str,
                rationale=proposal.rationale or "",
                shadow=is_shadow,
            )
            self._app._signal_log.append(entry)
            if self._app._telegram_bot is not None:
                try:
                    await self._app._telegram_bot.notify_signal(entry)
                except Exception as exc:
                    log.warning("telegram.notify_signal_failed", error=str(exc))

            # Track shadow position for TP/SL simulation
            if is_shadow and proposal.stop_loss and proposal.take_profit:
                engine_position = (
                    self._app._execution_engine._open_positions.get(symbol) or {}
                    if self._app._execution_engine is not None
                    else {}
                )
                entry_price = engine_position.get("entry_price") or proposal.entry_price or Decimal(str(current_price))
                _shadow_positions[symbol] = {
                    "side": proposal.side.value,
                    "entry": float(entry_price),
                    "tp": float(proposal.take_profit),
                    "sl": float(proposal.stop_loss),
                    "spread_bps": proposal.spread_bps,
                    "opened_at": datetime.now(tz=UTC),
                }

        async def strategy_loop() -> None:
            nonlocal _balance_tick, _effective_blocked_symbols

            while not self._app._shutdown_event.is_set():
                cycle_start = time.monotonic()
                self._app._last_strategy_loop_at = datetime.now(tz=UTC)
                # Refresh balance every N iterations
                _balance_tick += 1
                refresh_every = max(1, int(_BALANCE_REFRESH_INTERVAL / _STRATEGY_LOOP_INTERVAL))
                try:
                    if _balance_tick % refresh_every == 0:
                        await self._app._refresh_balance()
                        await self._app._refresh_closed_pnl_memory()
                    await self._app._sync_execution_positions()
                    await self._app._manage_open_positions()
                    self._app._check_zero_trading()
                except Exception:
                    log.exception("strategy_loop.cycle_maintenance_error")

                balance = self._app._cached_balance
                capital = balance

                # Feature pipeline runs on full active_symbols universe (set at startup)
                active_symbols = (
                    self._app._screener.active_symbols if self._app._screener is not None else list(_SYMBOLS)
                )
                _effective_blocked_symbols = (
                    set()
                    if self._app._initial_shadow_mode()
                    else self._app._effective_performance_blocks(active_symbols)
                )

                # Strategy evaluation uses execution_candidates only (Starter-optimized subset)
                exec_symbols = (
                    self._app._screener.execution_candidates if self._app._screener is not None else list(_SYMBOLS)
                )

                results = await asyncio.gather(
                    *[process_symbol(symbol, balance, capital) for symbol in exec_symbols],
                    return_exceptions=True,
                )
                for symbol, result in zip(exec_symbols, results, strict=False):
                    if isinstance(result, Exception):
                        log.warning(
                            "strategy_loop.symbol_task_failed",
                            symbol=symbol,
                            error=str(result),
                            error_type=type(result).__name__,
                        )

                # Measure processing time only — the deliberate sleep below is not overload.
                self._app._last_strategy_cycle_ms = (time.monotonic() - cycle_start) * 1000.0

                try:
                    await asyncio.wait_for(
                        self._app._shutdown_event.wait(),
                        timeout=_STRATEGY_LOOP_INTERVAL,
                    )
                except TimeoutError:
                    pass

        task = asyncio.create_task(strategy_loop(), name="strategy-loop")
        self._app._background_tasks.append(task)
        shadow = self._app._initial_shadow_mode()
        log.info(
            "strategy_loop.started",
            shadow_mode=shadow,
            initial_capital=str(initial_capital),
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
