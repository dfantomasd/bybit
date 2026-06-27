"""Execution runtime: risk stack, private WS, position management."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal
from typing import Any

from trader.domain.enums import TradingMode
from trader.modules.base import AppBoundModule
from trader.monitoring.logging import get_logger
from trader.runtime.constants import _JOURNAL_FALLBACK_UUID

log = get_logger(__name__)


class ExecutionRuntimeModule(AppBoundModule):
    name = "execution"

    async def init_risk_manager(self, initial_capital: Decimal) -> None:
        """Initialise RiskManager and all its dependencies."""
        from trader.risk.circuit_breakers import CircuitBreakerManager
        from trader.risk.drawdown import DrawdownTracker
        from trader.risk.exposure import ExposureTracker
        from trader.risk.kelly_adapter import KellyAdapter
        from trader.risk.kill_switch import KillSwitch
        from trader.risk.manager import RiskManager
        from trader.risk.profiles import get_risk_limits

        assert self._app._settings is not None
        profile = self._app._settings.RISK_PROFILE
        limits = get_risk_limits(profile)

        drawdown = DrawdownTracker(initial_equity=initial_capital)
        self._app._exposure_tracker = ExposureTracker(
            total_capital=initial_capital,
            risk_limits=limits,
        )
        breakers = CircuitBreakerManager(risk_limits=limits)
        kill_switch = KillSwitch()

        # Initialize unified ML system with all 5 models
        try:
            from trader.ml.entry_exit_optimizer_enhanced import EntryExitOptimizerEnhanced
            from trader.ml.kelly_predictor import MLKellyPredictor
            from trader.ml.regime_predictor_enhanced import RegimePredictorEnhanced
            from trader.ml.signal_fusion_enhanced import SignalFusionEnhanced
            from trader.ml.spread_predictor_enhanced import SpreadPredictorEnhanced
            from trader.ml.stoploss_optimizer_enhanced import StopLossOptimizerEnhanced
            from trader.ml.unified_controller import UnifiedMLController

            # Initialize each model
            kelly_predictor = MLKellyPredictor()
            regime_predictor = RegimePredictorEnhanced()
            signal_fusion = SignalFusionEnhanced()
            spread_predictor = SpreadPredictorEnhanced()
            stoploss_optimizer = StopLossOptimizerEnhanced()
            entry_exit_optimizer = EntryExitOptimizerEnhanced()

            # Create unified controller
            ml_controller = UnifiedMLController(
                kelly_predictor=kelly_predictor,
                regime_predictor=regime_predictor,
                signal_fusion=signal_fusion,
                spread_predictor=spread_predictor,
                stoploss_optimizer=stoploss_optimizer,
                entry_exit_optimizer=entry_exit_optimizer,
                model_dir="/tmp/ml_models",  # noqa: S108
                auto_save=True,
            )

            # Try to load previously trained models
            await ml_controller.load_models()

            # Store in app
            self._app._ml_controller = ml_controller
            self._app._kelly_predictor = kelly_predictor
            self._app._regime_predictor = regime_predictor
            self._app._signal_fusion = signal_fusion
            self._app._spread_predictor = spread_predictor
            self._app._stoploss_optimizer = stoploss_optimizer
            self._app._entry_exit_optimizer = entry_exit_optimizer

            kelly_adapter = KellyAdapter(ml_kelly_predictor=kelly_predictor)

            log.info("ml_unified_controller.initialized")
        except Exception as e:
            log.warning(f"ml_unified_controller.import_failed: {e}, using fallback")
            kelly_adapter = KellyAdapter()
            self._app._ml_controller = None
            self._app._kelly_predictor = None

        self._app._risk_manager = RiskManager(
            risk_profile=profile,
            drawdown_tracker=drawdown,
            exposure_tracker=self._app._exposure_tracker,
            circuit_breaker_manager=breakers,
            kill_switch=kill_switch,
            require_liquidity_for_sizing=(
                self._app._settings.LIVE_REQUIRE_LIQUIDITY_FOR_SIZING
                and self._app._settings.TRADING_MODE in (TradingMode.LIVE, TradingMode.CANARY_LIVE)
            ),
            max_correlated_positions=int(self._app._settings.MAX_CORRELATED_POSITIONS),
            kelly_adapter=kelly_adapter,
            trade_journal=self._app._trade_journal,
            ml_controller=self._app._ml_controller,
        )
        self._app._kill_switch = kill_switch
        log.info(
            "risk_manager.initialized",
            profile=profile.value,
            initial_capital=str(initial_capital),
        )

    async def refresh_balance(self) -> Decimal:
        """Fetch current available balance from exchange; fall back to cached value.

        Also updates ExposureTracker capital when balance changes.
        """
        assert self._app._settings is not None
        has_key = bool(self._app._settings.BYBIT_API_KEY.get_secret_value())
        if not has_key or self._app._bybit_adapter is None:
            return self._app._cached_balance

        try:
            balance = await self._app._bybit_adapter.get_balance()
            # Use available; if zero (e.g. all collateralised) fall back to wallet
            available = balance.available_balance
            if available <= Decimal("0") and balance.wallet_balance > Decimal("0"):
                available = balance.wallet_balance
            if available > Decimal("0"):
                if self._app._balance_refreshed_at is not None and balance.updated_at < self._app._balance_refreshed_at:
                    log.debug(
                        "balance.refresh_ignored_stale",
                        available_usd=str(available),
                        updated_at=balance.updated_at.isoformat(),
                        current_updated_at=self._app._balance_refreshed_at.isoformat(),
                    )
                    return self._app._cached_balance
                old_capital = self._app._cached_balance
                self._app._cached_balance = available
                self._app._balance_refreshed_at = balance.updated_at
                log.info(
                    "balance.refreshed",
                    available_usd=str(available),
                    wallet_usd=str(balance.wallet_balance),
                    updated_at=self._app._balance_refreshed_at.isoformat(),
                )
                # P1: Update ExposureTracker capital so exposure_pct is always current
                if self._app._exposure_tracker is not None and available != old_capital:
                    self._app._exposure_tracker.update_capital(available, updated_at=self._app._balance_refreshed_at)
                    log.debug(
                        "exposure.capital_updated",
                        old_capital=old_capital,
                        new_capital=available,
                        total_exposure_pct=str(self._app._exposure_tracker.total_exposure_pct),
                    )
            return self._app._cached_balance
        except Exception as exc:
            log.warning("balance.refresh_failed", error=str(exc))
            return self._app._cached_balance

    async def init_execution_engine(self) -> None:
        """Initialise ExecutionEngine after RiskManager is ready."""
        from trader.execution.engine import ExecutionEngine

        assert self._app._settings is not None
        assert self._app._risk_manager is not None
        assert self._app._exposure_tracker is not None
        assert self._app._bybit_adapter is not None
        from trader.config import get_risk_profile_config

        profile_cfg = get_risk_profile_config(self._app._settings.RISK_PROFILE)

        shadow = self._app._initial_shadow_mode()
        is_canary = self._app._settings.TRADING_MODE == TradingMode.CANARY_LIVE
        max_open_positions = self._app._settings.MAX_POSITIONS
        if shadow and self._app._settings.SHADOW_PROBE_ENABLED:
            if self._app._settings.SHADOW_PROBE_RESEARCH_PROFILE_V2:
                max_open_positions = 4
            else:
                max_open_positions = self._app._settings.SHADOW_PROBE_MAX_OPEN_POSITIONS
        self._app._execution_engine = ExecutionEngine(
            adapter=self._app._bybit_adapter,
            risk_manager=self._app._risk_manager,
            exposure_tracker=self._app._exposure_tracker,
            shadow_mode=shadow,
            shadow_apply_net_edge_gate=self._app._scalp_strict_shadow(),
            cooldown_s=profile_cfg.cooldown_seconds,
            category=self._app._settings.DEFAULT_MARKET_CATEGORY,
            trade_journal=self._app._trade_journal,
            min_notional_safety_buffer_pct=self._app._settings.MIN_NOTIONAL_SAFETY_BUFFER_PCT,
            micro_account_balance_usd=self._app._settings.MICRO_ACCOUNT_BALANCE_USD,
            micro_account_min_notional_buffer_pct=self._app._settings.MICRO_ACCOUNT_MIN_NOTIONAL_BUFFER_PCT,
            max_new_entries_per_minute=self._app._settings.MAX_NEW_ENTRIES_PER_MINUTE,
            max_concurrent_pending_entries=self._app._settings.MAX_CONCURRENT_PENDING_ENTRIES,
            max_queue_utilization_pct=float(self._app._settings.MAX_QUEUE_UTILIZATION_PCT),
            max_same_side_positions=self._app._settings.MAX_SAME_SIDE_POSITIONS,
            max_open_positions=max_open_positions,
            startup_warmup_seconds=self._app._settings.STARTUP_WARMUP_SECONDS,
            is_canary=is_canary,
            fee_provider=self._app._fee_provider,
            max_spread_bps=self._app._settings.SCREENER_MAX_SPREAD_BPS,
            expected_slippage_pct=self._app._settings.EXPECTED_SLIPPAGE_PCT,
            funding_buffer_pct=self._app._settings.FUNDING_BUFFER_PCT,
            min_net_edge_pct=self._app._settings.MIN_EXPECTED_NET_EDGE_PCT,
            net_edge_safety_margin_pct=self._app._settings.NET_EDGE_SAFETY_MARGIN_PCT,
            entry_order_mode=self._app._settings.ENTRY_ORDER_MODE,
            maker_timeout_s=self._app._settings.MAKER_TIMEOUT_SECONDS,
            maker_ttl_s=self._app._settings.MAKER_TTL_SECONDS,
            maker_allow_escalation=self._app._settings.MAKER_ALLOW_ESCALATION,
            # Late-bound: the tracker is created when the public WS starts
            imbalance_provider=lambda s: (
                self._app._orderbook_tracker.latest_imbalance(s) if self._app._orderbook_tracker is not None else None
            ),
            live_armed=self._app._settings.LIVE_ARMED,
            shadow_min_atr_multiple=(
                self._app._settings.SHADOW_MIN_ATR_MULTIPLE if shadow and not self._app._scalp_strict_shadow() else None
            ),
        )

        # Initialize ML integration for ExecutionEngine
        if self._app._ml_controller is not None:
            try:
                from trader.ml.execution_integration import ExecutionMLIntegrator

                ml_integrator = ExecutionMLIntegrator(ml_controller=self._app._ml_controller)
                self._app._ml_integrator = ml_integrator
                self._app._execution_engine._ml_integrator = ml_integrator
                log.info("ml_integration.enabled_in_execution_engine")
            except Exception as e:
                log.warning(f"ml_integration.setup_failed: {e}")
                self._app._ml_integrator = None
        else:
            self._app._ml_integrator = None

        # P0.2: Restore unresolved pending entries from durable storage before any new entries.
        await self._app._restore_execution_pending_entries()

        # Sync open positions from exchange so we don't double-enter on restart
        await self._app._execution_engine.sync_positions()

        # Reconcile restored pending entries against live exchange state
        try:
            await self._app._execution_engine.reconcile_restored_pending_entries()
        except Exception as exc:
            log.warning("execution_engine.reconcile_failed", error=str(exc))

        log.info("execution_engine.initialized", shadow_mode=shadow, is_canary=is_canary)

    async def start_private_ws(self) -> None:
        """Start Bybit private WebSocket for real-time order/position/balance events."""
        from trader.exchange.bybit_ws_private import BybitPrivateWebSocket
        from trader.exchange.endpoint_selector import EndpointSelector

        assert self._app._settings is not None
        api_key = self._app._settings.BYBIT_API_KEY.get_secret_value()
        api_secret = self._app._settings.BYBIT_API_SECRET.get_secret_value()

        if not api_key or not api_secret:
            log.info("private_ws.skipped", reason="no_api_credentials_configured")
            return

        selector = EndpointSelector(
            self._app._settings.BYBIT_REGION,
            self._app._settings.BYBIT_USE_TESTNET,
        )

        private_event_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1000)

        self._app._ws_private = BybitPrivateWebSocket(
            endpoint=selector.ws_private_base,
            api_key=api_key,
            api_secret=api_secret,
            event_queue=private_event_queue,
        )

        async def consume_private_events() -> None:
            from trader.domain.enums import OrderStatus
            from trader.domain.events import (
                BalanceUpdateEvent,
                ExecutionUpdateEvent,
                OrderUpdateEvent,
                PositionUpdateEvent,
            )

            _terminal_order_states = {
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
                OrderStatus.EXPIRED,
            }

            seen_exec_ids: set[str] = set()
            # In-process cache of released order_link_ids: avoids a DB roundtrip per
            # duplicate terminal event. The authoritative record is order_pending_state
            # (resolved_at) which survives restarts.
            _released_cache: set[str] = set()

            async def _release_pending(order_link_id: str, symbol: str) -> None:
                """Release a pending entry slot exactly once and persist the resolution."""
                if order_link_id in _released_cache:
                    return
                if (
                    self._app._trade_journal is not None
                    and self._app._trade_journal.is_enabled
                    and await self._app._trade_journal.is_order_resolved(order_link_id)
                ):
                    _released_cache.add(order_link_id)
                    return
                if self._app._execution_engine is not None:
                    self._app._execution_engine.mark_entry_resolved(order_link_id)
                _released_cache.add(order_link_id)
                if self._app._trade_journal is not None and self._app._trade_journal.is_enabled:
                    try:
                        await self._app._trade_journal.mark_order_resolved(order_link_id, symbol)
                    except Exception as _res_exc:
                        log.debug(
                            "private_ws.mark_order_resolved_failed",
                            order_link_id=order_link_id,
                            error=str(_res_exc),
                        )

            while not self._app._shutdown_event.is_set():
                try:
                    event = await asyncio.wait_for(private_event_queue.get(), timeout=1.0)
                    if isinstance(event, BalanceUpdateEvent) and event.available_balance > Decimal("0"):
                        if (
                            self._app._balance_refreshed_at is not None
                            and event.timestamp < self._app._balance_refreshed_at
                        ):
                            log.debug(
                                "private_ws.balance_update_ignored_stale",
                                available=str(event.available_balance),
                                updated_at=event.timestamp.isoformat(),
                                current_updated_at=self._app._balance_refreshed_at.isoformat(),
                            )
                            continue
                        old_capital = self._app._cached_balance
                        self._app._cached_balance = event.available_balance
                        self._app._balance_refreshed_at = event.timestamp
                        # P1: Update ExposureTracker capital from WS balance push
                        if self._app._exposure_tracker is not None and event.available_balance != old_capital:
                            self._app._exposure_tracker.update_capital(
                                event.available_balance,
                                updated_at=self._app._balance_refreshed_at,
                            )
                            log.debug(
                                "exposure.capital_updated_ws",
                                old_capital=old_capital,
                                new_capital=event.available_balance,
                                total_exposure_pct=str(self._app._exposure_tracker.total_exposure_pct),
                            )
                        log.debug(
                            "private_ws.balance_update",
                            available=str(event.available_balance),
                            updated_at=self._app._balance_refreshed_at.isoformat(),
                        )
                    elif isinstance(event, OrderUpdateEvent):
                        # Wire OrderUpdateEvent → both idempotency AND durable state via adapter
                        # P0: Never use exchange orderId directly as pending-ID.
                        # Use order_link_id if present, otherwise reverse-lookup via exchange_order_id.
                        order_link_id = event.order_link_id
                        exchange_order_id = event.order_id
                        if order_link_id is None and exchange_order_id:
                            if self._app._trade_journal is not None:
                                order_link_id = await self._app._trade_journal.find_order_link_id_by_exchange_order_id(
                                    exchange_order_id
                                )
                            # If lookup fails, we still process the event but can't tie it to a pending slot
                        if order_link_id is None:
                            # Generate a fallback ID for logging only — never used for pending slot
                            order_link_id = f"unknown:{exchange_order_id or 'no_exchange_id'}"

                        order_status = event.status  # OrderUpdateEvent.status is the correct field
                        log.info(
                            "private_ws.order_update",
                            order_link_id=order_link_id,
                            exchange_order_id=exchange_order_id,
                            symbol=event.symbol,
                            status=order_status.value if order_status else "unknown",
                            side=event.side.value if event.side else "unknown",
                        )
                        # Update both idempotency and durable state atomically via adapter
                        if self._app._bybit_adapter is not None:
                            try:
                                is_terminal = await self._app._bybit_adapter.handle_order_update(event)
                            except Exception as _h_exc:
                                log.debug(
                                    "private_ws.handle_order_update_failed",
                                    error=str(_h_exc),
                                )
                                is_terminal = order_status in _terminal_order_states
                        else:
                            # Fallback: write directly to journal when adapter unavailable
                            if self._app._trade_journal is not None:
                                try:
                                    await self._app._trade_journal.record_order_update_event(
                                        order_link_id=order_link_id,
                                        exchange_order_id=exchange_order_id,
                                        symbol=event.symbol,
                                        side=event.side.value if event.side else "unknown",
                                        qty=event.qty if hasattr(event, "qty") and event.qty else Decimal("0"),
                                        state=order_status.value if order_status else "UNKNOWN",
                                    )
                                except Exception as _j_exc:
                                    log.debug(
                                        "private_ws.order_update_journal_failed",
                                        error=str(_j_exc),
                                    )
                            is_terminal = order_status in _terminal_order_states

                        # Release pending entry slot on terminal — exactly once per order.
                        # Resolution is persisted to order_pending_state so a restart
                        # never re-blocks the slot. Skip "unknown:" prefix IDs — they
                        # are fallback logging IDs, not real pending slots.
                        if is_terminal and order_link_id:
                            if not order_link_id.startswith("unknown:"):
                                await _release_pending(order_link_id, event.symbol)
                            else:
                                _released_cache.add(order_link_id)
                        # Trigger position sync on fill
                        if order_status == OrderStatus.FILLED and self._app._execution_engine is not None:
                            try:
                                await self._app._execution_engine.sync_positions()
                            except Exception as _sync_exc:
                                log.debug(
                                    "private_ws.order_fill_sync_failed",
                                    error=str(_sync_exc),
                                )
                    elif isinstance(event, ExecutionUpdateEvent):
                        if event.exec_id in seen_exec_ids:
                            continue
                        # Bound the dedup set: duplicates after a reset are harmless
                        # (downstream journal writes are idempotent on exec_id).
                        if len(seen_exec_ids) >= 10_000:
                            seen_exec_ids.clear()
                        seen_exec_ids.add(event.exec_id)
                        # P0: Reverse lookup order_link_id if not present
                        order_link_id = event.order_link_id
                        exchange_order_id = event.order_id
                        if order_link_id is None and exchange_order_id:
                            if self._app._trade_journal is not None:
                                order_link_id = await self._app._trade_journal.find_order_link_id_by_exchange_order_id(
                                    exchange_order_id
                                )

                        log.info(
                            "private_ws.execution_fill",
                            exec_id=event.exec_id,
                            symbol=event.symbol,
                            exec_price=str(event.exec_price),
                            exec_qty=str(event.exec_qty),
                            side=event.side.value,
                            order_link_id=order_link_id,
                            exchange_order_id=exchange_order_id,
                        )
                        if self._app._trade_journal is not None:
                            try:
                                # P0.5: persist to execution_events (nullable proposal/decision)
                                await self._app._trade_journal.record_execution_event(
                                    exec_id=event.exec_id,
                                    order_link_id=order_link_id
                                    if order_link_id and not order_link_id.startswith("unknown:")
                                    else None,
                                    exchange_order_id=exchange_order_id,
                                    symbol=event.symbol,
                                    side=event.side.value,
                                    exec_price=event.exec_price,
                                    exec_qty=event.exec_qty,
                                    exec_fee=event.exec_fee if event.exec_fee else None,
                                    exec_value=event.exec_value if event.exec_value else None,
                                    is_maker=event.is_maker if hasattr(event, "is_maker") else None,
                                    closed_size=event.closed_size if event.closed_size else None,
                                )
                                await self._app._trade_journal.record_order_event(
                                    order_link_id=order_link_id
                                    if order_link_id and not order_link_id.startswith("unknown:")
                                    else event.exec_id,
                                    proposal_id=_JOURNAL_FALLBACK_UUID,
                                    decision_id=_JOURNAL_FALLBACK_UUID,
                                    symbol=event.symbol,
                                    side=event.side.value,
                                    qty=event.exec_qty,
                                    status="FILLED",
                                    exchange_order_id=exchange_order_id,
                                )
                            except Exception as _journal_exc:
                                log.warning(
                                    "private_ws.execution_journal_failed",
                                    exec_id=event.exec_id,
                                    error=str(_journal_exc),
                                )

                        # P0.3: Release pending entry slot for this order_link_id only.
                        # Use the resolved order_link_id (after reverse lookup), skip "unknown:" prefixes.
                        # Resolution is persisted to order_pending_state for restart safety.
                        if order_link_id and not order_link_id.startswith("unknown:"):
                            await _release_pending(order_link_id, event.symbol)

                        if self._app._execution_engine is not None:
                            try:
                                await self._app._execution_engine.sync_positions()
                            except Exception as _sync_exc:
                                log.warning(
                                    "private_ws.execution_sync_failed",
                                    exec_id=event.exec_id,
                                    error=str(_sync_exc),
                                )

                        if self._app._bybit_adapter is not None and not self._app._initial_shadow_mode():
                            try:
                                await self._app._bybit_adapter.reconcile()
                            except Exception as _rec_exc:
                                log.debug(
                                    "private_ws.execution_reconcile_failed",
                                    exec_id=event.exec_id,
                                    error=str(_rec_exc),
                                )
                    elif isinstance(event, PositionUpdateEvent):
                        if self._app._execution_engine is not None:
                            try:
                                await self._app._execution_engine.apply_position_update(event)
                                self.cache_exchange_position_update(event)
                            except Exception as _pos_exc:
                                log.warning(
                                    "private_ws.position_update_failed",
                                    symbol=event.symbol,
                                    error=str(_pos_exc),
                                )
                except TimeoutError:
                    pass
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    log.warning("private_ws_consumer.error", error=str(exc))

        ws_task = asyncio.create_task(self._app._ws_private.start(), name="ws-private")
        consumer_task = asyncio.create_task(consume_private_events(), name="ws-private-consumer")
        self._app._background_tasks.extend([ws_task, consumer_task])
        log.info("private_ws.started", endpoint=selector.ws_private_base)

    async def run_risk_monitor(self) -> None:
        """Periodic risk monitor: update equity, check WS freshness, feed circuit breakers."""
        assert self._app._settings is not None
        interval = 15.0

        while not self._app._shutdown_event.is_set():
            try:
                # Refresh balance and update DrawdownTracker with current equity
                if (
                    self._app._bybit_adapter is not None
                    and self._app._risk_manager is not None
                    and bool(self._app._settings.BYBIT_API_KEY.get_secret_value())
                ):
                    try:
                        balance = await self._app._bybit_adapter.get_balance()
                        wallet = balance.wallet_balance
                        if wallet > Decimal("0"):
                            await self._app._risk_manager._drawdown.update(wallet)
                    except Exception as exc:
                        log.debug("risk_monitor.balance_update_failed", error=str(exc))

                # P1: Fetch daily realized PnL and feed to RiskManager for daily loss limit tracking
                if self._app._trade_journal is not None and self._app._risk_manager is not None:
                    try:
                        net_results = await self._app._trade_journal.get_daily_net_results()
                        net_pnl = Decimal(str(net_results.get("net_pnl_usd", 0)))
                        # Replace daily_pnl entirely (get_daily_net_results returns today's total)
                        # RiskManager.update_daily_pnl is additive, so we track the delta
                        old_daily_pnl = self._app._risk_manager.daily_pnl
                        delta = net_pnl - old_daily_pnl
                        if delta != Decimal("0"):
                            await self._app._risk_manager.update_daily_pnl(delta)
                            log.debug(
                                "risk_monitor.daily_pnl_synced",
                                old=old_daily_pnl,
                                new=net_pnl,
                                delta=delta,
                            )
                    except Exception as exc:
                        log.debug("risk_monitor.daily_pnl_sync_failed", error=str(exc))

                # P1: Evaluate circuit breakers
                if self._app._risk_manager is not None and self._app._risk_manager._breakers is not None:
                    breakers = self._app._risk_manager._breakers
                    # Daily loss limit
                    await breakers.check_daily_loss(
                        self._app._risk_manager.daily_pnl,
                        self._app._cached_balance,
                    )
                    # Max drawdown
                    await breakers.check_drawdown(self._app._risk_manager._drawdown.drawdown_pct)
                    # WebSocket staleness
                    if (
                        self._app._health_checker is not None
                        and self._app._health_checker._last_ws_message_at is not None
                    ):
                        age = (datetime.now(tz=UTC) - self._app._health_checker._last_ws_message_at).total_seconds()
                        await breakers.check_websocket_staleness(age)
                    # REST error rate (track from adapter if available)
                    if self._app._bybit_adapter is not None and hasattr(
                        self._app._bybit_adapter, "_rest_errors_last_minute"
                    ):
                        await breakers.check_rest_error_rate(self._app._bybit_adapter._rest_errors_last_minute)
                    # Feature quality
                    if self._app._feature_pipeline is not None and hasattr(
                        self._app._feature_pipeline, "quality_score"
                    ):
                        await breakers.check_feature_quality(self._app._feature_pipeline.quality_score)
                    # NTP drift
                    if self._app._bybit_adapter is not None and hasattr(self._app._bybit_adapter, "ntp_drift_seconds"):
                        await breakers.check_ntp_drift(self._app._bybit_adapter.ntp_drift_seconds)
                    # Auto-reset eligible breakers
                    await breakers.reset_all_auto()

                # Check WS freshness and alert if stale (legacy logging)
                if self._app._health_checker is not None and self._app._health_checker._last_ws_message_at is not None:
                    age = (datetime.now(tz=UTC) - self._app._health_checker._last_ws_message_at).total_seconds()
                    if age > 60.0:
                        log.warning("risk_monitor.ws_stale", age_s=age)
                        self._app._record_diag("ws_stale")
                    await self.maybe_recover_stale_ws(age)

            except Exception as exc:
                log.warning("risk_monitor.error", error=str(exc))

            try:
                await asyncio.wait_for(
                    self._app._shutdown_event.wait(),
                    timeout=interval,
                )
            except TimeoutError:
                pass

    async def maybe_recover_stale_ws(self, market_data_age_s: float) -> None:
        """Nudge the public WS to reconnect when market data stops flowing."""
        assert self._app._settings is not None
        threshold = float(self._app._settings.WS_MARKET_DATA_STALE_RECONNECT_SECONDS)
        if market_data_age_s < threshold or self._app._ws_public is None:
            return
        now = datetime.now(tz=UTC)
        if self._app._last_ws_recovery_at is not None:
            if (now - self._app._last_ws_recovery_at).total_seconds() < threshold:
                return
        self._app._last_ws_recovery_at = now
        log.warning(
            "ws_public.recovery_requested",
            market_data_age_s=round(market_data_age_s, 1),
            threshold_s=threshold,
        )
        try:
            await self._app._ws_public.force_reconnect()
        except Exception as exc:
            log.warning("ws_public.recovery_failed", error=str(exc))

    async def refresh_closed_pnl_memory(self) -> None:
        """Import recent Bybit closed PnL and update performance symbol blocks."""
        assert self._app._settings is not None
        if (
            self._app._trade_journal is None
            or not self._app._settings.PERFORMANCE_FILTER_ENABLED
            or self._app._bybit_adapter is None
            or not self._app._trade_journal.is_enabled
        ):
            return

        now = datetime.now(tz=UTC)
        if self._app._closed_pnl_refreshed_at is not None:
            elapsed = (now - self._app._closed_pnl_refreshed_at).total_seconds()
            if elapsed < self._app._settings.CLOSED_PNL_REFRESH_INTERVAL_SECONDS:
                return

        try:
            resp = await self._app._bybit_adapter._rest.get_closed_pnl(
                category=self._app._settings.DEFAULT_MARKET_CATEGORY,
                limit=100,
            )
            records = resp.get("result", {}).get("list", [])
            await self._app._trade_journal.record_closed_pnl_records(records)
            blocked = await self._app._trade_journal.get_blocked_symbols(
                min_closed_trades=self._app._settings.PERFORMANCE_MIN_CLOSED_TRADES,
                max_loss_usd=Decimal(str(self._app._settings.PERFORMANCE_MAX_SYMBOL_LOSS_USD)),
                lookback_days=self._app._settings.PERFORMANCE_LOOKBACK_DAYS,
            )
            if blocked != self._app._performance_blocked_symbols:
                log.info(
                    "performance_filter.updated",
                    blocked_symbols=sorted(blocked),
                    min_closed_trades=self._app._settings.PERFORMANCE_MIN_CLOSED_TRADES,
                    max_loss_usd=self._app._settings.PERFORMANCE_MAX_SYMBOL_LOSS_USD,
                    lookback_days=self._app._settings.PERFORMANCE_LOOKBACK_DAYS,
                )
            self._app._performance_blocked_symbols = blocked
            self._app._closed_pnl_refreshed_at = now

            # Trigger ML model retraining on closed trades
            if self._app._ml_controller is not None and self._app._ml_integrator is not None:
                try:
                    # Get recent trades for context
                    recent_trades = await self._app._trade_journal.get_recent_closed_trades(limit=20)
                    if recent_trades:
                        # Build trade outcomes for ML
                        for trade in recent_trades:
                            trade_outcome = {
                                "symbol": trade.get("symbol", ""),
                                "pnl_usd": trade.get("pnl_usdt", 0),
                                "pnl_bps": trade.get("net_bps", 0),
                                "entry_price": trade.get("entry", 0),
                                "exit_price": trade.get("exit", 0),
                                "side": trade.get("side", "LONG"),
                                "qty": trade.get("qty", 0),
                            }
                            # Record for ML training
                            await self._app._ml_integrator.record_trade_outcome(
                                trade_data=trade_outcome,
                                recent_trades=recent_trades,
                            )
                        log.debug(f"ml_training.triggered_on_closed_trades: {len(recent_trades)}")
                except Exception as ml_exc:
                    log.debug(f"ml_training.trigger_failed: {str(ml_exc)}")

        except Exception as exc:
            log.debug("performance_filter.refresh_failed", error=str(exc))

    async def manage_open_positions(self) -> None:
        """Move profitable positions to breakeven and enable exchange trailing stop."""
        assert self._app._settings is not None
        if (
            not self._app._settings.PROFIT_MANAGER_ENABLED
            or not self._app._settings.TRAILING_STOP_ENABLED
            or self._app._bybit_adapter is None
        ):
            return

        now = datetime.now(tz=UTC)
        if self._app._positions_managed_at is not None:
            elapsed = (now - self._app._positions_managed_at).total_seconds()
            if elapsed < self._app._settings.POSITION_MANAGEMENT_INTERVAL_SECONDS:
                return
        self._app._positions_managed_at = now

        positions = self.recent_exchange_positions()
        if positions is None:
            try:
                positions = await self._app._bybit_adapter.get_positions(self._app._settings.DEFAULT_MARKET_CATEGORY)
                self._app._cache_exchange_positions(positions)
            except Exception as exc:
                log.debug("profit_manager.positions_fetch_failed", error=str(exc))
                return

        # Prune stale trailing-stop keys for positions that are no longer open
        active_keys = {
            f"{p.symbol}:{p.side.value}:{p.size}:{p.entry_price}"
            for p in positions
            if p.size > Decimal("0") and p.entry_price > Decimal("0")
        }
        self._app._trailing_stop_keys &= active_keys

        for pos in positions:
            if pos.size <= Decimal("0") or pos.entry_price <= Decimal("0"):
                continue
            mark_price = pos.mark_price or pos.entry_price
            if mark_price <= Decimal("0"):
                continue

            pnl_pct = (
                (mark_price - pos.entry_price) / pos.entry_price * Decimal("100")
                if pos.side.value == "Buy"
                else (pos.entry_price - mark_price) / pos.entry_price * Decimal("100")
            )
            if pnl_pct < Decimal(str(self._app._settings.TRAILING_ACTIVATION_PCT)):
                continue

            position_key = f"{pos.symbol}:{pos.side.value}:{pos.size}:{pos.entry_price}"
            if position_key in self._app._trailing_stop_keys:
                continue

            try:
                info = (
                    await self._app._execution_engine.get_instrument_info(pos.symbol)
                    if self._app._execution_engine is not None
                    else await self._app._bybit_adapter.get_instrument_info(
                        self._app._settings.DEFAULT_MARKET_CATEGORY,
                        pos.symbol,
                    )
                )
                active_price = self.round_to_tick(
                    self.activation_price(pos.entry_price, pos.side.value),
                    info.tick_size,
                    round_up=pos.side.value == "Buy",
                )
                trailing_distance = self.round_to_tick(
                    mark_price * Decimal(str(self._app._settings.TRAILING_DISTANCE_PCT)) / Decimal("100"),
                    info.tick_size,
                    round_up=True,
                )
                fee_rates = None
                if self._app._fee_provider is not None:
                    try:
                        fee_rates = await self._app._fee_provider.get(pos.symbol)
                    except Exception as _fee_exc:
                        log.debug(
                            "profit_manager.fee_rate_failed",
                            symbol=pos.symbol,
                            error=str(_fee_exc),
                        )
                breakeven_stop = self.round_to_tick(
                    self.breakeven_stop(pos.entry_price, pos.side.value, fee_rates=fee_rates),
                    info.tick_size,
                    round_up=pos.side.value == "Sell",
                )
                if trailing_distance < info.tick_size:
                    trailing_distance = info.tick_size

                await self._app._bybit_adapter.set_trading_stop(
                    category=self._app._settings.DEFAULT_MARKET_CATEGORY,
                    symbol=pos.symbol,
                    stop_loss=str(breakeven_stop),
                    trailing_stop=str(trailing_distance),
                    active_price=str(active_price),
                    position_idx=0,
                    tpsl_mode="Full",
                )
                self._app._trailing_stop_keys.add(position_key)
                log.info(
                    "profit_manager.trailing_stop_set",
                    symbol=pos.symbol,
                    side=pos.side.value,
                    pnl_pct=float(round(pnl_pct, 4)),
                    stop_loss=str(breakeven_stop),
                    trailing_stop=str(trailing_distance),
                    active_price=str(active_price),
                )
            except Exception as exc:
                log.debug(
                    "profit_manager.trailing_stop_failed",
                    symbol=pos.symbol,
                    error=str(exc),
                )

    async def sync_execution_positions(self) -> None:
        """Keep local execution/risk state aligned with Bybit TP/SL closures."""
        assert self._app._settings is not None
        if self._app._execution_engine is None or self._app._bybit_adapter is None:
            return
        if self._app._execution_engine._shadow_mode:
            return

        now = datetime.now(tz=UTC)
        if self._app._positions_synced_at is not None:
            elapsed = (now - self._app._positions_synced_at).total_seconds()
            if elapsed < self._app._settings.POSITION_SYNC_INTERVAL_SECONDS:
                return
        positions = await self._app._execution_engine.sync_positions()
        if positions is not None:
            self._app._positions_synced_at = now
            self._app._cache_exchange_positions(positions)

    def cache_exchange_positions(self, positions: list[Any]) -> None:
        self._app._latest_exchange_positions = positions
        self._app._latest_exchange_positions_at = datetime.now(tz=UTC)

    def cache_exchange_position_update(self, position: Any) -> None:
        positions = list(self._app._latest_exchange_positions or [])
        positions = [p for p in positions if getattr(p, "symbol", None) != position.symbol]
        if position.size > Decimal("0"):
            positions.append(position)
        self._app._cache_exchange_positions(positions)

    def recent_exchange_positions(self) -> list[Any] | None:
        assert self._app._settings is not None
        if self._app._latest_exchange_positions_at is None:
            return None
        age = (datetime.now(tz=UTC) - self._app._latest_exchange_positions_at).total_seconds()
        if age <= max(
            self._app._settings.POSITION_SYNC_INTERVAL_SECONDS,
            self._app._settings.POSITION_MANAGEMENT_INTERVAL_SECONDS,
        ):
            return self._app._latest_exchange_positions
        return None

    def effective_performance_blocks(self, active_symbols: list[str]) -> set[str]:
        assert self._app._settings is not None
        blocked = {symbol for symbol in self._app._performance_blocked_symbols if symbol in active_symbols}
        tradable_count = len(active_symbols) - len(blocked)
        min_tradable = max(0, self._app._settings.PERFORMANCE_MIN_TRADABLE_SYMBOLS)
        if blocked and tradable_count < min_tradable:
            log.warning(
                "performance_filter.relaxed",
                reason="too_few_tradable_symbols",
                blocked_symbols=sorted(blocked),
                active_symbols=active_symbols,
                min_tradable=min_tradable,
            )
            return set()
        return blocked

    def activation_price(self, entry_price: Decimal, side: str) -> Decimal:
        assert self._app._settings is not None
        delta = entry_price * Decimal(str(self._app._settings.TRAILING_ACTIVATION_PCT)) / Decimal("100")
        return entry_price + delta if side == "Buy" else entry_price - delta

    def breakeven_stop(self, entry_price: Decimal, side: str, fee_rates: Any | None = None) -> Decimal:
        """Compute a breakeven stop that covers round-trip taker fees + spread + slippage + buffer."""
        assert self._app._settings is not None
        # Default to config taker rate if no live fee data
        if fee_rates is not None:
            taker = Decimal(str(fee_rates.taker_fee_rate))
        else:
            taker = Decimal(str(self._app._settings.DEFAULT_LINEAR_TAKER_FEE_RATE))
        entry_fee_pct = taker * Decimal("100")
        exit_fee_pct = taker * Decimal("100")
        spread_pct = Decimal(str(self._app._settings.SCREENER_MAX_SPREAD_BPS)) / Decimal("100")
        slippage_pct = Decimal(str(self._app._settings.EXPECTED_SLIPPAGE_PCT)) * Decimal("2")
        buffer_pct = Decimal(str(self._app._settings.MIN_NET_PROFIT_BUFFER_PCT))
        total_offset_pct = entry_fee_pct + exit_fee_pct + spread_pct + slippage_pct + buffer_pct
        # Also respect the legacy static offset as a minimum floor
        static_pct = Decimal(str(self._app._settings.BREAKEVEN_STOP_OFFSET_PCT))
        offset_pct = max(total_offset_pct, static_pct)
        offset = entry_price * offset_pct / Decimal("100")
        return entry_price + offset if side == "Buy" else entry_price - offset

    def round_to_tick(
        self,
        price: Decimal,
        tick_size: Decimal,
        *,
        round_up: bool,
    ) -> Decimal:
        if tick_size <= Decimal("0"):
            return price
        rounding = ROUND_CEILING if round_up else ROUND_DOWN
        ticks = (price / tick_size).to_integral_value(rounding=rounding)
        return ticks * tick_size
