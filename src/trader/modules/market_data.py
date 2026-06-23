"""market_data."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

from trader.domain.errors import RateLimitError
from trader.monitoring.logging import get_logger
from trader.runtime.constants import (
    _INTERVAL_MS,
    _MIN_SEED_BARS,
    _SYMBOLS,
    _WS_INTERVAL,
)
from trader.runtime.module import ModuleTaskMixin

log = get_logger(__name__)


class MarketDataModule(ModuleTaskMixin):
    name = "market_data"

    def spawn_background_tasks(self, tasks: list[asyncio.Task[object]]) -> None:
        self._spawn(tasks, self.reconcile_unconfirmed_candles(), "candle-reconciler")
        self._spawn(tasks, self.run_startup_backfill(), "startup-backfill")
        self._spawn(tasks, self.run_load_governor(), "load-governor")
        self._spawn(tasks, self.run_symbol_subscribe_watchdog(), "subscribe-watchdog")

    async def on_screener_symbols_added(self, symbols: list[str]) -> None:
        """Seed candles and subscribe WebSocket for newly added screener symbols."""
        if self._app._subscribe_watchdog is not None:
            self._app._subscribe_watchdog.register(symbols)
        for symbol in symbols:
            # Seed historical candles (also invalidates cache, triggers recompute,
            # and pre-warms turnover_24h — see _seed_candle_store).
            await self.seed_candle_store(symbols=[symbol])
            # Subscribe WebSocket to the new symbol's topics
            if self._app._ws_public is not None:
                topics = self._app._ws_topics_for_symbol(symbol)
                await self._app._ws_public.subscribe(topics)
                log.info("screener.symbol_subscribed", symbol=symbol, topics=topics)

    async def on_screener_symbols_removed(self, symbols: list[str]) -> None:
        log.info("screener.symbols_removed", symbols=symbols)
        for symbol in symbols:
            self._app._last_candle_sample_at.pop(symbol, None)
            self._app._last_signal_at.pop(symbol, None)

    async def start_screener(self) -> list[str]:
        """Run the market screener and return initial symbol list."""
        from trader.features.screener import MarketScreener
        from trader.features.subscribe_watchdog import SubscribeWatchdog

        assert self._app._bybit_adapter is not None
        assert self._app._settings is not None

        self._app._subscribe_watchdog = SubscribeWatchdog(
            timeout_s=float(self._app._settings.SCREENER_SUBSCRIBE_TIMEOUT_SECONDS),
            max_retries=int(self._app._settings.SCREENER_SUBSCRIBE_MAX_RETRIES),
        )

        self._app._screener = MarketScreener(
            rest_client=self._app._bybit_adapter._rest,
            wide_max_symbols=self._app._settings.SCREENER_WIDE_MAX_SYMBOLS,
            feature_max_symbols=self._app._settings.SCREENER_FEATURE_MAX_SYMBOLS,
            execution_candidates=self._app._settings.SCREENER_EXECUTION_CANDIDATES,
            min_volume_usd=self._app._settings.SCREENER_MIN_VOLUME_USD,
            max_spread_bps=self._app._settings.SCREENER_MAX_SPREAD_BPS,
            min_top_book_depth_usd=self._app._settings.SCREENER_MIN_TOP_BOOK_DEPTH_USD,
            min_price_usd=self._app._settings.SCREENER_MIN_PRICE_USD,
            max_price_usd=self._app._settings.SCREENER_MAX_PRICE_USD,
            interval_s=self._app._settings.SCREENER_REFRESH_SECONDS,
            denylist=list(self._app._settings.SCREENER_DENYLIST),
            on_symbols_added=self.on_screener_symbols_added,
            on_symbols_removed=self.on_screener_symbols_removed,
            has_open_position=lambda symbol: (
                self._app._execution_engine is not None and self._app._execution_engine.has_open_position(symbol)
            ),
            has_pending_order=lambda symbol: (
                self._app._execution_engine is not None
                and self._app._execution_engine.has_pending_order_for_symbol(symbol)
            ),
        )

        # Run first screen synchronously so we have symbols before WS starts
        try:
            task = asyncio.create_task(self._app._screener.run(), name="screener")
            self._app._background_tasks.append(task)
            await self._app._screener.wait_ready()
            symbols = self._app._screener.active_symbols
            log.info("screener.initial_symbols", symbols=symbols)
            return symbols
        except Exception as exc:
            log.warning(
                "screener.startup_failed",
                error=str(exc),
                fallback=_SYMBOLS,
            )
            return list(_SYMBOLS)

    # ------------------------------------------------------------------
    # Market data & features
    # ------------------------------------------------------------------

    async def seed_candle_store(self, symbols: list[str] | None = None) -> None:
        """Fetch recent historical klines via REST to seed the CandleStore."""
        from trader.data.candles import Candle

        assert self._app._settings is not None
        assert self._app._bybit_adapter is not None

        if self._app._candle_store is None:
            self._app._candle_store = self._app._new_candle_store()

        has_api_key = bool(self._app._settings.BYBIT_API_KEY.get_secret_value())
        seed_symbols = symbols or _SYMBOLS
        retry_attempts = max(1, int(getattr(self._app._settings, "CANDLE_SEED_RETRY_ATTEMPTS", 3)))
        retry_base_delay_s = max(0.0, float(getattr(self._app._settings, "CANDLE_SEED_RETRY_BASE_DELAY_SECONDS", 1.0)))

        for symbol in seed_symbols:
            # Clear any pre-existing cached vector before touching the candle store.
            # This prevents the watchdog from reading a stale vector during the seeding
            # window and then re-caching it, which would persist until the next WS kline.
            if self._app._feature_pipeline is not None:
                self._app._feature_pipeline.invalidate_symbol(symbol)
            for interval in self._app._market_data_intervals():
                try:
                    for attempt in range(1, retry_attempts + 1):
                        try:
                            resp = await self._app._bybit_adapter._rest.get_kline(
                                category="linear",
                                symbol=symbol,
                                interval=interval,
                                limit=_MIN_SEED_BARS,
                            )
                            break
                        except RateLimitError:
                            if attempt >= retry_attempts:
                                raise
                            wait_s = retry_base_delay_s * (2 ** (attempt - 1))
                            log.warning(
                                "candle_store.seed_rate_limited_retrying",
                                symbol=symbol,
                                interval=interval,
                                attempt=attempt,
                                max_attempts=retry_attempts,
                                wait_seconds=round(wait_s, 3),
                            )
                            await asyncio.sleep(wait_s)
                    items = resp.get("result", {}).get("list", [])
                    # Bybit returns newest-first; reverse to oldest-first
                    items = list(reversed(items))
                    now = datetime.now(tz=UTC)
                    count = 0
                    for row in items:
                        # row: [startTime, open, high, low, close, volume, turnover]
                        try:
                            ts_ms = int(row[0])
                            open_time = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
                            bar_ms = _INTERVAL_MS.get(interval, 60_000)
                            # A candle is confirmed only after its full interval has elapsed.
                            # close_epoch_ms is the exclusive start of the next bar.
                            close_epoch_ms = ts_ms + bar_ms
                            close_time = datetime.fromtimestamp((close_epoch_ms - 1) / 1000, tz=UTC)
                            confirmed = now.timestamp() * 1000 >= close_epoch_ms
                            candle = Candle(
                                open_time=open_time,
                                open=float(row[1]),
                                high=float(row[2]),
                                low=float(row[3]),
                                close=float(row[4]),
                                volume=float(row[5]),
                                confirm=confirmed,
                            )
                            self._app._candle_store.add(symbol, interval, candle)
                            # Only persist confirmed candles — active REST candles
                            # may carry intermediate prices and must not be stored as
                            # confirmed=true in the training database.
                            if (
                                confirmed
                                and self._app._should_persist_candle_interval(interval)
                                and self._app._trade_journal is not None
                                and self._app._trade_journal.is_enabled
                            ):
                                await self._app._trade_journal.upsert_market_candle(
                                    symbol=symbol,
                                    interval=interval,
                                    open_time=open_time,
                                    close_time=close_time,
                                    open=Decimal(str(row[1])),
                                    high=Decimal(str(row[2])),
                                    low=Decimal(str(row[3])),
                                    close=Decimal(str(row[4])),
                                    volume=Decimal(str(row[5])),
                                    turnover=Decimal(str(row[6])),
                                    confirmed=True,
                                    source="rest_seed",
                                )
                            count += 1
                        except (IndexError, ValueError):
                            continue
                    log.info(
                        "candle_store.seeded",
                        symbol=symbol,
                        interval=interval,
                        bars=count,
                    )
                except Exception as exc:
                    log.warning(
                        "candle_store.seed_failed",
                        symbol=symbol,
                        interval=interval,
                        error=str(exc),
                        has_api_key=has_api_key,
                    )
            # After all intervals seeded: invalidate again (watchdog may have run during
            # the seeding window and re-cached a stale vector), then trigger an immediate
            # recompute so the next strategy call gets a valid vector without waiting for
            # the next WS kline or the 60-second watchdog cycle.
            if self._app._feature_pipeline is not None:
                self._app._feature_pipeline.invalidate_symbol(symbol)
                for interval in self._app._market_data_intervals():
                    await self._app._feature_pipeline.on_confirmed_candle(symbol, interval)

        # Pre-warm InstrumentInfo.turnover_24h for all seeded symbols in ONE batch call
        # so position_sizer never hits liquidity_data_missing on the first signal.
        # A single batch GET avoids per-symbol rate-limit pressure during startup.
        if self._app._execution_engine is not None and self._app._bybit_adapter is not None:
            await self.prefetch_ticker_turnover(seed_symbols)

    async def prefetch_ticker_turnover(self, symbols: list[str]) -> None:
        """Batch-fetch 24h turnover for position sizing (safe to call after execution init)."""
        if (
            not symbols
            or self._app._execution_engine is None
            or self._app._bybit_adapter is None
        ):
            return
        try:
            resp = await self._app._bybit_adapter._rest.get_tickers(category="linear")
            items = resp.get("result", {}).get("list", [])
            seed_set = set(symbols)
            for item in items:
                sym = item.get("symbol", "")
                if sym not in seed_set:
                    continue
                raw_t24h = item.get("turnover24h")
                if raw_t24h:
                    self._app._execution_engine.update_ticker_turnover(sym, Decimal(str(raw_t24h)))
        except Exception as exc:
            log.debug("seed.ticker_prefetch_failed", symbols=symbols, error=str(exc))

    async def reconcile_unconfirmed_candles(self) -> None:
        """Backfill candles that have become confirmed since the last write.

        Unconfirmed candles are never persisted (look-ahead bias guard), so a WS
        gap or a restart mid-bar can leave holes. Every 5 minutes this re-fetches
        the most recent klines via REST and upserts only those whose close_time
        has already passed (confirmed by clock, not by stream).
        """
        assert self._app._settings is not None
        reconcile_interval = 300  # 5 minutes
        bars_to_check = 30

        while not self._app._shutdown_event.is_set():
            try:
                await asyncio.wait_for(self._app._shutdown_event.wait(), timeout=reconcile_interval)
                break
            except TimeoutError:
                pass

            if (
                self._app._bybit_adapter is None
                or self._app._trade_journal is None
                or not self._app._trade_journal.is_enabled
            ):
                continue

            symbols = self._app._screener.active_symbols if self._app._screener is not None else list(_SYMBOLS)
            backfilled = 0
            for symbol in symbols:
                for interval in self._app._market_data_intervals():
                    try:
                        resp = await self._app._bybit_adapter._rest.get_kline(
                            category="linear",
                            symbol=symbol,
                            interval=interval,
                            limit=bars_to_check,
                        )
                        items = resp.get("result", {}).get("list", [])
                        now_ms = datetime.now(tz=UTC).timestamp() * 1000
                        bar_ms = _INTERVAL_MS.get(interval, 60_000)
                        for row in items:
                            try:
                                ts_ms = int(row[0])
                                close_epoch_ms = ts_ms + bar_ms
                                if now_ms < close_epoch_ms:
                                    continue  # still open — skip, no look-ahead
                                if not self._app._should_persist_candle_interval(interval):
                                    continue
                                await self._app._trade_journal.upsert_market_candle(
                                    symbol=symbol,
                                    interval=interval,
                                    open_time=datetime.fromtimestamp(ts_ms / 1000, tz=UTC),
                                    close_time=datetime.fromtimestamp((close_epoch_ms - 1) / 1000, tz=UTC),
                                    open=Decimal(str(row[1])),
                                    high=Decimal(str(row[2])),
                                    low=Decimal(str(row[3])),
                                    close=Decimal(str(row[4])),
                                    volume=Decimal(str(row[5])),
                                    turnover=Decimal(str(row[6])),
                                    confirmed=True,
                                    source="rest_reconcile",
                                )
                                backfilled += 1
                            except (IndexError, ValueError):
                                continue
                    except Exception as exc:
                        log.debug(
                            "candle_reconcile.fetch_failed",
                            symbol=symbol,
                            interval=interval,
                            error=str(exc),
                        )
            if backfilled:
                log.info(
                    "candle_reconcile.completed",
                    upserted=backfilled,
                    symbols=len(symbols),
                )

    async def run_startup_backfill(self) -> None:
        """One-shot historical candle backfill at startup.

        With a fresh/cleared DB the canary checklist needs ~1000 1m candles and
        model training needs labelled history — waiting for WS alone takes many
        hours. This pages back through REST klines for the active symbols and
        persists clock-confirmed candles only, respecting a hard request cap.
        Idempotent: upsert_market_candle deduplicates on (symbol, interval, open_time).

        Behaviour:
        - Waits for the screener to publish its first symbol universe (so the
          backfill targets real trading symbols, not the static fallback list).
        - Waits up to 60s for the DB connection (it may still be bootstrapping).
        - Skips (symbol, interval) pairs whose stored history already covers
          >= 90% of the requested window — restarts cost near-zero REST quota.
        - Never raises: a backfill failure must not take down the supervisor.
        """
        assert self._app._settings is not None
        if not self._app._settings.STARTUP_BACKFILL_ENABLED:
            log.info("startup_backfill.disabled")
            return
        try:
            await self._app._startup_backfill()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("startup_backfill.failed", error=str(exc), error_type=type(exc).__name__)

    async def startup_backfill(self) -> None:
        assert self._app._settings is not None

        # Wait for the screener's first refresh so we backfill the real universe.
        if self._app._screener is not None:
            try:
                await asyncio.wait_for(self._app._screener.wait_ready(), timeout=120)
            except TimeoutError:
                log.warning(
                    "startup_backfill.screener_not_ready",
                    fallback_symbols=list(_SYMBOLS),
                )

        # The trade journal connects concurrently at startup — give it up to 60s.
        for _ in range(12):
            if self._app._trade_journal is not None and self._app._trade_journal.is_enabled:
                break
            if self._app._shutdown_event.is_set():
                return
            await asyncio.sleep(5)
        if (
            self._app._bybit_adapter is None
            or self._app._trade_journal is None
            or not self._app._trade_journal.is_enabled
        ):
            log.info("startup_backfill.skipped", reason="no_adapter_or_db")
            return

        days = max(1, int(self._app._settings.STARTUP_BACKFILL_DAYS))
        max_requests = max(1, int(self._app._settings.STARTUP_BACKFILL_MAX_REQUESTS))
        window_ms = days * 86_400_000
        symbols = self._app._screener.active_symbols if self._app._screener is not None else list(_SYMBOLS)
        if not symbols:
            symbols = list(_SYMBOLS)

        # Gap detection: skip pairs whose history already covers the window.
        try:
            existing_counts = await self._app._trade_journal.get_candle_counts_per_symbol()
        except Exception as exc:
            log.debug("startup_backfill.count_check_failed", error=str(exc))
            existing_counts = {}

        requests_used = 0
        total_upserted = 0
        skipped_pairs = 0

        for symbol in symbols:
            for interval in self._app._market_data_intervals():
                bar_ms = _INTERVAL_MS.get(interval, 60_000)
                expected_bars = window_ms // bar_ms
                have = existing_counts.get((symbol, interval), 0)
                if have >= expected_bars * 0.9:
                    skipped_pairs += 1
                    continue

                end_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
                oldest_needed_ms = end_ms - window_ms
                while end_ms > oldest_needed_ms and requests_used < max_requests:
                    if self._app._shutdown_event.is_set():
                        return
                    try:
                        resp = await self._app._bybit_adapter._rest.get_kline(
                            category="linear",
                            symbol=symbol,
                            interval=interval,
                            end=end_ms,
                            limit=1000,
                        )
                    except Exception as exc:
                        log.warning(
                            "startup_backfill.fetch_failed",
                            symbol=symbol,
                            interval=interval,
                            error=str(exc),
                        )
                        break
                    requests_used += 1
                    items = resp.get("result", {}).get("list", [])
                    if not items:
                        break
                    now_ms = datetime.now(tz=UTC).timestamp() * 1000
                    oldest_in_page = end_ms
                    for row in items:
                        try:
                            ts_ms = int(row[0])
                            oldest_in_page = min(oldest_in_page, ts_ms)
                            close_epoch_ms = ts_ms + bar_ms
                            if now_ms < close_epoch_ms:
                                continue  # unconfirmed — never persist (look-ahead guard)
                            if not self._app._should_persist_candle_interval(interval):
                                continue
                            await self._app._trade_journal.upsert_market_candle(
                                symbol=symbol,
                                interval=interval,
                                open_time=datetime.fromtimestamp(ts_ms / 1000, tz=UTC),
                                close_time=datetime.fromtimestamp((close_epoch_ms - 1) / 1000, tz=UTC),
                                open=Decimal(str(row[1])),
                                high=Decimal(str(row[2])),
                                low=Decimal(str(row[3])),
                                close=Decimal(str(row[4])),
                                volume=Decimal(str(row[5])),
                                turnover=Decimal(str(row[6])),
                                confirmed=True,
                                source="rest_backfill",
                            )
                            total_upserted += 1
                        except (IndexError, ValueError):
                            continue
                    if oldest_in_page >= end_ms:
                        break  # no progress — avoid infinite loop
                    end_ms = oldest_in_page - 1
                    await asyncio.sleep(0.25)  # be gentle on REST rate limits
            if requests_used >= max_requests:
                log.info("startup_backfill.request_cap_reached", cap=max_requests)
                break

        log.info(
            "startup_backfill.completed",
            symbols=len(symbols),
            requests_used=requests_used,
            candles_upserted=total_upserted,
            pairs_skipped_already_full=skipped_pairs,
        )
        if total_upserted > 0 and self._app._telegram_bot is not None:
            try:
                await self._app._telegram_bot.notify(
                    f"📥 <b>Стартовый backfill завершен</b>\n"
                    f"Свечей записано: <code>{total_upserted}</code> | "
                    f"REST-запросов: <code>{requests_used}/{max_requests}</code>\n"
                    f"Монет: <code>{len(symbols)}</code> | "
                    f"Пар пропущено (история уже есть): <code>{skipped_pairs}</code>\n"
                    f"Модель начнет обучение после накопления размеченных исходов."
                )
            except Exception as exc:
                log.debug("startup_backfill.notify_failed", error=str(exc))

    async def start_public_ws(self, symbols: list[str]) -> None:
        """Start the public WebSocket and wire events to CandleStore."""
        from trader.exchange.bybit_ws_public import BybitPublicWebSocket
        from trader.exchange.endpoint_selector import EndpointSelector

        assert self._app._settings is not None
        assert self._app._health_checker is not None

        if self._app._candle_store is None:
            self._app._candle_store = self._app._new_candle_store()

        selector = EndpointSelector(
            self._app._settings.BYBIT_REGION,
            self._app._settings.BYBIT_USE_TESTNET,
        )

        # Build subscription list from screened symbols
        category = self._app._settings.DEFAULT_MARKET_CATEGORY
        if self._app._subscribe_watchdog is not None:
            self._app._subscribe_watchdog.register(symbols)
        subs: list[str] = []
        for symbol in symbols:
            for interval in self._app._market_data_intervals():
                subs.append(f"kline.{interval}.{symbol}")
            subs.append(f"tickers.{symbol}")

        # Orderbook L2 feed only for execution candidates (not the whole
        # universe) — imbalance/microprice features cost ~5-10 KB/s per symbol.
        if self._app._settings.ORDERBOOK_FEED_ENABLED:
            from trader.data.orderbook_tracker import OrderbookTracker

            self._app._orderbook_tracker = OrderbookTracker()
            ob_symbols = self._app._screener.execution_candidates if self._app._screener is not None else symbols[:5]
            max_ob = max(1, int(self._app._settings.MAX_ORDERBOOK_ACTIVE_SYMBOLS))
            if str(self._app._settings.ORDERBOOK_MODE).upper() == "STREAMING":
                ob_symbols = symbols[:max_ob]
            else:
                ob_symbols = ob_symbols[:max_ob]
            for symbol in ob_symbols:
                if symbol in symbols:
                    subs.append(f"orderbook.50.{symbol}")

        flow_symbols: list[str] = []
        if self._app._settings.TRADE_FLOW_FEED_ENABLED or self._app._settings.LIQUIDATION_FEED_ENABLED:
            from trader.data.flow_tracker import FlowTracker

            self._app._flow_tracker = FlowTracker(
                window_s=self._app._settings.FLOW_TRACKER_WINDOW_SECONDS,
                large_trade_notional_usd=self._app._settings.FLOW_LARGE_TRADE_NOTIONAL_USD,
            )
            flow_symbols = self._app._screener.execution_candidates if self._app._screener is not None else symbols[:5]
            for symbol in flow_symbols:
                if symbol not in symbols:
                    continue
                if self._app._settings.TRADE_FLOW_FEED_ENABLED:
                    subs.append(f"publicTrade.{symbol}")
                if self._app._settings.LIQUIDATION_FEED_ENABLED:
                    subs.append(f"allLiquidation.{symbol}")

        # Orderbook deltas add ~150-300 events/s on top of klines/tickers —
        # size the buffer so a consumer stall never drops a confirmed kline.
        from trader.domain.events import BaseEvent

        event_queue: asyncio.Queue[BaseEvent] = asyncio.Queue(maxsize=5000)

        self._app._ws_public = BybitPublicWebSocket(
            endpoint=f"{selector.ws_public_base}/{category}",
            subscriptions=subs,
            event_queue=event_queue,
        )

        # Event consumer: feeds CandleStore, triggers features, writes candle journal
        async def consume_events() -> None:

            from trader.data.candles import candle_from_kline_event
            from trader.domain.events import KlineEvent, LiquidationEvent, OrderBookEvent, TickerEvent, TradeEvent

            while not self._app._shutdown_event.is_set():
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                    if isinstance(event, OrderBookEvent):
                        if self._app._orderbook_tracker is not None:
                            self._app._orderbook_tracker.record(event.symbol, event.bids, event.asks)
                    elif isinstance(event, TradeEvent):
                        if self._app._flow_tracker is not None:
                            self._app._flow_tracker.record_trade(
                                event.symbol,
                                event.side,
                                event.price,
                                event.qty,
                                event.executed_at,
                            )
                    elif isinstance(event, LiquidationEvent):
                        if self._app._flow_tracker is not None:
                            self._app._flow_tracker.record_liquidation(
                                event.symbol,
                                event.side,
                                event.price,
                                event.qty,
                                event.timestamp,
                            )
                    elif isinstance(event, TickerEvent):
                        if (
                            event.turnover_24h is not None
                            and event.turnover_24h > 0
                            and self._app._execution_engine is not None
                        ):
                            self._app._execution_engine.update_ticker_turnover(event.symbol, event.turnover_24h)
                    elif isinstance(event, KlineEvent):
                        candle = candle_from_kline_event(event)
                        if self._app._candle_store is None:
                            continue
                        self._app._candle_store.add(event.symbol, event.interval, candle)
                        if self._app._subscribe_watchdog is not None:
                            self._app._subscribe_watchdog.confirm_ws_kline(event.symbol, event.interval)

                        if event.confirm:
                            self._app._last_confirmed_candle_at = datetime.now(tz=UTC)
                            # Event-driven feature recompute for this (symbol, interval)
                            if self._app._feature_pipeline is not None:
                                vec = await self._app._feature_pipeline.on_confirmed_candle(
                                    event.symbol, event.interval
                                )
                                # MTF pattern features on 1m depend on closed 5m/15m bars — refresh 1m
                                # when a higher-TF candle closes so pat5_/pat15_ stay current.
                                if event.interval in ("5", "15"):
                                    self._app._feature_pipeline.invalidate_symbol(event.symbol)
                                    await self._app._feature_pipeline.on_confirmed_candle(
                                        event.symbol, _WS_INTERVAL
                                    )
                                # Per-candle training sampler: a labelled sample per
                                # confirmed 1m candle instead of per trade signal
                                if vec is not None and event.interval == _WS_INTERVAL:
                                    await self._app._sample_confirmed_candle(event.symbol, event.interval, vec)

                            # Persist confirmed candle to PostgreSQL (best-effort, selected intervals only)
                            if (
                                self._app._should_persist_candle_interval(event.interval)
                                and self._app._trade_journal is not None
                                and self._app._trade_journal.is_enabled
                            ):
                                bar_ms = _INTERVAL_MS.get(event.interval, 60_000)
                                close_time = datetime.fromtimestamp(
                                    (event.open_time.timestamp() * 1000 + bar_ms - 1) / 1000,
                                    tz=UTC,
                                )
                                try:
                                    await self._app._trade_journal.upsert_market_candle(
                                        symbol=event.symbol,
                                        interval=event.interval,
                                        open_time=event.open_time,
                                        close_time=close_time,
                                        open=event.open,
                                        high=event.high,
                                        low=event.low,
                                        close=event.close,
                                        volume=event.volume,
                                        turnover=event.turnover,
                                        confirmed=True,
                                        source="ws",
                                    )
                                except Exception as _candle_exc:
                                    log.debug(
                                        "ws_consumer.candle_journal_failed",
                                        symbol=event.symbol,
                                        error=str(_candle_exc),
                                    )

                    # Update WS health on any message
                    if self._app._health_checker:
                        self._app._health_checker.set_ws_status(
                            connected=True,
                            last_message_at=datetime.now(tz=UTC),
                        )
                except TimeoutError:
                    # Check if WS is still connected
                    if self._app._ws_public and not self._app._ws_public.is_connected:
                        if self._app._health_checker:
                            self._app._health_checker.set_ws_status(connected=False)
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    log.warning("ws_consumer.error", error=str(exc))

        ws_task = asyncio.create_task(self._app._ws_public.start(), name="ws-public")
        consumer_task = asyncio.create_task(consume_events(), name="ws-consumer")
        self._app._background_tasks.extend([ws_task, consumer_task])
        log.info(
            "public_ws.started",
            endpoint=selector.ws_public_base,
            subscriptions=subs,
        )

    async def run_load_governor(self) -> None:
        """Adaptive load governor: reduce feature symbols when system is under pressure.

        Monitors event-loop lag and WS queue utilisation every
        LOAD_GOVERNOR_CHECK_SECONDS. When any metric exceeds its threshold,
        the screener's feature universe is narrowed by one symbol (down to the
        configured minimum). When all metrics are healthy the universe is
        gradually restored toward the original maximum.
        """
        assert self._app._settings is not None
        if not self._app._settings.ADAPTIVE_LOAD_GOVERNOR_ENABLED:
            return

        check_interval = float(self._app._settings.LOAD_GOVERNOR_CHECK_SECONDS)
        max_lag_ms = float(self._app._settings.MAX_EVENT_LOOP_LAG_MS)
        min_symbols = int(self._app._settings.LOAD_GOVERNOR_MIN_FEATURE_SYMBOLS)
        min_exec = int(self._app._settings.LOAD_GOVERNOR_MIN_EXECUTION_CANDIDATES)
        max_feature_cycle_ms = float(self._app._settings.MAX_FEATURE_CYCLE_MS)

        # Original feature_max from screener (set at startup)
        original_max: int | None = None
        original_exec: int | None = None
        overload_streak = 0
        restore_streak = 0
        ws_stale_threshold_s = 90.0

        while not self._app._shutdown_event.is_set():
            await asyncio.sleep(check_interval)
            if self._app._screener is None:
                continue

            if original_max is None:
                original_max = getattr(self._app._screener, "_original_feature_max", self._app._screener._feature_max)
            if original_exec is None:
                original_exec = getattr(
                    self._app._screener, "_original_exec_candidates", self._app._screener._exec_candidates
                )

            # --- Measure event-loop lag ---
            t0 = asyncio.get_event_loop().time()
            await asyncio.sleep(0)  # yield and immediately return
            lag_ms = (asyncio.get_event_loop().time() - t0) * 1000

            # --- Measure WS queue utilisation (if accessible) ---
            # The event queue is local to _start_public_ws, so we track pressure
            # by checking if health checker reports recent WS staleness
            ws_stale = False
            if self._app._health_checker is not None and self._app._health_checker._last_ws_message_at is not None:
                ws_age = (datetime.now(tz=UTC) - self._app._health_checker._last_ws_message_at).total_seconds()
                ws_stale = ws_age > ws_stale_threshold_s

            symbol_count = max(1, len(self._app._active_symbols()))
            per_symbol_cycle_ms = self._app._last_strategy_cycle_ms / symbol_count
            feature_cycle_overload = max_feature_cycle_ms > 0 and per_symbol_cycle_ms > max_feature_cycle_ms
            overloaded = lag_ms > max_lag_ms or ws_stale or feature_cycle_overload
            if overloaded:
                overload_streak += 1
                restore_streak = 0
            else:
                restore_streak += 1
                overload_streak = 0
            current = self._app._screener._feature_max
            current_exec = self._app._screener._exec_candidates

            if overloaded and overload_streak >= 2 and current > min_symbols:
                streak = overload_streak
                new_max = max(min_symbols, current - 1)
                self._app._screener._feature_max = new_max
                if current_exec > min_exec:
                    self._app._screener._exec_candidates = max(min_exec, current_exec - 1)
                overload_streak = 0
                log.warning(
                    "load_governor.reducing_symbols",
                    lag_ms=round(lag_ms, 1),
                    ws_stale=ws_stale,
                    feature_cycle_ms=round(self._app._last_strategy_cycle_ms, 1),
                    per_symbol_cycle_ms=round(per_symbol_cycle_ms, 1),
                    symbol_count=symbol_count,
                    overload_streak=streak,
                    from_max=current,
                    to_max=new_max,
                    from_exec=current_exec,
                    to_exec=self._app._screener._exec_candidates,
                    min_symbols=min_symbols,
                )
            elif not overloaded and restore_streak >= 2 and current < original_max:
                # Restore one symbol at a time
                streak = restore_streak
                new_max = min(original_max, current + 1)
                self._app._screener._feature_max = new_max
                if original_exec is not None and current_exec < original_exec:
                    self._app._screener._exec_candidates = min(original_exec, current_exec + 1)
                restore_streak = 0
                log.info(
                    "load_governor.restoring_symbols",
                    lag_ms=round(lag_ms, 1),
                    restore_streak=streak,
                    from_max=current,
                    to_max=new_max,
                    from_exec=current_exec,
                    to_exec=self._app._screener._exec_candidates,
                )

    async def run_symbol_subscribe_watchdog(self) -> None:
        """Retry or reconnect WS when screener symbols never receive 1m klines."""
        assert self._app._settings is not None
        if self._app._subscribe_watchdog is None:
            return
        interval_s = max(2.0, min(10.0, float(self._app._settings.SCREENER_SUBSCRIBE_TIMEOUT_SECONDS) / 2.0))
        while not self._app._shutdown_event.is_set():
            try:
                await asyncio.wait_for(self._app._shutdown_event.wait(), timeout=interval_s)
                break
            except TimeoutError:
                pass

            expired = self._app._subscribe_watchdog.expired()
            if not expired:
                continue

            for symbol in expired:
                self._app._subscribe_watchdog.record_timeout(symbol)
                force_reconnect = self._app._subscribe_watchdog.mark_retry(symbol)
                if self._app._ws_public is not None:
                    if force_reconnect:
                        log.warning(
                            "screener.subscribe_watchdog.force_reconnect",
                            symbol=symbol,
                            timeout_s=self._app._settings.SCREENER_SUBSCRIBE_TIMEOUT_SECONDS,
                        )
                        await self._app._ws_public.force_reconnect()
                    else:
                        topics = self._app._ws_topics_for_symbol(symbol)
                        log.warning(
                            "screener.subscribe_watchdog.resubscribe",
                            symbol=symbol,
                            topics=topics,
                        )
                        await self._app._ws_public.subscribe(topics)
