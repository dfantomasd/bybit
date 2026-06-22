"""Background ops: retention, outcomes, reconciliation, transaction log."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from trader.monitoring.logging import get_logger
from trader.runtime.constants import (
    _TRADE_JOURNAL_RECONNECT_INTERVAL,
)
from trader.runtime.module import ModuleTaskMixin

log = get_logger(__name__)


class OpsModule(ModuleTaskMixin):
    """Postgres hygiene and exchange reconciliation loops."""

    name = "ops"

    def spawn_background_tasks(self, tasks: list[asyncio.Task[object]]) -> None:
        self._spawn(tasks, self.run_data_retention(), "data-retention")
        self._spawn(tasks, self.run_outcome_resolver(), "outcome-resolver")
        self._spawn(tasks, self.run_reconciliation(), "reconciliation")
        self._spawn(tasks, self.run_transaction_log_sync(), "transaction-log-sync")

    async def maybe_run_startup_retention(self) -> None:
        """One-shot purge after Postgres connects to trim historical bloat."""
        assert self._app._settings is not None
        if (
            self._app._startup_retention_done
            or not self._app._settings.DATA_RETENTION_ENABLED
            or not self._app._settings.DATA_RETENTION_RUN_ON_STARTUP
            or self._app._trade_journal is None
            or not self._app._trade_journal.is_enabled
        ):
            return
        self._app._startup_retention_done = True
        try:
            report = await self._app._trade_journal.run_data_retention_policy(self._app._settings)
            self._app._last_retention_run_at = datetime.now(tz=UTC)
            log.info("data_retention.startup_complete", **report)
        except Exception as exc:
            log.warning("data_retention.startup_failed", error=str(exc))

    async def run_data_retention(self) -> None:
        assert self._app._settings is not None
        if not self._app._settings.DATA_RETENTION_ENABLED:
            log.info("data_retention.disabled")
            return
        interval_h = max(1.0, float(self._app._settings.DATA_RETENTION_INTERVAL_HOURS))
        while not self._app._shutdown_event.is_set():
            if self._app._trade_journal is not None and self._app._trade_journal.is_enabled:
                try:
                    report = await self._app._trade_journal.run_data_retention_policy(self._app._settings)
                    self._app._last_retention_run_at = datetime.now(tz=UTC)
                    log.info("data_retention.run_complete", **report)
                except Exception as exc:
                    log.warning("data_retention.failed", error=str(exc))
            try:
                await asyncio.wait_for(self._app._shutdown_event.wait(), timeout=interval_h * 3600.0)
            except TimeoutError:
                pass

    async def run_outcome_resolver(self) -> None:
        """Resolve prediction outcomes by comparing feature snapshot prices with market_candles."""
        assert self._app._settings is not None
        interval = float(self._app._settings.OUTCOME_RESOLVER_INTERVAL_SECONDS)
        batch_limit = int(self._app._settings.OUTCOME_RESOLVER_BATCH_LIMIT)
        horizons = [5, 15, 30, 60]

        while not self._app._shutdown_event.is_set():
            if self._app._trade_journal is not None and self._app._trade_journal.is_enabled:
                for horizon in horizons:
                    try:
                        resolved = await self._app._trade_journal.resolve_outcomes_from_candles(
                            horizon_minutes=horizon,
                            limit=batch_limit,
                        )
                        if resolved > 0:
                            log.info(
                                "outcome_resolver.resolved",
                                horizon_minutes=horizon,
                                count=resolved,
                            )
                    except Exception as exc:
                        log.warning("outcome_resolver.error", horizon=horizon, error=str(exc))
                await self._maybe_apply_online_learning()

            try:
                await asyncio.wait_for(
                    self._app._shutdown_event.wait(),
                    timeout=interval,
                )
            except TimeoutError:
                pass

    async def run_reconciliation(self) -> None:
        """Periodic reconciliation: compare local order state with exchange."""
        assert self._app._settings is not None
        interval = float(self._app._settings.RECONCILIATION_INTERVAL_SECONDS)

        while not self._app._shutdown_event.is_set():
            try:
                if self._app._bybit_adapter is not None and not self._app._initial_shadow_mode():
                    result = await self._app._bybit_adapter.reconcile()
                    if result.discrepancies_found > 0:
                        log.warning(
                            "reconciliation.discrepancies_found",
                            discrepancies=result.discrepancies_found,
                            mismatched=result.mismatched_order_ids[:10],
                            summary=result.summary,
                        )
                    else:
                        log.debug("reconciliation.clean", summary=result.summary)
            except Exception as exc:
                log.warning("reconciliation.error", error=str(exc))

            try:
                await asyncio.wait_for(
                    self._app._shutdown_event.wait(),
                    timeout=interval,
                )
            except TimeoutError:
                pass

    async def run_transaction_log_sync(self) -> None:
        """Periodically sync Bybit transaction log outside the hot strategy loop."""
        assert self._app._settings is not None
        interval = max(1.0, float(self._app._settings.TRANSACTION_LOG_SYNC_INTERVAL_SECONDS))

        while not self._app._shutdown_event.is_set():
            self._app._last_tx_log_sync_at = datetime.now(tz=UTC)
            try:
                await self._app._sync_transaction_log()
            except Exception as exc:
                log.debug("transaction_log.periodic_sync_failed", error=str(exc))

            try:
                await asyncio.wait_for(
                    self._app._shutdown_event.wait(),
                    timeout=interval,
                )
            except TimeoutError:
                pass

    async def sync_transaction_log(self) -> None:
        """Sync Bybit transaction log to database — supports pagination up to 5 pages."""
        assert self._app._settings is not None
        if self._app._trade_journal is None or self._app._bybit_adapter is None:
            return

        log.info("transaction_log.sync_started")
        total_fetched = 0
        total_inserted = 0
        cursor: str | None = None
        max_pages = 5

        try:
            for _page in range(max_pages):
                resp = await self._app._bybit_adapter._rest.get_transaction_log(
                    account_type="UNIFIED",
                    category=self._app._settings.DEFAULT_MARKET_CATEGORY,
                    currency="USDT",
                    limit=50,
                    cursor=cursor,
                )
                result = resp.get("result") or {}
                entries = result.get("list", [])
                next_cursor = result.get("nextPageCursor") or ""

                if entries:
                    total_fetched += len(entries)
                    log.info(
                        "transaction_log.page_fetched",
                        page=_page + 1,
                        count=len(entries),
                    )
                    inserted = await self._app._trade_journal.record_transaction_log_entries(entries)
                    total_inserted += inserted
                    log.info(
                        "transaction_log.entries_inserted",
                        page=_page + 1,
                        inserted=inserted,
                        fetched=len(entries),
                    )

                if not next_cursor:
                    break
                cursor = next_cursor

            log.info(
                "transaction_log.sync_complete",
                total_fetched=total_fetched,
                total_inserted=total_inserted,
            )
        except Exception as exc:
            log.warning("transaction_log.sync_failed", error=str(exc))

    async def run_trade_journal_reconnector(self) -> None:
        """Keep trying Postgres after transient Render startup/network failures."""
        if self._app._trade_journal is None:
            return
        while not self._app._shutdown_event.is_set():
            # Older tests/fakes may not expose durable_state_healthy; treat them as healthy.
            durable_healthy = bool(getattr(self._app._trade_journal, "durable_state_healthy", True))
            if not self._app._trade_journal.is_enabled or not durable_healthy:
                try:
                    connected = await self._app._trade_journal.reconnect_if_needed(
                        min_interval=_TRADE_JOURNAL_RECONNECT_INTERVAL,
                        force=False,
                    )
                    if connected:
                        log.info("trade_journal.reconnected")
                        await self._app._restore_execution_pending_entries()
                        await self._app._maybe_run_startup_retention()
                except Exception as exc:
                    log.debug("trade_journal.reconnect_failed", error=str(exc))
            blocked_remaining = getattr(self._app._trade_journal, "reconnect_blocked_remaining_seconds", lambda: 0.0)()
            sleep_s = max(_TRADE_JOURNAL_RECONNECT_INTERVAL, float(blocked_remaining or 0.0))
            sleep_s = min(sleep_s, 120.0)
            try:
                await asyncio.wait_for(
                    self._app._shutdown_event.wait(),
                    timeout=sleep_s,
                )
            except TimeoutError:
                continue
