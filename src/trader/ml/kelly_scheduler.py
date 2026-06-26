"""Scheduler for periodic Kelly model training.

Automatically triggers training when:
- Enough new trades have accumulated
- Sufficient time has elapsed since last training
- Explicitly requested
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class KellyTrainingScheduler:
    """Manages periodic training of Kelly predictor models."""

    def __init__(
        self,
        kelly_trainer: Any,
        kelly_predictor: Any,
        min_trades_per_training: int = 50,
        min_hours_between_trainings: int = 24,
    ):
        """Initialize scheduler.

        Args:
            kelly_trainer: KellyTrainer instance
            kelly_predictor: MLKellyPredictor instance
            min_trades_per_training: Trigger training after this many new trades
            min_hours_between_trainings: Minimum hours between training runs
        """
        self._trainer = kelly_trainer
        self._predictor = kelly_predictor
        self._min_trades = min_trades_per_training
        self._min_hours = min_hours_between_trainings

        self._last_training_time: Optional[datetime] = None
        self._trades_since_training: int = 0
        self._background_task: Optional[asyncio.Task[None]] = None
        self._is_training: bool = False

    async def start(self, trade_getter: Callable[[], list[dict[str, Any]]]) -> None:
        """Start background training scheduler.

        Args:
            trade_getter: Async callable that returns list of trades
        """
        if self._background_task is not None:
            return

        self._background_task = asyncio.create_task(
            self._training_loop(trade_getter)
        )
        logger.info("kelly_training_scheduler.started")

    async def stop(self) -> None:
        """Stop background scheduler."""
        if self._background_task is not None:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None
        logger.info("kelly_training_scheduler.stopped")

    async def on_trade_closed(self, trade: dict[str, Any]) -> None:
        """Notify scheduler that a trade closed (trigger potential training)."""
        self._trades_since_training += 1

        if self._should_train():
            await self.trigger_training([])

    async def trigger_training(self, trades: list[dict[str, Any]]) -> tuple[bool, str]:
        """Manually trigger training immediately."""
        if self._is_training:
            return False, "Training already in progress"

        self._is_training = True
        try:
            success, message = await self._trainer.train_from_trades(
                trades or [],
                self._predictor,
            )
            if success:
                self._trades_since_training = 0
                self._last_training_time = datetime.now(UTC)
            return success, message
        finally:
            self._is_training = False

    def _should_train(self) -> bool:
        """Check if training should be triggered."""
        if self._is_training:
            return False

        if self._trades_since_training >= self._min_trades:
            return True

        if self._last_training_time is not None:
            time_since = datetime.now(UTC) - self._last_training_time
            if time_since >= timedelta(hours=self._min_hours):
                return True

        return False

    async def _training_loop(
        self,
        trade_getter: Callable[[], list[dict[str, Any]]],
    ) -> None:
        """Background loop that periodically checks and trains."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                if self._should_train():
                    logger.info(
                        f"kelly_training_scheduler.triggering: "
                        f"trades_since={self._trades_since_training}, "
                        f"min_required={self._min_trades}"
                    )
                    trades = trade_getter()
                    await self.trigger_training(trades)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"kelly_training_scheduler.error: {e}")
                await asyncio.sleep(30)
