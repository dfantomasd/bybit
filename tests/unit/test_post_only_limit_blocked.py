"""Tests for POST_ONLY_LIMIT fail-fast rejection (P0.5)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_post_only_limit_raises_on_init() -> None:
    from trader.execution.engine import ExecutionEngine

    with pytest.raises(ValueError, match="MARKET"):
        ExecutionEngine(
            adapter=MagicMock(),
            risk_manager=MagicMock(),
            exposure_tracker=MagicMock(),
            entry_order_mode="POST_ONLY_LIMIT",
        )


def test_unknown_mode_raises_on_init() -> None:
    from trader.execution.engine import ExecutionEngine

    with pytest.raises(ValueError, match="MARKET"):
        ExecutionEngine(
            adapter=MagicMock(),
            risk_manager=MagicMock(),
            exposure_tracker=MagicMock(),
            entry_order_mode="LIMIT",
        )


def test_market_mode_accepted() -> None:
    from trader.execution.engine import ExecutionEngine

    engine = ExecutionEngine(
        adapter=MagicMock(),
        risk_manager=MagicMock(),
        exposure_tracker=MagicMock(),
        entry_order_mode="MARKET",
    )
    assert engine._entry_order_mode == "MARKET"


def test_default_mode_is_market() -> None:
    from trader.execution.engine import ExecutionEngine

    engine = ExecutionEngine(
        adapter=MagicMock(),
        risk_manager=MagicMock(),
        exposure_tracker=MagicMock(),
    )
    assert engine._entry_order_mode == "MARKET"
