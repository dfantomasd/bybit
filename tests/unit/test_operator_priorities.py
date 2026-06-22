"""Tests for operator priority legends."""

from __future__ import annotations

from trader.operator_priorities import (
    full_priority_overview,
    strategy_priority_text,
)


def test_strategy_priority_text_lists_order() -> None:
    text = strategy_priority_text(
        risk_profile="MODERATE",
        order_raw="order_flow_v1,scalp_micro_v1,ema_crossover_v1",
    )
    assert "order_flow_v1" in text
    assert "scalp_micro_v1" in text
    assert "MODERATE" in text


def test_full_priority_overview_uses_runtime_strategy_order() -> None:
    text = full_priority_overview(
        runtime_settings={
            "risk_profile": "SCALP",
            "scalp_strategy_priority_order": "scalp_micro_v1,order_flow_v1",
        }
    )
    assert "Приоритет безопасности" in text
    assert "scalp_micro_v1" in text
    assert "SCALP" in text
