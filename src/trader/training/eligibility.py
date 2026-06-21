"""Shared SQL helpers for ML training sample eligibility."""

from __future__ import annotations

# Candle-sampling baselines store RULE_BASELINE_V1 labels without strategy_id.
_CANDLE_BASELINE_DECISIONS = ("SHADOW_CANDLE", "HISTORICAL_REAL")


def training_strategy_filter_sql(param: str = "$4") -> str:
    """Return SQL predicate for optional TRAIN_STRATEGY_ALLOWLIST.

    When an allowlist is configured, candle-sampling baselines without
    ``strategy_id`` remain eligible because they carry most of the labelled data.
    """
    decisions = ", ".join(f"'{item}'" for item in _CANDLE_BASELINE_DECISIONS)
    return f"""(
        {param}::text[] IS NULL
        OR pe.metadata->>'strategy_id' = ANY({param}::text[])
        OR (
            pe.metadata->>'strategy_id' IS NULL
            AND COALESCE(pe.decision, '') IN ({decisions})
        )
    )"""
