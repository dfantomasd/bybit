"""Shared SQL helpers for ML training sample eligibility."""

from __future__ import annotations

# Candle-sampling baselines store RULE_BASELINE_V1 labels without strategy_id.
_CANDLE_BASELINE_DECISIONS = ("SHADOW_CANDLE", "HISTORICAL_REAL")
_STRATEGY_BASELINE_DECISIONS = ("GATE_PASS", "GATE_BLOCK", "SHADOW_BASELINE")


def training_strategy_filter_sql(
    allowlist_param: str = "$4",
    include_candle_baseline_param: str = "$5",
) -> str:
    """Return SQL predicate for TRAIN_STRATEGY_ALLOWLIST / TRAIN_INCLUDE_CANDLE_BASELINE.

    When the allowlist is empty, all RULE_BASELINE_V1 Buy/Sell labels are eligible.

    When the allowlist is set and ``TRAIN_INCLUDE_CANDLE_BASELINE`` is false, only
    rows whose ``metadata.strategy_id`` is in the allowlist are used. This keeps
    training aligned with the live scalp (or other) strategy instead of the EMA
    candle sampler.

    When the allowlist is set and ``TRAIN_INCLUDE_CANDLE_BASELINE`` is true, candle
    baselines (``SHADOW_CANDLE`` / ``HISTORICAL_REAL``) are also included regardless
    of ``strategy_id`` — the live candle sampler tags rows as ``candle_sampler_v1``.
    """
    decisions = ", ".join(f"'{item}'" for item in _CANDLE_BASELINE_DECISIONS)
    return f"""(
        {allowlist_param}::text[] IS NULL
        OR cardinality({allowlist_param}::text[]) = 0
        OR pe.metadata->>'strategy_id' = ANY({allowlist_param}::text[])
        OR (
            {include_candle_baseline_param}::boolean IS TRUE
            AND COALESCE(pe.decision, '') IN ({decisions})
        )
    )"""


def training_decision_filter_sql(include_candle_baseline_param: str = "$5") -> str:
    """Return SQL predicate for decisions train.py may consume.

    Strategy signals use ``SHADOW_BASELINE`` / gate decisions. Candle baselines
    are only included when TRAIN_INCLUDE_CANDLE_BASELINE is enabled, matching
    ``training_strategy_filter_sql`` so diagnostics and training do not drift.
    """

    strategy_decisions = ", ".join(f"'{item}'" for item in _STRATEGY_BASELINE_DECISIONS)
    candle_decisions = ", ".join(f"'{item}'" for item in _CANDLE_BASELINE_DECISIONS)
    return f"""(
        COALESCE(pe.decision, '') IN ({strategy_decisions})
        OR (
            {include_candle_baseline_param}::boolean IS TRUE
            AND COALESCE(pe.decision, '') IN ({candle_decisions})
        )
    )"""
