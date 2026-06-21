"""Shared helpers for augmenting feature vectors with trade direction."""

from __future__ import annotations

from trader.domain.models import FeatureVector


def feature_values_for_side(vec: FeatureVector, side: str) -> tuple[list[str], list[float]]:
    """Return model features augmented with the proposed trade direction."""
    normalized_side = str(side).strip().lower()
    if normalized_side not in {"buy", "sell"}:
        raise ValueError(f"unsupported proposal side for ML features: {side!r}")
    by_name = dict(zip(vec.feature_names, vec.values, strict=True))
    by_name["proposal_side"] = 1.0 if normalized_side == "buy" else -1.0
    names = sorted(by_name.keys())
    values = [float(by_name[name]) for name in names]
    return names, values
