"""Runtime compatibility shim — no longer active.

The legacy observational score alias and Canary block were needed while
``TradingApplication`` used a single ``ModelRegistry.score()`` call for both
shadow logging and live gating. After the direct Challenger / Champion split
in app.py (explicit ``score_shadow()`` / ``score_live()`` calls), these
monkey-patches are no longer applied.

This module is kept as a tombstone to preserve import compatibility in any
external code that references it. ``install_observational_score_alias()``
is now a no-op.
"""

from __future__ import annotations


def install_observational_score_alias() -> None:
    """No-op after direct score_shadow() / score_live() split in app.py."""
