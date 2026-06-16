"""Guard against the shielded-event-wait task leak pattern.

``await asyncio.wait_for(asyncio.shield(event.wait()), timeout=...)`` leaks one
pending task plus one Event waiter on EVERY timeout: shield protects the inner
``wait()`` task from the timeout cancellation, so it stays alive (and in the
event's waiter list) until the event is finally set. In periodic loops this
grew by >15k tasks/day. A plain ``wait_for(event.wait(), timeout)`` is correct:
cancelling ``Event.wait()`` is safe and removes the waiter.
"""

from __future__ import annotations

import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "trader"

_LEAK_PATTERN = re.compile(r"asyncio\.shield\(\s*self\._\w*event\.wait\(\)\s*\)")


def test_no_shielded_event_wait_in_source() -> None:
    offenders: list[str] = []
    for path in _SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in _LEAK_PATTERN.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(_SRC.parent.parent)}:{line}")
    assert not offenders, (
        "shielded Event.wait() leaks a pending task per loop iteration; "
        f"use await asyncio.wait_for(event.wait(), timeout=...) instead: {offenders}"
    )


def test_wait_for_event_wait_does_not_leak() -> None:
    """Empirical check of the fixed pattern: no waiters accumulate."""

    import asyncio

    async def scenario() -> int:
        event = asyncio.Event()
        for _ in range(5):
            try:
                await asyncio.wait_for(event.wait(), timeout=0.001)
            except TimeoutError:
                pass
        await asyncio.sleep(0)
        return len(event._waiters)  # noqa: SLF001 - intentional internals check

    assert asyncio.run(scenario()) == 0
