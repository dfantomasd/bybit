"""Tests for WS subscribe watchdog."""

from __future__ import annotations

import time

from trader.features.subscribe_watchdog import SubscribeWatchdog


def test_register_and_confirm_clears_pending() -> None:
    wd = SubscribeWatchdog(timeout_s=10.0)
    wd.register(["BTCUSDT", "ETHUSDT"])
    assert set(wd.pending_symbols()) == {"BTCUSDT", "ETHUSDT"}
    wd.confirm_ws_kline("BTCUSDT", "1")
    assert wd.pending_symbols() == ["ETHUSDT"]


def test_non_1m_kline_does_not_confirm() -> None:
    wd = SubscribeWatchdog(timeout_s=10.0)
    wd.register(["BTCUSDT"])
    wd.confirm_ws_kline("BTCUSDT", "5")
    assert wd.pending_symbols() == ["BTCUSDT"]


def test_expired_symbols_after_timeout() -> None:
    wd = SubscribeWatchdog(timeout_s=5.0)
    start = time.monotonic()
    wd.register(["SOLUSDT"], now=start)
    assert wd.expired(now=start + 4.0) == []
    assert wd.expired(now=start + 6.0) == ["SOLUSDT"]


def test_mark_retry_forces_reconnect_after_max_retries() -> None:
    wd = SubscribeWatchdog(timeout_s=5.0, max_retries=3)
    wd.register(["ADAUSDT"])
    assert wd.mark_retry("ADAUSDT") is False
    assert wd.mark_retry("ADAUSDT") is False
    assert wd.mark_retry("ADAUSDT") is True
    assert wd.reconnects_total == 1
