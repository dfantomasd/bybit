"""Application-level multi-timeframe market data wiring tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from trader.app import TradingApplication


class _Secret:
    def get_secret_value(self) -> str:
        return "configured"


@pytest.mark.asyncio
async def test_seed_candle_store_fetches_and_persists_configured_intervals() -> None:
    app = TradingApplication()
    app._settings = SimpleNamespace(
        MULTITIMEFRAME_ENABLED=True,
        MULTITIMEFRAME_INTERVALS=["1", "5", "15", "60"],
        BYBIT_API_KEY=_Secret(),
        CANDLE_STORE_MAX_BARS_1M=250,
        CANDLE_STORE_MAX_BARS_5M=250,
        CANDLE_STORE_MAX_BARS_15M=200,
        CANDLE_STORE_MAX_BARS_1H=120,
    )

    rest = SimpleNamespace(
        get_kline=AsyncMock(
            return_value={
                "result": {
                    "list": [
                        [
                            "1700000000000",
                            "1.0",
                            "1.1",
                            "0.9",
                            "1.05",
                            "1000",
                            "1050",
                        ]
                    ]
                }
            }
        )
    )
    app._bybit_adapter = SimpleNamespace(_rest=rest)
    app._trade_journal = SimpleNamespace(is_enabled=True, upsert_market_candle=AsyncMock())

    await app._seed_candle_store(symbols=["DOGEUSDT"])

    fetched_intervals = [call.kwargs["interval"] for call in rest.get_kline.await_args_list]
    persisted_intervals = [call.kwargs["interval"] for call in app._trade_journal.upsert_market_candle.await_args_list]

    assert fetched_intervals == ["1", "5", "15", "60"]
    assert persisted_intervals == ["1", "5", "15", "60"]


def test_market_data_intervals_keep_1m_first_and_deduplicate() -> None:
    app = TradingApplication()
    app._settings = SimpleNamespace(MULTITIMEFRAME_ENABLED=True, MULTITIMEFRAME_INTERVALS=["5", "1", "15", "15", "60"])

    assert app._market_data_intervals() == ["1", "5", "15", "60"]
