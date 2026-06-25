from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from trader.storage.directional_trade_journal import DirectionalTradeJournal, _BaseTradeJournal


@pytest.mark.asyncio
async def test_lite_diagnostics_use_active_label_schema_and_mark_model_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = DirectionalTradeJournal("postgresql://example/db")
    journal._pool = object()  # type: ignore[assignment]
    monkeypatch.setattr(
        "trader.storage.directional_trade_journal._training_eligibility_params",
        lambda: (["scalp_micro_v1", "candle_sampler_v1"], True, "directional_net_v2", 2.0),
    )
    monkeypatch.setattr("trader.storage.directional_trade_journal._optional_settings", lambda: None)

    async def fake_base_diagnostics(*, lite: bool = False) -> dict[str, Any]:
        return {
            "connected": True,
            "lite": lite,
            "latest_model_version": {
                "version": "v2",
                "training_samples": 10_000,
                "metrics": {"label_schema_version": "directional_net_v2"},
            },
        }

    monkeypatch.setattr(_BaseTradeJournal, "get_db_diagnostics", AsyncMock(side_effect=fake_base_diagnostics))

    diag = await journal.get_db_diagnostics(lite=True)

    assert diag["label_schema_version"] == "directional_net_v2"
    assert diag["training_config"]["strategy_allowlist"] == [
        "scalp_micro_v1",
        "candle_sampler_v1",
    ]
    assert diag["latest_model_version"]["schema_compatible"] is True
    assert diag["latest_model_version"]["training_samples_compatible"] == 10_000
