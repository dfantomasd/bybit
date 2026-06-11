"""Tests for the cost-aware micro-scalping strategy."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from trader.data.candles import Candle, CandleStore
from trader.domain.enums import OrderSide
from trader.domain.models import FeatureVector
from trader.strategies.scalp_micro import ScalpMicroStrategy

_SYMBOL = "TESTUSDT"
_INTERVAL = "1"


def _make_store(closes: list[float], volumes: list[float]) -> CandleStore:
    store = CandleStore(max_bars=500)
    base = datetime(2026, 6, 11, tzinfo=UTC)
    for i, (close, vol) in enumerate(zip(closes, volumes, strict=True)):
        prev = closes[i - 1] if i > 0 else close
        store.add(
            _SYMBOL,
            _INTERVAL,
            Candle(
                open_time=base + timedelta(minutes=i),
                open=prev,
                high=max(prev, close) * 1.001,
                low=min(prev, close) * 0.999,
                close=close,
                volume=vol,
                confirm=True,
            ),
        )
    return store


def _vector(rsi: float = 0.50, adx: float = 0.30, atr_pct: float = 0.004) -> FeatureVector:
    names = ["rsi_14", "adx_14", "atr_14_pct"]
    return FeatureVector(
        feature_id=uuid.uuid4(),
        symbol=_SYMBOL,
        timestamp=datetime.now(tz=UTC),
        values=[rsi, adx, atr_pct],
        feature_names=names,
        quality_score=1.0,
        lookback_bars=60,
    )


def _cross_up_data() -> tuple[list[float], list[float]]:
    """Closes engineered so EMA9 crosses above EMA21 on the last bar with a volume
    spike, while price stays within the bounce zone of the 5-bar low."""
    closes = [100.0 - i * 0.01 for i in range(40)]  # shallow downtrend (EMAs close)
    closes += [closes[-1] + 0.2 * (i + 1) for i in range(2)]  # fresh pop up
    volumes = [100.0] * (len(closes) - 1) + [500.0]  # impulse on last bar
    return closes, volumes


def _strategy(
    store: CandleStore,
    spread_bps: float | None = 1.0,
    taker_fee_pct: float = 0.055,
    **kwargs,
) -> ScalpMicroStrategy:
    rejections: list[str] = kwargs.pop("rejections", [])
    return ScalpMicroStrategy(
        candle_store=store,
        interval=_INTERVAL,
        spread_provider=lambda _s: spread_bps,
        taker_fee_pct=taker_fee_pct,
        expected_slippage_pct=0.01,
        min_net_return_pct=0.05,
        max_spread_bps=3.0,
        cooldown_seconds=60,
        max_trades_per_minute=10,
        max_position_notional_usd=100.0,
        diag_hook=rejections.append,
        **kwargs,
    )


class TestScalpMicroStrategy:
    def setup_method(self) -> None:
        ScalpMicroStrategy._global_signal_times.clear()

    def test_buy_signal_on_cross_up_with_impulse(self) -> None:
        closes, volumes = _cross_up_data()
        store = _make_store(closes, volumes)
        strat = _strategy(store)
        proposal = strat.evaluate(_vector(), closes[-1], 1000.0)
        assert proposal is not None
        assert proposal.side == OrderSide.BUY
        assert proposal.take_profit is not None and proposal.stop_loss is not None
        # TP distance must be ~2x SL distance (reward:risk 2:1)
        entry = float(proposal.entry_price)
        tp_dist = float(proposal.take_profit) - entry
        sl_dist = entry - float(proposal.stop_loss)
        assert tp_dist > 0 and sl_dist > 0
        assert abs(tp_dist / sl_dist - 2.0) < 0.05

    def test_wide_spread_rejected(self) -> None:
        closes, volumes = _cross_up_data()
        store = _make_store(closes, volumes)
        rejections: list[str] = []
        strat = _strategy(store, spread_bps=5.0, rejections=rejections)
        assert strat.evaluate(_vector(), closes[-1], 1000.0) is None
        assert "spread_rejected" in rejections

    def test_unknown_spread_fails_closed(self) -> None:
        closes, volumes = _cross_up_data()
        store = _make_store(closes, volumes)
        strat = _strategy(store, spread_bps=None)
        assert strat.evaluate(_vector(), closes[-1], 1000.0) is None

    def test_flat_market_adx_rejected(self) -> None:
        closes, volumes = _cross_up_data()
        store = _make_store(closes, volumes)
        strat = _strategy(store)
        assert strat.evaluate(_vector(adx=0.15), closes[-1], 1000.0) is None

    def test_net_edge_rejected_when_costs_exceed_edge(self) -> None:
        closes, volumes = _cross_up_data()
        store = _make_store(closes, volumes)
        rejections: list[str] = []
        # Round-trip fees 0.4% > gross TP edge 0.2% (ATR 0.4% * 0.5) → reject
        strat = _strategy(store, taker_fee_pct=0.2, rejections=rejections)
        assert strat.evaluate(_vector(), closes[-1], 1000.0) is None
        assert "scalp_net_edge_rejected" in rejections

    def test_no_volume_impulse_no_signal(self) -> None:
        closes, _ = _cross_up_data()
        volumes = [100.0] * len(closes)  # flat volume, no impulse
        store = _make_store(closes, volumes)
        strat = _strategy(store)
        assert strat.evaluate(_vector(), closes[-1], 1000.0) is None

    def test_symbol_cooldown(self) -> None:
        closes, volumes = _cross_up_data()
        store = _make_store(closes, volumes)
        strat = _strategy(store)
        first = strat.evaluate(_vector(), closes[-1], 1000.0)
        assert first is not None
        second = strat.evaluate(_vector(), closes[-1], 1000.0)
        assert second is None  # within cooldown

    def test_global_rate_limit(self) -> None:
        closes, volumes = _cross_up_data()
        store = _make_store(closes, volumes)
        strat = _strategy(store)
        now = datetime.now(tz=UTC)
        for _ in range(10):
            ScalpMicroStrategy._global_signal_times.append(now)
        assert strat.evaluate(_vector(), closes[-1], 1000.0) is None

    def test_notional_cap_applied(self) -> None:
        closes, volumes = _cross_up_data()
        store = _make_store(closes, volumes)
        strat = _strategy(store)
        proposal = strat.evaluate(_vector(), closes[-1], 100_000.0)
        assert proposal is not None
        assert float(proposal.requested_notional_usd) <= 100.0
