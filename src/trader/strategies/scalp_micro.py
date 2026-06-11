"""Micro-scalping strategy with hard cost control.

Designed for high-frequency small-edge entries where fees, spread and slippage
dominate. Every signal must clear an explicit net-edge check: expected move
minus round-trip taker fees, spread and slippage must exceed
``min_net_return_pct`` or the signal is dropped.

Entry logic
-----------
BUY:
  - EMA9 crosses above EMA21 on the last confirmed bar (fresh cross only)
  - RSI14 < 70 (not overbought)
  - Volume impulse: last volume > 1.5 x SMA(volume, 20)
  - Spread <= max_spread_bps (default 3 bps)
  - Price bouncing from the low of the last 5 candles
  - ADX14 >= 20 (no flat market)
SELL: mirrored.

Exits (fixed, no trailing)
--------------------------
  TP = entry +/- ATR(14) * 0.5
  SL = entry -/+ ATR(14) * 0.25   (reward:risk ~= 2:1)

Rate limits
-----------
  - Per-symbol cooldown (default 60 s)
  - Global cap across all symbols (default 10 signals/minute), shared via a
    class-level deque so multiple instances cannot exceed it together.
"""

from __future__ import annotations

import uuid
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, ClassVar

import structlog

from trader.domain.enums import MarketRegime, MarketType, OrderSide
from trader.domain.models import FeatureVector, TradeProposal
from trader.features.technical import ema
from trader.strategies.base import BaseStrategy

log = structlog.get_logger(__name__)

_STRATEGY_ID = "scalp_micro_v1"

_EMA_FAST = 9
_EMA_SLOW = 21
_RSI_OVERBOUGHT = 0.70  # rsi_14 feature is normalised to [0, 1]
_RSI_OVERSOLD = 0.30
_ADX_FLAT = 0.20  # adx_14 feature is normalised to [0, 1]; 0.20 == ADX 20
_VOLUME_IMPULSE_MULT = 1.5
_BOUNCE_LOOKBACK = 5
_BOUNCE_ZONE_ATR_MULT = 0.35  # price must be within this many ATRs of the extreme
_TP_ATR_MULT = 0.5
_SL_ATR_MULT = 0.25
_MIN_ATR_PCT = 0.0005  # skip dead markets: TP would be inside the spread
_MAX_ATR_PCT = 0.03  # skip violent markets: SL gets blown through

_PRICE_DECIMALS = Decimal("0.00000001")


def _price(value: float) -> Decimal:
    return Decimal(str(value)).quantize(_PRICE_DECIMALS)


class ScalpMicroStrategy(BaseStrategy):
    """Cost-aware micro-scalping strategy.

    Args:
        candle_store:       CandleStore for closes/highs/lows/volumes series.
        interval:           Candle interval the strategy operates on (e.g. "1").
        spread_provider:    Callable(symbol) -> current spread in bps, or None
                            when unknown. Unknown spread fails closed (no signal).
        taker_fee_pct:      One-way taker fee in percent (0.055 for 0.055%).
        expected_slippage_pct: Expected slippage in percent per round trip.
        min_net_return_pct: Minimum expected NET return in percent after all
                            costs; signals below this are dropped.
        max_spread_bps:     Maximum allowed bid-ask spread in bps.
        cooldown_seconds:   Minimum seconds between signals per symbol.
        max_trades_per_minute: Global cap across all symbols and instances.
        risk_pct:           Fraction of balance risked per trade (0.01 = 1%).
        max_position_notional_usd: Hard notional cap per position.
        min_qty_usd:        Exchange minimum notional.
        diag_hook:          Optional callable(reason) for rejection diagnostics
                            (e.g. "spread_rejected", "scalp_net_edge_rejected").
    """

    # Shared across all instances: global trade-rate governor
    _global_signal_times: ClassVar[deque[datetime]] = deque(maxlen=256)

    def __init__(
        self,
        candle_store: Any,
        interval: str = "1",
        spread_provider: Callable[[str], float | None] | None = None,
        taker_fee_pct: float = 0.055,
        expected_slippage_pct: float = 0.03,
        min_net_return_pct: float = 0.05,
        max_spread_bps: float = 3.0,
        cooldown_seconds: int = 60,
        max_trades_per_minute: int = 10,
        risk_pct: float = 0.01,
        max_position_notional_usd: float = 100.0,
        min_qty_usd: float = 5.0,
        diag_hook: Callable[[str], None] | None = None,
    ) -> None:
        self._store = candle_store
        self._interval = interval
        self._spread_provider = spread_provider
        self._taker_fee_pct = taker_fee_pct
        self._expected_slippage_pct = expected_slippage_pct
        self._min_net_return_pct = min_net_return_pct
        self._max_spread_bps = max_spread_bps
        self._cooldown_seconds = cooldown_seconds
        self._max_trades_per_minute = max_trades_per_minute
        self._risk_pct = risk_pct
        self._max_position_notional_usd = max_position_notional_usd
        self._min_qty_usd = min_qty_usd
        self._diag_hook = diag_hook
        self._last_signal_at: dict[str, datetime] = {}

    @property
    def strategy_id(self) -> str:
        return _STRATEGY_ID

    # ------------------------------------------------------------------

    def _diag(self, reason: str) -> None:
        if self._diag_hook is not None:
            try:
                self._diag_hook(reason)
            except Exception as exc:
                log.debug("scalp_micro.diag_hook_failed", error=str(exc))

    def _rate_limited(self, symbol: str) -> bool:
        now = datetime.now(tz=UTC)
        last = self._last_signal_at.get(symbol)
        if last is not None and (now - last).total_seconds() < self._cooldown_seconds:
            return True
        recent = [t for t in self._global_signal_times if (now - t).total_seconds() < 60]
        return len(recent) >= self._max_trades_per_minute

    def _register_signal(self, symbol: str) -> None:
        now = datetime.now(tz=UTC)
        self._last_signal_at[symbol] = now
        self._global_signal_times.append(now)

    def _net_edge_pct(self, gross_edge_pct: float, spread_bps: float) -> float:
        """Expected NET return in percent after round-trip costs."""
        spread_pct = spread_bps / 100.0  # bps -> percent
        return gross_edge_pct - self._taker_fee_pct * 2 - spread_pct - self._expected_slippage_pct

    # ------------------------------------------------------------------

    def evaluate(
        self,
        feature_vector: FeatureVector,
        current_price: float,
        available_balance_usd: float,
    ) -> TradeProposal | None:
        symbol = feature_vector.symbol
        if current_price <= 0 or feature_vector.quality_score < 0.6:
            return None

        if self._rate_limited(symbol):
            return None

        f = dict(zip(feature_vector.feature_names, feature_vector.values, strict=True))
        rsi14 = f.get("rsi_14")
        adx14 = f.get("adx_14")
        atr_pct = f.get("atr_14_pct")
        if rsi14 is None or atr_pct is None:
            return None

        # Flat-market filter: no scalping when there is no movement to capture
        if adx14 is None or adx14 < _ADX_FLAT:
            return None

        if atr_pct < _MIN_ATR_PCT or atr_pct > _MAX_ATR_PCT:
            return None

        closes = list(self._store.closes(symbol, self._interval))
        highs = list(self._store.highs(symbol, self._interval))
        lows = list(self._store.lows(symbol, self._interval))
        volumes = list(self._store.volumes(symbol, self._interval))
        if len(closes) < _EMA_SLOW + 2 or len(volumes) < 21 or len(lows) < _BOUNCE_LOOKBACK + 1:
            return None

        # --- EMA cross on the last confirmed bar (fresh cross only) ---
        ema_fast = ema(closes, _EMA_FAST)
        ema_slow = ema(closes, _EMA_SLOW)
        if len(ema_fast) < 2 or len(ema_slow) < 2:
            return None
        fast_now, fast_prev = ema_fast[-1], ema_fast[-2]
        slow_now, slow_prev = ema_slow[-1], ema_slow[-2]
        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now
        if not crossed_up and not crossed_down:
            return None

        # --- Volume impulse ---
        vol_sma20 = sum(volumes[-21:-1]) / 20 if len(volumes) >= 21 else 0.0
        if vol_sma20 <= 0 or volumes[-1] < _VOLUME_IMPULSE_MULT * vol_sma20:
            return None

        # --- Spread filter (fail closed: unknown spread = no trade) ---
        spread_bps = self._spread_provider(symbol) if self._spread_provider is not None else None
        if spread_bps is None or spread_bps > self._max_spread_bps:
            self._diag("spread_rejected")
            log.debug(
                "scalp_micro.spread_rejected",
                symbol=symbol,
                spread_bps=spread_bps,
                max_spread_bps=self._max_spread_bps,
            )
            return None

        atr_abs = atr_pct * current_price
        bounce_zone = atr_abs * _BOUNCE_ZONE_ATR_MULT

        side: OrderSide | None = None
        if crossed_up and rsi14 < _RSI_OVERBOUGHT:
            # Bounce from the low of the last 5 candles
            low5 = min(lows[-_BOUNCE_LOOKBACK:])
            if current_price > low5 and (current_price - low5) <= bounce_zone + atr_abs:
                side = OrderSide.BUY
        elif crossed_down and rsi14 > _RSI_OVERSOLD:
            high5 = max(highs[-_BOUNCE_LOOKBACK:])
            if current_price < high5 and (high5 - current_price) <= bounce_zone + atr_abs:
                side = OrderSide.SELL
        if side is None:
            return None

        # --- Net edge check: gross edge is the TP distance ---
        gross_edge_pct = atr_pct * _TP_ATR_MULT * 100.0
        net_edge_pct = self._net_edge_pct(gross_edge_pct, spread_bps)
        if net_edge_pct < self._min_net_return_pct:
            self._diag("scalp_net_edge_rejected")
            log.debug(
                "scalp_micro.net_edge_rejected",
                symbol=symbol,
                gross_edge_pct=round(gross_edge_pct, 4),
                net_edge_pct=round(net_edge_pct, 4),
                min_required_pct=self._min_net_return_pct,
                spread_bps=spread_bps,
            )
            return None

        # --- Position sizing with notional cap ---
        sl_dist_pct = atr_pct * _SL_ATR_MULT
        if sl_dist_pct <= 0:
            return None
        qty_usd = available_balance_usd * self._risk_pct / sl_dist_pct
        qty_usd = min(qty_usd, self._max_position_notional_usd, available_balance_usd * 0.30)
        if qty_usd < self._min_qty_usd:
            return None
        qty = qty_usd / current_price

        entry = current_price
        if side == OrderSide.BUY:
            tp = entry + atr_abs * _TP_ATR_MULT
            sl = entry - atr_abs * _SL_ATR_MULT
            regime = MarketRegime.BULL_TREND
        else:
            tp = entry - atr_abs * _TP_ATR_MULT
            sl = entry + atr_abs * _SL_ATR_MULT
            regime = MarketRegime.BEAR_TREND

        # Confidence scales with how much net edge clears the minimum
        confidence = min(0.90, 0.55 + min(0.25, net_edge_pct / max(self._min_net_return_pct, 1e-9) * 0.05))

        self._register_signal(symbol)
        log.info(
            "scalp_micro.signal",
            symbol=symbol,
            side=side.value,
            net_edge_pct=round(net_edge_pct, 4),
            spread_bps=round(spread_bps, 2),
            atr_pct=round(atr_pct, 5),
            qty_usd=round(qty_usd, 2),
        )

        return TradeProposal(
            proposal_id=uuid.uuid4(),
            strategy_id=_STRATEGY_ID,
            symbol=symbol,
            market_type=MarketType.LINEAR,
            side=side,
            requested_qty=Decimal(str(round(qty, 4))),
            requested_notional_usd=Decimal(str(round(qty_usd, 2))),
            entry_price=_price(entry),
            stop_loss=_price(sl),
            take_profit=_price(tp),
            confidence=confidence,
            regime=regime,
            rationale=(
                f"scalp: EMA{_EMA_FAST}x{_EMA_SLOW} cross, vol impulse, "
                f"spread={spread_bps:.1f}bps, net_edge={net_edge_pct:.3f}%"
            ),
        )
