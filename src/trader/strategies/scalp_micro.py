"""VWAP-Pullback micro-strategy with orderbook and multi-EMA confirmation.

Entry logic
-----------
BUY (long scalp):
  - ewma_tier_signal > threshold  → EMAs stacked bullishly
  - vwap_distance_pct in [-0.6, +0.2]  → price pulled back near VWAP, not free-falling
  - ob_imbalance_l5 > threshold  → buyers visible in L5 order book
  - macd_hist > 0  → momentum positive / recovering
  - rsi_14 in [0.32, 0.67]  → not overbought/oversold
  - adx_14 > threshold  → detectable trend (not pure noise)
  - volume_zscore > -1.2  → not a dead market
SELL: mirrored.

Economic basis: VWAP acts as an intraday anchor. When price pulls back toward
VWAP in the direction of the EMA stack with buy-side book pressure and
recovering MACD momentum, there is a measurable edge in the direction of
continuation after the pullback completes.

Exits
-----
  TP = entry +/- ATR(14) * 1.6   (reward:risk ~= 2.46)
  SL = entry -/+ ATR(14) * 0.65

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
from trader.strategies.base import BaseStrategy

log = structlog.get_logger(__name__)

_STRATEGY_ID = "scalp_micro_v1"

# === VWAP Pullback thresholds ===
_EWMA_THRESHOLD = 0.003          # multi-EMA bullish/bearish stack strength
_EWMA_THRESHOLD_SHADOW = 0.001   # relaxed for shadow data collection
_OB_IMBALANCE_THRESHOLD = 0.08   # L5 book imbalance required for direction confirmation
_OB_IMBALANCE_THRESHOLD_SHADOW = 0.05
_VWAP_BUY_LOW = -0.6             # BUY: price at most 0.6% below VWAP (not free-falling)
_VWAP_BUY_HIGH = 0.2             # BUY: price at most 0.2% above VWAP (no breakout chasing)
_VWAP_SELL_LOW = -0.2            # SELL: price at most 0.2% below VWAP
_VWAP_SELL_HIGH = 0.6            # SELL: price at most 0.6% above VWAP (pullback from above)
_RSI_BUY_MIN = 0.32              # BUY: not oversold
_RSI_BUY_MAX = 0.67              # BUY: not overbought
_RSI_SELL_MIN = 0.33             # SELL: not oversold
_RSI_SELL_MAX = 0.68             # SELL: not overbought
_ADX_THRESHOLD = 0.18            # normalised ADX (0.18 == ADX 18); flat markets rejected
_ADX_THRESHOLD_SHADOW = 0.14
_VOLUME_ZSCORE_MIN = -1.2        # reject completely dead markets
_TP_ATR_MULT = 1.6               # reward:risk ~= 2.46
_SL_ATR_MULT = 0.65
_MIN_ATR_PCT = 0.0005            # skip dead markets where TP would be inside the spread
_MAX_ATR_PCT = 0.03              # skip violent markets where SL gets blown through

_PRICE_DECIMALS = Decimal("0.00000001")


def _price(value: float) -> Decimal:
    return Decimal(str(value)).quantize(_PRICE_DECIMALS)


class ScalpMicroStrategy(BaseStrategy):
    """Cost-aware VWAP-pullback micro-scalping strategy.

    Signals come entirely from the FeatureVector (pipeline-computed). The
    candle_store argument is retained for interface compatibility but not read
    during evaluation.

    Args:
        candle_store:       Retained for interface compatibility; not used.
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
                            (e.g. "spread_rejected", "imbalance_missing").
        imbalance_provider: Retained for interface compatibility; not used.
                            OB confirmation comes from ob_imbalance_l5 feature.
        min_imbalance:      Retained for interface compatibility; not used.
        shadow_relaxed:     When True, loosens all thresholds and skips hard
                            net-edge / OB-data fails to maximise paper data volume.
    """

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
        imbalance_provider: Callable[[str], float | None] | None = None,
        min_imbalance: float = 0.15,
        shadow_relaxed: bool = False,
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
        self._imbalance_provider = imbalance_provider  # kept for interface compat
        self._min_imbalance = min_imbalance            # kept for interface compat
        self._shadow_relaxed = shadow_relaxed
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
                log.debug("scalp_micro.diag_hook_failed", reason=reason, error=str(exc))

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
        return gross_edge_pct - self._taker_fee_pct * 2 - spread_pct - self._expected_slippage_pct * 2

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
        ewma = f.get("ewma_tier_signal")
        vwap_dist = f.get("vwap_distance_pct")
        ob_imbalance = f.get("ob_imbalance_l5")
        ob_present = f.get("ob_data_present", 0.0)
        macd_hist = f.get("macd_hist")
        vol_z = f.get("volume_zscore")

        if rsi14 is None or atr_pct is None or ewma is None or vwap_dist is None:
            return None

        if atr_pct < _MIN_ATR_PCT or atr_pct > _MAX_ATR_PCT:
            return None

        adx_floor = _ADX_THRESHOLD_SHADOW if self._shadow_relaxed else _ADX_THRESHOLD
        if adx14 is None or adx14 < adx_floor:
            return None

        if vol_z is not None and vol_z < _VOLUME_ZSCORE_MIN:
            return None

        ewma_threshold = _EWMA_THRESHOLD_SHADOW if self._shadow_relaxed else _EWMA_THRESHOLD

        # --- Determine candidate direction ---
        buy_candidate = (
            ewma > ewma_threshold
            and _VWAP_BUY_LOW < vwap_dist < _VWAP_BUY_HIGH
            and _RSI_BUY_MIN < rsi14 < _RSI_BUY_MAX
            and (macd_hist is None or macd_hist > 0)
        )
        sell_candidate = (
            ewma < -ewma_threshold
            and _VWAP_SELL_LOW < vwap_dist < _VWAP_SELL_HIGH
            and _RSI_SELL_MIN < rsi14 < _RSI_SELL_MAX
            and (macd_hist is None or macd_hist < 0)
        )

        side: OrderSide | None = None
        if buy_candidate:
            side = OrderSide.BUY
        elif sell_candidate:
            side = OrderSide.SELL

        if side is None:
            return None

        # --- Orderbook imbalance confirmation from FeatureVector ---
        ob_threshold = _OB_IMBALANCE_THRESHOLD_SHADOW if self._shadow_relaxed else _OB_IMBALANCE_THRESHOLD
        if ob_present < 1.0:
            if self._shadow_relaxed:
                log.debug("scalp_micro.ob_missing_shadow", symbol=symbol)
            else:
                self._diag("imbalance_missing")
                return None
        else:
            imb = ob_imbalance if ob_imbalance is not None else 0.0
            confirms = (imb >= ob_threshold if side == OrderSide.BUY else imb <= -ob_threshold)
            if not confirms:
                if self._shadow_relaxed:
                    log.debug(
                        "scalp_micro.imbalance_skipped_shadow",
                        symbol=symbol,
                        side=side.value,
                        imbalance=round(imb, 3),
                    )
                else:
                    self._diag("imbalance_rejected")
                    return None

        # --- Spread filter (fail closed: unknown spread = no trade) ---
        spread_bps = self._spread_provider(symbol) if self._spread_provider is not None else None
        if spread_bps is None and self._shadow_relaxed:
            spread_bps = min(self._max_spread_bps, 4.0)
        if spread_bps is None or spread_bps > self._max_spread_bps:
            self._diag("spread_rejected")
            log.debug(
                "scalp_micro.spread_rejected",
                symbol=symbol,
                spread_bps=spread_bps,
                max_spread_bps=self._max_spread_bps,
            )
            return None

        # --- Net edge check: gross edge is the TP distance ---
        atr_abs = atr_pct * current_price
        gross_edge_pct = atr_pct * _TP_ATR_MULT * 100.0
        net_edge_pct = self._net_edge_pct(gross_edge_pct, spread_bps)
        if not self._shadow_relaxed and net_edge_pct < self._min_net_return_pct:
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
        if self._shadow_relaxed and net_edge_pct < self._min_net_return_pct:
            log.debug(
                "scalp_micro.net_edge_skipped_shadow",
                symbol=symbol,
                net_edge_pct=round(net_edge_pct, 4),
                min_required_pct=self._min_net_return_pct,
            )

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

        confidence = min(0.90, 0.55 + min(0.25, net_edge_pct / max(self._min_net_return_pct, 1e-9) * 0.05))

        self._register_signal(symbol)
        ob_imb_str = f"{ob_imbalance:.3f}" if ob_imbalance is not None else "N/A"
        log.info(
            "scalp_micro.signal",
            symbol=symbol,
            side=side.value,
            ewma=round(ewma, 5),
            vwap_dist_pct=round(vwap_dist, 3),
            ob_imbalance=ob_imb_str,
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
                f"vwap_pullback: ewma={ewma:.4f}, vwap_dist={vwap_dist:.2f}%, "
                f"ob_imb={ob_imb_str}, "
                f"spread={spread_bps:.1f}bps, net_edge={net_edge_pct:.3f}%"
            ),
        )
