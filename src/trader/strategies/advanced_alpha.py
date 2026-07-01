"""Advanced alpha strategies adapted to the local strategy interface."""

from __future__ import annotations

import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import structlog

from trader.domain.enums import MarketRegime, MarketType, OrderSide
from trader.domain.models import FeatureVector, TradeProposal
from trader.risk.net_edge import NetEdgeParams, passes_min_net_edge
from trader.strategies.base import BaseStrategy

log = structlog.get_logger(__name__)
_PRICE_DECIMALS = Decimal("0.00000001")


def _features(vec: FeatureVector) -> dict[str, float]:
    return dict(zip(vec.feature_names, vec.values, strict=True))


def _price(value: float) -> Decimal:
    return Decimal(str(value)).quantize(_PRICE_DECIMALS)


def _reject(strategy_id: str, symbol: str, reason: str, **extra: Any) -> None:
    log.debug("advanced_alpha.rejected", strategy_id=strategy_id, symbol=symbol, reason=reason, **extra)


def _proposal(
    *,
    strategy_id: str,
    symbol: str,
    side: OrderSide,
    current_price: float,
    available_balance_usd: float,
    atr_pct: float,
    confidence: float,
    rationale: str,
    feature_id: UUID | None = None,
    risk_pct: float = 0.004,
    tp_mult: float = 1.4,
    sl_mult: float = 0.7,
    max_notional_usd: float = 100.0,
    cost_params: NetEdgeParams | None = None,
    min_net_return_pct: float = 0.08,
    spread_bps: float | None = None,
) -> TradeProposal | None:
    if current_price <= 0 or atr_pct <= 0:
        return None
    sl_dist = max(atr_pct * sl_mult, 0.001)
    tp_dist = max(atr_pct * tp_mult, sl_dist * 1.5)
    if cost_params is not None and not passes_min_net_edge(
        tp_dist,
        cost_params,
        min_net_return_pct,
        spread_bps=spread_bps,
    ):
        _reject(
            strategy_id,
            symbol,
            "net_edge_below_minimum",
            gross_tp_pct=round(tp_dist * 100.0, 4),
            min_net_return_pct=min_net_return_pct,
            spread_bps=spread_bps,
        )
        return None
    qty_usd = min(available_balance_usd * risk_pct / sl_dist, available_balance_usd * 0.20, max_notional_usd)
    if qty_usd < 5.0:
        return None
    qty = qty_usd / current_price
    if side == OrderSide.BUY:
        stop = current_price * (1 - sl_dist)
        take = current_price * (1 + tp_dist)
        regime = MarketRegime.BULL_TREND
    else:
        stop = current_price * (1 + sl_dist)
        take = current_price * (1 - tp_dist)
        regime = MarketRegime.BEAR_TREND
    return TradeProposal(
        proposal_id=uuid.uuid4(),
        strategy_id=strategy_id,
        symbol=symbol,
        market_type=MarketType.LINEAR,
        side=side,
        requested_qty=Decimal(str(round(qty, 4))),
        requested_notional_usd=Decimal(str(round(qty_usd, 2))),
        entry_price=_price(current_price),
        stop_loss=_price(stop),
        take_profit=_price(take),
        confidence=max(0.0, min(confidence, 0.95)),
        expected_return=tp_dist * 100.0,
        expected_risk=sl_dist * 100.0,
        spread_bps=spread_bps,
        regime=regime,
        feature_id=feature_id,
        rationale=rationale,
    )


def _cost_kwargs(
    strategy: Any,
    *,
    spread_bps: float | None = None,
) -> dict[str, Any]:
    return {
        "cost_params": strategy._cost_params,
        "min_net_return_pct": strategy._min_net_return_pct,
        "spread_bps": spread_bps,
    }


class OrderFlowStrategy(BaseStrategy):
    """Trade with aligned tape pressure, book imbalance, and microprice."""

    def __init__(
        self,
        flow_tracker: Any,
        orderbook_tracker: Any | None,
        min_flow_imbalance: float = 0.35,
        min_book_imbalance: float = 0.18,
        max_spread_bps: float = 3.0,
        cost_params: NetEdgeParams | None = None,
        min_net_return_pct: float = 0.08,
    ) -> None:
        self._flow = flow_tracker
        self._book = orderbook_tracker
        self._min_flow_imbalance = min_flow_imbalance
        self._min_book_imbalance = min_book_imbalance
        self._max_spread_bps = max_spread_bps
        self._cost_params = cost_params
        self._min_net_return_pct = min_net_return_pct

    @property
    def strategy_id(self) -> str:
        return "order_flow_v1"

    def evaluate(
        self, feature_vector: FeatureVector, current_price: float, available_balance_usd: float
    ) -> TradeProposal | None:
        f = _features(feature_vector)
        atr_pct = f.get("atr_14_pct")
        spread = f.get("spread_bps")
        if feature_vector.quality_score < 0.6 or atr_pct is None:
            _reject(self.strategy_id, feature_vector.symbol, "low_quality_or_missing_atr")
            return None
        if spread is not None and spread > self._max_spread_bps:
            _reject(self.strategy_id, feature_vector.symbol, "spread_too_wide", spread_bps=spread)
            return None
        stats = self._flow.trade_stats(feature_vector.symbol)
        if stats is None or stats.total_notional < 2_000:
            _reject(self.strategy_id, feature_vector.symbol, "insufficient_trade_flow")
            return None
        if self._book is None:
            _reject(self.strategy_id, feature_vector.symbol, "missing_orderbook_confirmation")
            return None
        try:
            book = self._book.latest_imbalance(feature_vector.symbol)
            micro = self._book.microprice_deviation_bps(feature_vector.symbol)
        except Exception as exc:
            _reject(self.strategy_id, feature_vector.symbol, "orderbook_confirmation_failed", error=str(exc))
            return None
        if book is None or micro is None:
            _reject(self.strategy_id, feature_vector.symbol, "stale_orderbook_confirmation")
            return None
        if stats.imbalance >= self._min_flow_imbalance:
            if book < self._min_book_imbalance:
                _reject(self.strategy_id, feature_vector.symbol, "book_not_confirming_buy", book_imbalance=book)
                return None
            if micro < 0:
                _reject(self.strategy_id, feature_vector.symbol, "microprice_not_confirming_buy", microprice_bps=micro)
                return None
            return _proposal(
                strategy_id=self.strategy_id,
                symbol=feature_vector.symbol,
                side=OrderSide.BUY,
                current_price=current_price,
                available_balance_usd=available_balance_usd,
                atr_pct=atr_pct,
                confidence=0.58 + min(0.2, abs(stats.imbalance) * 0.2),
                rationale=f"order flow buy pressure imbalance={stats.imbalance:.2f}",
                feature_id=feature_vector.feature_id,
                **_cost_kwargs(self, spread_bps=spread),
            )
        if stats.imbalance <= -self._min_flow_imbalance:
            if book > -self._min_book_imbalance:
                _reject(self.strategy_id, feature_vector.symbol, "book_not_confirming_sell", book_imbalance=book)
                return None
            if micro > 0:
                _reject(self.strategy_id, feature_vector.symbol, "microprice_not_confirming_sell", microprice_bps=micro)
                return None
            return _proposal(
                strategy_id=self.strategy_id,
                symbol=feature_vector.symbol,
                side=OrderSide.SELL,
                current_price=current_price,
                available_balance_usd=available_balance_usd,
                atr_pct=atr_pct,
                confidence=0.58 + min(0.2, abs(stats.imbalance) * 0.2),
                rationale=f"order flow sell pressure imbalance={stats.imbalance:.2f}",
                feature_id=feature_vector.feature_id,
                **_cost_kwargs(self, spread_bps=spread),
            )
        _reject(self.strategy_id, feature_vector.symbol, "flow_imbalance_below_threshold", imbalance=stats.imbalance)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# FUNDING RATE SQUEEZE  (v2)
# ──────────────────────────────────────────────────────────────────────────────

# Improved thresholds:
#   8 bps (was 5) — only genuinely extreme funding qualifies
#   OI bug fix: both sides confirm when OI is rising (more trapped positions)
#   RSI exhaustion filter added
#   R:R improved to ~2.7 (tp=1.5, sl=0.55) from 1.54 (tp=1.0, sl=0.65)
#   Volatility cap: skip panic markets (realized_vol > 2.5%)

_FA_MIN_FUNDING_BPS = 8.0  # absolute threshold; 5 was too noisy
_FA_MAX_FUNDING_BPS = 80.0  # clip to avoid stale/corrupt data
_FA_RSI_SELL_MIN = 0.60  # price elevated enough to fade
_FA_RSI_BUY_MAX = 0.40  # price depressed enough to fade
_FA_MAX_1BAR_RETURN = 0.004  # reject if still in explosive momentum
_FA_MAX_REALIZED_VOL = 0.025  # skip panic markets (> 2.5% vol)
_FA_OI_MIN_CHANGE = 0.0  # OI must be non-negative (positions building)


class FundingArbitrageStrategy(BaseStrategy):
    """Fade extreme funding rates when price momentum is exhausting.

    Setup:
      - Funding rate above threshold → too many longs (positive) or shorts (negative)
      - OI still rising → more trapped positioning = bigger eventual squeeze
      - RSI confirms the price is elevated / depressed (exhaustion zone)
      - 1-bar return small → momentum not still explosive (wait for slowdown)
      - Realized vol within bounds → no panic market

    Edge: perpetual funding creates a steady cost on the crowded side.
    When extreme funding persists, mean reversion typically occurs within
    1-4 hours as traders reduce the costly positions.
    """

    def __init__(
        self,
        min_abs_funding_bps: float = _FA_MIN_FUNDING_BPS,
        max_abs_return_1: float = _FA_MAX_1BAR_RETURN,
        cost_params: NetEdgeParams | None = None,
        min_net_return_pct: float = 0.08,
    ) -> None:
        self._min_abs_funding_bps = min_abs_funding_bps
        self._max_abs_return_1 = max_abs_return_1
        self._cost_params = cost_params
        self._min_net_return_pct = min_net_return_pct

    @property
    def strategy_id(self) -> str:
        return "funding_arbitrage_v1"

    def evaluate(
        self, feature_vector: FeatureVector, current_price: float, available_balance_usd: float
    ) -> TradeProposal | None:
        f = _features(feature_vector)
        atr_pct = f.get("atr_14_pct")
        # Prefer unclipped value for threshold detection; fall back to clipped
        funding = f.get("funding_rate_bps", f.get("funding_rate_bps_clipped"))
        oi_change = f.get("oi_change_pct_60m", f.get("oi_change_pct_60m_clipped"))
        r1 = f.get("log_return_1")
        rsi = f.get("rsi_14")
        vol = f.get("realized_vol_20")

        if feature_vector.quality_score < 0.6 or atr_pct is None or funding is None:
            _reject(self.strategy_id, feature_vector.symbol, "low_quality_or_missing_funding")
            return None
        if oi_change is None or r1 is None or rsi is None:
            _reject(self.strategy_id, feature_vector.symbol, "missing_oi_momentum_or_rsi")
            return None

        # Sanity bounds on funding (stale/corrupt data protection)
        if abs(funding) > _FA_MAX_FUNDING_BPS:
            _reject(self.strategy_id, feature_vector.symbol, "funding_exceeds_sanity_cap", funding_bps=funding)
            return None

        # Skip if funding is not extreme enough
        if abs(funding) < self._min_abs_funding_bps:
            _reject(self.strategy_id, feature_vector.symbol, "funding_not_extreme", funding_bps=funding)
            return None

        # Skip panic markets (realized vol too high)
        if vol is not None and vol > _FA_MAX_REALIZED_VOL:
            _reject(self.strategy_id, feature_vector.symbol, "panic_market_vol_too_high", realized_vol=vol)
            return None

        # Skip if momentum is still explosive (wait for it to slow)
        if abs(r1) > self._max_abs_return_1:
            _reject(self.strategy_id, feature_vector.symbol, "momentum_still_explosive", log_return_1=r1)
            return None

        side = OrderSide.SELL if funding > 0 else OrderSide.BUY

        # OI confirmation: both cases benefit from rising OI (more trapped positions)
        # Positive funding + OI rising = more longs entering = bigger eventual long squeeze
        # Negative funding + OI rising = more shorts entering = bigger eventual short squeeze
        if oi_change < _FA_OI_MIN_CHANGE:
            _reject(
                self.strategy_id,
                feature_vector.symbol,
                "oi_not_confirming_trapped_positioning",
                side=side.value,
                oi_change=oi_change,
                funding_bps=funding,
            )
            return None

        # RSI exhaustion filter: price should be elevated / depressed
        if side == OrderSide.SELL and rsi < _FA_RSI_SELL_MIN:
            _reject(self.strategy_id, feature_vector.symbol, "rsi_not_elevated_for_sell", rsi=rsi)
            return None
        if side == OrderSide.BUY and rsi > _FA_RSI_BUY_MAX:
            _reject(self.strategy_id, feature_vector.symbol, "rsi_not_depressed_for_buy", rsi=rsi)
            return None

        # Confidence scales with how extreme the funding rate is
        funding_intensity = min(1.0, (abs(funding) - self._min_abs_funding_bps) / 20.0)
        confidence = 0.58 + funding_intensity * 0.17

        return _proposal(
            strategy_id=self.strategy_id,
            symbol=feature_vector.symbol,
            side=side,
            current_price=current_price,
            available_balance_usd=available_balance_usd,
            atr_pct=atr_pct,
            confidence=confidence,
            rationale=(f"funding_squeeze: funding={funding:.2f}bps oi={oi_change:.3f} rsi={rsi:.2f} r1={r1:.4f}"),
            feature_id=feature_vector.feature_id,
            tp_mult=1.5,  # R:R ≈ 2.73
            sl_mult=0.55,
            **_cost_kwargs(self),
        )


# ──────────────────────────────────────────────────────────────────────────────
# LIQUIDATION CASCADE FADE  (v2)
# ──────────────────────────────────────────────────────────────────────────────

# Improved:
#   Per-symbol cooldown (was absent — double-fired on same cascade)
#   RSI extreme filter added (fade only when RSI at extreme)
#   Volume spike confirmation (volume_zscore > 1.5)
#   Raised thresholds: $50k (was $20k) and 0.72 imbalance (was 0.65)
#   R:R improved to ~2.77 (tp=1.8, sl=0.65) — counter-trend needs bigger TP
#   ATR cap: skip panic-spike markets

_LH_MIN_NOTIONAL = 50_000.0  # must be a real cascade, not minor noise
_LH_MIN_IMBALANCE = 0.72  # strong directional liquidation dominance
_LH_RSI_SELL_MIN = 0.70  # RSI overbought = longs got squeezed too far
_LH_RSI_BUY_MAX = 0.30  # RSI oversold = shorts got squeezed too far
_LH_VOLUME_SPIKE_MIN = 1.5  # volume z-score: volume must spike with cascade
_LH_MAX_ATR_PCT = 0.025  # skip extreme panic candles
_LH_COOLDOWN_SECONDS = 300  # 5 minutes between signals per symbol


class LiquidationHuntingStrategy(BaseStrategy):
    """Fade exhaustion after heavy one-sided liquidation cascades.

    Setup:
      - Large notional liquidated ($50k+) on one side (longs or shorts)
      - Imbalance > 0.72 → cascade is clearly one-directional
      - RSI at extreme confirming price overshot
      - Volume spike confirms panic (not quiet cascade)
      - Per-symbol cooldown prevents doubling into same cascade
      - ATR within bounds (not in outright panic market)

    Edge: forced liquidations overshoot fair value. Once the cascade
    exhausts the trapped side, price snaps back to the pre-cascade level.
    The fade captures the snap-back, not the cascade itself.
    """

    def __init__(
        self,
        flow_tracker: Any,
        min_liq_notional_usd: float = _LH_MIN_NOTIONAL,
        min_liq_imbalance: float = _LH_MIN_IMBALANCE,
        cooldown_seconds: int = _LH_COOLDOWN_SECONDS,
        cost_params: NetEdgeParams | None = None,
        min_net_return_pct: float = 0.08,
    ) -> None:
        self._flow = flow_tracker
        self._min_liq_notional_usd = min_liq_notional_usd
        self._min_liq_imbalance = min_liq_imbalance
        self._cooldown_seconds = cooldown_seconds
        self._cost_params = cost_params
        self._min_net_return_pct = min_net_return_pct
        self._last_signal_at: dict[str, datetime] = {}

    @property
    def strategy_id(self) -> str:
        return "liquidation_hunting_v1"

    def _rate_limited(self, symbol: str) -> bool:
        last = self._last_signal_at.get(symbol)
        if last is None:
            return False
        return (datetime.now(UTC) - last).total_seconds() < self._cooldown_seconds

    def evaluate(
        self, feature_vector: FeatureVector, current_price: float, available_balance_usd: float
    ) -> TradeProposal | None:
        symbol = feature_vector.symbol
        f = _features(feature_vector)
        atr_pct = f.get("atr_14_pct")
        rsi = f.get("rsi_14")
        vol_z = f.get("volume_zscore")

        if feature_vector.quality_score < 0.6 or atr_pct is None:
            _reject(self.strategy_id, symbol, "low_quality_or_missing_atr")
            return None

        if self._rate_limited(symbol):
            _reject(self.strategy_id, symbol, "cooldown_active")
            return None

        # Skip extreme panic markets
        if atr_pct > _LH_MAX_ATR_PCT:
            _reject(self.strategy_id, symbol, "atr_too_high_panic_market", atr_pct=atr_pct)
            return None

        liq = self._flow.liquidation_stats(symbol)
        if liq is None or liq.total_notional < self._min_liq_notional_usd:
            _reject(
                self.strategy_id, symbol, "insufficient_liquidation_notional", notional=liq.total_notional if liq else 0
            )
            return None

        # Volume must spike during the cascade
        if vol_z is not None and vol_z < _LH_VOLUME_SPIKE_MIN:
            _reject(self.strategy_id, symbol, "volume_not_spiking_during_cascade", vol_z=vol_z)
            return None

        if liq.imbalance >= self._min_liq_imbalance:
            # Shorts being liquidated → price spiked up → fade by SELLING
            if rsi is not None and rsi < _LH_RSI_SELL_MIN:
                _reject(self.strategy_id, symbol, "rsi_not_overbought_for_sell_fade", rsi=rsi)
                return None
            side = OrderSide.SELL
        elif liq.imbalance <= -self._min_liq_imbalance:
            # Longs being liquidated → price dropped too far → fade by BUYING
            if rsi is not None and rsi > _LH_RSI_BUY_MAX:
                _reject(self.strategy_id, symbol, "rsi_not_oversold_for_buy_fade", rsi=rsi)
                return None
            side = OrderSide.BUY
        else:
            _reject(self.strategy_id, symbol, "liquidation_imbalance_below_threshold", imbalance=liq.imbalance)
            return None

        # Confidence scales with cascade size and imbalance extremity
        size_factor = min(1.0, liq.total_notional / 200_000.0)
        imb_factor = min(1.0, (abs(liq.imbalance) - self._min_liq_imbalance) / 0.28)
        confidence = 0.60 + size_factor * 0.12 + imb_factor * 0.10

        self._last_signal_at[symbol] = datetime.now(UTC)
        return _proposal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            side=side,
            current_price=current_price,
            available_balance_usd=available_balance_usd,
            atr_pct=atr_pct,
            confidence=confidence,
            rationale=(
                f"liq_cascade_fade: total={liq.total_notional:.0f}$ "
                f"imb={liq.imbalance:.2f} rsi={f'{rsi:.2f}' if rsi is not None else 'N/A'}"
            ),
            feature_id=feature_vector.feature_id,
            tp_mult=1.8,  # R:R ≈ 2.77 (counter-trend needs bigger cushion)
            sl_mult=0.65,
            **_cost_kwargs(self),
        )


# ──────────────────────────────────────────────────────────────────────────────
# VOLATILITY SQUEEZE BREAKOUT  (new)
# ──────────────────────────────────────────────────────────────────────────────

# Uses Bollinger Band width compression as a squeeze detector:
#   Low bb_bandwidth = price ranges are compressed = energy building
#   bb_pct_b at extreme = price breaking out of the tight band
#   Volume spike = breakout confirmed by participation
#   Strong candle body = directional conviction (not wick-heavy)
#   MACD histogram direction = momentum agrees
#   ewma_tier_signal direction = EMA stack agrees

_VS_SQUEEZE_BW = 0.018  # bb_bandwidth below this = squeeze
_VS_BREAKOUT_HIGH = 0.82  # bb_pct_b above = upper band breakout
_VS_BREAKOUT_LOW = 0.18  # bb_pct_b below = lower band breakout
_VS_VOLUME_SPIKE_MIN = 1.0  # volume_zscore > 1 std above mean
_VS_BODY_RATIO_MIN = 0.58  # strong directional candle
_VS_ADX_MIN = 0.14  # some trend forming (very low bar)
_VS_ADX_MAX = 0.45  # skip established trends (already moved)
_VS_MIN_ATR_PCT = 0.0006
_VS_MAX_ATR_PCT = 0.022  # skip panic markets
_VS_COOLDOWN_SECONDS = 120  # 2 minutes between signals per symbol


class VolatilitySqueezeBreakoutStrategy(BaseStrategy):
    """Enter breakouts from Bollinger Band compression with volume confirmation.

    Setup:
      - bb_bandwidth below squeeze threshold → range contracted
      - bb_pct_b at extreme (>0.82 or <0.18) → price breaking out of the band
      - volume_zscore > 1.0 → above-average volume confirms the break
      - candle_body_ratio > 0.58 → directional candle, not wick-heavy indecision
      - MACD histogram + ewma_tier_signal agree with breakout direction
      - ADX in range: some trend forming but not already extended

    Edge: volatility compression always resolves with expansion. When price
    simultaneously compresses AND starts breaking out with volume, the
    directional move tends to persist for several candles before exhausting.
    This is the 'NR-breakout' pattern adapted for crypto perps.
    """

    def __init__(
        self,
        squeeze_bw_threshold: float = _VS_SQUEEZE_BW,
        breakout_high: float = _VS_BREAKOUT_HIGH,
        breakout_low: float = _VS_BREAKOUT_LOW,
        cooldown_seconds: int = _VS_COOLDOWN_SECONDS,
        cost_params: NetEdgeParams | None = None,
        min_net_return_pct: float = 0.08,
    ) -> None:
        self._squeeze_bw = squeeze_bw_threshold
        self._breakout_high = breakout_high
        self._breakout_low = breakout_low
        self._cooldown_seconds = cooldown_seconds
        self._cost_params = cost_params
        self._min_net_return_pct = min_net_return_pct
        self._last_signal_at: dict[str, datetime] = {}
        # Track recent bb_bandwidth per symbol to confirm genuine squeeze buildup
        self._bw_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=10))

    @property
    def strategy_id(self) -> str:
        return "volatility_squeeze_v1"

    def _rate_limited(self, symbol: str) -> bool:
        last = self._last_signal_at.get(symbol)
        if last is None:
            return False
        return (datetime.now(UTC) - last).total_seconds() < self._cooldown_seconds

    def evaluate(
        self, feature_vector: FeatureVector, current_price: float, available_balance_usd: float
    ) -> TradeProposal | None:
        symbol = feature_vector.symbol
        f = _features(feature_vector)
        atr_pct = f.get("atr_14_pct")
        bb_bw = f.get("bb_bandwidth")
        bb_pb = f.get("bb_pct_b")
        vol_z = f.get("volume_zscore")
        body = f.get("candle_body_ratio")
        macd_hist = f.get("macd_hist")
        ewma = f.get("ewma_tier_signal")
        adx = f.get("adx_14")

        if feature_vector.quality_score < 0.6 or atr_pct is None:
            _reject(self.strategy_id, symbol, "low_quality_or_missing_atr")
            return None
        if bb_bw is None or bb_pb is None:
            _reject(self.strategy_id, symbol, "missing_bollinger_features")
            return None

        if self._rate_limited(symbol):
            _reject(self.strategy_id, symbol, "cooldown_active")
            return None

        # ATR bounds: skip dead and panic markets
        if atr_pct < _VS_MIN_ATR_PCT or atr_pct > _VS_MAX_ATR_PCT:
            _reject(self.strategy_id, symbol, "atr_out_of_bounds", atr_pct=atr_pct)
            return None

        # ADX bounds: need some trend but not already extended
        if adx is not None and (adx < _VS_ADX_MIN or adx > _VS_ADX_MAX):
            _reject(self.strategy_id, symbol, "adx_out_of_bounds", adx=adx)
            return None

        # Track bandwidth history to confirm squeeze buildup
        history = self._bw_history[symbol]
        prior = list(history)  # snapshot before appending current bar
        history.append(bb_bw)

        # Require squeeze: current AND recent bandwidth low
        if bb_bw > self._squeeze_bw:
            _reject(self.strategy_id, symbol, "no_squeeze_bandwidth_too_wide", bb_bandwidth=bb_bw)
            return None
        # Verify sustained squeeze in prior bars (checked before current append to avoid
        # always-false: current bar already passed the squeeze guard above)
        if len(prior) >= 3 and min(prior[-3:]) > self._squeeze_bw * 1.5:
            _reject(self.strategy_id, symbol, "no_sustained_squeeze_in_history")
            return None

        # Determine breakout direction from bb_pct_b
        if bb_pb > self._breakout_high:
            side = OrderSide.BUY
        elif bb_pb < self._breakout_low:
            side = OrderSide.SELL
        else:
            _reject(self.strategy_id, symbol, "no_band_breakout", bb_pct_b=bb_pb)
            return None

        # Volume must be above average on the breakout
        if vol_z is not None and vol_z < _VS_VOLUME_SPIKE_MIN:
            _reject(self.strategy_id, symbol, "volume_not_spiking_on_breakout", vol_z=vol_z)
            return None

        # Candle body confirms direction (not wick reversal)
        if body is not None and abs(body) < _VS_BODY_RATIO_MIN:
            _reject(self.strategy_id, symbol, "weak_candle_body", body_ratio=body)
            return None
        if body is not None:
            body_direction_ok = body > 0 if side == OrderSide.BUY else body < 0
            if not body_direction_ok:
                _reject(self.strategy_id, symbol, "candle_body_against_breakout", body=body)
                return None

        # MACD histogram direction must agree
        if macd_hist is not None and (
            (side == OrderSide.BUY and macd_hist <= 0) or (side == OrderSide.SELL and macd_hist >= 0)
        ):
            _reject(self.strategy_id, symbol, "macd_against_breakout", macd_hist=macd_hist)
            return None

        # EMA stack direction must agree (or neutral)
        if ewma is not None and (
            (side == OrderSide.BUY and ewma < -0.001) or (side == OrderSide.SELL and ewma > 0.001)
        ):
            _reject(self.strategy_id, symbol, "ewma_against_breakout", ewma=ewma)
            return None

        # Confidence scales with how far into the band the price has broken
        if side == OrderSide.BUY:
            breakout_intensity = min(1.0, (bb_pb - self._breakout_high) / 0.15)
        else:
            breakout_intensity = min(1.0, (self._breakout_low - bb_pb) / 0.15)
        vol_boost = min(0.10, max(0.0, (vol_z or 0.0) - _VS_VOLUME_SPIKE_MIN) * 0.03)
        confidence = 0.58 + breakout_intensity * 0.16 + vol_boost

        self._last_signal_at[symbol] = datetime.now(UTC)
        return _proposal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            side=side,
            current_price=current_price,
            available_balance_usd=available_balance_usd,
            atr_pct=atr_pct,
            confidence=confidence,
            rationale=(
                f"bb_squeeze_breakout: bw={bb_bw:.4f} pct_b={bb_pb:.2f} "
                f"vol_z={(vol_z or 0.0):.2f} body={(body or 0.0):.2f}"
            ),
            feature_id=feature_vector.feature_id,
            tp_mult=1.6,  # R:R ≈ 2.46
            sl_mult=0.65,
            **_cost_kwargs(self),
        )


class MarketMakingStrategy(BaseStrategy):
    """Maker-first mean reversion proxy for the current single-order engine."""

    def __init__(
        self,
        spread_provider: Callable[[str], float | None],
        min_spread_bps: float = 1.2,
        max_spread_bps: float = 4.0,
        cost_params: NetEdgeParams | None = None,
        min_net_return_pct: float = 0.08,
    ) -> None:
        self._spread_provider = spread_provider
        self._min_spread_bps = min_spread_bps
        self._max_spread_bps = max_spread_bps
        self._cost_params = cost_params
        self._min_net_return_pct = min_net_return_pct

    @property
    def strategy_id(self) -> str:
        return "market_making_v1"

    def evaluate(
        self, feature_vector: FeatureVector, current_price: float, available_balance_usd: float
    ) -> TradeProposal | None:
        f = _features(feature_vector)
        atr_pct = f.get("atr_14_pct")
        rsi = f.get("rsi_14")
        r1 = f.get("log_return_1", 0.0) or 0.0
        spread = self._spread_provider(feature_vector.symbol)
        if feature_vector.quality_score < 0.6 or atr_pct is None or rsi is None or spread is None:
            _reject(self.strategy_id, feature_vector.symbol, "low_quality_or_missing_spread")
            return None
        if spread < self._min_spread_bps or spread > self._max_spread_bps:
            _reject(self.strategy_id, feature_vector.symbol, "spread_outside_maker_band", spread_bps=spread)
            return None
        if rsi < 0.35 and r1 < 0:
            side = OrderSide.BUY
        elif rsi > 0.65 and r1 > 0:
            side = OrderSide.SELL
        else:
            _reject(self.strategy_id, feature_vector.symbol, "mean_reversion_setup_absent", rsi=rsi, log_return_1=r1)
            return None
        return _proposal(
            strategy_id=self.strategy_id,
            symbol=feature_vector.symbol,
            side=side,
            current_price=current_price,
            available_balance_usd=available_balance_usd,
            atr_pct=atr_pct,
            confidence=0.56,
            rationale=f"market making mean-reversion proxy spread={spread:.2f}bps rsi={rsi:.2f} r1={r1:.4f}",
            feature_id=feature_vector.feature_id,
            risk_pct=0.0025,
            tp_mult=0.80,
            sl_mult=0.50,
            max_notional_usd=50.0,
            **_cost_kwargs(self, spread_bps=spread),
        )


class StatisticalArbitrageStrategy(BaseStrategy):
    """Z-score mean reversion using the symbol's own short/medium returns."""

    def __init__(
        self,
        min_zscore: float = 2.0,
        max_adx: float = 0.35,
        cost_params: NetEdgeParams | None = None,
        min_net_return_pct: float = 0.08,
    ) -> None:
        self._min_zscore = min_zscore
        self._max_adx = max_adx
        self._cost_params = cost_params
        self._min_net_return_pct = min_net_return_pct

    @property
    def strategy_id(self) -> str:
        return "statistical_arbitrage_v1"

    def evaluate(
        self, feature_vector: FeatureVector, current_price: float, available_balance_usd: float
    ) -> TradeProposal | None:
        f = _features(feature_vector)
        atr_pct = f.get("atr_14_pct")
        r1 = f.get("log_return_1")
        vol = f.get("realized_vol_20")
        adx = f.get("adx_14", 0.0) or 0.0
        if feature_vector.quality_score < 0.6 or atr_pct is None or r1 is None:
            _reject(self.strategy_id, feature_vector.symbol, "low_quality_or_missing_returns")
            return None
        if vol is None or vol <= 0 or adx > self._max_adx:
            _reject(self.strategy_id, feature_vector.symbol, "volatility_or_adx_filter", volatility=vol, adx=adx)
            return None
        z = r1 / max(vol, 1e-9)
        if z >= self._min_zscore:
            side = OrderSide.SELL
        elif z <= -self._min_zscore:
            side = OrderSide.BUY
        else:
            _reject(self.strategy_id, feature_vector.symbol, "zscore_below_threshold", zscore=z)
            return None
        return _proposal(
            strategy_id=self.strategy_id,
            symbol=feature_vector.symbol,
            side=side,
            current_price=current_price,
            available_balance_usd=available_balance_usd,
            atr_pct=atr_pct,
            confidence=0.56 + min(0.16, abs(z) * 0.03),
            rationale=f"stat-arb z-score mean reversion z={z:.2f}",
            feature_id=feature_vector.feature_id,
            tp_mult=0.9,
            sl_mult=0.65,
            **_cost_kwargs(self),
        )
