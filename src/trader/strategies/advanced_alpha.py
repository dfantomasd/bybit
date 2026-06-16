"""Advanced alpha strategies adapted to the local strategy interface."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from decimal import Decimal
from typing import Any
from uuid import UUID

import structlog

from trader.domain.enums import MarketRegime, MarketType, OrderSide
from trader.domain.models import FeatureVector, TradeProposal
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
) -> TradeProposal | None:
    if current_price <= 0 or atr_pct <= 0:
        return None
    sl_dist = max(atr_pct * sl_mult, 0.001)
    tp_dist = max(atr_pct * tp_mult, sl_dist * 1.5)
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
        regime=regime,
        feature_id=feature_id,
        rationale=rationale,
    )

class OrderFlowStrategy(BaseStrategy):
    """Trade with aligned tape pressure, book imbalance, and microprice."""

    def __init__(
        self,
        flow_tracker: Any,
        orderbook_tracker: Any | None,
        min_flow_imbalance: float = 0.35,
        min_book_imbalance: float = 0.18,
        max_spread_bps: float = 3.0,
    ) -> None:
        self._flow = flow_tracker
        self._book = orderbook_tracker
        self._min_flow_imbalance = min_flow_imbalance
        self._min_book_imbalance = min_book_imbalance
        self._max_spread_bps = max_spread_bps

    @property
    def strategy_id(self) -> str:
        return "order_flow_v1"

    def evaluate(self, feature_vector: FeatureVector, current_price: float, available_balance_usd: float) -> TradeProposal | None:
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
        book = self._book.latest_imbalance(feature_vector.symbol) if self._book is not None else None
        micro = self._book.microprice_deviation_bps(feature_vector.symbol) if self._book is not None else None
        if stats.imbalance >= self._min_flow_imbalance:
            if book is not None and book < self._min_book_imbalance:
                _reject(self.strategy_id, feature_vector.symbol, "book_not_confirming_buy", book_imbalance=book)
                return None
            if micro is not None and micro < 0:
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
            )
        if stats.imbalance <= -self._min_flow_imbalance:
            if book is not None and book > -self._min_book_imbalance:
                _reject(self.strategy_id, feature_vector.symbol, "book_not_confirming_sell", book_imbalance=book)
                return None
            if micro is not None and micro > 0:
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
            )
        _reject(self.strategy_id, feature_vector.symbol, "flow_imbalance_below_threshold", imbalance=stats.imbalance)
        return None


class FundingArbitrageStrategy(BaseStrategy):
    """Position against extreme funding when price momentum is exhausted."""

    def __init__(self, min_abs_funding_bps: float = 5.0, max_abs_return_3: float = 0.003) -> None:
        self._min_abs_funding_bps = min_abs_funding_bps
        self._max_abs_return_3 = max_abs_return_3

    @property
    def strategy_id(self) -> str:
        return "funding_arbitrage_v1"

    def evaluate(self, feature_vector: FeatureVector, current_price: float, available_balance_usd: float) -> TradeProposal | None:
        f = _features(feature_vector)
        atr_pct = f.get("atr_14_pct")
        funding = f.get("funding_rate_bps_clipped", f.get("funding_rate_bps"))
        oi_change = f.get("oi_change_pct_60m_clipped", f.get("oi_change_pct_60m", 0.0)) or 0.0
        r3 = f.get("return_3", 0.0) or 0.0
        if feature_vector.quality_score < 0.6 or atr_pct is None or funding is None:
            _reject(self.strategy_id, feature_vector.symbol, "low_quality_or_missing_funding")
            return None
        if abs(funding) < self._min_abs_funding_bps or abs(r3) > self._max_abs_return_3:
            _reject(self.strategy_id, feature_vector.symbol, "funding_or_momentum_not_extreme", funding_bps=funding, return_3=r3)
            return None
        side = OrderSide.SELL if funding > 0 else OrderSide.BUY
        if side == OrderSide.SELL and oi_change < 0:
            _reject(self.strategy_id, feature_vector.symbol, "oi_not_confirming_positive_funding_fade", oi_change=oi_change)
            return None
        if side == OrderSide.BUY and oi_change > 0:
            _reject(self.strategy_id, feature_vector.symbol, "oi_not_confirming_negative_funding_fade", oi_change=oi_change)
            return None
        return _proposal(
            strategy_id=self.strategy_id,
            symbol=feature_vector.symbol,
            side=side,
            current_price=current_price,
            available_balance_usd=available_balance_usd,
            atr_pct=atr_pct,
            confidence=0.57 + min(0.18, abs(funding) / 100.0),
            rationale=f"funding mean reversion funding={funding:.2f}bps oi={oi_change:.3f}",
            feature_id=feature_vector.feature_id,
            tp_mult=1.0,
            sl_mult=0.65,
        )


class LiquidationHuntingStrategy(BaseStrategy):
    """Fade exhaustion after one-sided liquidation bursts."""

    def __init__(self, flow_tracker: Any, min_liq_notional_usd: float = 20_000.0, min_liq_imbalance: float = 0.65) -> None:
        self._flow = flow_tracker
        self._min_liq_notional_usd = min_liq_notional_usd
        self._min_liq_imbalance = min_liq_imbalance

    @property
    def strategy_id(self) -> str:
        return "liquidation_hunting_v1"

    def evaluate(self, feature_vector: FeatureVector, current_price: float, available_balance_usd: float) -> TradeProposal | None:
        f = _features(feature_vector)
        atr_pct = f.get("atr_14_pct")
        if feature_vector.quality_score < 0.6 or atr_pct is None:
            _reject(self.strategy_id, feature_vector.symbol, "low_quality_or_missing_atr")
            return None
        liq = self._flow.liquidation_stats(feature_vector.symbol)
        if liq is None or liq.total_notional < self._min_liq_notional_usd:
            _reject(self.strategy_id, feature_vector.symbol, "insufficient_liquidation_flow")
            return None
        if liq.imbalance >= self._min_liq_imbalance:
            side = OrderSide.SELL
        elif liq.imbalance <= -self._min_liq_imbalance:
            side = OrderSide.BUY
        else:
            _reject(self.strategy_id, feature_vector.symbol, "liquidation_imbalance_below_threshold", imbalance=liq.imbalance)
            return None
        return _proposal(
            strategy_id=self.strategy_id,
            symbol=feature_vector.symbol,
            side=side,
            current_price=current_price,
            available_balance_usd=available_balance_usd,
            atr_pct=atr_pct,
            confidence=0.59 + min(0.16, abs(liq.imbalance) * 0.16),
            rationale=f"liquidation exhaustion total={liq.total_notional:.0f} imbalance={liq.imbalance:.2f}",
            feature_id=feature_vector.feature_id,
            tp_mult=1.1,
            sl_mult=0.75,
        )


class MarketMakingStrategy(BaseStrategy):
    """Maker-first mean reversion proxy for the current single-order engine."""

    def __init__(self, spread_provider: Callable[[str], float | None], min_spread_bps: float = 1.2, max_spread_bps: float = 4.0) -> None:
        self._spread_provider = spread_provider
        self._min_spread_bps = min_spread_bps
        self._max_spread_bps = max_spread_bps

    @property
    def strategy_id(self) -> str:
        return "market_making_v1"

    def evaluate(self, feature_vector: FeatureVector, current_price: float, available_balance_usd: float) -> TradeProposal | None:
        f = _features(feature_vector)
        atr_pct = f.get("atr_14_pct")
        rsi = f.get("rsi_14")
        r3 = f.get("return_3", 0.0) or 0.0
        spread = self._spread_provider(feature_vector.symbol)
        if feature_vector.quality_score < 0.6 or atr_pct is None or rsi is None or spread is None:
            _reject(self.strategy_id, feature_vector.symbol, "low_quality_or_missing_spread")
            return None
        if spread < self._min_spread_bps or spread > self._max_spread_bps:
            _reject(self.strategy_id, feature_vector.symbol, "spread_outside_maker_band", spread_bps=spread)
            return None
        if rsi < 0.35 and r3 < 0:
            side = OrderSide.BUY
        elif rsi > 0.65 and r3 > 0:
            side = OrderSide.SELL
        else:
            _reject(self.strategy_id, feature_vector.symbol, "mean_reversion_setup_absent", rsi=rsi, return_3=r3)
            return None
        return _proposal(
            strategy_id=self.strategy_id,
            symbol=feature_vector.symbol,
            side=side,
            current_price=current_price,
            available_balance_usd=available_balance_usd,
            atr_pct=atr_pct,
            confidence=0.56,
            rationale=f"market making mean-reversion proxy spread={spread:.2f}bps rsi={rsi:.2f}",
            feature_id=feature_vector.feature_id,
            risk_pct=0.0025,
            tp_mult=0.55,
            sl_mult=0.55,
            max_notional_usd=50.0,
        )


class StatisticalArbitrageStrategy(BaseStrategy):
    """Z-score mean reversion using the symbol's own short/medium returns."""

    def __init__(self, min_zscore: float = 2.0, max_adx: float = 0.35) -> None:
        self._min_zscore = min_zscore
        self._max_adx = max_adx

    @property
    def strategy_id(self) -> str:
        return "statistical_arbitrage_v1"

    def evaluate(self, feature_vector: FeatureVector, current_price: float, available_balance_usd: float) -> TradeProposal | None:
        f = _features(feature_vector)
        atr_pct = f.get("atr_14_pct")
        r1 = f.get("return_1")
        r5 = f.get("return_5")
        vol = f.get("realized_volatility", f.get("volatility_20"))
        adx = f.get("adx_14", 0.0) or 0.0
        if feature_vector.quality_score < 0.6 or atr_pct is None or r1 is None or r5 is None:
            _reject(self.strategy_id, feature_vector.symbol, "low_quality_or_missing_returns")
            return None
        if vol is None or vol <= 0 or adx > self._max_adx:
            _reject(self.strategy_id, feature_vector.symbol, "volatility_or_adx_filter", volatility=vol, adx=adx)
            return None
        z = (r1 - r5 / 5.0) / max(vol, 1e-9)
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
        )
