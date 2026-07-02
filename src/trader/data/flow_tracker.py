"""Short-horizon trade and liquidation flow analytics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from trader.domain.enums import OrderSide

_DEFAULT_WINDOW_S = 60.0


@dataclass(frozen=True)
class FlowStats:
    buy_notional: float
    sell_notional: float
    imbalance: float
    total_notional: float
    large_trade_count: int


@dataclass(frozen=True)
class LiquidationStats:
    buy_notional: float
    sell_notional: float
    imbalance: float
    total_notional: float


@dataclass(frozen=True)
class _FlowPrint:
    ts: datetime
    side: OrderSide
    notional: float


class FlowTracker:
    """Keeps rolling public trade and liquidation pressure per symbol."""

    def __init__(
        self,
        window_s: float = _DEFAULT_WINDOW_S,
        history_slots: int = 512,
        large_trade_notional_usd: float = 10_000.0,
    ) -> None:
        self._window_s = window_s
        self._large_trade_notional_usd = large_trade_notional_usd
        self._trades: dict[str, deque[_FlowPrint]] = {}
        self._liquidations: dict[str, deque[_FlowPrint]] = {}
        self._history_slots = history_slots

    def record_trade(
        self,
        symbol: str,
        side: OrderSide,
        price: Decimal,
        qty: Decimal,
        ts: datetime | None = None,
    ) -> None:
        self._record(self._trades, symbol, side, price, qty, ts)

    def record_liquidation(
        self,
        symbol: str,
        side: OrderSide,
        price: Decimal,
        qty: Decimal,
        ts: datetime | None = None,
    ) -> None:
        self._record(self._liquidations, symbol, side, price, qty, ts)

    def trade_stats(self, symbol: str, now: datetime | None = None) -> FlowStats | None:
        prints = self._fresh(self._trades.get(symbol), now)
        if not prints:
            return None
        buy = sum(p.notional for p in prints if p.side == OrderSide.BUY)
        sell = sum(p.notional for p in prints if p.side == OrderSide.SELL)
        total = buy + sell
        if total <= 0:
            return None
        large = sum(1 for p in prints if p.notional >= self._large_trade_notional_usd)
        return FlowStats(
            buy_notional=buy,
            sell_notional=sell,
            imbalance=(buy - sell) / total,
            total_notional=total,
            large_trade_count=large,
        )

    def liquidation_stats(self, symbol: str, now: datetime | None = None) -> LiquidationStats | None:
        prints = self._fresh(self._liquidations.get(symbol), now)
        if not prints:
            return None
        buy = sum(p.notional for p in prints if p.side == OrderSide.BUY)
        sell = sum(p.notional for p in prints if p.side == OrderSide.SELL)
        total = buy + sell
        if total <= 0:
            return None
        return LiquidationStats(
            buy_notional=buy,
            sell_notional=sell,
            imbalance=(buy - sell) / total,
            total_notional=total,
        )

    def _record(
        self,
        store: dict[str, deque[_FlowPrint]],
        symbol: str,
        side: OrderSide,
        price: Decimal,
        qty: Decimal,
        ts: datetime | None,
    ) -> None:
        notional = float(price * qty)
        if notional <= 0:
            return
        bucket = store.setdefault(symbol.upper(), deque(maxlen=self._history_slots))
        bucket.append(_FlowPrint(ts=ts or datetime.now(tz=UTC), side=side, notional=notional))
        self._prune(bucket)

    def _fresh(self, prints: deque[_FlowPrint] | None, now: datetime | None) -> list[_FlowPrint]:
        if not prints:
            return []
        self._prune(prints, now)
        return list(prints)

    def _prune(self, prints: deque[_FlowPrint], now: datetime | None = None) -> None:
        cutoff = (now or datetime.now(tz=UTC)) - timedelta(seconds=self._window_s)
        while prints and prints[0].ts < cutoff:
            prints.popleft()

    def remove_symbol(self, symbol: str) -> None:
        sym = symbol.upper()
        self._trades.pop(sym, None)
        self._liquidations.pop(sym, None)
