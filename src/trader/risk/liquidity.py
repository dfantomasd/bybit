"""Liquidity awareness and market depth analysis.

Ensures trades are only taken in sufficiently liquid markets to avoid
slippage, execution issues, and price impact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class LiquidityAssessment:
    """Liquidity evaluation for a trading opportunity."""

    symbol: str
    bid_ask_spread_bps: float  # spread in basis points
    bid_ask_spread_pct: float  # spread as percentage
    bid_depth_usd: Decimal  # USD value of bids at top levels
    ask_depth_usd: Decimal  # USD value of asks at top levels
    total_depth_usd: Decimal  # combined bid+ask depth
    is_liquid: bool  # passes liquidity checks
    liquidity_score: float  # 0-1, higher = more liquid
    estimated_slippage_bps: float  # expected slippage in bps
    rejection_reason: str = ""


def assess_liquidity(
    symbol: str,
    bid_price: Decimal,
    ask_price: Decimal,
    bid_volumes: list[tuple[Decimal, Decimal]],  # [(price, volume), ...]
    ask_volumes: list[tuple[Decimal, Decimal]],
    position_size_usd: Decimal,
    max_spread_bps: float = 5.0,
    min_depth_usd: Decimal = Decimal("10000"),
    side: str | None = None,
) -> LiquidityAssessment:
    """Assess liquidity for a trading opportunity.

    Args:
        symbol: Trading symbol
        bid_price: Best bid price
        ask_price: Best ask price
        bid_volumes: Bid side orderbook levels [(price, volume), ...]
        ask_volumes: Ask side orderbook levels [(price, volume), ...]
        position_size_usd: Size of intended position in USD
        max_spread_bps: Maximum acceptable spread in basis points
        min_depth_usd: Minimum depth required in USD
        side: "Buy" consumes ask depth, "Sell" consumes bid depth. If not
            given, the more conservative (smaller) of the two sides is used
            since the eventual trade direction is unknown.

    Returns:
        LiquidityAssessment with evaluation and reasoning.
    """

    if bid_price <= 0 or ask_price <= 0:
        return LiquidityAssessment(
            symbol=symbol,
            bid_ask_spread_bps=0.0,
            bid_ask_spread_pct=0.0,
            bid_depth_usd=Decimal("0"),
            ask_depth_usd=Decimal("0"),
            total_depth_usd=Decimal("0"),
            is_liquid=False,
            liquidity_score=0.0,
            estimated_slippage_bps=0.0,
            rejection_reason="Invalid prices",
        )

    # Calculate spread
    mid_price = (bid_price + ask_price) / 2
    spread_pct = float((ask_price - bid_price) / mid_price * 100)
    spread_bps = spread_pct * 100  # convert % to bps

    # Calculate orderbook depth
    bid_depth = Decimal("0")
    for price, volume in bid_volumes[:10]:  # top 10 levels
        bid_depth += price * volume

    ask_depth = Decimal("0")
    for price, volume in ask_volumes[:10]:
        ask_depth += price * volume

    total_depth = bid_depth + ask_depth

    # Depth on the side the trade will actually consume: a buy eats ask
    # depth, a sell eats bid depth. When the side is unknown, use the
    # smaller of the two as a conservative proxy.
    if side == "Buy":
        directional_depth = ask_depth
    elif side == "Sell":
        directional_depth = bid_depth
    else:
        directional_depth = min(bid_depth, ask_depth)

    # Determine if liquid enough
    is_liquid = True
    rejection_reasons = []

    if spread_bps > max_spread_bps:
        is_liquid = False
        rejection_reasons.append(f"Spread {spread_bps:.1f}bps > max {max_spread_bps}bps")

    if total_depth < min_depth_usd:
        is_liquid = False
        rejection_reasons.append(f"Depth ${total_depth} < minimum ${min_depth_usd}")

    if position_size_usd > directional_depth / 2:
        # Position would be >50% of the relevant side's depth (risky)
        is_liquid = False
        rejection_reasons.append(f"Position size ${position_size_usd} > 50% of directional depth ${directional_depth}")

    # Calculate liquidity score (0-1)
    depth_ratio = min(1.0, float(total_depth / max(Decimal("100000"), min_depth_usd)))
    spread_ratio = max(0.0, 1.0 - (spread_bps / 10.0))  # 10bps = 0 score
    size_ratio = min(1.0, float(min_depth_usd / (position_size_usd + Decimal("1"))))

    liquidity_score = depth_ratio * 0.4 + spread_ratio * 0.4 + size_ratio * 0.2

    # Estimate slippage using the depth the trade will actually consume
    estimated_slippage = estimate_market_impact(
        position_size_usd=position_size_usd,
        total_depth_usd=directional_depth,
        spread_bps=spread_bps,
    )

    return LiquidityAssessment(
        symbol=symbol,
        bid_ask_spread_bps=spread_bps,
        bid_ask_spread_pct=spread_pct,
        bid_depth_usd=bid_depth,
        ask_depth_usd=ask_depth,
        total_depth_usd=total_depth,
        is_liquid=is_liquid,
        liquidity_score=float(liquidity_score),
        estimated_slippage_bps=estimated_slippage,
        rejection_reason="; ".join(rejection_reasons),
    )


def estimate_market_impact(
    position_size_usd: Decimal,
    total_depth_usd: Decimal,
    spread_bps: float,
) -> float:
    """Estimate market impact (slippage) from position size and depth.

    Args:
        position_size_usd: Position size in USD
        total_depth_usd: Total available depth in USD
        spread_bps: Bid-ask spread in basis points

    Returns:
        Estimated slippage in basis points.
    """

    if total_depth_usd <= 0:
        return 100.0  # very high slippage if no depth

    # Base slippage = spread/2 (half the spread for market order)
    base_slippage = spread_bps / 2.0

    # Additional slippage from position size relative to depth
    size_ratio = float(position_size_usd / total_depth_usd)

    # Empirical formula: slippage increases with sqrt(size_ratio)
    # 5% of depth = ~0.22 additional slippage
    # 10% of depth = ~0.32 additional slippage
    # 50% of depth = ~0.70 additional slippage
    impact_slippage = (size_ratio**0.5) * 100

    total_slippage = base_slippage + impact_slippage

    return min(500.0, total_slippage)  # cap at 500bps


def filter_tradeable_symbols(
    candidates: dict[str, dict[str, any]],  # {symbol: {bid, ask, bid_volumes, ask_volumes, ...}}
    position_size_usd: Decimal,
    max_spread_bps: float = 5.0,
    min_depth_usd: Decimal = Decimal("10000"),
    min_liquidity_score: float = 0.4,
) -> dict[str, any]:
    """Filter which symbols are liquid enough to trade.

    Args:
        candidates: Candidate symbols with orderbook data
        position_size_usd: Intended position size
        max_spread_bps: Maximum acceptable spread
        min_depth_usd: Minimum depth required
        min_liquidity_score: Minimum liquidity score (0-1)

    Returns:
        {approved: [symbol, ...], rejected: [(symbol, reason), ...]}
    """

    approved = []
    rejected = []

    for symbol, data in candidates.items():
        assessment = assess_liquidity(
            symbol=symbol,
            bid_price=data.get("bid_price", Decimal("0")),
            ask_price=data.get("ask_price", Decimal("0")),
            bid_volumes=data.get("bid_volumes", []),
            ask_volumes=data.get("ask_volumes", []),
            position_size_usd=position_size_usd,
            max_spread_bps=max_spread_bps,
            min_depth_usd=min_depth_usd,
        )

        if not assessment.is_liquid or assessment.liquidity_score < min_liquidity_score:
            reason = assessment.rejection_reason or "Insufficient liquidity"
            rejected.append((symbol, reason))
        else:
            approved.append(symbol)

    return {
        "approved": approved,
        "rejected": rejected,
        "total_candidates": len(candidates),
        "approval_rate": len(approved) / len(candidates) if candidates else 0.0,
    }
