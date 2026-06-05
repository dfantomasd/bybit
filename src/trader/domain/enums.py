"""Domain enumerations for the Bybit AI trading system."""
from __future__ import annotations

from enum import Enum


class TradingMode(str, Enum):
    """Trading execution mode — controls whether orders hit real markets."""

    TESTNET = "TESTNET"
    """Orders sent to Bybit testnet; no real money at risk."""

    SHADOW = "SHADOW"
    """Orders computed but never submitted; pure paper-trading on live prices."""

    CANARY_LIVE = "CANARY_LIVE"
    """Live trading with severely reduced size limits (canary / pilot mode)."""

    LIVE = "LIVE"
    """Full live trading — requires explicit operator activation."""


class SystemStatus(str, Enum):
    """High-level lifecycle status of the trading system."""

    STARTING = "STARTING"
    """System is initialising components."""

    PREFLIGHT = "PREFLIGHT"
    """Running preflight checks before enabling trading."""

    RUNNING = "RUNNING"
    """Normal operating state — strategies may generate signals."""

    SAFE_MODE = "SAFE_MODE"
    """Safe-mode engaged — no new entries; existing positions managed conservatively."""

    PAUSED = "PAUSED"
    """Temporarily paused by operator or risk rule."""

    STOPPING = "STOPPING"
    """Graceful shutdown in progress — orders being cancelled."""

    STOPPED = "STOPPED"
    """System fully stopped."""

    BLOCKED = "BLOCKED"
    """Blocked by kill-switch or regulatory trigger."""

    ERROR = "ERROR"
    """Unrecoverable error state requiring operator intervention."""


class RiskProfile(str, Enum):
    """Pre-defined risk appetite configurations."""

    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"
    SCALP = "SCALP"


class MarketRegime(str, Enum):
    """Detected market regime used to gate strategy behaviour."""

    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    EVENT_RISK = "EVENT_RISK"
    UNCERTAIN = "UNCERTAIN"


class OrderStatus(str, Enum):
    """Fine-grained order lifecycle state (local + exchange)."""

    CREATED_LOCAL = "CREATED_LOCAL"
    """Order record created locally, not yet submitted."""

    SUBMITTING = "SUBMITTING"
    """REST request in-flight."""

    REST_ACCEPTED = "REST_ACCEPTED"
    """Exchange REST endpoint returned 200/OK."""

    WS_CONFIRMED = "WS_CONFIRMED"
    """WebSocket order-update confirmed order active on exchange."""

    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    """Order partially executed."""

    FILLED = "FILLED"
    """Order fully executed."""

    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    """Cancel request sent, awaiting confirmation."""

    CANCELLED = "CANCELLED"
    """Order cancelled and confirmed by exchange."""

    REJECTED = "REJECTED"
    """Exchange rejected the order."""

    EXPIRED = "EXPIRED"
    """Order expired (e.g. IOC/FOK)."""

    UNKNOWN_RECONCILIATION_REQUIRED = "UNKNOWN_RECONCILIATION_REQUIRED"
    """State is unknown; reconciliation loop must resolve."""


class RiskDecisionStatus(str, Enum):
    """Outcome of the Risk Manager's evaluation of a trade proposal."""

    APPROVED = "APPROVED"
    """Proposal accepted as-is."""

    RESIZED = "RESIZED"
    """Proposal approved with quantity reduced to fit risk limits."""

    REJECTED = "REJECTED"
    """Proposal rejected; no order should be submitted."""

    SAFE_MODE_ONLY = "SAFE_MODE_ONLY"
    """Only safe-mode operations permitted (reduce / close only)."""

    PAUSED = "PAUSED"
    """Risk manager is paused; no new entries allowed."""


class MarketType(str, Enum):
    """Bybit market category."""

    SPOT = "spot"
    LINEAR = "linear"
    INVERSE = "inverse"
    OPTION = "option"


class OrderSide(str, Enum):
    """Order direction as expected by Bybit v5 API."""

    BUY = "Buy"
    SELL = "Sell"


class OrderType(str, Enum):
    """Order execution type."""

    MARKET = "Market"
    LIMIT = "Limit"


class BybitRegion(str, Enum):
    """Bybit regional entity for regulatory compliance."""

    GLOBAL = "GLOBAL"
    NL = "NL"
    EEA = "EEA"
    TR = "TR"
    KZ = "KZ"
    GE = "GE"
    AE = "AE"
    ID = "ID"


class VolatilityLevel(str, Enum):
    """Discretised volatility level used in regime classification."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class KillSwitchMode(str, Enum):
    """Escalating kill-switch severities."""

    PAUSE_NEW_ENTRIES = "PAUSE_NEW_ENTRIES"
    """Stop opening new positions but leave existing ones running."""

    CANCEL_OPEN_ORDERS = "CANCEL_OPEN_ORDERS"
    """Cancel all open (unfilled) orders."""

    REDUCE_RISK = "REDUCE_RISK"
    """Halve position sizes."""

    CLOSE_ALL_IF_CONFIGURED = "CLOSE_ALL_IF_CONFIGURED"
    """Close all positions if the operator has enabled close-on-kill."""

    FULL_STOP = "FULL_STOP"
    """Cancel orders, optionally close positions, halt the system."""
