"""Prometheus metrics registry (singleton).

All metrics are defined here so they can be imported from a single location.
The registry is created once at module load time.

Usage::

    from trader.monitoring.metrics import METRICS
    METRICS.orders_submitted.labels(symbol="BTCUSDT", side="Buy", order_type="Market").inc()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
)

# ---------------------------------------------------------------------------
# Shared registry (use default REGISTRY so prometheus_client HTTP server
# picks it up automatically; override in tests with a private registry)
# ---------------------------------------------------------------------------

_REGISTRY: CollectorRegistry | None = None


def _get_registry() -> CollectorRegistry:
    """Return the global prometheus registry."""
    from prometheus_client import REGISTRY

    return REGISTRY


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# Standard histogram buckets (in seconds)
_LATENCY_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
# In milliseconds
_MS_BUCKETS = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)


@dataclass
class TradingMetrics:
    """Singleton container for all Prometheus metrics."""

    _instance: ClassVar[TradingMetrics | None] = None

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    market_data_received_total: Counter = field(init=False)
    market_data_latency_ms: Histogram = field(init=False)
    market_data_stale_total: Counter = field(init=False)
    orderbook_depth_bid: Gauge = field(init=False)
    orderbook_depth_ask: Gauge = field(init=False)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    features_computed_total: Counter = field(init=False)
    feature_computation_seconds: Histogram = field(init=False)
    feature_quality_score: Gauge = field(init=False)
    feature_staleness_seconds: Gauge = field(init=False)

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------
    regime_detected_total: Counter = field(init=False)
    regime_current: Gauge = field(init=False)
    regime_confidence: Gauge = field(init=False)

    # ------------------------------------------------------------------
    # Trade proposals
    # ------------------------------------------------------------------
    proposals_generated_total: Counter = field(init=False)
    proposals_confidence: Histogram = field(init=False)

    # ------------------------------------------------------------------
    # Risk manager
    # ------------------------------------------------------------------
    risk_decisions_total: Counter = field(init=False)
    risk_approved_total: Counter = field(init=False)
    risk_rejected_total: Counter = field(init=False)
    risk_resized_total: Counter = field(init=False)
    portfolio_heat_pct: Gauge = field(init=False)
    daily_drawdown_pct: Gauge = field(init=False)
    weekly_drawdown_pct: Gauge = field(init=False)
    open_positions_count: Gauge = field(init=False)

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------
    orders_submitted_total: Counter = field(init=False)
    orders_confirmed_total: Counter = field(init=False)
    orders_rejected_total: Counter = field(init=False)
    orders_cancelled_total: Counter = field(init=False)
    orders_filled_total: Counter = field(init=False)
    order_fill_latency_ms: Histogram = field(init=False)
    order_slippage_bps: Histogram = field(init=False)

    # ------------------------------------------------------------------
    # Position / PnL
    # ------------------------------------------------------------------
    position_size_usd: Gauge = field(init=False)
    unrealised_pnl_usd: Gauge = field(init=False)
    realised_pnl_usd_total: Gauge = field(init=False)
    trade_pnl_usd: Histogram = field(init=False)

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------
    reconciliation_runs_total: Counter = field(init=False)
    reconciliation_discrepancies_total: Counter = field(init=False)
    reconciliation_duration_seconds: Histogram = field(init=False)

    # ------------------------------------------------------------------
    # WebSocket connectivity
    # ------------------------------------------------------------------
    ws_connection_status: Gauge = field(init=False)
    ws_reconnects_total: Counter = field(init=False)
    ws_messages_received_total: Counter = field(init=False)
    ws_message_processing_ms: Histogram = field(init=False)

    # ------------------------------------------------------------------
    # REST API
    # ------------------------------------------------------------------
    rest_requests_total: Counter = field(init=False)
    rest_errors_total: Counter = field(init=False)
    rest_latency_ms: Histogram = field(init=False)
    rate_limit_hits_total: Counter = field(init=False)

    # ------------------------------------------------------------------
    # System health
    # ------------------------------------------------------------------
    system_status: Gauge = field(init=False)
    component_healthy: Gauge = field(init=False)
    preflight_checks_passed_total: Counter = field(init=False)
    preflight_checks_failed_total: Counter = field(init=False)

    # ------------------------------------------------------------------
    # ML model
    # ------------------------------------------------------------------
    model_inference_total: Counter = field(init=False)
    model_inference_seconds: Histogram = field(init=False)
    model_drift_score: Gauge = field(init=False)
    model_staleness_seconds: Gauge = field(init=False)

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------
    kill_switch_activations_total: Counter = field(init=False)

    def __post_init__(self) -> None:
        """Create all metrics after dataclass initialisation."""
        # Market data
        self.market_data_received_total = Counter(
            "trader_market_data_received_total",
            "Total market data messages received",
            ["symbol", "topic"],
        )
        self.market_data_latency_ms = Histogram(
            "trader_market_data_latency_ms",
            "Market data end-to-end latency in milliseconds",
            ["symbol"],
            buckets=_MS_BUCKETS,
        )
        self.market_data_stale_total = Counter(
            "trader_market_data_stale_total",
            "Total stale market data events dropped",
            ["symbol"],
        )
        self.orderbook_depth_bid = Gauge(
            "trader_orderbook_depth_bid_usd",
            "Total USD depth on bid side (top 10 levels)",
            ["symbol"],
        )
        self.orderbook_depth_ask = Gauge(
            "trader_orderbook_depth_ask_usd",
            "Total USD depth on ask side (top 10 levels)",
            ["symbol"],
        )

        # Feature engineering
        self.features_computed_total = Counter(
            "trader_features_computed_total",
            "Total feature vectors computed",
            ["symbol", "version"],
        )
        self.feature_computation_seconds = Histogram(
            "trader_feature_computation_seconds",
            "Time to compute a feature vector",
            ["symbol"],
            buckets=_LATENCY_BUCKETS,
        )
        self.feature_quality_score = Gauge(
            "trader_feature_quality_score",
            "Latest feature quality score [0,1]",
            ["symbol"],
        )
        self.feature_staleness_seconds = Gauge(
            "trader_feature_staleness_seconds",
            "Age of the most recent feature vector in seconds",
            ["symbol"],
        )

        # Regime detection
        self.regime_detected_total = Counter(
            "trader_regime_detected_total",
            "Total regime classifications",
            ["symbol", "regime"],
        )
        self.regime_current = Gauge(
            "trader_regime_current",
            "Current regime index (for time-series tracking)",
            ["symbol", "regime"],
        )
        self.regime_confidence = Gauge(
            "trader_regime_confidence",
            "Confidence of current regime classification [0,1]",
            ["symbol"],
        )

        # Trade proposals
        self.proposals_generated_total = Counter(
            "trader_proposals_generated_total",
            "Total trade proposals generated by strategies",
            ["strategy_id", "symbol", "side"],
        )
        self.proposals_confidence = Histogram(
            "trader_proposals_confidence",
            "Distribution of trade proposal confidence scores",
            ["strategy_id"],
            buckets=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0),
        )

        # Risk manager
        self.risk_decisions_total = Counter(
            "trader_risk_decisions_total",
            "Total risk decisions made",
            ["status"],
        )
        self.risk_approved_total = Counter(
            "trader_risk_approved_total",
            "Total proposals approved (as-is or resized)",
            ["symbol"],
        )
        self.risk_rejected_total = Counter(
            "trader_risk_rejected_total",
            "Total proposals rejected",
            ["symbol", "reason"],
        )
        self.risk_resized_total = Counter(
            "trader_risk_resized_total",
            "Total proposals resized by the risk manager",
            ["symbol"],
        )
        self.portfolio_heat_pct = Gauge(
            "trader_portfolio_heat_pct",
            "Current portfolio heat as % of equity",
        )
        self.daily_drawdown_pct = Gauge(
            "trader_daily_drawdown_pct",
            "Current daily drawdown as % of equity",
        )
        self.weekly_drawdown_pct = Gauge(
            "trader_weekly_drawdown_pct",
            "Current weekly drawdown as % of equity",
        )
        self.open_positions_count = Gauge(
            "trader_open_positions_count",
            "Current number of open positions",
        )

        # Order execution
        self.orders_submitted_total = Counter(
            "trader_orders_submitted_total",
            "Total orders submitted to the exchange",
            ["symbol", "side", "order_type"],
        )
        self.orders_confirmed_total = Counter(
            "trader_orders_confirmed_total",
            "Total orders confirmed by the exchange",
            ["symbol", "confirmed_via"],
        )
        self.orders_rejected_total = Counter(
            "trader_orders_rejected_total",
            "Total orders rejected by the exchange",
            ["symbol", "reason"],
        )
        self.orders_cancelled_total = Counter(
            "trader_orders_cancelled_total",
            "Total orders cancelled",
            ["symbol"],
        )
        self.orders_filled_total = Counter(
            "trader_orders_filled_total",
            "Total orders fully filled",
            ["symbol", "side"],
        )
        self.order_fill_latency_ms = Histogram(
            "trader_order_fill_latency_ms",
            "Time from order submission to fill confirmation",
            ["symbol"],
            buckets=_MS_BUCKETS,
        )
        self.order_slippage_bps = Histogram(
            "trader_order_slippage_bps",
            "Slippage in basis points (actual vs expected fill price)",
            ["symbol", "side"],
            buckets=(0, 1, 2, 5, 10, 20, 50, 100, 200),
        )

        # Position / PnL
        self.position_size_usd = Gauge(
            "trader_position_size_usd",
            "Current position size in USD notional",
            ["symbol", "side"],
        )
        self.unrealised_pnl_usd = Gauge(
            "trader_unrealised_pnl_usd",
            "Current unrealised P&L in USD",
            ["symbol"],
        )
        self.realised_pnl_usd_total = Gauge(
            "trader_realised_pnl_usd_total",
            "Cumulative realised P&L in USD (can be negative — use .set(), Counter cannot decrease)",
            ["symbol", "strategy_id"],
        )
        self.trade_pnl_usd = Histogram(
            "trader_trade_pnl_usd",
            "P&L per closed trade in USD",
            ["symbol", "strategy_id"],
            buckets=(-500, -200, -100, -50, -20, -10, 0, 10, 20, 50, 100, 200, 500),
        )

        # Reconciliation
        self.reconciliation_runs_total = Counter(
            "trader_reconciliation_runs_total",
            "Total reconciliation passes",
            ["success"],
        )
        self.reconciliation_discrepancies_total = Counter(
            "trader_reconciliation_discrepancies_total",
            "Total order/position discrepancies found",
        )
        self.reconciliation_duration_seconds = Histogram(
            "trader_reconciliation_duration_seconds",
            "Duration of a reconciliation pass",
            buckets=_LATENCY_BUCKETS,
        )

        # WebSocket
        self.ws_connection_status = Gauge(
            "trader_ws_connection_status",
            "WebSocket connection health (1=connected, 0=disconnected)",
            ["channel"],
        )
        self.ws_reconnects_total = Counter(
            "trader_ws_reconnects_total",
            "Total WebSocket reconnection attempts",
            ["channel"],
        )
        self.ws_messages_received_total = Counter(
            "trader_ws_messages_received_total",
            "Total WebSocket messages received",
            ["channel", "topic"],
        )
        self.ws_message_processing_ms = Histogram(
            "trader_ws_message_processing_ms",
            "Time to process a WebSocket message",
            ["topic"],
            buckets=_MS_BUCKETS,
        )

        # REST API
        self.rest_requests_total = Counter(
            "trader_rest_requests_total",
            "Total REST API requests to exchange",
            ["endpoint", "method"],
        )
        self.rest_errors_total = Counter(
            "trader_rest_errors_total",
            "Total REST API errors",
            ["endpoint", "error_code"],
        )
        self.rest_latency_ms = Histogram(
            "trader_rest_latency_ms",
            "REST API round-trip latency in milliseconds",
            ["endpoint"],
            buckets=_MS_BUCKETS,
        )
        self.rate_limit_hits_total = Counter(
            "trader_rate_limit_hits_total",
            "Total rate-limit (429) responses from exchange",
            ["endpoint"],
        )

        # System health
        self.system_status = Gauge(
            "trader_system_status",
            "Current system status as integer index",
        )
        self.component_healthy = Gauge(
            "trader_component_healthy",
            "Health of individual components (1=healthy, 0=unhealthy)",
            ["component"],
        )
        self.preflight_checks_passed_total = Counter(
            "trader_preflight_checks_passed_total",
            "Total preflight checks that passed",
        )
        self.preflight_checks_failed_total = Counter(
            "trader_preflight_checks_failed_total",
            "Total preflight checks that failed",
        )

        # ML model
        self.model_inference_total = Counter(
            "trader_model_inference_total",
            "Total model inference calls",
            ["model_id", "algorithm"],
        )
        self.model_inference_seconds = Histogram(
            "trader_model_inference_seconds",
            "Model inference wall-clock time",
            ["model_id"],
            buckets=_LATENCY_BUCKETS,
        )
        self.model_drift_score = Gauge(
            "trader_model_drift_score",
            "Current model drift score [0,1]",
            ["model_id"],
        )
        self.model_staleness_seconds = Gauge(
            "trader_model_staleness_seconds",
            "Seconds since last model inference",
            ["model_id"],
        )

        # Kill switch
        self.kill_switch_activations_total = Counter(
            "trader_kill_switch_activations_total",
            "Total kill-switch activations",
            ["mode"],
        )

    @classmethod
    def get_instance(cls) -> TradingMetrics:
        """Return the singleton instance, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# Module-level convenience alias
METRICS: TradingMetrics = TradingMetrics.get_instance()
