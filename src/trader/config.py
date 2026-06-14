"""Application configuration via pydantic-settings.

Secrets are loaded from:
1. Environment variables (highest priority)
2. .env file
3. Docker secrets at /run/secrets/

CRITICAL: SecretStr fields MUST NOT be logged or serialised in plaintext.
The get_secret_value() method must only be called in the minimal execution
context that requires the actual credential.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from trader.domain.enums import BybitRegion, RiskProfile, TradingMode


class Settings(BaseSettings):
    """Top-level application settings.

    All SECRET fields use SecretStr so they are redacted in __str__ / repr.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        secrets_dir="/run/secrets",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Environment / mode
    # ------------------------------------------------------------------
    TRADING_MODE: TradingMode = TradingMode.TESTNET
    """Controls whether orders go to testnet, paper, canary, or live markets.
    NEVER changes to LIVE without explicit operator action."""

    RISK_PROFILE: RiskProfile = RiskProfile.CONSERVATIVE

    # ------------------------------------------------------------------
    # Bybit API credentials
    # ------------------------------------------------------------------
    BYBIT_API_KEY: SecretStr = SecretStr("")
    BYBIT_API_SECRET: SecretStr = SecretStr("")
    BYBIT_REGION: BybitRegion = BybitRegion.GLOBAL
    BYBIT_USE_TESTNET: bool = True
    """Must be True whenever TRADING_MODE is not LIVE."""
    BYBIT_CONNECTIVITY_REQUIRED: bool = False
    """When True, Bybit REST connectivity must pass preflight."""
    DEFAULT_MARKET_CATEGORY: str = "linear"
    """Default Bybit category for read-only monitoring commands."""

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    POSTGRES_DSN: SecretStr = SecretStr("postgresql+asyncpg://trader:trader@postgres:5432/trader")
    TRADE_JOURNAL_ENABLED: bool = True
    """Persist signals, risk decisions, order events, and closed PnL in Postgres."""
    PERFORMANCE_FILTER_ENABLED: bool = True
    """Use stored closed PnL to temporarily skip recently losing symbols."""
    PERFORMANCE_MIN_CLOSED_TRADES: int = 5
    PERFORMANCE_MAX_SYMBOL_LOSS_USD: float = -2.0
    PERFORMANCE_LOOKBACK_DAYS: int = 7
    PERFORMANCE_MIN_TRADABLE_SYMBOLS: int = 2
    """Relax performance blocks if they would leave too few symbols tradable."""
    CLOSED_PNL_REFRESH_INTERVAL_SECONDS: int = 300
    PROFIT_MANAGER_ENABLED: bool = True
    TRAILING_STOP_ENABLED: bool = True
    POSITION_MANAGEMENT_INTERVAL_SECONDS: int = 30
    TRAILING_ACTIVATION_PCT: float = 0.70
    """Unrealised profit percent before enabling exchange trailing stop."""
    TRAILING_DISTANCE_PCT: float = 0.25
    """Trailing stop distance as percent of current mark price."""
    BREAKEVEN_STOP_OFFSET_PCT: float = 0.20
    """Small offset beyond entry used when moving stop to breakeven."""
    MIN_NET_PROFIT_BUFFER_PCT: float = 0.08
    """Minimum profit above all costs required at breakeven stop."""
    EXPECTED_SLIPPAGE_PCT: float = 0.03
    """Expected slippage per side as percent of price."""

    DEFAULT_LINEAR_MAKER_FEE_RATE: float = 0.0002
    DEFAULT_LINEAR_TAKER_FEE_RATE: float = 0.00055
    """Fallback fee rates when API is unavailable (SHADOW only)."""

    MIN_EXPECTED_NET_EDGE_PCT: float = 0.25
    """Minimum expected net edge (after all costs) required to enter a trade."""
    NET_EDGE_SAFETY_MARGIN_PCT: float = 0.05
    """Extra safety margin subtracted from net edge before comparing to MIN_EXPECTED_NET_EDGE_PCT."""
    FUNDING_BUFFER_PCT: float = 0.01
    """Estimated funding cost buffer per position."""

    # ------------------------------------------------------------------
    # Scalping (ScalpMicroStrategy)
    # ------------------------------------------------------------------
    SCALP_STRATEGY_ENABLED: bool = True
    """Enable the cost-aware micro-scalping strategy alongside the trend strategy."""
    MAX_SPREAD_BPS_SCALP: float = 2.5
    """Maximum bid-ask spread (bps) for scalp entries. Unknown spread fails closed."""
    MIN_NET_SCALP_RETURN_PCT: float = 0.08
    """Minimum expected NET return (percent) after fees+spread+slippage for a scalp."""
    SCALP_COOLDOWN_SECONDS: int = 60
    """Minimum seconds between scalp signals per symbol."""
    SCALP_MAX_TRADES_PER_MINUTE: int = 10
    """Global cap on scalp signals per minute across the whole portfolio."""
    SCALP_MAX_POSITION_NOTIONAL_USD: float = 100.0
    """Hard notional cap per scalp position."""
    SCALP_MIN_OB_IMBALANCE: float = 0.15
    """Required L5 orderbook imbalance agreeing with the signal side (BUY needs
    >= +value, SELL needs <= -value). Missing/stale book data fails OPEN."""
    TREND_MIN_ADX: float = 0.25
    """Minimum normalized ADX for EMA trend entries. 0.25 means ADX 25."""
    TREND_BLOCK_NEGATIVE_FUNDING_OI: bool = True
    """Block fragile trend entries when funding and open interest context disagrees."""

    # ------------------------------------------------------------------
    # Orderbook microstructure feed
    # ------------------------------------------------------------------
    ORDERBOOK_FEED_ENABLED: bool = True
    """Subscribe to orderbook.50 for execution candidates and derive imbalance/
    microprice features. Adds ~5-10 KB/s WS traffic per tracked symbol."""

    # ------------------------------------------------------------------
    # Per-candle training sampler
    # ------------------------------------------------------------------
    CANDLE_SAMPLING_ENABLED: bool = True
    """Record a feature snapshot + rule-direction baseline event on EVERY
    confirmed 1m candle (decision=SHADOW_CANDLE), not only on trade signals.
    Multiplies training-sample accumulation ~100x. Sampler events are excluded
    from signal statistics (buckets, healthcheck, bootstrap baseline)."""

    # ------------------------------------------------------------------
    # Regime-bucket performance gating
    # ------------------------------------------------------------------
    BUCKET_BLOCK_ENABLED: bool = True
    """Skip strategy evaluation in (regime, volatility, UTC hour) buckets whose
    own historical signals show persistent negative expectancy."""
    BUCKET_MIN_SAMPLES: int = 30
    """Minimum resolved outcomes in a bucket before it can be blocked."""
    BUCKET_BLOCK_AVG_BPS: float = -2.0
    """Block a bucket when its average net return is below this (bps)."""
    BUCKET_STATS_REFRESH_SECONDS: int = 3600
    """How often the in-memory bucket statistics are refreshed from Postgres."""

    # ------------------------------------------------------------------
    # Startup candle backfill
    # ------------------------------------------------------------------
    STARTUP_BACKFILL_ENABLED: bool = True
    """Backfill missing market_candles history via REST once at startup so the
    canary history requirements and model training don't wait days for WS data."""
    STARTUP_BACKFILL_DAYS: int = 2
    """How many days of history to backfill per symbol/interval."""
    STARTUP_BACKFILL_MAX_REQUESTS: int = 200
    """Hard cap on REST kline requests per startup (rate-limit protection)."""

    # ------------------------------------------------------------------
    # Anti zero-trading guards
    # ------------------------------------------------------------------
    MIN_SIGNALS_PER_HOUR: int = 1
    """Expected minimum signals/hour; below this with zero fills a warning is logged."""
    AUTO_SOFTEN_FILTERS_ENABLED: bool = False
    """Reserved: when true, filters may be relaxed automatically on zero trading. Off by default."""
    FALLBACK_TO_RULE_WHEN_MODEL_UNSURE: bool = True
    """When the model exists but its score is below the gate threshold, keep the
    rule-based proposal instead of dropping it (hybrid mode fallback)."""

    ENTRY_ORDER_MODE: str = "MAKER_FIRST"
    """MARKET or MAKER_FIRST. MAKER_FIRST places a POST_ONLY limit at the best
    bid/ask first (maker fee/rebate), then escalates to a market order or aborts
    after MAKER_TIMEOUT_SECONDS — see MAKER_* settings."""

    # ------------------------------------------------------------------
    # Maker-first execution (ENTRY_ORDER_MODE = "MAKER_FIRST")
    # ------------------------------------------------------------------
    MAKER_TIMEOUT_SECONDS: float = 3.0
    """How long to wait for the POST_ONLY limit to fill before deciding to
    escalate (when MAKER_ALLOW_ESCALATION) or keep resting until TTL."""
    MAKER_TTL_SECONDS: float = 5.0
    """Absolute lifetime of the maker order. With escalation disabled the order
    rests until TTL, then the remainder is cancelled and the entry aborted."""
    MAKER_ALLOW_ESCALATION: bool = True
    """Escalate the unfilled remainder to a market (taker) order after the
    timeout — only when price has not drifted and the orderbook imbalance does
    not contradict the direction; otherwise the entry is aborted."""
    REDIS_URL: SecretStr = SecretStr("")
    REDIS_REQUIRED: bool = False
    """When True, Redis must pass preflight. Render Free monitoring can run without Redis."""

    # ------------------------------------------------------------------
    # Telegram notifications
    # ------------------------------------------------------------------
    TELEGRAM_BOT_TOKEN: SecretStr = SecretStr("")
    TELEGRAM_ALLOWED_CHAT_IDS: Annotated[list[int], NoDecode] = []

    # ------------------------------------------------------------------
    # LLM (optional, disabled by default)
    # ------------------------------------------------------------------
    LLM_ENABLED: bool = False
    LLM_PROVIDER: str = "ollama"
    LLM_BASE_URL: str = "http://ollama:11434"
    LLM_MODEL: str = "llama3"
    LLM_BUDGET_CAP_USD: float = 5.0
    """Maximum USD spend on LLM API calls per day (external providers)."""

    # ------------------------------------------------------------------
    # Risk defaults (overridden by profile YAML)
    # ------------------------------------------------------------------
    MAX_POSITIONS: int = 2
    MAX_POSITION_SIZE_PCT: float = 2.0
    """Max single position as % of account equity."""

    MAX_PORTFOLIO_HEAT_PCT: float = 6.0
    """Sum of all open position risks as % of equity."""

    MAX_DAILY_DRAWDOWN_PCT: float = 3.0
    """Trigger safe-mode if daily drawdown exceeds this."""

    MAX_WEEKLY_DRAWDOWN_PCT: float = 8.0
    """Trigger kill-switch if weekly drawdown exceeds this."""

    MAX_POSITION_VALUE_USD: float = 10.0
    """Hard cap on the notional value (USD) of a single position.
    Applies after all risk multipliers; overrides profile sizing if lower.
    Useful for CANARY_LIVE where you want an absolute dollar guard."""

    ALLOW_ENTRIES_IN_SIDEWAYS: bool = False
    """When False (default), new entries are blocked in SIDEWAYS market regime.
    Set True to allow trading in sideways/ranging markets (increases false signals)."""

    # ------------------------------------------------------------------
    # Monitoring / observability
    # ------------------------------------------------------------------
    PROMETHEUS_PORT: int = 9090
    FASTAPI_PORT: int = 8080
    INTERNAL_API_KEY: SecretStr = SecretStr("")
    """Optional API key for observability endpoints. Generated at startup when empty."""
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    """'json' for production, 'console' for development."""

    # ------------------------------------------------------------------
    # Feature flags
    # ------------------------------------------------------------------
    SHADOW_MODE: bool = True
    """When True, orders are computed but never submitted."""

    LIVE_MODE: bool = False
    """Explicit opt-in required to allow live order submission."""

    TELEGRAM_ALLOW_RISK_INCREASE: bool = False
    """When False (default), Telegram /risk command cannot escalate to a riskier profile.
    Risk escalation requires explicit env-var opt-in to prevent accidental leverage bumps."""

    MIN_NOTIONAL_SAFETY_BUFFER_PCT: float = 3.0
    """Safety buffer applied on top of exchange min-notional (e.g. 3% → $5 min becomes $5.15).
    Prevents near-limit orders from being rejected by code=110094."""

    # ------------------------------------------------------------------
    # Screener
    # ------------------------------------------------------------------
    SCREENER_WIDE_MAX_SYMBOLS: int = 80
    """Maximum symbols in the wide universe (volume + spread + depth filter pass)."""

    SCREENER_FEATURE_MAX_SYMBOLS: int = 30
    """Symbols that receive WS subscriptions and have features computed."""

    SCREENER_EXECUTION_CANDIDATES: int = 15
    """Top symbols passed to the strategy for signal generation each cycle."""

    SCREENER_MIN_VOLUME_USD: float = 20_000_000.0
    """Minimum 24h turnover in USD to enter the wide universe."""

    SCREENER_MAX_SPREAD_BPS: float = 8.0
    """Maximum bid-ask spread in basis points (wider → excluded)."""

    SCREENER_MIN_TOP_BOOK_DEPTH_USD: float = 5_000.0
    """Minimum top-of-book depth in USD (bid1 + ask1 notional)."""

    SCREENER_MIN_PRICE_USD: float = 0.0
    """Optional minimum last price filter. 0 disables the lower bound."""

    SCREENER_MAX_PRICE_USD: float = 0.0
    """Optional maximum last price filter. 0 disables the upper bound."""

    SCREENER_REFRESH_SECONDS: int = 900
    """How often to refresh the screener universe (seconds)."""

    SCREENER_SUBSCRIBE_TIMEOUT_SECONDS: int = 10
    """Reserved: WS subscribe acknowledgement timeout per symbol (not yet enforced in screener)."""

    SCREENER_DENYLIST: list[str] = []
    """Symbols explicitly excluded (pre-market, innovation zone, etc.)."""

    # ------------------------------------------------------------------
    # Screener tuning for Render Starter
    # ------------------------------------------------------------------
    STARTER_OPTIMIZED_MODE: bool = True
    """Apply Render Starter memory/CPU conservative defaults."""

    # ------------------------------------------------------------------
    # Burst / entry rate limiting
    # ------------------------------------------------------------------
    MAX_NEW_ENTRIES_PER_MINUTE: int = 1
    """Maximum new position entries allowed per minute (burst guard)."""
    MAX_CONCURRENT_PENDING_ENTRIES: int = 1
    """Maximum simultaneous positions in SUBMITTING/REST_ACCEPTED state."""
    MAX_SAME_SIDE_POSITIONS: int = 2
    """Maximum open positions on the same side (Buy or Sell)."""
    MAX_CORRELATED_POSITIONS: int = 2
    """Reserved: correlation-based position limiting (not yet wired into execution)."""
    STARTUP_WARMUP_SECONDS: int = 180
    """Seconds after startup before new entries are allowed (monitoring-only phase)."""

    # ------------------------------------------------------------------
    # Database safety gates
    # ------------------------------------------------------------------
    TRADE_JOURNAL_REQUIRED_FOR_ACTIVE: bool = True
    """In CANARY_LIVE/LIVE modes, block new entries if TradeJournal is unavailable."""
    DURABLE_ORDER_STATE_REQUIRED_FOR_ACTIVE: bool = True
    """In CANARY_LIVE/LIVE modes, block new entries if durable order state is unavailable."""

    # ------------------------------------------------------------------
    # Multi-timeframe candle store
    # ------------------------------------------------------------------
    MULTITIMEFRAME_ENABLED: bool = True
    MULTITIMEFRAME_INTERVALS: Annotated[list[str], NoDecode] = ["1", "5", "15", "60"]
    CANDLE_STORE_MAX_BARS_1M: int = 250  # reserved: per-interval candle store capacity (not yet read by CandleStore)
    CANDLE_STORE_MAX_BARS_5M: int = 250
    CANDLE_STORE_MAX_BARS_15M: int = 200
    CANDLE_STORE_MAX_BARS_1H: int = 120

    # ------------------------------------------------------------------
    # Orderbook mode
    # ------------------------------------------------------------------
    ORDERBOOK_MODE: str = "ON_DEMAND"
    """ON_DEMAND = fetch only for top candidates; STREAMING = subscribe for all."""
    MAX_ORDERBOOK_ACTIVE_SYMBOLS: int = 5  # reserved: STREAMING mode symbol cap (not yet enforced)

    # ------------------------------------------------------------------
    # Adaptive load governor
    # ------------------------------------------------------------------
    ADAPTIVE_LOAD_GOVERNOR_ENABLED: bool = True
    LOAD_GOVERNOR_CHECK_SECONDS: int = 30
    MAX_FEATURE_CYCLE_MS: int = 8000  # reserved: governor cycle-time thresholds (not yet read)
    MAX_STRATEGY_CYCLE_MS: int = 8000
    MAX_EVENT_LOOP_LAG_MS: int = 500
    MAX_QUEUE_UTILIZATION_PCT: int = 70  # reserved: queue-utilization gate (not yet enforced)
    LOAD_GOVERNOR_MIN_FEATURE_SYMBOLS: int = 10
    LOAD_GOVERNOR_MIN_EXECUTION_CANDIDATES: int = 3  # reserved: load governor floor (not yet enforced)

    # ------------------------------------------------------------------
    # ML / model
    # ------------------------------------------------------------------
    MODEL_ENABLED: bool = True
    """Enable lightweight supervised challenger model."""
    MODEL_ALLOW_LIVE_DECISIONS: bool = False
    """When False, model only scores in shadow; rule-based strategy remains authoritative.
    When True, a compatible CHAMPION model may replace rule-based decisions (hybrid mode).
    Real orders are still gated by TRADING_MODE/LIVE_MODE/LIVE_ARMED."""
    MODEL_MIN_TRAINING_SAMPLES: int = 500
    MODEL_MIN_CLOSED_TRADES_FOR_PROMOTION: int = 50
    MODEL_ENCRYPT_KEY: SecretStr = SecretStr("")
    """Fernet key (or arbitrary passphrase) for encrypting model artifacts at
    rest in Postgres. Artifacts are pickle — without encryption a compromised
    database means code execution in the trader. Empty = legacy plaintext."""
    MODEL_TYPE: str = "GBDT"
    """Challenger architecture: "GBDT" (gradient-boosted trees, stronger on
    non-linear feature interactions), "LOGREG" (regularized linear baseline),
    or "SGD" (linear, online-updateable)."""
    MODEL_CANDIDATES: str = "GBDT,LOGREG"
    """CSV list of model families considered by offline walk-forward selection."""
    MODEL_WF_FOLDS: int = 5
    MODEL_WF_MIN_TRAIN_SAMPLES: int = 500
    MODEL_THRESHOLD_GRID: str = "0,2,5,8,12"
    """CSV label thresholds in bps evaluated during offline selection."""
    MODEL_MIN_PASS_COUNT_FOR_PROMOTION: int = 20
    """Minimum model-pass observations expected before trusting promotion metrics."""
    TRAIN_EXCLUDE_NEGATIVE_BUCKETS: bool = False
    TRAIN_MIN_BUCKET_SAMPLES: int = 50
    TRAIN_BUCKET_MIN_AVG_BPS: float = -5.0
    """Optional training filter: remove regime/hour buckets with stable negative expectancy."""
    MODEL_SHADOW_SCORING_ENABLED: bool = True
    """Always run shadow scoring even when live decisions disabled."""
    MODEL_AUTO_TRAIN_ENABLED: bool = True
    """Automatically train a shadow challenger when enough new labelled examples accumulate."""
    MODEL_AUTO_TRAIN_MIN_SAMPLES: int = 1000
    MODEL_AUTO_TRAIN_INCREMENT_SAMPLES: int = 5000
    MODEL_AUTO_TRAIN_CHECK_SECONDS: int = 300
    MODEL_AUTO_TRAIN_HORIZON_MINUTES: int = 5
    MODEL_AUTO_TRAIN_LABEL_BPS: float = 5.0
    MODEL_AUTO_PROMOTE_ENABLED: bool = False
    """Auto-promote challenger to champion when it beats the current champion
    AND the lift is statistically significant (bootstrap p-value).
    Disabled by default: let the model train and accumulate shadow evidence first,
    then enable explicitly via env."""
    MODEL_AUTO_PROMOTE_CHECK_SECONDS: int = 600
    MODEL_AUTO_PROMOTE_MIN_SIGNALS: int = 50
    MODEL_AUTO_PROMOTE_MIN_LIFT_BPS: float = 1.0
    """Minimum live lift (bps) the challenger must show before auto-promotion."""
    MODEL_AUTO_PROMOTE_MIN_PASS_EXPECTANCY_BPS: float = 0.0
    """Minimum average net return (bps) among challenger GATE_PASS outcomes."""
    MODEL_AUTO_PROMOTE_MIN_WF_BPS: float = 0.0
    """Minimum walk-forward expectancy (bps) stored in challenger training metrics."""
    MODEL_AUTO_PROMOTE_MIN_WF_POSITIVE_FOLDS: int = 3
    """Minimum positive walk-forward folds required before auto-promotion."""
    MODEL_AUTO_PROMOTE_MAX_WF_STD_BPS: float = 25.0
    """Maximum walk-forward fold standard deviation allowed before auto-promotion."""
    MODEL_AUTO_PROMOTE_PVALUE_THRESHOLD: float = 0.05
    """Maximum bootstrap p-value for auto-promotion: the challenger's mean net
    return must beat the baseline in >= (1 - threshold) of bootstrap resamples."""
    MODEL_AUTO_PROMOTE_BOOTSTRAP_ITERATIONS: int = 1000
    MODEL_AUTO_PROMOTE_MIN_BOOTSTRAP_SAMPLES: int = 50
    """Minimum resolved challenger returns required to run the bootstrap test."""
    MODEL_CHAMPION_MONITOR_SECONDS: int = 14_400
    """How often to check current CHAMPION for degradation and rollback."""
    MODEL_CHAMPION_MIN_WF_BPS: float = 0.0
    """Rollback CHAMPION when stored walk-forward expectancy falls below this."""
    MODEL_CHAMPION_MAX_DRAWDOWN_BPS: float = 1500.0
    """Rollback CHAMPION when recent model return drawdown exceeds this bps limit."""
    MODEL_CHAMPION_MIN_PAPER_GATE_COUNT: int = 20
    """Minimum paper-gate sample count required for walk-forward champion selection."""
    MODEL_SHADOW_GATE_ENABLED: bool = True
    """Evaluate a model-based pass/block gate in shadow, without affecting execution."""
    MODEL_SHADOW_GATE_THRESHOLD: float = 0.55
    MODEL_GATE_CANARY_ENABLED: bool = False
    """When enabled, allow the model gate to block entries only with conservative safeguards.
    Disabled by default until a promoted CHAMPION shows positive shadow-gate lift."""
    MODEL_GATE_CANARY_MIN_QUALITY: str = "GOOD"
    MODEL_GATE_CANARY_MAX_BLOCK_RATE_PCT: float = 60.0
    MODEL_GATE_CANARY_MIN_OBSERVATIONS: int = 50
    MODEL_GATE_CANARY_MIN_LIFT_BPS: float = 0.0
    MODEL_GATE_CANARY_ALLOW_EVERY_NTH_BLOCKED: int = 3
    MODEL_PAPER_NOTIONAL_USD: float = 5.0

    # ------------------------------------------------------------------
    # CANARY_LIVE safety
    # ------------------------------------------------------------------
    LIVE_ARMED: bool = False
    """Secondary safety gate required alongside LIVE_MODE=true for live execution."""
    CANARY_MAX_OPEN_POSITIONS: int = 2
    CANARY_MAX_TOTAL_EXPOSURE_PCT: float = 45.0

    # ------------------------------------------------------------------
    # Operational
    # ------------------------------------------------------------------
    TRANSACTION_LOG_SYNC_INTERVAL_SECONDS: int = 60
    """How often to sync the Bybit transaction log to the database."""
    RECONCILIATION_INTERVAL_SECONDS: int = 30
    POSITION_SYNC_INTERVAL_SECONDS: int = 30
    """How often to sync exchange positions into the local execution/risk state."""
    HEALTH_CHECK_INTERVAL_SECONDS: int = 15
    DATA_STALENESS_THRESHOLD_SECONDS: int = 5
    FEATURE_STALENESS_THRESHOLD_SECONDS: int = 10
    MODEL_STALENESS_THRESHOLD_SECONDS: int = 3600

    @field_validator("MULTITIMEFRAME_INTERVALS", mode="before")
    @classmethod
    def parse_intervals(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return ["1", "5", "15", "60"]
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str):
            stripped = value.strip().strip("[]")
            if not stripped:
                return ["1", "5", "15", "60"]
            return [v.strip() for v in stripped.split(",") if v.strip()]
        raise TypeError("MULTITIMEFRAME_INTERVALS must be a list or comma-separated string")

    @field_validator("TELEGRAM_ALLOWED_CHAT_IDS", mode="before")
    @classmethod
    def parse_chat_ids(cls, value: Any) -> list[int]:
        """Accept JSON-style, comma-separated, or single chat ID strings."""
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [int(item) for item in value]
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                stripped = stripped[1:-1]
            if not stripped:
                return []
            return [int(item.strip()) for item in stripped.split(",") if item.strip()]
        raise TypeError("TELEGRAM_ALLOWED_CHAT_IDS must be a list or string")

    def model_post_init(self, __context: Any) -> None:
        """Enforce critical safety invariants after field parsing."""
        if self.STARTER_OPTIMIZED_MODE and self.SCREENER_MAX_PRICE_USD <= 0:
            self.SCREENER_MAX_PRICE_USD = 25.0

        # TESTNET mode must use testnet endpoints to avoid spending real money.
        # SHADOW mode is safe with mainnet endpoints because orders are never
        # submitted — mainnet is needed on US-hosted deployments where Bybit
        # testnet blocks requests by IP.
        if self.TRADING_MODE == TradingMode.TESTNET and not self.BYBIT_USE_TESTNET:
            raise ValueError(
                "BYBIT_USE_TESTNET must be True when TRADING_MODE=TESTNET. "
                "To use mainnet endpoints without real orders, set TRADING_MODE=SHADOW."
            )

        # Live mode requires explicit opt-in flags
        if self.TRADING_MODE == TradingMode.LIVE:
            if not self.LIVE_MODE:
                raise ValueError(
                    "TRADING_MODE=LIVE requires LIVE_MODE=true to be explicitly set. This is a deliberate safety gate."
                )
            if not self.LIVE_ARMED:
                raise ValueError(
                    "TRADING_MODE=LIVE requires LIVE_ARMED=true to be explicitly set. This is a deliberate safety gate."
                )

        if self.TRADING_MODE == TradingMode.CANARY_LIVE:
            if not self.LIVE_MODE:
                raise ValueError("TRADING_MODE=CANARY_LIVE requires LIVE_MODE=true to be explicitly set.")
            if not self.LIVE_ARMED:
                raise ValueError("TRADING_MODE=CANARY_LIVE requires LIVE_ARMED=true to be explicitly set.")
            # SHADOW_MODE defaults to True but must be False in CANARY_LIVE so orders are
            # actually submitted. Auto-clear it here so operators don't need a separate env var.
            self.SHADOW_MODE = False

        if self.TRADING_MODE in (TradingMode.CANARY_LIVE, TradingMode.LIVE):
            if self.BYBIT_USE_TESTNET:
                raise ValueError(
                    f"TRADING_MODE={self.TRADING_MODE.value} requires BYBIT_USE_TESTNET=false. "
                    "Set BYBIT_USE_TESTNET=false to use real Bybit endpoints."
                )

        # Hybrid ML mode sanity check: live model decisions without the canary
        # gate means the model can replace signals but nothing blocks weak ones.
        if self.MODEL_ALLOW_LIVE_DECISIONS and not self.MODEL_GATE_CANARY_ENABLED:
            import warnings as _warnings

            _warnings.warn(
                "MODEL_ALLOW_LIVE_DECISIONS=true with MODEL_GATE_CANARY_ENABLED=false: "
                "the model can replace rule-based decisions but the canary gate will not "
                "block low-score signals. Consider enabling MODEL_GATE_CANARY_ENABLED.",
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# Risk profile parameter dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskProfileConfig:
    """Numerical risk parameters for a given RiskProfile.

    These values are the defaults; they can be overridden by config/profiles.yaml.
    """

    # Position sizing
    max_positions: int
    max_position_size_pct: float
    """Max single position risk as % of equity (Kelly-scaled)."""

    max_position_notional_usd: float
    """Hard cap in USD notional regardless of equity."""

    # Portfolio-level heat
    max_portfolio_heat_pct: float
    """Sum of all open position risks as % of equity."""

    # Drawdown circuit breakers
    max_daily_drawdown_pct: float
    max_weekly_drawdown_pct: float
    max_monthly_drawdown_pct: float

    # Kelly fraction applied to raw model bet sizing
    kelly_fraction: float

    # Leverage limits (1 = no leverage)
    max_leverage: float

    # Regime gates: list of MarketRegime values allowed for new entries
    allowed_regimes: list[str] = field(default_factory=list)

    # Volatility gate
    max_volatility_level: str = "HIGH"
    """Block new entries above this volatility level."""

    # Order parameters
    default_time_in_force: str = "GTC"
    use_post_only: bool = False

    # Slippage tolerance
    max_slippage_bps: float = 20.0

    # Minimum confidence required from the model
    min_confidence: float = 0.55

    # Cooldown between trades on same symbol (seconds)
    cooldown_seconds: int = 300


# Pre-built profile configs matching the spec table
CONSERVATIVE_PROFILE = RiskProfileConfig(
    max_positions=2,
    max_position_size_pct=1.0,
    max_position_notional_usd=500.0,
    max_portfolio_heat_pct=3.0,
    max_daily_drawdown_pct=2.0,
    max_weekly_drawdown_pct=5.0,
    max_monthly_drawdown_pct=10.0,
    kelly_fraction=0.1,
    max_leverage=1.0,
    allowed_regimes=["BULL_TREND", "SIDEWAYS"],
    max_volatility_level="NORMAL",
    use_post_only=True,
    max_slippage_bps=10.0,
    min_confidence=0.65,
    cooldown_seconds=600,
)

MODERATE_PROFILE = RiskProfileConfig(
    max_positions=4,
    max_position_size_pct=2.0,
    max_position_notional_usd=2000.0,
    max_portfolio_heat_pct=6.0,
    max_daily_drawdown_pct=3.0,
    max_weekly_drawdown_pct=8.0,
    max_monthly_drawdown_pct=15.0,
    kelly_fraction=0.2,
    max_leverage=2.0,
    allowed_regimes=["BULL_TREND", "BEAR_TREND", "SIDEWAYS"],
    max_volatility_level="HIGH",
    use_post_only=False,
    max_slippage_bps=20.0,
    min_confidence=0.55,
    cooldown_seconds=300,
)

AGGRESSIVE_PROFILE = RiskProfileConfig(
    max_positions=6,
    max_position_size_pct=4.0,
    max_position_notional_usd=10000.0,
    max_portfolio_heat_pct=12.0,
    max_daily_drawdown_pct=5.0,
    max_weekly_drawdown_pct=12.0,
    max_monthly_drawdown_pct=25.0,
    kelly_fraction=0.3,
    max_leverage=5.0,
    allowed_regimes=["BULL_TREND", "BEAR_TREND", "SIDEWAYS", "HIGH_VOLATILITY"],
    max_volatility_level="EXTREME",
    use_post_only=False,
    max_slippage_bps=40.0,
    min_confidence=0.50,
    cooldown_seconds=120,
)

SCALP_PROFILE = RiskProfileConfig(
    max_positions=8,
    max_position_size_pct=0.75,
    max_position_notional_usd=1500.0,
    max_portfolio_heat_pct=8.0,
    max_daily_drawdown_pct=2.5,
    max_weekly_drawdown_pct=6.0,
    max_monthly_drawdown_pct=12.0,
    kelly_fraction=0.15,
    max_leverage=7.0,
    allowed_regimes=["BULL_TREND", "BEAR_TREND", "SIDEWAYS", "HIGH_VOLATILITY"],
    max_volatility_level="EXTREME",
    use_post_only=False,
    max_slippage_bps=25.0,
    min_confidence=0.45,
    cooldown_seconds=45,
)

RISK_PROFILE_MAP: dict[RiskProfile, RiskProfileConfig] = {
    RiskProfile.CONSERVATIVE: CONSERVATIVE_PROFILE,
    RiskProfile.MODERATE: MODERATE_PROFILE,
    RiskProfile.AGGRESSIVE: AGGRESSIVE_PROFILE,
    RiskProfile.SCALP: SCALP_PROFILE,
}


def get_risk_profile_config(profile: RiskProfile) -> RiskProfileConfig:
    """Return the ``RiskProfileConfig`` for the given profile."""
    return RISK_PROFILE_MAP[profile]
