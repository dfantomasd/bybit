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

    ENTRY_ORDER_MODE: str = "MARKET"
    """MARKET or POST_ONLY_LIMIT. POST_ONLY_LIMIT uses maker orders with TTL."""
    ENTRY_LIMIT_TTL_SECONDS: int = 5
    """Seconds to wait for a limit entry fill before cancelling."""
    ENTRY_REPRICE_ATTEMPTS: int = 1
    """Max repricing attempts before abandoning a limit entry."""
    ALLOW_TAKER_ENTRY: bool = False
    """Allow market fallback if limit entry fails. False = skip trade."""
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
    """Timeout for WS subscribe acknowledgement per symbol (seconds)."""

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
    """Maximum correlated (same-quote) open positions."""
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
    CANDLE_STORE_MAX_BARS_1M: int = 250
    CANDLE_STORE_MAX_BARS_5M: int = 250
    CANDLE_STORE_MAX_BARS_15M: int = 200
    CANDLE_STORE_MAX_BARS_1H: int = 120

    # ------------------------------------------------------------------
    # Orderbook mode
    # ------------------------------------------------------------------
    ORDERBOOK_MODE: str = "ON_DEMAND"
    """ON_DEMAND = fetch only for top candidates; STREAMING = subscribe for all."""
    MAX_ORDERBOOK_ACTIVE_SYMBOLS: int = 5

    # ------------------------------------------------------------------
    # Adaptive load governor
    # ------------------------------------------------------------------
    ADAPTIVE_LOAD_GOVERNOR_ENABLED: bool = True
    LOAD_GOVERNOR_CHECK_SECONDS: int = 30
    MAX_FEATURE_CYCLE_MS: int = 8000
    MAX_STRATEGY_CYCLE_MS: int = 8000
    MAX_EVENT_LOOP_LAG_MS: int = 500
    MAX_QUEUE_UTILIZATION_PCT: int = 70
    LOAD_GOVERNOR_MIN_FEATURE_SYMBOLS: int = 10
    LOAD_GOVERNOR_MIN_EXECUTION_CANDIDATES: int = 3

    # ------------------------------------------------------------------
    # ML / model
    # ------------------------------------------------------------------
    MODEL_ENABLED: bool = True
    """Enable lightweight supervised challenger model."""
    MODEL_ALLOW_LIVE_DECISIONS: bool = True
    """When False, model only scores in shadow; rule-based strategy remains authoritative.
    When True, a compatible CHAMPION model may replace rule-based decisions (hybrid mode).
    Real orders are still gated by TRADING_MODE/LIVE_MODE/LIVE_ARMED."""
    MODEL_MIN_TRAINING_SAMPLES: int = 500
    MODEL_MIN_CLOSED_TRADES_FOR_PROMOTION: int = 50
    MODEL_SHADOW_SCORING_ENABLED: bool = True
    """Always run shadow scoring even when live decisions disabled."""
    MODEL_AUTO_TRAIN_ENABLED: bool = True
    """Automatically train a shadow challenger when enough new labelled examples accumulate."""
    MODEL_AUTO_TRAIN_MIN_SAMPLES: int = 1000
    MODEL_AUTO_TRAIN_INCREMENT_SAMPLES: int = 1000
    MODEL_AUTO_TRAIN_CHECK_SECONDS: int = 300
    MODEL_AUTO_TRAIN_HORIZON_MINUTES: int = 15
    MODEL_AUTO_TRAIN_LABEL_BPS: float = 5.0
    MODEL_AUTO_PROMOTE_ENABLED: bool = True
    """Auto-promote challenger to champion when it beats the current champion
    AND the lift is statistically significant (bootstrap p-value)."""
    MODEL_AUTO_PROMOTE_CHECK_SECONDS: int = 600
    MODEL_AUTO_PROMOTE_MIN_SIGNALS: int = 50
    MODEL_AUTO_PROMOTE_MIN_LIFT_BPS: float = 1.0
    """Minimum live lift (bps) the challenger must show before auto-promotion."""
    MODEL_AUTO_PROMOTE_PVALUE_THRESHOLD: float = 0.05
    """Maximum bootstrap p-value for auto-promotion: the challenger's mean net
    return must beat the baseline in >= (1 - threshold) of bootstrap resamples."""
    MODEL_AUTO_PROMOTE_BOOTSTRAP_ITERATIONS: int = 1000
    MODEL_AUTO_PROMOTE_MIN_BOOTSTRAP_SAMPLES: int = 50
    """Minimum resolved challenger returns required to run the bootstrap test."""
    MODEL_SHADOW_GATE_ENABLED: bool = True
    """Evaluate a model-based pass/block gate in shadow, without affecting execution."""
    MODEL_SHADOW_GATE_THRESHOLD: float = 0.55
    MODEL_GATE_CANARY_ENABLED: bool = True
    """When enabled, allow the model gate to block entries only with conservative safeguards."""
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
