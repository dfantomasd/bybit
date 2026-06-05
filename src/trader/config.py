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
from typing import Any

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    POSTGRES_DSN: SecretStr = SecretStr(
        "postgresql+asyncpg://trader:trader@postgres:5432/trader"
    )
    REDIS_URL: SecretStr = SecretStr("redis://redis:6379/0")

    # ------------------------------------------------------------------
    # Telegram notifications
    # ------------------------------------------------------------------
    TELEGRAM_BOT_TOKEN: SecretStr = SecretStr("")
    TELEGRAM_ALLOWED_CHAT_IDS: list[int] = []

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

    # ------------------------------------------------------------------
    # Operational
    # ------------------------------------------------------------------
    RECONCILIATION_INTERVAL_SECONDS: int = 30
    HEALTH_CHECK_INTERVAL_SECONDS: int = 15
    DATA_STALENESS_THRESHOLD_SECONDS: int = 5
    FEATURE_STALENESS_THRESHOLD_SECONDS: int = 10
    MODEL_STALENESS_THRESHOLD_SECONDS: int = 3600

    def model_post_init(self, __context: Any) -> None:
        """Enforce critical safety invariants after field parsing."""
        # Testnet must be enabled for non-live modes
        if self.TRADING_MODE in (TradingMode.TESTNET, TradingMode.SHADOW):
            if not self.BYBIT_USE_TESTNET:
                raise ValueError(
                    "BYBIT_USE_TESTNET must be True when TRADING_MODE is "
                    f"{self.TRADING_MODE}. Set BYBIT_USE_TESTNET=true in .env"
                )

        # Live mode requires explicit opt-in flags
        if self.TRADING_MODE == TradingMode.LIVE:
            if not self.LIVE_MODE:
                raise ValueError(
                    "TRADING_MODE=LIVE requires LIVE_MODE=true to be explicitly set. "
                    "This is a deliberate safety gate."
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

RISK_PROFILE_MAP: dict[RiskProfile, RiskProfileConfig] = {
    RiskProfile.CONSERVATIVE: CONSERVATIVE_PROFILE,
    RiskProfile.MODERATE: MODERATE_PROFILE,
    RiskProfile.AGGRESSIVE: AGGRESSIVE_PROFILE,
}


def get_risk_profile_config(profile: RiskProfile) -> RiskProfileConfig:
    """Return the ``RiskProfileConfig`` for the given profile."""
    return RISK_PROFILE_MAP[profile]
