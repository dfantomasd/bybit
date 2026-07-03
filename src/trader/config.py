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
    TRADE_JOURNAL_FETCH_TIMEOUT_SECONDS: float = 30.0
    """Per-query read timeout for Postgres diagnostics and journal reads."""
    TRADE_JOURNAL_POOL_MAX_SIZE: int = 5
    """Connection pool size for asyncpg (Render/Supabase free tier)."""
    TRADE_JOURNAL_RECONNECT_MAX_BACKOFF_SECONDS: float = 1800.0
    """Max reconnect backoff after repeated Postgres failures."""
    TRADE_JOURNAL_AUTH_CIRCUIT_BREAKER_MIN_BACKOFF_SECONDS: float = 900.0
    """Minimum reconnect wait after Supabase auth circuit-breaker errors."""
    WS_MARKET_DATA_STALE_RECONNECT_SECONDS: float = 120.0
    """Force public WS reconnect when market data is older than this."""
    # Persist signals, risk decisions, order events, and closed PnL in Postgres.
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
    MAX_SPREAD_BPS_SCALP: float = 5.0
    """Maximum bid-ask spread (bps) for scalp entries. Unknown spread fails closed."""
    MIN_NET_SCALP_RETURN_PCT: float = 0.08
    """Minimum expected NET return (percent) after fees+spread+slippage for a scalp."""
    MIN_NET_TREND_RETURN_PCT: float = 0.10
    """Minimum expected NET return (percent) for EMA trend entries after costs."""
    MIN_NET_ALPHA_RETURN_PCT: float = 0.15
    """Minimum expected NET return (percent) for advanced-alpha entries after costs."""
    MIN_NET_MARKET_MAKING_PCT: float = 0.06
    """Minimum expected NET return (percent) for market-making entries after costs."""
    MIN_NET_STAT_ARB_PCT: float = 0.07
    """Minimum expected NET return (percent) for statistical arbitrage entries after costs."""
    SCALP_COOLDOWN_SECONDS: int = 60
    """Minimum seconds between scalp signals per symbol."""
    SCALP_MAX_TRADES_PER_MINUTE: int = 10
    """Global cap on scalp signals per minute across the whole portfolio."""
    SCALP_MAX_POSITION_NOTIONAL_USD: float = 100.0
    """Hard notional cap per scalp position."""
    SCALP_MIN_OB_IMBALANCE: float = 0.15
    """Required L5 orderbook imbalance agreeing with the signal side (BUY needs
    >= +value, SELL needs <= -value). Missing/stale book data fails closed."""
    SCALP_DISABLE_TREND_STRATEGY: bool = True
    """When RISK_PROFILE=SCALP, skip the slow EMA trend strategy (wide TP/SL)."""
    SCALP_STRATEGY_PRIORITY_ORDER: str = (
        "scalp_micro_v1,order_flow_v1,liquidation_hunting_v1,funding_arbitrage_v1,"
        "statistical_arbitrage_v1,market_making_v1,discovered_rule_v1,ema_crossover_v1"
    )
    """Strategy priority override used when RISK_PROFILE=SCALP."""
    SCALP_STRICT_SHADOW: bool = True
    """On SCALP+SHADOW, apply expectancy and net-edge gates like LIVE (no toxic paper trades)."""
    SHADOW_RELAX_SCALP_FILTERS: bool = True
    """When SHADOW and not SCALP_STRICT_SHADOW, loosen scalp gates for paper-trade data collection."""
    SHADOW_PROBE_ENABLED: bool = True
    """Enable SHADOW-only paper probes so model/order outcomes accumulate when live strategies are silent."""
    SHADOW_PROBE_RESEARCH_PROFILE_V2: bool = True
    """Apply the current research profile in code.

    This version switch is intentionally separate from the older per-setting
    Render variables so a deploy cannot silently retain a stale probe policy.
    """
    SHADOW_PROBE_PAPER_COLLECTION_MODE: bool = True
    """Widen probe regimes in SHADOW so ML labels accumulate (no live orders)."""
    SHADOW_PROBE_PAPER_REGIMES: str = "SIDEWAYS,HIGH_VOLATILITY,UNCERTAIN,BULL_TREND,BEAR_TREND"
    """Regimes allowed for shadow_probe_hv_v2 when paper collection mode is on."""
    SHADOW_PROBE_MIN_ABS_IMBALANCE: float = 0.08
    SHADOW_PROBE_COOLDOWN_SECONDS: int = 180
    SHADOW_PROBE_MAX_NOTIONAL_USD: float = 8.0
    SHADOW_PROBE_MIN_TP_PCT: float = 0.75
    """Minimum gross TP distance for paper probes; must exceed round-trip costs."""
    SHADOW_PROBE_MAX_TP_PCT: float = 1.50
    """Maximum gross TP distance after cost-aware reward/risk adjustment."""
    SHADOW_PROBE_MIN_SL_PCT: float = 0.40
    """Minimum SL distance for paper probes to avoid instant noise stop-outs."""
    SHADOW_PROBE_MIN_NET_RETURN_PCT: float = 0.30
    """Minimum expected net return after round-trip costs before emitting a probe."""
    SHADOW_PROBE_MIN_NET_REWARD_RISK: float = 1.10
    """Minimum reward/loss ratio after round-trip costs; TP is adjusted up to satisfy it."""
    SHADOW_PROBE_MIN_NOTIONAL_BUFFER_PCT: float = 3.0
    """Safety buffer when pre-checking probe notional against exchange min_notional."""
    SHADOW_PROBE_SYMBOL_WARMUP_SECONDS: int = 300
    """Skip probe entries on symbols for this long after screener subscription."""
    SHADOW_PROBE_MAX_OPEN_POSITIONS: int = 4
    """Maximum total open positions allowed before new shadow probes are blocked."""
    SHADOW_PROBE_BURST_MAX_SIGNALS: int = 6
    """Maximum probe signals allowed inside the burst window."""
    SHADOW_PROBE_BURST_WINDOW_SECONDS: int = 300
    """Rolling window used by the probe burst limiter."""
    SHADOW_PROBE_BURST_COOLDOWN_SECONDS: int = 300
    """Global pause after the probe burst limit is reached."""
    SHADOW_PROBE_SELL_ENABLED: bool = False
    """Allow SELL-side paper probes. Disabled by default while SELL baseline is negative."""
    DISCOVERED_RULE_STRATEGY_ENABLED: bool = True
    """Enable SHADOW-only strategy_lab discovered-rule validation when a rule JSON exists."""
    DISCOVERED_RULES_PATH: str = "strategy_lab.json"
    """Path to JSON created by python -m trader.strategy_lab.discover."""
    DISCOVERED_RULE_AUTO_GENERATE: bool = True
    """In SHADOW, generate strategy_lab.json from DB outcomes when no valid rules are checked in."""
    DISCOVERED_RULE_AUTO_GENERATE_TIMEOUT_SECONDS: float = 20.0
    DISCOVERED_RULE_AUTO_GENERATE_MIN_SAMPLES: int = 1000
    DISCOVERED_RULE_AUTO_GENERATE_MIN_TRAIN_COUNT: int = 30
    DISCOVERED_RULE_MIN_VALIDATION_COUNT: int = 10
    DISCOVERED_RULE_MIN_VALIDATION_NET_BPS: float = 0.0
    DISCOVERED_RULE_MAX_RULES: int = 20
    DISCOVERED_RULE_MAX_NOTIONAL_USD: float = 8.0
    DISCOVERED_RULE_TP_PCT: float = 0.75
    DISCOVERED_RULE_SL_PCT: float = 0.40
    DISCOVERED_RULE_MIN_CONFIDENCE: float = 0.52
    SHADOW_PROBE_SIDE_BLOCK_ENABLED: bool = True
    """Block probe entries on symbol+side pairs with persistently negative paper baseline."""
    SHADOW_PROBE_SIDE_MIN_SAMPLES: int = 8
    SHADOW_PROBE_SIDE_BLOCK_AVG_BPS: float = -3.0
    SHADOW_PROBE_QUALITY_FILTER_ENABLED: bool = True
    """Require non-negative recent paper baseline for a symbol+side before new probe entries."""
    SHADOW_PROBE_BASELINE_MIN_AVG_BPS: float = 0.0
    SHADOW_PROBE_BASELINE_MIN_SAMPLES: int = 6
    SHADOW_PROBE_SYMBOL_TOP_N: int = 10
    """When >0, restrict probes to top-N symbols by recent paper baseline. 0 disables the cap."""
    SHADOW_PROBE_SYMBOL_MIN_SAMPLES: int = 6
    SHADOW_PROBE_SYMBOL_MIN_AVG_BPS: float = -1.0
    SHADOW_PROBE_STATS_LOOKBACK_DAYS: int = 7
    SHADOW_PROBE_SYMBOL_LOSS_COOLDOWN_ENABLED: bool = True
    """Temporarily block a probe symbol after a poor local run of shadow closes."""
    SHADOW_PROBE_SYMBOL_LOSS_COOLDOWN_MIN_CLOSED: int = 2
    SHADOW_PROBE_SYMBOL_LOSS_COOLDOWN_WINDOW: int = 3
    SHADOW_PROBE_SYMBOL_LOSS_COOLDOWN_MAX_LOSS_RATE: float = 1.0
    SHADOW_PROBE_SYMBOL_LOSS_COOLDOWN_MIN_AVG_PNL_PCT: float = 0.0
    SHADOW_PROBE_SYMBOL_LOSS_COOLDOWN_SECONDS: int = 900
    SHADOW_PROBE_ALLOWED_REGIMES: str = "HIGH_VOLATILITY"
    """Comma-separated market regimes where shadow probes may fire.

    Production defaults target the only currently positive out-of-sample
    regime. These are paper probes only; live entries remain independently
    gated.
    """
    SHADOW_MIN_ATR_MULTIPLE: float = 0.35
    """Min stop/ATR ratio in shadow sizing (scalp SL is 0.5×ATR; lower value avoids tick-round rejects)."""
    TREND_STRATEGY_ENABLED: bool = True
    """Enable the EMA crossover trend strategy in the ensemble."""
    TREND_MIN_ADX: float = 0.25
    """Minimum normalized ADX for EMA trend entries. 0.25 means ADX 25."""
    TREND_BLOCK_NEGATIVE_FUNDING_OI: bool = True
    """Block fragile trend entries when funding and open interest context disagrees."""
    TREND_MTF_CONFIRMATION_ENABLED: bool = True
    """Require higher-timeframe trend confirmation before accepting 1m EMA signals."""
    TREND_CONFIRMATION_INTERVALS: str = "5,15"
    """Comma-separated intervals used to confirm trend entries."""

    # ------------------------------------------------------------------
    # Orderbook microstructure feed
    # ------------------------------------------------------------------
    ORDERBOOK_FEED_ENABLED: bool = True
    """Subscribe to orderbook.50 for execution candidates and derive imbalance/
    microprice features. Adds ~5-10 KB/s WS traffic per tracked symbol."""
    TRADE_FLOW_FEED_ENABLED: bool = True
    """Subscribe to publicTrade for execution candidates and derive order-flow pressure."""
    LIQUIDATION_FEED_ENABLED: bool = True
    """Subscribe to liquidation prints for execution candidates."""
    FLOW_TRACKER_WINDOW_SECONDS: float = 60.0
    """Rolling window for trade-flow and liquidation-pressure strategies."""
    FLOW_LARGE_TRADE_NOTIONAL_USD: float = 10_000.0
    """Trade notional threshold considered a large tape print."""

    # ------------------------------------------------------------------
    # Advanced alpha strategies
    # ------------------------------------------------------------------
    ORDER_FLOW_STRATEGY_ENABLED: bool = True
    """Enable order-flow strategy using tape pressure + orderbook confirmation."""
    ORDER_FLOW_MIN_IMBALANCE: float = 0.35
    ORDER_FLOW_MIN_BOOK_IMBALANCE: float = 0.18
    FUNDING_ARB_STRATEGY_ENABLED: bool = True
    """Enable funding-rate mean-reversion entries."""
    FUNDING_ARB_MIN_ABS_BPS: float = 8.0
    """Minimum absolute funding rate (bps) to trigger fade. Raised from 5→8 to reduce noise."""
    LIQUIDATION_HUNTING_STRATEGY_ENABLED: bool = True
    """Enable liquidation-exhaustion fade entries."""
    LIQUIDATION_HUNTING_MIN_NOTIONAL_USD: float = 50_000.0
    """Minimum liquidation notional (USD) to trigger fade. Raised from $20k→$50k."""
    LIQUIDATION_HUNTING_MIN_IMBALANCE: float = 0.72
    """Minimum liquidation directional imbalance. Raised from 0.65→0.72."""
    VOLATILITY_SQUEEZE_STRATEGY_ENABLED: bool = True
    """Enable Bollinger Band squeeze-breakout entries."""
    VOLATILITY_SQUEEZE_BB_BANDWIDTH: float = 0.018
    """BB bandwidth threshold below which we detect a squeeze."""
    VOLATILITY_SQUEEZE_COOLDOWN_SECONDS: int = 120
    """Per-symbol cooldown between squeeze-breakout signals."""
    MEAN_REVERSION_STRATEGY_ENABLED: bool = True
    """Enable simple RSI mean-reversion entries (oversold/overbought)."""
    MACD_ZEROCROSS_STRATEGY_ENABLED: bool = True
    """Enable MACD histogram zero-cross momentum reversal entries."""
    ATR_BREAKOUT_STRATEGY_ENABLED: bool = True
    """Enable ATR range breakout entries with volume confirmation."""
    MARKET_MAKING_STRATEGY_ENABLED: bool = True
    """Enable maker-first mean-reversion proxy for the current single-order engine."""
    MARKET_MAKING_MIN_SPREAD_BPS: float = 1.2
    MARKET_MAKING_MAX_SPREAD_BPS: float = 4.0
    STAT_ARB_STRATEGY_ENABLED: bool = True
    """Enable single-symbol statistical mean-reversion entries."""
    STAT_ARB_MIN_ZSCORE: float = 2.0
    STRATEGY_PRIORITY_ORDER: str = (
        "order_flow_v1,liquidation_hunting_v1,funding_arbitrage_v1,"
        "mean_reversion_v1,macd_zerocross_v1,atr_breakout_v1,"
        "volatility_squeeze_v1,statistical_arbitrage_v1,market_making_v1,"
        "discovered_rule_v1,scalp_micro_v1,ema_crossover_v1"
    )
    """Higher-priority strategies win ensemble conflicts when directions disagree."""

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
    BUCKET_MIN_SAMPLES: int = 200
    """Minimum resolved outcomes in a bucket before it can be blocked."""
    BUCKET_BLOCK_AVG_BPS: float = -10.0
    """Block a bucket when its average net return is below this (bps)."""
    BUCKET_STATS_REFRESH_SECONDS: int = 3600
    """How often the in-memory bucket statistics are refreshed from Postgres."""
    HOUR_BLOCK_ENABLED: bool = True
    """Fallback gate for toxic UTC hours when regime buckets are too sparse."""
    HOUR_MIN_SAMPLES: int = 30
    """Minimum resolved outcomes in an hour before the fallback gate can block."""
    HOUR_BLOCK_AVG_BPS: float = -10.0
    """Block an UTC hour when average net return is below this threshold."""
    STRATEGY_BLOCK_ENABLED: bool = True
    """Block strategies whose resolved net expectancy is not positive."""
    STRATEGY_MIN_SAMPLES: int = 20
    """Exploration budget before a strategy must prove positive net expectancy."""
    STRATEGY_BLOCK_AVG_BPS: float = 0.0
    """Block a strategy once its average net return falls below this threshold."""
    STRATEGY_REGIME_BLOCK_ENABLED: bool = True
    """Block only the strategy+market-regime combinations with negative expectancy."""
    STRATEGY_REGIME_MIN_SAMPLES: int = 12
    """Minimum resolved outcomes for a strategy+regime pair before it can be blocked."""
    STRATEGY_REGIME_BLOCK_AVG_BPS: float = -2.0
    """Block a strategy+regime pair when its average net return is below this (bps)."""
    SYMBOL_SIDE_BLOCK_ENABLED: bool = True
    """Block symbol+side pairs whose resolved baseline expectancy is persistently negative."""
    SYMBOL_SIDE_MIN_SAMPLES: int = 20
    """Minimum resolved outcomes for a symbol+side pair before it can be blocked."""
    SYMBOL_SIDE_BLOCK_AVG_BPS: float = -2.0
    """Block a symbol+side pair when its average net return is below this (bps)."""

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
    CANDLE_SEED_RETRY_ATTEMPTS: int = 3
    """Retries for startup CandleStore seed requests that hit Bybit rate limits."""
    CANDLE_SEED_RETRY_BASE_DELAY_SECONDS: float = 1.0
    """Base exponential backoff delay after a rate-limited startup seed request."""
    CANDLE_SEED_USE_DB_CACHE: bool = True
    """Load startup CandleStore from Postgres before falling back to Bybit REST."""
    CANDLE_SEED_DB_MIN_BARS: int = 200
    """Minimum confirmed DB bars per symbol/interval required to skip REST seeding."""

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
    PREFLIGHT_POSTGRES_RETRY_ATTEMPTS: int = 6
    """Startup Postgres health retries before failing deploy (Supabase cold start / transient SSL)."""
    PREFLIGHT_POSTGRES_RETRY_DELAY_SECONDS: float = 2.0
    """Base delay between Postgres preflight retries (linear backoff)."""
    PREFLIGHT_POSTGRES_REQUIRED: bool | None = None
    """When None, required only for CANARY_LIVE/LIVE. SHADOW may start without Postgres."""
    PREFLIGHT_POSTGRES_OPTIONAL_MAX_ATTEMPTS: int = 3
    """Quick Postgres probes when preflight is optional (SHADOW); avoids long deploy stalls."""

    # ------------------------------------------------------------------
    # Data retention / cold export
    # ------------------------------------------------------------------
    DATA_RETENTION_ENABLED: bool = True
    DATA_RETENTION_INTERVAL_HOURS: float = 24.0
    DATA_RETENTION_RUN_ON_STARTUP: bool = True
    """Purge stale rows once after Postgres connects (reduces Supabase bloat)."""
    DATA_RETENTION_EXPORT_ENABLED: bool = True
    DATA_RETENTION_EXPORT_DIR: str = "data/retention_exports"
    MARKET_CANDLE_PERSIST_INTERVALS: str = "1,5,15,60"
    """Comma-separated kline intervals written to Postgres.

    1m: outcome resolver + training labels (largest volume; short retention).
    5m/15m: MTF patterns + horizon labels (moderate volume).
    60m: sparse (~24 bars/day/symbol); cheap regime context for 60m horizon models.
    """
    CANDLE_RETENTION_DAYS_1M: int = 14
    CANDLE_RETENTION_DAYS_5M: int = 60
    CANDLE_RETENTION_DAYS_15M: int = 90
    CANDLE_RETENTION_DAYS_60M: int = 180
    FEATURE_SNAPSHOT_RETENTION_DAYS: int = 45
    FEATURE_SNAPSHOT_INVALID_RETENTION_DAYS: int = 3
    FEATURE_SNAPSHOT_ORPHAN_RETENTION_DAYS: int = 14
    """Delete snapshots with no linked prediction_events after this many days."""
    PREDICTION_EVENT_ORPHAN_RETENTION_DAYS: int = 14
    PREDICTION_OUTCOME_RETENTION_DAYS: int = 90
    SHADOW_SIGNAL_RETENTION_DAYS: int = 14
    RESOLVED_SNAPSHOT_EXPORT_BEFORE_DELETE_DAYS: int = 45

    # ------------------------------------------------------------------
    # Telegram notifications
    # ------------------------------------------------------------------
    TELEGRAM_BOT_TOKEN: SecretStr = SecretStr("")
    TELEGRAM_ALLOWED_CHAT_IDS: Annotated[list[int], NoDecode] = []
    TELEGRAM_DELIVERY_MODE: str = "auto"
    """Telegram update delivery: auto (webhook on Render), polling, or webhook."""
    TELEGRAM_WEBHOOK_URL: str = ""
    """Full webhook URL. When empty, auto uses RENDER_EXTERNAL_URL + /telegram/webhook."""
    TELEGRAM_WEBHOOK_SECRET: SecretStr = SecretStr("")
    """Optional X-Telegram-Bot-Api-Secret-Token for webhook requests."""
    TELEGRAM_POLLING_CONFLICT_RECOVERY_WAIT_SECONDS: float = 10.0
    """Seconds to wait after repeated getUpdates conflicts before resuming polling."""
    TELEGRAM_POLLING_WATCHDOG_INTERVAL_SECONDS: float = 30.0
    """How often the in-bot polling watchdog checks Telegram health."""
    TELEGRAM_POLLING_ZOMBIE_SILENCE_SECONDS: float = 180.0
    """Force-restart polling when updater claims running but no handler activity."""

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
    MICRO_ACCOUNT_BALANCE_USD: float = 50.0
    """Balances below this use MICRO_ACCOUNT_MIN_NOTIONAL_BUFFER_PCT instead."""
    MICRO_ACCOUNT_MIN_NOTIONAL_BUFFER_PCT: float = 1.0
    """Reduced min-notional buffer for micro accounts (e.g. ~$23 testnet wallets)."""
    LIVE_REQUIRE_LIQUIDITY_FOR_SIZING: bool = True
    """Require fresh turnover_24h data before sizing entries in LIVE/CANARY_LIVE."""

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
    """WS subscribe watchdog: retry/resubscribe if no 1m kline arrives within this many seconds."""
    SCREENER_SUBSCRIBE_MAX_RETRIES: int = 3
    """Force WS reconnect after this many subscribe retries for the same symbol."""

    SCREENER_DENYLIST: list[str] = []
    """Symbols explicitly excluded (pre-market, innovation zone, etc.)."""

    # ------------------------------------------------------------------
    # Screener tuning for Render Starter
    # ------------------------------------------------------------------
    STARTER_OPTIMIZED_MODE: bool = True
    """Apply Render Starter memory/CPU conservative defaults."""
    STARTER_LIGHT_WS_FEEDS: bool = True
    """On Starter, skip publicTrade/allLiquidation WS feeds (keeps klines + selective orderbook)."""
    STARTER_SHADOW_MINIMAL_STRATEGIES: bool = True
    """On Starter+SHADOW, disable advanced alpha strategies to save RAM/CPU."""
    STARTER_DEFER_TRAINING_UNDER_LOAD: bool = True
    """Skip auto-training while the load governor has reduced the feature universe."""
    WS_PUBLIC_EVENT_QUEUE_MAXSIZE: int = 5000
    """Max buffered public WS events before backpressure/drops."""
    FLOW_TRACKER_HISTORY_SLOTS: int = 512
    """Ring buffer size per symbol in FlowTracker."""

    # ------------------------------------------------------------------
    # Burst / entry rate limiting
    # ------------------------------------------------------------------
    MAX_NEW_ENTRIES_PER_MINUTE: int = 4
    """Maximum new position entries allowed per minute (burst guard)."""
    MAX_CONCURRENT_PENDING_ENTRIES: int = 1
    """Maximum simultaneous positions in SUBMITTING/REST_ACCEPTED state."""
    MAX_SAME_SIDE_POSITIONS: int = 2
    """Maximum open positions on the same side (Buy or Sell)."""
    MAX_CORRELATED_POSITIONS: int = 2
    """Maximum open positions in the same crypto family (BTC/ETH/SOL cluster)."""
    STARTUP_WARMUP_SECONDS: int = 180
    """Seconds after startup before new entries are allowed (monitoring-only phase)."""
    SHADOW_LOSS_GUARD_ENABLED: bool = False
    """Temporarily block new entries after a poor run of shadow TP/SL outcomes."""
    SHADOW_LOSS_GUARD_MIN_CLOSED: int = 3
    """Minimum recent shadow closes before the loss guard can activate."""
    SHADOW_LOSS_GUARD_WINDOW: int = 5
    """How many recent shadow closes are evaluated for the loss guard."""
    SHADOW_LOSS_GUARD_MAX_LOSS_RATE: float = 0.6
    """Activate when recent loss rate is at or above this fraction."""
    SHADOW_LOSS_GUARD_MIN_AVG_PNL_PCT: float = -0.05
    """Activate when recent average shadow PnL percent is at or below this value."""
    SHADOW_LOSS_GUARD_COOLDOWN_SECONDS: int = 900
    """How long new entries stay blocked after the shadow loss guard activates."""

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
    """Cap orderbook WS subscriptions in STREAMING mode."""

    # ------------------------------------------------------------------
    # Adaptive load governor
    # ------------------------------------------------------------------
    ADAPTIVE_LOAD_GOVERNOR_ENABLED: bool = True
    LOAD_GOVERNOR_CHECK_SECONDS: int = 30
    OUTCOME_RESOLVER_INTERVAL_SECONDS: int = 60
    """How often to resolve pending prediction outcomes from stored candles."""
    OUTCOME_RESOLVER_BATCH_LIMIT: int = 1000
    """Maximum prediction outcomes to resolve per horizon on each resolver pass."""
    MAX_FEATURE_CYCLE_MS: int = 8000
    MAX_STRATEGY_CYCLE_MS: int = 8000
    MAX_EVENT_LOOP_LAG_MS: int = 500
    MAX_QUEUE_UTILIZATION_PCT: int = 70
    LOAD_GOVERNOR_MIN_FEATURE_SYMBOLS: int = 10
    LOAD_GOVERNOR_MIN_EXECUTION_CANDIDATES: int = 3

    # ------------------------------------------------------------------
    # ML / model
    # ------------------------------------------------------------------
    ML_UNIFIED_MODEL_DIR: str = "data/ml_unified_models"
    """Directory for unified ML artifacts (Kelly, regime, fusion, spread, stoploss).
    Prefer a persistent volume path on Render; /tmp is cleared on redeploy."""
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
    """CSV label thresholds in bps evaluated during offline selection.
    0 bps = break-even after costs (net-cost-aware label already deducts fees)."""
    MODEL_LABEL_HORIZON: int = 5
    """Bars ahead to measure label outcome. Use 5 for SCALP, 15-30 for swing."""
    MODEL_LABEL_USE_TPSL_EXIT: bool = True
    """Resolve training labels using scalp TP/SL first-touch exits instead of horizon close."""
    MODEL_LABEL_TP_ATR_MULT: float = 1.0
    """Take-profit distance as ATR(14) multiple for TP/SL label resolution."""
    MODEL_LABEL_SL_ATR_MULT: float = 0.5
    """Stop-loss distance as ATR(14) multiple for TP/SL label resolution."""
    MODEL_MIN_PASS_COUNT_FOR_PROMOTION: int = 20
    """Minimum model-pass observations expected before trusting promotion metrics."""
    TRAIN_EXCLUDE_NEGATIVE_BUCKETS: bool = True
    TRAIN_STRATEGY_ALLOWLIST: str = "scalp_micro_v1,shadow_probe_hv_v2,discovered_rule_v1"
    """CSV strategy ids for training. Empty = all RULE_BASELINE_V1 labels."""
    TRAIN_INCLUDE_CANDLE_BASELINE: bool = True
    """When allowlist is set, also include SHADOW_CANDLE/HISTORICAL_REAL baselines.

    This is safe for live execution because challenger promotion still requires
    positive walk-forward, shadow-gate lift, and paper-gate PnL. Keeping it on
    prevents fresh v2 outcomes collected by the candle sampler from being
    invisible to auto-training when strategy-signal samples are sparse.
    """
    TRAIN_LABEL_SPREAD_BPS: float = 4.0
    """Spread component in the training label cost model (scalp max spread is ~5 bps)."""
    TRAIN_MIN_BUCKET_SAMPLES: int = 50
    TRAIN_BUCKET_MIN_AVG_BPS: float = -5.0
    """Optional training filter: remove regime/hour buckets with stable negative expectancy."""
    MODEL_SHADOW_SCORING_ENABLED: bool = True
    """Always run shadow scoring even when live decisions disabled."""
    MODEL_AUTO_TRAIN_ENABLED: bool = True
    """Automatically train a shadow challenger when enough new labelled examples accumulate."""
    MODEL_AUTO_TRAIN_MIN_SAMPLES: int = 1000
    MODEL_AUTO_TRAIN_SCHEMA_CHANGE_MIN_SAMPLES: int = 50
    """Minimum samples required to fire auto-training when the loaded model uses a stale feature
    schema (predict() returns None for every candle). Lower than MIN_SAMPLES because any working
    model is better than no model — we can retrain again once more samples accumulate."""
    MODEL_AUTO_TRAIN_INCREMENT_SAMPLES: int = 5000
    MODEL_AUTO_TRAIN_CHECK_SECONDS: int = 300
    MODEL_AUTO_TRAIN_MIN_INTERVAL_SECONDS: int = 21600
    """Minimum time between successful auto-training runs. Prevents checkpoint churn
    while a challenger has not yet accumulated shadow evidence."""
    MODEL_AUTO_TRAIN_HORIZON_MINUTES: int = 5
    MODEL_AUTO_TRAIN_RETRAIN_IF_WEAK: bool = False
    """Retrain automatically when the latest model quality is WEAK/missing."""
    MODEL_DRIFT_DETECTION_ENABLED: bool = True
    """Monitor feature distribution drift (PSI) and optionally trigger retrain."""
    MODEL_DRIFT_PSI_THRESHOLD: float = 0.25
    MODEL_DRIFT_MIN_SAMPLES: int = 200
    MODEL_DRIFT_AUTO_RETRAIN: bool = False
    MODEL_ONLINE_LEARNING_ENABLED: bool = False
    """Apply challenger partial_fit after resolved outcomes (SGD/LOGREG only)."""
    MODEL_ONLINE_LEARNING_MAX_UPDATES_PER_CYCLE: int = 50
    MODEL_ONLINE_LEARNING_CHECKPOINT_EVERY: int = 25
    """Persist challenger artifact to Postgres after this many online partial_fit updates."""
    MODEL_AUTO_TRAIN_LABEL_BPS: float = 2.0
    MODEL_AUTO_PROMOTE_ENABLED: bool = False
    """Auto-promote challenger to champion when it beats the current champion
    AND the lift is statistically significant (bootstrap p-value).
    Disabled by default: let the model train and accumulate shadow evidence first,
    then enable explicitly via env."""
    MODEL_AUTO_PROMOTE_CHECK_SECONDS: int = 600
    MODEL_AUTO_PROMOTE_MIN_SIGNALS: int = 30
    MODEL_AUTO_PROMOTE_MIN_LIFT_BPS: float = 0.5
    """Minimum live lift (bps) the challenger must show before auto-promotion."""
    MODEL_AUTO_PROMOTE_MIN_PASS_EXPECTANCY_BPS: float = 0.0
    """Minimum average net return (bps) among challenger GATE_PASS outcomes."""
    MODEL_AUTO_PROMOTE_MIN_WF_BPS: float = 0.0
    """Minimum walk-forward expectancy (bps) stored in challenger training metrics."""
    MODEL_AUTO_PROMOTE_MIN_WF_POSITIVE_FOLDS: int = 2
    """Minimum positive walk-forward folds required before auto-promotion."""
    MODEL_AUTO_PROMOTE_MAX_WF_STD_BPS: float = 25.0
    """Maximum walk-forward fold standard deviation allowed before auto-promotion."""
    MODEL_AUTO_PROMOTE_MIN_QUALITY: str = "WEAK"
    """Minimum stored training quality allowed for shadow auto-promotion.

    Canary/live gating remains controlled separately by MODEL_GATE_CANARY_MIN_QUALITY.
    """
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
    MODEL_CHAMPION_MIN_PAPER_GATE_COUNT: int = 50
    """Minimum paper-gate sample count required for walk-forward champion selection."""
    ECONOMIC_READINESS_REQUIRED_FOR_ACTIVE: bool = True
    """Require positive model/paper evidence before CANARY_LIVE or LIVE startup."""
    MODEL_SHADOW_GATE_ENABLED: bool = True
    """Evaluate a model-based pass/block gate in shadow, without affecting execution."""
    MODEL_SHADOW_GATE_THRESHOLD: float = 0.50
    CANDLE_SAMPLER_SHADOW_GATE_MIN_PASS_RATE_PCT: float = 20.0
    """Minimum exploratory pass rate for candle-sampler shadow scoring.

    This never affects live/canary execution. It only prevents observational
    shadow-gate stats from starving when a weak/new challenger scores every
    candle below the strict production threshold.
    """
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

    def _apply_starter_memory_caps(self) -> None:
        """Clamp resource-heavy settings to Starter-safe ceilings."""
        ob_ceiling = 4
        if self.SHADOW_PROBE_PAPER_COLLECTION_MODE:
            ob_ceiling = min(6, max(4, int(self.SCREENER_EXECUTION_CANDIDATES)))

        ceilings: dict[str, int | float] = {
            "SCREENER_WIDE_MAX_SYMBOLS": 15,
            "SCREENER_FEATURE_MAX_SYMBOLS": 8,
            "SCREENER_EXECUTION_CANDIDATES": 6,
            "MAX_ORDERBOOK_ACTIVE_SYMBOLS": ob_ceiling,
            "CANDLE_STORE_MAX_BARS_1M": 150,
            "CANDLE_STORE_MAX_BARS_5M": 150,
            "CANDLE_STORE_MAX_BARS_15M": 120,
            "CANDLE_STORE_MAX_BARS_1H": 80,
            "OUTCOME_RESOLVER_BATCH_LIMIT": 300,
            "LOAD_GOVERNOR_MIN_FEATURE_SYMBOLS": 4,
            "FLOW_TRACKER_HISTORY_SLOTS": 128,
            "WS_PUBLIC_EVENT_QUEUE_MAXSIZE": 2000,
        }
        for setting_name, ceiling in ceilings.items():
            current = getattr(self, setting_name)
            if isinstance(current, int | float) and current > ceiling:
                setattr(self, setting_name, type(current)(ceiling))

        self.MODEL_ONLINE_LEARNING_ENABLED = False
        if self.STARTER_LIGHT_WS_FEEDS:
            self.TRADE_FLOW_FEED_ENABLED = False
            self.LIQUIDATION_FEED_ENABLED = False
        if self.STARTER_SHADOW_MINIMAL_STRATEGIES and self.TRADING_MODE == TradingMode.SHADOW:
            self.ORDER_FLOW_STRATEGY_ENABLED = False
            self.FUNDING_ARB_STRATEGY_ENABLED = False
            self.LIQUIDATION_HUNTING_STRATEGY_ENABLED = False
            self.VOLATILITY_SQUEEZE_STRATEGY_ENABLED = False
            self.MARKET_MAKING_STRATEGY_ENABLED = False
            self.STAT_ARB_STRATEGY_ENABLED = False

    def model_post_init(self, __context: Any) -> None:
        """Enforce critical safety invariants after field parsing."""
        if self.SHADOW_PROBE_RESEARCH_PROFILE_V2:
            # Existing Render services retain dashboard env values even when
            # render.yaml changes. Keep the versioned research cohort isolated
            # in every Settings consumer (runtime, diagnostics, and trainer
            # subprocess) regardless of stale deployment overrides.
            # Include basic ensemble strategies so their signals feed training
            # data and speed up schema-change sample accumulation.
            self.SHADOW_PROBE_MIN_ABS_IMBALANCE = 0.04
            self.SHADOW_PROBE_MIN_TP_PCT = 0.60
            self.SHADOW_PROBE_MAX_TP_PCT = 1.50
            self.SHADOW_PROBE_MIN_SL_PCT = 0.25
            self.SHADOW_PROBE_MIN_NET_RETURN_PCT = 0.12
            self.SHADOW_PROBE_MIN_NET_REWARD_RISK = 1.10
            self.SHADOW_PROBE_SYMBOL_WARMUP_SECONDS = 60
            self.SHADOW_PROBE_SELL_ENABLED = True
            self.SHADOW_PROBE_SIDE_BLOCK_ENABLED = True
            self.SHADOW_PROBE_QUALITY_FILTER_ENABLED = True
            self.SCALP_STRICT_SHADOW = True
            self.BUCKET_STATS_REFRESH_SECONDS = 300
            self.SHADOW_LOSS_GUARD_ENABLED = True
            self.SHADOW_LOSS_GUARD_MIN_CLOSED = 5
            self.SHADOW_LOSS_GUARD_WINDOW = 5
            self.SHADOW_LOSS_GUARD_COOLDOWN_SECONDS = 300
            self.TRAIN_STRATEGY_ALLOWLIST = (
                "scalp_micro_v1,shadow_probe_hv_v2,discovered_rule_v1,"
                "mean_reversion_v1,macd_zerocross_v1,atr_breakout_v1"
            )
            self.TRAIN_INCLUDE_CANDLE_BASELINE = True

        if (
            self.STARTER_OPTIMIZED_MODE
            and self.SCREENER_MAX_PRICE_USD <= 0
            and self.SCREENER_MIN_PRICE_USD < 25.0
        ):
            self.SCREENER_MAX_PRICE_USD = 25.0
        if (
            self.SCREENER_MAX_PRICE_USD > 0
            and self.SCREENER_MIN_PRICE_USD > 0
            and self.SCREENER_MIN_PRICE_USD > self.SCREENER_MAX_PRICE_USD
        ):
            raise ValueError(
                f"SCREENER_MIN_PRICE_USD ({self.SCREENER_MIN_PRICE_USD}) must be <= "
                f"SCREENER_MAX_PRICE_USD ({self.SCREENER_MAX_PRICE_USD})"
            )
        if self.STARTER_OPTIMIZED_MODE:
            self._apply_starter_memory_caps()
        if self.STARTER_OPTIMIZED_MODE and self.SHADOW_PROBE_PAPER_COLLECTION_MODE:
            min_ob = min(6, max(4, int(self.SCREENER_EXECUTION_CANDIDATES)))
            if self.MAX_ORDERBOOK_ACTIVE_SYMBOLS < min_ob:
                self.MAX_ORDERBOOK_ACTIVE_SYMBOLS = min_ob

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
            if self.MODEL_ALLOW_LIVE_DECISIONS and not self.MODEL_ENCRYPT_KEY.get_secret_value().strip():
                raise ValueError(
                    "MODEL_ENCRYPT_KEY must be set when MODEL_ALLOW_LIVE_DECISIONS=true in LIVE/CANARY_LIVE. "
                    "Model artifacts are pickle — encrypt at rest to prevent code execution from a compromised database."
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

        if not self.MULTITIMEFRAME_ENABLED and self.TREND_MTF_CONFIRMATION_ENABLED:
            import warnings as _warnings

            _warnings.warn(
                "TREND_MTF_CONFIRMATION_ENABLED=true with MULTITIMEFRAME_ENABLED=false: "
                "disabling MTF trend confirmation because higher-TF candles are unavailable.",
                stacklevel=2,
            )
            self.TREND_MTF_CONFIRMATION_ENABLED = False

    def market_candle_persist_intervals(self) -> frozenset[str]:
        """Kline intervals persisted to Postgres (others remain in-memory only)."""
        return frozenset(part.strip() for part in self.MARKET_CANDLE_PERSIST_INTERVALS.split(",") if part.strip())


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
