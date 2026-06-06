"""Custom exception hierarchy for the Bybit AI trading system.

All exceptions derive from ``TradingSystemError`` so callers can catch the
entire family with a single except clause when needed.
"""

from __future__ import annotations


class TradingSystemError(Exception):
    """Base class for all trading system errors.

    Attributes:
        message:  Human-readable description.
        code:     Optional short error code for structured logging.
        retryable: Whether the operation can be retried automatically.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.retryable = retryable

    def __repr__(self) -> str:
        return f"{type(self).__name__}(message={self.message!r}, code={self.code!r}, retryable={self.retryable})"


# ---------------------------------------------------------------------------
# Configuration / startup errors
# ---------------------------------------------------------------------------


class ConfigurationError(TradingSystemError):
    """Raised when the system configuration is invalid or incomplete."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message, code="CONFIG_ERROR", retryable=False)
        self.field = field


# ---------------------------------------------------------------------------
# Authentication / connectivity errors
# ---------------------------------------------------------------------------


class AuthenticationError(TradingSystemError):
    """Raised when API authentication fails (invalid key, signature, IP block)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="AUTH_ERROR", retryable=False)


class RateLimitError(TradingSystemError):
    """Raised when the exchange rate-limit is hit."""

    def __init__(self, message: str, *, retry_after_ms: int | None = None) -> None:
        super().__init__(message, code="RATE_LIMIT", retryable=True)
        self.retry_after_ms = retry_after_ms


class RegionBlockedError(TradingSystemError):
    """Raised when the operator's region is not permitted by the exchange."""

    def __init__(self, message: str, *, region: str | None = None) -> None:
        super().__init__(message, code="REGION_BLOCKED", retryable=False)
        self.region = region


# ---------------------------------------------------------------------------
# Order / execution errors
# ---------------------------------------------------------------------------


class InsufficientFundsError(TradingSystemError):
    """Raised when account balance is too low to execute the order."""

    def __init__(
        self,
        message: str,
        *,
        required: float | None = None,
        available: float | None = None,
    ) -> None:
        super().__init__(message, code="INSUFFICIENT_FUNDS", retryable=False)
        self.required = required
        self.available = available


class OrderRejectedError(TradingSystemError):
    """Raised when the exchange explicitly rejects an order."""

    def __init__(
        self,
        message: str,
        *,
        order_link_id: str | None = None,
        exchange_code: str | None = None,
    ) -> None:
        super().__init__(message, code="ORDER_REJECTED", retryable=False)
        self.order_link_id = order_link_id
        self.exchange_code = exchange_code


class ReconciliationError(TradingSystemError):
    """Raised when local state cannot be reconciled with exchange state."""

    def __init__(
        self,
        message: str,
        *,
        order_link_ids: list[str] | None = None,
    ) -> None:
        super().__init__(message, code="RECONCILIATION_ERROR", retryable=True)
        self.order_link_ids = order_link_ids or []


class StaleDataError(TradingSystemError):
    """Raised when market data or features are too old to be used safely."""

    def __init__(
        self,
        message: str,
        *,
        data_age_seconds: float | None = None,
        max_age_seconds: float | None = None,
    ) -> None:
        super().__init__(message, code="STALE_DATA", retryable=True)
        self.data_age_seconds = data_age_seconds
        self.max_age_seconds = max_age_seconds


# ---------------------------------------------------------------------------
# Risk / safety errors
# ---------------------------------------------------------------------------


class SafeModeError(TradingSystemError):
    """Raised when an operation is blocked because the system is in safe mode."""

    def __init__(self, message: str, *, triggered_by: str | None = None) -> None:
        super().__init__(message, code="SAFE_MODE", retryable=False)
        self.triggered_by = triggered_by


class KillSwitchError(TradingSystemError):
    """Raised when the kill switch blocks an operation."""

    def __init__(
        self,
        message: str,
        *,
        kill_switch_mode: str | None = None,
    ) -> None:
        super().__init__(message, code="KILL_SWITCH", retryable=False)
        self.kill_switch_mode = kill_switch_mode


# ---------------------------------------------------------------------------
# ML / model errors
# ---------------------------------------------------------------------------


class FeatureError(TradingSystemError):
    """Raised when feature engineering fails or produces invalid output."""

    def __init__(
        self,
        message: str,
        *,
        symbol: str | None = None,
        feature_version: str | None = None,
    ) -> None:
        super().__init__(message, code="FEATURE_ERROR", retryable=True)
        self.symbol = symbol
        self.feature_version = feature_version


class ModelError(TradingSystemError):
    """Raised when model inference fails or produces invalid output."""

    def __init__(
        self,
        message: str,
        *,
        model_id: str | None = None,
        algorithm: str | None = None,
    ) -> None:
        super().__init__(message, code="MODEL_ERROR", retryable=True)
        self.model_id = model_id
        self.algorithm = algorithm


class DataQualityError(TradingSystemError):
    """Raised when incoming market data fails quality checks."""

    def __init__(
        self,
        message: str,
        *,
        symbol: str | None = None,
        quality_score: float | None = None,
        min_required: float | None = None,
    ) -> None:
        super().__init__(message, code="DATA_QUALITY", retryable=True)
        self.symbol = symbol
        self.quality_score = quality_score
        self.min_required = min_required
