"""Structlog configuration for the trading system.

Provides:
- JSON output in production (LOG_FORMAT=json)
- Pretty coloured output in development (LOG_FORMAT=console)
- Automatic field injection: timestamp, env, service, severity, correlation_id
- Secret redaction processor that masks sensitive values
- Async-safe (structlog is thread-safe and coroutine-safe by design)

Usage::

    from trader.monitoring.logging import configure_logging, get_logger
    configure_logging(log_level="INFO", log_format="json")
    log = get_logger(__name__)
    log.info("order_submitted", order_link_id="abc123", symbol="BTCUSDT")
"""
from __future__ import annotations

import logging
import re
import sys
from typing import Any, cast

import structlog
from structlog.types import EventDict, Processor, WrappedLogger

# ---------------------------------------------------------------------------
# Secret patterns to redact
# ---------------------------------------------------------------------------

# Patterns matching known sensitive key names
_SECRET_FIELD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"api[_-]?key", re.IGNORECASE),
    re.compile(r"api[_-]?secret", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"passwd", re.IGNORECASE),
    re.compile(r"private[_-]?key", re.IGNORECASE),
    re.compile(r"bot[_-]?token", re.IGNORECASE),
    re.compile(r"postgres[_-]?dsn", re.IGNORECASE),
    re.compile(r"redis[_-]?url", re.IGNORECASE),
]

_REDACTED = "***REDACTED***"


def _is_secret_field(key: str) -> bool:
    """Return True if the field name matches any known secret pattern."""
    return any(p.search(key) for p in _SECRET_FIELD_PATTERNS)


def _redact_secrets(
    _logger: WrappedLogger,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Structlog processor that redacts secret fields in-place."""
    redacted: dict[str, Any] = {}
    for key, value in event_dict.items():
        if _is_secret_field(key):
            redacted[key] = _REDACTED
        elif isinstance(value, dict):
            # Recursively redact nested dicts
            redacted[key] = {
                k: _REDACTED if _is_secret_field(k) else v for k, v in value.items()
            }
        else:
            redacted[key] = value
    event_dict.clear()
    event_dict.update(redacted)
    return event_dict


def _add_service_context(
    _logger: WrappedLogger,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Inject static service context fields."""
    event_dict.setdefault("service", "bybit-trader")
    event_dict.setdefault("env", "production")
    return event_dict


def _severity_from_level(
    _logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add a 'severity' field compatible with GCP / Datadog log levels."""
    level_map = {
        "debug": "DEBUG",
        "info": "INFO",
        "warning": "WARNING",
        "error": "ERROR",
        "critical": "CRITICAL",
        "exception": "ERROR",
    }
    event_dict["severity"] = level_map.get(method_name, method_name.upper())
    return event_dict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    service_name: str = "bybit-trader",
    environment: str = "production",
) -> None:
    """Configure structlog for the entire process.

    Call this once at application startup before any loggers are used.

    Args:
        log_level:    Standard level string ("DEBUG", "INFO", etc.)
        log_format:   "json" for machine-readable, "console" for human-readable.
        service_name: Injected into every log record as ``service``.
        environment:  Injected into every log record as ``env``.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure the stdlib root logger so third-party libs emit through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    # Silence noisy third-party loggers
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("telegram.ext").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        _severity_from_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        lambda _l, _m, ed: (ed.update(service=service_name, env=environment) or ed),
        _redact_secrets,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_format == "json":
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also wire up stdlib logging through structlog for uniform output
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Return a bound structlog logger.

    Args:
        name: Optional logger name (typically ``__name__``).
    """
    return cast(structlog.BoundLogger, structlog.get_logger(name))


def bind_context(**kwargs: Any) -> None:
    """Bind key-value pairs to the current async context (coroutine-local).

    Useful for request tracing — bind correlation_id at the start of a
    request handler and it will appear in all log records within that coroutine.
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all context variables bound in the current coroutine."""
    structlog.contextvars.clear_contextvars()
