"""Unit tests for the Settings configuration class."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from trader.domain.enums import RiskProfile, TradingMode


class TestSettingsDefaults:
    """Settings loaded from environment variables behave correctly."""

    def _make_settings(self, **env_overrides: str) -> object:
        """Helper: create a Settings instance with controlled env vars."""
        from trader.config import Settings

        base_env = {
            "BYBIT_API_KEY": "test-api-key",
            "BYBIT_API_SECRET": "test-api-secret",
            "BYBIT_USE_TESTNET": "true",
            "POSTGRES_DSN": "postgresql+asyncpg://u:p@localhost:5432/db",
            "REDIS_URL": "redis://localhost:6379/0",
            "TELEGRAM_BOT_TOKEN": "1234567890:test",
            "TELEGRAM_ALLOWED_CHAT_IDS": "[123]",
            "TRADING_MODE": "TESTNET",
            "LIVE_MODE": "false",
        }
        base_env.update(env_overrides)

        with patch.dict(os.environ, base_env, clear=False):
            return Settings(_env_file=None)  # type: ignore[call-arg]

    def test_trading_mode_defaults_to_testnet(self) -> None:
        settings = self._make_settings()
        assert settings.TRADING_MODE == TradingMode.TESTNET  # type: ignore[union-attr]

    def test_risk_profile_defaults_to_conservative(self) -> None:
        settings = self._make_settings()
        assert settings.RISK_PROFILE == RiskProfile.CONSERVATIVE  # type: ignore[union-attr]

    def test_bybit_use_testnet_defaults_true(self) -> None:
        settings = self._make_settings()
        assert settings.BYBIT_USE_TESTNET is True  # type: ignore[union-attr]

    def test_live_mode_defaults_to_false(self) -> None:
        settings = self._make_settings()
        assert settings.LIVE_MODE is False  # type: ignore[union-attr]

    def test_shadow_mode_defaults_to_true(self) -> None:
        settings = self._make_settings()
        assert settings.SHADOW_MODE is True  # type: ignore[union-attr]

    def test_trade_journal_defaults_enabled(self) -> None:
        settings = self._make_settings()
        assert settings.TRADE_JOURNAL_ENABLED is True  # type: ignore[union-attr]
        assert settings.PERFORMANCE_FILTER_ENABLED is True  # type: ignore[union-attr]
        assert settings.PERFORMANCE_MIN_TRADABLE_SYMBOLS == 2  # type: ignore[union-attr]

    def test_profit_manager_defaults_enabled(self) -> None:
        settings = self._make_settings()
        assert settings.PROFIT_MANAGER_ENABLED is True  # type: ignore[union-attr]
        assert settings.TRAILING_STOP_ENABLED is True  # type: ignore[union-attr]

    def test_position_sync_interval_default(self) -> None:
        settings = self._make_settings()
        assert settings.POSITION_SYNC_INTERVAL_SECONDS == 30  # type: ignore[union-attr]

    def test_max_positions_default(self) -> None:
        settings = self._make_settings()
        assert settings.MAX_POSITIONS == 2  # type: ignore[union-attr]

    def test_single_telegram_chat_id_string(self) -> None:
        settings = self._make_settings(TELEGRAM_ALLOWED_CHAT_IDS="-1003976706688")
        assert settings.TELEGRAM_ALLOWED_CHAT_IDS == [-1003976706688]  # type: ignore[union-attr]

    def test_comma_separated_telegram_chat_ids(self) -> None:
        settings = self._make_settings(TELEGRAM_ALLOWED_CHAT_IDS="123, -456")
        assert settings.TELEGRAM_ALLOWED_CHAT_IDS == [123, -456]  # type: ignore[union-attr]


class TestSettingsSecrets:
    """Secret fields must not appear in string representations."""

    def _make_settings(self) -> object:
        from trader.config import Settings

        env = {
            "BYBIT_API_KEY": "super-secret-key",
            "BYBIT_API_SECRET": "super-secret-secret",
            "BYBIT_USE_TESTNET": "true",
            "POSTGRES_DSN": "postgresql+asyncpg://u:p@localhost:5432/db",
            "REDIS_URL": "redis://localhost:6379/0",
            "TELEGRAM_BOT_TOKEN": "bot:token",
            "TELEGRAM_ALLOWED_CHAT_IDS": "[999]",
            "TRADING_MODE": "TESTNET",
            "LIVE_MODE": "false",
        }
        with patch.dict(os.environ, env, clear=False):
            return Settings(_env_file=None)  # type: ignore[call-arg]

    def test_api_key_not_in_str_repr(self) -> None:
        settings = self._make_settings()
        repr_str = str(settings)
        assert "super-secret-key" not in repr_str

    def test_api_secret_not_in_str_repr(self) -> None:
        settings = self._make_settings()
        repr_str = str(settings)
        assert "super-secret-secret" not in repr_str

    def test_bot_token_not_in_str_repr(self) -> None:
        settings = self._make_settings()
        repr_str = str(settings)
        assert "bot:token" not in repr_str

    def test_secret_accessible_via_get_secret_value(self) -> None:
        settings = self._make_settings()
        # SecretStr.get_secret_value() should return the actual value
        assert settings.BYBIT_API_KEY.get_secret_value() == "super-secret-key"  # type: ignore[union-attr]


class TestSettingsSafetyGates:
    """Critical safety invariants are enforced."""

    def _make_settings(self, **env_overrides: str) -> object:
        from trader.config import Settings

        base_env = {
            "BYBIT_API_KEY": "key",
            "BYBIT_API_SECRET": "secret",
            "BYBIT_USE_TESTNET": "true",
            "POSTGRES_DSN": "postgresql+asyncpg://u:p@localhost:5432/db",
            "REDIS_URL": "redis://localhost:6379/0",
            "TELEGRAM_BOT_TOKEN": "tok",
            "TELEGRAM_ALLOWED_CHAT_IDS": "[1]",
            "TRADING_MODE": "TESTNET",
            "LIVE_MODE": "false",
        }
        base_env.update(env_overrides)
        with patch.dict(os.environ, base_env, clear=False):
            return Settings(_env_file=None)  # type: ignore[call-arg]

    def test_live_mode_requires_explicit_opt_in(self) -> None:
        """TRADING_MODE=LIVE without LIVE_MODE=true must raise an error."""
        with pytest.raises(ValueError, match="LIVE_MODE"):
            self._make_settings(
                TRADING_MODE="LIVE",
                BYBIT_USE_TESTNET="false",
                LIVE_MODE="false",
            )

    def test_testnet_mode_with_testnet_false_raises(self) -> None:
        """TESTNET mode must use BYBIT_USE_TESTNET=true."""
        with pytest.raises(ValueError, match="BYBIT_USE_TESTNET"):
            self._make_settings(
                TRADING_MODE="TESTNET",
                BYBIT_USE_TESTNET="false",
            )

    def test_shadow_mode_allows_mainnet_data(self) -> None:
        """SHADOW mode may use mainnet endpoints because it never submits orders."""
        settings = self._make_settings(
            TRADING_MODE="SHADOW",
            BYBIT_USE_TESTNET="false",
        )
        assert settings.TRADING_MODE == TradingMode.SHADOW  # type: ignore[union-attr]
        assert settings.BYBIT_USE_TESTNET is False  # type: ignore[union-attr]

    def test_live_trading_mode_allowed_when_live_mode_true(self) -> None:
        """LIVE mode is permitted only when LIVE_MODE=true is explicitly set."""
        settings = self._make_settings(
            TRADING_MODE="LIVE",
            BYBIT_USE_TESTNET="false",
            LIVE_MODE="true",
        )
        assert settings.TRADING_MODE == TradingMode.LIVE  # type: ignore[union-attr]


class TestRiskProfileConfig:
    """RiskProfileConfig dataclasses have correct values."""

    def test_conservative_profile(self) -> None:
        from trader.config import CONSERVATIVE_PROFILE

        assert CONSERVATIVE_PROFILE.max_positions == 2
        assert CONSERVATIVE_PROFILE.kelly_fraction < 0.2
        assert CONSERVATIVE_PROFILE.max_leverage == 1.0

    def test_moderate_profile(self) -> None:
        from trader.config import MODERATE_PROFILE

        assert MODERATE_PROFILE.max_positions > 2
        assert MODERATE_PROFILE.max_leverage >= 1.0

    def test_aggressive_profile(self) -> None:
        from trader.config import AGGRESSIVE_PROFILE

        assert AGGRESSIVE_PROFILE.max_positions > 4
        assert AGGRESSIVE_PROFILE.max_leverage > 2.0

    def test_scalp_profile(self) -> None:
        from trader.config import SCALP_PROFILE

        assert SCALP_PROFILE.max_positions >= 8
        assert SCALP_PROFILE.min_confidence < 0.50
        assert SCALP_PROFILE.cooldown_seconds <= 60

    def test_get_risk_profile_config_returns_correct(self) -> None:
        from trader.config import CONSERVATIVE_PROFILE, get_risk_profile_config

        cfg = get_risk_profile_config(RiskProfile.CONSERVATIVE)
        assert cfg is CONSERVATIVE_PROFILE

    def test_profiles_frozen(self) -> None:
        from trader.config import CONSERVATIVE_PROFILE

        with pytest.raises((AttributeError, TypeError)):
            CONSERVATIVE_PROFILE.max_positions = 999  # type: ignore[misc]
