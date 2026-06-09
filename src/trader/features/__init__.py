"""Feature engineering module."""

from trader.features.source_candle_guard import install_source_candle_guard

install_source_candle_guard()

__all__ = ["install_source_candle_guard"]
