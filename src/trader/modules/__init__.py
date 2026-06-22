"""Pluggable runtime modules bound to ``TradingApplication``."""

from trader.modules.diagnostics import DiagnosticsModule
from trader.modules.market_data import MarketDataModule
from trader.modules.ops import OpsModule
from trader.modules.registry import ModuleRegistry
from trader.modules.telegram_bridge import TelegramBridgeModule
from trader.modules.trading_loop import TradingLoopModule
from trader.modules.training import TrainingModule

__all__ = [
    "DiagnosticsModule",
    "MarketDataModule",
    "ModuleRegistry",
    "OpsModule",
    "TradingLoopModule",
    "TelegramBridgeModule",
    "TrainingModule",
]
