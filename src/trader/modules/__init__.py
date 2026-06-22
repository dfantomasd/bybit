"""Pluggable runtime modules bound to ``TradingApplication``."""

from trader.modules.market_data import MarketDataModule
from trader.modules.ops import OpsModule
from trader.modules.registry import ModuleRegistry
from trader.modules.trading_loop import TradingLoopModule
from trader.modules.training import TrainingModule

__all__ = [
    "MarketDataModule",
    "ModuleRegistry",
    "OpsModule",
    "TradingLoopModule",
    "TrainingModule",
]
