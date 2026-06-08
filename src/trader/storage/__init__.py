"""Storage integrations."""

from trader.storage.directional_trade_journal import install_directional_trade_journal

install_directional_trade_journal()

__all__ = ["install_directional_trade_journal"]
