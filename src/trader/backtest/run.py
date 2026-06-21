"""CLI entry point for simplified strategy backtests."""

from __future__ import annotations

import argparse

from trader.backtest.engine import BacktestConfig, BacktestEngine, generate_synthetic_trend
from trader.strategies.trend import EMAcrossoverStrategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simplified strategy backtest")
    parser.add_argument("--bars", type=int, default=800, help="Number of synthetic candles")
    parser.add_argument("--balance", type=float, default=10_000.0, help="Starting balance in USD")
    parser.add_argument("--min-adx", type=float, default=0.25, help="Minimum normalized ADX")
    parser.add_argument("--min-net-return", type=float, default=0.10, help="Minimum net return percent")
    args = parser.parse_args()

    closes, highs, lows, volumes = generate_synthetic_trend(args.bars)
    strategy = EMAcrossoverStrategy(
        min_adx=args.min_adx,
        min_net_return_pct=args.min_net_return,
    )
    result = BacktestEngine(
        strategy,
        BacktestConfig(initial_balance_usd=args.balance),
    ).run(closes, highs, lows, volumes)

    metrics = result.metrics
    assert metrics is not None
    print("=== Backtest summary ===")
    print(f"Trades:          {metrics.total_trades}")
    print(f"Win rate:        {metrics.win_rate:.1f}%")
    print(f"Net PnL:         {metrics.net_pnl_pct:.2f}%")
    print(f"Avg trade:       {metrics.avg_trade_pnl_pct:.3f}%")
    print(f"Max drawdown:    {metrics.max_drawdown_pct:.2f}%")
    print(f"Sharpe:          {metrics.sharpe_ratio:.2f}")
    print(f"Profit factor:   {metrics.profit_factor:.2f}")


if __name__ == "__main__":
    main()
