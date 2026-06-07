"""Offline backfill CLI — fetches historical candles from Bybit REST and writes to PostgreSQL.

Usage:
    python -m trader.training.backfill --symbol BTCUSDT --interval 1 --days 7

NEVER runs inside the trading process.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import click


async def _backfill(symbol: str, interval: str, days: int) -> None:
    import asyncpg

    from trader.config import Settings

    settings = Settings()
    dsn = settings.POSTGRES_DSN.get_secret_value().replace("postgresql+asyncpg://", "postgresql://", 1)

    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2, statement_cache_size=0)

    # Minimal import of REST client for historical data
    try:
        import aiohttp

        base = "https://api.bybit.com"
        end_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
        start_ms = int((datetime.now(tz=UTC) - timedelta(days=days)).timestamp() * 1000)
        interval_ms_map = {"1": 60_000, "5": 300_000, "15": 900_000, "60": 3_600_000}
        bar_ms = interval_ms_map.get(interval, 60_000)
        limit = 200
        inserted = 0

        async with aiohttp.ClientSession() as session:
            cursor = end_ms
            while cursor > start_ms:
                url = f"{base}/v5/market/kline"
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "interval": interval,
                    "end": cursor,
                    "limit": limit,
                }
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()

                if data.get("retCode") != 0:
                    click.echo(f"API error: {data.get('retMsg')}", err=True)
                    break

                candles = data.get("result", {}).get("list", [])
                if not candles:
                    break

                for c in candles:
                    open_time_ms = int(c[0])
                    if open_time_ms < start_ms:
                        continue
                    open_time = datetime.fromtimestamp(open_time_ms / 1000, tz=UTC)
                    close_time = datetime.fromtimestamp((open_time_ms + bar_ms - 1) / 1000, tz=UTC)
                    async with pool.acquire() as conn:
                        await conn.execute(
                            """
                            INSERT INTO market_candles (
                                symbol, interval, open_time, close_time, open, high, low, close,
                                volume, turnover, confirmed, source
                            )
                            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                            ON CONFLICT (symbol, interval, open_time) DO NOTHING
                            """,
                            symbol,
                            interval,
                            open_time,
                            close_time,
                            Decimal(c[1]),
                            Decimal(c[2]),
                            Decimal(c[3]),
                            Decimal(c[4]),
                            Decimal(c[5]),
                            Decimal(c[6]),
                            True,
                            "rest_backfill",
                        )
                    inserted += 1

                oldest_in_batch = int(candles[-1][0])
                cursor = oldest_in_batch - bar_ms
                await asyncio.sleep(0.2)

        click.echo(f"Backfill complete: {inserted} candles inserted for {symbol}/{interval}")
    finally:
        await pool.close()


@click.command()
@click.option("--symbol", default="BTCUSDT", help="Trading symbol")
@click.option("--interval", default="1", help="Candle interval (1, 5, 15, 60)")
@click.option("--days", default=7, type=int, help="Days of history to fetch")
def main(symbol: str, interval: str, days: int) -> None:
    """Backfill historical candles from Bybit REST API into PostgreSQL."""
    asyncio.run(_backfill(symbol, interval, days))


if __name__ == "__main__":
    main()
