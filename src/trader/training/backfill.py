"""Offline backfill CLI — fetches historical candles from Bybit REST and writes to PostgreSQL.

Usage:
    python -m trader.training.backfill --symbols BTCUSDT,ETHUSDT --intervals 1,5,15,60 --days 7

NEVER runs inside the trading process.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import click


async def _backfill_one(pool: Any, session: Any, *, symbol: str, interval: str, days: int, category: str) -> int:
    import aiohttp

    base = "https://api.bybit.com"
    end_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
    start_ms = int((datetime.now(tz=UTC) - timedelta(days=days)).timestamp() * 1000)
    interval_ms_map = {"1": 60_000, "3": 180_000, "5": 300_000, "15": 900_000, "30": 1_800_000, "60": 3_600_000}
    bar_ms = interval_ms_map.get(interval, 60_000)
    limit = 1000
    inserted = 0

    cursor = end_ms
    while cursor > start_ms:
        url = f"{base}/v5/market/kline"
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "end": cursor,
            "limit": limit,
        }
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            data = await resp.json()

        if data.get("retCode") != 0:
            click.echo(f"API error for {symbol}/{interval}: {data.get('retMsg')}", err=True)
            break

        candles = data.get("result", {}).get("list", [])
        if not candles:
            break

        async with pool.acquire() as conn:
            for c in candles:
                open_time_ms = int(c[0])
                if open_time_ms < start_ms:
                    continue
                open_time = datetime.fromtimestamp(open_time_ms / 1000, tz=UTC)
                close_time = datetime.fromtimestamp((open_time_ms + bar_ms - 1) / 1000, tz=UTC)
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
    return inserted


async def _backfill(symbols: list[str], intervals: list[str], days: int, category: str) -> None:
    import asyncpg

    from trader.config import Settings
    from trader.storage.trade_journal import asyncpg_pool_connect_kwargs

    settings = Settings()

    pool = await asyncpg.create_pool(
        **asyncpg_pool_connect_kwargs(settings.POSTGRES_DSN.get_secret_value()),
        min_size=1,
        max_size=2,
        statement_cache_size=0,
    )

    try:
        import aiohttp

        total = 0
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                for interval in intervals:
                    total += await _backfill_one(
                        pool,
                        session,
                        symbol=symbol,
                        interval=interval,
                        days=days,
                        category=category,
                    )
        click.echo(f"Backfill all complete: {total} candles inserted")
    finally:
        await pool.close()


@click.command()
@click.option("--symbols", "--symbol", default="BTCUSDT", help="Comma-separated trading symbols")
@click.option("--intervals", "--interval", default="1", help="Comma-separated intervals (1, 3, 5, 15, 30, 60)")
@click.option("--days", default=7, type=int, help="Days of history to fetch")
@click.option("--category", default="linear", help="Bybit market category")
def main(symbols: str, intervals: str, days: int, category: str) -> None:
    """Backfill historical candles from Bybit REST API into PostgreSQL."""
    symbol_list = [item.strip().upper() for item in symbols.split(",") if item.strip()]
    interval_list = [item.strip() for item in intervals.split(",") if item.strip()]
    asyncio.run(_backfill(symbol_list, interval_list, days, category))


if __name__ == "__main__":
    main()
