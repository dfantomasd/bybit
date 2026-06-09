"""Alembic environment configuration for async PostgreSQL migrations."""
from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Alembic Config object for access to .ini values
config = context.config

# Set up logging from the alembic.ini [loggers] section
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import all SQLAlchemy models here so Alembic can detect changes
# When models are created in Phase 2, add imports like:
#   from trader.db.models import Base  # noqa: F401
#   target_metadata = Base.metadata
target_metadata = None

# Read the database URL from environment if not set in alembic.ini
# This allows using docker secrets / env vars without hardcoding
def get_url() -> str:
    """Get database URL from environment or alembic config."""
    url = os.environ.get("POSTGRES_DSN") or config.get_main_option("sqlalchemy.url")
    if not url:
        raise ValueError(
            "Database URL not configured. "
            "Set POSTGRES_DSN environment variable or sqlalchemy.url in alembic.ini"
        )
    # asyncpg driver is required for async migrations
    if "asyncpg" not in url and "postgresql" in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://")
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    In offline mode, the database does not need to be reachable;
    migrations are rendered as SQL statements to stdout or a file.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations using an existing connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in async mode using asyncpg."""
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (requires a live DB connection)."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
