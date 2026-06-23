"""Telegram webhook delivery mode tests."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from trader.telegram_bot import TelegramBotConfig, TelegramMonitorBot


def _make_webhook_bot() -> TelegramMonitorBot:
    return TelegramMonitorBot(
        config=TelegramBotConfig(
            token="fake:TOKEN",
            allowed_chat_ids={12345},
            trading_mode="SHADOW",
            risk_profile="CONSERVATIVE",
            bybit_use_testnet=True,
            delivery_mode="webhook",
            webhook_url="https://example.onrender.com/telegram/webhook",
            webhook_secret="secret-token",
        ),
        health_provider=AsyncMock(),
        adapter_factory=lambda: None,
    )


def _mock_application() -> MagicMock:
    mock_app = MagicMock()
    mock_app.initialize = AsyncMock()
    mock_app.start = AsyncMock()
    mock_app.stop = AsyncMock()
    mock_app.shutdown = AsyncMock()
    mock_app.running = True
    mock_app.updater = None
    mock_app.bot = MagicMock(
        set_webhook=AsyncMock(),
        delete_webhook=AsyncMock(),
        get_webhook_info=AsyncMock(return_value=MagicMock(url="https://example.onrender.com/telegram/webhook")),
    )
    mock_app.process_update = AsyncMock()
    return mock_app


@pytest.mark.asyncio
async def test_telegram_webhook_start_registers_route_and_webhook() -> None:
    bot = _make_webhook_bot()
    http_app = FastAPI()
    mock_app = _mock_application()

    with patch("trader.telegram_bot.Application") as mock_app_cls:
        mock_app_cls.builder.return_value.token.return_value.build.return_value = mock_app
        started = await bot.start(http_app=http_app)

    assert started is True
    assert bot._webhook_route_mounted is True
    paths = {getattr(route, "path", None) for route in http_app.routes}
    assert "/telegram/webhook" in paths
    assert "/telegram/livez" in paths
    mock_app.bot.set_webhook.assert_awaited_once()
    health = bot.health_snapshot()
    assert health["delivery_mode"] == "webhook"
    assert health["webhook_active"] is True


@pytest.mark.asyncio
async def test_telegram_webhook_requires_http_app() -> None:
    bot = _make_webhook_bot()
    mock_app = _mock_application()

    with patch("trader.telegram_bot.Application") as mock_app_cls:
        mock_app_cls.builder.return_value.token.return_value.build.return_value = mock_app
        started = await bot.start(http_app=None)

    assert started is False
    assert bot.health_snapshot()["polling_disabled_reason"] == "webhook_http_app_missing"


@pytest.mark.asyncio
async def test_telegram_webhook_processes_update() -> None:
    bot = _make_webhook_bot()
    http_app = FastAPI()
    mock_app = _mock_application()

    with patch("trader.telegram_bot.Application") as mock_app_cls:
        mock_app_cls.builder.return_value.token.return_value.build.return_value = mock_app
        with patch("trader.telegram_bot.Update.de_json", return_value=MagicMock(update_id=1)):
            assert await bot.start(http_app=http_app) is True

    transport = ASGITransport(app=http_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/telegram/webhook",
            json={"update_id": 1},
            headers={"X-Telegram-Bot-Api-Secret-Token": "secret-token"},
        )

    assert response.status_code == 200
    await asyncio.sleep(0)
    mock_app.process_update.assert_awaited_once()


@pytest.mark.asyncio
async def test_telegram_webhook_start_starts_delivery_watchdog() -> None:
    bot = _make_webhook_bot()
    http_app = FastAPI()
    mock_app = _mock_application()

    with patch("trader.telegram_bot.Application") as mock_app_cls:
        mock_app_cls.builder.return_value.token.return_value.build.return_value = mock_app
        with patch.object(bot, "_start_polling_watchdog") as mock_watchdog:
            assert await bot.start(http_app=http_app) is True
            mock_watchdog.assert_called_once()


@pytest.mark.asyncio
async def test_telegram_webhook_shutdown_preserves_registration() -> None:
    bot = _make_webhook_bot()
    http_app = FastAPI()
    mock_app = _mock_application()

    with patch("trader.telegram_bot.Application") as mock_app_cls:
        mock_app_cls.builder.return_value.token.return_value.build.return_value = mock_app
        assert await bot.start(http_app=http_app) is True

    await bot.stop()
    mock_app.bot.delete_webhook.assert_not_awaited()
