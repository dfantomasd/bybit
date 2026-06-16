# Промт для другого ИИ: выбор торговых пар через Telegram

Ты работаешь с репозиторием `dfantomasd/bybit`, Bybit AI Trading Bot. Нужно реализовать и улучшить Telegram-интерфейс, где русскоязычный трейдер может выбрать торговые пары галочками прямо в боте. Пользователь не программист, поэтому все тексты должны быть простыми и на русском.

## Цель

Добавить в Telegram-бот экран `✅ Выбрать пары`, где бот показывает до 100 монет/пар, подходящих для торговли с текущим балансом и фильтрами сканера. Пользователь нажимает на пары, видит галочки, а выбранные пары дальше используются:

- для загрузки истории свечей;
- для feature pipeline;
- для обучения модели;
- для торгового отбора strategy/execution, если пара продолжает проходить фильтры ликвидности, спреда, цены и риска.

## Контекст архитектуры

Изучи файлы:

- `src/trader/telegram_bot.py`
- `src/trader/app.py`
- `src/trader/features/screener.py`
- `src/trader/training/backfill.py`
- `src/trader/storage/trade_journal.py`
- `tests/unit/test_pr4_telegram_dashboard.py`
- `tests/unit/test_screener.py`

Сейчас `MarketScreener` строит:

- `wide_universe`: широкий список пар, прошедших фильтры;
- `feature_universe` / `active_symbols`: пары для свечей и признаков;
- `execution_candidates`: пары для стратегии.

Telegram работает через `TradingController` в `telegram_bot.py`, который получает callbacks из `TradingApplication`.

## Требования к UX

1. В главном меню должна быть кнопка `✅ Выбрать пары`.
2. Экран должен показывать до 100 подходящих пар.
3. Каждая пара отображается кнопкой:
   - `✅ BTCUSDT`, если выбрана;
   - `☐ BTCUSDT`, если не выбрана.
4. Нажатие на пару переключает выбор.
5. Нужна пагинация по 10 пар на страницу: `◀️`, `1/10`, `▶️`.
6. Должна быть кнопка `🔄 Обновить`.
7. Текст должен объяснять:
   - пары уже прошли фильтры ликвидности/спреда/цены;
   - выбранные пары будут использоваться для обучения и торговли;
   - если рынок ухудшится, сканер может временно не торговать пару.
8. Не добавляй кнопку, которая включает live-торговлю. Выбор пар не должен обходить RiskManager.

## Требования к логике

1. Кандидаты берутся из `screener.wide_universe`, максимум 100.
2. Если `wide_universe` еще пустой, fallback: `screener.active_symbols`.
3. Выбранные пары хранятся в `MarketScreener.manual_symbols`.
4. В `MarketScreener._screen()` ручные пары должны попадать в начало `feature_universe` и `execution_candidates`, но только если они есть в `wide_universe`, то есть прошли фильтры.
5. При добавлении новой пары нужно вызвать существующую логику:
   - seed candles;
   - subscribe WebSocket.
6. При снятии галочки пара убирается из manual list, но не надо насильно закрывать позиции или отписывать WS, если есть открытая позиция.
7. Runtime settings должны показывать `manual_symbols`.
8. Все изменения должны быть безопасны для `SHADOW`, `CANARY_LIVE` и `LIVE`: итоговое исполнение все равно идет через RiskManager и ExecutionEngine.

## Тесты

Добавь или обнови тесты:

- `tests/unit/test_screener.py`: ручная eligible-пара приоритетно попадает в `feature_universe` и `execution_candidates`.
- `tests/unit/test_pr4_telegram_dashboard.py`: главное меню содержит кнопку выбора пар.
- `tests/unit/test_pr4_telegram_dashboard.py`: меню выбора пар показывает `✅` и `☐`.
- `tests/unit/test_pr4_telegram_dashboard.py`: нажатие toggle вызывает `TradingController.toggle_symbol`.
- Если добавляешь persistence в Postgres, добавь тесты `TradeJournal`.

Команды проверки:

```bash
uv run --extra dev ruff format src/trader/telegram_bot.py src/trader/app.py src/trader/features/screener.py tests/unit/test_pr4_telegram_dashboard.py tests/unit/test_screener.py
uv run --extra dev ruff check src/trader/telegram_bot.py src/trader/app.py src/trader/features/screener.py tests/unit/test_pr4_telegram_dashboard.py tests/unit/test_screener.py
uv run --extra dev pytest tests/unit/test_pr4_telegram_dashboard.py tests/unit/test_screener.py --no-cov -q
uv run --extra dev pytest tests/unit --no-cov -q
```

## Улучшение второго этапа

После первой реализации добавь persistence:

- таблица `operator_symbol_selection`;
- сохранение выбранных пар по chat_id/operator_id;
- загрузка selections при старте;
- audit log: кто добавил/убрал пару и когда.

Важно: persistence не должна ломать запуск, если Postgres временно недоступен. В live-режиме можно показывать предупреждение, но не падать без необходимости.
