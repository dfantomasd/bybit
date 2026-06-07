# Аудит готовности к CANARY_LIVE

Дата: 2026-06-08

## Раздел 1. Технический аудит

Исправлено:

- `src/trader/training/promote.py`: промоут в `CHAMPION` был фактически заблокирован, потому что `walk_forward_expectancy` не передавался в `can_promote()` и оставался `0.0`. Теперь CLI берет метрику из `model_versions.metrics` и промоутит только при положительном ожидании.
- `src/trader/app.py`: запуск обучения защищен `asyncio.Lock`, чтобы автообучение и ручная кнопка Telegram не стартовали два процесса одновременно.
- `src/trader/telegram_bot.py`: убраны тихие ошибки в fallback редактирования Telegram-сообщения и в `/net`; теперь сбои логируются.

Проверено:

- `create_task` в долгоживущих задачах добавляется в `_background_tasks` и отменяется в `_graceful_shutdown`.
- `TradeJournal` закрывает pool в `close()` и training/backfill закрывают pool в `finally`.
- `callback_data` основных меню покрыты обработчиками `_handle_view_button`, `_handle_control_button`, `_handle_train_button`, `_handle_limit_button`, `_handle_mode_button`.

Остаточный риск:

- В `TradeJournal` парсеры `_decimal_or_none`, `_parse_dec`, `_parse_ts`, `_dt_from_ms` остаются best-effort и возвращают `None`/текущее время на кривых данных биржи. Это не скрывает критические ошибки записи, но может маскировать отдельные сырые поля Bybit.

## Раздел 2. UX-аудит

Главная проблема была в смешении русского меню с английскими терминами на экранах: `Status`, `Balance`, `Feature snapshots`, `Prediction outcomes`, `Quality`, `Precision`, `Lift`, `Walk-forward`, `Paper baseline`, `Model gate`.

Оценка после правок:

- Главное меню: 4/5. Кнопки понятны, порядок рабочий: позиции, сканер, результаты, причины отсутствия сделок, база/модель, нагрузка, управление.
- База и модель: 4/5. Метрики переведены, добавлено пояснение `bps`, `Precision`, `Lift`.
- Готовность к CANARY: 5/5. Каждый красный/желтый пункт показывает, как исправить, примерное время и env vars Render.
- Модель/статус: 4/5. `CHAMPION`, `SHADOW_CHALLENGER`, `GOOD`, `WEAK` отображаются русскими статусами.
- Лимиты: 4/5. Добавлены пояснения, как лимит влияет на риск или нагрузку.
- Пауза/возобновить: 5/5. Текст явно говорит, что новые входы остановлены, а сбор/управление продолжаются.
- Результаты и `/net`: 4/5. Показаны валовый PnL, комиссии, фандинг, проскальзывание, чистый PnL.
- `/diagnostics`: 4/5. Технические счетчики переведены, но это все еще экран оператора.

## Раздел 5. Ускорение выхода на реальные деньги

Backfill есть и расширен:

```bash
python -m trader.training.backfill --symbols BTCUSDT,ETHUSDT,DOGEUSDT --intervals 1,5,15,60 --days 7
```

Рекомендованный порядок:

1. Загрузить историю по 3-5 самым ликвидным символам и всем интервалам `1,5,15,60`.
2. Запустить сервис в `SHADOW`, дождаться `feature_snapshots >= 1000` и `labelled_samples_15m >= 1000`.
3. Запустить обучение: `python -m trader.training.train --min-samples 1000 --horizon 15 --label-bps 5`.
4. Проверить `Lift`, `Precision`, `walk-forward`; при положительном ожидании выполнить `promote.py --confirm`.
5. Включать `CANARY_LIVE` только после зеленого Telegram-экрана готовности.

`MODEL_AUTO_TRAIN_INCREMENT_SAMPLES=1000` безопасен для продакшена. Для первого ускоренного кандидата можно вручную запускать `/train 500 15 5`, но промоут в `CHAMPION` должен оставаться через положительный walk-forward.

## Раздел 6. Конкурентное преимущество в скальпинге

Текущий проект уже сильнее типичного retail ML-скальпера, потому что есть:

- журнал сигналов, исходов и бумажного PnL;
- модель-кандидат отдельно от `CHAMPION`;
- shadow gate, который оценивает pass/block до влияния на реальные деньги;
- fee-aware execution и риск-менеджер перед отправкой ордера.

Что добавить за неделю для измеримого преимущества:

- Order book imbalance и bid/ask depth imbalance в feature pipeline: сложность средняя, вероятный lift высокий на 15m входах с плохой ликвидностью.
- Funding/open-interest momentum: сложность низкая/средняя, помогает не входить против перегретого perp-потока.
- Time-of-day/session features: сложность низкая, помогает фильтровать часы с плохим spread/slippage.

Maker-режим `POST_ONLY_LIMIT` уже предусмотрен через `ENTRY_ORDER_MODE`. Для горизонта 15m он выгоден, если fill-rate не режет лучшие сигналы; включать его стоит после сравнения maker/taker результата на paper.

Для A/B тестирования следующая архитектура: сохранять `strategy_id` и `model_version` для каждой ветки, вести параллельные paper outcomes, затем выбирать `CHAMPION` по net expectancy после fees/funding/slippage, а не по raw accuracy.
