"""Operator-facing priority legends for Telegram and diagnostics."""

from __future__ import annotations

import html
from typing import Any

_STRATEGY_LABELS: dict[str, str] = {
    "order_flow_v1": "Order flow (лента + стакан)",
    "liquidation_hunting_v1": "Liquidation hunting (охота за ликвидациями)",
    "funding_arbitrage_v1": "Funding arb (перекос фандинга)",
    "statistical_arbitrage_v1": "Stat arb (z-score mean-reversion)",
    "market_making_v1": "Market making (узкий спред)",
    "scalp_micro_v1": "Scalp micro (1m микро-скальп)",
    "ema_crossover_v1": "EMA trend (медленный тренд)",
    "candle_sampler_v1": "Candle sampler (только обучение)",
}


def _format_strategy_order(raw: str) -> list[str]:
    lines: list[str] = []
    for index, strategy_id in enumerate(raw.split(","), start=1):
        sid = strategy_id.strip()
        if not sid:
            continue
        label = _STRATEGY_LABELS.get(sid, sid)
        lines.append(f"{index}. <code>{html.escape(sid)}</code> — {html.escape(label)}")
    return lines


def safety_priority_text() -> str:
    """What overrides what in the safety stack."""
    return "\n".join(
        [
            "<b>🛡 Приоритет безопасности</b> (сверху вниз — что важнее)",
            "1. <b>Emergency stop / Kill switch</b> — мгновенно блокирует новые сделки.",
            "   Почему выше всего: оператор или риск-система должны иметь абсолютный veto.",
            "2. <b>Риск-менеджер</b> — дневной лимит, просадка, exposure, circuit breakers.",
            "   Почему выше стратегии: даже хороший сигнал нельзя исполнять при превышении лимитов.",
            "3. <b>Execution engine</b> — cooldown, max positions, min notional, API rejects.",
            "   Почему важно: защищает от спама заявок и технических отказов биржи.",
            "4. <b>Фильтры сигналов</b> — bucket/symbol/strategy gates, shadow loss guard, MTF, model gate.",
            "   Почему ниже риска: отсекают слабые сигналы, но не заменяют hard limits.",
            "5. <b>Стратегии ensemble</b> — генерируют идеи; сами по себе не отправляют ордера.",
        ]
    )


def signal_filter_priority_text() -> str:
    """Order of signal filters inside the strategy loop."""
    return "\n".join(
        [
            "<b>🚦 Приоритет фильтров сигнала</b> (проверяются по порядку)",
            "1. <b>Пауза / shadow loss guard</b> — глобальная пауза или серия плохих paper-сделок.",
            "2. <b>Bucket gate</b> — токсичный режим рынка (regime × volatility × час UTC).",
            "3. <b>Symbol-side gate</b> — отрицательный expectancy по паре+стороне.",
            "4. <b>Strategy gates</b> — strategy, strategy×side и strategy×regime режут доказанно токсичные контексты.",
            "5. <b>MTF trend confirm</b> — для LIVE: 5m/15m должны согласоваться с 1m трендом.",
            "6. <b>Model gate (CANARY)</b> — ML-фильтр только при доказанном lift; в SHADOW — наблюдение.",
            "7. <b>RiskManager.validate</b> — финальная проверка размера и лимитов перед ордером.",
            "",
            "В SHADOW bucket/symbol gates обычно <b>выключены</b>, чтобы копить данные.",
            "При <code>SCALP_STRICT_SHADOW=true</code> gates включены — paper ближе к LIVE.",
        ]
    )


def canary_readiness_priority_text() -> str:
    """What matters most when deciding CANARY_LIVE readiness."""
    return "\n".join(
        [
            "<b>🎯 Приоритет готовности CANARY</b> (что блокирует реальные деньги)",
            "<b>Критично (без этого CANARY нельзя):</b>",
            "• Инфраструктура: Postgres, Bybit WS, свежие свечи 1m/5m/15m/1h",
            "• Модель: quality=GOOD, walk-forward &gt; 0, CHAMPION, gate lift &gt; 0 на 50+ сигналах",
            "• Экономика: paper model-gate ≥20 сделок и PnL &gt; 0",
            "",
            "<b>Важно, но вторично:</b>",
            "• 2000+ размеченных примеров (1000 — минимум для первого кандидата)",
            "• 3+ активных монет, feature snapshots, prediction outcomes",
            "",
            "<b>Некритично для старта CANARY:</b>",
            "• Предупреждения по baseline paper, единичные API rejects",
            "• Model gate canary уже включен (лучше сначала наблюдать в SHADOW)",
            "",
            "Telegram <b>не включает</b> LIVE — только env vars на Render.",
        ]
    )


def strategy_priority_text(*, risk_profile: str, order_raw: str) -> str:
    """Explain ensemble conflict resolution for the active profile."""
    profile = (risk_profile or "MODERATE").upper()
    lines = _format_strategy_order(order_raw)
    header = (
        f"<b>📊 Приоритет стратегий ({html.escape(profile)})</b>\n"
        "При конфликте направлений побеждает стратегия <b>выше в списке</b>.\n"
        "Равный приоритет → сигнал пропускается (ensemble.conflict_blocked).\n"
    )
    if not lines:
        return header + "Порядок не задан в конфиге."
    return header + "\n".join(lines)


def full_priority_overview(*, runtime_settings: dict[str, Any] | None = None, settings: Any | None = None) -> str:
    """Combined priority legend for /priorities and Telegram buttons."""
    runtime = runtime_settings or {}
    risk_profile = str(runtime.get("risk_profile") or getattr(settings, "RISK_PROFILE", "MODERATE"))
    if hasattr(risk_profile, "value"):
        risk_profile = str(risk_profile.value)

    is_scalp = str(risk_profile).upper() == "SCALP"
    if runtime:
        order_raw = str(runtime.get("scalp_strategy_priority_order" if is_scalp else "strategy_priority_order") or "")
    elif settings is not None:
        order_raw = (
            str(getattr(settings, "SCALP_STRATEGY_PRIORITY_ORDER", ""))
            if is_scalp
            else str(getattr(settings, "STRATEGY_PRIORITY_ORDER", ""))
        )
    else:
        order_raw = ""

    sections = [
        safety_priority_text(),
        "",
        signal_filter_priority_text(),
        "",
        canary_readiness_priority_text(),
    ]
    if order_raw.strip():
        sections.extend(["", strategy_priority_text(risk_profile=risk_profile, order_raw=order_raw)])
    return "\n".join(sections)
