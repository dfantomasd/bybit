"""SHADOW runtime strategy for rules discovered by strategy_lab.

The strategy is intentionally simple and explainable: it reads a JSON report
created by ``python -m trader.strategy_lab.discover`` and emits a paper-only
proposal when a side-aware rule matches current features. No rules means no
signals; live safety is enforced by only wiring this strategy in SHADOW mode.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import structlog

from trader.domain.enums import MarketType, OrderSide
from trader.domain.models import FeatureVector, TradeProposal
from trader.strategies.base import BaseStrategy

log = structlog.get_logger(__name__)

DISCOVERED_RULE_STRATEGY_ID = "discovered_rule_v1"
_PRICE_DECIMALS = Decimal("0.00000001")
_PLACEHOLDER_PATH_PREFIXES = ("/path/to/", "path/to/")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DiscoveredCondition:
    feature: str
    op: str
    threshold: float


@dataclass(frozen=True)
class DiscoveredRule:
    rule_id: str
    conditions: tuple[DiscoveredCondition, ...]
    side: str
    symbol: str | None
    validation_count: int
    validation_avg_net_bps: float
    validation_lift_bps: float | None = None
    score: float = 0.0


def _price(value: float) -> Decimal:
    return Decimal(str(value)).quantize(_PRICE_DECIMALS)


def _normalise_side(side: str | None) -> str | None:
    raw = str(side or "").strip().lower()
    if raw in {"buy", "long"}:
        return "Buy"
    if raw in {"sell", "short"}:
        return "Sell"
    return None


def _parse_rule(
    raw: dict[str, Any], *, min_validation_count: int, min_validation_net_bps: float
) -> DiscoveredRule | None:
    side = _normalise_side(raw.get("side"))
    if side is None:
        return None
    validation_count = int(raw.get("validation_count") or 0)
    validation_avg = raw.get("validation_avg_net_bps")
    if validation_avg is None:
        return None
    validation_avg_f = float(validation_avg)
    if validation_count < min_validation_count or validation_avg_f < min_validation_net_bps:
        return None
    conditions: list[DiscoveredCondition] = []
    for item in list(raw.get("conditions") or []):
        feature = str(item.get("feature") or "")
        op = str(item.get("op") or "")
        if not feature or op not in {">=", "<="}:
            return None
        conditions.append(DiscoveredCondition(feature=feature, op=op, threshold=float(item.get("threshold"))))
    if not conditions:
        return None
    symbol = raw.get("symbol")
    return DiscoveredRule(
        rule_id=str(raw.get("rule_id") or "discovered_rule"),
        conditions=tuple(conditions),
        side=side,
        symbol=str(symbol).upper() if symbol else None,
        validation_count=validation_count,
        validation_avg_net_bps=validation_avg_f,
        validation_lift_bps=(float(raw["validation_lift_bps"]) if raw.get("validation_lift_bps") is not None else None),
        score=float(raw["score"]) if raw.get("score") is not None else validation_avg_f,
    )


def load_discovered_rules(
    path: str | Path,
    *,
    min_validation_count: int = 10,
    min_validation_net_bps: float = 0.0,
    max_rules: int = 20,
) -> list[DiscoveredRule]:
    """Load side-aware rules from a strategy_lab JSON report."""

    rule_path = Path(path).expanduser()
    if not rule_path.exists():
        return []
    try:
        payload = json.loads(rule_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("discovered_rule.load_failed", path=str(rule_path), error=str(exc))
        return []
    rules: list[DiscoveredRule] = []
    for raw in list(payload.get("rules") or []):
        if not isinstance(raw, dict):
            continue
        rule = _parse_rule(
            raw,
            min_validation_count=min_validation_count,
            min_validation_net_bps=min_validation_net_bps,
        )
        if rule is not None:
            rules.append(rule)
    rules.sort(key=lambda item: (item.validation_avg_net_bps, item.score, item.validation_count), reverse=True)
    return rules[: max(0, int(max_rules))]


def writable_discovered_rules_path(path: str | Path) -> Path:
    """Return a readable path for a strategy-lab report.

    Render envs sometimes keep documentation placeholders such as
    ``/path/to/strategy_lab.json``. Treat those as not operator intent and use
    the app working directory first, then fall back to the checked-in repo file.
    """

    raw = str(path or "").strip()
    if not raw or raw.startswith(_PLACEHOLDER_PATH_PREFIXES):
        raw = "strategy_lab.json"
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    cwd_candidate = candidate
    if cwd_candidate.exists():
        return cwd_candidate
    repo_candidate = _repo_root() / candidate
    if repo_candidate.exists():
        return repo_candidate
    return cwd_candidate


def runtime_discovered_rules_path(path: str | Path) -> Path:
    """Return the writable runtime path for generated strategy-lab reports."""

    raw = str(path or "").strip()
    if not raw or raw.startswith(_PLACEHOLDER_PATH_PREFIXES):
        raw = "strategy_lab.json"
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    return candidate


def write_discovered_rules_failure_report(path: str | Path, *, error: Exception, stage: str) -> Path:
    """Persist a small diagnostic report so Telegram can explain generator failures."""

    target = runtime_discovered_rules_path(path)
    payload = {
        "status": "auto_generate_failed",
        "rules": [],
        "sample_count": 0,
        "error": f"{type(error).__name__}: {error}",
        "stage": stage,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


async def auto_generate_discovered_rules_file(
    path: str | Path,
    *,
    horizon: int,
    min_samples: int,
    min_train_count: int,
    min_validation_count: int,
    min_validation_net_bps: float,
    top_n: int,
    timeout_seconds: float,
) -> tuple[Path, dict[str, Any]]:
    """Generate a strategy-lab JSON report from DB outcomes with a hard timeout."""

    target = runtime_discovered_rules_path(path)

    async def _build() -> dict[str, Any]:
        from trader.strategy_lab.discover import build_strategy_lab_report_from_db

        return await build_strategy_lab_report_from_db(
            horizon=horizon,
            min_samples=min_samples,
            min_train_count=min_train_count,
            min_validation_count=min_validation_count,
            min_validation_net_bps=min_validation_net_bps,
            top_n=top_n,
            segmented=True,
        )

    report = await asyncio.wait_for(_build(), timeout=max(1.0, float(timeout_seconds)))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target, report


class DiscoveredRuleStrategy(BaseStrategy):
    """Emit SHADOW proposals from offline-discovered positive-expectancy rules."""

    def __init__(
        self,
        *,
        rules: list[DiscoveredRule],
        max_notional_usd: float = 8.0,
        tp_pct: float = 0.75,
        sl_pct: float = 0.40,
        min_confidence: float = 0.52,
        diag_hook: Any | None = None,
    ) -> None:
        self._rules = list(rules)
        self._max_notional_usd = max(1.0, float(max_notional_usd))
        self._tp_pct = max(0.05, float(tp_pct))
        self._sl_pct = max(0.05, float(sl_pct))
        self._min_confidence = max(0.0, min(0.95, float(min_confidence)))
        self._diag_hook = diag_hook

    @property
    def strategy_id(self) -> str:
        return DISCOVERED_RULE_STRATEGY_ID

    def _diag(self, reason: str, **extra: Any) -> None:
        if self._diag_hook is None:
            return
        try:
            self._diag_hook(reason if not extra.get("symbol") else f"{reason}:{extra['symbol']}")
        except Exception as exc:
            log.debug("discovered_rule.diag_hook_failed", reason=reason, error=str(exc))

    @staticmethod
    def _matches(rule: DiscoveredRule, features: dict[str, float]) -> bool:
        for condition in rule.conditions:
            value = features.get(condition.feature)
            if value is None:
                return False
            if condition.op == ">=" and value < condition.threshold:
                return False
            if condition.op == "<=" and value > condition.threshold:
                return False
        return True

    def evaluate(
        self,
        feature_vector: FeatureVector,
        current_price: float,
        available_balance_usd: float,
    ) -> TradeProposal | None:
        if current_price <= 0 or available_balance_usd <= 0:
            return None
        if not self._rules:
            self._diag("discovered_rule_no_rules", symbol=feature_vector.symbol)
            return None
        features = dict(zip(feature_vector.feature_names, feature_vector.values, strict=False))
        symbol = feature_vector.symbol.upper()
        matches = [
            rule
            for rule in self._rules
            if (rule.symbol is None or rule.symbol == symbol) and self._matches(rule, features)
        ]
        if not matches:
            self._diag("discovered_rule_no_match", symbol=symbol)
            return None
        best = max(matches, key=lambda item: (item.validation_avg_net_bps, item.score, item.validation_count))
        side = OrderSide.BUY if best.side == "Buy" else OrderSide.SELL
        entry = _price(current_price)
        tp_mult = Decimal("1") + Decimal(str(self._tp_pct / 100.0))
        sl_mult = Decimal("1") - Decimal(str(self._sl_pct / 100.0))
        if side == OrderSide.SELL:
            tp_mult = Decimal("1") - Decimal(str(self._tp_pct / 100.0))
            sl_mult = Decimal("1") + Decimal(str(self._sl_pct / 100.0))
        notional = min(self._max_notional_usd, max(1.0, available_balance_usd * 0.002))
        qty = Decimal(str(notional)) / entry
        confidence = min(
            0.95,
            max(self._min_confidence, 0.50 + min(best.validation_avg_net_bps, 100.0) / 250.0),
        )
        self._diag("discovered_rule_match", symbol=symbol)
        return TradeProposal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            market_type=MarketType.LINEAR,
            side=side,
            requested_qty=qty,
            requested_notional_usd=Decimal(str(notional)),
            entry_price=entry,
            take_profit=(entry * tp_mult).quantize(_PRICE_DECIMALS),
            stop_loss=(entry * sl_mult).quantize(_PRICE_DECIMALS),
            confidence=confidence,
            expected_return=best.validation_avg_net_bps / 10000.0,
            expected_risk=self._sl_pct,
            feature_id=feature_vector.feature_id,
            rationale=(
                f"discovered rule {best.rule_id}: validation "
                f"{best.validation_count} avg {best.validation_avg_net_bps:+.2f} bps"
            ),
        )
