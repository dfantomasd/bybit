"""Async LLM client for optional risk-multiplier scoring.

Calls a local Ollama-compatible endpoint (or OpenAI-compatible API) to get a
risk sentiment score [0.0, 1.0] for a proposed trade.  The score is used as a
*reducing* multiplier on the approved position size — it can only shrink, never
grow, the risk manager's base sizing.

The client is intentionally minimal: one structured prompt, one number back.
If the LLM is unavailable or exceeds the daily budget the multiplier defaults
to 1.0 (no reduction).
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import aiohttp
import structlog

log = structlog.get_logger(__name__)

_DEFAULT_TIMEOUT_S = 5.0
_RETRY_ATTEMPTS = 2
_PROMPT_TEMPLATE = """\
You are a risk analyst for a cryptocurrency derivatives trading system.
Given the trade proposal below, return a JSON object with a single key
"risk_multiplier" whose value is a float between 0.0 and 1.0.
1.0 means full confidence in the trade; 0.0 means skip it entirely.
Base your answer on the regime, side, and rationale only.
Respond with ONLY the JSON object — no markdown, no explanation.

Trade:
  symbol: {symbol}
  side: {side}
  regime: {regime}
  confidence: {confidence:.2f}
  rationale: {rationale}
"""


class LLMClient:
    """Lightweight async LLM risk-multiplier client.

    Parameters
    ----------
    base_url:
        Base URL of the Ollama-compatible API, e.g. ``http://ollama:11434``.
    model:
        Model name, e.g. ``llama3``.
    budget_cap_usd:
        Maximum USD spend per day (tracked via token-count heuristic for
        external providers; Ollama is free so this only logs a warning).
    timeout_s:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        budget_cap_usd: float = 5.0,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._budget_cap_usd = budget_cap_usd
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()

        # Daily spend tracking (heuristic: $0.001 per call for cloud providers)
        self._spend_date: str = ""
        self._daily_spend_usd: float = 0.0
        self._cost_per_call_usd: float = 0.001

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def _check_budget(self) -> bool:
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        if today != self._spend_date:
            self._spend_date = today
            self._daily_spend_usd = 0.0
        if self._daily_spend_usd >= self._budget_cap_usd:
            log.warning(
                "llm.budget_cap_reached",
                daily_spend=self._daily_spend_usd,
                cap=self._budget_cap_usd,
            )
            return False
        return True

    def _record_call(self) -> None:
        self._daily_spend_usd += self._cost_per_call_usd

    async def get_risk_multiplier(
        self,
        symbol: str,
        side: str,
        regime: str,
        confidence: float,
        rationale: str,
    ) -> float:
        """Return a risk multiplier in [0.0, 1.0].

        Returns 1.0 (no reduction) on any error so trading is never blocked
        by LLM unavailability.
        """
        # Budget check + pre-record under lock so concurrent per-symbol calls
        # cannot all slip past a near-full budget simultaneously.
        async with self._lock:
            if not self._check_budget():
                return 1.0
            self._record_call()

        prompt = _PROMPT_TEMPLATE.format(
            symbol=symbol,
            side=side,
            regime=regime,
            confidence=confidence,
            rationale=rationale or "none",
        )

        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 32},
        }

        for attempt in range(_RETRY_ATTEMPTS):
            try:
                session = await self._get_session()
                async with session.post(
                    f"{self._base_url}/api/generate",
                    json=payload,
                ) as resp:
                    if resp.status != 200:
                        log.debug(
                            "llm.bad_status",
                            status=resp.status,
                            attempt=attempt,
                        )
                        if attempt < _RETRY_ATTEMPTS - 1:
                            await asyncio.sleep(0.5)
                        continue
                    data = await resp.json(content_type=None)
                    # Ollama returns {"error": "..."} with HTTP 200 on model/config errors.
                    if "error" in data:
                        log.warning(
                            "llm.model_error",
                            error=data["error"],
                            model=self._model,
                        )
                        return 1.0
                    raw = data.get("response") or ""
                    if not raw:
                        log.debug("llm.empty_response", attempt=attempt)
                        if attempt < _RETRY_ATTEMPTS - 1:
                            await asyncio.sleep(0.5)
                        continue
                    parsed = json.loads(raw)
                    if not isinstance(parsed, dict):
                        log.debug(
                            "llm.unexpected_json_type",
                            json_type=type(parsed).__name__,
                            attempt=attempt,
                        )
                        if attempt < _RETRY_ATTEMPTS - 1:
                            await asyncio.sleep(0.5)
                        continue
                    mult = float(parsed.get("risk_multiplier", 1.0))
                    mult = max(0.0, min(1.0, mult))
                    log.debug(
                        "llm.risk_multiplier",
                        symbol=symbol,
                        side=side,
                        multiplier=mult,
                    )
                    return mult
            except Exception as exc:
                log.debug("llm.call_failed", attempt=attempt, error=str(exc))
                if attempt < _RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(0.5)

        return 1.0
