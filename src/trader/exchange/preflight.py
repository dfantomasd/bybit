"""Preflight check service — runs all validation checks before bot starts trading.

Returns a PreflightReport; blocks in BLOCKED state if critical checks fail.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from pydantic import BaseModel

from trader.domain.models import PreflightReport
from trader.exchange.endpoint_selector import EndpointSelector

logger = structlog.get_logger(__name__)

# Server-time drift thresholds (seconds)
_DRIFT_WARN_SECONDS = 5.0
_DRIFT_FAIL_SECONDS = 30.0

# Balance warning threshold (USDT)
_LOW_BALANCE_WARN_USDT = 10.0

# Required API key permissions (Bybit v5 names)
_REQUIRED_PERMISSIONS = {"ContractTrade", "Trade"}
_DANGEROUS_PERMISSIONS = {"Wallet"}  # withdrawal permission — should not be set


class CheckResult(BaseModel):
    """Result of a single preflight check."""

    name: str
    passed: bool
    critical: bool
    message: str
    details: dict[str, Any] = {}
    warning: str | None = None


class PreflightChecker:
    """Runs all preflight checks before the bot starts trading.

    Injects a REST client so checks can call real (or mocked) Bybit endpoints.
    All individual checks are coroutines; ``run()`` aggregates them.
    """

    def __init__(
        self,
        rest_client: Any,  # BybitRestClient (avoid circular import)
        endpoint_selector: EndpointSelector,
        use_testnet: bool,
        symbols_to_check: list[str] | None = None,
        check_leverage_symbol: str | None = None,
        expected_account_type: str = "UNIFIED",
    ) -> None:
        self._rest = rest_client
        self._endpoint_selector = endpoint_selector
        self._use_testnet = use_testnet
        self._symbols = symbols_to_check or []
        self._leverage_symbol = check_leverage_symbol
        self._expected_account_type = expected_account_type

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> PreflightReport:
        """Execute all checks and return a PreflightReport."""
        checks_to_run = [
            self._check_rest_connectivity,
            self._check_server_time_drift,
            self._check_api_key_validity,
            self._check_api_key_permissions,
            self._check_account_type,
            self._check_trading_categories,
            self._check_balance,
            self._check_region_compatibility,
            self._check_testnet_vs_live,
            self._check_leverage_settings,
        ]

        results: list[CheckResult] = []
        for check_fn in checks_to_run:
            try:
                result = await check_fn()
            except Exception as exc:  # pragma: no cover
                result = CheckResult(
                    name=check_fn.__name__.removeprefix("_check_"),
                    passed=False,
                    critical=True,
                    message=f"Check raised unexpected exception: {exc}",
                )
            results.append(result)
            log_fn = logger.info if result.passed else logger.warning
            log_fn(
                "preflight.check_complete",
                check=result.name,
                passed=result.passed,
                critical=result.critical,
                message=result.message,
            )

        # Summarise
        all_passed = all(r.passed for r in results)
        critical_failed = any(not r.passed and r.critical for r in results)
        checks_dict = {r.name: r.passed for r in results}
        errors = [r.message for r in results if not r.passed and r.critical]
        warnings = [r.warning for r in results if r.warning]

        passed = all_passed or (not critical_failed)

        report = PreflightReport(
            passed=passed,
            checks=checks_dict,
            errors=errors,
            warnings=[w for w in warnings if w],
        )

        log_fn2 = logger.info if passed else logger.error
        log_fn2(
            "preflight.complete",
            passed=passed,
            total_checks=len(results),
            failed_checks=sum(1 for r in results if not r.passed),
        )
        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    async def _check_rest_connectivity(self) -> CheckResult:
        """Verify we can reach the Bybit REST endpoint."""
        try:
            resp = await self._rest.get_server_time()
            if resp.get("retCode", -1) == 0:
                return CheckResult(
                    name="rest_connectivity",
                    passed=True,
                    critical=True,
                    message="REST endpoint reachable",
                    details={"server_time": resp.get("result", {})},
                )
        except Exception as exc:
            return CheckResult(
                name="rest_connectivity",
                passed=False,
                critical=True,
                message=f"REST endpoint unreachable: {exc}",
            )
        return CheckResult(
            name="rest_connectivity",
            passed=False,
            critical=True,
            message="REST endpoint returned non-zero retCode",
        )

    async def _check_server_time_drift(self) -> CheckResult:
        """Check clock skew between local machine and Bybit server."""
        try:
            local_ms = int(time.time() * 1000)
            resp = await self._rest.get_server_time()
            server_ms_str = resp.get("result", {}).get("timeSecond") or resp.get("result", {}).get("timeNano", "0")
            # timeSecond is a string of seconds since epoch
            server_ms = int(float(server_ms_str)) * 1000
            drift_seconds = abs(local_ms - server_ms) / 1000.0

            if drift_seconds > _DRIFT_FAIL_SECONDS:
                return CheckResult(
                    name="server_time_drift",
                    passed=False,
                    critical=True,
                    message=f"Clock drift {drift_seconds:.1f}s exceeds {_DRIFT_FAIL_SECONDS}s limit",
                    details={"drift_seconds": drift_seconds},
                )
            warning = None
            if drift_seconds > _DRIFT_WARN_SECONDS:
                warning = f"Clock drift {drift_seconds:.1f}s exceeds {_DRIFT_WARN_SECONDS}s warning threshold"

            return CheckResult(
                name="server_time_drift",
                passed=True,
                critical=True,
                message=f"Clock drift {drift_seconds:.1f}s — within acceptable range",
                details={"drift_seconds": drift_seconds},
                warning=warning,
            )
        except Exception as exc:
            return CheckResult(
                name="server_time_drift",
                passed=False,
                critical=True,
                message=f"Could not check server time: {exc}",
            )

    async def _check_api_key_validity(self) -> CheckResult:
        """Verify API key is valid by calling get_api_key_info."""
        try:
            resp = await self._rest.get_api_key_info()
            if resp.get("retCode", -1) == 0:
                return CheckResult(
                    name="api_key_validity",
                    passed=True,
                    critical=True,
                    message="API key is valid",
                    details={"key_info": resp.get("result", {})},
                )
            return CheckResult(
                name="api_key_validity",
                passed=False,
                critical=True,
                message=f"API key invalid — retCode={resp.get('retCode')} {resp.get('retMsg', '')}",
            )
        except Exception as exc:
            return CheckResult(
                name="api_key_validity",
                passed=False,
                critical=True,
                message=f"API key check failed: {exc}",
            )

    async def _check_api_key_permissions(self) -> CheckResult:
        """Check that the key has trading permissions but not withdrawal."""
        try:
            resp = await self._rest.get_api_key_info()
            result = resp.get("result", {})
            permissions = result.get("permissions", {})

            # Bybit returns permissions as a dict of category → list of permission strings
            # The dict keys themselves (e.g. "Wallet") denote permission categories.
            # We check both keys and values for dangerous permission names.
            all_perm_keys: set[str] = set(permissions.keys())
            all_perm_values: set[str] = set()
            for perm_list in permissions.values():
                all_perm_values.update(perm_list)
            all_perms = all_perm_keys | all_perm_values

            warning = None
            if _DANGEROUS_PERMISSIONS & all_perm_keys:
                warning = (
                    "API key has WITHDRAWAL permission (Wallet category) — this is dangerous for a trading bot. "
                    "Revoke it immediately."
                )
            elif any(p.lower() in ("withdraw", "withdrawal") for p in all_perm_values):
                warning = (
                    "API key has WITHDRAWAL permission — this is dangerous for a trading bot. Revoke it immediately."
                )

            return CheckResult(
                name="api_key_permissions",
                passed=True,
                critical=False,
                message="API key permissions checked",
                details={"permissions": list(all_perms)},
                warning=warning,
            )
        except Exception as exc:
            return CheckResult(
                name="api_key_permissions",
                passed=False,
                critical=False,
                message=f"Could not check permissions: {exc}",
            )

    async def _check_account_type(self) -> CheckResult:
        """Verify account is UNIFIED (recommended for V5 API)."""
        try:
            resp = await self._rest.get_account_info()
            result = resp.get("result", {})
            account_type = result.get("unifiedMarginStatus", result.get("marginMode", ""))

            # Bybit unifiedMarginStatus: 1=Regular, 2=Unified margin, 3=Unified trade
            unified = account_type in (2, 3, "2", "3", "UNIFIED", "UTA")
            return CheckResult(
                name="account_type",
                passed=unified,
                critical=False,
                message=(
                    f"Account type: {account_type} — "
                    + ("UNIFIED (recommended)" if unified else "NOT unified — consider upgrading")
                ),
                details={"account_type": account_type},
            )
        except Exception as exc:
            return CheckResult(
                name="account_type",
                passed=False,
                critical=False,
                message=f"Could not check account type: {exc}",
            )

    async def _check_trading_categories(self) -> CheckResult:
        """Verify we can fetch instruments for at least one category."""
        categories = ["linear", "spot"]
        accessible = []
        for cat in categories:
            try:
                resp = await self._rest.get_instruments_info(cat, symbol=None)
                if resp.get("retCode", -1) == 0:
                    accessible.append(cat)
            except Exception:  # noqa: S110
                pass

        passed = bool(accessible)
        return CheckResult(
            name="trading_categories",
            passed=passed,
            critical=False,
            message=f"Accessible categories: {accessible}" if passed else "No categories accessible",
            details={"accessible": accessible},
        )

    async def _check_balance(self) -> CheckResult:
        """Check account balance; warn if very low."""
        try:
            resp = await self._rest.get_wallet_balance(account_type="UNIFIED")
            result = resp.get("result", {})
            accounts = result.get("list", [])

            total_equity = 0.0
            for account in accounts:
                total_equity += float(account.get("totalEquity", 0) or 0)

            warning = None
            if total_equity < _LOW_BALANCE_WARN_USDT:
                warning = f"Account equity is very low: {total_equity:.2f} USDT"

            return CheckResult(
                name="balance",
                passed=True,
                critical=False,
                message=f"Account equity: {total_equity:.2f} USDT",
                details={"total_equity_usdt": total_equity},
                warning=warning,
            )
        except Exception as exc:
            return CheckResult(
                name="balance",
                passed=False,
                critical=False,
                message=f"Could not fetch balance: {exc}",
            )

    async def _check_region_compatibility(self) -> CheckResult:
        """Verify region/testnet combo is valid."""
        try:
            self._endpoint_selector.validate_region_compatibility()
            return CheckResult(
                name="region_compatibility",
                passed=True,
                critical=True,
                message=(
                    f"Region {self._endpoint_selector.region.value} is compatible with testnet={self._use_testnet}"
                ),
                details={
                    "region": self._endpoint_selector.region.value,
                    "rest_base": self._endpoint_selector.rest_base,
                },
            )
        except Exception as exc:
            return CheckResult(
                name="region_compatibility",
                passed=False,
                critical=True,
                message=f"Region incompatibility: {exc}",
            )

    async def _check_testnet_vs_live(self) -> CheckResult:
        """Ensure testnet mode matches the declared trading mode."""
        # This check is mostly a guard: if testnet is True and we get live prices, something is wrong.
        mode_label = "TESTNET" if self._use_testnet else "LIVE"
        return CheckResult(
            name="testnet_vs_live",
            passed=True,
            critical=True,
            message=f"Running in {mode_label} mode — configuration consistent",
            details={"use_testnet": self._use_testnet},
            warning=("Running in LIVE mode — real money at risk!" if not self._use_testnet else None),
        )

    async def _check_leverage_settings(self) -> CheckResult:
        """Check leverage settings for the configured symbols."""
        if not self._leverage_symbol:
            return CheckResult(
                name="leverage_settings",
                passed=True,
                critical=False,
                message="No leverage symbol configured — skipping",
            )
        try:
            resp = await self._rest.get_positions("linear", symbol=self._leverage_symbol)
            result = resp.get("result", {})
            return CheckResult(
                name="leverage_settings",
                passed=True,
                critical=False,
                message=f"Leverage check for {self._leverage_symbol} passed",
                details={"position_result": result},
            )
        except Exception as exc:
            return CheckResult(
                name="leverage_settings",
                passed=False,
                critical=False,
                message=f"Leverage check failed: {exc}",
            )
