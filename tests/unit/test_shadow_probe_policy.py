from __future__ import annotations

from types import SimpleNamespace

from trader.modules.signal_policy import SignalPolicyModule


def _policy_module(
    *,
    side_stats: dict[tuple[str, str], tuple[float, int]] | None = None,
    symbol_stats: dict[str, tuple[float, int]] | None = None,
    eligible: set[str] | None = None,
) -> SignalPolicyModule:
    settings = SimpleNamespace(
        SHADOW_PROBE_SIDE_BLOCK_ENABLED=True,
        SHADOW_PROBE_SIDE_MIN_SAMPLES=8,
        SHADOW_PROBE_SIDE_BLOCK_AVG_BPS=-3.0,
        SHADOW_PROBE_QUALITY_FILTER_ENABLED=True,
        SHADOW_PROBE_BASELINE_MIN_AVG_BPS=0.0,
        SHADOW_PROBE_BASELINE_MIN_SAMPLES=6,
        SHADOW_PROBE_SYMBOL_TOP_N=2,
        SHADOW_PROBE_SYMBOL_MIN_SAMPLES=5,
        SHADOW_PROBE_SYMBOL_MIN_AVG_BPS=-1.0,
        SHADOW_PROBE_SYMBOL_WARMUP_SECONDS=300,
        SHADOW_PROBE_ALLOWED_REGIMES="BULL_TREND,BEAR_TREND",
    )
    app = SimpleNamespace(
        _settings=settings,
        _shadow_probe_side_stats=side_stats or {},
        _shadow_probe_symbol_stats=symbol_stats or {},
        _shadow_probe_eligible_symbols=eligible,
        _shadow_probe_symbol_subscribed_at={},
    )
    return SignalPolicyModule(app)


def test_shadow_probe_side_blocked_for_losing_side() -> None:
    policy = _policy_module(side_stats={("XRPUSDT", "Sell"): (-5.0, 10)})

    assert policy.shadow_probe_side_blocked("XRPUSDT", "Sell") is True
    assert policy.shadow_probe_side_blocked("XRPUSDT", "Buy") is False


def test_shadow_probe_quality_requires_non_negative_baseline() -> None:
    policy = _policy_module(side_stats={("XRPUSDT", "Buy"): (-1.0, 8)})

    assert policy.shadow_probe_quality_allows("XRPUSDT", "Buy") is False
    assert policy.shadow_probe_quality_allows("XRPUSDT", "Sell") is True


def test_shadow_probe_symbol_allowed_uses_eligible_set() -> None:
    policy = _policy_module(eligible={"XRPUSDT", "SOLUSDT"})

    assert policy.shadow_probe_symbol_allowed("XRPUSDT") is True
    assert policy.shadow_probe_symbol_allowed("DOGEUSDT") is False


def test_shadow_probe_empty_eligible_set_is_warmup_not_block_all() -> None:
    policy = _policy_module(eligible=set())

    assert policy.shadow_probe_symbol_allowed("DOGEUSDT") is True


def test_compute_shadow_probe_eligible_symbols_returns_top_n() -> None:
    stats = {
        "XRPUSDT": (4.0, 10),
        "SOLUSDT": (2.0, 8),
        "DOGEUSDT": (-2.0, 12),
        "BTCUSDT": (1.0, 7),
    }

    eligible = SignalPolicyModule.compute_shadow_probe_eligible_symbols(
        stats,
        top_n=2,
        min_samples=5,
        min_avg_bps=-1.0,
    )

    assert eligible == {"XRPUSDT", "SOLUSDT"}


def test_shadow_probe_symbol_warmup_blocks_recent_subscriptions() -> None:
    from datetime import UTC, datetime, timedelta

    policy = _policy_module()
    policy.record_shadow_probe_symbol_subscribed(["WLDUSDT"])

    assert policy.shadow_probe_symbol_warmed_up("XRPUSDT") is False
    assert policy.shadow_probe_symbol_warmed_up("WLDUSDT") is False

    policy._app._shadow_probe_symbol_subscribed_at["WLDUSDT"] = datetime.now(tz=UTC) - timedelta(seconds=400)
    assert policy.shadow_probe_symbol_warmed_up("WLDUSDT") is True


def test_shadow_probe_regime_allows_trending_only() -> None:
    from trader.domain.enums import MarketRegime

    policy = _policy_module()
    bull = SimpleNamespace(regime=MarketRegime.BULL_TREND)
    sideways = SimpleNamespace(regime=MarketRegime.SIDEWAYS)

    assert policy.shadow_probe_regime_allows(bull) is True
    assert policy.shadow_probe_regime_allows(sideways) is False
    assert policy.shadow_probe_regime_allows(None) is False


def test_shadow_probe_research_v2_targets_high_volatility() -> None:
    from trader.domain.enums import MarketRegime

    policy = _policy_module()
    policy._app._settings.SHADOW_PROBE_RESEARCH_PROFILE_V2 = True
    policy._app._settings.SHADOW_PROBE_PAPER_COLLECTION_MODE = False

    assert policy.shadow_probe_regime_allows(SimpleNamespace(regime=MarketRegime.HIGH_VOLATILITY)) is True
    assert policy.shadow_probe_regime_allows(SimpleNamespace(regime=MarketRegime.BULL_TREND)) is False


def test_shadow_probe_paper_collection_expands_regimes() -> None:
    from trader.domain.enums import MarketRegime

    policy = _policy_module()
    policy._app._settings.SHADOW_PROBE_RESEARCH_PROFILE_V2 = True
    policy._app._settings.SHADOW_PROBE_PAPER_COLLECTION_MODE = True
    policy._app._settings.SHADOW_PROBE_PAPER_REGIMES = "SIDEWAYS,HIGH_VOLATILITY,UNCERTAIN,BULL_TREND,BEAR_TREND"

    assert policy.shadow_probe_regime_allows(SimpleNamespace(regime=MarketRegime.SIDEWAYS)) is True
    assert policy.shadow_probe_regime_allows(SimpleNamespace(regime=MarketRegime.HIGH_VOLATILITY)) is True
    assert policy.shadow_probe_regime_allows(SimpleNamespace(regime=MarketRegime.UNCERTAIN)) is True
    assert policy.shadow_probe_regime_allows(SimpleNamespace(regime=MarketRegime.BULL_TREND)) is True
    assert policy.shadow_probe_regime_allows(SimpleNamespace(regime=MarketRegime.BEAR_TREND)) is True
