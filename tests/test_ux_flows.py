"""UX / high-traffic flow regression tests (pure helpers + pipeline smoke)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from fx_report.market.pairs import (
    edges_from_spot,
    get_pair,
    list_pairs,
    resolve_pair_for_bullish,
)
from fx_report.model.backtest import compare_peak_engines
from fx_report.model.calibrate import (
    load_calib_oos_summary,
    resolve_calibrated_params_path,
)
from fx_report.model.monte_carlo import run_mixture_monte_carlo
from fx_report.model.weights import default_weights, resolve_bucket_edges
from fx_report.pipeline import step4_evaluate_impact
from fx_report.ui.ux_helpers import (
    PCT_CUT_MIN,
    abs_edges_to_pct_cuts,
    app_password_expected,
    bb_jump_compensate_warning,
    password_accepted,
    pct_cuts_in_bounds,
    seed_pct_widget_value,
    should_heal_floor_clamp,
)


# ---------------------------------------------------------------------------
# 1. Password gate
# ---------------------------------------------------------------------------


def test_password_default_uniocean():
    assert app_password_expected(environ={}) == "uniocean"


def test_password_env_override():
    assert app_password_expected(environ={"APP_PASSWORD": "secret"}) == "secret"
    assert app_password_expected(environ={"FX_REPORT_PASSWORD": "fx"}) == "fx"
    # blank env falls through
    assert app_password_expected(environ={"APP_PASSWORD": "  "}) == "uniocean"


def test_password_accept_reject_empty():
    expected = "uniocean"
    assert password_accepted("uniocean", expected) is True
    assert password_accepted("wrong", expected) is False
    assert password_accepted("", expected) is False
    assert password_accepted(None, expected) is False


# ---------------------------------------------------------------------------
# 2. Pair + bullish
# ---------------------------------------------------------------------------


def test_resolve_bullish_flip_and_block_empty():
    spec = get_pair("USD/AUD")
    assert resolve_pair_for_bullish(spec, "USD").pair == "USD/AUD"
    flipped = resolve_pair_for_bullish(spec, "AUD")
    assert flipped.pair == "AUD/USD"
    with pytest.raises(ValueError):
        resolve_pair_for_bullish(spec, "")
    with pytest.raises(ValueError):
        resolve_pair_for_bullish(spec, "EUR")


def test_catalog_pairs_include_usdaud():
    assert "USD/AUD" in list_pairs()


# ---------------------------------------------------------------------------
# 3. Spot buckets / -20 clamp regression
# ---------------------------------------------------------------------------


def test_default_cuts_are_not_minus_twenty():
    cuts = get_pair("USD/AUD").bucket_pct_cuts
    assert cuts == (0.0, 2.0, 4.0, 6.0)
    assert not all(abs(c - PCT_CUT_MIN) < 1e-9 for c in cuts)


def test_abs_wrong_quote_overflow_not_pushed_to_widgets():
    """AUD/USD-ish levels on USD/AUD spot → huge negative % → must stay OOB."""
    spot_usdaud = 1.43
    # Mistakenly pasted AUD/USD-style edges (~0.70)
    bad_edges = (0.70, 0.71, 0.72, 0.73)
    raw = abs_edges_to_pct_cuts(spot_usdaud, bad_edges)
    assert min(raw) < PCT_CUT_MIN
    assert not pct_cuts_in_bounds(raw)
    # Seeding widgets would clamp — that marks OOB for heal
    seeded = [seed_pct_widget_value(p) for p in raw]
    assert all(oob for _, oob in seeded)
    assert all(abs(v - PCT_CUT_MIN) < 1e-9 for v, _ in seeded)


def test_heal_floor_only_when_seeded_from_oob():
    floor = [PCT_CUT_MIN] * 4
    assert should_heal_floor_clamp(floor, seeded_from_oob=True) is True
    # Intentional user edit to all -20 must NOT auto-reset
    assert should_heal_floor_clamp(floor, seeded_from_oob=False) is False
    assert should_heal_floor_clamp([0, 2, 4, 6], seeded_from_oob=True) is False


def test_relative_edges_roundtrip():
    spot = 1.4287
    cuts = (0.0, 2.0, 4.0, 6.0)
    edges = edges_from_spot(spot, cuts)
    back = abs_edges_to_pct_cuts(spot, edges)
    assert pct_cuts_in_bounds(back)
    for a, b in zip(cuts, back):
        assert abs(a - b) < 1e-9


# ---------------------------------------------------------------------------
# 4. MC settings / BB jump_compensate warning
# ---------------------------------------------------------------------------


def test_bb_jump_compensate_warning():
    assert bb_jump_compensate_warning(peak_engine="path_max", jump_compensate=True) is None
    assert bb_jump_compensate_warning(peak_engine="brownian_bridge", jump_compensate=False) is None
    msg = bb_jump_compensate_warning(peak_engine="brownian_bridge", jump_compensate=True)
    assert msg is not None
    assert "jump_compensate" in msg
    assert "path_max" in msg


def test_jump_none_vs_merton_changes_path_max():
    w = default_weights("USD/AUD")
    # Force jump intensity so jump_model matters
    scenarios = [
        replace(s, expected_jumps=2.0, jump_mean=0.01, jump_std=0.01) for s in w.scenarios
    ]
    edges = (1.40, 1.43, 1.46, 1.49)
    common = dict(
        spot=1.43,
        sigma_daily_base=0.005,
        scenarios=scenarios,
        trading_days=21,
        n_sims=6_000,
        seed=11,
        bucket_edges=edges,
        peak_engine="path_max",
    )
    merton = run_mixture_monte_carlo(**common, jump_model="merton")
    none = run_mixture_monte_carlo(**common, jump_model="none")
    assert merton.raw_probs != none.raw_probs


# ---------------------------------------------------------------------------
# 5. Calibrated params / OOS for USDAUD
# ---------------------------------------------------------------------------


def test_usdaud_calibrated_and_oos_present():
    path = resolve_calibrated_params_path("USD/AUD")
    assert path is not None and path.exists()
    oos = load_calib_oos_summary("USD/AUD")
    assert oos is not None
    hold = oos.get("holdout") or {}
    assert hold.get("n")
    # Skill / Brier keys exist (values may be low)
    assert "brier" in hold
    assert "skill_brier" in hold or "hit_rate" in hold


# ---------------------------------------------------------------------------
# 6. Empty evidence honesty
# ---------------------------------------------------------------------------


def test_empty_evidence_policy_off(monkeypatch):
    from fx_report.market.fetch_data import MarketSnapshot
    from fx_report.model.weights import default_weights as dw

    spec = get_pair("USD/AUD")
    base = dw(spec)
    market = MarketSnapshot(
        asof="test",
        pair=spec.pair,
        spot=1.55,
        provider_raw=1.55,
        sigma_daily=0.006,
        sigma_annual=0.095,
        mean_daily_return=0.0,
        n_returns=60,
        lookback_days=60,
        history_start="2020-01-01",
        history_end="2026-01-01",
        source="test",
        brent=None,
        dxy_proxy=None,
        notes=[],
        ret_1d=0.0,
        ret_5d=0.0,
    )
    ev, meta = step4_evaluate_impact(
        [],
        spec,
        market,
        base,
        mode="rules",
        template_policy="off",
    )
    assert ev == []
    assert meta.get("fallback_templates") is False
    assert meta.get("evidence_quality") == "news_empty_no_prior"


# ---------------------------------------------------------------------------
# 7. Dual-engine compare respects jump_model from weights
# ---------------------------------------------------------------------------


def test_compare_peak_engines_respects_jump_model():
    w = default_weights("USD/AUD")
    w.scenarios = [
        replace(s, expected_jumps=2.5, jump_mean=0.012, jump_std=0.01) for s in w.scenarios
    ]
    edges = resolve_bucket_edges(w, 1.43)
    w.jump_model = "none"
    w.jump_compensate = False
    cmp_none = compare_peak_engines(
        1.43, 0.005, w, edges, n_sims=4_000, seed=3, score=0.0
    )
    w.jump_model = "merton"
    cmp_merton = compare_peak_engines(
        1.43, 0.005, w, edges, n_sims=4_000, seed=3, score=0.0
    )
    # path_max leg must differ when jumps are on vs off; BB may stay close
    assert cmp_none["path_max"] != cmp_merton["path_max"]
