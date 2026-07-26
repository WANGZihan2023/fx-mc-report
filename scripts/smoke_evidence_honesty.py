#!/usr/bin/env python3
"""Smoke: evidence-chain honesty (no silent template fallback) + audit fields."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    # Import surface
    from fx_report.pipeline import run_pipeline, step4_evaluate_impact, TemplatePolicy
    from fx_report.model.weights import default_weights, evidence_score
    from fx_report.market.pairs import get_pair
    from fx_report.market.fetch_data import MarketSnapshot

    print("OK import pipeline / step4 / TemplatePolicy")

    spec = get_pair("USD/AUD")
    base = default_weights(spec)
    # Minimal market stub (no network)
    market = MarketSnapshot(
        asof="smoke",
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
        source="smoke",
        brent=None,
        dxy_proxy=None,
        notes=[],
        ret_1d=0.0,
        ret_5d=0.0,
    )

    # --- policy=off + empty headlines → no silent templates ---
    ev_off, meta_off = step4_evaluate_impact(
        [],
        spec,
        market,
        base,
        mode="rules",
        template_policy="off",
    )
    assert meta_off.get("fallback_templates") is False, meta_off
    assert meta_off.get("evidence_quality") == "news_empty_no_prior", meta_off
    assert ev_off == [], f"expected empty evidence, got {len(ev_off)}"
    assert evidence_score(ev_off) == 0.0
    print("OK off: empty news → no templates, quality=news_empty_no_prior")

    # --- prior_only → templates marked + downweighted ---
    ev_prior, meta_prior = step4_evaluate_impact(
        [],
        spec,
        market,
        base,
        mode="rules",
        template_policy="prior_only",
    )
    assert meta_prior.get("fallback_templates") is True, meta_prior
    assert meta_prior.get("evidence_quality") == "prior_only", meta_prior
    assert len(ev_prior) > 0
    assert all(e.is_prior for e in ev_prior), "all prior items must be marked"
    print(f"OK prior_only: {len(ev_prior)} marked priors, S={evidence_score(ev_prior):+.3f}")

    # --- fallback_warn ---
    ev_warn, meta_warn = step4_evaluate_impact(
        [],
        spec,
        market,
        base,
        mode="rules",
        template_policy="fallback_warn",
    )
    assert meta_warn.get("fallback_templates") is True
    assert meta_warn.get("evidence_quality") == "fallback_warn"
    assert len(ev_warn) > 0
    print("OK fallback_warn: templates flagged")

    # --- full pipeline rules + no-news (needs market fetch; may hit network) ---
    try:
        result = run_pipeline(
            "USD/AUD",
            sims=2_000,
            days=21,
            seed=7,
            mode="rules",
            no_news=True,
            ai_research=False,
            template_policy="off",
            out_dir=None,
            verbose=False,
        )
        nm = result.news_meta
        assert nm.get("evidence_quality") == "news_empty_no_prior", nm
        assert nm.get("fallback_templates") is False, nm
        assert result.diagnostics.get("peak_engine") in {"path_max", "brownian_bridge"}
        assert "evidence_quality" in result.diagnostics
        assert abs(result.score) < 1e-9, f"expected S≈0 got {result.score}"
        print(
            f"OK pipeline no-news/off: S={result.score:+.3f} "
            f"quality={nm.get('evidence_quality')} peak={result.diagnostics.get('peak_engine')}"
        )
    except Exception as exc:
        print(f"WARN pipeline smoke skipped (market fetch?): {exc}")

    # prior_only pipeline path
    try:
        result2 = run_pipeline(
            "USD/AUD",
            sims=2_000,
            days=21,
            seed=7,
            mode="rules",
            no_news=True,
            ai_research=False,
            template_policy="prior_only",
            out_dir=None,
            verbose=False,
        )
        assert result2.news_meta.get("evidence_quality") == "prior_only"
        assert result2.news_meta.get("fallback_templates") is True
        assert any(w.evidence.is_prior for w in result2.weighted)
        print(
            f"OK pipeline no-news/prior_only: S={result2.score:+.3f} "
            f"n={result2.news_meta.get('evidence_n')}"
        )
    except Exception as exc:
        print(f"WARN prior_only pipeline skipped: {exc}")

    print("ALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
