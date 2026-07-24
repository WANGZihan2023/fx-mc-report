#!/usr/bin/env python3
"""CLI: market + headlines → LLM/rules evidence → MC → report."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

from fetch_data import calibrate_unpriced_from_market, fetch_market
from monte_carlo import enforce_math_floor, run_mixture_monte_carlo
from news_evidence import build_evidence_from_news
from news_fetch import fetch_headlines_for_pair
from news_llm import resolve_llm_config
from pairs import get_pair, list_pairs, make_custom_pair
from report_text import build_diagnostics, build_report_markdown
from weights import (
    apply_evidence_to_scenarios,
    default_weights,
    evidence_score,
    resolve_bucket_edges,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Multi-pair FX peak-bucket Monte Carlo")
    p.add_argument("--pair", default="USD/AUD", help=f"One of: {', '.join(list_pairs())}")
    p.add_argument("--ticker", default=None, help="Override Yahoo ticker (custom pair)")
    p.add_argument("--invert", action="store_true", help="Invert Yahoo close")
    p.add_argument("--sims", type=int, default=100_000)
    p.add_argument("--days", type=int, default=66)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--out", type=str, default="output")
    p.add_argument("--no-news", action="store_true", help="Skip headline fetch; use templates only")
    p.add_argument("--keep-templates", action="store_true", help="Merge templates with news evidence")
    p.add_argument("--max-news", type=int, default=10)
    p.add_argument(
        "--mode",
        choices=["hybrid", "llm", "rules"],
        default="hybrid",
        help="Evidence classifier: hybrid|llm|rules",
    )
    p.add_argument("--no-fulltext", action="store_true", help="Do not fetch article HTML for LLM")
    args = p.parse_args()

    if args.ticker:
        spec = make_custom_pair(args.pair, args.ticker, args.invert)
    else:
        spec = get_pair(args.pair)

    w = default_weights(spec)
    w.n_sims = args.sims
    w.trading_days = args.days
    w.seed = args.seed
    w.vol_lookback_days = args.lookback

    print(f"Pair {spec.pair}  ticker={spec.yahoo_ticker} invert={spec.invert}")
    market = fetch_market(spec, lookback_days=w.vol_lookback_days)
    print(
        f"  spot={market.spot:.5f}  sigma_d={market.sigma_daily:.4%}  "
        f"sigma_ann={market.sigma_annual:.2%}  hist={market.history_ticker} "
        f"proxy={market.used_proxy}"
    )
    for n in market.notes:
        print(f"  note: {n}")

    headlines = []
    news_meta: dict = {}
    if not args.no_news:
        print("Fetching headlines…")
        headlines = fetch_headlines_for_pair(spec, max_items=30)
        print(f"  headlines fetched: {len(headlines)}")
        suggested_up = calibrate_unpriced_from_market(market.ret_1d, market.ret_5d)
        cfg = resolve_llm_config()
        mode = args.mode
        if mode in {"llm", "hybrid"} and cfg is None:
            print("  no LLM API key; using rules")
            mode = "rules"
        elif cfg is not None:
            print(f"  LLM: model={cfg.model} base={cfg.base_url}")

        auto, news_meta = build_evidence_from_news(
            headlines,
            spec,
            mode=mode,  # type: ignore[arg-type]
            max_items=args.max_news,
            unpriced_cap=suggested_up,
            llm_cfg=cfg,
            fetch_fulltext=not args.no_fulltext,
        )
        print(f"  evidence: {len(auto)}  meta={ {k: news_meta.get(k) for k in ('mode','rules_used')} }")
        if news_meta.get("llm"):
            print(f"  llm meta: {news_meta['llm']}")
        if auto:
            w.evidence = auto + (list(w.evidence) if args.keep_templates else [])
        else:
            print("  fallback to template evidence")

    suggested_up = calibrate_unpriced_from_market(market.ret_1d, market.ret_5d)
    for e in w.evidence:
        e.unpriced = min(e.unpriced, suggested_up)

    score = evidence_score(w.evidence)
    mu_shift = w.score_to_mu_a * score
    sigma_extra = 1.0 + w.score_to_sigma_b * abs(score)
    scenarios = apply_evidence_to_scenarios(
        w.scenarios,
        score,
        logit_scale=w.evidence_logit_scale,
        temperature=w.scenario_temperature,
        max_shift=w.max_scenario_shift,
    )
    print(f"S={score:+.3f}  mu_shift={mu_shift:+.4f}  sigma_extra={sigma_extra:.3f}")
    for e in w.evidence:
        print(
            f"  evidence {e.id} [{e.strength_label}] dir={e.direction:+d} "
            f"cat={e.category} strength={e.strength:.2f} | {e.title[:70]}"
        )
    for s in scenarios:
        print(f"  scenario {s.name}: {s.weight:.1%}")

    edges = resolve_bucket_edges(w, market.spot)
    mc = run_mixture_monte_carlo(
        spot=market.spot,
        sigma_daily_base=market.sigma_daily,
        scenarios=scenarios,
        trading_days=w.trading_days,
        n_sims=w.n_sims,
        seed=w.seed,
        bucket_edges=edges,
        mu_annual_shift=mu_shift,
        sigma_mult_extra=sigma_extra,
    )
    probs = enforce_math_floor(mc.raw_probs, market.spot, edges)
    for k, v in probs.items():
        print(f"  {k}: {v:.1%}")

    start = date.today()
    end = start + timedelta(days=max(int(w.trading_days * 1.4), 1))
    report = build_report_markdown(
        market,
        w,
        scenarios,
        mc,
        probs,
        score=score,
        mu_shift=mu_shift,
        sigma_extra=sigma_extra,
        horizon_label=f"{start} 至 {end}",
        bucket_edges=edges,
    )
    diag = build_diagnostics(
        market, w, scenarios, mc, probs, score, mu_shift, sigma_extra, edges
    )
    diag["headlines"] = [
        {
            "title": h.title,
            "source": h.source,
            "published": h.published.isoformat() if h.published else None,
            "url": h.url,
            "provider": h.provider,
        }
        for h in headlines
    ]
    diag["news_meta"] = news_meta

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    safe = spec.pair.replace("/", "")
    (out / f"{safe}_report.md").write_text(report, encoding="utf-8")
    (out / f"{safe}_diagnostics.json").write_text(
        json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out / (safe + '_report.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
