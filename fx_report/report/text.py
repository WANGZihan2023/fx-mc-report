"""Build FX Analyse report markdown — any FX pair."""

from __future__ import annotations

from typing import Any

from fx_report.format_rate import format_rate
from fx_report.market.fetch_data import MarketSnapshot
from fx_report.model.monte_carlo import MCResult
from fx_report.model.strength import label_strength, rubric_markdown
from fx_report.model.weights import EvidenceItem, ModelWeights, ScenarioSpec, evidence_score
from fx_report.report.strings import L, normalize_report_lang, scenario_narrative


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _most_likely(probs: dict[str, float]) -> tuple[str, float]:
    k = max(probs, key=probs.get)
    return k, probs[k]


def _scenario_line(s: ScenarioSpec, pair: str, *, lang: str) -> str:
    narr = scenario_narrative(s.name, pair, lang=lang) or (s.narrative or "")
    return (
        f"| {s.name} | {s.weight:.1%} | μ={s.mu_annual:+.2%} | σ×{s.sigma_mult:.2f} | "
        f"E[jumps]={s.expected_jumps:.2f} | {narr} |"
    )


def build_report_markdown(
    market: MarketSnapshot,
    weights: ModelWeights,
    scenarios_adj: list[ScenarioSpec],
    mc: MCResult,
    probs: dict[str, float],
    *,
    score: float,
    mu_shift: float,
    sigma_extra: float,
    horizon_label: str,
    bucket_edges: tuple[float, float, float, float],
    lang: str = "zh",
) -> str:
    lang = normalize_report_lang(lang)
    pair = market.pair
    top, top_p = _most_likely(probs)
    rows = "\n".join(f"| {k} | {_pct(v)} |" for k, v in probs.items())

    up = [e for e in weights.evidence if e.direction > 0]
    down = [e for e in weights.evidence if e.direction < 0]
    sep = "｜"

    def evid_lines(items: list[EvidenceItem]) -> str:
        lines = []
        for e in items:
            s = e.direction * e.strength * e.freshness * e.unpriced
            lab = e.strength_label or label_strength(e.strength)
            br = e.strength_breakdown
            br_s = ""
            if br:
                br_s = (
                    sep
                    + L("evid_scoring", lang=lang)
                    + " "
                    + ", ".join(f"{k}={v:.2f}" for k, v in br.items() if k != "sum")
                )
            stance = (getattr(e, "stance_summary", None) or "").strip()
            i18n = getattr(e, "stance_summary_i18n", None) or {}
            if isinstance(i18n, dict) and (i18n.get(lang) or "").strip():
                stance = str(i18n[lang]).strip()
            stance_bit = (
                f"{sep}{L('stance_summary', lang=lang)}：{stance}" if stance else ""
            )
            lines.append(
                f"- **{e.id}** [{lab}] {e.title}{sep}{L('evid_contrib', lang=lang)} "
                f"{s:+.2f}{sep}"
                f"strength={e.strength:.2f} × freshness={e.freshness:.2f} × "
                f"unpriced={e.unpriced:.2f}"
                f"{br_s}{stance_bit}"
                + (f"{sep}{e.note}" if e.note else "")
            )
        return "\n".join(lines) if lines else L("md_none", lang=lang)

    scen_tbl = "\n".join(_scenario_line(s, pair, lang=lang) for s in scenarios_adj)
    raw_rows = "\n".join(f"| {k} | {_pct(v)} |" for k, v in mc.raw_probs.items())
    edge_s = " / ".join(format_rate(x) for x in bucket_edges)
    peak = getattr(mc, "peak_engine", weights.peak_engine)
    lookback = (
        f"{market.lookback_days}d ({market.history_start} → {market.history_end})"
        if lang == "en"
        else f"{market.lookback_days} 日（{market.history_start} → {market.history_end}）"
    )
    proxy = L("md_proxy", lang=lang) if market.used_proxy else ""
    notes = (
        ("; " if lang == "en" else "；").join(market.notes)
        if market.notes
        else L("md_notes_empty", lang=lang)
    )
    ret1 = f"{100 * market.ret_1d:+.2f}%" if market.ret_1d is not None else "N/A"
    ret5 = f"{100 * market.ret_5d:+.2f}%" if market.ret_5d is not None else "N/A"
    ret20 = f"{100 * market.ret_20d:+.2f}%" if market.ret_20d is not None else "N/A"
    vol2060 = (
        f"{(market.sigma_20d_ann or 0) * 100:.2f}% / {(market.sigma_60d_ann or 0) * 100:.2f}%"
        if market.sigma_60d_ann
        else "N/A"
    )

    md = f"""{L("md_title", lang=lang)}

{L("md_question", lang=lang, horizon=horizon_label, pair=pair)}

{L("md_generated", lang=lang, d=market.asof)}  
{L("md_sims_line", lang=lang, n=f"{mc.n_sims:,}", days=mc.trading_days, seed=weights.seed, peak=peak)}  
{L("md_market_src", lang=lang, s=market.source)}  
{L("md_bucket_edges", lang=lang, e=edge_s)}

---

{L("md_prob", lang=lang)}

| {L("md_col_range", lang=lang)} | {L("md_col_prob", lang=lang)} |
|------|------|
{rows}

{L("md_most", lang=lang, top=top, p=_pct(top_p))}

---

{L("md_anchor", lang=lang)}

| {L("md_col_field", lang=lang)} | {L("md_col_value", lang=lang)} |
|------|-----|
| {L("md_field_pair", lang=lang)} | {pair} |
| {L("md_field_spot", lang=lang)} | {format_rate(market.spot)} |
| {L("md_field_raw", lang=lang)} | {format_rate(market.provider_raw)} |
| {L("md_field_source", lang=lang)} | {market.source} |
| {L("md_field_sigma_d", lang=lang)} | {market.sigma_daily:.4%} |
| {L("md_field_sigma_a", lang=lang)} | {market.sigma_annual:.2%} |
| {L("md_field_lookback", lang=lang)} | {lookback} |
| Brent | {f"{market.brent:.2f}" if market.brent else "N/A"} |
| DXY | {f"{market.dxy_proxy:.2f}" if market.dxy_proxy else "N/A"} |
| {L("md_field_ret1", lang=lang)} | {ret1} |
| {L("md_field_ret5", lang=lang)} | {ret5} |
| {L("md_field_ret20", lang=lang)} | {ret20} |
| {L("md_field_vol2060", lang=lang)} | {vol2060} |
| {L("md_field_tickers", lang=lang)} | {market.history_ticker} / {market.spot_ticker}{proxy} |
| {L("md_field_basis", lang=lang)} | {format_rate(market.cnh_cny_basis, signed=True, na="N/A")} |

{L("md_notes", lang=lang, n=notes)}

---

{L("md_up", lang=lang, pair=pair)}

{evid_lines(up)}

{L("md_down", lang=lang, pair=pair)}

{evid_lines(down)}

---

{L("md_exec", lang=lang)}

{L("md_exec_body", lang=lang, pair=pair, spot=format_rate(market.spot), days=mc.trading_days, n_sims=f"{mc.n_sims:,}", peak=peak, score=f"{score:+.2f}", mu=f"{mu_shift:+.2%}", sigma=f"{sigma_extra:.3f}")}

{L("md_exec_most", lang=lang, top=top, p=_pct(top_p), p50=format_rate(mc.percentiles['p50']), p90=format_rate(mc.percentiles['p90']), p95=format_rate(mc.percentiles['p95']))}

{L("md_math_floor", lang=lang)}

---

{L("md_scen", lang=lang)}

{L("md_scen_cols", lang=lang)}
|------|------|------|----------|------|------|
{scen_tbl}

---

{L("md_raw_mc", lang=lang)}

| {L("md_col_range", lang=lang)} | {L("md_col_freq", lang=lang)} |
|------|------|
{raw_rows}

---

{L("md_rubric", lang=lang)}

{rubric_markdown(lang=lang)}

---

## {L("watch_title", lang=lang)}

{L("md_watch1", lang=lang)}  
{L("md_watch2", lang=lang)}  
{L("md_watch3", lang=lang)}  
{L("md_watch4", lang=lang)}

---

{L("md_disclaimer", lang=lang)}
"""
    return md


def build_diagnostics(
    market: MarketSnapshot,
    weights: ModelWeights,
    scenarios_adj: list[ScenarioSpec],
    mc: MCResult,
    probs: dict[str, float],
    score: float,
    mu_shift: float,
    sigma_extra: float,
    bucket_edges: tuple[float, float, float, float],
) -> dict[str, Any]:
    return {
        "market": market.to_dict(),
        "score_S": score,
        "mu_annual_shift": mu_shift,
        "sigma_mult_extra": sigma_extra,
        "mapping": {
            "score_to_mu_a": weights.score_to_mu_a,
            "score_to_sigma_b": weights.score_to_sigma_b,
            "evidence_logit_scale": weights.evidence_logit_scale,
            "scenario_temperature": weights.scenario_temperature,
            "max_scenario_shift": weights.max_scenario_shift,
            "variance_reduction": getattr(mc, "variance_reduction", "none"),
        },
        "strength_rubric": "strength.py v1: source + surprise + scope (cap 3)",
        "scenarios_adjusted": [s.__dict__ for s in scenarios_adj],
        "scenario_path_counts": mc.scenario_counts,
        "raw_probs": mc.raw_probs,
        "calibrated_probs": probs,
        "percentiles": mc.percentiles,
        "bucket_edges": list(bucket_edges),
        "bucket_pct_cuts": list(weights.bucket_pct_cuts),
        "n_sims": mc.n_sims,
        "seed": weights.seed,
        "peak_engine": getattr(mc, "peak_engine", getattr(weights, "peak_engine", "path_max")),
        "variance_reduction": getattr(mc, "variance_reduction", "none"),
        "jump_model": getattr(mc, "jump_model", getattr(weights, "jump_model", "merton")),
        "jump_compensate": bool(
            getattr(mc, "jump_compensate", getattr(weights, "jump_compensate", False))
        ),
        "bb_jumps_caveat": getattr(mc, "bb_jumps_caveat", None),
        "evidence_score_check": evidence_score(weights.evidence),
        "evidence_quality": None,
        "fallback_templates": False,
    }
