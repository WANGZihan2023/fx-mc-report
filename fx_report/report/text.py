"""Build FX Analyse report markdown — any FX pair."""

from __future__ import annotations

from typing import Any

from fx_report.format_rate import format_rate
from fx_report.market.fetch_data import MarketSnapshot
from fx_report.model.monte_carlo import MCResult
from fx_report.model.strength import label_strength, rubric_markdown
from fx_report.model.weights import EvidenceItem, ModelWeights, ScenarioSpec, evidence_score
from fx_report.report.strings import L, normalize_report_lang


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _most_likely(probs: dict[str, float]) -> tuple[str, float]:
    k = max(probs, key=probs.get)
    return k, probs[k]


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

    def evid_lines(items: list[EvidenceItem]) -> str:
        lines = []
        for e in items:
            s = e.direction * e.strength * e.freshness * e.unpriced
            lab = e.strength_label or label_strength(e.strength)
            br = e.strength_breakdown
            br_s = ""
            if br:
                br_s = "｜计分 " + ", ".join(f"{k}={v:.2f}" for k, v in br.items() if k != "sum")
            stance = (getattr(e, "stance_summary", None) or "").strip()
            stance_bit = f"｜{L('stance_summary', lang=lang)}：{stance}" if stance else ""
            lines.append(
                f"- **{e.id}** [{lab}] {e.title}｜贡献 {s:+.2f}｜"
                f"strength={e.strength:.2f} × freshness={e.freshness:.2f} × unpriced={e.unpriced:.2f}"
                f"{br_s}{stance_bit}"
                + (f"｜{e.note}" if e.note else "")
            )
        return "\n".join(lines) if lines else L("md_none", lang=lang)

    scen_tbl = "\n".join(
        f"| {s.name} | {s.weight:.1%} | μ={s.mu_annual:+.2%} | σ×{s.sigma_mult:.2f} | "
        f"E[jumps]={s.expected_jumps:.2f} | {s.narrative} |"
        for s in scenarios_adj
    )
    raw_rows = "\n".join(f"| {k} | {_pct(v)} |" for k, v in mc.raw_probs.items())
    edge_s = " / ".join(format_rate(x) for x in bucket_edges)
    peak = getattr(mc, "peak_engine", weights.peak_engine)

    if lang == "en":
        md = f"""{L("md_title", lang=lang)}

{L("md_question", lang=lang, horizon=horizon_label, pair=pair)}

**Generated:** {market.asof}  
**Sims:** {mc.n_sims:,}｜**Trading days:** {mc.trading_days}｜**Seed:** {weights.seed}｜**Peak engine:** {peak}  
**Market source:** {market.source}  
**Bucket edges:** {edge_s}

---

{L("md_prob", lang=lang)}

| Range | Probability |
|------|------|
{rows}

{L("md_most", lang=lang, top=top, p=_pct(top_p))}

---

{L("md_anchor", lang=lang)}

| Field | Value |
|------|-----|
| Pair | {pair} |
| Spot (analysis quote) | {format_rate(market.spot)} |
| Provider raw | {format_rate(market.provider_raw)} |
| Source | {market.source} |
| Daily σ | {market.sigma_daily:.4%} |
| Annual σ | {market.sigma_annual:.2%} |
| Lookback | {market.lookback_days}d ({market.history_start} → {market.history_end}) |
| Brent | {f"{market.brent:.2f}" if market.brent else "N/A"} |
| DXY | {f"{market.dxy_proxy:.2f}" if market.dxy_proxy else "N/A"} |

**Notes:** {"; ".join(market.notes) if market.notes else "—"}

---

{L("md_up", lang=lang, pair=pair)}

{evid_lines(up)}

{L("md_down", lang=lang, pair=pair)}

{evid_lines(down)}

---

{L("md_exec", lang=lang)}

Start **{pair} ≈ {format_rate(market.spot)}**. {mc.trading_days} trading days, **{mc.n_sims:,}** Monte Carlo mixture (peak `{peak}`), evidence score **S={score:+.2f}** (μ shift {mu_shift:+.2%} ann., σ ×{sigma_extra:.3f}).

Most likely: **`{top}` ({_pct(top_p)})**. Peak percentiles P50={format_rate(mc.percentiles['p50'])}, P90={format_rate(mc.percentiles['p90'])}, P95={format_rate(mc.percentiles['p95'])}.

Math floor: ranges strictly below spot are zeroed then renormalized.

---

## Scenario weights (calibrated)

| Scenario | Weight | Drift | σ mult | Jumps | Narrative |
|------|------|------|----------|------|------|
{scen_tbl}

---

## Raw MC frequencies

| Range | Frequency |
|------|------|
{raw_rows}

---

## Strength rubric

{rubric_markdown()}

---

## {L("watch_title", lang=lang)}

1. **Core central bank / data for the pair** — surprise → re-score and re-run.  
2. **Risk assets / safe haven** — systemic shock lifts escalation; easing lifts de-escalation.  
3. **Commodities / China demand** (if relevant) — adjust U-CN / oil evidence.  
4. **Already priced** — large spot jump → lower unpriced to avoid double-count.

---

{L("md_disclaimer", lang=lang)}
"""
        return md

    md = f"""{L("md_title", lang=lang)}

{L("md_question", lang=lang, horizon=horizon_label, pair=pair)}

**预测生成：** {market.asof}  
**模拟次数：** {mc.n_sims:,}｜**交易日：** {mc.trading_days}｜**种子：** {weights.seed}｜**峰值引擎：** {peak}  
**行情来源：** {market.source}  
**分档边界：** {edge_s}

---

{L("md_prob", lang=lang)}

| 区间 | 概率 |
|------|------|
{rows}

{L("md_most", lang=lang, top=top, p=_pct(top_p))}

---

{L("md_anchor", lang=lang)}

| 字段 | 值 |
|------|-----|
| 货币对 | {pair} |
| 现价（分析口径） | {format_rate(market.spot)} |
| 源端原始报价 | {format_rate(market.provider_raw)} |
| 行情来源 | {market.source} |
| 日波动 σ_d | {market.sigma_daily:.4%} |
| 年化 σ | {market.sigma_annual:.2%} |
| 回看 | {market.lookback_days} 日（{market.history_start} → {market.history_end}） |
| Brent | {f"{market.brent:.2f}" if market.brent else "N/A"} |
| DXY | {f"{market.dxy_proxy:.2f}" if market.dxy_proxy else "N/A"} |
| 近1日涨跌 | {f"{100*market.ret_1d:+.2f}%" if market.ret_1d is not None else "N/A"} |
| 近5日涨跌 | {f"{100*market.ret_5d:+.2f}%" if market.ret_5d is not None else "N/A"} |
| 近20日涨跌 | {f"{100*market.ret_20d:+.2f}%" if market.ret_20d is not None else "N/A"} |
| 20D/60D 年化波动 | {f"{(market.sigma_20d_ann or 0)*100:.2f}% / {(market.sigma_60d_ann or 0)*100:.2f}%" if market.sigma_60d_ann else "N/A"} |
| 历史代码 / 现价代码 | {market.history_ticker} / {market.spot_ticker}{"（代理）" if market.used_proxy else ""} |
| CNH−CNY 价差 | {format_rate(market.cnh_cny_basis, signed=True, na="N/A")} |

**数据说明：** {"；".join(market.notes) if market.notes else "无额外备注"}

---

{L("md_up", lang=lang, pair=pair)}

{evid_lines(up)}

{L("md_down", lang=lang, pair=pair)}

{evid_lines(down)}

---

{L("md_exec", lang=lang)}

起点 **{pair} ≈ {format_rate(market.spot)}**。{mc.trading_days} 个交易日、**{mc.n_sims:,}** 次情景混合蒙特卡洛（峰值引擎 `{peak}`），证据分 **S={score:+.2f}** 校准权重与参数（μ 平移 {mu_shift:+.2%} 年化，σ ×{sigma_extra:.3f}）。

最可能：**`{top}`（{_pct(top_p)}）**。峰值分位 P50={format_rate(mc.percentiles['p50'])}，P90={format_rate(mc.percentiles['p90'])}，P95={format_rate(mc.percentiles['p95'])}。

数学地板：严格低于起点的最高价区间归零后归一化。

---

## 情景权重（校准后）

| 情景 | 权重 | 漂移 | 波动倍数 | 跳跃 | 叙事 |
|------|------|------|----------|------|------|
{scen_tbl}

---

## 原始 MC 频率

| 区间 | 频率 |
|------|------|
{raw_rows}

---

## 信息强弱判定规则

{rubric_markdown()}

---

## {L("watch_title", lang=lang)}

1. **该货币对核心央行/数据** — 决议或重磅意外 → 改 surprise/scope 并重跑。  
2. **风险资产与避险** — 系统性冲击抬 escalation；缓和抬 deescalation。  
3. **商品/中国需求**（若相关）— 改 U-CN / 油价证据方向与未定价。  
4. **已定价程度** — 即期已大跳则下调 unpriced，避免双计。

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
