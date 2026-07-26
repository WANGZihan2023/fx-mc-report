"""Build FX Analyse report markdown — any FX pair."""

from __future__ import annotations

from typing import Any

from fx_report.market.fetch_data import MarketSnapshot
from fx_report.model.monte_carlo import MCResult
from fx_report.model.strength import label_strength, rubric_markdown
from fx_report.model.weights import EvidenceItem, ModelWeights, ScenarioSpec, evidence_score


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
) -> str:
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
            lines.append(
                f"- **{e.id}** [{lab}] {e.title}｜贡献 {s:+.2f}｜"
                f"strength={e.strength:.2f} × freshness={e.freshness:.2f} × unpriced={e.unpriced:.2f}"
                f"{br_s}"
                + (f"｜{e.note}" if e.note else "")
            )
        return "\n".join(lines) if lines else "_（无）_"

    scen_tbl = "\n".join(
        f"| {s.name} | {s.weight:.1%} | μ={s.mu_annual:+.2%} | σ×{s.sigma_mult:.2f} | "
        f"E[jumps]={s.expected_jumps:.2f} | {s.narrative} |"
        for s in scenarios_adj
    )
    raw_rows = "\n".join(f"| {k} | {_pct(v)} |" for k, v in mc.raw_probs.items())
    edge_s = " / ".join(f"{x:.4f}" for x in bucket_edges)

    md = f"""# FX ANALYSE · 情报报告（多货币对引擎）

**问题：** {horizon_label} 内，**{pair}** 的**最高日高**将落在哪一档？

**预测生成：** {market.asof}  
**模拟次数：** {mc.n_sims:,}｜**交易日：** {mc.trading_days}｜**种子：** {weights.seed}｜**峰值引擎：** {getattr(mc, "peak_engine", weights.peak_engine)}  
**行情来源：** {market.source}  
**分档边界：** {edge_s}

---

## 概率分布

| 区间 | 概率 |
|------|------|
{rows}

**最可能区间：`{top}`（{_pct(top_p)}）**

---

## 行情锚点

| 字段 | 值 |
|------|-----|
| 货币对 | {pair} |
| 现价（分析口径） | {market.spot:.5f} |
| 源端原始报价 | {market.provider_raw:.5f} |
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
| CNH−CNY 价差 | {f"{market.cnh_cny_basis:+.4f}" if market.cnh_cny_basis is not None else "N/A"} |

**数据说明：** {"；".join(market.notes) if market.notes else "无额外备注"}

---

## 上行驱动（推高 {pair} 峰值）

{evid_lines(up)}

## 下行驱动（压制 {pair} 峰值）

{evid_lines(down)}

---

## 执行摘要

起点 **{pair} ≈ {market.spot:.5f}**。{mc.trading_days} 个交易日、**{mc.n_sims:,}** 次情景混合蒙特卡洛（峰值引擎 `{getattr(mc, "peak_engine", weights.peak_engine)}`），证据分 **S={score:+.2f}** 校准权重与参数（μ 平移 {mu_shift:+.2%} 年化，σ ×{sigma_extra:.3f}）。

最可能：**`{top}`（{_pct(top_p)}）**。峰值分位 P50={mc.percentiles['p50']:.5f}，P90={mc.percentiles['p90']:.5f}，P95={mc.percentiles['p95']:.5f}。

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

## What to Watch

1. **该货币对核心央行/数据** — 决议或重磅意外 → 改 surprise/scope 并重跑。  
2. **风险资产与避险** — 系统性冲击抬 escalation；缓和抬 deescalation。  
3. **商品/中国需求**（若相关）— 改 U-CN / 油价证据方向与未定价。  
4. **已定价程度** — 即期已大跳则下调 unpriced，避免双计。

---

*概率模型输出，不构成投资建议。*
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
        "evidence_score_check": evidence_score(weights.evidence),
    }
