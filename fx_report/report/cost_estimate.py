"""Per-report USD cost estimates (DeepSeek Flash + Tavily) for UI / docs.

Figures are planning estimates — not meter readings. DeepSeek Flash rates
(cache-miss in / out) as of mid-2026: ~$0.14 / $0.28 per 1M tokens.
"""

from __future__ import annotations

from typing import Any

# DeepSeek V4 Flash (cache miss) — USD per 1M tokens
DEEPSEEK_IN_PER_M = 0.14
DEEPSEEK_OUT_PER_M = 0.28

# Rough token budgets observed / assumed for a LIVE report
# AI research (~10 rounds with Tavily): plan + select per round + final extract
_AI_RESEARCH_IN = 45_000
_AI_RESEARCH_OUT = 12_000
# Classify batch (~80 headlines snippets)
_CLASSIFY_IN = 35_000
_CLASSIFY_OUT = 8_000
# Per-ref stance summary: ~350 in + ~80 out tokens per item (batched)
_STANCE_IN_PER = 350
_STANCE_OUT_PER = 80
# Full-report LLM translate (if used; template path is $0)
_TRANSLATE_IN = 8_000
_TRANSLATE_OUT = 8_000

# Tavily: free tier credits; paid ~$0.008–0.01 / search — use $0.01 planning
TAVILY_USD_PER_SEARCH = 0.01
TAVILY_SEARCHES_LIVE = 10  # ~max_rounds with search hands


def _llm_usd(n_in: float, n_out: float) -> float:
    return (n_in / 1_000_000.0) * DEEPSEEK_IN_PER_M + (
        n_out / 1_000_000.0
    ) * DEEPSEEK_OUT_PER_M


def estimate_stance_summary_usd(n_refs: int) -> float:
    n = max(0, int(n_refs))
    return _llm_usd(n * _STANCE_IN_PER, n * _STANCE_OUT_PER)


def estimate_live_report_cost(
    *,
    n_refs: int = 80,
    report_lang_via_template: bool = True,
    include_stance_summaries: bool = False,
    include_ai_research: bool = True,
    include_classify: bool = True,
    include_tavily: bool = True,
) -> dict[str, Any]:
    """
    Return a breakdown dict with USD floats (approx).

    ``report_lang_via_template=True`` → ZH/EN switch costs ~$0 (no LLM translate).
    """
    parts: dict[str, float] = {}
    if include_ai_research:
        parts["ai_research_llm"] = _llm_usd(_AI_RESEARCH_IN, _AI_RESEARCH_OUT)
    else:
        parts["ai_research_llm"] = 0.0
    if include_classify:
        parts["classify_llm"] = _llm_usd(_CLASSIFY_IN, _CLASSIFY_OUT)
    else:
        parts["classify_llm"] = 0.0
    if include_tavily and include_ai_research:
        parts["tavily_searches"] = TAVILY_SEARCHES_LIVE * TAVILY_USD_PER_SEARCH
    else:
        parts["tavily_searches"] = 0.0
    if report_lang_via_template:
        parts["report_language"] = 0.0
    else:
        parts["report_language"] = _llm_usd(_TRANSLATE_IN, _TRANSLATE_OUT)
    if include_stance_summaries:
        parts["stance_summaries"] = estimate_stance_summary_usd(n_refs)
    else:
        parts["stance_summaries"] = 0.0

    total = sum(parts.values())
    return {
        "n_refs": int(n_refs),
        "parts_usd": parts,
        "total_usd": round(total, 4),
        "pricing_note": (
            f"DeepSeek Flash ~${DEEPSEEK_IN_PER_M}/M in · "
            f"${DEEPSEEK_OUT_PER_M}/M out; Tavily ~${TAVILY_USD_PER_SEARCH}/search "
            f"× {TAVILY_SEARCHES_LIVE} (planning)."
        ),
    }


def cost_table_rows_zh() -> list[dict[str, str]]:
    """Concrete Chinese rows for docs / UI caption."""
    base = estimate_live_report_cost(n_refs=80, include_stance_summaries=False)
    zh_tmpl = estimate_live_report_cost(
        n_refs=80, report_lang_via_template=True, include_stance_summaries=False
    )
    zh_llm = estimate_live_report_cost(
        n_refs=80, report_lang_via_template=False, include_stance_summaries=False
    )
    rows = [
        {
            "场景": "基线 Live（AI 检索≈10 轮 + 分类，无逐条总结）",
            "约 USD/份": f"${base['total_usd']:.3f}",
        },
        {
            "场景": "+ 中文 / 英 / 双语报告（模板切换，无整篇 LLM 翻译）",
            "约 USD/份": f"+$0.000（合计 ${zh_tmpl['total_usd']:.3f}；双语不重跑新闻/MC）",
        },
        {
            "场景": "+ 中文报告（若整篇 LLM 翻译，不推荐）",
            "约 USD/份": f"+${zh_llm['parts_usd']['report_language']:.3f}",
        },
    ]
    for n in (30, 80, 100):
        add = estimate_stance_summary_usd(n)
        tot = estimate_live_report_cost(
            n_refs=n,
            report_lang_via_template=True,
            include_stance_summaries=True,
        )["total_usd"]
        rows.append(
            {
                "场景": f"+ 逐条引用总结（{n} 条，DeepSeek 分批）",
                "约 USD/份": f"+${add:.4f}（合计约 ${tot:.3f}）",
            }
        )
    rows.append(
        {
            "场景": "日更 2–3 份（基线+模板中文+80 条总结）",
            "约 USD/份": (
                f"约 ${estimate_live_report_cost(n_refs=80, include_stance_summaries=True)['total_usd']:.3f}"
                f"/份 → 日约 "
                f"${2 * estimate_live_report_cost(n_refs=80, include_stance_summaries=True)['total_usd']:.2f}"
                f"–"
                f"${3 * estimate_live_report_cost(n_refs=80, include_stance_summaries=True)['total_usd']:.2f}"
            ),
        }
    )
    return rows


def cost_caption_zh() -> str:
    rows = cost_table_rows_zh()
    lines = ["**每份 Live 报告费用粗估**（DeepSeek Flash + Tavily；历史 cheap 路径接近 $0）："]
    for r in rows:
        lines.append(f"- {r['场景']}：{r['约 USD/份']}")
    lines.append(
        "说明：模板中/英/双语双渲染≈$0（不重跑新闻与 MC）；逐条总结按批处理（约 20 条/次），无 Key 时退回抽取式。"
    )
    return "\n".join(lines)
