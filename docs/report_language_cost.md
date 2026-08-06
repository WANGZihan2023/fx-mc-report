# 报告语言、引用总结与费用粗估

## 报告语言（中 / 英）

- UI 侧栏 **报告语言** 与 **界面语言** 独立；默认 **中文**。
- PDF / HTML / Markdown 的章节标题、执行摘要、叙事、关注事项跟随报告语言。
- References 里的 **支撑引用** 保持来源原文（不编造、不伪翻译）；**总结** 为 1–2 句报告语言说明「来源说了什么 + 如何支撑 Higher/Lower/Context」。

## 逐条引用总结

| 路径 | 行为 |
|------|------|
| Live + 已配置 LLM | DeepSeek 等按批（约 20 条/次）写 `stance_summary` |
| Live / 无 Key | 抽取式模板（title / summary / support_quote），不编造事实 |
| 历史 cheap（默认） | 跳过 LLM，仅抽取式（≈$0）；侧栏可勾选强制 LLM |

实现：`fx_report/news/summarize.py` → `apply_stance_summaries`；展示：`evidence_refs` + Torchcast HTML/PDF。

## 每份 Live 报告费用粗估

计价假设：DeepSeek Flash 约 $0.14 / $0.28 per 1M tokens（in/out）；Tavily 约 $0.01/次 × ~10 轮。**非账单读数**，供规划。

详见 UI 侧栏「每份报告费用粗估」与 `fx_report/report/cost_estimate.py`。

典型量级（随实现微调，以 `cost_estimate.py` 为准）：

- 基线 Live（AI 检索 + 分类，无逐条总结）：约 **$0.12**/份
- 模板中文切换：**+$0**
- + 逐条总结（30 / 80 / 100 条）：额外约 **$0.002–0.007**（合计约 $0.12–0.12）
- 日更 2–3 份（基线 + 模板中文 + 80 条总结）：约 **$0.24–0.37**/日

历史 cheap 回放接近 **$0**（无 Tavily / 无 LLM 总结，除非勾选覆盖）。
