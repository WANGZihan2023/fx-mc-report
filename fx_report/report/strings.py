"""Bilingual (zh / en) string templates for Torchcast / Markdown reports.

UI language (`fx_report.ui.i18n`) is separate; this module drives **report body**
labels, section titles, and narrative templates. Evidence source quotes stay in
their original language; stance summaries follow ``report_lang``.
"""

from __future__ import annotations

from typing import Any  # EvidenceItem duck-typed in format_impact_note

LANG_ZH = "zh"
LANG_EN = "en"
LANG_BOTH = "both"
DEFAULT_REPORT_LANG = LANG_ZH
# UI / pipeline mode: zh | en | both (default bilingual — one MC, two renders)
DEFAULT_REPORT_MODE = LANG_BOTH
REPORT_LANGS = (LANG_ZH, LANG_EN)
REPORT_MODES = (LANG_BOTH, LANG_ZH, LANG_EN)


def normalize_report_lang(value: object | None) -> str:
    """Normalize to a single render language ``zh`` or ``en``.

    ``both`` / bilingual aliases map to ``zh`` (primary) for single-lang helpers.
    """
    raw = str(value or "").strip().lower()
    if raw in ("en", "en-us", "en-gb", "english"):
        return LANG_EN
    if raw in ("zh", "zh-cn", "zh-hans", "cn", "chinese", "中文"):
        return LANG_ZH
    if raw in ("both", "bilingual", "zh+en", "en+zh", "zh_en", "中英", "中英双语", "双语"):
        return LANG_ZH
    return DEFAULT_REPORT_LANG


def normalize_report_mode(value: object | None) -> str:
    """Normalize report language *mode*: ``zh`` | ``en`` | ``both``."""
    raw = str(value or "").strip().lower()
    if raw in ("both", "bilingual", "zh+en", "en+zh", "zh_en", "中英", "中英双语", "双语"):
        return LANG_BOTH
    if raw in ("en", "en-us", "en-gb", "english"):
        return LANG_EN
    if raw in ("zh", "zh-cn", "zh-hans", "cn", "chinese", "中文"):
        return LANG_ZH
    return DEFAULT_REPORT_MODE


def report_langs_for_mode(mode: object | None) -> list[str]:
    """Languages to render for a mode (news/MC run once; templates twice if both)."""
    m = normalize_report_mode(mode)
    if m == LANG_BOTH:
        return [LANG_ZH, LANG_EN]
    return [m]


def report_lang_suffix(lang: object | None) -> str:
    """Filename token: ``_zh`` or ``_en``."""
    return f"_{normalize_report_lang(lang)}"


CCY_NAME: dict[str, dict[str, str]] = {
    LANG_EN: {
        "USD": "US Dollar",
        "AUD": "Australian Dollar",
        "EUR": "Euro",
        "GBP": "British Pound",
        "JPY": "Japanese Yen",
        "CNH": "Chinese Yuan (offshore)",
        "CNY": "Chinese Yuan",
        "CAD": "Canadian Dollar",
        "NZD": "New Zealand Dollar",
        "CHF": "Swiss Franc",
    },
    LANG_ZH: {
        "USD": "美元",
        "AUD": "澳元",
        "EUR": "欧元",
        "GBP": "英镑",
        "JPY": "日元",
        "CNH": "离岸人民币",
        "CNY": "人民币",
        "CAD": "加元",
        "NZD": "纽元",
        "CHF": "瑞郎",
    },
}

CATEGORY: dict[str, dict[str, str]] = {
    LANG_EN: {
        "geopolitics": "Geopolitical / safe-haven pressure on the pair",
        "oil": "Oil / energy shock affecting terms of trade and risk",
        "fed": "Federal Reserve policy / hike-path pricing",
        "ecb": "ECB policy path",
        "rba": "RBA cash-rate / carry support",
        "rbnz": "RBNZ policy path",
        "boj": "BOJ policy path",
        "boe": "Bank of England policy path",
        "boc": "Bank of Canada policy path",
        "snb": "SNB policy path",
        "pboc": "PBOC / CNH policy signals",
        "cpi": "Inflation print vs consensus",
        "china_iron": "Iron ore / China metals demand",
        "china_growth": "China growth / stimulus impulse",
        "growth": "Growth / labour-market data",
        "yields": "Yield differentials",
        "positioning": "Speculative positioning / sentiment",
        "other": "Other macro / market evidence",
    },
    LANG_ZH: {
        "geopolitics": "地缘 / 避险压力",
        "oil": "油价 / 能源冲击与贸易条件",
        "fed": "美联储政策 / 加息路径定价",
        "ecb": "欧洲央行政策路径",
        "rba": "澳联储利率 / 利差支撑",
        "rbnz": "新西兰联储政策路径",
        "boj": "日本央行政策路径",
        "boe": "英格兰银行政策路径",
        "boc": "加拿大央行政策路径",
        "snb": "瑞士央行政策路径",
        "pboc": "央行 / 离岸人民币政策信号",
        "cpi": "通胀数据相对预期",
        "china_iron": "铁矿石 / 中国金属需求",
        "china_growth": "中国增长 / 刺激脉冲",
        "growth": "增长 / 就业数据",
        "yields": "利差",
        "positioning": "投机仓位 / 情绪",
        "other": "其他宏观 / 市场证据",
    },
}

SIDE_LABEL = {
    LANG_EN: {"upside": "upside", "downside": "downside", "context": "context"},
    LANG_ZH: {"upside": "上行", "downside": "下行", "context": "背景"},
}

# UI / section chrome
LABELS: dict[str, dict[str, str]] = {
    LANG_EN: {
        "kicker": "FX ANALYSE · Intelligence Report",
        "tag_ordered": "Ordered",
        "tag_buckets": "{n} Buckets",
        "meta_forecast": "Forecast date {d}",
        "meta_evidence": "{n} evidence items",
        "meta_bullish": "Bullish currency {c}",
        "meta_quote": "Analysis quote {p}",
        "meta_peak": "Peak engine {e}",
        "meta_ev_quality": "Evidence {q}",
        "meta_clusters": "Clusters {c} / raw {r}",
        "sec_prob": "Probability Distribution",
        "most_likely": "Most Likely Range",
        "probability": "{p} probability",
        "upside": "↑ UPSIDE",
        "downside": "↓ DOWNSIDE",
        "sec_exec": "Executive Summary",
        "sec_evidence": "Evidence Base / References",
        "ev_higher": "Higher",
        "ev_lower": "Lower",
        "ev_context": "Context",
        "support": "Support quote",
        "support_weak": "Support quote (weak)",
        "stance_summary": "Summary",
        "link_dead": "Link may be dead",
        "link_fragile": "Link may be unstable (Google News redirect)",
        "watch_title": "What to Watch",
        "if_up": "If → Then (upside)",
        "if_down": "If → Then (downside)",
        "if_word": "If",
        "then_word": "Then",
        "disclaimer": (
            "FX Analyse forecasts are probabilistic. Use as one input among many "
            "— not as investment advice."
        ),
        "thin_up": "Upside evidence thin — check news / templates.",
        "thin_down": "Downside evidence thin — check news / templates.",
        "md_title": "# FX ANALYSE · Intelligence Report",
        "md_question": (
            "**Question:** Within {horizon}, where will the **highest daily high** "
            "of **{pair}** land?"
        ),
        "md_prob": "## Probability distribution",
        "md_most": "**Most likely bucket: `{top}` ({p})**",
        "md_anchor": "## Market anchors",
        "md_up": "## Upside drivers (raise {pair} peak)",
        "md_down": "## Downside drivers (cap {pair} peak)",
        "md_exec": "## Executive summary",
        "md_none": "_(none)_",
        "md_disclaimer": "*Model output only — not investment advice.*",
        "md_generated": "**Generated:** {d}",
        "md_sims_line": (
            "**Sims:** {n}｜**Trading days:** {days}｜**Seed:** {seed}｜**Peak engine:** {peak}"
        ),
        "md_market_src": "**Market source:** {s}",
        "md_bucket_edges": "**Bucket edges:** {e}",
        "md_col_range": "Range",
        "md_col_prob": "Probability",
        "md_col_field": "Field",
        "md_col_value": "Value",
        "md_field_pair": "Pair",
        "md_field_spot": "Spot (analysis quote)",
        "md_field_raw": "Provider raw",
        "md_field_source": "Source",
        "md_field_sigma_d": "Daily σ",
        "md_field_sigma_a": "Annual σ",
        "md_field_lookback": "Lookback",
        "md_field_ret1": "1D return",
        "md_field_ret5": "5D return",
        "md_field_ret20": "20D return",
        "md_field_vol2060": "20D/60D ann. vol",
        "md_field_tickers": "History / spot ticker",
        "md_field_basis": "CNH−CNY basis",
        "md_proxy": "(proxy)",
        "md_notes": "**Notes:** {n}",
        "md_notes_empty": "—",
        "md_exec_body": (
            "Start **{pair} ≈ {spot}**. {days} trading days, **{n_sims}** Monte Carlo "
            "mixture (peak `{peak}`), evidence score **S={score}** (μ shift {mu} ann., "
            "σ ×{sigma})."
        ),
        "md_exec_most": (
            "Most likely: **`{top}` ({p})**. Peak percentiles P50={p50}, P90={p90}, P95={p95}."
        ),
        "md_math_floor": (
            "Math floor: ranges strictly below spot are zeroed then renormalized."
        ),
        "md_scen": "## Scenario weights (calibrated)",
        "md_scen_cols": (
            "| Scenario | Weight | Drift | σ mult | Jumps | Narrative |"
        ),
        "md_raw_mc": "## Raw MC frequencies",
        "md_col_freq": "Frequency",
        "md_rubric": "## Strength rubric",
        "md_watch1": (
            "1. **Core central bank / data for the pair** — surprise → re-score and re-run."
        ),
        "md_watch2": (
            "2. **Risk assets / safe haven** — systemic shock lifts escalation; "
            "easing lifts de-escalation."
        ),
        "md_watch3": (
            "3. **Commodities / China demand** (if relevant) — adjust U-CN / oil evidence."
        ),
        "md_watch4": (
            "4. **Already priced** — large spot jump → lower unpriced to avoid double-count."
        ),
        "evid_contrib": "contrib",
        "evid_scoring": "scoring",
        "scen_escalation": (
            "Risk-on / safe-haven or one-sided shock → thicker {pair} upper tail"
        ),
        "scen_baseline": "Range-bound / sticky mid → {pair} peak often in mid buckets",
        "scen_deescalation": "De-escalation / headwinds fade → {pair} peak capped",
        "impact_up": "Lifts upper tail via {cat}",
        "impact_down": "Caps peak via {cat}",
        "impact_neutral": "Neutral",
        "impact_unclassified": "Unclassified — excluded from main score",
        "impact_prior": "[prior]",
        "impact_dup": "[cluster downweight {cid}]",
        "impact_rep": "[cluster rep {cid} n={n}]",
        "preface_title": "## Analysis pipeline (fixed 7 steps)",
        "preface_bullish": "Bullish: **{c}**｜Analysis quote: **{p}**",
        "preface_1": "1. Select pair → **{pair}**",
        "preface_2": "2. Assess information needs → {n} items",
        "preface_3": "3. Store influential statements → {n} rows",
        "preface_4": "4. Score impact → {n} evidence items",
        "preface_5": "5. Assign weights → see table below",
        "preface_6": "6. Math analysis → Monte Carlo {n} runs",
        "preface_7": "7. Emit this report (Torchcast PDF / HTML primary)",
        "preface_needs": "### Step 2 · Information needs",
        "preface_needs_cols": "| ID | Need | Why | Sources |",
        "preface_weights": "### Step 5 · Weight contributions",
        "preface_weights_cols": "| ID | Strength | Contrib | Impact note |",
        "refs_heading": "## References / Evidence base (id · Summary · Support quote · URL)",
        "refs_empty": "_(none)_",
        "refs_note": (
            "Each row prefers a stance summary + verbatim support quote "
            "(from summary/snippet; never invented). Dead links are unlinked and marked."
        ),
        "refs_generated": "_Generated {t}_",
        "warns": "warns",
        "preview_zh": "Chinese preview",
        "preview_en": "English preview",
        "report_expander": "Full report (FX Analyse)",
        "dl_pdf": "Download PDF ({label})",
        "dl_html": "Download HTML ({label})",
        "dl_md": "Download Markdown ({label})",
        "lang_zh": "Chinese",
        "lang_en": "English",
        "report_for_lang": "**{label} report**",
    },
    LANG_ZH: {
        "kicker": "FX ANALYSE · 情报报告",
        "tag_ordered": "有序分档",
        "tag_buckets": "{n} 档",
        "meta_forecast": "预测日 {d}",
        "meta_evidence": "{n} 条证据",
        "meta_bullish": "看涨货币 {c}",
        "meta_quote": "分析报价 {p}",
        "meta_peak": "峰值引擎 {e}",
        "meta_ev_quality": "证据质量 {q}",
        "meta_clusters": "聚类 {c} / 原始 {r}",
        "sec_prob": "概率分布",
        "most_likely": "最可能区间",
        "probability": "概率 {p}",
        "upside": "↑ 上行",
        "downside": "↓ 下行",
        "sec_exec": "执行摘要",
        "sec_evidence": "证据库 / References",
        "ev_higher": "上行 Higher",
        "ev_lower": "下行 Lower",
        "ev_context": "背景 Context",
        "support": "支撑引用",
        "support_weak": "支撑引用（弱）",
        "stance_summary": "总结",
        "link_dead": "链接可能失效",
        "link_fragile": "链接可能不稳定（Google News 跳转）",
        "watch_title": "关注事项",
        "if_up": "若→则（上行）",
        "if_down": "若→则（下行）",
        "if_word": "若",
        "then_word": "则",
        "disclaimer": "FX Analyse 预测为概率输出，仅供参考，不构成投资建议。",
        "thin_up": "上行证据偏少 — 请检查新闻 / 模板。",
        "thin_down": "下行证据偏少 — 请检查新闻 / 模板。",
        "md_title": "# FX ANALYSE · 情报报告（多货币对引擎）",
        "md_question": (
            "**问题：** {horizon} 内，**{pair}** 的**最高日高**将落在哪一档？"
        ),
        "md_prob": "## 概率分布",
        "md_most": "**最可能区间：`{top}`（{p}）**",
        "md_anchor": "## 行情锚点",
        "md_up": "## 上行驱动（推高 {pair} 峰值）",
        "md_down": "## 下行驱动（压制 {pair} 峰值）",
        "md_exec": "## 执行摘要",
        "md_none": "_（无）_",
        "md_disclaimer": "*概率模型输出，不构成投资建议。*",
        "md_generated": "**预测生成：** {d}",
        "md_sims_line": (
            "**模拟次数：** {n}｜**交易日：** {days}｜**种子：** {seed}｜**峰值引擎：** {peak}"
        ),
        "md_market_src": "**行情来源：** {s}",
        "md_bucket_edges": "**分档边界：** {e}",
        "md_col_range": "区间",
        "md_col_prob": "概率",
        "md_col_field": "字段",
        "md_col_value": "值",
        "md_field_pair": "货币对",
        "md_field_spot": "现价（分析口径）",
        "md_field_raw": "源端原始报价",
        "md_field_source": "行情来源",
        "md_field_sigma_d": "日波动 σ_d",
        "md_field_sigma_a": "年化 σ",
        "md_field_lookback": "回看",
        "md_field_ret1": "近1日涨跌",
        "md_field_ret5": "近5日涨跌",
        "md_field_ret20": "近20日涨跌",
        "md_field_vol2060": "20D/60D 年化波动",
        "md_field_tickers": "历史代码 / 现价代码",
        "md_field_basis": "CNH−CNY 价差",
        "md_proxy": "（代理）",
        "md_notes": "**数据说明：** {n}",
        "md_notes_empty": "无额外备注",
        "md_exec_body": (
            "起点 **{pair} ≈ {spot}**。{days} 个交易日、**{n_sims}** 次情景混合蒙特卡洛"
            "（峰值引擎 `{peak}`），证据分 **S={score}** 校准权重与参数"
            "（μ 平移 {mu} 年化，σ ×{sigma}）。"
        ),
        "md_exec_most": (
            "最可能：**`{top}`（{p}）**。峰值分位 P50={p50}，P90={p90}，P95={p95}。"
        ),
        "md_math_floor": "数学地板：严格低于起点的最高价区间归零后归一化。",
        "md_scen": "## 情景权重（校准后）",
        "md_scen_cols": "| 情景 | 权重 | 漂移 | 波动倍数 | 跳跃 | 叙事 |",
        "md_raw_mc": "## 原始 MC 频率",
        "md_col_freq": "频率",
        "md_rubric": "## 信息强弱判定规则",
        "md_watch1": (
            "1. **该货币对核心央行/数据** — 决议或重磅意外 → 改 surprise/scope 并重跑。"
        ),
        "md_watch2": (
            "2. **风险资产与避险** — 系统性冲击抬 escalation；缓和抬 deescalation。"
        ),
        "md_watch3": (
            "3. **商品/中国需求**（若相关）— 改 U-CN / 油价证据方向与未定价。"
        ),
        "md_watch4": (
            "4. **已定价程度** — 即期已大跳则下调 unpriced，避免双计。"
        ),
        "evid_contrib": "贡献",
        "evid_scoring": "计分",
        "scen_escalation": "风险升高 / 避险或单边冲击 → {pair} 上尾加厚",
        "scen_baseline": "中性胶着 → {pair} 峰值多落在中档",
        "scen_deescalation": "缓和 / 逆风消退 → {pair} 峰值受压",
        "impact_up": "推高 {cat} 路径上尾",
        "impact_down": "压制 {cat} 路径峰值",
        "impact_neutral": "中性",
        "impact_unclassified": "未分类｜不计入主分",
        "impact_prior": "[先验]",
        "impact_dup": "[簇内降权 {cid}]",
        "impact_rep": "[簇代表 {cid} n={n}]",
        "preface_title": "## 分析流程（固定七步）",
        "preface_bullish": "看涨：**{c}**｜分析报价：**{p}**",
        "preface_1": "1. 选择货币对 → **{pair}**",
        "preface_2": "2. 评估所需信息 → {n} 项",
        "preface_3": "3. 存储有影响语句 → {n} 条",
        "preface_4": "4. 评估影响 → 证据 {n} 条",
        "preface_5": "5. 赋予权重 → 见下表",
        "preface_6": "6. 数学分析 → 蒙特卡洛 {n} 次",
        "preface_7": "7. 输出本报告（Torchcast PDF / HTML 为主）",
        "preface_needs": "### 步骤2 · 信息需求",
        "preface_needs_cols": "| ID | 需要什么 | 为何需要 | 来源设想 |",
        "preface_weights": "### 步骤5 · 权重贡献",
        "preface_weights_cols": "| ID | 强弱 | 贡献分 | 影响说明 |",
        "refs_heading": "## References / 证据库（id · 总结 · 支撑引用 · 来源链接）",
        "refs_empty": "_（无存储语句）_",
        "refs_note": (
            "每条尽量含「总结」与「支撑引用」摘录（来自 summary/snippet，非编造）。"
            "失效链接已去掉超链并标注「链接可能失效」。"
        ),
        "refs_generated": "_生成时间 {t}_",
        "warns": "告警",
        "preview_zh": "中文预览",
        "preview_en": "English preview",
        "report_expander": "完整报告（FX Analyse 格式）",
        "dl_pdf": "下载 PDF（{label}）",
        "dl_html": "下载 HTML（{label}）",
        "dl_md": "下载 Markdown（{label}）",
        "lang_zh": "中文",
        "lang_en": "English",
        "report_for_lang": "**{label}报告**",
    },
}


SCENARIO_NARRATIVE_KEYS = {
    "escalation": "scen_escalation",
    "baseline": "scen_baseline",
    "deescalation": "scen_deescalation",
}


def scenario_narrative(name: str, pair: str, *, lang: str | None = None) -> str:
    """Localize default three-regime scenario narratives by scenario name."""
    key = SCENARIO_NARRATIVE_KEYS.get((name or "").strip().lower())
    if key:
        return L(key, lang=lang, pair=pair)
    return ""


def format_impact_note(
    e: Any,
    contrib: float,
    *,
    lang: str | None = None,
) -> str:
    """Language-aware impact note for Markdown weight tables."""
    lang = normalize_report_lang(lang)
    cat = getattr(e, "category", None) or "other"
    if (cat or "").lower() == "unclassified":
        impact = L("impact_unclassified", lang=lang)
    else:
        direction = int(getattr(e, "direction", 0) or 0)
        if direction > 0:
            impact = L("impact_up", lang=lang, cat=cat)
        elif direction < 0:
            impact = L("impact_down", lang=lang, cat=cat)
        else:
            impact = L("impact_neutral", lang=lang)
        if getattr(e, "is_prior", False):
            impact = f"{L('impact_prior', lang=lang)} {impact}"
        role = getattr(e, "cluster_role", None) or ""
        cid = getattr(e, "cluster_id", None) or ""
        if role == "dup" and cid:
            impact = f"{L('impact_dup', lang=lang, cid=cid)} {impact}"
        elif cid and int(getattr(e, "cluster_size", 0) or 0) > 1 and role == "rep":
            impact = f"{L('impact_rep', lang=lang, cid=cid, n=int(e.cluster_size))} {impact}"
    label = getattr(e, "strength_label", None) or "n/a"
    note = f"{impact}｜label={label}｜contrib={contrib:+.3f}"
    sid = getattr(e, "statement_id", None) or ""
    if sid:
        note += f"｜{sid}"
    cid = getattr(e, "cluster_id", None) or ""
    if cid:
        note += f"｜{cid}/{getattr(e, 'cluster_role', None) or 'n/a'}"
    return note


def L(key: str, *, lang: str | None = None, **kwargs: Any) -> str:
    lang = normalize_report_lang(lang)
    table = LABELS.get(lang) or LABELS[LANG_ZH]
    s = table.get(key) or LABELS[LANG_EN].get(key) or key
    if kwargs:
        try:
            return s.format(**kwargs)
        except (KeyError, ValueError):
            return s
    return s


def pair_phrase(pair: str, *, lang: str | None = None) -> str:
    lang = normalize_report_lang(lang)
    parts = (pair or "").split("/")
    if len(parts) != 2:
        return pair
    names = CCY_NAME.get(lang) or CCY_NAME[LANG_EN]
    b = names.get(parts[0], parts[0])
    q = names.get(parts[1], parts[1])
    if lang == LANG_ZH:
        return f"{b}兑{q}（{pair}）"
    return f"{b} to {q} ({pair})"


def category_label(category: str, *, lang: str | None = None) -> str:
    lang = normalize_report_lang(lang)
    table = CATEGORY.get(lang) or CATEGORY[LANG_EN]
    return table.get(category, category or ("证据" if lang == LANG_ZH else "Evidence"))


def side_word(direction: int, *, lang: str | None = None) -> str:
    lang = normalize_report_lang(lang)
    table = SIDE_LABEL.get(lang) or SIDE_LABEL[LANG_EN]
    if direction > 0:
        return table["upside"]
    if direction < 0:
        return table["downside"]
    return table["context"]
