"""Bilingual (zh / en) string templates for Torchcast / Markdown reports.

UI language (`fx_report.ui.i18n`) is separate; this module drives **report body**
labels, section titles, and narrative templates. Evidence source quotes stay in
their original language; stance summaries follow ``report_lang``.
"""

from __future__ import annotations

from typing import Any

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
    },
}


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
