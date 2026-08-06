"""
Torchcast-style intelligence report: HTML + PDF.

Layout mirrors the reference PDF:
  cover (question, probability bars, upside/downside, executive summary)
  narrative sections with ↑/↓
  evidence base cards
  what-to-watch if/then cards
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

from fx_report.format_rate import format_rate
from fx_report.market.fetch_data import MarketSnapshot
from fx_report.model.monte_carlo import MCResult
from fx_report.model.weights import EvidenceItem, ModelWeights, ScenarioSpec
from fx_report.report.evidence_refs import (
    evidence_link_meta,
    evidence_stance_summary_meta,
    evidence_support_meta,
)
from fx_report.report.strings import (
    L,
    category_label,
    normalize_report_lang,
    pair_phrase,
    side_word,
)

# ---------------------------------------------------------------------------
# Colors (Torchcast palette)
# ---------------------------------------------------------------------------

GOLD = "#C4A35A"
GOLD_SOFT = "#E8D5A3"
GREEN = "#2F6B4F"
RED = "#9B3B3B"
BADGE = "#3D7A6A"
TEXT = "#1A1A1A"
MUTED = "#6B6B6B"
RULE = "#E5E5E5"
BOX_BG = "#FAFAF8"
CHIP_BG = "#F3E7C4"
LINK = "#2B5C9E"


CCY_NAME = {
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
}


@dataclass
class NarrativeSection:
    title: str
    direction: str  # "up" | "down" | "neutral"
    lede: str
    body: str  # may include [[C-1]] citation markers


@dataclass
class WatchItem:
    title: str
    lede: str
    upside_if: str
    upside_then: str
    downside_if: str
    downside_then: str


@dataclass
class TorchcastReport:
    pair: str
    question: str
    forecast_date: str
    n_evidence: int
    n_buckets: int
    probs: dict[str, float]
    top_bucket: str
    top_prob: float
    upside_bullets: list[str]
    downside_bullets: list[str]
    executive_summary: str
    narratives: list[NarrativeSection]
    higher_evidence: list[EvidenceItem]
    lower_evidence: list[EvidenceItem]
    context_evidence: list[EvidenceItem]
    watches: list[WatchItem]
    spot: float
    lang: str = "zh"
    disclaimer: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.lang = normalize_report_lang(self.lang)
        if not (self.disclaimer or "").strip():
            self.disclaimer = L("disclaimer", lang=self.lang)


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def _pair_english(pair: str) -> str:
    return pair_phrase(pair, lang="en")


def _question(pair: str, start: date, end: date, *, lang: str = "en") -> str:
    lang = normalize_report_lang(lang)
    phrase = pair_phrase(pair, lang=lang)
    if lang == "zh":
        return (
            f"{start.strftime('%Y年%m月%d日')}至{end.strftime('%Y年%m月%d日')}期间，"
            f"{phrase}的最高日高汇率将落在哪一档？"
        )
    return (
        f"What will be the highest daily high exchange rate of the "
        f"{phrase} between {start.strftime('%B %d, %Y')}, "
        f"and {end.strftime('%B %d, %Y')}?"
    )


def _chip_cite(text: str) -> str:
    """Replace [[ID]] markers with yellow citation chips in HTML."""

    def repl(m: re.Match[str]) -> str:
        return f'<span class="cite">{_esc(m.group(1))}</span>'

    return re.sub(r"\[\[([A-Za-z0-9\-]+)\]\]", repl, text)


def _has_cjk(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


def _evidence_display_title(e: EvidenceItem, *, lang: str = "en") -> str:
    lang = normalize_report_lang(lang)
    title = (e.title or "").strip()
    # Keep original Latin titles; for CJK titles in EN reports, fall back to category
    if title and not _has_cjk(title):
        return title
    if title and lang == "zh":
        return title
    base = category_label(e.category or "", lang=lang)
    side = side_word(int(e.direction or 0), lang=lang)
    return f"{base}（{side}）" if lang == "zh" else f"{base} ({side})"


def _evidence_bullets(
    items: Sequence[EvidenceItem], *, limit: int = 3, lang: str = "en"
) -> list[str]:
    out: list[str] = []
    for e in items[:limit]:
        title = _evidence_display_title(e, lang=lang)
        if len(title) > 140:
            title = title[:137] + "…"
        out.append(title)
    return out


def _build_narratives(
    market: MarketSnapshot,
    probs: dict[str, float],
    edges: Sequence[float],
    mc: MCResult,
    weights: ModelWeights,
    up: list[EvidenceItem],
    down: list[EvidenceItem],
    *,
    lang: str = "en",
) -> list[NarrativeSection]:
    lang = normalize_report_lang(lang)
    e = list(edges)
    floor = e[0] if e else market.spot
    labels = list(probs.keys())
    top = max(probs, key=probs.get)
    mid = labels[len(labels) // 2] if labels else top
    engine = getattr(mc, "peak_engine", getattr(weights, "peak_engine", "path_max"))
    if lang == "zh":
        if engine == "brownian_bridge":
            engine_phrase = (
                f"基于 {mc.n_sims:,} 次蒙特卡洛混合、布朗桥连续峰值引擎"
                f"（日端点间反射原理；不含复合泊松跳跃）给出定量锚点"
            )
        else:
            engine_phrase = (
                f"基于 {mc.n_sims:,} 次蒙特卡洛混合（含跳跃的 GBM；离散路径最大值）"
                f"给出定量锚点"
            )
        floor_body = (
            f"预测日 {market.pair} 现价约 {format_rate(market.spot)}，"
            f"数学上最高日高不可能低于 {format_rate(floor)} [[C-1]]。"
            f"因此完全低于现价地板的分档概率置 0.0%。{engine_phrase}。"
            f"所用日频历史为 {market.history_start} → {market.history_end}，"
            f"日波动约 {market.sigma_daily:.3%}（年化 {market.sigma_annual:.2%}）。"
            f"峰值引擎：<strong>{_esc(engine)}</strong>。"
        )
        up_ids = " ".join(f"[[{x.id}]]" for x in up[:4]) or "[[U-GEO]]"
        up_titles = "；".join(
            _evidence_display_title(x, lang=lang)[:80] for x in up[:3]
        ) or "避险 / 政策驱动"
        up_body = (
            f"最高概率分档为 <strong>{_esc(top)}</strong>（{_pct(probs[top])}），"
            f"上行证据包括 {_esc(up_titles)}。{up_ids} "
            f"若风险溢价在 {mc.trading_days} 个交易日内延续，报价更易测试更高技术 / 政策位。"
        )
        down_ids = " ".join(f"[[{x.id}]]" for x in down[:4]) or "[[D-CPI]]"
        down_titles = "；".join(
            _evidence_display_title(x, lang=lang)[:80] for x in down[:3]
        ) or "通胀降温 / 本地政策支撑"
        lower = next((k for k, v in probs.items() if k != top and v > 0), mid)
        down_body = (
            f"中间区间 <strong>{_esc(lower)}</strong>（{_pct(probs.get(lower, 0.0))}）"
            f"由下行 / 封顶证据锚定：{_esc(down_titles)}。{down_ids} "
            f"若风险溢价回吐，{market.pair} 峰值更可能落在该较低带 [[C-2]]。"
        )
        return [
            NarrativeSection(
                title="数学地板与定量基线",
                direction="up",
                lede=(
                    f"起点现价 {format_rate(market.spot)} 从数学上阻止最高日高"
                    f"落在 {format_rate(floor)} 以下。"
                ),
                body=floor_body,
            ),
            NarrativeSection(
                title="支撑上尾的驱动",
                direction="up",
                lede=f"上行证据倾向 {market.pair} 峰值靠近 {top}。",
                body=up_body,
            ),
            NarrativeSection(
                title="下行 / 封顶压力",
                direction="down",
                lede=f"对冲力量更常把峰值压在 {lower} 附近。",
                body=down_body,
            ),
        ]

    # English (legacy)
    if engine == "brownian_bridge":
        engine_phrase = (
            f"A {mc.n_sims:,}-run Monte Carlo mixture using a Brownian-bridge continuous "
            f"peak engine (reflection principle between daily log-GBM endpoints; "
            f"compound-Poisson jumps excluded) provides "
            f"the quantitative anchor for the distribution across higher intervals"
        )
    else:
        engine_phrase = (
            f"A {mc.n_sims:,}-run Monte Carlo mixture (GBM with jumps; discrete path max) "
            f"provides the quantitative anchor for the distribution across higher intervals"
        )

    floor_body = (
        f"With a starting {market.pair} spot rate of approximately {format_rate(market.spot)} "
        f"on the forecast date, the daily high exchange rate cannot mathematically resolve "
        f"below {format_rate(floor)} during the forecast horizon [[C-1]]. Consequently, probabilities "
        f"for buckets entirely below the spot floor are set at 0.0%. {engine_phrase}. "
        f"This modeling uses daily exchange-rate history "
        f"from {market.history_start} to {market.history_end}, applying an estimated daily "
        f"volatility of {market.sigma_daily:.3%} ({market.sigma_annual:.2%} annualized) "
        f"against the starting spot. Peak engine: <strong>{_esc(engine)}</strong>."
    )

    up_ids = " ".join(f"[[{x.id}]]" for x in up[:4]) or "[[U-GEO]]"
    up_titles = "; ".join(
        _evidence_display_title(x, lang=lang)[:80] for x in up[:3]
    ) or "safe-haven / policy drivers"
    up_body = (
        f"The highest-probability bucket is <strong>{_esc(top)}</strong> "
        f"({_pct(probs[top])}). This expectation is supported by upside evidence including "
        f"{_esc(up_titles)}. {up_ids} "
        f"These forces leave the quote vulnerable to testing higher technical / policy levels "
        f"if risk premia persist through the {mc.trading_days}-trading-day window."
    )

    down_ids = " ".join(f"[[{x.id}]]" for x in down[:4]) or "[[D-CPI]]"
    down_titles = "; ".join(
        _evidence_display_title(x, lang=lang)[:80] for x in down[:3]
    ) or "cooling inflation / local-policy support"
    lower = next((k for k, v in probs.items() if k != top and v > 0), mid)
    down_body = (
        f"The intermediate range <strong>{_esc(lower)}</strong> "
        f"({_pct(probs.get(lower, 0.0))}) is anchored by downside / capping evidence: "
        f"{_esc(down_titles)}. {down_ids} "
        f"If risk premia unwind, the peak {market.pair} path is more likely to remain "
        f"contained in this lower band [[C-2]]."
    )

    return [
        NarrativeSection(
            title="Mathematical Floor and Quantitative Baseline",
            direction="up",
            lede=(
                f"The starting spot rate of {format_rate(market.spot)} mathematically prevents "
                f"the highest daily high from resolving below {format_rate(floor)}."
            ),
            body=floor_body,
        ),
        NarrativeSection(
            title="Drivers Supporting the Upper Tail",
            direction="up",
            lede=f"Upside evidence favors a {market.pair} peak toward {top}.",
            body=up_body,
        ),
        NarrativeSection(
            title="Downside / Capping Pressure",
            direction="down",
            lede=f"Offsetting forces point to a peak more often contained near {lower}.",
            body=down_body,
        ),
    ]


def _default_watches(
    pair: str, top: str, lower: str, *, lang: str = "en"
) -> list[WatchItem]:
    lang = normalize_report_lang(lang)
    base, quote = pair.split("/")
    if lang == "zh":
        items = [
            WatchItem(
                title="地缘政治与能源价格",
                lede="风险溢价与油价冲击对避险货币需求杠杆很高。",
                upside_if="冲突升级且能源价格维持高位",
                upside_then=f"避险流动与贸易条件压力把峰值推向 {top}（上档概率上升）。",
                downside_if="停火 / 缓和令风险溢价回吐、油价降温",
                downside_then=f"{pair} 峰值更可能封顶在 {lower} 附近（上档概率下降）。",
            ),
            WatchItem(
                title=(
                    "美联储政策决议"
                    if "USD" in (base, quote)
                    else f"{base} / {quote} 政策决议"
                ),
                lede="下一次政策决议将厘清分歧的利率路径预期。",
                upside_if="反应函数偏鹰派得到确认",
                upside_then=f"利差重定价有利于 {pair} 峰值靠近 {top}。",
                downside_if="持稳 / 偏鸽基调在软数据后占主导",
                downside_then=f"近端美元（或高息币）动能回落，峰值更多集中在 {lower}。",
            ),
        ]
        if "AUD" in (base, quote) or "NZD" in (base, quote):
            items.append(
                WatchItem(
                    title="中国 / 商品需求",
                    lede="中国宏观与工业金属需求影响澳纽贸易条件。",
                    upside_if="刺激不及预期且铁矿石 / 金属偏软",
                    upside_then=f"商品货币走弱，抬高 {pair} 峰值进入更高分档的概率。",
                    downside_if="大规模财政 / 地产刺激提振商品需求",
                    downside_then=f"澳纽走强，{pair} 峰值更可能封顶在 {lower} 附近。",
                )
            )
        if "AUD" in (base, quote):
            items.append(
                WatchItem(
                    title="澳联储政策会议",
                    lede="国内政策决定缓冲澳元贬值的利差 carry。",
                    upside_if="增长偏软下澳联储按兵不动、carry 支撑减弱",
                    upside_then=f"澳元支撑消退，{pair} 更易冲入更高分档。",
                    downside_if="澳联储加息 / 维持明显紧缩",
                    downside_then=f"Carry 支撑有助于把 {pair} 峰值压在最高档以下。",
                )
            )
        return items[:4]

    items = [
        WatchItem(
            title="Geopolitics and Energy Prices",
            lede="Risk premia and oil shocks are high-leverage drivers of safe-haven currency demand.",
            upside_if="Escalation persists and energy prices remain elevated",
            upside_then=f"Safe-haven flows and terms-of-trade stress push the maximum toward {top} (upside for higher_buckets).",
            downside_if="A ceasefire / de-escalation unwinds the risk premium and oil cools",
            downside_then=f"The {pair} maximum is more likely capped near {lower} (downside for higher_buckets).",
        ),
        WatchItem(
            title=f"{'Federal Reserve' if 'USD' in (base, quote) else base + ' / ' + quote} Policy Decision",
            lede="The next policy decision resolves divergent rate-path expectations.",
            upside_if="The hawkish side of the reaction function is confirmed",
            upside_then=f"Rate differentials reprice in favor of a higher {pair} peak toward {top}.",
            downside_if="The hold / dovish tone dominates after soft data",
            downside_then=f"Near-term dollar (or high-yielder) strength fades, concentrating the peak near {lower}.",
        ),
    ]
    if "AUD" in (base, quote) or "NZD" in (base, quote):
        items.append(
            WatchItem(
                title="China / Commodity Demand",
                lede="Chinese macro and industrial metals demand influence AUD/NZD terms of trade.",
                upside_if="Stimulus disappoints and iron ore / metals stay soft",
                upside_then=f"Commodity FX weakens, raising odds the {pair} peak enters higher buckets.",
                downside_if="A large fiscal / property stimulus lifts commodity demand",
                downside_then=f"AUD/NZD firms and the {pair} maximum is more likely capped near {lower}.",
            )
        )
    if "RBA" in pair.upper() or "AUD" in (base, quote):
        items.append(
            WatchItem(
                title="Reserve Bank of Australia Policy Meeting",
                lede="Domestic policy dictates the yield carry that cushions AUD depreciation.",
                upside_if="The RBA holds amid softer growth, weakening carry support",
                upside_then=f"AUD support fades and {pair} is more likely to push into higher bands.",
                downside_if="The RBA hikes / stays firmly restrictive",
                downside_then=f"Carry support helps keep the {pair} maximum below the top bucket.",
            )
        )
    return items[:4]


def build_torchcast_report(
    market: MarketSnapshot,
    weights: ModelWeights,
    scenarios_adj: list[ScenarioSpec],
    mc: MCResult,
    probs: dict[str, float],
    *,
    score: float,
    mu_shift: float,
    sigma_extra: float,
    horizon_start: date | None = None,
    horizon_end: date | None = None,
    bucket_edges: Sequence[float] | None = None,
    bullish_currency: str | None = None,
    lang: str = "zh",
) -> TorchcastReport:
    lang = normalize_report_lang(lang)
    start = horizon_start or date.today()
    end = horizon_end or (start + timedelta(days=max(int(weights.trading_days * 1.4), 1)))
    edges = tuple(bucket_edges or weights.bucket_edges)
    top = max(probs, key=probs.get)
    top_p = probs[top]
    up = [e for e in weights.evidence if e.direction > 0]
    down = [e for e in weights.evidence if e.direction < 0]
    ctx = [e for e in weights.evidence if e.direction == 0]
    labels = list(probs.keys())
    lower = next((k for k, v in probs.items() if k != top and v > 0), labels[1] if len(labels) > 1 else top)
    phrase = pair_phrase(market.pair, lang=lang)
    p50 = format_rate(mc.percentiles.get("p50", 0))
    p90 = format_rate(mc.percentiles.get("p90", 0))
    p95 = format_rate(mc.percentiles.get("p95", 0))

    if lang == "zh":
        exec_sum = (
            f"{start.strftime('%Y年%m月%d日')}至{end.strftime('%Y年%m月%d日')}期间，"
            f"{phrase}的最高日高最可能落在 <strong>{_esc(top)}</strong>，"
            f"概率 {_pct(top_p)}。"
            f"起点现价 {format_rate(market.spot)} 从数学上阻止峰值落在地板档以下"
            f"（本窗口 {mc.trading_days} 个交易日）"
            f'<span class="cite">C-1</span>。'
            f"其余概率分布在中间档，由证据分 S={score:+.2f}"
            f"（μ 平移 {mu_shift:+.2%} 年化，σ ×{sigma_extra:.3f}）驱动。"
            f"峰值分位：P50={p50}，P90={p90}，P95={p95}。"
        )
    else:
        exec_sum = (
            f"The highest daily high exchange rate of {_pair_english(market.pair)} between "
            f"{start.strftime('%B %d, %Y')}, and {end.strftime('%B %d, %Y')}, is projected to "
            f"most likely peak in <strong>{_esc(top)}</strong>, carrying a {_pct(top_p)} probability. "
            f"Because the starting spot rate is {format_rate(market.spot)}, the exchange rate cannot "
            f"mathematically peak below the floor bucket during this {mc.trading_days}-trading-day "
            f"window <span class=\"cite\">C-1</span>. Remaining likelihood is split across intermediate "
            f"ranges, driven by counter-balancing forces captured in the evidence score "
            f"S={score:+.2f} (μ shift {mu_shift:+.2%} ann., σ ×{sigma_extra:.3f}). "
            f"Peak path percentiles: P50={p50}, P90={p90}, P95={p95}."
        )

    return TorchcastReport(
        pair=market.pair,
        question=_question(market.pair, start, end, lang=lang),
        forecast_date=start.isoformat(),
        n_evidence=len(weights.evidence),
        n_buckets=len(probs),
        probs=dict(probs),
        top_bucket=top,
        top_prob=top_p,
        upside_bullets=_evidence_bullets(up, lang=lang)
        or [L("thin_up", lang=lang)],
        downside_bullets=_evidence_bullets(down, lang=lang)
        or [L("thin_down", lang=lang)],
        executive_summary=exec_sum,
        narratives=_build_narratives(
            market, probs, edges, mc, weights, up, down, lang=lang
        ),
        higher_evidence=up,
        lower_evidence=down,
        context_evidence=ctx,
        watches=_default_watches(market.pair, top, lower, lang=lang),
        spot=market.spot,
        lang=lang,
        extra={
            "scenarios": [s.__dict__ for s in scenarios_adj],
            "source": market.source,
            "n_sims": mc.n_sims,
            "peak_engine": getattr(mc, "peak_engine", getattr(weights, "peak_engine", "path_max")),
            "bullish_currency": (bullish_currency or market.pair.split("/")[0]).upper(),
            "evidence_quality": None,  # filled by pipeline step7 when available
            "fallback_templates": False,
            "report_lang": lang,
        },
    )


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = f"""
@page {{
  size: letter;
  margin: 16mm 15mm 18mm 15mm;
  @bottom-center {{
    content: "FX Analyse · Page " counter(page) " of " counter(pages);
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 9pt;
    color: #9A9A9A;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  color: {TEXT};
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, "Times New Roman", serif;
  font-size: 10.5pt;
  line-height: 1.48;
}}
.kicker {{
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 8.5pt;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  color: {MUTED};
  margin: 0 0 12px 0;
}}
.tags {{ margin: 0 0 14px 0; }}
.tag {{
  display: inline-block;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 8pt;
  color: {MUTED};
  border: 1px solid {RULE};
  border-radius: 999px;
  padding: 3px 11px;
  margin-right: 6px;
  background: #fff;
}}
h1.question {{
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  font-size: 19pt;
  font-weight: 700;
  line-height: 1.28;
  margin: 0 0 12px 0;
  color: {TEXT};
}}
.meta {{
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 9.5pt;
  color: {MUTED};
  margin-bottom: 18px;
}}
.meta span + span::before {{ content: " · "; color: #C0C0C0; }}
.panel {{
  border: 1px solid {RULE};
  border-radius: 12px;
  background: {BOX_BG};
  padding: 16px 18px;
  margin-bottom: 16px;
}}
.section-label {{
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 8pt;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: {MUTED};
  margin: 0 0 12px 0;
}}
.prob-grid {{
  display: flex;
  gap: 18px;
  align-items: flex-start;
}}
.prob-left {{ width: 34%; flex: 0 0 34%; }}
.prob-right {{ flex: 1; }}
.most-label {{
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 8pt;
  letter-spacing: 0.14em;
  color: {GOLD};
  text-transform: uppercase;
  margin: 0 0 6px 0;
}}
.most-bucket {{
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 24pt;
  font-weight: 700;
  margin: 0;
  letter-spacing: -0.02em;
}}
.most-prob {{
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 11pt;
  color: {MUTED};
  margin: 4px 0 0 0;
}}
.bar-row {{
  display: grid;
  grid-template-columns: 1.55in 1fr 0.55in;
  gap: 8px;
  align-items: center;
  margin: 6px 0;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 8.5pt;
}}
.bar-label {{ color: {MUTED}; }}
.bar-track {{
  height: 9px;
  background: #ECECEC;
  border-radius: 999px;
  overflow: hidden;
}}
.bar-track > span {{
  display: block;
  height: 100%;
  border-radius: 999px;
  background: #C8C8C8;
}}
.bar-track > span.top {{ background: {GOLD}; }}
.bar-pct {{ text-align: right; color: {TEXT}; font-variant-numeric: tabular-nums; }}
.split {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0;
  margin-top: 14px;
  border-top: 1px solid {RULE};
  padding-top: 12px;
}}
.col {{ padding-right: 14px; }}
.col + .col {{
  padding-right: 0;
  padding-left: 14px;
  border-left: 1px solid {RULE};
}}
.col h3 {{
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 9.5pt;
  margin: 0 0 8px 0;
  letter-spacing: 0.06em;
}}
.up {{ color: {GREEN}; }}
.down {{ color: {RED}; }}
.col ul {{
  margin: 0;
  padding-left: 16px;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 9pt;
  color: {TEXT};
}}
.col li {{ margin: 0 0 7px 0; }}
.exec {{
  border: 1px solid {RULE};
  border-radius: 12px;
  padding: 14px 16px 14px 18px;
  border-left: 4px solid {GOLD};
  background: #fff;
  margin-bottom: 8px;
}}
.exec p {{ margin: 0; }}
.cite {{
  display: inline-block;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 7.5pt;
  background: {CHIP_BG};
  color: #5A4A20;
  border-radius: 4px;
  padding: 1px 5px;
  margin: 0 1px;
  vertical-align: baseline;
  white-space: nowrap;
}}
.narrative {{ margin: 20px 0; page-break-inside: avoid; }}
.narrative h2 {{
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  font-size: 14.5pt;
  margin: 0 0 5px 0;
}}
.arrow {{
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 11pt;
  margin-left: 6px;
}}
.lede {{
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 10pt;
  color: {MUTED};
  margin: 0 0 9px 0;
}}
.narrative p {{ margin: 0; }}
.evidence-box {{
  border: 1px solid {RULE};
  border-radius: 12px;
  padding: 16px 18px;
  margin: 20px 0;
  background: {BOX_BG};
}}
.ev-group {{
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 11pt;
  font-weight: 700;
  margin: 10px 0 12px 0;
}}
.ev-item {{
  margin: 0 0 13px 0;
  page-break-inside: avoid;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 9pt;
  line-height: 1.42;
}}
.badge {{
  display: inline-block;
  font-size: 7.5pt;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: #fff;
  background: {BADGE};
  border-radius: 4px;
  padding: 2px 6px;
  margin-right: 4px;
}}
.ev-links a {{
  color: {LINK};
  text-decoration: none;
  font-size: 8pt;
  word-break: break-all;
}}
.ev-quote {{
  margin: 4px 0 2px 0;
  padding: 4px 8px;
  border-left: 3px solid {GOLD_SOFT};
  color: {MUTED};
  font-size: 8.5pt;
  font-style: italic;
  line-height: 1.4;
}}
.ev-quote-label {{
  font-style: normal;
  font-size: 7.5pt;
  color: #8A7A50;
  margin-right: 4px;
  letter-spacing: 0.02em;
}}
.ev-stance {{
  margin: 3px 0 2px 0;
  color: {TEXT};
  font-size: 8.5pt;
  font-style: normal;
  line-height: 1.4;
}}
.ev-stance-label {{
  font-size: 7.5pt;
  color: #5A6A7A;
  margin-right: 4px;
  letter-spacing: 0.02em;
  font-weight: 600;
}}
.ev-link-warn {{
  color: #8A5A3A;
  font-size: 7.5pt;
}}
.watch-title-page {{
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 22pt;
  letter-spacing: 0.24em;
  text-transform: uppercase;
  color: {MUTED};
  text-align: center;
  margin-top: 240px;
  page-break-before: always;
}}
.watch {{
  page-break-inside: avoid;
  margin: 0 0 20px 0;
}}
.watch .num {{
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 18pt;
  font-weight: 700;
  color: {TEXT};
  margin: 0 0 4px 0;
}}
.watch h3 {{
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  font-size: 13pt;
  margin: 0 0 4px 0;
}}
.ifthen {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-top: 8px;
}}
.ifbox {{
  border: 1px solid {RULE};
  border-radius: 8px;
  padding: 10px 12px;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 9pt;
}}
.ifbox .lab {{
  font-size: 8pt;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 4px;
}}
.disclaimer {{
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 8.5pt;
  color: {MUTED};
  margin-top: 22px;
  border-top: 1px solid {RULE};
  padding-top: 10px;
}}
"""


def _oos_meta_span(calib_oos: Any) -> str:
    """Short holdout/calib trust line for HTML/PDF meta strip (empty if none)."""
    if not isinstance(calib_oos, dict) or not calib_oos:
        return ""
    hit = calib_oos.get("holdout_hit_rate")
    brier = calib_oos.get("holdout_brier")
    skill = calib_oos.get("holdout_skill_brier")
    n = calib_oos.get("holdout_n")
    if hit is None and brier is None:
        return ""
    hit_s = _pct(float(hit)) if hit is not None else "—"
    try:
        brier_s = f"{float(brier):.3f}" if brier is not None else "—"
    except (TypeError, ValueError):
        brier_s = "—"
    try:
        skill_s = f"{float(skill):.3f}" if skill is not None and skill == skill else "—"
    except (TypeError, ValueError):
        skill_s = "—"
    try:
        n_s = str(int(n)) if n is not None and n == n else "—"
    except (TypeError, ValueError):
        n_s = "—"
    return (
        f'<span>OOS holdout hit {hit_s} · Brier {brier_s} · Skill {skill_s} · n={n_s}</span>'
    )


def render_html(report: TorchcastReport) -> str:
    lang = normalize_report_lang(getattr(report, "lang", None) or "zh")
    # probability bars
    max_p = max(report.probs.values()) if report.probs else 1.0
    bars = []
    for label, p in report.probs.items():
        width = 0 if max_p <= 0 else int(round(100 * p / max_p))
        cls = "top" if label == report.top_bucket else ""
        bars.append(
            f'<div class="bar-row"><div class="bar-label">{_esc(label)}</div>'
            f'<div class="bar-track"><span class="{cls}" style="width:{width}%"></span></div>'
            f'<div class="bar-pct">{_pct(p)}</div></div>'
        )
    up_li = "".join(f"<li>{_esc(b)}</li>" for b in report.upside_bullets)
    down_li = "".join(f"<li>{_esc(b)}</li>" for b in report.downside_bullets)

    narr_html = []
    for n in report.narratives:
        arrow = "↑" if n.direction == "up" else ("↓" if n.direction == "down" else "·")
        arrow_cls = "up" if n.direction == "up" else ("down" if n.direction == "down" else "")
        narr_html.append(
            f'<section class="narrative">'
            f'<h2>{_esc(n.title)} <span class="arrow {arrow_cls}">{arrow}</span></h2>'
            f'<p class="lede">{_esc(n.lede)}</p>'
            f"<p>{_chip_cite(n.body)}</p>"
            f"</section>"
        )

    def ev_block(title: str, items: list[EvidenceItem]) -> str:
        if not items:
            return ""
        rows = []
        for e in items:
            lab = (e.strength_label or "MODERATE").upper()
            sq = evidence_support_meta(e, lang=lang)
            stance = evidence_stance_summary_meta(e, lang=lang)
            quote = sq.get("quote") or ""
            link_meta = evidence_link_meta(e, lang=lang)
            url = link_meta.get("url") or ""
            stance_html = ""
            if stance.get("text"):
                stance_html = (
                    f'<div class="ev-stance">'
                    f'<span class="ev-stance-label">{_esc(str(stance.get("label") or ""))}</span>'
                    f"{_esc(str(stance['text']))}"
                    f"</div>"
                )
            quote_html = ""
            if quote:
                qlab = _esc(str(sq.get("label") or L("support", lang=lang)))
                q_open, q_close = ('"', '"') if lang == "en" else ("「", "」")
                quote_html = (
                    f'<div class="ev-quote">'
                    f'<span class="ev-quote-label">{qlab}</span>'
                    f"{q_open}{_esc(quote)}{q_close}"
                    f"</div>"
                )
            warn_lab = link_meta.get("label") or ""
            if url:
                link = (
                    f'<div class="ev-links"><a href="{_esc(url)}">{_esc(url)}</a></div>'
                )
            elif warn_lab:
                link = (
                    f'<div class="ev-links ev-link-warn">'
                    f"{_esc(str(warn_lab))}"
                    f"</div>"
                )
            else:
                link = ""
            rows.append(
                f'<div class="ev-item">'
                f'<span class="cite">{_esc(e.id)}</span> '
                f'<span class="badge">{_esc(lab)}</span> '
                f"{_esc(_evidence_display_title(e, lang=lang))}"
                f"{stance_html}"
                f"{quote_html}"
                f"{link}"
                f"</div>"
            )
        return f'<div class="ev-group">{_esc(title)}</div>' + "".join(rows)

    evidence_html = (
        '<div class="evidence-box">'
        f'<div class="section-label">{_esc(L("sec_evidence", lang=lang))}</div>'
        + ev_block(L("ev_higher", lang=lang), report.higher_evidence)
        + ev_block(L("ev_lower", lang=lang), report.lower_evidence)
        + ev_block(L("ev_context", lang=lang), report.context_evidence)
        + "</div>"
    )

    watch_html = [
        f'<div class="watch-title-page">{_esc(L("watch_title", lang=lang))}</div>'
    ]
    for i, w in enumerate(report.watches, 1):
        watch_html.append(
            f'<section class="watch">'
            f'<div class="num">{i:02d}</div>'
            f"<h3>{_esc(w.title)}</h3>"
            f'<p class="lede">{_esc(w.lede)}</p>'
            f'<div class="ifthen">'
            f'<div class="ifbox"><div class="lab up">{_esc(L("if_up", lang=lang))}</div>'
            f"<div><strong>{_esc(L('if_word', lang=lang))}</strong> {_esc(w.upside_if)}</div>"
            f"<div><strong>{_esc(L('then_word', lang=lang))}</strong> {_esc(w.upside_then)}</div></div>"
            f'<div class="ifbox" style="margin-left:10px"><div class="lab down">{_esc(L("if_down", lang=lang))}</div>'
            f"<div><strong>{_esc(L('if_word', lang=lang))}</strong> {_esc(w.downside_if)}</div>"
            f"<div><strong>{_esc(L('then_word', lang=lang))}</strong> {_esc(w.downside_then)}</div></div>"
            f"</div></section>"
        )

    html_lang = "zh-CN" if lang == "zh" else "en"
    return f"""<!DOCTYPE html>
<html lang="{html_lang}">
<head>
<meta charset="utf-8"/>
<title>{_esc(report.question)}</title>
<style>{CSS}</style>
</head>
<body>
  <p class="kicker">{_esc(L("kicker", lang=lang))}</p>
  <div class="tags">
    <span class="tag">{_esc(L("tag_ordered", lang=lang))}</span>
    <span class="tag">{_esc(L("tag_buckets", lang=lang, n=report.n_buckets))}</span>
  </div>
  <h1 class="question">{_esc(report.question)}</h1>
  <div class="meta">
    <span>{_esc(L("meta_forecast", lang=lang, d=report.forecast_date))}</span>
    <span>{_esc(L("meta_evidence", lang=lang, n=report.n_evidence))}</span>
    <span>{_esc(L("meta_bullish", lang=lang, c=str(report.extra.get("bullish_currency") or report.pair.split("/")[0])))}</span>
    <span>{_esc(L("meta_quote", lang=lang, p=report.pair))}</span>
    <span>{_esc(L("meta_peak", lang=lang, e=str(report.extra.get("peak_engine") or "path_max")))}</span>
    <span>{_esc(L("meta_ev_quality", lang=lang, q=str(report.extra.get("evidence_quality") or "n/a")))}</span>
    <span>{_esc(L("meta_clusters", lang=lang, c=str(report.extra.get("cluster_n") if report.extra.get("cluster_n") is not None else "n/a"), r=str(report.extra.get("evidence_raw_n") if report.extra.get("evidence_raw_n") is not None else report.n_evidence)))}{" · dedup" if report.extra.get("cluster_dedup_applied") else ""}{(" · ⚠" + str(len(report.extra.get("cluster_warnings") or [])) + " " + L("warns", lang=lang)) if (report.extra.get("cluster_warnings") or []) else ""}</span>
    {_oos_meta_span(report.extra.get("calib_oos"))}
  </div>

  <div class="panel">
    <div class="section-label">{_esc(L("sec_prob", lang=lang))}</div>
    <div class="prob-grid">
      <div class="prob-left">
        <p class="most-label">{_esc(L("most_likely", lang=lang))}</p>
        <p class="most-bucket">{_esc(report.top_bucket)}</p>
        <p class="most-prob">{_esc(L("probability", lang=lang, p=_pct(report.top_prob)))}</p>
      </div>
      <div class="prob-right">{"".join(bars)}</div>
    </div>
    <div class="split">
      <div class="col">
        <h3 class="up">{_esc(L("upside", lang=lang))}</h3>
        <ul>{up_li}</ul>
      </div>
      <div class="col">
        <h3 class="down">{_esc(L("downside", lang=lang))}</h3>
        <ul>{down_li}</ul>
      </div>
    </div>
  </div>

  <div class="exec">
    <div class="section-label">{_esc(L("sec_exec", lang=lang))}</div>
    <p>{report.executive_summary}</p>
  </div>

  {"".join(narr_html)}
  {evidence_html}
  {"".join(watch_html)}
  <p class="disclaimer">{_esc(report.disclaimer or L("disclaimer", lang=lang))}</p>
</body>
</html>
"""


def write_html(report: TorchcastReport, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(render_html(report), encoding="utf-8")
    return path


def _ensure_homebrew_libs() -> None:
    """Make Homebrew pango/gobject visible to WeasyPrint on macOS."""
    import os
    from pathlib import Path

    candidates = [
        Path("/opt/homebrew/lib"),
        Path("/usr/local/lib"),
    ]
    existing = [str(p) for p in candidates if p.is_dir()]
    if not existing:
        return
    key = "DYLD_FALLBACK_LIBRARY_PATH"
    cur = os.environ.get(key, "")
    parts = [p for p in cur.split(":") if p]
    for p in existing:
        if p not in parts:
            parts.insert(0, p)
    os.environ[key] = ":".join(parts)
    # Also help pkg-config based lookups when present
    for brew in (Path("/opt/homebrew"), Path("/usr/local")):
        pc = brew / "lib" / "pkgconfig"
        if pc.is_dir():
            pkg = os.environ.get("PKG_CONFIG_PATH", "")
            if str(pc) not in pkg.split(":"):
                os.environ["PKG_CONFIG_PATH"] = f"{pc}:{pkg}" if pkg else str(pc)


def write_pdf(report: TorchcastReport, path: str | Path) -> Path:
    """
    Render PDF.

    Default engine: weasyprint (Torchcast HTML/CSS → PDF).
    Fallback: reportlab if WeasyPrint / system libs unavailable.
    Override: FX_PDF_ENGINE=reportlab|weasyprint|playwright
    """
    path = Path(path)
    import os

    engine = os.environ.get("FX_PDF_ENGINE", "weasyprint").lower().strip()
    html_str = render_html(report)
    errors: list[str] = []

    if engine in {"weasyprint", "auto", ""}:
        try:
            _ensure_homebrew_libs()
            from weasyprint import HTML  # type: ignore

            HTML(string=html_str, base_url=str(path.parent.resolve())).write_pdf(str(path))
            return path
        except Exception as e:
            errors.append(f"weasyprint:{e}")
            if engine == "weasyprint":
                # still fall through unless user insists — keep report usable
                pass

    if engine == "playwright":
        try:
            return _write_pdf_playwright(html_str, path)
        except Exception as e:
            errors.append(f"playwright:{e}")

    try:
        return _write_pdf_reportlab(report, path)
    except Exception as e:
        errors.append(f"reportlab:{e}")
        raise RuntimeError("PDF export failed: " + " | ".join(errors)) from e


def _write_pdf_playwright(html_str: str, path: Path) -> Path:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html_str, wait_until="load")
        page.pdf(
            path=str(path),
            format="Letter",
            print_background=True,
            margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"},
        )
        browser.close()
    return path


def _write_pdf_reportlab(report: TorchcastReport, path: Path) -> Path:
    """ReportLab fallback approximating the Torchcast cover + body."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable,
        KeepTogether,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    gold = colors.HexColor(GOLD)
    green = colors.HexColor(GREEN)
    red = colors.HexColor(RED)
    muted = colors.HexColor(MUTED)
    rule = colors.HexColor(RULE)

    def cite_rl(s: str) -> str:
        # Convert [[ID]] or existing cite spans to ReportLab font tags
        s = re.sub(r"<span class=\"cite\">([^<]+)</span>", r"<font backColor='#F3E7C4' size='7'> \1 </font>", s)
        s = re.sub(r"\[\[([A-Za-z0-9\-]+)\]\]", r"<font backColor='#F3E7C4' size='7'> \1 </font>", s)
        s = s.replace("<strong>", "<b>").replace("</strong>", "</b>")
        return s

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Kicker", fontName="Helvetica", fontSize=8, textColor=muted, spaceAfter=6, leading=10))
    styles.add(ParagraphStyle(name="Q", fontName="Times-Bold", fontSize=15, leading=19, spaceAfter=8, textColor=colors.HexColor(TEXT)))
    styles.add(ParagraphStyle(name="Meta", fontName="Helvetica", fontSize=9, textColor=muted, spaceAfter=10))
    styles.add(ParagraphStyle(name="SecLabel", fontName="Helvetica", fontSize=8, textColor=muted, spaceBefore=4, spaceAfter=6))
    styles.add(ParagraphStyle(name="MostLabel", fontName="Helvetica", fontSize=8, textColor=gold, spaceAfter=2))
    styles.add(ParagraphStyle(name="MostBucket", fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=2))
    styles.add(ParagraphStyle(name="MostProb", fontName="Helvetica", fontSize=10, textColor=muted, spaceAfter=4))
    styles.add(ParagraphStyle(name="Body", fontName="Times-Roman", fontSize=9.5, leading=13, alignment=TA_JUSTIFY, spaceAfter=6))
    styles.add(ParagraphStyle(name="Lede", fontName="Helvetica", fontSize=9, textColor=muted, leading=12, spaceAfter=4))
    styles.add(ParagraphStyle(name="H2", fontName="Times-Bold", fontSize=12, leading=15, spaceBefore=10, spaceAfter=3))
    styles.add(ParagraphStyle(name="TcBullet", fontName="Helvetica", fontSize=8.5, leading=11, leftIndent=8, spaceAfter=3))
    styles.add(ParagraphStyle(name="TcEv", fontName="Helvetica", fontSize=8.5, leading=11, spaceAfter=6))
    styles.add(ParagraphStyle(name="TcWatchH", fontName="Times-Bold", fontSize=12, leading=14, spaceAfter=3))
    styles.add(ParagraphStyle(name="TcWatchNum", fontName="Helvetica-Bold", fontSize=16, spaceBefore=8, spaceAfter=2))
    styles.add(ParagraphStyle(name="TcDisc", fontName="Helvetica", fontSize=8, textColor=muted, spaceBefore=12))
    styles.add(ParagraphStyle(name="TcCenterBig", fontName="Helvetica", fontSize=18, textColor=muted, alignment=TA_CENTER, spaceBefore=180, spaceAfter=40))
    styles.add(ParagraphStyle(name="TcBarLab", fontName="Helvetica", fontSize=8, textColor=muted))
    styles.add(ParagraphStyle(name="TcBarPct", fontName="Helvetica", fontSize=8, alignment=TA_RIGHT))
    styles.add(ParagraphStyle(name="TcUpH", fontName="Helvetica-Bold", fontSize=9, textColor=green, spaceAfter=4))
    styles.add(ParagraphStyle(name="TcDownH", fontName="Helvetica-Bold", fontSize=9, textColor=red, spaceAfter=4))

    story: list[Any] = []
    lang = normalize_report_lang(getattr(report, "lang", None) or "zh")
    story.append(Paragraph(_esc(L("kicker", lang=lang)).upper(), styles["Kicker"]))
    story.append(
        Paragraph(
            f"{_esc(L('tag_ordered', lang=lang))} &nbsp;&nbsp; "
            f"{_esc(L('tag_buckets', lang=lang, n=report.n_buckets))}",
            styles["Meta"],
        )
    )
    story.append(Paragraph(_esc(report.question), styles["Q"]))
    bullish_meta = str(
        report.extra.get("bullish_currency") or report.pair.split("/")[0]
    )
    peak_meta = str(report.extra.get("peak_engine") or "path_max")
    eq_meta = str(report.extra.get("evidence_quality") or "n/a")
    cluster_n = report.extra.get("cluster_n")
    raw_n = report.extra.get("evidence_raw_n", report.n_evidence)
    dedup_bit = " · dedup" if report.extra.get("cluster_dedup_applied") else ""
    warn_list = list(report.extra.get("cluster_warnings") or [])
    warn_bit = (
        f" · ⚠{len(warn_list)} {L('warns', lang=lang)}"
        if warn_list
        else ""
    )
    cluster_meta = (
        f"{L('meta_clusters', lang=lang, c=cluster_n, r=raw_n)}{dedup_bit}{warn_bit}"
        if cluster_n is not None
        else ""
    )
    story.append(
        Paragraph(
            f"{_esc(L('meta_forecast', lang=lang, d=report.forecast_date))} &nbsp;·&nbsp; "
            f"{_esc(L('meta_evidence', lang=lang, n=report.n_evidence))} &nbsp;·&nbsp; "
            f"{_esc(L('meta_bullish', lang=lang, c=bullish_meta))} &nbsp;·&nbsp; "
            f"{_esc(L('meta_quote', lang=lang, p=report.pair))} &nbsp;·&nbsp; "
            f"{_esc(L('meta_peak', lang=lang, e=peak_meta))} &nbsp;·&nbsp; "
            f"{_esc(L('meta_ev_quality', lang=lang, q=eq_meta))}"
            + (f" &nbsp;·&nbsp; {_esc(cluster_meta)}" if cluster_meta else ""),
            styles["Meta"],
        )
    )

    # Probability panel
    max_p = max(report.probs.values()) if report.probs else 1.0
    bar_rows = []
    for label, p in report.probs.items():
        w = 0.01 if max_p <= 0 else max(0.01, 2.2 * (p / max_p))
        fill = gold if label == report.top_bucket else colors.HexColor("#D0D0D0")
        bar = Table([[""]], colWidths=[w * inch], rowHeights=[7])
        bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), fill), ("ROUNDEDCORNERS", [3, 3, 3, 3])]))
        bar_rows.append(
            [
                Paragraph(_esc(label), styles["TcBarLab"]),
                bar,
                Paragraph(_pct(p), styles["TcBarPct"]),
            ]
        )
    bars_tbl = Table(bar_rows, colWidths=[1.5 * inch, 2.4 * inch, 0.7 * inch])
    bars_tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))

    left = [
        Paragraph(_esc(L("most_likely", lang=lang)).upper(), styles["MostLabel"]),
        Paragraph(_esc(report.top_bucket), styles["MostBucket"]),
        Paragraph(
            _esc(L("probability", lang=lang, p=_pct(report.top_prob))),
            styles["MostProb"],
        ),
    ]
    left_t = Table([[left]], colWidths=[2.2 * inch])
    top_grid = Table([[left_t, bars_tbl]], colWidths=[2.3 * inch, 4.6 * inch])
    top_grid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))

    up_paras = [Paragraph(_esc(L("upside", lang=lang)), styles["TcUpH"])] + [
        Paragraph(f"• {_esc(b)}", styles["TcBullet"]) for b in report.upside_bullets
    ]
    down_paras = [Paragraph(_esc(L("downside", lang=lang)), styles["TcDownH"])] + [
        Paragraph(f"• {_esc(b)}", styles["TcBullet"]) for b in report.downside_bullets
    ]
    split = Table([[up_paras, down_paras]], colWidths=[3.4 * inch, 3.4 * inch])
    split.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBEFORE", (1, 0), (1, 0), 0.5, rule),
                ("LEFTPADDING", (1, 0), (1, 0), 10),
                ("RIGHTPADDING", (0, 0), (0, 0), 10),
            ]
        )
    )

    panel = Table([[top_grid], [Spacer(1, 6)], [HRFlowable(width="100%", thickness=0.5, color=rule)], [Spacer(1, 4)], [split]], colWidths=[6.9 * inch])
    panel.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.7, rule),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(BOX_BG)),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("ROUNDEDCORNERS", [6, 6, 6, 6]),
            ]
        )
    )
    story.append(panel)
    story.append(Spacer(1, 10))

    exec_tbl = Table(
        [
            [Paragraph(_esc(L("sec_exec", lang=lang)).upper(), styles["SecLabel"])],
            [Paragraph(cite_rl(report.executive_summary), styles["Body"])],
        ],
        colWidths=[6.9 * inch],
    )
    exec_tbl.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.7, rule),
                ("LINEBEFORE", (0, 0), (0, -1), 3, gold),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(exec_tbl)

    for n in report.narratives:
        arrow = "↑" if n.direction == "up" else "↓"
        story.append(Paragraph(f"{_esc(n.title)} {arrow}", styles["H2"]))
        story.append(Paragraph(_esc(n.lede), styles["Lede"]))
        story.append(Paragraph(cite_rl(n.body), styles["Body"]))

    # Evidence / References (id · claim/quote · source link)
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(_esc(L("sec_evidence", lang=lang)).upper(), styles["SecLabel"])
    )

    def add_ev_group(title: str, items: list[EvidenceItem]) -> None:
        if not items:
            return
        story.append(Paragraph(_esc(title), styles["TcWatchH"]))
        for e in items:
            lab = (e.strength_label or "MODERATE").upper()
            body = (
                f"<font backColor='#F3E7C4' size='7'> { _esc(e.id) } </font> "
                f"<font backColor='#3D7A6A' color='white' size='7'> {lab} </font> "
                f"{_esc(_evidence_display_title(e, lang=lang))}"
            )
            stance = evidence_stance_summary_meta(e, lang=lang)
            if stance.get("text"):
                body += (
                    f"<br/><font color='#1A1A1A' size='8'>"
                    f"<b>{_esc(str(stance.get('label') or ''))}</b> "
                    f"{_esc(str(stance['text']))}"
                    f"</font>"
                )
            sq = evidence_support_meta(e, lang=lang)
            quote = sq.get("quote") or ""
            if quote:
                qlab = _esc(str(sq.get("label") or L("support", lang=lang)))
                q_open, q_close = ('"', '"') if lang == "en" else ("「", "」")
                body += (
                    f"<br/><font color='#6B6B6B' size='8'><i>"
                    f"{qlab} {q_open}{_esc(quote)}{q_close}"
                    f"</i></font>"
                )
            link_meta = evidence_link_meta(e, lang=lang)
            url = link_meta.get("url") or ""
            if url:
                body += f"<br/><font color='#2B5C9E' size='7'><u>{_esc(url)}</u></font>"
            elif link_meta.get("label"):
                body += (
                    f"<br/><font color='#8A5A3A' size='7'>"
                    f"{_esc(str(link_meta['label']))}"
                    f"</font>"
                )
            story.append(Paragraph(body, styles["TcEv"]))

    add_ev_group(L("ev_higher", lang=lang), report.higher_evidence)
    add_ev_group(L("ev_lower", lang=lang), report.lower_evidence)
    add_ev_group(L("ev_context", lang=lang), report.context_evidence)

    # What to watch
    story.append(
        Paragraph(_esc(L("watch_title", lang=lang)).upper(), styles["TcCenterBig"])
    )
    for i, w in enumerate(report.watches, 1):
        block = [
            Paragraph(f"{i:02d}", styles["TcWatchNum"]),
            Paragraph(_esc(w.title), styles["TcWatchH"]),
            Paragraph(_esc(w.lede), styles["Lede"]),
        ]
        if_tbl = Table(
            [
                [
                    Paragraph(
                        f"<font color='#2F6B4F'><b>{_esc(L('if_up', lang=lang))}</b></font><br/>"
                        f"<b>{_esc(L('if_word', lang=lang))}</b> {_esc(w.upside_if)}<br/>"
                        f"<b>{_esc(L('then_word', lang=lang))}</b> {_esc(w.upside_then)}",
                        styles["TcEv"],
                    ),
                    Paragraph(
                        f"<font color='#9B3B3B'><b>{_esc(L('if_down', lang=lang))}</b></font><br/>"
                        f"<b>{_esc(L('if_word', lang=lang))}</b> {_esc(w.downside_if)}<br/>"
                        f"<b>{_esc(L('then_word', lang=lang))}</b> {_esc(w.downside_then)}",
                        styles["TcEv"],
                    ),
                ]
            ],
            colWidths=[3.35 * inch, 3.35 * inch],
        )
        if_tbl.setStyle(
            TableStyle(
                [
                    ("BOX", (0, 0), (0, 0), 0.5, rule),
                    ("BOX", (1, 0), (1, 0), 0.5, rule),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (1, 0), (1, 0), 8),
                ]
            )
        )
        block.append(if_tbl)
        story.append(KeepTogether(block))

    story.append(
        Paragraph(
            _esc(report.disclaimer or L("disclaimer", lang=lang)),
            styles["TcDisc"],
        )
    )

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(muted)
        canvas.drawCentredString(letter[0] / 2, 0.55 * inch, f"FX Analyse · Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.75 * inch,
        title=report.question[:120],
        author="FX Analyse",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return path


def export_torchcast(
    report: TorchcastReport,
    out_dir: str | Path,
    *,
    stem: str,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    html_path = write_html(report, out / f"{stem}_fx_analyse.html")
    pdf_path = write_pdf(report, out / f"{stem}_fx_analyse.pdf")
    return {"html": html_path, "pdf": pdf_path}
