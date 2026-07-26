"""
显式七步流水线（按产品约定）：

1. 选择货币对
2. 评估所需要的信息
3. 抓取并存储有影响的语句（供赋权 / References / 数学分析）
4. 评估每条信息对货币对的影响
5. 对每条信息赋予权重
6. 数学分析（蒙特卡洛分档）
7. 输出规范格式报告
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from fx_report.market.fetch_data import MarketSnapshot, calibrate_unpriced_from_market, fetch_market
from fx_report.model.monte_carlo import MCResult, enforce_math_floor, run_mixture_monte_carlo
from fx_report.news.evidence import build_evidence_from_news
from fx_report.news.fetch import Headline, fetch_headlines_for_pair
from fx_report.news.llm import LLMConfig, resolve_llm_config
from fx_report.market.pair_drivers import DRIVER_CATALOG, describe_pair_factors, info_needs_for_drivers
from fx_report.market.pairs import PairSpec, get_pair, make_custom_pair
from fx_report.report.text import build_diagnostics, build_report_markdown
from fx_report.report.torchcast import (
    TorchcastReport,
    build_torchcast_report,
    export_torchcast,
    render_html,
)
from fx_report.model.weights import (
    EvidenceItem,
    ModelWeights,
    ScenarioSpec,
    apply_evidence_to_scenarios,
    default_weights,
    evidence_score,
    resolve_bucket_edges,
)

ClassifyMode = Literal["hybrid", "llm", "rules"]

# 驱动词表与信息需求见 pair_drivers.py（任意货币对共用）


@dataclass
class InfoNeed:
    id: str
    need: str
    why: str
    sources: str
    driver: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class StoredStatement:
    """步骤 3：落库的有影响语句，供赋权 / References / 数学分析。"""

    id: str
    statement: str
    source: str
    url: str
    provider: str
    published: str | None
    related_drivers: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WeightedEvidence:
    """步骤 4–5：影响评估 + 权重。"""

    evidence: EvidenceItem
    impact_note: str
    weight_contrib: float  # direction × strength × freshness × unpriced

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": asdict(self.evidence),
            "impact_note": self.impact_note,
            "weight_contrib": self.weight_contrib,
        }


@dataclass
class PipelineResult:
    pair: str
    stage_log: list[str]
    info_needs: list[InfoNeed]
    market: MarketSnapshot
    statements: list[StoredStatement]
    weighted: list[WeightedEvidence]
    score: float
    mu_shift: float
    sigma_extra: float
    scenarios: list[ScenarioSpec]
    edges: tuple[float, float, float, float]
    mc: MCResult
    probs: dict[str, float]
    weights: ModelWeights
    report_md: str
    report_html: str
    torchcast: TorchcastReport
    diagnostics: dict[str, Any]
    news_meta: dict[str, Any]
    horizon_label: str

    def save(self, out_dir: str | Path) -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        safe = self.pair.replace("/", "")
        paths: dict[str, Path] = {
            "report": out / f"{safe}_report.md",
            "diagnostics": out / f"{safe}_diagnostics.json",
            "statements": out / f"{safe}_statements.json",
            "info_needs": out / f"{safe}_info_needs.json",
            "pipeline": out / f"{safe}_pipeline.json",
        }
        paths["report"].write_text(self.report_md, encoding="utf-8")
        paths["diagnostics"].write_text(
            json.dumps(self.diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        paths["statements"].write_text(
            json.dumps([s.to_dict() for s in self.statements], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths["info_needs"].write_text(
            json.dumps([n.to_dict() for n in self.info_needs], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths["pipeline"].write_text(
            json.dumps(
                {
                    "pair": self.pair,
                    "stage_log": self.stage_log,
                    "score": self.score,
                    "probs": self.probs,
                    "horizon_label": self.horizon_label,
                    "weighted": [w.to_dict() for w in self.weighted],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        # Torchcast-style primary deliverable
        tc_paths = export_torchcast(self.torchcast, out, stem=safe)
        paths["html"] = tc_paths["html"]
        paths["pdf"] = tc_paths["pdf"]
        return paths


# ---------------------------------------------------------------------------
# 步骤函数
# ---------------------------------------------------------------------------


def step1_select_pair(
    pair: str,
    *,
    ticker: str | None = None,
    invert: bool = False,
) -> PairSpec:
    """1. 选择货币对（同时锁定该对的影响因子清单）"""
    if ticker:
        return make_custom_pair(pair, ticker, invert)
    return get_pair(pair)


def step2_assess_info_needs(spec: PairSpec) -> list[InfoNeed]:
    """
    2. 评估所需要的信息

    不同货币对影响因子不同 → 先列出本对要找什么，再进入抓取。
    目录内货币对用精调 drivers；任意自定义对用币种自动推断。
    """
    rows = info_needs_for_drivers(spec.default_drivers)
    needs = [
        InfoNeed(
            id=r["id"],
            need=r["need"],
            why=r["why"],
            sources=r["sources"],
            driver=r.get("driver", ""),
        )
        for r in rows
    ]
    return needs


def step3_collect_and_store_statements(
    spec: PairSpec,
    info_needs: list[InfoNeed],
    *,
    lookback_days: int = 60,
    max_items: int = 30,
    skip_news: bool = False,
    ai_research: bool = True,
    llm_cfg: LLMConfig | None = None,
) -> tuple[MarketSnapshot, list[StoredStatement], list[Headline], dict[str, Any]]:
    """
    3. 按信息需求抓取，并存储有影响的语句
    （行情 + 官方/RSS 头条 + 可选 AI 检索员；供赋权 / References / 数学分析）
    """
    meta: dict[str, Any] = {"ai_research": None}
    market = fetch_market(spec, lookback_days=lookback_days)
    statements: list[StoredStatement] = [
        StoredStatement(
            id="MKT-SPOT",
            statement=(
                f"{spec.pair} 现价 {market.spot:.5f}；"
                f"日波动 {market.sigma_daily:.4%}；年化波动 {market.sigma_annual:.2%}；"
                f"来源 {market.source}"
            ),
            source=market.source,
            url="",
            provider="market",
            published=market.asof,
            related_drivers=["spot_vol"],
            raw=market.to_dict(),
        )
    ]
    for note in market.notes:
        statements.append(
            StoredStatement(
                id=f"MKT-NOTE-{len(statements)}",
                statement=note,
                source=market.source,
                url="",
                provider="market",
                published=market.asof,
                related_drivers=["spot_vol"],
            )
        )

    headlines: list[Headline] = []
    if not skip_news:
        headlines = fetch_headlines_for_pair(spec, max_items=max_items)

        # AI 检索员：白名单投行页 + 搜索 API + LLM 抽取展望（像人工补搜）
        if ai_research:
            try:
                from fx_report.news.ai_research import run_ai_research

                ai = run_ai_research(
                    spec,
                    info_need_ids=[n.id for n in info_needs],
                    llm_cfg=llm_cfg,
                    max_headlines=12,
                )
                meta["ai_research"] = ai.meta
                # 去重后合并：AI 结果优先展示在前半，便于步骤4看到投行展望
                seen = {(h.title or "").strip().lower() for h in headlines}
                for h in ai.headlines:
                    key = (h.title or "").strip().lower()
                    if key and key not in seen:
                        headlines.insert(0, h)
                        seen.add(key)
            except Exception as e:
                meta["ai_research"] = {"enabled": True, "errors": [f"ai_research_failed:{e}"]}

        drivers = list(spec.default_drivers)
        for i, h in enumerate(headlines):
            # 粗匹配：语句关联到信息需求驱动
            blob = f"{h.title} {h.summary}".lower()
            related = [d for d in drivers if d.replace("_", " ") in blob or d in blob]
            if not related:
                # 保底挂上政策类需求
                related = ["fed", "rba"] if any(x in blob for x in ("fed", "rba", "fomc")) else []
            statements.append(
                StoredStatement(
                    id=f"STMT-{i+1:02d}",
                    statement=(h.title + ((" — " + h.summary[:240]) if h.summary else "")).strip(),
                    source=h.source,
                    url=h.url,
                    provider=h.provider,
                    published=h.published.isoformat() if h.published else None,
                    related_drivers=related or [n.driver for n in info_needs if n.driver][:2],
                    raw=h.to_dict(),
                )
            )
    return market, statements, headlines, meta


def step4_evaluate_impact(
    headlines: list[Headline],
    spec: PairSpec,
    market: MarketSnapshot,
    base_weights: ModelWeights,
    *,
    mode: ClassifyMode = "hybrid",
    max_items: int = 10,
    keep_templates: bool = False,
    llm_cfg: LLMConfig | None = None,
    fetch_fulltext: bool = True,
) -> tuple[list[EvidenceItem], dict[str, Any]]:
    """4. 评估每条信息对货币对的影响（方向 / 类别 / 强弱输入）"""
    suggested_up = calibrate_unpriced_from_market(market.ret_1d, market.ret_5d)
    cfg = llm_cfg
    use_mode = mode
    if use_mode in {"llm", "hybrid"} and cfg is None:
        cfg = resolve_llm_config()
    if use_mode in {"llm", "hybrid"} and cfg is None:
        use_mode = "rules"

    auto: list[EvidenceItem] = []
    meta: dict[str, Any] = {"mode": use_mode}
    if headlines:
        auto, meta = build_evidence_from_news(
            headlines,
            spec,
            mode=use_mode,
            max_items=max_items,
            unpriced_cap=suggested_up,
            llm_cfg=cfg,
            fetch_fulltext=fetch_fulltext,
        )
    if auto:
        evidence = auto + (list(base_weights.evidence) if keep_templates else [])
    else:
        evidence = list(base_weights.evidence)
        meta["fallback_templates"] = True

    for e in evidence:
        e.unpriced = min(e.unpriced, suggested_up)
    return evidence, meta


def step5_assign_weights(evidence: list[EvidenceItem]) -> list[WeightedEvidence]:
    """5. 对每条信息赋予权重（贡献分）"""
    out: list[WeightedEvidence] = []
    for e in evidence:
        contrib = e.direction * e.strength * e.freshness * e.unpriced
        if e.direction > 0:
            impact = f"推高 {e.category} 路径上尾"
        elif e.direction < 0:
            impact = f"压制 {e.category} 路径峰值"
        else:
            impact = "中性"
        note = f"{impact}｜label={e.strength_label or 'n/a'}｜contrib={contrib:+.3f}"
        out.append(WeightedEvidence(evidence=e, impact_note=note, weight_contrib=contrib))
    return out


def step6_math_analysis(
    market: MarketSnapshot,
    weights: ModelWeights,
    evidence: list[EvidenceItem],
) -> tuple[float, float, float, list[ScenarioSpec], tuple[float, float, float, float], MCResult, dict[str, float]]:
    """6. 数学分析：证据分 → 情景/漂移/波动 → 蒙特卡洛分档"""
    weights.evidence = evidence
    score = evidence_score(evidence)
    mu_shift = weights.score_to_mu_a * score
    sigma_extra = 1.0 + weights.score_to_sigma_b * abs(score)
    scenarios = apply_evidence_to_scenarios(
        weights.scenarios,
        score,
        logit_scale=weights.evidence_logit_scale,
        temperature=weights.scenario_temperature,
        max_shift=weights.max_scenario_shift,
    )
    edges = resolve_bucket_edges(weights, market.spot)
    mc = run_mixture_monte_carlo(
        spot=market.spot,
        sigma_daily_base=market.sigma_daily,
        scenarios=scenarios,
        trading_days=weights.trading_days,
        n_sims=weights.n_sims,
        seed=weights.seed,
        bucket_edges=edges,
        mu_annual_shift=mu_shift,
        sigma_mult_extra=sigma_extra,
    )
    probs = enforce_math_floor(mc.raw_probs, market.spot, edges)
    return score, mu_shift, sigma_extra, scenarios, edges, mc, probs


def step7_build_report(
    *,
    market: MarketSnapshot,
    weights: ModelWeights,
    scenarios: list[ScenarioSpec],
    mc: MCResult,
    probs: dict[str, float],
    score: float,
    mu_shift: float,
    sigma_extra: float,
    edges: tuple[float, float, float, float],
    info_needs: list[InfoNeed],
    statements: list[StoredStatement],
    weighted: list[WeightedEvidence],
    stage_log: list[str],
    headlines: list[Headline],
    news_meta: dict[str, Any],
) -> tuple[str, str, TorchcastReport, dict[str, Any], str]:
    """7. Torchcast 风格报告（HTML/PDF）+ Markdown 副本 + diagnostics"""
    start = date.today()
    end = start + timedelta(days=max(int(weights.trading_days * 1.4), 1))
    horizon = f"{start} 至 {end}"

    tc = build_torchcast_report(
        market,
        weights,
        scenarios,
        mc,
        probs,
        score=score,
        mu_shift=mu_shift,
        sigma_extra=sigma_extra,
        horizon_start=start,
        horizon_end=end,
        bucket_edges=edges,
    )
    report_html = render_html(tc)

    report = build_report_markdown(
        market,
        weights,
        scenarios,
        mc,
        probs,
        score=score,
        mu_shift=mu_shift,
        sigma_extra=sigma_extra,
        horizon_label=horizon,
        bucket_edges=edges,
    )

    # 前置流程说明 + References（Markdown 调试副本）
    needs_md = "\n".join(
        f"| {n.id} | {n.need} | {n.why} | {n.sources} |" for n in info_needs
    )
    refs_md = "\n".join(
        f"{i}. [{s.source}] {s.statement[:160]}"
        + (f" — {s.url}" if s.url else "")
        for i, s in enumerate(statements[:25], 1)
    )
    weights_md = "\n".join(
        f"| {w.evidence.id} | {w.evidence.strength_label} | {w.weight_contrib:+.3f} | {w.impact_note} |"
        for w in weighted
    )
    preface = f"""## 分析流程（固定七步）

1. 选择货币对 → **{market.pair}**
2. 评估所需信息 → {len(info_needs)} 项
3. 存储有影响语句 → {len(statements)} 条
4. 评估影响 → 证据 {len(weights.evidence)} 条
5. 赋予权重 → 见下表
6. 数学分析 → 蒙特卡洛 {mc.n_sims:,} 次
7. 输出本报告（Torchcast PDF / HTML 为主）

### 步骤2 · 信息需求

| ID | 需要什么 | 为何需要 | 来源设想 |
|----|----------|----------|----------|
{needs_md}

### 步骤5 · 权重贡献

| ID | 强弱 | 贡献分 | 影响说明 |
|----|------|--------|----------|
{weights_md}

---
"""
    appendix = f"""

---

## References（来自步骤3存储语句）

{refs_md if refs_md else "_（无存储语句）_"}

_生成时间 {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}_
"""
    full_report = preface + report + appendix

    diag = build_diagnostics(
        market, weights, scenarios, mc, probs, score, mu_shift, sigma_extra, edges
    )
    diag["stage_log"] = stage_log
    diag["info_needs"] = [n.to_dict() for n in info_needs]
    diag["statements"] = [s.to_dict() for s in statements]
    diag["weighted"] = [w.to_dict() for w in weighted]
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
    diag["torchcast_question"] = tc.question
    return full_report, report_html, tc, diag, horizon


def run_pipeline(
    pair: str = "USD/AUD",
    *,
    ticker: str | None = None,
    invert: bool = False,
    sims: int = 100_000,
    days: int = 66,
    seed: int = 42,
    lookback: int = 60,
    mode: ClassifyMode = "hybrid",
    max_news: int = 10,
    keep_templates: bool = False,
    no_news: bool = False,
    no_fulltext: bool = False,
    ai_research: bool = True,
    llm_cfg: LLMConfig | None = None,
    out_dir: str | Path | None = "output",
    verbose: bool = True,
) -> PipelineResult:
    """跑完整七步；可选写入 output/。"""
    log: list[str] = []

    def say(msg: str) -> None:
        log.append(msg)
        if verbose:
            print(msg)

    # 1
    say("【1/7】选择货币对")
    spec = step1_select_pair(pair, ticker=ticker, invert=invert)
    say(f"  → {spec.pair}｜{spec.description}")
    say(f"  → {describe_pair_factors(spec.base, spec.quote, spec.default_drivers)}")

    base = default_weights(spec)
    base.n_sims = sims
    base.trading_days = days
    base.seed = seed
    base.vol_lookback_days = lookback

    # 2
    say("【2/7】评估所需要的信息（因货币对而异）")
    info_needs = step2_assess_info_needs(spec)
    for n in info_needs:
        label = DRIVER_CATALOG[n.id].label if n.id in DRIVER_CATALOG else n.id
        say(f"  · [{label}] {n.need}")

    # 3
    say("【3/7】抓取并存储有影响的语句" + ("（含 AI 检索员）" if ai_research and not no_news else ""))
    market, statements, headlines, step3_meta = step3_collect_and_store_statements(
        spec,
        info_needs,
        lookback_days=lookback,
        max_items=30,
        skip_news=no_news,
        ai_research=ai_research,
        llm_cfg=llm_cfg,
    )
    say(f"  → 行情 {market.source} spot={market.spot:.5f}")
    say(f"  → 存储语句 {len(statements)} 条｜头条 {len(headlines)} 条")
    ai_meta = step3_meta.get("ai_research") or {}
    if ai_meta:
        say(
            f"  → AI 检索：白名单 {ai_meta.get('whitelist_ok', 0)}｜"
            f"命中 {ai_meta.get('search_hits', 0)}｜"
            f"产出 {ai_meta.get('headlines_out', 0)}｜"
            f"LLM={'on' if ai_meta.get('llm') else 'off'}"
        )

    # 4
    say("【4/7】评估每条信息对货币对的影响")
    evidence, news_meta = step4_evaluate_impact(
        headlines,
        spec,
        market,
        base,
        mode=mode,
        max_items=max_news,
        keep_templates=keep_templates,
        llm_cfg=llm_cfg,
        fetch_fulltext=not no_fulltext,
    )
    news_meta["ai_research"] = ai_meta
    say(f"  → 证据 {len(evidence)} 条｜mode={news_meta.get('mode')}")

    # 5
    say("【5/7】对每条信息赋予权重")
    weighted = step5_assign_weights(evidence)
    for w in weighted[:12]:
        say(
            f"  · {w.evidence.id} [{w.evidence.strength_label}] "
            f"contrib={w.weight_contrib:+.3f} | {w.evidence.title[:56]}"
        )

    # 6
    say("【6/7】数学分析（蒙特卡洛）")
    score, mu_shift, sigma_extra, scenarios, edges, mc, probs = step6_math_analysis(
        market, base, evidence
    )
    say(f"  → S={score:+.3f}  μ_shift={mu_shift:+.4f}  σ×={sigma_extra:.3f}")
    for k, v in probs.items():
        say(f"  · {k}: {v:.1%}")

    # 7
    say("【7/7】输出 Torchcast 格式报告（PDF / HTML）")
    report_md, report_html, torchcast, diagnostics, horizon = step7_build_report(
        market=market,
        weights=base,
        scenarios=scenarios,
        mc=mc,
        probs=probs,
        score=score,
        mu_shift=mu_shift,
        sigma_extra=sigma_extra,
        edges=edges,
        info_needs=info_needs,
        statements=statements,
        weighted=weighted,
        stage_log=log,
        headlines=headlines,
        news_meta=news_meta,
    )

    result = PipelineResult(
        pair=spec.pair,
        stage_log=log,
        info_needs=info_needs,
        market=market,
        statements=statements,
        weighted=weighted,
        score=score,
        mu_shift=mu_shift,
        sigma_extra=sigma_extra,
        scenarios=scenarios,
        edges=edges,
        mc=mc,
        probs=probs,
        weights=base,
        report_md=report_md,
        report_html=report_html,
        torchcast=torchcast,
        diagnostics=diagnostics,
        news_meta=news_meta,
        horizon_label=horizon,
    )
    if out_dir is not None:
        paths = result.save(out_dir)
        say(f"  → PDF  {paths.get('pdf')}")
        say(f"  → HTML {paths.get('html')}")
        say(f"  → MD   {paths.get('report')}")
    return result
