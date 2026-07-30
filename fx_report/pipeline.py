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
from fx_report.news.cluster import (
    ClusterMethod,
    assign_event_clusters,
    propagate_cluster_to_statements,
)
from fx_report.news.drift import DEFAULT_SOFT_ADAPT
from fx_report.news.evidence import build_evidence_from_news
from fx_report.news.fetch import (
    Headline,
    fetch_headlines_for_pair,
    fetch_historical_headlines_for_pair,
)
from fx_report.news.classify import MIN_PAIR_RELEVANCE, match_drivers_from_text, pair_relevance
from fx_report.news.llm import LLMConfig, resolve_llm_config
from fx_report.market.pair_drivers import DRIVER_CATALOG, describe_pair_factors, info_needs_for_drivers
from fx_report.market.pairs import PairSpec, get_pair, make_custom_pair, resolve_pair_for_bullish
from fx_report.report.text import build_diagnostics, build_report_markdown
from fx_report.ui.ux_helpers import (
    format_cheap_historical_caption,
    refs_statement_cap,
    step3_pool_size,
)
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
    evidence_item_contrib,
    evidence_score,
    resolve_bucket_edges,
)

ClassifyMode = Literal["hybrid", "llm", "rules"]
TemplatePolicy = Literal["off", "prior_only", "fallback_warn"]
HumanReviewMode = Literal["off", "pause", "auto_skip"]

# Prior templates: mark + downweight so they cannot pass as news-driven.
_PRIOR_STRENGTH_MULT = 0.5
_PRIOR_UNPRICED_MULT = 0.5

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
    cluster_id: str = ""  # linked event cluster after step4

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


@dataclass
class PipelineCheckpoint:
    """
    Phase A 产物：抓取+分类+聚类完成，赋权/MC/报告之前可暂停等人审。
    可序列化进 Streamlit session_state，刷新后尽量恢复。
    """

    pair: str
    bullish_currency: str
    stage_log: list[str]
    info_needs: list[InfoNeed]
    market: MarketSnapshot
    statements: list[StoredStatement]
    headlines: list[Headline]
    evidence: list[EvidenceItem]
    news_meta: dict[str, Any]
    weights: ModelWeights
    pending_reviews: list[Any]  # PendingReview（避免循环 import 用 Any）
    variance_reduction: str = "none"
    as_of_date: str | None = None
    display_pair: str = ""
    ticker: str | None = None
    invert: bool = False
    cal_source: str = "default"

    def to_session_dict(self) -> dict[str, Any]:
        from fx_report.model.human_review import PendingReview

        return {
            "pair": self.pair,
            "bullish_currency": self.bullish_currency,
            "stage_log": list(self.stage_log),
            "info_needs": [n.to_dict() for n in self.info_needs],
            "market": self.market.to_dict(),
            "statements": [s.to_dict() for s in self.statements],
            "headlines": [h.to_dict() for h in self.headlines],
            "evidence": [asdict(e) for e in self.evidence],
            "news_meta": dict(self.news_meta),
            "weights": self.weights.to_dict(),
            "pending_reviews": [
                p.to_dict() if hasattr(p, "to_dict") else dict(p)
                for p in self.pending_reviews
            ],
            "variance_reduction": self.variance_reduction,
            "as_of_date": self.as_of_date,
            "display_pair": self.display_pair or self.pair,
            "ticker": self.ticker,
            "invert": bool(self.invert),
            "cal_source": self.cal_source,
            "_schema": "PipelineCheckpoint.v1",
        }

    @classmethod
    def from_session_dict(cls, raw: dict[str, Any]) -> "PipelineCheckpoint":
        from fx_report.model.human_review import PendingReview
        from fx_report.model.weights import ScenarioSpec

        market_d = dict(raw.get("market") or {})
        market = MarketSnapshot(
            asof=str(market_d.get("asof") or ""),
            pair=str(market_d.get("pair") or raw.get("pair") or ""),
            spot=float(market_d.get("spot") or 0.0),
            provider_raw=float(market_d.get("provider_raw") or market_d.get("spot") or 0.0),
            sigma_daily=float(market_d.get("sigma_daily") or 0.01),
            sigma_annual=float(market_d.get("sigma_annual") or 0.16),
            mean_daily_return=float(market_d.get("mean_daily_return") or 0.0),
            n_returns=int(market_d.get("n_returns") or 0),
            lookback_days=int(market_d.get("lookback_days") or 60),
            history_start=str(market_d.get("history_start") or ""),
            history_end=str(market_d.get("history_end") or ""),
            source=str(market_d.get("source") or ""),
            brent=market_d.get("brent"),
            dxy_proxy=market_d.get("dxy_proxy"),
            notes=list(market_d.get("notes") or []),
            ret_1d=market_d.get("ret_1d"),
            ret_5d=market_d.get("ret_5d"),
            ret_20d=market_d.get("ret_20d"),
            sigma_20d_ann=market_d.get("sigma_20d_ann"),
            sigma_60d_ann=market_d.get("sigma_60d_ann"),
            history_ticker=str(market_d.get("history_ticker") or ""),
            spot_ticker=str(market_d.get("spot_ticker") or ""),
            used_proxy=bool(market_d.get("used_proxy", False)),
            cnh_cny_basis=market_d.get("cnh_cny_basis"),
        )
        info_needs = [
            InfoNeed(
                id=str(n.get("id") or ""),
                need=str(n.get("need") or ""),
                why=str(n.get("why") or ""),
                sources=str(n.get("sources") or ""),
                driver=str(n.get("driver") or ""),
            )
            for n in (raw.get("info_needs") or [])
        ]
        statements = [
            StoredStatement(
                id=str(s.get("id") or ""),
                statement=str(s.get("statement") or ""),
                source=str(s.get("source") or ""),
                url=str(s.get("url") or ""),
                provider=str(s.get("provider") or ""),
                published=s.get("published"),
                related_drivers=list(s.get("related_drivers") or []),
                raw=dict(s.get("raw") or {}),
                cluster_id=str(s.get("cluster_id") or ""),
            )
            for s in (raw.get("statements") or [])
        ]
        headlines: list[Headline] = []
        for h in raw.get("headlines") or []:
            pub = h.get("published")
            pub_dt: datetime | None = None
            if pub:
                try:
                    pub_dt = datetime.fromisoformat(str(pub).replace("Z", "+00:00"))
                except Exception:
                    pub_dt = None
            headlines.append(
                Headline(
                    title=str(h.get("title") or ""),
                    summary=str(h.get("summary") or ""),
                    source=str(h.get("source") or ""),
                    url=str(h.get("url") or ""),
                    published=pub_dt,
                    provider=str(h.get("provider") or ""),
                )
            )
        evidence: list[EvidenceItem] = []
        for row in raw.get("evidence") or []:
            evidence.append(
                EvidenceItem(
                    id=str(row.get("id") or ""),
                    title=str(row.get("title") or ""),
                    direction=int(row.get("direction") or 0),
                    strength=float(row.get("strength") or 0.0),
                    freshness=float(row.get("freshness") if row.get("freshness") is not None else 1.0),
                    unpriced=float(row.get("unpriced") if row.get("unpriced") is not None else 1.0),
                    category=str(row.get("category") or "other"),
                    note=str(row.get("note") or ""),
                    strength_label=str(row.get("strength_label") or ""),
                    strength_breakdown=dict(row.get("strength_breakdown") or {}),
                    source_tier=str(row.get("source_tier") or ""),
                    surprise=str(row.get("surprise") or ""),
                    scope=str(row.get("scope") or ""),
                    statement_id=str(row.get("statement_id") or ""),
                    url=str(row.get("url") or ""),
                    is_prior=bool(row.get("is_prior", False)),
                    cluster_id=str(row.get("cluster_id") or ""),
                    cluster_size=int(row.get("cluster_size") or 1),
                    cluster_role=str(row.get("cluster_role") or ""),
                    summary=str(row.get("summary") or ""),
                )
            )
        wraw = dict(raw.get("weights") or {})
        scenarios = [
            ScenarioSpec(
                name=str(s.get("name") or ""),
                weight=float(s.get("weight") or 0.0),
                mu_annual=float(s.get("mu_annual") or 0.0),
                sigma_mult=float(s.get("sigma_mult") or 1.0),
                expected_jumps=float(s.get("expected_jumps") or 0.0),
                jump_mean=float(s.get("jump_mean") or 0.0),
                jump_std=float(s.get("jump_std") or 0.0),
                narrative=str(s.get("narrative") or ""),
            )
            for s in (wraw.get("scenarios") or [])
        ]
        weights = ModelWeights(
            n_sims=int(wraw.get("n_sims") or 100_000),
            seed=int(wraw.get("seed") or 42),
            trading_days=int(wraw.get("trading_days") or 66),
            vol_lookback_days=int(wraw.get("vol_lookback_days") or 60),
            bucket_edges=tuple(wraw.get("bucket_edges") or (1.40, 1.43, 1.46, 1.49)),  # type: ignore[arg-type]
            use_relative_buckets=bool(wraw.get("use_relative_buckets", True)),
            bucket_pct_cuts=tuple(wraw.get("bucket_pct_cuts") or (0.0, 2.0, 4.0, 6.0)),  # type: ignore[arg-type]
            score_to_mu_a=float(wraw.get("score_to_mu_a") or 0.012),
            score_to_sigma_b=float(wraw.get("score_to_sigma_b") or 0.035),
            scenario_temperature=float(wraw.get("scenario_temperature") or 1.0),
            max_scenario_shift=float(wraw.get("max_scenario_shift") or 0.18),
            evidence_logit_scale=float(wraw.get("evidence_logit_scale") or 0.08),
            peak_engine=str(wraw.get("peak_engine") or "path_max"),
            jump_model=str(wraw.get("jump_model") or "merton"),
            jump_compensate=bool(wraw.get("jump_compensate", False)),
            vol_estimator=str(wraw.get("vol_estimator") or "window"),
            ewma_lambda=float(wraw.get("ewma_lambda") or 0.94),
            drift_mode=str(wraw.get("drift_mode") or "scenario"),
            carry_mu_annual=float(wraw.get("carry_mu_annual") or 0.0),
            scenarios=scenarios,
            evidence=list(evidence),
        )
        pending = [PendingReview.from_dict(p) for p in (raw.get("pending_reviews") or [])]
        return cls(
            pair=str(raw.get("pair") or market.pair),
            bullish_currency=str(raw.get("bullish_currency") or ""),
            stage_log=list(raw.get("stage_log") or []),
            info_needs=info_needs,
            market=market,
            statements=statements,
            headlines=headlines,
            evidence=evidence,
            news_meta=dict(raw.get("news_meta") or {}),
            weights=weights,
            pending_reviews=pending,
            variance_reduction=str(raw.get("variance_reduction") or "none"),
            as_of_date=raw.get("as_of_date"),
            display_pair=str(raw.get("display_pair") or raw.get("pair") or ""),
            ticker=raw.get("ticker"),
            invert=bool(raw.get("invert", False)),
            cal_source=str(raw.get("cal_source") or "default"),
        )


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
    allow_historical_ai: bool = False,
    llm_cfg: LLMConfig | None = None,
    vol_estimator: str = "window",
    ewma_lambda: float = 0.94,
    as_of_date: date | datetime | str | None = None,
) -> tuple[MarketSnapshot, list[StoredStatement], list[Headline], dict[str, Any]]:
    """
    3. 按信息需求抓取，并存储有影响的语句
    （行情 + 官方/RSS 头条 + 可选 AI 检索员；供赋权 / References / 数学分析）

    Historical (`as_of_date` set): AI / Tavily stay OFF unless
    ``allow_historical_ai=True`` (expensive; may inject non-historical web hits).
    """
    meta: dict[str, Any] = {"ai_research": None}
    cheap_hist = as_of_date is not None and not allow_historical_ai
    effective_ai = bool(ai_research) and not cheap_hist
    if as_of_date is not None:
        meta["cheap_historical"] = bool(cheap_hist)
        meta["allow_historical_ai"] = bool(allow_historical_ai)

    market = fetch_market(
        spec,
        lookback_days=lookback_days,
        vol_estimator=vol_estimator,
        ewma_lambda=ewma_lambda,
        as_of_date=as_of_date,
    )
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
        if as_of_date is not None:
            headlines, hist_meta = fetch_historical_headlines_for_pair(
                spec,
                as_of_date=as_of_date,
                lookback_days=lookback_days,
                max_items=max_items,
            )
            meta.update(hist_meta)
            meta["cheap_historical"] = bool(cheap_hist)
            meta["allow_historical_ai"] = bool(allow_historical_ai)
        else:
            headlines = fetch_headlines_for_pair(spec, max_items=max_items)

        # AI 检索员：白名单投行页 + 搜索 API + LLM 抽取展望（像人工补搜）
        if effective_ai and as_of_date is None:
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
        elif effective_ai and as_of_date is not None and allow_historical_ai:
            try:
                from fx_report.news.ai_research import run_ai_research

                ai = run_ai_research(
                    spec,
                    info_need_ids=[n.id for n in info_needs],
                    llm_cfg=llm_cfg,
                    max_headlines=12,
                    as_of_date=as_of_date,
                    allow_historical=True,
                )
                meta["ai_research"] = ai.meta
                seen = {(h.title or "").strip().lower() for h in headlines}
                for h in ai.headlines:
                    key = (h.title or "").strip().lower()
                    if key and key not in seen:
                        headlines.insert(0, h)
                        seen.add(key)
            except Exception as e:
                meta["ai_research"] = {
                    "enabled": True,
                    "allow_historical": True,
                    "errors": [f"ai_research_failed:{e}"],
                }
        elif as_of_date is not None and not effective_ai:
            meta["ai_research"] = {
                "enabled": False,
                "historical_disabled": True,
                "cheap_historical": True,
                "limitation": (
                    "历史回放默认省钱模式：已禁用 AI 检索员与 Tavily/Brave；"
                    "证据靠 GDELT+磁盘缓存+近窗 NewsAPI/inbox。"
                    "仅当显式 allow_historical_ai 时才可开启（贵且可能引入非历史信息）。"
                ),
            }

        allowed = list({n.driver for n in info_needs if n.driver} | set(spec.default_drivers))
        stmt_i = 0
        low_rel_skipped = 0
        for h in headlines:
            blob = f"{h.title} {h.summary}"
            rel = pair_relevance(blob, spec.pair)
            # Keep low-relevance out of the main statement pool used for scoring lineage
            if rel < MIN_PAIR_RELEVANCE:
                low_rel_skipped += 1
                continue
            related = match_drivers_from_text(blob, allowed=allowed or None)
            # If only unclassified, still store but do not invent first-2 drivers
            stmt_i += 1
            statements.append(
                StoredStatement(
                    id=f"STMT-{stmt_i:02d}",
                    statement=(h.title + ((" — " + h.summary[:240]) if h.summary else "")).strip(),
                    source=h.source,
                    url=h.url,
                    provider=h.provider,
                    published=h.published.isoformat() if h.published else None,
                    related_drivers=related,
                    raw={**(h.to_dict()), "pair_relevance": rel},
                )
            )
        meta["low_rel_skipped"] = low_rel_skipped
        meta["statements_news_n"] = stmt_i
    return market, statements, headlines, meta


def _mark_prior_templates(
    templates: list[EvidenceItem],
    *,
    downweight: bool,
) -> list[EvidenceItem]:
    """Copy template evidence, mark is_prior; optionally shrink strength/unpriced."""
    out: list[EvidenceItem] = []
    for e in templates:
        strength = e.strength * _PRIOR_STRENGTH_MULT if downweight else e.strength
        unpriced = e.unpriced * _PRIOR_UNPRICED_MULT if downweight else e.unpriced
        note = (e.note or "").strip()
        tag = "prior_template｜downweighted" if downweight else "prior_template"
        note = f"{tag}｜{note}" if note else tag
        out.append(
            EvidenceItem(
                id=e.id if e.id.startswith("P-") else f"P-{e.id}",
                title=e.title,
                direction=e.direction,
                strength=strength,
                freshness=e.freshness,
                unpriced=unpriced,
                category=e.category,
                note=note,
                strength_label=e.strength_label,
                strength_breakdown={**e.strength_breakdown, "prior": 1.0},
                source_tier=e.source_tier,
                surprise=e.surprise,
                scope=e.scope,
                statement_id=e.statement_id,
                url=e.url,
                is_prior=True,
                cluster_id=e.cluster_id,
                cluster_size=e.cluster_size,
                cluster_role=e.cluster_role,
                summary=str(e.summary or ""),
            )
        )
    return out


def _link_evidence_to_statements(
    evidence: list[EvidenceItem],
    statements: list[StoredStatement],
) -> None:
    """Attach statement_id (and url if missing) by URL / title match. Mutates in place."""
    by_url: dict[str, StoredStatement] = {}
    by_title: dict[str, StoredStatement] = {}
    for s in statements:
        if s.url:
            by_url[s.url.strip().lower()] = s
        title = (s.statement or "").split(" — ")[0].strip().lower()
        if title:
            by_title[title] = s
    for e in evidence:
        if e.statement_id:
            continue
        hit: StoredStatement | None = None
        if e.url:
            hit = by_url.get(e.url.strip().lower())
        if hit is None:
            hit = by_title.get((e.title or "").strip().lower())
        if hit is not None:
            e.statement_id = hit.id
            if not e.url and hit.url:
                e.url = hit.url


def step4_evaluate_impact(
    headlines: list[Headline],
    spec: PairSpec,
    market: MarketSnapshot,
    base_weights: ModelWeights,
    *,
    mode: ClassifyMode = "hybrid",
    max_items: int = 10,
    keep_templates: bool = False,
    template_policy: TemplatePolicy = "off",
    llm_cfg: LLMConfig | None = None,
    fetch_fulltext: bool = True,
    statements: list[StoredStatement] | None = None,
    as_of_date: date | datetime | str | None = None,
    cluster_events: bool = True,
    cluster_method: ClusterMethod = "jaccard",
    soft_adapt_drift: bool = DEFAULT_SOFT_ADAPT,
) -> tuple[list[EvidenceItem], dict[str, Any]]:
    """
    4. 评估每条信息对货币对的影响（方向 / 类别 / 强弱输入）

    template_policy (empty news→evidence):
      off           — do NOT silently use default_evidence (prefer; S→0)
      prior_only    — templates allowed, marked is_prior + downweighted
      fallback_warn — debug: use templates as-is, flag fallback_templates
    keep_templates: when news evidence non-empty, also append marked prior templates.
    cluster_events: assign EVT-* clusters and keep_strongest when summing S (default on).
    cluster_method: jaccard (default) | tfidf (sklearn if available, else Jaccard).
    soft_adapt_drift: if True and TV drift warns, soft-shrink over-represented
      strengths toward neutral (default False — warn-only).
    """
    suggested_up = calibrate_unpriced_from_market(market.ret_1d, market.ret_5d)
    cfg = llm_cfg
    use_mode = mode
    if use_mode in {"llm", "hybrid"} and cfg is None:
        cfg = resolve_llm_config()
    if use_mode in {"llm", "hybrid"} and cfg is None:
        use_mode = "rules"

    policy: TemplatePolicy = template_policy or "off"
    meta: dict[str, Any] = {
        "mode": use_mode,
        "template_policy": policy,
        "fallback_templates": False,
        "evidence_quality": "pending",
        "fetched": len(headlines),
        "kept": 0,
        "classified": 0,
        "evidence_n": 0,
        "evidence_raw_n": 0,
        "cluster_n": 0,
        "cluster_dup_n": 0,
        "cluster_dedup_applied": False,
        "cluster_dedup_mode": "off",
        "cluster_warnings": [],
        "summary_meta": {},
        "drift_meta": {},
    }

    auto: list[EvidenceItem] = []
    reference_now: datetime | None = None
    if as_of_date is not None:
        ref_date = as_of_date if isinstance(as_of_date, date) and not isinstance(as_of_date, datetime) else None
        if ref_date is None:
            text = str(as_of_date)
            if "T" in text:
                text = text.split("T", 1)[0]
            if " " in text:
                text = text.split(" ", 1)[0]
            ref_date = date.fromisoformat(text)
        reference_now = datetime.combine(ref_date, datetime.min.time(), tzinfo=timezone.utc)
    if headlines:
        auto, news_meta = build_evidence_from_news(
            headlines,
            spec,
            mode=use_mode,
            max_items=max_items,
            unpriced_cap=suggested_up,
            llm_cfg=cfg,
            fetch_fulltext=fetch_fulltext,
            reference_now=reference_now,
        )
        meta.update({k: v for k, v in news_meta.items() if k != "mode" or True})
        meta["mode"] = news_meta.get("mode", use_mode)

    if auto:
        evidence = list(auto)
        if keep_templates:
            evidence = evidence + _mark_prior_templates(
                list(base_weights.evidence), downweight=True
            )
        meta["fallback_templates"] = False
        meta["evidence_quality"] = "news_driven"
    else:
        # Honest empty path — no silent fake news evidence
        if policy == "off":
            evidence = []
            meta["fallback_templates"] = False
            meta["evidence_quality"] = "news_empty_no_prior"
        elif policy == "prior_only":
            evidence = _mark_prior_templates(list(base_weights.evidence), downweight=True)
            meta["fallback_templates"] = True
            meta["evidence_quality"] = "prior_only"
        else:  # fallback_warn
            evidence = list(base_weights.evidence)
            for e in evidence:
                e.is_prior = True
                if e.note and "fallback_warn" not in e.note:
                    e.note = f"fallback_warn｜{e.note}"
                elif not e.note:
                    e.note = "fallback_warn｜template"
            meta["fallback_templates"] = True
            meta["evidence_quality"] = "fallback_warn"

    if statements:
        _link_evidence_to_statements(evidence, statements)

    for e in evidence:
        e.unpriced = min(e.unpriced, suggested_up)
        # Cap unclassified so they cannot dominate even if somehow scored
        if (e.category or "").lower() == "unclassified":
            e.strength = min(e.strength, 0.25)
            e.direction = 0

    # ECDA-style summarization: short auditable blurb before cluster / HITL / S
    from fx_report.news.summarize import apply_evidence_summaries

    summary_meta = apply_evidence_summaries(
        evidence,
        headlines=headlines,
        prefer_llm=False,  # offline-safe default; LLM only if explicitly enabled later
    )
    meta["summary_meta"] = summary_meta

    # ECDA-style event clustering: same-theme headlines → one cluster before S
    # Pass quality/limitation so empty-news / 429 land in cluster_warnings (same ZH style).
    cluster_meta = assign_event_clusters(
        evidence,
        enabled=bool(cluster_events),
        news_meta=meta,
        cluster_method=cluster_method,
    )
    if statements:
        propagate_cluster_to_statements(evidence, statements)
    meta.update(cluster_meta.to_dict())
    if cluster_events and cluster_meta.cluster_dedup_applied:
        meta["cluster_dedup_mode"] = "keep_strongest"
    elif cluster_events:
        meta["cluster_dedup_mode"] = "keep_strongest"
    else:
        meta["cluster_dedup_mode"] = "off"

    # Light drift monitor: category/direction mix vs rolling baseline → audit warn
    # soft_adapt_drift default OFF: warn only; when ON, soft-shrink over-rep strengths
    from fx_report.news.drift import check_evidence_drift

    drift_report = check_evidence_drift(
        evidence,
        pair=spec.pair,
        out_dir="output",
        update_baseline=True,
        soft_adapt=bool(soft_adapt_drift),
    )
    meta["drift_meta"] = drift_report.to_dict()
    warn_list = list(cluster_meta.cluster_warnings or [])
    for w in drift_report.warnings:
        if w not in warn_list:
            warn_list.append(w)

    meta["evidence_n"] = len(evidence)
    meta["evidence_raw_n"] = cluster_meta.evidence_raw_n
    meta["prior_n"] = sum(1 for e in evidence if e.is_prior)
    meta["news_n"] = sum(1 for e in evidence if not e.is_prior)
    meta["cluster_warnings"] = warn_list
    return evidence, meta


def step5_assign_weights(evidence: list[EvidenceItem]) -> list[WeightedEvidence]:
    """5. 对每条信息赋予权重（贡献分）；unclassified / prior / 簇内去重已在 score 层处理"""
    out: list[WeightedEvidence] = []
    for e in evidence:
        if (e.category or "").lower() == "unclassified":
            contrib = 0.0
            impact = "未分类｜不计入主分"
        else:
            contrib = evidence_item_contrib(e)
            if e.direction > 0:
                impact = f"推高 {e.category} 路径上尾"
            elif e.direction < 0:
                impact = f"压制 {e.category} 路径峰值"
            else:
                impact = "中性"
            if e.is_prior:
                impact = f"[先验] {impact}"
            if e.cluster_role == "dup":
                impact = f"[簇内降权 {e.cluster_id}] {impact}"
            elif e.cluster_id and e.cluster_size > 1 and e.cluster_role == "rep":
                impact = f"[簇代表 {e.cluster_id} n={e.cluster_size}] {impact}"
        note = f"{impact}｜label={e.strength_label or 'n/a'}｜contrib={contrib:+.3f}"
        if e.statement_id:
            note += f"｜{e.statement_id}"
        if e.cluster_id:
            note += f"｜{e.cluster_id}/{e.cluster_role or 'n/a'}"
        out.append(WeightedEvidence(evidence=e, impact_note=note, weight_contrib=contrib))
    return out


def step6_math_analysis(
    market: MarketSnapshot,
    weights: ModelWeights,
    evidence: list[EvidenceItem],
    variance_reduction: str = "none",
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
        peak_engine=getattr(weights, "peak_engine", "path_max"),
        drift_mode=getattr(weights, "drift_mode", "scenario"),
        carry_mu_annual=float(getattr(weights, "carry_mu_annual", 0.0)),
        variance_reduction=variance_reduction,
        jump_model=getattr(weights, "jump_model", "merton"),
        jump_compensate=bool(getattr(weights, "jump_compensate", False)),
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
    bullish_currency: str | None = None,
    as_of_date: date | datetime | str | None = None,
) -> tuple[str, str, TorchcastReport, dict[str, Any], str]:
    """7. Torchcast 风格报告（HTML/PDF）+ Markdown 副本 + diagnostics"""
    if as_of_date is not None:
        if isinstance(as_of_date, date) and not isinstance(as_of_date, datetime):
            start = as_of_date
        elif isinstance(as_of_date, datetime):
            start = as_of_date.date()
        else:
            start = date.fromisoformat(str(as_of_date).split("T", 1)[0].split(" ", 1)[0])
    else:
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
        bullish_currency=bullish_currency,
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
    ref_cap = refs_statement_cap(int(news_meta.get("max_news") or len(weighted) or 10))
    refs_md = "\n".join(
        f"{i}. [{s.source}] {s.statement[:160]}"
        + (f" — {s.url}" if s.url else "")
        for i, s in enumerate(statements[:ref_cap], 1)
    )
    weights_md = "\n".join(
        f"| {w.evidence.id} | {w.evidence.strength_label} | {w.weight_contrib:+.3f} | {w.impact_note} |"
        for w in weighted
    )
    bullish_line = (
        f"Bullish: **{bullish_currency}**｜Analysis quote: **{market.pair}**\n\n"
        if bullish_currency
        else ""
    )
    preface = f"""## 分析流程（固定七步）

{bullish_line}1. 选择货币对 → **{market.pair}**
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
    diag["evidence_quality"] = news_meta.get("evidence_quality")
    diag["fallback_templates"] = bool(news_meta.get("fallback_templates"))
    diag["template_policy"] = news_meta.get("template_policy", "off")
    diag["evidence_counts"] = {
        "fetched": news_meta.get("fetched", 0),
        "kept": news_meta.get("kept", 0),
        "classified": news_meta.get("classified", 0),
        "evidence_n": news_meta.get("evidence_n", len(weights.evidence)),
        "evidence_raw_n": news_meta.get("evidence_raw_n", news_meta.get("evidence_n", 0)),
        "prior_n": news_meta.get("prior_n", 0),
        "news_n": news_meta.get("news_n", 0),
        "cluster_n": news_meta.get("cluster_n", 0),
        "cluster_dup_n": news_meta.get("cluster_dup_n", 0),
        "cluster_dedup_applied": bool(news_meta.get("cluster_dedup_applied")),
        "cluster_dedup_mode": news_meta.get("cluster_dedup_mode", "off"),
        "cluster_warnings": list(news_meta.get("cluster_warnings") or []),
        "summary_meta": dict(news_meta.get("summary_meta") or {}),
        "drift_meta": dict(news_meta.get("drift_meta") or {}),
    }
    diag["cluster_warnings"] = list(news_meta.get("cluster_warnings") or [])
    diag["summary_meta"] = dict(news_meta.get("summary_meta") or {})
    diag["drift_meta"] = dict(news_meta.get("drift_meta") or {})
    if bullish_currency:
        diag["bullish_currency"] = bullish_currency
    # Push audit fields into torchcast HTML/PDF meta
    tc.extra["evidence_quality"] = diag["evidence_quality"]
    tc.extra["fallback_templates"] = diag["fallback_templates"]
    tc.extra["template_policy"] = diag["template_policy"]
    tc.extra["cluster_n"] = news_meta.get("cluster_n", 0)
    tc.extra["evidence_raw_n"] = news_meta.get("evidence_raw_n", news_meta.get("evidence_n", 0))
    tc.extra["cluster_dedup_applied"] = bool(news_meta.get("cluster_dedup_applied"))
    tc.extra["cluster_warnings"] = list(news_meta.get("cluster_warnings") or [])
    # OOS / calib trust line when bundled or local summary exists
    try:
        from fx_report.model.calibrate import load_calib_oos_summary

        oos = load_calib_oos_summary(market.pair)
    except Exception:
        oos = None
    if oos:
        hold = oos.get("holdout") or {}
        train = oos.get("train") or {}
        tc.extra["calib_oos"] = {
            "holdout_hit_rate": hold.get("hit_rate"),
            "holdout_brier": hold.get("brier"),
            "holdout_skill_brier": hold.get("skill_brier"),
            "holdout_logloss": hold.get("logloss"),
            "holdout_ece": hold.get("reliability_ece"),
            "holdout_n": hold.get("n"),
            "train_brier": train.get("brier"),
            "train_skill_brier": train.get("skill_brier"),
            "train_hit_rate": train.get("hit_rate"),
            "source": oos.get("source"),
        }
        diag["calib_oos"] = tc.extra["calib_oos"]
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
    peak_engine: str = "path_max",
    variance_reduction: str = "none",
    jump_model: str = "merton",
    jump_compensate: bool = False,
    mode: ClassifyMode = "hybrid",
    max_news: int = 10,
    keep_templates: bool = False,
    template_policy: TemplatePolicy = "off",
    no_news: bool = False,
    no_fulltext: bool = False,
    ai_research: bool = True,
    allow_historical_ai: bool = False,
    llm_cfg: LLMConfig | None = None,
    out_dir: str | Path | None = "output",
    verbose: bool = True,
    bullish_currency: str | None = None,
    model_weights: ModelWeights | None = None,
    calibrated_params_path: str | Path | None = None,
    calibrated_params_label: str | None = None,
    use_label_learned_strength: bool = False,
    as_of_date: date | datetime | str | None = None,
    human_review_mode: HumanReviewMode = "auto_skip",
    max_uncertain: int = 5,
    review_overrides: dict[str, Any] | None = None,
    _phase_a_only: bool = False,
) -> PipelineResult | PipelineCheckpoint:
    """跑完整七步；可选写入 output/。

    `model_weights`：UI 传入时，在默认权重上覆盖分档切点 / 映射 / 情景 / 模板证据等，
    确保蒙特卡洛与 FX Analyse 使用同一套用户分档。

    `calibrated_params_path`：可选 Stage-1 JSON，覆盖 score_to_* / 情景先验等。
    `template_policy`：off（默认，无静默模板）/ prior_only / fallback_warn。
    `use_label_learned_strength`：若 label_audit 标注 ≥N 条，用类别强度倍率缩放证据。

    人机协同（不确定证据）：
      human_review_mode:
        off        — 不检测，直接赋权
        auto_skip  — 检测并记入日志/meta，保留模型方向继续（CLI 默认）
        pause      — 有 pending 且无 overrides 时返回 PipelineCheckpoint
      review_overrides: {evidence_id|statement_id → up|down|neutral|skip}
      `_phase_a_only`：Phase A 专用；返回 checkpoint（不跑 MC）。
    """
    log: list[str] = []

    def say(msg: str) -> None:
        log.append(msg)
        if verbose:
            print(msg)

    # 1
    say("【1/7】选择货币对")
    display_spec = step1_select_pair(pair, ticker=ticker, invert=invert)
    say(f"  → 展示对 {display_spec.pair}｜{display_spec.description}")

    bullish = (bullish_currency or display_spec.base).strip().upper()
    if bullish_currency is None:
        say(
            f"  → 未指定看涨货币，默认看涨 base={bullish} "
            f"（分析报价升高 = {bullish} 走强）"
        )
    else:
        say(f"  → 看涨货币 {bullish}")

    spec = resolve_pair_for_bullish(display_spec, bullish)
    if spec.pair != display_spec.pair:
        say(f"  → 分析口径翻转 {display_spec.pair} → {spec.pair}")
    else:
        say(f"  → 分析口径 {spec.pair}（不变）")
    say(f"  → {spec.pair}｜{spec.description}")
    say(f"  → {describe_pair_factors(spec.base, spec.quote, spec.default_drivers)}")

    base = default_weights(spec)
    cal_source = "default"
    if calibrated_params_label:
        cal_source = str(calibrated_params_label)
    cal_path: Path | None = None
    if calibrated_params_path:
        cal_path = Path(calibrated_params_path)
    elif model_weights is None:
        from fx_report.model.calibrate import resolve_calibrated_params_path

        cal_path = resolve_calibrated_params_path(spec.pair)
    if cal_path is not None:
        from fx_report.model.calibrate import apply_calibrated_params, load_calibrated_params

        if cal_path.exists():
            apply_calibrated_params(base, load_calibrated_params(cal_path))
            cal_source = calibrated_params_label or str(cal_path)
            say(f"  → 已加载校准参数 {cal_path}")
        else:
            say(f"  → 警告：校准文件不存在，跳过：{cal_path}")
    if model_weights is not None:
        base.n_sims = int(model_weights.n_sims)
        base.trading_days = int(model_weights.trading_days)
        base.seed = int(model_weights.seed)
        base.vol_lookback_days = int(model_weights.vol_lookback_days)
        base.use_relative_buckets = bool(model_weights.use_relative_buckets)
        base.bucket_pct_cuts = tuple(model_weights.bucket_pct_cuts)  # type: ignore[assignment]
        base.bucket_edges = tuple(model_weights.bucket_edges)  # type: ignore[assignment]
        base.score_to_mu_a = float(model_weights.score_to_mu_a)
        base.score_to_sigma_b = float(model_weights.score_to_sigma_b)
        base.evidence_logit_scale = float(model_weights.evidence_logit_scale)
        base.scenario_temperature = float(model_weights.scenario_temperature)
        base.max_scenario_shift = float(model_weights.max_scenario_shift)
        if getattr(model_weights, "peak_engine", None):
            base.peak_engine = str(model_weights.peak_engine)
        if getattr(model_weights, "jump_model", None):
            base.jump_model = str(model_weights.jump_model)
        base.jump_compensate = bool(getattr(model_weights, "jump_compensate", False))
        if model_weights.scenarios:
            base.scenarios = list(model_weights.scenarios)
        if model_weights.evidence:
            base.evidence = list(model_weights.evidence)
        base.vol_estimator = str(getattr(model_weights, "vol_estimator", "window"))
        base.ewma_lambda = float(getattr(model_weights, "ewma_lambda", 0.94))
        base.drift_mode = str(getattr(model_weights, "drift_mode", "scenario"))
        base.carry_mu_annual = float(getattr(model_weights, "carry_mu_annual", 0.0))
    else:
        base.n_sims = sims
        base.trading_days = days
        base.seed = seed
        base.vol_lookback_days = lookback
        base.peak_engine = str(peak_engine or "path_max")
        base.jump_model = str(jump_model or "merton")
        base.jump_compensate = bool(jump_compensate)
    say(
        f"  → 分档 "
        f"{'相对% ' + str(base.bucket_pct_cuts) if base.use_relative_buckets else '绝对 ' + str(base.bucket_edges)}"
    )
    say(f"  → peak_engine={base.peak_engine}")
    say(f"  → jump_model={base.jump_model}  jump_compensate={base.jump_compensate}")
    say(f"  → vol_estimator={base.vol_estimator}  drift_mode={base.drift_mode}")
    say(f"  → calibrated_params={cal_source}")
    say(f"  → template_policy={template_policy}")
    if as_of_date is not None:
        say(f"  → historical_as_of={as_of_date}")

    # 2
    say("【2/7】评估所需要的信息（因货币对而异）")
    info_needs = step2_assess_info_needs(spec)
    for n in info_needs:
        label = DRIVER_CATALOG[n.id].label if n.id in DRIVER_CATALOG else n.id
        say(f"  · [{label}] {n.need}")

    # 3 — historical as_of defaults to cheap (no AI/Tavily) unless allow_historical_ai
    cheap_hist = as_of_date is not None and not allow_historical_ai
    effective_ai = bool(ai_research) and not cheap_hist
    if as_of_date is not None and cheap_hist:
        say("  → cheap_historical=ON（AI 检索员 / Tavily 关闭；证据靠 GDELT+缓存）")
    say(
        "【3/7】抓取并存储有影响的语句"
        + ("（含 AI 检索员）" if effective_ai and not no_news else "")
    )
    step3_items = step3_pool_size(max_news, historical=as_of_date is not None)
    market, statements, headlines, step3_meta = step3_collect_and_store_statements(
        spec,
        info_needs,
        lookback_days=lookback,
        max_items=step3_items,
        skip_news=no_news,
        ai_research=ai_research,
        allow_historical_ai=allow_historical_ai,
        llm_cfg=llm_cfg,
        vol_estimator=getattr(base, "vol_estimator", "window"),
        ewma_lambda=float(getattr(base, "ewma_lambda", 0.94)),
        as_of_date=as_of_date,
    )
    say(f"  → 行情 {market.source} spot={market.spot:.5f}")
    say(f"  → 存储语句 {len(statements)} 条｜头条 {len(headlines)} 条｜step3_pool={step3_items}")
    if step3_meta.get("low_rel_skipped"):
        say(f"  → 低相关头条跳过 {step3_meta['low_rel_skipped']} 条")
    if as_of_date is not None:
        say(f"  → {format_cheap_historical_caption(step3_meta, cheap_historical=cheap_hist)}")
    ai_meta = step3_meta.get("ai_research") or {}
    if ai_meta:
        rounds = ai_meta.get("rounds") or []
        n_search = sum(1 for r in rounds if r.get("action") == "search")
        say(
            f"  → AI 检索（迭代）：白名单 {ai_meta.get('whitelist_ok', 0)}｜"
            f"轮次 {n_search}｜命中 {ai_meta.get('search_hits', 0)}｜"
            f"精选 {ai_meta.get('kept_hits', ai_meta.get('search_hits', 0))}｜"
            f"产出 {ai_meta.get('headlines_out', 0)}｜"
            f"脑={'on' if ai_meta.get('llm') else 'off'}"
        )
        lim = ai_meta.get("limitation")
        if lim:
            say(f"  → AI 检索限制：{lim}")

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
        template_policy=template_policy,
        llm_cfg=llm_cfg,
        fetch_fulltext=not no_fulltext,
        statements=statements,
        as_of_date=as_of_date,
    )
    news_meta["ai_research"] = ai_meta
    news_meta["step3"] = {
        k: step3_meta[k]
        for k in ("low_rel_skipped", "statements_news_n")
        if k in step3_meta
    }
    for k in (
        "historical_mode",
        "historical_as_of",
        "historical_lookback_days",
        "historical_lookback_days_effective",
        "newsapi_from",
        "newsapi_from_requested",
        "newsapi_from_clamped",
        "newsapi_outside_window",
        "providers_used",
        "newsapi_enabled",
        "newsapi_hits",
        "newsapi_error",
        "newsapi_http_status",
        "newsapi_from_cache",
        "gdelt_enabled",
        "gdelt_hits",
        "gdelt_from",
        "gdelt_from_requested",
        "gdelt_from_clamped",
        "gdelt_outside_window",
        "gdelt_error",
        "gdelt_http_status",
        "gdelt_from_cache",
        "gdelt_query",
        "inbox_dated_hits",
        "cheap_historical",
        "allow_historical_ai",
        "historical_news_quality",
        "limitation",
    ):
        if k in step3_meta and k not in news_meta:
            news_meta[k] = step3_meta[k]
    news_meta["max_news"] = int(max_news)
    news_meta["step3_pool"] = int(step3_items)
    news_meta["calibrated_params"] = cal_source
    say(
        f"  → 证据 {len(evidence)} 条｜mode={news_meta.get('mode')}｜"
        f"quality={news_meta.get('evidence_quality')}｜"
        f"fallback_templates={news_meta.get('fallback_templates')}"
    )

    label_learn_meta: dict[str, Any] = {
        "requested": bool(use_label_learned_strength),
        "applied": False,
        "ready": False,
        "message": "",
    }
    if use_label_learned_strength:
        from fx_report.model.label_learn import (
            apply_label_learned_strength,
            fit_label_learned_params,
            save_label_learned_params,
        )

        learned = fit_label_learned_params()
        label_learn_meta["ready"] = bool(learned.ready)
        label_learn_meta["n_labeled"] = learned.n_labeled
        label_learn_meta["min_required"] = learned.min_required
        label_learn_meta["message"] = learned.message
        if learned.ready:
            evidence, apply_meta = apply_label_learned_strength(evidence, learned)
            label_learn_meta.update(apply_meta)
            try:
                save_label_learned_params(learned)
            except Exception:
                pass
            say(
                f"  → 标签学习强度：已应用 "
                f"（scaled={apply_meta.get('n_strength_scaled', 0)}，"
                f"nudged={apply_meta.get('n_dir_nudged', 0)}）"
            )
        else:
            say(f"  → 标签学习强度未启用：{learned.message}")
    news_meta["label_learn"] = label_learn_meta

    # 4b · 不确定证据（人机协同，赋权前）
    from fx_report.model.human_review import (
        detect_uncertain_evidence,
        direction_zh,
    )

    pending: list[Any] = []
    if human_review_mode != "off":
        pending = detect_uncertain_evidence(
            evidence,
            pair=spec.pair,
            max_items=int(max_uncertain),
        )
    news_meta["pending_reviews"] = [p.to_dict() for p in pending]
    news_meta["human_review"] = {
        "mode": human_review_mode,
        "n_pending": len(pending),
        "auto_skipped": False,
        "n_overridden": 0,
    }
    if pending:
        say(f"  → 不确定证据 {len(pending)} 条（上限 {max_uncertain}）")
        for p in pending:
            say(
                f"    · {p.evidence_id}: {', '.join(p.reasons_zh)}｜"
                f"模型={direction_zh(p.model_direction)}｜{p.title[:56]}"
            )
    as_of_s: str | None = None
    if as_of_date is not None:
        if isinstance(as_of_date, datetime):
            as_of_s = as_of_date.date().isoformat()
        elif isinstance(as_of_date, date):
            as_of_s = as_of_date.isoformat()
        else:
            as_of_s = str(as_of_date).split("T", 1)[0].split(" ", 1)[0]

    checkpoint = PipelineCheckpoint(
        pair=spec.pair,
        bullish_currency=bullish,
        stage_log=list(log),
        info_needs=info_needs,
        market=market,
        statements=statements,
        headlines=headlines,
        evidence=list(evidence),
        news_meta=dict(news_meta),
        weights=base,
        pending_reviews=pending,
        variance_reduction=str(variance_reduction or "none"),
        as_of_date=as_of_s,
        display_pair=display_spec.pair,
        ticker=ticker,
        invert=bool(invert),
        cal_source=cal_source,
    )

    if _phase_a_only:
        say("  → Phase A 完成（等待人工确认 / 或继续 Phase B）")
        return checkpoint

    if human_review_mode == "pause" and pending and not review_overrides:
        say("  → 已暂停：请在界面确认方向后再继续（或传入 review_overrides）")
        return checkpoint

    if human_review_mode == "auto_skip" and pending and not review_overrides:
        news_meta["human_review"]["auto_skipped"] = True
        checkpoint.news_meta = dict(news_meta)
        say("  → 不确定证据已自动跳过（保留模型方向继续赋权）")

    return run_pipeline_phase_b(
        checkpoint,
        review_overrides=review_overrides,
        out_dir=out_dir,
        verbose=verbose,
    )


def run_pipeline_phase_a(
    pair: str = "USD/AUD",
    **kwargs: Any,
) -> PipelineCheckpoint:
    """Phase A：抓取 + 分类 + 聚类 → 返回 pending_reviews（不赋权/MC）。"""
    kwargs = dict(kwargs)
    kwargs["_phase_a_only"] = True
    kwargs.setdefault("human_review_mode", "pause")
    kwargs.setdefault("out_dir", None)
    out = run_pipeline(pair, **kwargs)
    assert isinstance(out, PipelineCheckpoint)
    return out


def run_pipeline_phase_b(
    checkpoint: PipelineCheckpoint | dict[str, Any],
    *,
    review_overrides: dict[str, Any] | None = None,
    out_dir: str | Path | None = "output",
    verbose: bool = True,
    save_label_audit: bool = True,
) -> PipelineResult:
    """Phase B：应用人工选择 → 赋权 → MC → 报告。"""
    from fx_report.model.human_review import (
        apply_review_overrides,
        reviews_to_label_rows,
    )
    from fx_report.model.label_audit import (
        agree_rate_stats,
        label_audit_path,
        save_label_audit as _save_label_audit,
        save_spotcheck_stats,
    )
    import pandas as pd

    if isinstance(checkpoint, dict):
        cp = PipelineCheckpoint.from_session_dict(checkpoint)
    else:
        cp = checkpoint

    log = list(cp.stage_log)

    def say(msg: str) -> None:
        log.append(msg)
        if verbose:
            print(msg)

    evidence = list(cp.evidence)
    news_meta = dict(cp.news_meta)
    base = cp.weights
    market = cp.market
    info_needs = cp.info_needs
    statements = cp.statements
    headlines = cp.headlines
    bullish = cp.bullish_currency
    pending = list(cp.pending_reviews)
    cal_source = cp.cal_source
    variance_reduction = cp.variance_reduction
    as_of_date = cp.as_of_date

    hr = dict(news_meta.get("human_review") or {})
    choices = dict(review_overrides or {})
    if choices:
        evidence, ov_meta = apply_review_overrides(evidence, choices)
        hr.update(ov_meta)
        hr["applied_overrides"] = True
        say(
            f"  → 人工覆盖方向 {ov_meta.get('n_overridden', 0)} 条"
            f"（跳过 {ov_meta.get('n_skipped', 0)}）"
        )
        news_meta["human_review"] = hr
        if save_label_audit and pending:
            try:
                rows = reviews_to_label_rows(pending, choices, pair=cp.pair)
                df = pd.DataFrame(rows)
                labeled = df[df["human_direction"].astype(str).str.len() > 0]
                if not labeled.empty:
                    path = label_audit_path(cp.pair)
                    if path.exists():
                        try:
                            old = pd.read_csv(path)
                            keep = old[
                                ~old["statement_id"]
                                .astype(str)
                                .isin(labeled["statement_id"].astype(str))
                            ]
                            labeled = pd.concat([keep, labeled], ignore_index=True)
                        except Exception:
                            pass
                    _save_label_audit(labeled, path)
                    stats = agree_rate_stats(labeled)
                    save_spotcheck_stats(stats, cp.pair)
                    say(f"  → 已写入 label_audit：{path.name}")
            except Exception as exc:
                say(f"  → label_audit 保存失败（忽略）：{exc}")
    elif pending:
        hr["auto_skipped"] = True
        news_meta["human_review"] = hr

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
        market, base, evidence, variance_reduction=variance_reduction
    )
    say(f"  → S={score:+.3f}  μ_shift={mu_shift:+.4f}  σ×={sigma_extra:.3f}")
    if getattr(mc, "bb_jumps_caveat", None):
        say(f"  ⚠ {mc.bb_jumps_caveat}")
    for k, v in probs.items():
        say(f"  · {k}: {v:.1%}")

    # 7
    say("【7/7】输出 FX Analyse 格式报告（PDF / HTML）")
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
        bullish_currency=bullish,
        as_of_date=as_of_date,
    )
    diagnostics["calibrated_params"] = cal_source
    diagnostics["human_review"] = news_meta.get("human_review")
    diagnostics["pending_reviews"] = news_meta.get("pending_reviews")
    torchcast.extra["calibrated_params"] = cal_source
    torchcast.extra["human_review"] = news_meta.get("human_review")

    result = PipelineResult(
        pair=cp.pair,
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
