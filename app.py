"""
Multi-pair FX peak-bucket forecaster — Streamlit UI.

Sidebar: pair selector + all hidden weights + strength rubric checklist.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from fetch_data import calibrate_unpriced_from_market, fetch_market
from monte_carlo import enforce_math_floor, run_mixture_monte_carlo
from news_evidence import build_evidence_from_news
from news_fetch import fetch_headlines_for_pair
from pairs import get_pair, list_pairs, make_custom_pair
from report_text import build_diagnostics, build_report_markdown
from strength import (
    SOURCE_TIER_POINTS,
    SURPRISE_POINTS,
    SCOPE_POINTS,
    StrengthInputs,
    label_strength,
    rubric_markdown,
    score_strength,
)
from weights import (
    EvidenceItem,
    ModelWeights,
    ScenarioSpec,
    apply_evidence_to_scenarios,
    default_weights,
    evidence_score,
    resolve_bucket_edges,
)

# LLM helpers — keep resilient so Cloud doesn't crash on partial deploys
try:
    from news_llm import FREE_PROVIDERS, ollama_available, resolve_llm_config
except Exception:  # pragma: no cover
    FREE_PROVIDERS = {
        "ollama": {
            "label": "Ollama 本机（免费）",
            "api_key": "ollama",
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "llama3.1:latest",
        },
        "groq": {
            "label": "Groq",
            "api_key": "",
            "base_url": "https://api.groq.com/openai/v1",
            "model": "llama-3.1-8b-instant",
            "signup": "https://console.groq.com/keys",
        },
    }

    def ollama_available(timeout: float = 1.5) -> bool:
        return False

    def resolve_llm_config(*, api_key=None, base_url=None, model=None, allow_ollama_auto=True):
        try:
            from news_llm import resolve_llm_config as _resolve

            return _resolve(
                api_key=api_key,
                base_url=base_url,
                model=model,
                allow_ollama_auto=allow_ollama_auto,
            )
        except Exception:
            return None


st.set_page_config(
    page_title="FX Peak MC 情报报告",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _horizon_label(start: date, end: date) -> str:
    return f"{start.isoformat()} 至 {end.isoformat()}"


def pick_pair_spec():
    st.sidebar.title("货币对")
    mode = st.sidebar.radio("选择方式", ["目录", "自定义"], horizontal=True)
    if mode == "目录":
        pair = st.sidebar.selectbox("货币对", list_pairs(), index=list_pairs().index("USD/AUD"))
        return get_pair(pair)

    pair = st.sidebar.text_input("BASE/QUOTE", value="EUR/USD")
    ticker = st.sidebar.text_input("Yahoo ticker", value="EURUSD=X")
    invert = st.sidebar.checkbox("需要对 Yahoo 收盘取倒数", value=False)
    return make_custom_pair(pair, ticker, invert)


def render_strength_rules_sidebar() -> None:
    """Always-visible strength rubric in the sidebar."""
    st.sidebar.header("信息强弱判定")
    st.sidebar.markdown(
        """
**贡献分**  
`contrib = direction × strength × freshness × unpriced`

**strength（0–3）**  
`= min(3, 来源分 + 意外分 + 范围分)`

**标签**：≤1 SLIGHT｜≤2 MODERATE｜>2 STRONG
        """
    )
    st.sidebar.caption("来源档 source_tier")
    st.sidebar.dataframe(
        pd.DataFrame(
            {"档位": list(SOURCE_TIER_POINTS.keys()), "分": list(SOURCE_TIER_POINTS.values())}
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.sidebar.caption("意外程度 surprise")
    st.sidebar.dataframe(
        pd.DataFrame(
            {"档位": list(SURPRISE_POINTS.keys()), "分": list(SURPRISE_POINTS.values())}
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.sidebar.caption("影响范围 scope")
    st.sidebar.dataframe(
        pd.DataFrame(
            {"档位": list(SCOPE_POINTS.keys()), "分": list(SCOPE_POINTS.values())}
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.sidebar.markdown(
        """
**freshness** = `0.5 ** (年龄日 / 半衰期)`  
地缘≈5日｜CPI/央行≈7–8日｜仓位≈12日

**unpriced**：0=已定价完，1=几乎未定价；即期已大跳应下调，避免双计。

下方每条证据可选「自动打分」：按来源/意外/范围三项相加，结果会显示 strength 拆解。
        """
    )
    with st.sidebar.expander("完整规则说明（Markdown）", expanded=False):
        st.markdown(rubric_markdown())


def sidebar_weights(base: ModelWeights, pair_name: str) -> tuple[ModelWeights, dict]:
    st.sidebar.title("隐藏参数 / 权重")
    st.sidebar.caption(f"当前分析口径：{pair_name}。+1 方向 = 推高该报价的路径最高值。")

    render_strength_rules_sidebar()

    st.sidebar.header("0. 头条自动填证据")
    use_news = st.sidebar.checkbox("运行时抓取头条并自动填证据", value=True)
    keep_templates = st.sidebar.checkbox("保留模板证据（与头条合并）", value=False)
    max_news_ev = st.sidebar.slider("最多采用头条证据条数", 3, 20, 10, 1)
    classify_mode = st.sidebar.selectbox(
        "证据判定方式",
        options=["hybrid", "llm", "rules"],
        index=0,
        help="hybrid=大模型优先，失败回退关键词；llm=仅大模型；rules=仅关键词",
    )
    fetch_fulltext = st.sidebar.checkbox("抓取文章正文供大模型精读", value=True)

    has_ollama = ollama_available()
    provider_labels = {
        "ollama": f"Ollama 本机免费{'（已检测到）' if has_ollama else '（未运行）'}",
        "groq": "Groq 云端免费额度（需自行申请 Key）",
        "custom": "自定义 / OpenAI 兼容",
    }
    default_provider = "ollama" if has_ollama else "custom"
    provider = st.sidebar.selectbox(
        "大模型通道",
        options=list(provider_labels.keys()),
        format_func=lambda k: provider_labels[k],
        index=list(provider_labels.keys()).index(default_provider),
    )
    if provider == "ollama":
        st.sidebar.info("使用本机 Ollama，无需云端 Key，不会公开、不产生费用。")
        llm_key = "ollama"
        llm_base = FREE_PROVIDERS["ollama"]["base_url"]
        llm_model = st.sidebar.text_input("LLM Model", value=FREE_PROVIDERS["ollama"]["model"])
    elif provider == "groq":
        st.sidebar.markdown(f"免费申请：[Groq Console]({FREE_PROVIDERS['groq']['signup']})")
        llm_key = st.sidebar.text_input("Groq API Key", type="password", value="")
        llm_base = FREE_PROVIDERS["groq"]["base_url"]
        llm_model = st.sidebar.text_input("LLM Model", value=FREE_PROVIDERS["groq"]["model"])
    else:
        st.sidebar.caption(
            "Key 只存在于你的会话/本机 secrets，不会写入 Git。"
            "公开网页若配置云端 Secrets，访客可能消耗你的额度（但看不到 Key）。"
        )
        llm_key = st.sidebar.text_input("LLM API Key", type="password", value="")
        llm_base = st.sidebar.text_input("LLM Base URL", value="")
        llm_model = st.sidebar.text_input("LLM Model", value="")

    st.sidebar.header("1. 模拟设置")
    n_sims = st.sidebar.number_input("蒙特卡洛次数", 10_000, 500_000, base.n_sims, 10_000)
    seed = st.sidebar.number_input("随机种子", 0, 10_000_000, base.seed, 1)
    trading_days = st.sidebar.number_input("交易日窗口", 5, 252, base.trading_days, 1)
    vol_lookback = st.sidebar.number_input("波动率回看交易日", 20, 252, base.vol_lookback_days, 5)

    st.sidebar.header("2. 分档（相对现价 %）")
    use_rel = st.sidebar.checkbox("用相对现价分档（推荐，跨货币对通用）", value=True)
    c0 = st.sidebar.number_input("切点1 %", -5.0, 20.0, float(base.bucket_pct_cuts[0]), 0.5)
    c1 = st.sidebar.number_input("切点2 %", -5.0, 20.0, float(base.bucket_pct_cuts[1]), 0.5)
    c2 = st.sidebar.number_input("切点3 %", -5.0, 20.0, float(base.bucket_pct_cuts[2]), 0.5)
    c3 = st.sidebar.number_input("切点4 %", -5.0, 20.0, float(base.bucket_pct_cuts[3]), 0.5)
    cuts = tuple(sorted([c0, c1, c2, c3]))

    st.sidebar.header("3. 证据 → 参数映射")
    a = st.sidebar.slider("a：S→年化漂移", 0.0, 0.05, float(base.score_to_mu_a), 0.001, format="%.3f")
    b = st.sidebar.slider("b：|S|→波动放大", 0.0, 0.15, float(base.score_to_sigma_b), 0.001, format="%.3f")
    logit_scale = st.sidebar.slider("证据→情景 logit", 0.0, 0.3, float(base.evidence_logit_scale), 0.01)
    temperature = st.sidebar.slider("情景温度", 0.3, 2.5, float(base.scenario_temperature), 0.1)
    max_shift = st.sidebar.slider("单情景最大位移", 0.0, 0.4, float(base.max_scenario_shift), 0.01)

    st.sidebar.header("4. 情景先验")
    scenarios: list[ScenarioSpec] = []
    for sc in base.scenarios:
        st.sidebar.subheader(sc.name)
        w = st.sidebar.slider(f"{sc.name} 权重", 0.0, 1.0, float(sc.weight), 0.01, key=f"w_{sc.name}")
        mu = st.sidebar.slider(f"{sc.name} μ", -0.2, 0.2, float(sc.mu_annual), 0.005, key=f"mu_{sc.name}")
        sm = st.sidebar.slider(f"{sc.name} σ×", 0.5, 2.5, float(sc.sigma_mult), 0.05, key=f"sm_{sc.name}")
        ej = st.sidebar.slider(f"{sc.name} E[jumps]", 0.0, 3.0, float(sc.expected_jumps), 0.05, key=f"ej_{sc.name}")
        jm = st.sidebar.slider(f"{sc.name} jump μ", -0.03, 0.03, float(sc.jump_mean), 0.001, key=f"jm_{sc.name}")
        js = st.sidebar.slider(f"{sc.name} jump σ", 0.001, 0.03, float(sc.jump_std), 0.001, key=f"js_{sc.name}")
        scenarios.append(
            ScenarioSpec(sc.name, w, mu, sm, ej, jm, js, sc.narrative)
        )

    st.sidebar.header("5. 证据计分卡（可自动打分）")
    evidence: list[EvidenceItem] = []
    tier_keys = list(SOURCE_TIER_POINTS.keys())
    sur_keys = list(SURPRISE_POINTS.keys())
    sco_keys = list(SCOPE_POINTS.keys())

    for e in base.evidence:
        with st.sidebar.expander(f"{e.id} · {e.title}", expanded=False):
            enabled = st.checkbox("启用", value=True, key=f"en_{e.id}")
            direction = st.selectbox(
                "方向 +1推高峰值 / -1压制",
                [1, -1],
                index=0 if e.direction > 0 else 1,
                key=f"dir_{e.id}",
            )
            auto = st.checkbox("用规则自动打 strength", value=True, key=f"auto_{e.id}")
            src = st.selectbox(
                "来源档",
                tier_keys,
                index=tier_keys.index(e.source_tier) if e.source_tier in tier_keys else 2,
                key=f"src_{e.id}",
            )
            sur = st.selectbox(
                "意外程度",
                sur_keys,
                index=sur_keys.index(e.surprise) if e.surprise in sur_keys else 2,
                key=f"sur_{e.id}",
            )
            sco = st.selectbox(
                "影响范围",
                sco_keys,
                index=sco_keys.index(e.scope) if e.scope in sco_keys else 1,
                key=f"sco_{e.id}",
            )
            age = st.number_input("信息年龄（日）", 0.0, 60.0, 2.0, 0.5, key=f"age_{e.id}")
            unpriced = st.slider("未定价", 0.0, 1.0, float(e.unpriced), 0.05, key=f"up_{e.id}")
            manual_s = st.slider(
                "手动 strength（关闭自动时用）",
                0.0,
                3.0,
                float(e.strength),
                0.1,
                key=f"ms_{e.id}",
            )

            if not enabled:
                continue

            if auto:
                scored = score_strength(
                    StrengthInputs(
                        source_tier=src,
                        surprise=sur,
                        scope=sco,
                        age_days=age,
                        category=e.category,
                        unpriced_hint=unpriced,
                    )
                )
            else:
                scored = score_strength(
                    StrengthInputs(
                        strength_override=manual_s,
                        age_days=age,
                        category=e.category,
                        unpriced_hint=unpriced,
                    )
                )

            st.caption(
                f"→ {label_strength(scored.strength)} strength={scored.strength:.2f} "
                f"freshness={scored.freshness:.2f} | {scored.breakdown}"
            )
            evidence.append(
                EvidenceItem(
                    id=e.id,
                    title=e.title,
                    direction=int(direction),
                    strength=scored.strength,
                    freshness=scored.freshness,
                    unpriced=scored.unpriced,
                    category=e.category,
                    note=e.note,
                    strength_label=label_strength(scored.strength),
                    strength_breakdown=scored.breakdown,
                    source_tier=src,
                    surprise=sur,
                    scope=sco,
                )
            )

    return ModelWeights(
        n_sims=int(n_sims),
        seed=int(seed),
        trading_days=int(trading_days),
        vol_lookback_days=int(vol_lookback),
        use_relative_buckets=use_rel,
        bucket_pct_cuts=cuts,  # type: ignore[arg-type]
        bucket_edges=base.bucket_edges,
        score_to_mu_a=float(a),
        score_to_sigma_b=float(b),
        scenario_temperature=float(temperature),
        max_scenario_shift=float(max_shift),
        evidence_logit_scale=float(logit_scale),
        scenarios=scenarios,
        evidence=evidence,
    ), {
        "use_news": use_news,
        "keep_templates": keep_templates,
        "max_news_ev": int(max_news_ev),
        "classify_mode": classify_mode,
        "fetch_fulltext": fetch_fulltext,
        "llm_key": llm_key.strip(),
        "llm_base": llm_base.strip(),
        "llm_model": llm_model.strip(),
    }


@st.cache_data(ttl=300, show_spinner="抓取头条…")
def cached_headlines(pair: str, max_items: int = 25):
    return fetch_headlines_for_pair(pair, max_items=max_items)


@st.cache_data(ttl=300, show_spinner="抓取行情…")
def cached_fetch(
    pair: str,
    ticker: str,
    invert: bool,
    lookback: int,
    cuts: tuple,
    fallbacks: tuple = (),
    spot_ticker: str | None = None,
):
    from pairs import PairSpec

    parts = pair.split("/")
    spec = PairSpec(
        pair=pair,
        yahoo_ticker=ticker,
        invert=invert,
        base=parts[0],
        quote=parts[1],
        description=pair,
        bucket_pct_cuts=cuts,
        fallback_tickers=fallbacks,
        spot_ticker=spot_ticker,
    )
    return fetch_market(spec, lookback_days=lookback)


def main() -> None:
    spec = pick_pair_spec()

    # Reset evidence defaults when pair changes
    if st.session_state.get("pair_key") != spec.pair:
        st.session_state["pair_key"] = spec.pair
        st.session_state.pop("last_report", None)

    base = default_weights(spec)
    weights, news_opts = sidebar_weights(base, spec.pair)

    st.title(f"{spec.pair} · 最高日高蒙特卡洛情报报告")
    st.caption(spec.description + "｜可抓取头条自动填证据；侧栏含强弱判定规则。")

    c1, c2, c3 = st.columns(3)
    with c1:
        start = st.date_input("窗口起点", value=date.today())
    with c2:
        end = st.date_input("窗口终点", value=date.today() + timedelta(days=92))
    with c3:
        run = st.button("抓取并运行蒙特卡洛", type="primary", use_container_width=True)

    if not run and "last_report" not in st.session_state:
        st.info("选择货币对后点击运行：将抓行情 + 头条并自动填证据。")
        st.markdown(rubric_markdown())
        st.markdown("**已支持目录：** " + ", ".join(list_pairs()))
        return

    if run:
        with st.spinner("抓取行情 / 头条并运行蒙特卡洛…"):
            market = cached_fetch(
                spec.pair,
                spec.yahoo_ticker,
                spec.invert,
                weights.vol_lookback_days,
                weights.bucket_pct_cuts,
                tuple(getattr(spec, "fallback_tickers", ()) or ()),
                getattr(spec, "spot_ticker", None),
            )
            if market.notes:
                st.warning(" / ".join(market.notes))

            headlines = []
            auto_evidence = []
            news_meta = {}
            if news_opts["use_news"]:
                headlines = cached_headlines(spec.pair, max_items=30)
                suggested_up = calibrate_unpriced_from_market(market.ret_1d, market.ret_5d)

                # Resolve API key: sidebar > env > streamlit secrets
                key = news_opts.get("llm_key") or ""
                if not key:
                    try:
                        key = str(st.secrets.get("LLM_API_KEY") or st.secrets.get("OPENAI_API_KEY") or "")
                    except Exception:
                        key = ""
                base_url = news_opts.get("llm_base") or None
                model = news_opts.get("llm_model") or None
                if not base_url:
                    try:
                        base_url = st.secrets.get("LLM_BASE_URL") or st.secrets.get("OPENAI_BASE_URL")
                    except Exception:
                        base_url = None
                if not model:
                    try:
                        model = st.secrets.get("LLM_MODEL") or st.secrets.get("OPENAI_MODEL")
                    except Exception:
                        model = None

                llm_cfg = resolve_llm_config(api_key=key or None, base_url=base_url, model=model)
                mode = news_opts.get("classify_mode") or "hybrid"
                if mode in {"llm", "hybrid"} and llm_cfg is None:
                    st.warning("未配置 LLM API Key，已用关键词规则填证据。可在侧栏粘贴 Key，或设环境变量 OPENAI_API_KEY / LLM_API_KEY。")
                    mode = "rules"

                auto_evidence, news_meta = build_evidence_from_news(
                    headlines,
                    spec,
                    mode=mode,  # type: ignore[arg-type]
                    max_items=news_opts["max_news_ev"],
                    unpriced_cap=suggested_up,
                    llm_cfg=llm_cfg,
                    fetch_fulltext=bool(news_opts.get("fetch_fulltext", True)),
                )
                if news_meta.get("llm") and news_meta["llm"].get("error"):
                    st.warning(f"大模型调用失败，已回退规则：{news_meta['llm']['error'][:200]}")
                elif news_meta.get("llm") and not news_meta.get("rules_used"):
                    st.success(
                        f"大模型精读完成（{news_meta['llm'].get('model')}），"
                        f"采用 {len(auto_evidence)} 条证据"
                    )

                if news_opts["keep_templates"]:
                    weights.evidence = auto_evidence + list(weights.evidence)
                else:
                    weights.evidence = auto_evidence or list(weights.evidence)
                if not auto_evidence:
                    st.info("未能生成头条证据；已回退到模板证据。")
            else:
                suggested_up = calibrate_unpriced_from_market(market.ret_1d, market.ret_5d)
                for e in weights.evidence:
                    e.unpriced = min(e.unpriced, suggested_up)

            # Apply priced-in cap again
            suggested_up = calibrate_unpriced_from_market(market.ret_1d, market.ret_5d)
            for e in weights.evidence:
                e.unpriced = min(e.unpriced, suggested_up)

            score = evidence_score(weights.evidence)
            mu_shift = weights.score_to_mu_a * score
            sigma_extra = 1.0 + weights.score_to_sigma_b * abs(score)
            if (
                market.sigma_20d_ann
                and market.sigma_60d_ann
                and market.sigma_60d_ann > 0
                and market.sigma_20d_ann / market.sigma_60d_ann > 1.25
            ):
                sigma_extra *= 1.08
            scenarios_adj = apply_evidence_to_scenarios(
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
                scenarios=scenarios_adj,
                trading_days=weights.trading_days,
                n_sims=weights.n_sims,
                seed=weights.seed,
                bucket_edges=edges,
                mu_annual_shift=mu_shift,
                sigma_mult_extra=sigma_extra,
            )
            probs = enforce_math_floor(mc.raw_probs, market.spot, edges)
            report = build_report_markdown(
                market,
                weights,
                scenarios_adj,
                mc,
                probs,
                score=score,
                mu_shift=mu_shift,
                sigma_extra=sigma_extra,
                horizon_label=_horizon_label(start, end),
                bucket_edges=edges,
            )
            diag = build_diagnostics(
                market, weights, scenarios_adj, mc, probs, score, mu_shift, sigma_extra, edges
            )
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
            diag["auto_evidence_count"] = len(auto_evidence)
            diag["news_meta"] = news_meta
            st.session_state["last_report"] = report
            st.session_state["last_diag"] = diag
            st.session_state["last_probs"] = probs
            st.session_state["last_headlines"] = diag["headlines"]
            st.session_state["last_news_meta"] = news_meta
            st.session_state["last_auto_evidence"] = [
                {
                    "id": e.id,
                    "title": e.title,
                    "dir": e.direction,
                    "label": e.strength_label,
                    "strength": e.strength,
                    "category": e.category,
                    "source_tier": e.source_tier,
                    "surprise": e.surprise,
                    "scope": e.scope,
                }
                for e in weights.evidence
            ]

    report = st.session_state["last_report"]
    diag = st.session_state["last_diag"]
    probs = st.session_state["last_probs"]

    k1, k2, k3, k4 = st.columns(4)
    top = max(probs, key=probs.get)
    k1.metric("货币对", diag["market"]["pair"])
    k2.metric("最可能档", top)
    k3.metric("概率", f"{100 * probs[top]:.1f}%")
    k4.metric("证据分 S", f"{diag['score_S']:+.2f}")

    st.subheader("分档概率")
    st.bar_chart(pd.DataFrame({"区间": list(probs), "概率": list(probs.values())}).set_index("区间"))

    if st.session_state.get("last_auto_evidence"):
        st.subheader("自动填入的证据（来自头条）")
        st.caption(f"采用 {diag.get('auto_evidence_count', len(st.session_state['last_auto_evidence']))} 条可判定方向的头条")
        st.dataframe(pd.DataFrame(st.session_state["last_auto_evidence"]), use_container_width=True)
    if st.session_state.get("last_headlines"):
        with st.expander(f"原始头条（{len(st.session_state['last_headlines'])}）", expanded=False):
            st.dataframe(pd.DataFrame(st.session_state["last_headlines"]), use_container_width=True)

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown(report)
        st.download_button(
            "下载 Markdown",
            report.encode("utf-8"),
            file_name=f"{diag['market']['pair'].replace('/', '')}_mc_report.md",
            mime="text/markdown",
        )
    with right:
        st.subheader("隐藏参数快照")
        st.json(diag["mapping"])
        st.markdown("**校准后情景**")
        st.dataframe(pd.DataFrame(diag["scenarios_adjusted"]), use_container_width=True)
        st.markdown("**原始 vs 校准**")
        cmp = pd.DataFrame({"原始MC": diag["raw_probs"], "校准后": diag["calibrated_probs"]})
        st.dataframe(cmp.map(lambda x: f"{100 * x:.1f}%"), use_container_width=True)
        st.download_button(
            "下载诊断 JSON",
            json.dumps(diag, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="diagnostics.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
