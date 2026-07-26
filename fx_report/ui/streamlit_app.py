"""
Multi-pair FX peak-bucket forecaster — Streamlit UI（精简折叠版）。

主区：选看涨 → 立刻看现价 → 自设分档 → 运行 → 看概率。
侧栏 / 折叠：API、蒙特卡洛次数、高级权重、规则说明、诊断明细。
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from fx_report.config.api_config import status_text
from fx_report.ui.api_panel import render_api_settings_panel
from fx_report.news.fetch import fetch_status_summary
from fx_report.market.fetch_data import fetch_market
from fx_report.market.pairs import (
    PairSpec,
    edges_from_spot,
    get_pair,
    list_pairs,
    make_custom_pair,
    resolve_pair_for_bullish,
)
from fx_report.pipeline import run_pipeline, step2_assess_info_needs
from fx_report.report.text import rubric_markdown
from fx_report.model.monte_carlo import bucket_labels_from_edges
from fx_report.model.strength import (
    SOURCE_TIER_POINTS,
    SURPRISE_POINTS,
    SCOPE_POINTS,
    StrengthInputs,
    label_strength,
    score_strength,
)
from fx_report.model.weights import (
    EvidenceItem,
    ModelWeights,
    ScenarioSpec,
    default_weights,
)

try:
    from fx_report.news.llm import resolve_llm_config
except Exception:  # pragma: no cover

    def resolve_llm_config(*, api_key=None, base_url=None, model=None, allow_ollama_auto=True):
        try:
            from fx_report.news.llm import resolve_llm_config as _resolve

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


def _horizon(start: date, end: date) -> str:
    return f"{start.isoformat()} → {end.isoformat()}"


def _fmt_px(x: float) -> str:
    ax = abs(x)
    if ax >= 100:
        return f"{x:.3f}"
    if ax >= 10:
        return f"{x:.4f}"
    return f"{x:.5f}"


def _spot_cache_key(pair: str) -> str:
    return f"spot_cache::{pair}"


def load_spot_for_pair(
    spec: PairSpec,
    *,
    lookback_days: int = 60,
    force: bool = False,
) -> dict:
    """Fetch analysis-quote spot without running the full MC/news pipeline."""
    key = _spot_cache_key(spec.pair)
    cached = st.session_state.get(key)
    if cached and not force and cached.get("ok") is True:
        return cached
    try:
        snap = fetch_market(spec, lookback_days=lookback_days)
        row = {
            "ok": True,
            "pair": snap.pair,
            "spot": float(snap.spot),
            "source": snap.source,
            "asof": snap.asof,
            "notes": list(snap.notes[:3]),
            "error": None,
        }
    except Exception as e:
        row = {
            "ok": False,
            "pair": spec.pair,
            "spot": None,
            "source": None,
            "asof": None,
            "notes": [],
            "error": str(e),
        }
    st.session_state[key] = row
    return row


def render_spot_panel(spec: PairSpec, bullish: str, lookback_days: int) -> dict:
    """Prominent spot after pair + bullish are chosen."""
    st.subheader("现价（分析报价）")
    c_a, c_b = st.columns([4, 1])
    with c_b:
        refresh = st.button("刷新现价", use_container_width=True, key="refresh_spot")
    spot_row = load_spot_for_pair(spec, lookback_days=lookback_days, force=refresh)
    with c_a:
        if spot_row["ok"]:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("分析口径", spot_row["pair"])
            m2.metric("现价", _fmt_px(spot_row["spot"]))
            m3.metric("来源", spot_row["source"] or "—")
            m4.metric("asof", spot_row["asof"] or "—")
            st.caption(
                f"看涨 {bullish} → 分析报价升高表示 {bullish} 走强。"
                + (" ｜ " + "｜".join(spot_row["notes"]) if spot_row["notes"] else "")
            )
        else:
            st.error(
                "无法获取现价，请检查网络或 API Key 后点「刷新现价」。\n\n"
                f"{spot_row.get('error') or '未知错误'}"
            )
    return spot_row


def render_bucket_editor(
    base: ModelWeights,
    spot: float | None,
    analysis_pair: str,
) -> tuple[bool, tuple[float, float, float, float], tuple[float, float, float, float]]:
    """
    Main-area bucket edges. 4 cut points → 5 Torchcast-style buckets.
    Returns (use_relative, pct_cuts, abs_edges).
    """
    st.subheader("概率区间（自己设边界）")
    st.caption(
        "4 个边界 → 5 个区间（与 Torchcast 一致）："
        "`< e1` · `e1–e2` · `e2–e3` · `e3–e4` · `≥ e4`。"
        "运行分析后，蒙特卡洛概率与 PDF 都用这套边界。"
    )

    mode_key = f"bucket_mode::{analysis_pair}"
    pct_key = f"bucket_pct::{analysis_pair}"
    abs_key = f"bucket_abs::{analysis_pair}"
    seeded_key = f"abs_seeded_from_spot::{analysis_pair}"

    if pct_key not in st.session_state:
        st.session_state[pct_key] = [float(x) for x in base.bucket_pct_cuts]

    if abs_key not in st.session_state:
        if spot is not None:
            st.session_state[abs_key] = [
                float(x) for x in edges_from_spot(spot, tuple(st.session_state[pct_key]))
            ]
            st.session_state[seeded_key] = True
        else:
            st.session_state[abs_key] = [float(x) for x in base.bucket_edges]
    elif spot is not None and not st.session_state.get(seeded_key):
        # First successful spot after placeholder abs edges
        st.session_state[abs_key] = [
            float(x) for x in edges_from_spot(spot, tuple(st.session_state[pct_key]))
        ]
        st.session_state[seeded_key] = True
        for i, e in enumerate(st.session_state[abs_key]):
            st.session_state[f"abs_cut_{analysis_pair}_{i}"] = float(e)

    for i, v in enumerate(st.session_state[pct_key]):
        wk = f"pct_cut_{analysis_pair}_{i}"
        if wk not in st.session_state:
            st.session_state[wk] = float(v)
    for i, v in enumerate(st.session_state[abs_key]):
        wk = f"abs_cut_{analysis_pair}_{i}"
        if wk not in st.session_state:
            st.session_state[wk] = float(v)

    # Migrate renamed radio options so old sessions don't crash
    _old_mode = st.session_state.get(mode_key)
    if _old_mode == "相对现价 %":
        st.session_state[mode_key] = "相对现价"
    elif _old_mode == "绝对水平":
        st.session_state[mode_key] = "绝对价位"

    mode = st.radio(
        "边界方式",
        ["相对现价", "绝对价位"],
        horizontal=True,
        key=mode_key,
        help="相对：边界 = 现价 × (1 + 涨幅%/100)。绝对：直接填汇率价位。",
    )
    use_rel = mode == "相对现价"

    cols = st.columns(4)
    if use_rel:
        pcts: list[float] = []
        for i, col in enumerate(cols):
            with col:
                pcts.append(
                    float(
                        st.number_input(
                            f"相对涨幅 {i + 1}（相对现价 +%）",
                            min_value=-20.0,
                            max_value=50.0,
                            step=0.5,
                            key=f"pct_cut_{analysis_pair}_{i}",
                            help="填上涨百分比，不是汇率本身。例：现价 1.43、填 2 → 边界 ≈ 1.4586。",
                        )
                    )
                )
        st.caption(
            "填的是相对现价的上涨百分比；例如现价 1.43、填 2 → 边界≈1.4586；不是直接填汇率。"
        )
        pct_cuts = tuple(sorted(pcts))  # type: ignore[assignment]
        st.session_state[pct_key] = list(pct_cuts)
        if spot is not None:
            abs_edges = edges_from_spot(spot, pct_cuts)
            st.session_state[abs_key] = list(abs_edges)
            for i, e in enumerate(abs_edges):
                st.session_state[f"abs_cut_{analysis_pair}_{i}"] = float(e)
        else:
            abs_edges = tuple(float(x) for x in st.session_state[abs_key])  # type: ignore[assignment]
    else:
        abss: list[float] = []
        step = 0.0001 if (spot is not None and spot < 10) else 0.01
        for i, col in enumerate(cols):
            with col:
                abss.append(
                    float(
                        st.number_input(
                            f"汇率边界 {i + 1}",
                            min_value=0.0,
                            max_value=1_000_000.0,
                            step=step,
                            format="%.5f",
                            key=f"abs_cut_{analysis_pair}_{i}",
                            help="直接填分析报价的绝对价位（汇率水平）。",
                        )
                    )
                )
        st.caption("填的是分析报价的绝对汇率价位（不是百分比）。")
        abs_edges = tuple(sorted(abss))  # type: ignore[assignment]
        st.session_state[abs_key] = list(abs_edges)
        if spot is not None and spot > 0:
            pct_cuts = tuple(  # type: ignore[assignment]
                sorted((e / spot - 1.0) * 100.0 for e in abs_edges)
            )
            st.session_state[pct_key] = list(pct_cuts)
            for i, p in enumerate(pct_cuts):
                st.session_state[f"pct_cut_{analysis_pair}_{i}"] = float(p)
        else:
            pct_cuts = tuple(float(x) for x in st.session_state[pct_key])  # type: ignore[assignment]

    preview_edges = (
        edges_from_spot(spot, pct_cuts)
        if use_rel and spot is not None
        else abs_edges
    )
    if use_rel and spot is None:
        st.info("将形成 5 档（需现价后换算绝对价位）：" + " · ".join(bucket_labels_from_edges(preview_edges)))
        st.caption("相对现价模式需先成功拉取现价，才能换算绝对边界并跑蒙特卡洛。")
    else:
        e = list(preview_edges)
        live = (
            f"< {_fmt_px(e[0])} | "
            + " | ".join(f"{_fmt_px(e[i])}–{_fmt_px(e[i + 1])}" for i in range(len(e) - 1))
            + f" | ≥ {_fmt_px(e[-1])}"
        )
        spot_note = f"（现价 {_fmt_px(spot)}）" if spot is not None else ""
        st.info(f"5 档预览：{live}{spot_note}")

    return use_rel, pct_cuts, abs_edges  # type: ignore[return-value]


def pick_pair_in_sidebar():
    """侧栏分区 1：货币对 + 看涨货币（默认可折叠，首次展开）。"""
    with st.sidebar.expander("① 货币对", expanded=True):
        mode = st.radio("方式", ["目录", "自定义"], horizontal=True, key="pair_mode")
        if mode == "目录":
            pair = st.selectbox("选择", list_pairs(), index=list_pairs().index("USD/AUD"))
            spec = get_pair(pair)
        else:
            pair = st.text_input("BASE/QUOTE", value="EUR/USD")
            ticker = st.text_input("内部符号", value="EURUSD")
            invert = st.checkbox("invert", value=False)
            spec = make_custom_pair(pair, ticker, invert)

        # 看涨货币必选：未选时 index=None，禁止静默开跑
        if st.session_state.get("bullish_pair_key") != spec.pair:
            st.session_state["bullish_pair_key"] = spec.pair
            st.session_state.pop("bullish_ccy", None)

        bullish = st.radio(
            "看涨货币（必选）",
            [spec.base, spec.quote],
            index=None,
            horizontal=True,
            key="bullish_ccy",
            help="看涨币走强 = 分析报价升高。选 quote 时自动翻转分析口径。",
        )
        return spec, bullish

def sidebar_weights(base: ModelWeights, pair_name: str) -> tuple[ModelWeights, dict]:
    st.sidebar.markdown(f"**{pair_name}**")
    st.sidebar.caption(
        "侧栏目录（点开分区）：①货币对 → ②抓取 → ③蒙特卡洛 → "
        "④映射 → ⑤情景 → ⑥证据 → ⑦规则 → ⑧数据源｜分档切点在主区"
    )

    # ② 抓取
    with st.sidebar.expander("② 抓取与判定", expanded=False):
        use_news = st.checkbox("官方 / vault 头条", value=True)
        ai_research = st.checkbox("AI 检索员", value=True)
        classify_mode = st.selectbox(
            "证据判定",
            ["hybrid", "llm", "rules"],
            index=0,
            help="hybrid=LLM优先；rules=仅关键词",
        )
        keep_templates = st.checkbox("保留模板证据", value=False)
        max_news_ev = st.slider("最多头条证据条数", 3, 20, 10, 1)
        fetch_fulltext = st.checkbox("抓正文供 LLM", value=True)

    # ③ 蒙特卡洛（分档切点在主区设置）
    with st.sidebar.expander("③ 蒙特卡洛", expanded=False):
        n_sims = st.number_input("蒙特卡洛次数", 10_000, 500_000, base.n_sims, 10_000)
        trading_days = st.number_input("交易日窗口", 5, 252, base.trading_days, 1)
        seed = st.number_input("随机种子", 0, 10_000_000, base.seed, 1)
        vol_lookback = st.number_input("波动回看日", 20, 252, base.vol_lookback_days, 5)
        st.caption("分档边界请在主区「概率区间」设置（相对现价涨幅% 或绝对汇率价位）。")
        use_rel = True
        cuts = tuple(base.bucket_pct_cuts)

    # ④ 映射
    with st.sidebar.expander("④ 证据 → 参数映射", expanded=False):
        a = st.slider("a：S→漂移", 0.0, 0.05, float(base.score_to_mu_a), 0.001, format="%.3f")
        b = st.slider("b：|S|→波动", 0.0, 0.15, float(base.score_to_sigma_b), 0.001, format="%.3f")
        logit_scale = st.slider("证据→情景 logit", 0.0, 0.3, float(base.evidence_logit_scale), 0.01)
        temperature = st.slider("情景温度", 0.3, 2.5, float(base.scenario_temperature), 0.1)
        max_shift = st.slider("单情景最大位移", 0.0, 0.4, float(base.max_scenario_shift), 0.01)

    # ⑤ 情景：用下拉选一个，避免一长串滑块
    with st.sidebar.expander("⑤ 情景先验", expanded=False):
        sc_names = [sc.name for sc in base.scenarios]
        focus = st.selectbox("编辑哪个情景", sc_names, key="sc_focus")
        # Keep all scenario params in session so unfocused ones persist
        if "scenario_edits" not in st.session_state:
            st.session_state["scenario_edits"] = {
                sc.name: {
                    "weight": float(sc.weight),
                    "mu": float(sc.mu_annual),
                    "sm": float(sc.sigma_mult),
                    "ej": float(sc.expected_jumps),
                    "jm": float(sc.jump_mean),
                    "js": float(sc.jump_std),
                    "narrative": sc.narrative,
                }
                for sc in base.scenarios
            }
        # reset if pair scenarios set changes
        for sc in base.scenarios:
            if sc.name not in st.session_state["scenario_edits"]:
                st.session_state["scenario_edits"][sc.name] = {
                    "weight": float(sc.weight),
                    "mu": float(sc.mu_annual),
                    "sm": float(sc.sigma_mult),
                    "ej": float(sc.expected_jumps),
                    "jm": float(sc.jump_mean),
                    "js": float(sc.jump_std),
                    "narrative": sc.narrative,
                }
        cur = st.session_state["scenario_edits"][focus]
        cur["weight"] = st.slider("权重", 0.0, 1.0, cur["weight"], 0.01, key=f"w_{focus}")
        cur["mu"] = st.slider("μ", -0.2, 0.2, cur["mu"], 0.005, key=f"mu_{focus}")
        cur["sm"] = st.slider("σ×", 0.5, 2.5, cur["sm"], 0.05, key=f"sm_{focus}")
        cur["ej"] = st.slider("E[jumps]", 0.0, 3.0, cur["ej"], 0.05, key=f"ej_{focus}")
        cur["jm"] = st.slider("jump μ", -0.03, 0.03, cur["jm"], 0.001, key=f"jm_{focus}")
        cur["js"] = st.slider("jump σ", 0.001, 0.03, cur["js"], 0.001, key=f"js_{focus}")
        st.session_state["scenario_edits"][focus] = cur
        scenarios = [
            ScenarioSpec(
                name,
                ed["weight"],
                ed["mu"],
                ed["sm"],
                ed["ej"],
                ed["jm"],
                ed["js"],
                ed["narrative"],
            )
            for name, ed in st.session_state["scenario_edits"].items()
            if name in sc_names
        ]

    # ⑥ 证据：下拉选一条再编辑（不再套娃 expander）
    with st.sidebar.expander("⑥ 模板证据计分卡", expanded=False):
        tier_keys = list(SOURCE_TIER_POINTS.keys())
        sur_keys = list(SURPRISE_POINTS.keys())
        sco_keys = list(SCOPE_POINTS.keys())
        ev_map = {e.id: e for e in base.evidence}
        ev_ids = list(ev_map.keys())
        # session store for edits
        if "evidence_edits" not in st.session_state or st.session_state.get("ev_pair") != pair_name:
            st.session_state["ev_pair"] = pair_name
            st.session_state["evidence_edits"] = {
                e.id: {
                    "enabled": True,
                    "direction": int(e.direction),
                    "auto": True,
                    "src": e.source_tier if e.source_tier in tier_keys else tier_keys[2],
                    "sur": e.surprise if e.surprise in sur_keys else sur_keys[2],
                    "sco": e.scope if e.scope in sco_keys else sco_keys[1],
                    "age": 2.0,
                    "unpriced": float(e.unpriced),
                    "manual_s": float(e.strength),
                }
                for e in base.evidence
            }
        pick = st.selectbox("选择证据条目", ev_ids, format_func=lambda i: f"{i} · {ev_map[i].title[:28]}")
        ed = st.session_state["evidence_edits"][pick]
        e0 = ev_map[pick]
        ed["enabled"] = st.checkbox("启用", value=ed["enabled"], key=f"en_{pick}")
        ed["direction"] = st.selectbox(
            "方向 +1推高 / -1压制",
            [1, -1],
            index=0 if ed["direction"] > 0 else 1,
            key=f"dir_{pick}",
        )
        ed["auto"] = st.checkbox("自动 strength", value=ed["auto"], key=f"auto_{pick}")
        ed["src"] = st.selectbox(
            "来源档", tier_keys, index=tier_keys.index(ed["src"]), key=f"src_{pick}"
        )
        ed["sur"] = st.selectbox(
            "意外", sur_keys, index=sur_keys.index(ed["sur"]), key=f"sur_{pick}"
        )
        ed["sco"] = st.selectbox(
            "范围", sco_keys, index=sco_keys.index(ed["sco"]), key=f"sco_{pick}"
        )
        ed["age"] = st.number_input("年龄（日）", 0.0, 60.0, float(ed["age"]), 0.5, key=f"age_{pick}")
        ed["unpriced"] = st.slider("未定价", 0.0, 1.0, float(ed["unpriced"]), 0.05, key=f"up_{pick}")
        ed["manual_s"] = st.slider(
            "手动 strength", 0.0, 3.0, float(ed["manual_s"]), 0.1, key=f"ms_{pick}"
        )
        st.session_state["evidence_edits"][pick] = ed

        evidence: list[EvidenceItem] = []
        for eid, e in ev_map.items():
            row = st.session_state["evidence_edits"].get(eid)
            if not row or not row["enabled"]:
                continue
            if row["auto"]:
                scored = score_strength(
                    StrengthInputs(
                        source_tier=row["src"],
                        surprise=row["sur"],
                        scope=row["sco"],
                        age_days=row["age"],
                        category=e.category,
                        unpriced_hint=row["unpriced"],
                    )
                )
            else:
                scored = score_strength(
                    StrengthInputs(
                        strength_override=row["manual_s"],
                        age_days=row["age"],
                        category=e.category,
                        unpriced_hint=row["unpriced"],
                    )
                )
            if eid == pick:
                st.caption(f"当前 → {label_strength(scored.strength)} · {scored.strength:.2f}")
            evidence.append(
                EvidenceItem(
                    id=e.id,
                    title=e.title,
                    direction=int(row["direction"]),
                    strength=scored.strength,
                    freshness=scored.freshness,
                    unpriced=scored.unpriced,
                    category=e.category,
                    note=e.note,
                    strength_label=label_strength(scored.strength),
                    strength_breakdown=scored.breakdown,
                    source_tier=row["src"],
                    surprise=row["sur"],
                    scope=row["sco"],
                )
            )

    # ⑦ 规则
    with st.sidebar.expander("⑦ 强弱判定规则", expanded=False):
        st.markdown(
            "`contrib = dir × strength × freshness × unpriced`  \n"
            "≤1 SLIGHT｜≤2 MODERATE｜>2 STRONG"
        )
        st.dataframe(
            pd.DataFrame({"档": list(SOURCE_TIER_POINTS), "分": list(SOURCE_TIER_POINTS.values())}),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown(rubric_markdown())

    # ⑧ 数据源状态
    with st.sidebar.expander("⑧ 数据源状态", expanded=False):
        st.code(fetch_status_summary(), language=None)
        st.text(status_text())

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
        "ai_research": ai_research,
        "keep_templates": keep_templates,
        "max_news_ev": int(max_news_ev),
        "classify_mode": classify_mode,
        "fetch_fulltext": fetch_fulltext,
        "llm_key": "",
        "llm_base": "",
        "llm_model": "",
    }


def main() -> None:
    display_spec, bullish = pick_pair_in_sidebar()

    if st.session_state.get("pair_key") != display_spec.pair:
        st.session_state["pair_key"] = display_spec.pair
        st.session_state.pop("last_report", None)
        st.session_state.pop("scenario_edits", None)
        st.session_state.pop("evidence_edits", None)

    bullish_ok = bullish in (display_spec.base, display_spec.quote)
    analysis_spec = (
        resolve_pair_for_bullish(display_spec, bullish) if bullish_ok else display_spec
    )

    if bullish_ok and st.session_state.get("analysis_pair_key") != analysis_spec.pair:
        st.session_state["analysis_pair_key"] = analysis_spec.pair
        st.session_state.pop("last_report", None)

    base = default_weights(analysis_spec)
    weights, news_opts = sidebar_weights(base, analysis_spec.pair)

    st.title(f"{display_spec.pair}")
    if bullish_ok:
        st.caption(
            f"分析口径：{analysis_spec.pair}（看涨 {bullish}）· "
            "先看现价 → 自设概率区间 → 再运行蒙特卡洛"
        )
    else:
        st.caption("最高日高分档 · 七步情报流水线 · 请先在侧栏选择看涨货币")

    with st.expander("API / AI Key（按需填写，可全空）", expanded=False):
        api_opts = render_api_settings_panel()

    if not bullish_ok:
        st.warning("请先在侧栏 ① 选择「看涨货币」（二选一）。选好后立刻显示现价与分档设置。")
        if "last_report" not in st.session_state:
            return

    spot_row: dict | None = None
    if bullish_ok:
        spot_row = render_spot_panel(
            analysis_spec, bullish, lookback_days=weights.vol_lookback_days
        )
        spot_val = float(spot_row["spot"]) if spot_row.get("ok") and spot_row.get("spot") is not None else None
        use_rel, pct_cuts, abs_edges = render_bucket_editor(
            base, spot_val, analysis_spec.pair
        )
        weights.use_relative_buckets = use_rel
        weights.bucket_pct_cuts = pct_cuts  # type: ignore[assignment]
        weights.bucket_edges = abs_edges  # type: ignore[assignment]

        with st.expander(f"本对信息需求 · {analysis_spec.pair}", expanded=False):
            needs = step2_assess_info_needs(analysis_spec)
            st.dataframe(
                pd.DataFrame([n.to_dict() for n in needs]),
                hide_index=True,
                use_container_width=True,
            )

        st.markdown("---")
        c1, c2, c3 = st.columns([1, 1, 1.2])
        with c1:
            start = st.date_input("窗口起点", value=date.today())
        with c2:
            end = st.date_input("窗口终点", value=date.today() + timedelta(days=92))
        with c3:
            st.write("")
            can_run = bool(spot_row and spot_row.get("ok"))
            run = st.button(
                "运行分析",
                type="primary",
                use_container_width=True,
                disabled=not can_run,
                help=None if can_run else "需先成功获取现价",
            )
        if not can_run:
            st.caption("现价未就绪时不能运行分析（分档与 MC 都依赖分析报价）。")
        elif "last_report" not in st.session_state and not run:
            st.info("确认概率区间后点「运行分析」。侧栏 ②–⑧ 可调抓取与模型参数；API 可全空。")
    else:
        start = date.today()
        end = date.today() + timedelta(days=92)
        run = False

    if run:
        if not bullish_ok:
            st.warning("未选择看涨货币，无法运行分析。")
            return
        if not spot_row or not spot_row.get("ok"):
            st.error("现价获取失败，无法运行分析。请先刷新现价。")
            return
        with st.spinner("流水线运行中…"):
            key = (api_opts.get("llm_key") or "").strip()
            if not key:
                try:
                    key = str(st.secrets.get("LLM_API_KEY") or st.secrets.get("OPENAI_API_KEY") or "")
                except Exception:
                    key = ""
            base_url = (api_opts.get("llm_base") or "").strip() or None
            model = (api_opts.get("llm_model") or "").strip() or None
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
            mode_cls = news_opts.get("classify_mode") or "hybrid"
            if mode_cls in {"llm", "hybrid"} and llm_cfg is None:
                st.warning("未配置 LLM，步骤4改用关键词规则。")
                mode_cls = "rules"

            result = run_pipeline(
                display_spec.pair,
                ticker=None if display_spec.pair in list_pairs() else display_spec.symbol_code,
                invert=display_spec.invert,
                sims=weights.n_sims,
                days=weights.trading_days,
                seed=weights.seed,
                lookback=weights.vol_lookback_days,
                mode=mode_cls,  # type: ignore[arg-type]
                max_news=news_opts["max_news_ev"],
                keep_templates=news_opts["keep_templates"],
                no_news=not news_opts["use_news"],
                no_fulltext=not bool(news_opts.get("fetch_fulltext", True)),
                ai_research=bool(news_opts.get("ai_research", True)),
                llm_cfg=llm_cfg,
                out_dir="output",
                verbose=False,
                bullish_currency=bullish,
                model_weights=weights,
            )

            if result.market.notes:
                st.caption("｜".join(result.market.notes[:3]))

            # Keep spot cache in sync with pipeline market print
            st.session_state[_spot_cache_key(analysis_spec.pair)] = {
                "ok": True,
                "pair": result.market.pair,
                "spot": float(result.market.spot),
                "source": result.market.source,
                "asof": result.market.asof,
                "notes": list(result.market.notes[:3]),
                "error": None,
            }

            st.session_state["last_report"] = result.report_md
            st.session_state["last_report_html"] = result.report_html
            st.session_state["last_pdf_bytes"] = None
            try:
                from fx_report.report.torchcast import write_pdf
                import tempfile
                from pathlib import Path

                with tempfile.TemporaryDirectory() as td:
                    pdf_path = write_pdf(result.torchcast, Path(td) / "report.pdf")
                    st.session_state["last_pdf_bytes"] = pdf_path.read_bytes()
            except Exception as e:
                st.session_state["last_pdf_error"] = str(e)
            st.session_state["last_diag"] = result.diagnostics
            st.session_state["last_probs"] = result.probs
            st.session_state["last_edges"] = list(result.edges)
            st.session_state["last_headlines"] = result.diagnostics.get("headlines", [])
            st.session_state["last_news_meta"] = result.news_meta
            st.session_state["last_info_needs"] = [n.to_dict() for n in result.info_needs]
            st.session_state["last_statements"] = [s.to_dict() for s in result.statements]
            st.session_state["last_auto_evidence"] = [
                {
                    "id": w.evidence.id,
                    "title": w.evidence.title,
                    "dir": w.evidence.direction,
                    "label": w.evidence.strength_label,
                    "strength": w.evidence.strength,
                    "category": w.evidence.category,
                    "contrib": w.weight_contrib,
                }
                for w in result.weighted
            ]
            st.session_state["last_stage_log"] = result.stage_log
            st.session_state["last_source"] = result.market.source

    if "last_report" not in st.session_state:
        return

    report = st.session_state["last_report"]
    diag = st.session_state["last_diag"]
    probs = st.session_state["last_probs"]

    st.markdown("---")
    st.subheader("分析结果")
    k1, k2, k3, k4 = st.columns(4)
    top = max(probs, key=probs.get)
    k1.metric("货币对", diag["market"]["pair"])
    k2.metric("最可能档", top)
    k3.metric("概率", f"{100 * probs[top]:.1f}%")
    k4.metric("证据分 S", f"{diag['score_S']:+.2f}")
    src = st.session_state.get("last_source") or diag.get("market", {}).get("source", "")
    edges_used = st.session_state.get("last_edges") or diag.get("bucket_edges") or []
    edge_s = " / ".join(_fmt_px(float(x)) for x in edges_used) if edges_used else "—"
    st.caption(f"行情源 {src} · 窗口 {_horizon(start, end)} · 切点 {edge_s}")

    st.bar_chart(
        pd.DataFrame({"区间": list(probs), "概率": list(probs.values())}).set_index("区间")
    )

    with st.expander("完整报告（Torchcast 格式）", expanded=True):
        pdf_bytes = st.session_state.get("last_pdf_bytes")
        html_doc = st.session_state.get("last_report_html")
        c1, c2, c3 = st.columns(3)
        pair_safe = diag["market"]["pair"].replace("/", "")
        if pdf_bytes:
            c1.download_button(
                "下载 PDF（Torchcast）",
                pdf_bytes,
                file_name=f"{pair_safe}_torchcast.pdf",
                mime="application/pdf",
            )
        elif st.session_state.get("last_pdf_error"):
            c1.caption(f"PDF 生成失败：{st.session_state['last_pdf_error']}")
        if html_doc:
            c2.download_button(
                "下载 HTML",
                html_doc.encode("utf-8"),
                file_name=f"{pair_safe}_torchcast.html",
                mime="text/html",
            )
        c3.download_button(
            "下载 Markdown（调试）",
            report.encode("utf-8"),
            file_name=f"{pair_safe}_mc_report.md",
            mime="text/markdown",
        )
        if html_doc:
            st.components.v1.html(html_doc, height=900, scrolling=True)
        else:
            st.markdown(report)

    with st.expander("流水线明细（日志 / 语句 / 证据 / 头条）", expanded=False):
        if st.session_state.get("last_stage_log"):
            st.markdown("**七步日志**")
            st.code("\n".join(st.session_state["last_stage_log"]), language=None)
        if st.session_state.get("last_info_needs"):
            st.markdown("**信息需求**")
            st.dataframe(pd.DataFrame(st.session_state["last_info_needs"]), use_container_width=True)
        if st.session_state.get("last_statements"):
            st.markdown(f"**语句（{len(st.session_state['last_statements'])}）**")
            df_s = pd.DataFrame(st.session_state["last_statements"])
            cols = [c for c in ("id", "source", "statement", "provider") if c in df_s.columns]
            st.dataframe(df_s[cols], use_container_width=True)
        if st.session_state.get("last_auto_evidence"):
            st.markdown("**权重**")
            st.dataframe(
                pd.DataFrame(st.session_state["last_auto_evidence"]),
                use_container_width=True,
            )
        if st.session_state.get("last_headlines"):
            st.markdown("**头条**")
            st.dataframe(
                pd.DataFrame(st.session_state["last_headlines"]),
                use_container_width=True,
            )

    with st.expander("诊断 JSON / 情景 / 映射", expanded=False):
        st.dataframe(pd.DataFrame(diag["scenarios_adjusted"]), use_container_width=True)
        cmp = pd.DataFrame({"原始MC": diag["raw_probs"], "校准后": diag["calibrated_probs"]})
        st.dataframe(cmp.map(lambda x: f"{100 * x:.1f}%"), use_container_width=True)
        st.json(diag["mapping"])
        st.download_button(
            "下载诊断 JSON",
            json.dumps(diag, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="diagnostics.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
