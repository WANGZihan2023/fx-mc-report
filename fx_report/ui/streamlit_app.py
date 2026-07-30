"""
Multi-pair FX peak-bucket forecaster — Streamlit UI（精简折叠版）。

主区：选看涨 → 立刻看现价 → 自设分档 → 运行 → 看概率。
侧栏 / 折叠：API、蒙特卡洛次数、高级权重、规则说明、诊断明细。
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from fx_report.config.api_config import status_text
from fx_report.ui.api_panel import render_api_settings_panel
from fx_report.order_doc import (
    OrderDocParse,
    order_doc_from_dict,
    parse_order_document,
    preview_lines,
)
from fx_report.ui.i18n import (
    choice_placeholder,
    get_lang,
    init_language,
    render_language_selector,
    start_field_label,
    t,
)
from fx_report.ui.ux_helpers import (
    PCT_CUT_MAX,
    PCT_CUT_MIN,
    START_EXPERT_DIALOG_KEYS,
    START_SIMPLE_DIALOG_KEYS,
    app_password_expected,
    bb_jump_compensate_warning,
    format_cheap_historical_caption,
    format_missing_start_message,
    is_unset_choice,
    missing_start_choices,
    password_accepted,
    pct_cuts_in_bounds,
    resolve_replay_ai_research,
    seed_pct_widget_value,
    should_heal_floor_clamp,
)
from fx_report.model.algo_recommend import (
    AlgoRecommendation,
    format_recommend_audit_zh,
    is_simple_setup_mode,
    recommend_algorithms,
)
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
from fx_report.pipeline import (
    PipelineCheckpoint,
    run_pipeline,
    run_pipeline_phase_a,
    run_pipeline_phase_b,
    step2_assess_info_needs,
)
from fx_report.report.text import rubric_markdown
from fx_report.model.calibrate import (
    BUNDLED_CALIBRATED_DIR,
    calib_oos_board_dataframe,
    load_calib_oos_summary,
    resolve_calibrated_params_path,
)
from fx_report.model.monte_carlo import bucket_labels_from_edges
from fx_report.model.replay_backtest import run_replay_backtest
from fx_report.model.replay_summary import replay_summary_dataframe
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
    page_title="FX Analyse",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _app_password() -> str:
    """Product shared gate: APP_PASSWORD / FX_REPORT_PASSWORD, else default uniocean."""
    return app_password_expected()


def _store_pipeline_result(result) -> None:
    """Persist PipelineResult fields into session_state for the results panel."""
    st.session_state["last_report"] = result.report_md
    st.session_state["last_report_html"] = result.report_html
    st.session_state["last_pdf_bytes"] = None
    try:
        from fx_report.report.torchcast import write_pdf
        import tempfile

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
            "statement_id": w.evidence.statement_id or w.evidence.id,
            "title": w.evidence.title,
            "dir": w.evidence.direction,
            "direction": w.evidence.direction,
            "label": w.evidence.strength_label,
            "strength": w.evidence.strength,
            "freshness": w.evidence.freshness,
            "unpriced": w.evidence.unpriced,
            "category": w.evidence.category,
            "url": getattr(w.evidence, "url", "") or "",
            "is_prior": bool(getattr(w.evidence, "is_prior", False)),
            "contrib": w.weight_contrib,
        }
        for w in result.weighted
    ]
    st.session_state["last_base_scenarios"] = [
        {
            "name": s.name,
            "weight": s.weight,
            "mu_annual": s.mu_annual,
            "sigma_mult": s.sigma_mult,
            "expected_jumps": s.expected_jumps,
            "jump_mean": s.jump_mean,
            "jump_std": s.jump_std,
            "narrative": s.narrative,
        }
        for s in result.weights.scenarios
    ]
    st.session_state["last_weight_mapping"] = {
        "score_to_mu_a": float(result.weights.score_to_mu_a),
        "score_to_sigma_b": float(result.weights.score_to_sigma_b),
        "evidence_logit_scale": float(result.weights.evidence_logit_scale),
        "scenario_temperature": float(result.weights.scenario_temperature),
        "max_scenario_shift": float(result.weights.max_scenario_shift),
        "n_sims": int(result.weights.n_sims),
        "trading_days": int(result.weights.trading_days),
        "seed": int(result.weights.seed),
        "peak_engine": str(getattr(result.weights, "peak_engine", "path_max")),
        "jump_model": str(getattr(result.weights, "jump_model", "merton")),
        "jump_compensate": bool(getattr(result.weights, "jump_compensate", False)),
        "variance_reduction": str(getattr(result.mc, "variance_reduction", "none")),
    }
    st.session_state["label_audit_use_demo"] = False
    st.session_state.pop("label_edits", None)
    st.session_state.pop("label_edit_fp", None)
    st.session_state.pop("last_label_agree_stats", None)
    st.session_state.pop("human_label_recomputed", None)
    try:
        from fx_report.model.backtest import evidence_to_label_audit

        audit_df = evidence_to_label_audit(st.session_state["last_auto_evidence"])
        st.session_state["last_label_audit_csv"] = audit_df.to_csv(index=False)
    except Exception:
        st.session_state["last_label_audit_csv"] = None
    st.session_state["last_stage_log"] = result.stage_log
    st.session_state["last_source"] = result.market.source
    # Clear HITL mid-flight once finished
    st.session_state.pop("hitl_checkpoint", None)
    st.session_state.pop("hitl_choices", None)


def render_hitl_uncertain_form() -> bool:
    """
    Blocking section before final results when Phase A left pending reviews.
    Returns True if Phase B just completed (caller should not return early).
    """
    from fx_report.model.human_review import direction_zh

    raw_cp = st.session_state.get("hitl_checkpoint")
    if not raw_cp:
        return False

    try:
        cp = (
            PipelineCheckpoint.from_session_dict(raw_cp)
            if isinstance(raw_cp, dict)
            else raw_cp
        )
    except Exception as exc:
        st.error(f"无法恢复待确认检查点：{exc}")
        if st.button("丢弃检查点并重新运行", key="hitl_drop_cp"):
            st.session_state.pop("hitl_checkpoint", None)
            st.rerun()
        return False

    pending = list(cp.pending_reviews or [])
    if not pending:
        st.session_state.pop("hitl_checkpoint", None)
        return False

    st.markdown("---")
    st.subheader("人工确认不确定证据")
    st.caption(
        f"分析口径 **{cp.pair}**｜看涨 **{cp.bullish_currency}**｜"
        f"共 {len(pending)} 条待确认（赋权与蒙特卡洛前）。"
        "刷新页面会尽量从会话恢复此表单。"
    )
    st.info(
        "请为每条选择 **利多 / 利空 / 中性 / 跳过**。"
        "「跳过」保留模型原方向。提交后继续计算证据分 S 与报告。"
    )

    if "hitl_choices" not in st.session_state:
        st.session_state["hitl_choices"] = {}

    choice_opts = ["利多", "利空", "中性", "跳过"]
    choice_map = {"利多": "up", "利空": "down", "中性": "neutral", "跳过": "skip"}
    rev_map = {v: k for k, v in choice_map.items()}

    with st.form("hitl_uncertain_form", clear_on_submit=False):
        picks: dict[str, str] = {}
        for i, p in enumerate(pending):
            eid = p.evidence_id if hasattr(p, "evidence_id") else p.get("evidence_id")
            title = p.title if hasattr(p, "title") else p.get("title", "")
            snippet = p.snippet if hasattr(p, "snippet") else p.get("snippet", title)
            reasons_zh = (
                p.reasons_zh if hasattr(p, "reasons_zh") else p.get("reasons_zh") or []
            )
            model_dir = (
                p.model_direction if hasattr(p, "model_direction") else p.get("model_direction", 0)
            )
            model_cat = (
                p.model_category if hasattr(p, "model_category") else p.get("model_category", "")
            )
            strength_label = (
                p.strength_label if hasattr(p, "strength_label") else p.get("strength_label", "")
            )
            url = p.url if hasattr(p, "url") else p.get("url", "")
            cluster_id = p.cluster_id if hasattr(p, "cluster_id") else p.get("cluster_id", "")
            rules_dir = (
                p.rules_direction
                if hasattr(p, "rules_direction")
                else p.get("rules_direction")
            )

            st.markdown(f"**{i + 1}. `{eid}`** · {title[:120]}")
            st.caption(snippet[:240] if snippet else "（无摘要）")
            al_rank = (
                p.priority_rank if hasattr(p, "priority_rank") else p.get("priority_rank", 0)
            )
            al_score = p.al_score if hasattr(p, "al_score") else p.get("al_score", 0)
            bits = [
                f"模型猜测：{direction_zh(int(model_dir))}（{model_cat or '—'}）",
                f"强弱：{strength_label or '—'}",
            ]
            if al_rank:
                bits.append(f"AL优先#{int(al_rank)}")
            if al_score:
                bits.append(f"不确定度={float(al_score):.2f}")
            if cluster_id:
                bits.append(f"簇：{cluster_id}")
            if rules_dir is not None:
                bits.append(f"规则方向：{direction_zh(int(rules_dir))}")
            if reasons_zh:
                bits.append("原因：" + "；".join(reasons_zh))
            st.caption(" · ".join(bits))
            if url:
                st.caption(f"[原文链接]({url})")

            prev = st.session_state["hitl_choices"].get(eid, "skip")
            default_label = rev_map.get(prev, "跳过")
            default_idx = choice_opts.index(default_label) if default_label in choice_opts else 3
            pick_label = st.radio(
                f"你的判断（{eid}）",
                choice_opts,
                index=default_idx,
                horizontal=True,
                key=f"hitl_radio_{eid}",
            )
            picks[str(eid)] = choice_map[pick_label]
            st.markdown("---")

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            submitted = st.form_submit_button("确认并继续生成报告", type="primary")
        with c2:
            skip_all = st.form_submit_button("全部跳过并继续")
        with c3:
            cancel = st.form_submit_button("取消本次分析")

    if cancel:
        st.session_state.pop("hitl_checkpoint", None)
        st.session_state.pop("hitl_choices", None)
        st.warning("已取消。可重新点「运行分析」。")
        return False

    if submitted or skip_all:
        if skip_all:
            overrides = {str(p.evidence_id if hasattr(p, "evidence_id") else p.get("evidence_id")): "skip" for p in pending}
        else:
            overrides = picks
            st.session_state["hitl_choices"] = dict(picks)
        with st.spinner("已收到人工判断，继续赋权与蒙特卡洛…"):
            try:
                result = run_pipeline_phase_b(
                    cp,
                    review_overrides=overrides,
                    out_dir="output",
                    verbose=False,
                )
            except Exception as exc:
                st.error(f"Phase B 失败：{exc}")
                return False
        if result.market.notes:
            st.caption("｜".join(result.market.notes[:3]))
        st.session_state[_spot_cache_key(result.market.pair)] = {
            "ok": True,
            "pair": result.market.pair,
            "spot": float(result.market.spot),
            "source": result.market.source,
            "asof": result.market.asof,
            "notes": list(result.market.notes[:3]),
            "error": None,
        }
        _store_pipeline_result(result)
        st.success("人工确认已应用，报告已生成。")
        st.rerun()
    return True


def _require_password() -> bool:
    """Return True if the user may see the main UI. Always requires a password."""
    init_language()
    expected = _app_password()
    if st.session_state.get("_auth_ok") is True:
        return True
    st.title("FX Analyse")
    render_language_selector(location="main", key="auth_ui_lang_select")
    st.caption(t("auth.caption"))
    entered = st.text_input(t("auth.password"), type="password", key="auth_password_input")
    if st.button(t("auth.enter"), type="primary", key="auth_submit"):
        if password_accepted(entered, expected):
            st.session_state["_auth_ok"] = True
            st.rerun()
        st.error(t("auth.wrong") if (entered or "").strip() else t("auth.empty"))
    return False


def _fmt_pct(x: float | None, *, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and (x != x)):  # NaN
        return "—"
    return f"{100 * float(x):.{digits}f}%"


def _fmt_num(x: float | None, *, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    return f"{float(x):.{digits}f}"


def render_calib_trust_panel(pair: str, *, cal_loaded: bool, cal_path: str | None) -> None:
    """Prominent holdout / OOS + calibrated-vs-prior status (trust surface)."""
    oos = load_calib_oos_summary(pair)
    if cal_loaded and cal_path:
        src_label = "已加载校准参数"
        path_hint = Path(cal_path).name
        if "data/calibrated" in str(cal_path).replace("\\", "/"):
            path_hint = f"{path_hint}（镜像内置）"
        elif str(cal_path).startswith("output") or "/output/" in str(cal_path).replace("\\", "/"):
            path_hint = f"{path_hint}（本地 output/）"
        st.success(f"**{src_label}** · `{path_hint}`")
    else:
        st.warning("**默认先验** · 未使用 Stage-1 校准 JSON（或文件不存在）")

    if not oos:
        st.caption(
            f"无 holdout 摘要（期望 `calib_oos_summary_{pair.replace('/', '')}.json` "
            f"于 output/ 或 `{BUNDLED_CALIBRATED_DIR.name}/`）。"
        )
        return

    hold = oos.get("holdout") or {}
    train = oos.get("train") or {}
    st.markdown("**校准 holdout（OOS）** — 时序末段样本，非样本内过拟合指标")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Holdout hit rate", _fmt_pct(hold.get("hit_rate")))
    c2.metric("Holdout Brier", _fmt_num(hold.get("brier")))
    c3.metric("Skill（Brier）", _fmt_num(hold.get("skill_brier")))
    c4.metric("Train hit rate", _fmt_pct(train.get("hit_rate")))
    c5.metric(
        "Holdout n",
        f"{int(hold.get('n') or 0)}" if hold.get("n") == hold.get("n") else "—",
    )
    if hold.get("reliability_ece") is not None and hold.get("reliability_ece") == hold.get(
        "reliability_ece"
    ):
        st.caption(
            f"评分规则：Brier / log-loss（严格恰当）· "
            f"Skill = 1 − 模型分/气候基准 · "
            f"可靠性 ECE={_fmt_num(hold.get('reliability_ece'))}"
        )
    rel = hold.get("reliability_buckets") or []
    if hold.get("reliability_argmax") or rel:
        render_probability_reliability(
            hold=hold,
            title="可靠性（预测概率 vs 实际命中）",
            expanded=False,
        )
    note = oos.get("note") or ""
    src = oos.get("source") or ""
    bits = [b for b in (src, note) if b]
    if bits:
        st.caption(" · ".join(bits))


def _reliability_argmax_frame(rows: list[dict] | None) -> pd.DataFrame | None:
    """Build Chinese-labeled predicted-vs-actual table from reliability_argmax bins."""
    if not rows:
        return None
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            n = float(r.get("n") or 0)
        except (TypeError, ValueError):
            continue
        if n <= 0 or n != n:
            continue
        mean_p = r.get("mean_p")
        hit = r.get("hit_rate")
        try:
            mp = float(mean_p)
            hr = float(hit)
        except (TypeError, ValueError):
            continue
        if mp != mp or hr != hr:
            continue
        lo = r.get("bin_lo")
        hi = r.get("bin_hi")
        try:
            label = f"{float(lo):.1f}–{float(hi):.1f}"
        except (TypeError, ValueError):
            label = "—"
        out.append(
            {
                "概率分箱": label,
                "预测概率均值": mp,
                "实际命中率": hr,
                "理想校准线": mp,
                "n": int(n),
            }
        )
    if not out:
        return None
    return pd.DataFrame(out)


def render_probability_reliability(
    *,
    hold: dict | None,
    title: str = "概率可靠性",
    expanded: bool = False,
    pair_label: str | None = None,
) -> None:
    """
    Gneiting-style reliability: binned predicted vs actual + ECE.
    Uses holdout (or live backtest summary) fields when present.
    """
    hold = hold or {}
    argmax = hold.get("reliability_argmax") or []
    buckets = hold.get("reliability_buckets") or []
    ece = hold.get("reliability_ece")
    if not argmax and not buckets and ece is None:
        return

    hdr = title if not pair_label else f"{title}（{pair_label}）"
    with st.expander(hdr, expanded=expanded):
        st.caption(
            "可靠性图 / 分箱表：横轴为模型给出的预测概率，纵轴为实际命中率；"
            "理想校准落在对角线上。数据来自校准 holdout / 回测 OOS（有则显示）。"
        )
        if ece is not None and ece == ece:
            st.metric("可靠性 ECE", _fmt_num(float(ece)))

        chart_df = _reliability_argmax_frame(list(argmax) if argmax else None)
        if chart_df is not None and not chart_df.empty:
            st.markdown("**可靠性图（argmax 概率分箱）**")
            plot_df = chart_df.set_index("预测概率均值")[
                ["实际命中率", "理想校准线"]
            ]
            st.line_chart(plot_df, use_container_width=True)
            st.dataframe(
                chart_df.rename(
                    columns={
                        "预测概率均值": "预测均值",
                        "实际命中率": "实际命中",
                        "理想校准线": "理想线",
                    }
                ).drop(columns=["理想线"], errors="ignore"),
                hide_index=True,
                use_container_width=True,
            )
        elif buckets:
            st.markdown("**分档预测概率 vs 实际频率**")
            st.dataframe(
                pd.DataFrame(buckets).rename(
                    columns={
                        "bucket": "分档",
                        "mean_p": "预测均值",
                        "emp_rate": "实际命中",
                        "n": "n",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("暂无分箱可靠性数据（需先跑校准/回测写出 OOS 摘要）。")


def render_cross_pair_quality_board(*, current_pair: str | None = None) -> None:
    """「跨对质量」— bundled/local calib_oos_summary for all calibrated pairs."""
    with st.expander("跨对质量（OOS / 校准 holdout）", expanded=False):
        st.caption(
            "来自 `calib_oos_summary_*.json`（优先本地 output/，否则镜像内置 "
            f"`{BUNDLED_CALIBRATED_DIR.as_posix()}`）。"
            "Holdout = 时序末段。评分规则：Brier / log-loss；"
            "Skill = 1 − 模型分/气候频率基准（越高越好）；"
            "可靠性 ECE = |预测概率 − 实际命中| 加权平均。"
        )
        try:
            board = calib_oos_board_dataframe()
        except Exception as exc:
            st.warning(f"无法加载跨对 OOS 摘要：{exc}")
            return
        if board.empty:
            st.info("未找到任何 calib_oos_summary_*.json。")
            return

        show = board.copy()

        def _safe_pct(x: object) -> str:
            try:
                if x is None or (isinstance(x, float) and x != x):
                    return "—"
                return _fmt_pct(float(x))
            except (TypeError, ValueError):
                return "—"

        def _safe_num(x: object) -> str:
            try:
                if x is None or (isinstance(x, float) and x != x):
                    return "—"
                return _fmt_num(float(x))
            except (TypeError, ValueError):
                return "—"

        def _safe_int(x: object) -> str:
            try:
                if x is None or (isinstance(x, float) and x != x):
                    return "—"
                return str(int(float(x)))
            except (TypeError, ValueError):
                return "—"

        show["holdout_hit"] = show["holdout_hit"].map(_safe_pct)
        show["train_hit"] = show["train_hit"].map(_safe_pct)
        show["holdout_brier"] = show["holdout_brier"].map(_safe_num)
        show["holdout_skill_brier"] = show["holdout_skill_brier"].map(_safe_num)
        show["holdout_logloss"] = show["holdout_logloss"].map(_safe_num)
        show["holdout_ece"] = show["holdout_ece"].map(_safe_num)
        show["train_brier"] = show["train_brier"].map(_safe_num)
        show["train_skill_brier"] = show["train_skill_brier"].map(_safe_num)
        show["holdout_n"] = show["holdout_n"].map(_safe_int)
        show["train_n"] = show["train_n"].map(_safe_int)
        show = show.rename(
            columns={
                "pair": "货币对",
                "holdout_hit": "Holdout hit",
                "holdout_brier": "Holdout Brier",
                "holdout_skill_brier": "Skill（Brier）",
                "holdout_logloss": "Holdout log-loss",
                "holdout_ece": "可靠性 ECE",
                "holdout_n": "Holdout n",
                "train_hit": "Train hit",
                "train_brier": "Train Brier",
                "train_skill_brier": "Train Skill",
                "train_n": "Train n",
                "source": "来源",
            }
        )
        st.dataframe(show, hide_index=True, use_container_width=True)

        # Reliability detail for current pair when available
        if current_pair:
            try:
                oos_cur = load_calib_oos_summary(current_pair)
            except Exception:
                oos_cur = None
            hold_cur = (oos_cur or {}).get("holdout") or {}
            if hold_cur.get("reliability_argmax") or hold_cur.get("reliability_buckets"):
                st.markdown(f"**{current_pair} 可靠性（预测概率 vs 实际命中）**")
                chart_df = _reliability_argmax_frame(
                    list(hold_cur.get("reliability_argmax") or [])
                )
                if chart_df is not None and not chart_df.empty:
                    st.line_chart(
                        chart_df.set_index("预测概率均值")[
                            ["实际命中率", "理想校准线"]
                        ],
                        use_container_width=True,
                    )
                    st.dataframe(
                        chart_df.drop(columns=["理想校准线"], errors="ignore"),
                        hide_index=True,
                        use_container_width=True,
                    )
                elif hold_cur.get("reliability_buckets"):
                    st.dataframe(
                        pd.DataFrame(hold_cur["reliability_buckets"]).rename(
                            columns={
                                "bucket": "分档",
                                "mean_p": "预测均值",
                                "emp_rate": "实际命中",
                                "n": "n",
                            }
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )

        if current_pair:
            st.caption(
                f"当前分析对 **{current_pair}**：可到下方「历史回测（argmax hit / Brier）」"
                f"跑小样本核对；完整 CLI：`python run_cli.py backtest --pair {current_pair}`"
            )
            cur = board[board["pair"] == current_pair]
            if not cur.empty:
                row = cur.iloc[0]
                st.caption(
                    f"本对 holdout hit={_fmt_pct(row.get('holdout_hit'))} · "
                    f"Brier={_fmt_num(row.get('holdout_brier'))} · "
                    f"Skill={_fmt_num(row.get('holdout_skill_brier'))} · "
                    f"n={_safe_int(row.get('holdout_n'))}"
                )


def render_current_pair_reliability_board(*, current_pair: str | None = None) -> None:
    """Standalone「概率可靠性」expander for the active pair's calib OOS."""
    if not current_pair:
        return
    try:
        oos = load_calib_oos_summary(current_pair)
    except Exception:
        oos = None
    hold = (oos or {}).get("holdout") or {}
    if not (
        hold.get("reliability_argmax")
        or hold.get("reliability_buckets")
        or hold.get("reliability_ece") is not None
    ):
        with st.expander("概率可靠性", expanded=False):
            st.info(
                f"当前对 **{current_pair}** 尚无 OOS 可靠性数据。"
                "完成日校准或回测后会显示分箱表与可靠性图。"
            )
        return
    render_probability_reliability(
        hold=hold,
        title="概率可靠性",
        expanded=False,
        pair_label=current_pair,
    )


def render_replay_summary_board(*, current_pair: str | None = None, out_dir: str = "output") -> None:
    with st.expander("历史冻结回放总览", expanded=False):
        st.caption(
            "汇总 `output/replay_backtest_*.json`。"
            "若 `historical_news_working=yes`，表示至少有一个回放时点同时满足 "
            "`historical_news_quality=date_filtered` 且 `evidence_n>0`。"
        )
        try:
            board = replay_summary_dataframe(out_dir)
        except Exception as exc:
            st.warning(f"无法加载 replay 汇总：{exc}")
            return
        if board.empty:
            st.info("尚未找到 replay_backtest_*.json。先跑一次「历史时点回放」或 CLI replay-backtest。")
            return

        show = board.copy()
        if current_pair:
            show = show[show["pair"] == current_pair].reset_index(drop=True)
            if show.empty:
                st.info(f"当前货币对 `{current_pair}` 暂无 replay 汇总；下方仍可运行新的小样本。")
                show = board.copy()
        show["argmax_hit_rate"] = show["argmax_hit_rate"].map(lambda x: _fmt_pct(x))
        show["mean_brier"] = show["mean_brier"].map(lambda x: _fmt_num(x, digits=4))
        show["mean_skill_brier"] = show["mean_skill_brier"].map(lambda x: _fmt_num(x, digits=4))
        show["evidence_mean"] = show["evidence_mean"].map(lambda x: _fmt_num(x, digits=2))
        show = show.rename(
            columns={
                "pair": "货币对",
                "window": "窗口",
                "n_rows": "回放时点数",
                "argmax_hit_rate": "hit_rate",
                "mean_brier": "mean_brier",
                "mean_skill_brier": "mean_skill_brier",
                "evidence_mean": "evidence_mean",
                "evidence_max": "evidence_max",
                "date_filtered_count": "date_filtered",
                "limited_count": "limited",
                "historical_news_working": "历史新闻是否工作",
            }
        )
        cols = [
            c
            for c in (
                "货币对",
                "窗口",
                "回放时点数",
                "hit_rate",
                "mean_brier",
                "mean_skill_brier",
                "evidence_mean",
                "evidence_max",
                "date_filtered",
                "limited",
                "历史新闻是否工作",
            )
            if c in show.columns
        ]
        st.dataframe(show[cols], hide_index=True, use_container_width=True)
        n_working = int((board["historical_news_working"] == "yes").sum())
        if n_working <= 0:
            st.warning(
                "当前已落盘 replay 结果里，还没有看到“历史日期过滤新闻 + 非零证据”同时成立的样本。"
                "这通常意味着历史新闻源尚未真正接通，或该窗口没有取到可用历史新闻。"
            )
        else:
            st.success(f"已发现 {n_working} 个 replay 窗口命中真实历史新闻证据。")
        st.caption("CLI 汇总：`python run_cli.py replay-summary --out output`")


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


# Relative-cut number_input bounds (NOT the default cuts).
_PCT_CUT_MIN = PCT_CUT_MIN
_PCT_CUT_MAX = PCT_CUT_MAX


def _default_pct_cuts(base: ModelWeights) -> list[float]:
    return [float(x) for x in base.bucket_pct_cuts]


def _apply_pct_cuts_to_session(
    analysis_pair: str,
    pct_key: str,
    abs_key: str,
    cuts: list[float],
    spot: float | None,
) -> None:
    st.session_state[pct_key] = [float(x) for x in cuts]
    for i, v in enumerate(cuts):
        st.session_state[f"pct_cut_{analysis_pair}_{i}"] = float(v)
    if spot is not None:
        abs_edges = list(edges_from_spot(spot, tuple(cuts)))  # type: ignore[arg-type]
        st.session_state[abs_key] = abs_edges
        for i, e in enumerate(abs_edges):
            st.session_state[f"abs_cut_{analysis_pair}_{i}"] = float(e)


def render_bucket_editor(
    base: ModelWeights,
    spot: float | None,
    analysis_pair: str,
) -> tuple[bool, tuple[float, float, float, float], tuple[float, float, float, float]]:
    """
    Main-area bucket edges. 4 cut points → 5 Torchcast-style buckets.
    Returns (use_relative, pct_cuts, abs_edges).
    """
    defaults = _default_pct_cuts(base)
    st.subheader("概率区间（自己设边界）")
    st.caption(
        "4 个边界 → 5 个区间（与 FX Analyse 一致）："
        "`< e1` · `e1–e2` · `e2–e3` · `e3–e4` · `≥ e4`。"
        "运行分析后，蒙特卡洛概率与 PDF 都用这套边界。"
        f" 相对模式默认是 **+{defaults[0]:g} / +{defaults[1]:g} / +{defaults[2]:g} / +{defaults[3]:g}%**"
        f"（相对现价）；**-20 只是输入下限**，不是默认值。"
    )

    mode_key = f"bucket_mode::{analysis_pair}"
    pct_key = f"bucket_pct::{analysis_pair}"
    abs_key = f"bucket_abs::{analysis_pair}"
    seeded_key = f"abs_seeded_from_spot::{analysis_pair}"

    # Apply 单子 PDF bucket hints once (mode / absolute or relative cuts)
    _apply_order_pdf_bucket_hint(analysis_pair, mode_key, pct_key, abs_key)
    hint = st.session_state.get("order_pdf_bucket_hint") or {}
    if hint.get("barrier") is not None or hint.get("strike") is not None:
        bits = []
        if hint.get("barrier") is not None:
            bits.append(f"Barrier={float(hint['barrier']):g}")
        if hint.get("strike") is not None:
            bits.append(f"Strike={float(hint['strike']):g}")
        st.caption("单子价位参考（请核对后再改切点）：" + " · ".join(bits))

    if pct_key not in st.session_state:
        st.session_state[pct_key] = list(defaults)

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

    oob_seed_flag = f"_pct_seed_oob::{analysis_pair}"
    for i, v in enumerate(st.session_state[pct_key]):
        wk = f"pct_cut_{analysis_pair}_{i}"
        if wk not in st.session_state:
            # Keep widget seeds inside number_input bounds (avoid silent clamp to -20).
            clamped, oob = seed_pct_widget_value(float(v))
            st.session_state[wk] = clamped
            if oob:
                st.session_state[oob_seed_flag] = True
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

    # No silent default: index=None until the user picks explicitly
    mode = st.radio(
        "边界方式（必选）",
        ["相对现价", "绝对价位"],
        index=None,
        horizontal=True,
        key=mode_key,
        help="相对：边界 = 现价 × (1 + 涨幅%/100)。绝对：直接填汇率价位。不预选，请手动选一项。",
    )
    if mode is None:
        st.info("请先选择边界方式：相对现价 或 绝对价位。")
        defaults_preview = _default_pct_cuts(base)
        preview_edges = (
            edges_from_spot(spot, tuple(defaults_preview))
            if spot is not None
            else tuple(float(x) for x in base.bucket_edges)
        )
        st.caption("选定边界方式后可编辑切点；未选时不会运行分析。")
        return True, tuple(defaults_preview), tuple(preview_edges)  # type: ignore[return-value]
    use_rel = mode == "相对现价"

    # Heal only when widgets were seeded from out-of-range abs→pct (not intentional -20).
    pct_widget_vals = [
        float(st.session_state.get(f"pct_cut_{analysis_pair}_{i}", defaults[i]))
        for i in range(4)
    ]
    seeded_oob = bool(st.session_state.get(oob_seed_flag))
    if use_rel and should_heal_floor_clamp(pct_widget_vals, seeded_from_oob=seeded_oob):
        st.session_state.pop(oob_seed_flag, None)
        _apply_pct_cuts_to_session(analysis_pair, pct_key, abs_key, defaults, spot)
        st.warning(
            "相对涨幅曾全部落在下限 **-20%**（常见原因：绝对价位与当前分析现价口径不一致，"
            "换算后的相对%超出输入框范围被钳住）。已恢复默认 "
            f"**+{defaults[0]:g}/+{defaults[1]:g}/+{defaults[2]:g}/+{defaults[3]:g}%**。"
        )

    reset = st.button(
        f"恢复默认相对涨幅 +{defaults[0]:g}/+{defaults[1]:g}/+{defaults[2]:g}/+{defaults[3]:g}%",
        key=f"reset_pct_cuts::{analysis_pair}",
        help="-20 不是默认；点此重置四个相对边界。",
    )
    if reset:
        st.session_state.pop(oob_seed_flag, None)
        _apply_pct_cuts_to_session(analysis_pair, pct_key, abs_key, defaults, spot)

    cols = st.columns(4)
    if use_rel:
        pcts: list[float] = []
        for i, col in enumerate(cols):
            with col:
                pcts.append(
                    float(
                        st.number_input(
                            f"相对涨幅 {i + 1}（相对现价 +%）",
                            min_value=_PCT_CUT_MIN,
                            max_value=_PCT_CUT_MAX,
                            step=0.5,
                            key=f"pct_cut_{analysis_pair}_{i}",
                            help=(
                                "填相对现价的上涨百分比，不是汇率。例：现价 1.43、填 2 → 边界≈1.4586。"
                                f"可调范围 {_PCT_CUT_MIN:g}%～{_PCT_CUT_MAX:g}%；"
                                f"默认 +{defaults[0]:g}/+{defaults[1]:g}/+{defaults[2]:g}/+{defaults[3]:g}。"
                            ),
                        )
                    )
                )
        st.caption(
            f"填的是相对现价的上涨百分比（默认 +{defaults[0]:g}/+{defaults[1]:g}/"
            f"+{defaults[2]:g}/+{defaults[3]:g}）；"
            f"输入框允许 {_PCT_CUT_MIN:g}%～{_PCT_CUT_MAX:g}%——"
            f"**看到 -20 多半是下限，不是默认档位**。"
            " 例：现价 1.43、填 2 → 边界≈1.4586。"
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
            raw_pcts = [(e / spot - 1.0) * 100.0 for e in abs_edges]
            pct_cuts = tuple(sorted(raw_pcts))  # type: ignore[assignment]
            st.session_state[pct_key] = list(pct_cuts)
            # Do NOT push out-of-range % into relative widgets — Streamlit would
            # clamp them all to min_value (-20) and poison the next relative view.
            if pct_cuts_in_bounds(raw_pcts):
                st.session_state.pop(oob_seed_flag, None)
                for i, p in enumerate(pct_cuts):
                    st.session_state[f"pct_cut_{analysis_pair}_{i}"] = float(p)
            else:
                # Mark so switching back to relative can one-shot heal if widgets
                # were ever seeded from these OOB percentages.
                st.session_state[oob_seed_flag] = True
                lo, hi = min(raw_pcts), max(raw_pcts)
                st.warning(
                    f"当前绝对边界相对现价约 {lo:+.1f}%～{hi:+.1f}%，"
                    f"超出相对模式可编辑范围（{_PCT_CUT_MIN:g}%～{_PCT_CUT_MAX:g}%）。"
                    "请继续用「绝对价位」，或点上方「恢复默认相对涨幅」后再切回相对模式。"
                    "（勿把 AUD/USD 价位填进 USD/AUD 分析口径。）"
                )
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
    """侧栏分区 ①：展示已确认的开始设置；点按钮打开弹窗（无预选）。"""
    with st.sidebar.expander(t("side.start"), expanded=True):
        cfg = st.session_state.get("start_cfg")
        if not cfg:
            st.caption(t("side.start.empty"))
        else:
            bull = cfg.get("bullish_currency") or "—"
            eng = cfg.get("peak_engine") or "—"
            cal = cfg.get("use_calibrated")
            cal_s = (
                t("val.use")
                if cal is True
                else (t("val.not_use") if cal is False else "—")
            )
            hr = cfg.get("human_review")
            hr_s = (
                t("val.review")
                if hr is True
                else (t("val.skip") if hr is False else "—")
            )
            mode_s = (
                t("opt.simple")
                if is_simple_setup_mode(cfg.get("setup_mode"))
                else t("opt.expert")
            )
            st.markdown(
                t(
                    "side.start.summary",
                    pair=cfg.get("pair") or "—",
                    bull=bull,
                    mode=mode_s,
                    eng=eng,
                    cal=cal_s,
                    hr=hr_s,
                )
            )
            if is_simple_setup_mode(cfg.get("setup_mode")):
                st.caption(t("side.start.algo_caption"))
                if st.button(
                    t("side.to_expert"),
                    use_container_width=True,
                    key="btn_switch_expert",
                    help=t("side.to_expert.help"),
                ):
                    st.session_state["_force_setup_mode"] = "专家"
                    st.session_state["_open_start_setup"] = True
                    st.rerun()
        if st.button(t("side.open_start"), use_container_width=True, key="btn_open_start_setup"):
            st.session_state["_open_start_setup"] = True
            st.rerun()

        if not cfg:
            return None, None

        try:
            spec = _spec_from_start_cfg(cfg)
        except Exception as exc:
            st.error(t("side.start.invalid", exc=exc))
            return None, None
        bullish = cfg.get("bullish_currency")
        return spec, bullish


def _spec_from_start_cfg(cfg: dict) -> PairSpec:
    mode = cfg.get("pair_mode") or "目录"
    pair = str(cfg.get("pair") or "").strip()
    if not pair:
        raise ValueError("货币对为空")
    if mode == "自定义":
        ticker = str(cfg.get("custom_ticker") or "").strip() or pair.replace("/", "")
        invert = bool(cfg.get("custom_invert", False))
        return make_custom_pair(pair, ticker, invert)
    return get_pair(pair)


def _start_cfg_choice_map(cfg: dict | None, *, bucket_mode: str | None) -> dict:
    """Build the dict expected by missing_start_choices."""
    cfg = cfg or {}
    return {
        "pair": cfg.get("pair"),
        "bullish_currency": cfg.get("bullish_currency"),
        "peak_engine": cfg.get("peak_engine"),
        "use_calibrated": cfg.get("use_calibrated"),
        "human_review": cfg.get("human_review"),
        "bucket_mode": bucket_mode,
        "setup_mode": cfg.get("setup_mode") or "expert",
    }


def _missing_for_start_cfg(cfg: dict | None, *, bucket_mode: str | None) -> list[str]:
    """Validate required start fields according to 简洁 / 专家 mode."""
    cfg = cfg or {}
    mode = str(cfg.get("setup_mode") or "expert")
    return missing_start_choices(
        _start_cfg_choice_map(cfg, bucket_mode=bucket_mode),
        setup_mode=mode,
        include_bucket=True,
        lang=get_lang(),
    )


def _dismiss_start_setup() -> None:
    st.session_state.pop("_open_start_setup", None)


def _dismiss_missing_start() -> None:
    st.session_state.pop("_show_missing_start", None)
    st.session_state.pop("_missing_start_labels", None)


def _apply_order_pdf_to_dialog_state(result: OrderDocParse) -> None:
    """
    Seed start-setup widget keys from a confident PDF parse.

    Does NOT touch peak_engine / use_calibrated / human_review.
    Bucket hints are applied later in the main-area bucket editor.
    """
    if not result.ok:
        return
    if result.pair_mode in ("目录", "自定义"):
        st.session_state["dlg_pair_mode"] = result.pair_mode
    if result.pair:
        if result.pair_mode == "自定义":
            st.session_state["dlg_pair_custom"] = result.pair
        else:
            # Catalog selectbox; fall back to custom if not listed
            if result.pair in list_pairs():
                st.session_state["dlg_pair_mode"] = "目录"
                st.session_state["dlg_pair_catalog"] = result.pair
            else:
                st.session_state["dlg_pair_mode"] = "自定义"
                st.session_state["dlg_pair_custom"] = result.pair
    if result.bullish_currency:
        st.session_state["dlg_bullish"] = result.bullish_currency
    st.session_state["order_pdf_bucket_hint"] = {
        "pair": result.pair,
        "bucket_mode": result.bucket_mode,
        "bucket_edges": list(result.bucket_edges) if result.bucket_edges else None,
        "bucket_pct_cuts": list(result.bucket_pct_cuts) if result.bucket_pct_cuts else None,
        "barrier": result.barrier,
        "strike": result.strike,
        "spot": result.spot,
    }
    # Allow re-apply when pair changes after confirm
    if result.pair:
        st.session_state.pop(f"_order_pdf_bucket_applied::{result.pair}", None)


def _apply_order_pdf_bucket_hint(analysis_pair: str, mode_key: str, pct_key: str, abs_key: str) -> None:
    """One-shot apply of PDF bucket mode / edges into the main bucket editor."""
    hint = st.session_state.get("order_pdf_bucket_hint") or {}
    if not hint:
        return
    hint_pair = str(hint.get("pair") or "")
    if hint_pair and hint_pair != analysis_pair:
        return
    flag = f"_order_pdf_bucket_applied::{analysis_pair}"
    if st.session_state.get(flag):
        return
    mode = hint.get("bucket_mode")
    if mode in ("相对现价", "绝对价位"):
        st.session_state[mode_key] = mode
    edges = hint.get("bucket_edges")
    if isinstance(edges, (list, tuple)) and len(edges) == 4:
        st.session_state[abs_key] = [float(x) for x in edges]
        for i, e in enumerate(edges):
            st.session_state[f"abs_cut_{analysis_pair}_{i}"] = float(e)
        st.session_state[f"abs_seeded_from_spot::{analysis_pair}"] = True
    pcts = hint.get("bucket_pct_cuts")
    if isinstance(pcts, (list, tuple)) and len(pcts) == 4:
        st.session_state[pct_key] = [float(x) for x in pcts]
        for i, p in enumerate(pcts):
            st.session_state[f"pct_cut_{analysis_pair}_{i}"] = float(p)
    st.session_state[flag] = True


def _fmt_pair_mode(v: str) -> str:
    return t("opt.catalog") if v == "目录" else (t("opt.custom") if v == "自定义" else v)


def _fmt_setup_mode(v: str) -> str:
    return t("opt.simple") if v == "简洁（推荐）" else (t("opt.expert") if v == "专家" else v)


def _fmt_cal_opt(v: str) -> str:
    return t("opt.cal_use") if v == "使用" else (t("opt.cal_skip") if v == "不使用" else v)


def _fmt_hr_opt(v: str) -> str:
    return (
        t("opt.need_review")
        if v == "需要人工确认"
        else (t("opt.auto_skip") if v == "自动跳过" else v)
    )


def _start_setup_dialog_body() -> None:
    """
    Modal body for must-have start choices.
    Internal option values stay Chinese for session compatibility;
    labels go through t().
    """
    prev = st.session_state.get("start_cfg") or {}
    editing = bool(prev)
    lang = get_lang()
    ph = choice_placeholder(lang)

    setup_mode_opts = ["简洁（推荐）", "专家"]
    force = st.session_state.pop("_force_setup_mode", None)
    if force in setup_mode_opts:
        prev_setup = force
    elif editing and is_simple_setup_mode(prev.get("setup_mode")):
        prev_setup = "简洁（推荐）"
    elif editing and prev.get("setup_mode") == "expert":
        prev_setup = "专家"
    else:
        prev_setup = "简洁（推荐）"
    setup_idx = setup_mode_opts.index(prev_setup)
    setup_pick = st.radio(
        t("dlg.setup_mode"),
        setup_mode_opts,
        index=setup_idx,
        horizontal=True,
        key="dlg_setup_mode",
        format_func=_fmt_setup_mode,
        help=t("dlg.setup_mode.help"),
    )
    simple = setup_pick == "简洁（推荐）"

    if simple:
        st.caption(t("dlg.simple.caption"))
    else:
        st.caption(t("dlg.expert.caption"))

    with st.expander(t("dlg.upload_pdf"), expanded=not editing):
        uploaded = st.file_uploader(
            t("dlg.upload_pdf.label"),
            type=["pdf", "jpg", "jpeg", "png"],
            key="dlg_order_pdf",
            help=t("dlg.upload_pdf.help"),
        )
        if uploaded is not None:
            raw = uploaded.getvalue()
            file_id = f"{uploaded.name}:{len(raw)}:{hash(raw[:4096])}"
            if st.session_state.get("_order_pdf_file_id") != file_id:
                with st.spinner(t("dlg.pdf.parsing")):
                    result = parse_order_document(
                        raw, filename=uploaded.name, use_llm=True
                    )
                st.session_state["_order_pdf_file_id"] = file_id
                st.session_state["order_pdf_result"] = result.to_dict()
                if result.ok:
                    _apply_order_pdf_to_dialog_state(result)
                st.rerun()

        pdf_res = order_doc_from_dict(st.session_state.get("order_pdf_result"))
        if pdf_res is not None:
            if not pdf_res.ok:
                st.error(pdf_res.error or t("dlg.pdf.fail"))
            else:
                st.success(t("dlg.pdf.ok"))
                for line in preview_lines(pdf_res):
                    st.caption(line)

    pair_modes = ["目录", "自定义"]
    prev_mode = prev.get("pair_mode") if editing else None
    if "dlg_pair_mode" in st.session_state and st.session_state["dlg_pair_mode"] in pair_modes:
        prev_mode = st.session_state["dlg_pair_mode"]
    mode_idx = pair_modes.index(prev_mode) if prev_mode in pair_modes else None
    mode = st.radio(
        t("dlg.pair_mode"),
        pair_modes,
        index=mode_idx,
        horizontal=True,
        key="dlg_pair_mode",
        format_func=_fmt_pair_mode,
    )

    pair: str | None = None
    custom_ticker = ""
    custom_invert = False
    if mode is None:
        st.info(t("dlg.pair_mode.hint"))
        base_opts: list[str] = []
    elif mode == "目录":
        pairs = list_pairs()
        prev_pair = prev.get("pair") if editing else None
        p_idx = pairs.index(prev_pair) if prev_pair in pairs else None
        pair = st.selectbox(
            t("dlg.pair"),
            pairs,
            index=p_idx,
            placeholder=ph,
            key="dlg_pair_catalog",
        )
        base_opts = []
        if pair and not is_unset_choice(pair):
            try:
                sp = get_pair(pair)
                base_opts = [sp.base, sp.quote]
            except Exception:
                base_opts = []
    else:
        default_pair = str(prev.get("pair") or "") if editing else ""
        pair_in = st.text_input(
            t("dlg.pair_custom"),
            value=default_pair,
            placeholder=t("dlg.pair_custom.ph"),
            key="dlg_pair_custom",
        )
        pair = (pair_in or "").strip() or None
        custom_ticker = st.text_input(
            t("dlg.ticker"),
            value=str(prev.get("custom_ticker") or "") if editing else "",
            key="dlg_custom_ticker",
        )
        custom_invert = st.checkbox(
            "invert",
            value=bool(prev.get("custom_invert", False)) if editing else False,
            key="dlg_custom_invert",
        )
        base_opts = []
        if pair and "/" in pair:
            parts = pair.split("/", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                base_opts = [parts[0].strip().upper(), parts[1].strip().upper()]
                pair = f"{base_opts[0]}/{base_opts[1]}"

    bullish = None
    if base_opts:
        prev_b = prev.get("bullish_currency") if editing else None
        b_idx = base_opts.index(prev_b) if prev_b in base_opts else None
        bullish = st.radio(
            t("dlg.bullish"),
            base_opts,
            index=b_idx,
            horizontal=True,
            key="dlg_bullish",
            help=t("dlg.bullish.help"),
        )
    else:
        st.caption(t("dlg.bullish.wait"))

    rec: AlgoRecommendation | None = None
    peak_engine: str | None = None
    use_calibrated: bool | None = None
    human_review: bool | None = None
    jump_model: str | None = None
    variance_reduction: str | None = None
    cluster_method: str | None = None

    if simple:
        if pair and not is_unset_choice(pair):
            rec = recommend_algorithms(str(pair))
            peak_engine = rec.peak_engine
            use_calibrated = rec.use_calibrated
            human_review = rec.human_review
            jump_model = rec.jump_model
            variance_reduction = rec.variance_reduction
            cluster_method = rec.cluster_method
            cal_s = t("val.use") if rec.use_calibrated else t("val.not_use")
            hr_s = t("val.review") if rec.human_review else t("val.skip")
            st.info(
                f"{t('dlg.algo.preview')}  \n"
                f"· peak_engine=`{rec.peak_engine}`　jump_model=`{rec.jump_model}`　"
                f"VR=`{rec.variance_reduction}`　cluster=`{rec.cluster_method}`  \n"
                f"· {cal_s}　{hr_s}  \n"
                + "  \n".join(f"· {r}" for r in rec.reasons)
            )
            if st.button(t("dlg.to_expert"), key="dlg_to_expert"):
                st.session_state["_force_setup_mode"] = "专家"
                st.rerun()
        else:
            st.caption(t("dlg.algo.wait"))
    else:
        engines = ["path_max", "brownian_bridge"]
        prev_eng = prev.get("peak_engine") if editing else None
        e_idx = engines.index(prev_eng) if prev_eng in engines else None
        peak_engine = st.selectbox(
            t("dlg.peak_engine"),
            engines,
            index=e_idx,
            placeholder=ph,
            key="dlg_peak_engine",
            help=t("dlg.peak_engine.help"),
        )

        cal_opts = ["使用", "不使用"]
        prev_cal = prev.get("use_calibrated") if editing else None
        cal_idx = 0 if prev_cal is True else (1 if prev_cal is False else None)
        cal_pick = st.radio(
            t("dlg.calibrated"),
            cal_opts,
            index=cal_idx,
            horizontal=True,
            key="dlg_use_cal",
            format_func=_fmt_cal_opt,
            help=t("dlg.calibrated.help"),
        )

        hr_opts = ["需要人工确认", "自动跳过"]
        prev_hr = prev.get("human_review") if editing else None
        hr_idx = 0 if prev_hr is True else (1 if prev_hr is False else None)
        hr_pick = st.radio(
            t("dlg.human_review"),
            hr_opts,
            index=hr_idx,
            horizontal=True,
            key="dlg_human_review",
            format_func=_fmt_hr_opt,
            help=t("dlg.human_review.help"),
        )

        use_calibrated = (
            True if cal_pick == "使用" else (False if cal_pick == "不使用" else None)
        )
        human_review = (
            True
            if hr_pick == "需要人工确认"
            else (False if hr_pick == "自动跳过" else None)
        )

    draft = {
        "pair": pair,
        "bullish_currency": bullish,
        "peak_engine": peak_engine,
        "use_calibrated": use_calibrated,
        "human_review": human_review,
    }
    dialog_keys = START_SIMPLE_DIALOG_KEYS if simple else START_EXPERT_DIALOG_KEYS
    dialog_missing = missing_start_choices(draft, keys=dialog_keys, lang=lang)
    if mode is None:
        pair_label = start_field_label("pair", lang=lang)
        dialog_missing = [
            start_field_label("pair_mode", lang=lang),
            *[x for x in dialog_missing if x != pair_label],
        ]

    c1, c2 = st.columns(2)
    with c1:
        confirmed = st.button(t("dlg.confirm"), type="primary", use_container_width=True)
    with c2:
        if st.button(t("dlg.cancel"), use_container_width=True):
            st.session_state.pop("_open_start_setup", None)
            st.rerun()

    if confirmed:
        if dialog_missing:
            st.error(format_missing_start_message(dialog_missing, lang=lang))
            return
        if bullish not in (base_opts or []):
            st.error(
                format_missing_start_message(
                    [start_field_label("bullish_currency", lang=lang)],
                    lang=lang,
                )
            )
            return
        if simple:
            if rec is None and pair:
                rec = recommend_algorithms(str(pair))
            if rec is None:
                st.error(t("dlg.algo_fail"))
                return
            peak_engine = rec.peak_engine
            use_calibrated = rec.use_calibrated
            human_review = rec.human_review
            jump_model = rec.jump_model
            variance_reduction = rec.variance_reduction
            cluster_method = rec.cluster_method

        cfg_out: dict[str, Any] = {
            "pair_mode": mode,
            "pair": pair,
            "custom_ticker": (custom_ticker or "").strip(),
            "custom_invert": bool(custom_invert),
            "bullish_currency": bullish,
            "peak_engine": peak_engine,
            "use_calibrated": bool(use_calibrated),
            "human_review": bool(human_review),
            "setup_mode": "simple" if simple else "expert",
        }
        if simple and rec is not None:
            cfg_out["jump_model"] = jump_model or rec.jump_model
            cfg_out["variance_reduction"] = (
                variance_reduction or rec.variance_reduction
            )
            cfg_out["cluster_method"] = cluster_method or rec.cluster_method
            cfg_out["algo_recommend"] = rec.to_dict()
        else:
            cfg_out.pop("algo_recommend", None)
            for k in ("jump_model", "variance_reduction", "cluster_method"):
                if k in prev and not simple:
                    pass
        st.session_state["start_cfg"] = cfg_out
        old_pair = (prev or {}).get("pair")
        if old_pair and old_pair != pair:
            st.session_state.pop("scenario_edits", None)
            st.session_state.pop("evidence_edits", None)
            st.session_state.pop("hitl_checkpoint", None)
            st.session_state.pop("hitl_choices", None)
            st.session_state.pop("last_report", None)
        st.session_state.pop("_open_start_setup", None)
        st.session_state["_start_setup_shown"] = True
        st.rerun()


@st.dialog("开始设置", width="large", on_dismiss=_dismiss_start_setup)
def _start_setup_dialog_zh() -> None:
    _start_setup_dialog_body()


@st.dialog("Start setup", width="large", on_dismiss=_dismiss_start_setup)
def _start_setup_dialog_en() -> None:
    _start_setup_dialog_body()


def start_setup_dialog() -> None:
    if get_lang() == "en":
        _start_setup_dialog_en()
    else:
        _start_setup_dialog_zh()


def _missing_start_dialog_body(missing_labels: list[str]) -> None:
    lang = get_lang()
    st.warning(format_missing_start_message(missing_labels, lang=lang))
    st.caption(t("dlg.missing.caption"))
    if st.button(t("dlg.open_start"), type="primary", use_container_width=True):
        st.session_state["_open_start_setup"] = True
        st.session_state.pop("_show_missing_start", None)
        st.rerun()
    if st.button(t("dlg.got_it"), use_container_width=True):
        st.session_state.pop("_show_missing_start", None)
        st.rerun()


@st.dialog("还不能运行", on_dismiss=_dismiss_missing_start)
def _missing_start_dialog_zh(missing_labels: list[str]) -> None:
    _missing_start_dialog_body(missing_labels)


@st.dialog("Can't run yet", on_dismiss=_dismiss_missing_start)
def _missing_start_dialog_en(missing_labels: list[str]) -> None:
    _missing_start_dialog_body(missing_labels)


def missing_start_dialog(missing_labels: list[str]) -> None:
    if get_lang() == "en":
        _missing_start_dialog_en(missing_labels)
    else:
        _missing_start_dialog_zh(missing_labels)


def sidebar_weights(base: ModelWeights, pair_name: str) -> tuple[ModelWeights, dict]:
    st.sidebar.markdown(f"**{pair_name}**")
    st.sidebar.caption(t("side.toc"))

    start_cfg = st.session_state.get("start_cfg") or {}
    # peak_engine / human_review come from 开始设置（必选，无侧栏静默默认）
    peak_engine = str(start_cfg.get("peak_engine") or getattr(base, "peak_engine", "path_max"))
    _hr = start_cfg.get("human_review")
    pause_uncertain = bool(_hr) if isinstance(_hr, bool) else False
    simple_mode = is_simple_setup_mode(start_cfg.get("setup_mode"))
    rec = AlgoRecommendation.from_dict(start_cfg.get("algo_recommend"))

    # ② 抓取
    with st.sidebar.expander(t("side.fetch"), expanded=False):
        use_news = st.checkbox("官方 / vault 头条", value=True)
        ai_research = st.checkbox(
            "AI 检索员",
            value=True,
            help=(
                "模仿人工一条条搜：LLM（DeepSeek 等）当「脑」拟下一句搜索词并挑选标题；"
                "Tavily/Brave/NewsAPI/Google News RSS 当「手」抓真链接。"
                "只填 DeepSeek 不会虚构 URL——请同时填搜索 Key 或依赖免费 Google News。"
            ),
        )
        classify_mode = st.selectbox(
            "证据判定",
            ["hybrid", "llm", "rules"],
            index=0,
            help="hybrid=LLM优先；rules=仅关键词",
        )
        template_policy = st.selectbox(
            "模板证据策略 template_policy",
            ["off", "prior_only", "fallback_warn"],
            index=0,
            help=(
                "新闻证据为空时：off=不用模板（默认，诚实）；"
                "prior_only=用模板但标记先验并降权；"
                "fallback_warn=调试用，用模板并告警"
            ),
        )
        keep_templates = st.checkbox(
            "新闻有证据时也合并模板（标记为先验）",
            value=False,
        )
        max_news_ev = st.slider("最多头条证据条数", 3, 40, 10, 1)
        st.caption(
            "仅影响当日 Live 报告证据池上限；历史回放仍走省钱路径，不会为此虚构证据。"
            "步骤3抓取池会按约 3× 比例放大（上限约 90）。"
        )
        fetch_fulltext = st.checkbox("抓正文供 LLM", value=True)
        use_label_learned = st.checkbox(
            "使用标签学习到的强度",
            value=False,
            help=(
                "Stage 3：若 output/label_audit_*.csv 中 human_direction ≥N 条，"
                "按类别拟合强度倍率并应用到本次证据；不足则提示「标注不足」。"
            ),
        )
        hr_label = "需要人工确认" if pause_uncertain else "自动跳过"
        st.caption(
            f"不确定证据：已在「开始设置」选为 **{hr_label}**"
            "（点侧栏①修改）。"
        )

    # ③ 蒙特卡洛（分档切点在主区设置）
    with st.sidebar.expander(t("side.mc"), expanded=False):
        n_sims = st.number_input("蒙特卡洛次数", 10_000, 500_000, base.n_sims, 10_000)
        trading_days = st.number_input("交易日窗口", 5, 252, base.trading_days, 1)
        seed = st.number_input("随机种子", 0, 10_000_000, base.seed, 1)
        vol_lookback = st.number_input("波动回看日", 20, 252, base.vol_lookback_days, 5)
        st.caption(
            f"峰值引擎：已在「开始设置」选为 `{peak_engine}`"
            "（点侧栏①修改）。"
        )
        _jm_opts = ["merton", "none"]
        if simple_mode and (start_cfg.get("jump_model") or (rec and rec.jump_model)):
            jump_model = str(
                start_cfg.get("jump_model")
                or (rec.jump_model if rec else "merton")
            )
            st.caption(
                f"跳跃模型：系统推荐 `{jump_model}`（简洁模式；改用专家设置可手选）。"
            )
        else:
            _jm_default = getattr(base, "jump_model", "merton")
            _jm_idx = _jm_opts.index(_jm_default) if _jm_default in _jm_opts else 0
            jump_model = st.selectbox(
                "跳跃模型 jump_model",
                _jm_opts,
                index=_jm_idx,
                help="merton=Cont–Tankov/Merton 复合泊松（对数正态跳跃）；none=关闭跳跃",
            )
        jump_compensate = st.checkbox(
            "Merton 补偿子 jump_compensate",
            value=bool(getattr(base, "jump_compensate", False)),
            help="开启后日度对数漂移减 λ(E[e^J]−1)Δt；默认关以保持旧行为",
        )
        _vr_opts = ["none", "antithetic"]
        if simple_mode and (
            start_cfg.get("variance_reduction")
            or (rec and rec.variance_reduction)
        ):
            variance_reduction = str(
                start_cfg.get("variance_reduction")
                or (rec.variance_reduction if rec else "antithetic")
            )
            st.caption(
                f"方差缩减：系统推荐 `{variance_reduction}`（简洁模式）。"
            )
        else:
            _vr_default = "none"
            # Prefer Stage-1 auto_tune / recommended_variance_reduction when present on weights
            _vr_cand = getattr(base, "recommended_variance_reduction", None)
            if not _vr_cand and isinstance(getattr(base, "calibration", None), dict):
                _vr_cand = base.calibration.get("recommended_variance_reduction")
            if _vr_cand in _vr_opts:
                _vr_default = str(_vr_cand)
            _vr_idx = _vr_opts.index(_vr_default)
            variance_reduction = st.selectbox(
                "方差缩减 variance_reduction",
                _vr_opts,
                index=_vr_idx,
                help="none=当前行为；antithetic=对扩散增量做反变量配对（常用于降MC方差）",
            )
        if simple_mode:
            _cm = str(
                start_cfg.get("cluster_method")
                or (rec.cluster_method if rec else "jaccard")
            )
            st.caption(f"事件聚类：系统推荐 `{_cm}`（简洁模式）。")
        if peak_engine == "brownian_bridge":
            st.info(
                "连续峰值：日端点之间用反射原理 / 布朗桥（Shreve II）抽取路径内最大值；"
                "复合泊松跳跃不计。"
            )
            if jump_model == "merton":
                st.warning(
                    "若情景 E[jumps]>0，跳跃会被忽略；"
                    "需要跳跃加厚峰值尾部时请改用 path_max。"
                )
            _jc_warn = bb_jump_compensate_warning(
                peak_engine=peak_engine, jump_compensate=bool(jump_compensate)
            )
            if _jc_warn:
                st.warning(_jc_warn)
        st.caption(
            "切换引擎后请重新「开始设置」再点「运行分析」；"
            "结果页「本次分析审计」会显示实际 peak_engine / jump_model。"
        )
        st.caption("分档边界请在主区「概率区间」设置（相对现价涨幅% 或绝对汇率价位）。")
        use_rel = True
        cuts = tuple(base.bucket_pct_cuts)

    # ④ 映射
    with st.sidebar.expander(t("side.map"), expanded=False):
        a = st.slider("a：S→漂移", 0.0, 0.05, float(base.score_to_mu_a), 0.001, format="%.3f")
        b = st.slider("b：|S|→波动", 0.0, 0.15, float(base.score_to_sigma_b), 0.001, format="%.3f")
        logit_scale = st.slider("证据→情景 logit", 0.0, 0.3, float(base.evidence_logit_scale), 0.01)
        temperature = st.slider("情景温度", 0.3, 2.5, float(base.scenario_temperature), 0.1)
        max_shift = st.slider("单情景最大位移", 0.0, 0.4, float(base.max_scenario_shift), 0.01)

    # ⑤ 情景：用下拉选一个，避免一长串滑块
    with st.sidebar.expander(t("side.scenario"), expanded=False):
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
        cur["ej"] = st.slider(
            "E[jumps]（窗口内期望跳跃次数）",
            0.0,
            3.0,
            cur["ej"],
            0.05,
            key=f"ej_{focus}",
            help="horizon E[N_T]；日强度 λ_daily=E[jumps]/交易日 = λ_ann·Δt，Δt=1/252",
        )
        cur["jm"] = st.slider("jump μ_J", -0.03, 0.03, cur["jm"], 0.001, key=f"jm_{focus}")
        cur["js"] = st.slider("jump σ_J", 0.001, 0.03, cur["js"], 0.001, key=f"js_{focus}")
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
    with st.sidebar.expander(t("side.evidence"), expanded=False):
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
    with st.sidebar.expander(t("side.rubric"), expanded=False):
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
    with st.sidebar.expander(t("side.sources"), expanded=False):
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
        peak_engine=str(peak_engine),
        jump_model=str(jump_model),
        jump_compensate=bool(jump_compensate),
        scenarios=scenarios,
        evidence=evidence,
    ), {
        "use_news": use_news,
        "ai_research": ai_research,
        "keep_templates": keep_templates,
        "template_policy": template_policy,
        "max_news_ev": int(max_news_ev),
        "classify_mode": classify_mode,
        "fetch_fulltext": fetch_fulltext,
        "use_label_learned_strength": use_label_learned,
        "pause_uncertain": bool(pause_uncertain),
        "variance_reduction": str(variance_reduction),
        "cluster_method": str(
            start_cfg.get("cluster_method")
            or (rec.cluster_method if rec else "jaccard")
        ),
        "llm_key": "",
        "llm_base": "",
        "llm_model": "",
    }


def _init_label_edits_from_df(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    edits: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        sid = str(row.get("statement_id") or "")
        if not sid:
            continue
        edits[sid] = {
            "human_direction": str(row.get("human_direction") or ""),
            "human_category": str(row.get("human_category") or ""),
            "agree": str(row.get("agree") or ""),
        }
    return edits


def _clear_label_widget_keys(sids: list[str] | None = None) -> None:
    """Drop Streamlit selectbox state so prefill/clear actually shows new values."""
    prefixes = ("hd_", "hc_", "ag_")
    if sids:
        for sid in sids:
            for p in prefixes:
                st.session_state.pop(f"{p}{sid}", None)
        return
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(prefixes):
            st.session_state.pop(k, None)


def _apply_human_label_recompute(filled: pd.DataFrame, *, is_demo: bool) -> dict:
    """
    Override evidence directions from human labels, recompute S + scenario weights,
    optionally re-run MC and refresh session diagnostics / probs.
    """
    from fx_report.model.label_audit import (
        apply_human_labels_to_evidence,
        recompute_score_and_scenarios,
    )
    from fx_report.model.monte_carlo import enforce_math_floor, run_mixture_monte_carlo

    if is_demo:
        return {"ok": False, "error": "练习样例不重算真实运行权重。"}

    evidence_rows = st.session_state.get("last_auto_evidence") or []
    base_scenarios = st.session_state.get("last_base_scenarios") or []
    mapping = st.session_state.get("last_weight_mapping") or {}
    diag = st.session_state.get("last_diag") or {}

    if not evidence_rows:
        return {"ok": False, "error": "无本次运行证据，无法重算。"}
    if not base_scenarios or not mapping:
        return {
            "ok": False,
            "error": "缺少情景/映射快照（请重新「运行分析」后再用人工标注重算）。",
        }

    overridden, n_overridden = apply_human_labels_to_evidence(evidence_rows, filled)
    if n_overridden == 0:
        return {"ok": False, "error": "没有可用的人工方向（unclear/空不覆盖）。请先标注。"}

    reco = recompute_score_and_scenarios(
        overridden,
        base_scenarios,
        score_to_mu_a=float(mapping["score_to_mu_a"]),
        score_to_sigma_b=float(mapping["score_to_sigma_b"]),
        evidence_logit_scale=float(mapping["evidence_logit_scale"]),
        scenario_temperature=float(mapping["scenario_temperature"]),
        max_scenario_shift=float(mapping["max_scenario_shift"]),
    )

    # Update lightweight evidence rows for UI tables
    id_to_dir = {
        str(e.get("statement_id") or e.get("id") or ""): e.get("dir", e.get("direction"))
        for e in overridden
    }
    id_to_cat = {
        str(e.get("statement_id") or e.get("id") or ""): e.get("category")
        for e in overridden
    }
    refreshed = []
    for row in evidence_rows:
        r = dict(row)
        sid = str(r.get("statement_id") or r.get("id") or "")
        if sid in id_to_dir and id_to_dir[sid] is not None:
            r["dir"] = id_to_dir[sid]
            r["direction"] = id_to_dir[sid]
        if sid in id_to_cat and id_to_cat[sid]:
            r["category"] = id_to_cat[sid]
        refreshed.append(r)
    st.session_state["last_auto_evidence"] = refreshed

    # Update diagnostics
    new_diag = dict(diag)
    new_diag["score_S"] = reco["score_S"]
    new_diag["mu_annual_shift"] = reco["mu_annual_shift"]
    new_diag["sigma_mult_extra"] = reco["sigma_mult_extra"]
    new_diag["scenarios_adjusted"] = [s.__dict__ for s in reco["scenarios_adjusted"]]
    new_diag["human_label_override"] = {
        "n_overridden": n_overridden,
        "score_S": reco["score_S"],
    }

    mc_note = ""
    market = diag.get("market") or {}
    edges = diag.get("bucket_edges") or st.session_state.get("last_edges")
    if (
        market.get("spot") is not None
        and market.get("sigma_daily") is not None
        and edges
    ):
        try:
            edges_t = tuple(float(x) for x in edges)
            mc = run_mixture_monte_carlo(
                spot=float(market["spot"]),
                sigma_daily_base=float(market["sigma_daily"]),
                scenarios=reco["scenarios_adjusted"],
                trading_days=int(mapping.get("trading_days") or 66),
                n_sims=int(mapping.get("n_sims") or 10_000),
                seed=int(mapping.get("seed") or 42),
                bucket_edges=edges_t,
                mu_annual_shift=float(reco["mu_annual_shift"]),
                sigma_mult_extra=float(reco["sigma_mult_extra"]),
                peak_engine=str(mapping.get("peak_engine") or "path_max"),
                variance_reduction=str(mapping.get("variance_reduction") or "none"),
                jump_model=str(mapping.get("jump_model") or "merton"),
                jump_compensate=bool(mapping.get("jump_compensate") or False),
            )
            probs = enforce_math_floor(mc.raw_probs, float(market["spot"]), edges_t)
            new_diag["raw_probs"] = mc.raw_probs
            new_diag["calibrated_probs"] = probs
            new_diag["percentiles"] = mc.percentiles
            new_diag["scenario_path_counts"] = mc.scenario_counts
            st.session_state["last_probs"] = probs
            mc_note = f"；已重跑 MC（n={mc.n_sims:,}）"
        except Exception as exc:
            mc_note = f"；MC 重跑失败（权重已更新）：{exc}"

    st.session_state["last_diag"] = new_diag
    st.session_state["human_label_recomputed"] = {
        "applied": True,
        "n_overridden": n_overridden,
        "score_S": reco["score_S"],
        "mu_annual_shift": reco["mu_annual_shift"],
        "sigma_mult_extra": reco["sigma_mult_extra"],
    }
    return {
        "ok": True,
        "n_overridden": n_overridden,
        "score_S": reco["score_S"],
        "note": mc_note,
    }


def render_label_audit_section(
    *,
    pair: str,
    bullish: str,
    evidence_rows: list[dict] | None,
    news_meta: dict | None = None,
    diag: dict | None = None,
) -> None:
    """In-app 证据人工标注：选方向/类别、自动 agree、落盘 CSV。"""
    from fx_report.config.api_config import has_news_api, load_config
    from fx_report.model.label_audit import (
        AGREE_VALUES,
        AGREE_ZH,
        HUMAN_DIRECTIONS,
        LABEL_CATEGORIES,
        agree_rate_stats,
        aggregate_spotcheck_stats,
        category_label,
        compute_agree,
        demo_evidence_rows,
        direction_label,
        empty_reason_message,
        evidence_rows_to_audit_df,
        help_markdown,
        label_audit_path,
        load_label_audit,
        load_spotcheck_stats,
        merge_human_labels,
        normalize_direction,
        prefill_from_model,
        railway_env_checklist_markdown,
        save_label_audit,
        save_spotcheck_stats,
    )

    # Anchor + prominent header (must stay above long report / charts)
    st.markdown('<div id="label-audit-section"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.subheader(t("audit.title"))
    st.caption(t("audit.caption"))

    news_meta = news_meta or {}
    diag = diag or {}
    counts = diag.get("evidence_counts") or {
        "fetched": news_meta.get("fetched", 0),
        "kept": news_meta.get("kept", 0),
        "classified": news_meta.get("classified", 0),
        "evidence_n": news_meta.get("evidence_n", 0),
    }
    quality = str(
        diag.get("evidence_quality") or news_meta.get("evidence_quality") or ""
    )

    with st.expander(t("audit.how"), expanded=True):
        st.markdown(help_markdown(pair=pair, bullish=bullish))
        st.caption(
            t(
                "audit.allowed",
                dirs=" / ".join(HUMAN_DIRECTIONS),
                agrees=" / ".join(AGREE_VALUES),
            )
        )

    # Source of statements: run evidence, or demo practice set
    use_demo = bool(st.session_state.get("label_audit_use_demo"))
    rows = list(evidence_rows or [])
    if use_demo:
        rows = demo_evidence_rows(pair=pair, bullish=bullish)

    if not rows:
        cfg = load_config()
        st.warning(
            empty_reason_message(
                evidence_n=int(counts.get("evidence_n") or 0),
                fetched=int(counts.get("fetched") or 0),
                quality=quality,
                news_keys_present=has_news_api(cfg),
            )
        )
        st.markdown(t("audit.no_rows"))
        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                t("audit.load_demo"),
                key="btn_load_demo_labels",
                type="primary",
            ):
                st.session_state["label_audit_use_demo"] = True
                st.session_state.pop("label_edits", None)
                st.session_state.pop("label_edit_fp", None)
                _clear_label_widget_keys()
                st.rerun()
        with c2:
            st.info(
                "要看**真实**新闻证据 / 更多 References：侧栏「API / AI Key」填写 "
                "`NEWSAPI_KEY` / `FINNHUB_API_KEY`（无 Key 也会试央行 + Google News RSS）。"
                "DeepSeek/LLM 只做证据判定，不会凭空增加链接。"
                "保存后重新运行分析。"
            )
        with st.expander("Railway / 环境变量检查清单", expanded=True):
            st.markdown(railway_env_checklist_markdown(news_keys_present=has_news_api(cfg)))
        return

    if use_demo:
        st.info(t("audit.demo_info"))
        if st.button(t("audit.exit_demo"), key="btn_exit_demo_labels"):
            st.session_state["label_audit_use_demo"] = False
            st.session_state.pop("label_edits", None)
            st.session_state.pop("label_edit_fp", None)
            _clear_label_widget_keys()
            st.rerun()

    base_df = evidence_rows_to_audit_df(rows)
    fp = "|".join(str(x) for x in base_df["statement_id"].tolist()) + f"|{pair}|demo={use_demo}"
    if st.session_state.get("label_edit_fp") != fp:
        # Try merge existing on-disk labels for same pair+date
        path_try = label_audit_path(pair)
        disk = load_label_audit(path_try) if path_try.exists() else None
        edits = _init_label_edits_from_df(base_df)
        if disk is not None and not disk.empty:
            by_id = {str(r["statement_id"]): r for _, r in disk.iterrows()}
            for sid, lab in edits.items():
                if sid in by_id:
                    lab["human_direction"] = str(by_id[sid].get("human_direction") or "")
                    lab["human_category"] = str(by_id[sid].get("human_category") or "")
                    lab["agree"] = str(by_id[sid].get("agree") or "")
        st.session_state["label_edits"] = edits
        st.session_state["label_edit_fp"] = fp
        _clear_label_widget_keys([str(x) for x in base_df["statement_id"].tolist()])
        # Sync widget keys from edits so selectboxes show disk/prefill values
        for sid, lab in edits.items():
            if lab.get("human_direction"):
                st.session_state[f"hd_{sid}"] = lab["human_direction"]
            if lab.get("human_category"):
                st.session_state[f"hc_{sid}"] = lab["human_category"]
            if lab.get("agree"):
                st.session_state[f"ag_{sid}"] = lab["agree"]

    edits: dict[str, dict[str, str]] = st.session_state.setdefault("label_edits", {})

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button(t("audit.prefill"), key="btn_prefill_labels", help=t("audit.prefill.help")):
            pref = prefill_from_model(base_df)
            edits_new = _init_label_edits_from_df(pref)
            st.session_state["label_edits"] = edits_new
            _clear_label_widget_keys([str(x) for x in base_df["statement_id"].tolist()])
            for sid, lab in edits_new.items():
                st.session_state[f"hd_{sid}"] = lab["human_direction"]
                st.session_state[f"hc_{sid}"] = lab["human_category"]
                st.session_state[f"ag_{sid}"] = lab["agree"]
            st.rerun()
    with b2:
        save_clicked = st.button(t("audit.save"), key="btn_save_labels")
    with b3:
        clear_clicked = st.button(t("audit.clear"), key="btn_clear_labels")

    if clear_clicked:
        st.session_state["label_edits"] = _init_label_edits_from_df(base_df)
        _clear_label_widget_keys([str(x) for x in base_df["statement_id"].tolist()])
        st.rerun()

    dir_opts = [""] + list(HUMAN_DIRECTIONS)
    cat_opts = [""] + list(LABEL_CATEGORIES)
    agree_opts = [""] + list(AGREE_VALUES)

    for _, row in base_df.iterrows():
        sid = str(row["statement_id"])
        lab = edits.setdefault(
            sid,
            {"human_direction": "", "human_category": "", "agree": ""},
        )
        title = str(row.get("title") or "")
        url = str(row.get("url") or "")
        md = normalize_direction(row.get("model_direction", ""))
        mc = str(row.get("model_category") or "")

        with st.container(border=True):
            head = f"**{sid}** · {title}"
            if url:
                st.markdown(f"{head}  \n[{t('audit.open_link')}]({url})")
            else:
                st.markdown(head)
            c_zh = category_label(mc) if mc else "—"
            md_zh = direction_label(md) if md else "未判定"
            st.caption(
                t(
                    "audit.model_ro",
                    md=md or "—",
                    md_zh=md_zh,
                    cat=c_zh,
                )
            )

            prev_d = lab.get("human_direction", "")
            prev_a = lab.get("agree", "")
            c_d, c_c, c_a = st.columns(3)
            with c_d:
                di = dir_opts.index(prev_d) if prev_d in dir_opts else 0
                new_d = st.selectbox(
                    t("audit.human_dir"),
                    dir_opts,
                    index=di,
                    format_func=lambda x: t("audit.unset") if x == "" else direction_label(x),
                    key=f"hd_{sid}",
                )
            with c_c:
                prev_c = lab.get("human_category", "")
                ci = cat_opts.index(prev_c) if prev_c in cat_opts else 0
                new_c = st.selectbox(
                    t("audit.human_cat"),
                    cat_opts,
                    index=ci,
                    format_func=lambda x: t("audit.unset") if x == "" else category_label(x),
                    key=f"hc_{sid}",
                )
            with c_a:
                # Auto-suggest agree from model vs human direction
                suggested = compute_agree(md, new_d) if new_d else ""
                if new_d and (not prev_a or prev_d != new_d):
                    cur_ag = suggested
                else:
                    cur_ag = prev_a or suggested
                ai = agree_opts.index(cur_ag) if cur_ag in agree_opts else 0
                new_a = st.selectbox(
                    t("audit.agree"),
                    agree_opts,
                    index=ai,
                    format_func=lambda x: (
                        t("audit.unset")
                        if x == ""
                        else f"{x} — {AGREE_ZH.get(x, x)}"
                    ),
                    key=f"ag_{sid}",
                )
            lab["human_direction"] = new_d
            lab["human_category"] = new_c
            lab["agree"] = new_a if new_a else (suggested if new_d else "")

    filled = merge_human_labels(base_df, edits)
    # Recompute agree column for consistency in CSV
    for i, r in filled.iterrows():
        hd = str(r.get("human_direction") or "")
        if hd:
            filled.at[i, "agree"] = compute_agree(r.get("model_direction", ""), hd)

    csv_bytes = filled.to_csv(index=False).encode("utf-8")
    st.session_state["last_label_audit_csv"] = filled.to_csv(index=False)

    stats = agree_rate_stats(filled, is_demo=use_demo)
    st.session_state["last_label_agree_stats"] = stats
    # Alias for audit panel / spot-check wording
    if stats.get("agree_rate") is not None:
        stats = {**stats, "抽检准确率": stats["agree_rate"]}

    out_path = label_audit_path(pair)
    n_done = int((filled["human_direction"].astype(str).str.len() > 0).sum())
    st.caption(t("audit.filled", n=n_done, total=len(filled), path=out_path))

    # 抽检准确率 (= agree_rate) — visible whenever labels exist
    if stats["has_labels"]:
        m1, m2, m3, m4 = st.columns(4)
        if stats["agree_rate"] is not None:
            m1.metric(t("audit.spot_rate"), f"{100 * float(stats['agree_rate']):.0f}%")
            m1.caption(t("audit.spot_rate.cap"))
        else:
            m1.metric(t("audit.spot_rate"), "—")
        m2.metric("一致 yes", int(stats["n_yes"]))
        m3.metric("不一致 no", int(stats["n_no"]))
        m4.metric("unsure", int(stats["n_unsure"]))
        st.caption(stats["caption"])
    else:
        st.caption(stats["caption"])
        saved_sc = load_spotcheck_stats(pair)
        if saved_sc and saved_sc.get("agree_rate") is not None:
            st.info(
                f"磁盘已有抽检准确率 "
                f"{100 * float(saved_sc['agree_rate']):.0f}% "
                f"（{saved_sc.get('as_of', '')} · yes={saved_sc.get('n_yes')} / "
                f"no={saved_sc.get('n_no')}）"
            )

    # Aggregate across all label_audit files (non-demo)
    agg = aggregate_spotcheck_stats()
    if agg.get("has_labels") and agg.get("agree_rate") is not None and not use_demo:
        st.caption(
            f"全部已保存标注合计抽检准确率："
            f"{100 * float(agg['agree_rate']):.0f}% "
            f"（{agg.get('n_yes', 0)}/{int(agg.get('n_yes', 0)) + int(agg.get('n_no', 0))}）"
        )

    # Stage 3 learn status
    from fx_report.model.label_learn import (
        MIN_LABELS_FOR_LEARN,
        fit_label_learned_params,
    )

    learned_preview = fit_label_learned_params()
    if learned_preview.ready:
        st.success(f"标签学习可用：{learned_preview.message}（侧栏勾选即可应用到下次运行）")
    else:
        st.caption(
            f"标签学习（Stage 3）：{learned_preview.message or f'标注不足，需至少 {MIN_LABELS_FOR_LEARN} 条'}"
        )

    # Human-label → recompute S / scenario weights for current run
    st.markdown("##### 标注重算")
    use_human_reweight = st.checkbox(
        "用人工标注重算权重",
        key="use_human_reweight",
        help=(
            "用 human_direction 覆盖模型方向，重算证据分 S 与情景权重；"
            "有行情快照时一并重跑 MC。练习样例不会改真实运行。"
        ),
        disabled=use_demo,
    )
    apply_reweight = st.button(
        "应用人工标注重算",
        key="btn_apply_human_reweight",
        disabled=use_demo or not use_human_reweight,
        help="需勾选上方开关，且至少有一条明确人工方向（非 unclear）",
    )

    if save_clicked:
        saved = save_label_audit(filled, out_path)
        sc_path = save_spotcheck_stats(stats, pair)
        st.success(f"已保存：{saved}；抽检准确率 → `{sc_path.name}`")
        if use_human_reweight and not use_demo and stats["has_labels"]:
            apply_reweight = True
        # Refresh learned params preview after save
        try:
            from fx_report.model.label_learn import save_label_learned_params

            lp = fit_label_learned_params()
            if lp.ready:
                save_label_learned_params(lp)
        except Exception:
            pass

    if apply_reweight and use_human_reweight and not use_demo:
        result = _apply_human_label_recompute(filled, is_demo=use_demo)
        if result.get("ok"):
            st.success(
                f"已用人工标注覆盖 {result['n_overridden']} 条方向，"
                f"S={result['score_S']:+.3f}{result.get('note') or ''}"
            )
            st.rerun()
        else:
            st.warning(result.get("error") or "重算失败")

    pair_safe = pair.replace("/", "")
    st.download_button(
        "下载当前标注 CSV（label_audit）",
        csv_bytes,
        file_name=out_path.name,
        mime="text/csv",
        key="dl_label_audit_filled",
        help="含已填/部分填写的 human_* 与 agree",
    )


def main() -> None:
    init_language()
    if not _require_password():
        return

    # Language toggle at top of sidebar (persistent via session + ?lang=)
    render_language_selector(location="sidebar", key="sidebar_ui_lang_select")

    # First visit: open 开始设置 once (no silent defaults)
    if "start_cfg" not in st.session_state and not st.session_state.get("_start_setup_shown"):
        st.session_state["_open_start_setup"] = True
        st.session_state["_start_setup_shown"] = True

    # Keep calling while open — dialog widgets need the function invoked each rerun
    if st.session_state.get("_open_start_setup"):
        start_setup_dialog()

    if st.session_state.get("_show_missing_start"):
        missing_start_dialog(list(st.session_state.get("_missing_start_labels") or []))

    display_spec, bullish = pick_pair_in_sidebar()

    if display_spec is None:
        st.title("FX Analyse")
        st.warning(t("main.need_start"))
        with st.expander("API / AI Key（按需填写，可全空）", expanded=False):
            render_api_settings_panel()
        if "last_report" not in st.session_state:
            return
        # Allow viewing a previous report even if setup was cleared
        display_spec = get_pair(
            (st.session_state.get("last_diag") or {}).get("market", {}).get("pair")
            or "USD/AUD"
        )
        bullish = st.session_state.get("start_cfg", {}).get("bullish_currency") or display_spec.base

    if st.session_state.get("pair_key") != display_spec.pair:
        st.session_state["pair_key"] = display_spec.pair
        st.session_state.pop("last_report", None)
        st.session_state.pop("scenario_edits", None)
        st.session_state.pop("evidence_edits", None)
        st.session_state.pop("hitl_checkpoint", None)
        st.session_state.pop("hitl_choices", None)

    bullish_ok = bullish in (display_spec.base, display_spec.quote)
    analysis_spec = (
        resolve_pair_for_bullish(display_spec, bullish) if bullish_ok else display_spec
    )

    if bullish_ok and st.session_state.get("analysis_pair_key") != analysis_spec.pair:
        st.session_state["analysis_pair_key"] = analysis_spec.pair
        st.session_state.pop("last_report", None)

    base = default_weights(analysis_spec)
    # Apply peak_engine from start setup before calib overlay (calib may also set it)
    start_cfg = st.session_state.get("start_cfg") or {}
    if start_cfg.get("peak_engine") in ("path_max", "brownian_bridge"):
        base.peak_engine = str(start_cfg["peak_engine"])
    if start_cfg.get("jump_model") in ("merton", "none"):
        base.jump_model = str(start_cfg["jump_model"])
    if start_cfg.get("variance_reduction") in ("none", "antithetic"):
        try:
            setattr(base, "recommended_variance_reduction", str(start_cfg["variance_reduction"]))
        except Exception:
            pass

    default_cal = resolve_calibrated_params_path(analysis_spec.pair)
    # Explicit choice from 开始设置 — never auto-check because a JSON exists
    use_cal = start_cfg.get("use_calibrated")
    if use_cal is True:
        st.sidebar.caption(t("side.cal.use"))
    elif use_cal is False:
        st.sidebar.caption(t("side.cal.skip"))
    else:
        st.sidebar.caption(t("side.cal.unset"))

    cal_path: str | None = None
    cal_label = "default"
    cal_loaded = False
    if use_cal is True and default_cal is not None:
        from fx_report.model.calibrate import apply_calibrated_params, load_calibrated_params

        apply_calibrated_params(base, load_calibrated_params(default_cal))
        cal_path = str(default_cal)
        cal_label = str(default_cal)
        cal_loaded = True
        st.sidebar.caption(f"已加载校准参数 · {default_cal.name}")
        # Keep user's explicit peak_engine over calibrated default when set in 开始设置
        if start_cfg.get("peak_engine") in ("path_max", "brownian_bridge"):
            base.peak_engine = str(start_cfg["peak_engine"])
        if start_cfg.get("jump_model") in ("merton", "none"):
            base.jump_model = str(start_cfg["jump_model"])
    elif use_cal is True:
        st.sidebar.caption("未找到校准 JSON，用默认先验")
    elif use_cal is False:
        st.sidebar.caption("默认先验（开始设置选不使用校准）")

    weights, news_opts = sidebar_weights(base, analysis_spec.pair)

    with st.sidebar.expander(t("side.label_audit"), expanded=False):
        if "last_report" in st.session_state:
            n_ev = len(st.session_state.get("last_auto_evidence") or [])
            st.markdown(t("side.label_audit.has_run"))
            st.caption(
                t("side.label_audit.n", n=n_ev)
                + (t("side.label_audit.demo_hint") if n_ev == 0 else "")
            )
            st.markdown(t("side.label_audit.jump"))
        else:
            st.caption(t("side.label_audit.need_run"))

    with st.sidebar.expander(t("side.todo"), expanded=True):
        from fx_report.config.api_config import has_news_api, load_config
        from fx_report.model.label_learn import MIN_LABELS_FOR_LEARN, fit_label_learned_params

        cfg_side = load_config()
        news_ok = has_news_api(cfg_side)
        learned_side = fit_label_learned_params()
        n_lab = int(learned_side.n_labeled or 0)
        st.markdown(
            f"""
**仍需你在 Railway / 本机完成：**

- {'✅' if news_ok else '☐'} 填 `NEWSAPI_KEY` / `FINNHUB_API_KEY`（**决定 References 条数**；LLM/DeepSeek 可选，只做判定）
- ✅ 访问口令已启用（默认 `uniocean`；可用 `APP_PASSWORD` 覆盖）
- {'✅' if n_lab >= MIN_LABELS_FOR_LEARN else '☐'} 实盘标注 ≥{MIN_LABELS_FOR_LEARN} 条（当前 **{n_lab}**）

无 Key 时仍会试央行 RSS + Google News；只填 DeepSeek **不会**自动多出参考链接。标注够后可勾「使用标签学习到的强度」。
详见 `docs/deploy-docker.md` / `docs/label_audit.md`。
""".strip()
        )

    st.title(f"FX Analyse · {display_spec.pair}")
    if bullish_ok:
        st.caption(
            t("main.caption_ok", pair=analysis_spec.pair, bull=bullish)
        )
    else:
        st.caption(t("main.caption_need"))

    render_calib_trust_panel(
        analysis_spec.pair, cal_loaded=cal_loaded, cal_path=cal_path
    )
    render_cross_pair_quality_board(current_pair=analysis_spec.pair)
    render_current_pair_reliability_board(current_pair=analysis_spec.pair)
    render_replay_summary_board(current_pair=analysis_spec.pair)

    with st.expander("API / AI Key（按需填写，可全空）", expanded=False):
        api_opts = render_api_settings_panel()

    if not bullish_ok:
        st.warning(t("main.need_start_short"))
        if "last_report" not in st.session_state:
            return

    spot_row: dict | None = None
    bucket_mode_choice: str | None = None
    if bullish_ok:
        spot_row = render_spot_panel(
            analysis_spec, bullish, lookback_days=weights.vol_lookback_days
        )
        spot_val = float(spot_row["spot"]) if spot_row.get("ok") and spot_row.get("spot") is not None else None
        use_rel, pct_cuts, abs_edges = render_bucket_editor(
            base, spot_val, analysis_spec.pair
        )
        mode_key = f"bucket_mode::{analysis_spec.pair}"
        bucket_mode_choice = st.session_state.get(mode_key)
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
        c1, c2, c3, c4 = st.columns([1, 1, 1.0, 1.0])
        with c1:
            start = st.date_input(t("main.window_start"), value=date.today())
        with c2:
            end = st.date_input(t("main.window_end"), value=date.today() + timedelta(days=92))
        with c3:
            st.write("")
            can_run = bool(spot_row and spot_row.get("ok"))
            run = st.button(
                t("main.run"),
                type="primary",
                use_container_width=True,
                disabled=not can_run,
                help=None if can_run else t("main.run.need_spot"),
            )
        with c4:
            st.write("")
            compare = st.button(
                t("main.compare"),
                use_container_width=True,
                disabled=not can_run,
                help=t("main.compare.help"),
            )
        if not can_run:
            st.caption(t("main.no_spot"))
        elif "last_report" not in st.session_state and not run:
            st.info(t("main.ready_hint"))

        # —— 双引擎对比（MC-only，降采样）——
        if compare and can_run and spot_val is not None:
            missing_cmp = _missing_for_start_cfg(
                st.session_state.get("start_cfg"),
                bucket_mode=bucket_mode_choice,
            )
            if missing_cmp:
                st.session_state["_missing_start_labels"] = missing_cmp
                st.session_state["_show_missing_start"] = True
                missing_start_dialog(missing_cmp)
            else:
                with st.spinner("双引擎对比中（降采样 MC，不含新闻重跑）…"):
                    try:
                        from fx_report.model.backtest import compare_peak_engines
                        from fx_report.model.weights import resolve_bucket_edges

                        snap = fetch_market(analysis_spec, lookback_days=weights.vol_lookback_days)
                        edges_cmp = resolve_bucket_edges(weights, float(snap.spot))
                        cmp_n = min(int(weights.n_sims), 8_000)
                        cmp = compare_peak_engines(
                            float(snap.spot),
                            float(snap.sigma_daily),
                            weights,
                            edges_cmp,
                            n_sims=cmp_n,
                            seed=int(weights.seed),
                            variance_reduction=str(news_opts.get("variance_reduction") or "none"),
                        )
                        st.session_state["last_engine_compare"] = {
                            "n_sims": cmp["n_sims"],
                            "score_S": cmp["score_S"],
                            "table": cmp["table"].to_dict(orient="list"),
                            "note": cmp["note"],
                            "pair": analysis_spec.pair,
                        }
                    except Exception as exc:
                        st.session_state["last_engine_compare"] = None
                        st.warning(f"双引擎对比失败：{exc}")

        if st.session_state.get("last_engine_compare"):
            ec = st.session_state["last_engine_compare"]
            st.subheader("双引擎对比")
            st.caption(
                f"同一现价/分档/情景 · n_sims={ec.get('n_sims'):,}（对比模式降采样）· "
                f"S={ec.get('score_S', 0):+.2f} · {ec.get('note', '')}"
            )
            df_ec = pd.DataFrame(ec["table"])
            show = df_ec.copy()
            for col in ("path_max", "brownian_bridge", "delta_bb_minus_pm"):
                if col in show.columns:
                    show[col] = show[col].map(lambda x: f"{100 * float(x):.1f}%")
            st.dataframe(show, hide_index=True, use_container_width=True)
            chart_df = pd.DataFrame(
                {
                    "path_max": df_ec["path_max"].values,
                    "brownian_bridge": df_ec["brownian_bridge"].values,
                },
                index=df_ec["bucket"].values,
            )
            st.bar_chart(chart_df)

        # —— 历史回测（小样本 UI）——
        with st.expander("历史回测（argmax hit / Brier）", expanded=False):
            st.caption(
                "对 peak_samples 跑降采样 MC，核对预测档 vs 实现档。"
                "UI 最多 30 行；完整回测用 CLI：`python run_cli.py backtest --pair …`"
            )
            bt_cols = st.columns([1, 1, 2])
            with bt_cols[0]:
                bt_rows = st.number_input("回测行数", min_value=5, max_value=30, value=20, step=5)
            with bt_cols[1]:
                bt_sims = st.number_input("每次模拟", min_value=500, max_value=5000, value=1500, step=500)
            with bt_cols[2]:
                st.write("")
                run_bt = st.button("运行小回测", use_container_width=True)
            if run_bt:
                samples = Path("output") / f"peak_samples_{analysis_spec.pair.replace('/', '')}.csv"
                try:
                    from fx_report.model.backtest import run_backtest

                    if not samples.exists():
                        st.info("本地无 peak_samples，尝试 build-peaks（需网络）…")
                        from fx_report.model.history_peaks import export_peak_samples

                        export_peak_samples(
                            analysis_spec.pair,
                            out_dir="output",
                            horizon_days=weights.trading_days,
                            vol_lookback=weights.vol_lookback_days,
                        )
                    with st.spinner("历史回测中…"):
                        bt = run_backtest(
                            analysis_spec.pair,
                            out_dir="output",
                            n_sims=int(bt_sims),
                            max_rows=int(bt_rows),
                            seed=int(weights.seed),
                            peak_engine=getattr(weights, "peak_engine", None),
                            variance_reduction=str(news_opts.get("variance_reduction") or "none"),
                            verbose=False,
                        )
                    st.session_state["last_backtest"] = {
                        "hit_rate": bt.hit_rate_argmax,
                        "brier": bt.mean_brier,
                        "logloss": bt.mean_logloss,
                        "skill_brier": (bt.summary or {}).get("skill_brier"),
                        "skill_logloss": (bt.summary or {}).get("skill_logloss"),
                        "reliability_ece": (bt.summary or {}).get("reliability_ece"),
                        "reliability_buckets": (bt.summary or {}).get("reliability_buckets"),
                        "reliability_argmax": (bt.summary or {}).get("reliability_argmax"),
                        "n": bt.n_rows,
                        "params": bt.params_source,
                        "engine": bt.peak_engine,
                        "table": bt.table[
                            [
                                c
                                for c in (
                                    "asof",
                                    "spot",
                                    "realized_max",
                                    "pred_bucket",
                                    "true_bucket",
                                    "hit",
                                    "brier",
                                    "logloss",
                                )
                                if c in bt.table.columns
                            ]
                        ].to_dict(orient="list"),
                        "summary": bt.summary,
                    }
                except Exception as exc:
                    st.session_state["last_backtest"] = None
                    st.warning(
                        f"回测失败（无样本/无网络/参数问题均可忽略）：{exc}"
                    )
            if st.session_state.get("last_backtest"):
                bt = st.session_state["last_backtest"]
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("argmax hit", f"{100 * bt['hit_rate']:.1f}%")
                m2.metric("mean Brier", f"{bt['brier']:.4f}")
                m3.metric("mean log-loss", f"{bt['logloss']:.4f}")
                sk = bt.get("skill_brier")
                m4.metric(
                    "Skill（Brier）",
                    f"{float(sk):.4f}" if sk is not None and sk == sk else "—",
                )
                m5.metric("n", bt["n"])
                st.caption(
                    f"engine=`{bt.get('engine')}` · params=`{bt.get('params')}` · "
                    "评分规则：严格恰当 Brier / log-loss；Skill vs 气候频率基准"
                )
                st.dataframe(pd.DataFrame(bt["table"]), hide_index=True, use_container_width=True)
                rel_b = bt.get("reliability_buckets") or []
                rel_a = bt.get("reliability_argmax") or []
                if rel_b or rel_a or bt.get("reliability_ece") is not None:
                    render_probability_reliability(
                        hold={
                            "reliability_buckets": rel_b,
                            "reliability_argmax": rel_a,
                            "reliability_ece": bt.get("reliability_ece"),
                        },
                        title="概率可靠性（本次回测）",
                        expanded=False,
                    )
                hold = (bt.get("summary") or {}).get("holdout_oos") or {}
                if hold.get("n"):
                    st.caption(
                        f"OOS holdout：n={int(hold['n'])}  "
                        f"brier={hold.get('brier', float('nan')):.4f}  "
                        f"skill_brier={hold.get('skill_brier', float('nan')):.4f}  "
                        f"logloss={hold.get('logloss', float('nan')):.4f}  "
                        f"hit={100 * hold.get('hit_rate', 0):.1f}%"
                    )
        with st.expander("历史时点回放", expanded=False):
            st.caption(
                "按历史 as_of 冻结跑完整流水线，再对照未来窗口实际最高价。"
                "UI 仅建议跑 2-5 个时点的小样本；若历史新闻仅能部分还原，会明确显示限制。"
            )
            st.info(
                "**历史回测自动省钱模式**：已强制关闭 AI 检索员与 Tavily/Brave 网页搜索，"
                "侧栏「AI 检索员」开关在此无效。证据优先 GDELT + 磁盘缓存"
                "（`output/.cache/gdelt/`），近窗才用 NewsAPI；AI/Tavily 仅用于当日 Live 报告。"
            )
            allow_hist_ai = st.checkbox(
                "允许历史启用 AI 检索（贵，且可能引入非历史信息）",
                value=False,
                key="replay_allow_historical_ai",
                help=(
                    "默认关闭。勾选后才会调用 AI 检索员 / Tavily，"
                    "可能把 as_of 之后的实时网页结果混进回放，仅调试用。"
                ),
            )
            if allow_hist_ai:
                st.warning(
                    "已开启昂贵覆盖：历史回放将调用 AI/Tavily，可能消耗配额且污染历史信息集。"
                )
            else:
                st.caption("🔒 历史回测已自动关闭 AI 检索员与网页搜索以省配额。")
            rp1, rp2, rp3, rp4 = st.columns(4)
            with rp1:
                replay_start = st.date_input(
                    "回放起点",
                    value=max(date.today() - timedelta(days=30), date(2024, 1, 1)),
                    key="replay_start",
                )
            with rp2:
                replay_end = st.date_input(
                    "回放终点",
                    value=max(date.today() - timedelta(days=7), date(2024, 2, 1)),
                    key="replay_end",
                )
            with rp3:
                replay_step = st.number_input("步长（日）", min_value=1, max_value=30, value=7, step=1)
            with rp4:
                replay_max_dates = st.number_input("最多日期数", min_value=2, max_value=5, value=3, step=1)
            rp5, rp6 = st.columns([1, 1])
            with rp5:
                replay_sims = st.number_input("每时点模拟", min_value=300, max_value=3000, value=800, step=100)
            with rp6:
                st.write("")
                run_replay = st.button("运行历史时点回放", use_container_width=True)
            if run_replay:
                replay_ai = resolve_replay_ai_research(
                    allow_historical_ai=bool(allow_hist_ai),
                    sidebar_ai_research=bool(news_opts.get("ai_research", True)),
                )
                try:
                    with st.spinner("历史时点回放中…"):
                        replay = run_replay_backtest(
                            display_spec.pair,
                            bullish_currency=bullish,
                            start_date=replay_start,
                            end_date=replay_end,
                            step_days=int(replay_step),
                            out_dir="output",
                            sims=int(replay_sims),
                            days=int(weights.trading_days),
                            seed=int(weights.seed),
                            lookback=int(weights.vol_lookback_days),
                            peak_engine=str(getattr(weights, "peak_engine", "path_max")),
                            variance_reduction=str(news_opts.get("variance_reduction") or "none"),
                            jump_model=str(getattr(weights, "jump_model", "merton")),
                            jump_compensate=bool(getattr(weights, "jump_compensate", False)),
                            mode=str(news_opts.get("classify_mode") or "hybrid"),
                            max_news=int(news_opts.get("max_news_ev") or 10),
                            keep_templates=bool(news_opts.get("keep_templates")),
                            template_policy=str(news_opts.get("template_policy") or "off"),
                            no_news=not bool(news_opts.get("use_news", True)),
                            no_fulltext=not bool(news_opts.get("fetch_fulltext", True)),
                            ai_research=replay_ai,
                            allow_historical_ai=bool(allow_hist_ai),
                            calibrated_params_path=cal_path if cal_loaded else None,
                            use_label_learned_strength=bool(news_opts.get("use_label_learned_strength")),
                            max_dates=int(replay_max_dates),
                            verbose=False,
                        )
                    st.session_state["last_replay_backtest"] = {
                        "summary": replay.summary,
                        "table": replay.table.to_dict(orient="list"),
                    }
                except Exception as exc:
                    st.session_state["last_replay_backtest"] = None
                    st.warning(f"历史时点回放失败：{exc}")
            if st.session_state.get("last_replay_backtest"):
                rp = st.session_state["last_replay_backtest"]
                summary = rp["summary"]
                table_df = pd.DataFrame(rp["table"])
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("命中率 argmax", f"{100 * float(summary.get('argmax_hit_rate', 0)):.1f}%")
                m2.metric("平均 Brier", f"{float(summary.get('mean_brier', float('nan'))):.4f}")
                m3.metric("平均 Skill", f"{float(summary.get('mean_skill_brier', float('nan'))):.4f}")
                m4.metric("时点数 n", int(summary.get("n_rows", 0)))
                # Observability: cheap mode / AI / sources / cache
                sample_meta = {}
                if not table_df.empty:
                    last = table_df.iloc[-1]
                    sample_meta = {
                        "cheap_historical": summary.get("cheap_historical", True),
                        "allow_historical_ai": summary.get("allow_historical_ai", False),
                        "gdelt_hits": int(last["gdelt_hits"]) if "gdelt_hits" in table_df.columns else None,
                        "newsapi_hits": int(last["newsapi_hits"]) if "newsapi_hits" in table_df.columns else None,
                        "inbox_dated_hits": (
                            int(last["inbox_dated_hits"]) if "inbox_dated_hits" in table_df.columns else None
                        ),
                        "gdelt_from_cache": (
                            bool(last["gdelt_from_cache"]) if "gdelt_from_cache" in table_df.columns else False
                        ),
                        "newsapi_from_cache": (
                            bool(last["newsapi_from_cache"]) if "newsapi_from_cache" in table_df.columns else False
                        ),
                        "providers_used": (
                            [
                                p
                                for p in str(
                                    last["providers_used"] if "providers_used" in table_df.columns else ""
                                ).split(",")
                                if p
                            ]
                        ),
                    }
                st.caption(
                    "观测｜"
                    + format_cheap_historical_caption(
                        sample_meta,
                        cheap_historical=bool(summary.get("cheap_historical", True)),
                    )
                    + f"｜合计 GDELT命中={summary.get('gdelt_hits_sum', '—')} "
                    f"NewsAPI命中={summary.get('newsapi_hits_sum', '—')} "
                    f"缓存命中行={summary.get('cache_hit_rows', '—')}"
                )
                st.caption(
                    "磁盘缓存目录：`output/.cache/gdelt/`、`output/.cache/newsapi/`"
                    "（失败/空结果短 TTL，成功命中约 7 天；可用环境变量 "
                    "`FX_GDELT_CACHE` / `FX_NEWSAPI_CACHE` 改路径）。"
                )
                quality_counts = summary.get("historical_news_quality_counts") or {}
                if quality_counts.get("limited"):
                    st.warning(
                        "历史新闻保真度有限：本次至少有部分时点无法用真实日期过滤新闻完整回放，"
                        "结果已在 `historical_news_quality` 列中标记。"
                    )
                else:
                    st.info("历史新闻来源为可日期过滤路径（仍建议把它视为 best-effort，而非完美史料库）。")
                cols = [
                    c
                    for c in (
                        "as_of",
                        "spot",
                        "pred_bucket",
                        "true_bucket",
                        "argmax_hit",
                        "brier",
                        "skill_brier",
                        "evidence_n",
                        "historical_news_quality",
                        "cheap_historical",
                        "gdelt_hits",
                        "newsapi_hits",
                        "gdelt_from_cache",
                        "newsapi_from_cache",
                    )
                    if c in table_df.columns
                ]
                st.dataframe(table_df[cols], hide_index=True, use_container_width=True)
                st.caption(
                    f"输出文件：`{summary.get('csv', '')}` / `{summary.get('json', '')}`"
                )
    else:
        start = date.today()
        end = date.today() + timedelta(days=92)
        run = False

    # HITL: restore pending reviews across refresh (blocking section before results)
    if st.session_state.get("hitl_checkpoint"):
        render_hitl_uncertain_form()
        # While waiting for human, do not show a stale finished report underneath
        if st.session_state.get("hitl_checkpoint"):
            return

    if run:
        missing = _missing_for_start_cfg(
            st.session_state.get("start_cfg"),
            bucket_mode=bucket_mode_choice,
        )
        if missing:
            st.session_state["_missing_start_labels"] = missing
            st.session_state["_show_missing_start"] = True
            missing_start_dialog(missing)
            return
        if not bullish_ok:
            st.session_state["_missing_start_labels"] = ["看涨货币"]
            st.session_state["_show_missing_start"] = True
            missing_start_dialog(["看涨货币"])
            return
        if not spot_row or not spot_row.get("ok"):
            st.error("现价获取失败，无法运行分析。请先刷新现价。")
            return
        with st.spinner("流水线运行中（抓取与判定）…"):
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

            pipe_kwargs = dict(
                ticker=None if display_spec.pair in list_pairs() else display_spec.symbol_code,
                invert=display_spec.invert,
                sims=weights.n_sims,
                days=weights.trading_days,
                seed=weights.seed,
                lookback=weights.vol_lookback_days,
                variance_reduction=str(news_opts.get("variance_reduction") or "none"),
                mode=mode_cls,
                max_news=news_opts["max_news_ev"],
                keep_templates=news_opts["keep_templates"],
                template_policy=news_opts.get("template_policy") or "off",
                no_news=not news_opts["use_news"],
                no_fulltext=not bool(news_opts.get("fetch_fulltext", True)),
                ai_research=bool(news_opts.get("ai_research", True)),
                llm_cfg=llm_cfg,
                verbose=False,
                bullish_currency=bullish,
                model_weights=weights,
                calibrated_params_path=None,
                calibrated_params_label=cal_label,
                use_label_learned_strength=bool(
                    news_opts.get("use_label_learned_strength")
                ),
                max_uncertain=5,
            )

            pause = bool(news_opts.get("pause_uncertain", True))
            if pause:
                checkpoint = run_pipeline_phase_a(
                    display_spec.pair,
                    out_dir=None,
                    human_review_mode="pause",
                    **pipe_kwargs,
                )
                if checkpoint.pending_reviews:
                    st.session_state["hitl_checkpoint"] = checkpoint.to_session_dict()
                    st.session_state["hitl_choices"] = {
                        str(getattr(p, "evidence_id", "")): "skip"
                        for p in checkpoint.pending_reviews
                    }
                    st.session_state.pop("last_report", None)
                    st.info(
                        f"发现 {len(checkpoint.pending_reviews)} 条不确定证据，"
                        "请先确认方向后再生成报告。"
                    )
                    st.rerun()
                result = run_pipeline_phase_b(
                    checkpoint,
                    review_overrides=None,
                    out_dir="output",
                    verbose=False,
                )
            else:
                result = run_pipeline(
                    display_spec.pair,
                    out_dir="output",
                    human_review_mode="auto_skip",
                    **pipe_kwargs,
                )
                if isinstance(result, PipelineCheckpoint):
                    result = run_pipeline_phase_b(result, out_dir="output", verbose=False)

            if result.market.notes:
                st.caption("｜".join(result.market.notes[:3]))

            st.session_state[_spot_cache_key(analysis_spec.pair)] = {
                "ok": True,
                "pair": result.market.pair,
                "spot": float(result.market.spot),
                "source": result.market.source,
                "asof": result.market.asof,
                "notes": list(result.market.notes[:3]),
                "error": None,
            }
            _store_pipeline_result(result)


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

    # —— 本次分析审计（prominent）——
    news_meta = st.session_state.get("last_news_meta") or diag.get("news_meta") or {}
    peak_eng = diag.get("peak_engine") or news_meta.get("peak_engine") or "path_max"
    jump_mdl = diag.get("jump_model") or news_meta.get("jump_model") or "merton"
    jump_comp = bool(diag.get("jump_compensate") or news_meta.get("jump_compensate") or False)
    bb_caveat = diag.get("bb_jumps_caveat") or news_meta.get("bb_jumps_caveat")
    cal_used = diag.get("calibrated_params") or news_meta.get("calibrated_params") or "default"
    cal_is_default = str(cal_used).strip().lower() in {"default", "", "none"}
    cal_zh = "默认先验" if cal_is_default else f"已加载校准参数（`{Path(str(cal_used)).name}`）"
    eq = diag.get("evidence_quality") or news_meta.get("evidence_quality") or "n/a"
    fb = bool(diag.get("fallback_templates") or news_meta.get("fallback_templates"))
    mode_used = news_meta.get("mode") or "n/a"
    counts = diag.get("evidence_counts") or {
        "fetched": news_meta.get("fetched", 0),
        "kept": news_meta.get("kept", 0),
        "classified": news_meta.get("classified", 0),
        "evidence_n": news_meta.get("evidence_n", 0),
    }
    sc_adj = diag.get("scenarios_adjusted") or []
    weight_bits = ", ".join(
        f"{s.get('name', '?')}={float(s.get('weight', 0)):.1%}" for s in sc_adj
    ) or "—"
    note_bits: list[str] = []
    if fb or eq in {"prior_only", "fallback_warn"}:
        note_bits.append("本次使用了模板/先验证据（非纯新闻驱动），已标记并降权或告警。")
    if eq == "news_empty_no_prior":
        note_bits.append("新闻未产出证据且 template_policy=off → S≈0，未静默填模板。")
    if bool(counts.get("cluster_dedup_applied") or news_meta.get("cluster_dedup_applied")):
        note_bits.append(
            "同主题新闻已事件聚类去重（簇内仅最强代表计入 S），避免重复加权重。"
        )
    cluster_warnings = list(
        diag.get("cluster_warnings")
        or counts.get("cluster_warnings")
        or news_meta.get("cluster_warnings")
        or []
    )
    for w in cluster_warnings:
        if w and w not in note_bits:
            note_bits.append(str(w))
    ai_meta = news_meta.get("ai_research") or {}
    if isinstance(ai_meta, dict):
        if ai_meta.get("limitation"):
            note_bits.append(f"AI 检索员：{ai_meta['limitation']}")
        elif ai_meta.get("mode") == "iterative":
            note_bits.append(
                f"AI 检索员迭代 {len([r for r in (ai_meta.get('rounds') or []) if r.get('action')=='search'])} 轮，"
                f"精选 {ai_meta.get('kept_hits', 0)} 条原料 → 产出 {ai_meta.get('headlines_out', 0)}。"
            )
    if news_meta.get("historical_mode") or news_meta.get("cheap_historical") is not None:
        note_bits.append(format_cheap_historical_caption(news_meta))
    if bb_caveat:
        note_bits.append(str(bb_caveat))
    elif peak_eng == "brownian_bridge":
        from fx_report.model.brownian_bridge_max import BB_CONTINUOUS_MAX_NOTE_ZH

        note_bits.append(BB_CONTINUOUS_MAX_NOTE_ZH)
    note = " ".join(note_bits) if note_bits else "证据链按新闻驱动（或空证据诚实路径）。"

    oos = load_calib_oos_summary(diag["market"]["pair"])
    oos_line = ""
    if oos:
        h = oos.get("holdout") or {}
        oos_line = (
            f"· Holdout hit={_fmt_pct(h.get('hit_rate'))}　"
            f"Brier={_fmt_num(h.get('brier'))}　"
            f"Skill={_fmt_num(h.get('skill_brier'))}　"
            f"n={int(h.get('n') or 0)}  \n"
        )

    agree_stats = st.session_state.get("last_label_agree_stats") or {}
    agree_line = ""
    if agree_stats.get("has_labels") and agree_stats.get("agree_rate") is not None:
        demo_tag = "（练习）" if agree_stats.get("is_demo") else ""
        agree_line = (
            f"· 抽检准确率={100 * float(agree_stats['agree_rate']):.0f}%"
            f"（同意率 yes={agree_stats.get('n_yes', 0)} / "
            f"decisive={int(agree_stats.get('n_yes', 0)) + int(agree_stats.get('n_no', 0))}）"
            f"{demo_tag}  \n"
        )
    elif agree_stats.get("has_labels"):
        agree_line = f"· 标注：{agree_stats.get('caption', '已有标注但无明确对错')}  \n"
    else:
        # Fall back to disk spotcheck for this pair
        from fx_report.model.label_audit import load_spotcheck_stats

        disk_sc = load_spotcheck_stats(diag["market"]["pair"])
        if disk_sc and disk_sc.get("agree_rate") is not None:
            agree_line = (
                f"· 抽检准确率（已保存）={100 * float(disk_sc['agree_rate']):.0f}% "
                f"（{disk_sc.get('as_of', '')}）  \n"
            )

    human_re = st.session_state.get("human_label_recomputed") or {}
    human_line = ""
    if human_re.get("applied"):
        human_line = (
            f"· 已用人工标注重算：S={human_re.get('score_S', 0):+.3f}　"
            f"覆盖方向 {human_re.get('n_overridden', 0)} 条  \n"
        )

    ll = news_meta.get("label_learn") or {}
    ll_line = ""
    if ll.get("requested"):
        if ll.get("applied"):
            ll_line = (
                f"· 标签学习强度：已应用 "
                f"（scaled={ll.get('n_strength_scaled', 0)}，"
                f"nudged={ll.get('n_dir_nudged', 0)}）  \n"
            )
        else:
            ll_line = f"· 标签学习强度：未应用 — {ll.get('message') or '标注不足'}  \n"

    drift_meta = dict(
        diag.get("drift_meta")
        or news_meta.get("drift_meta")
        or counts.get("drift_meta")
        or {}
    )
    drift_line = ""
    if drift_meta:
        tv_c = drift_meta.get("tv_category")
        tv_d = drift_meta.get("tv_direction")
        adapted = bool(drift_meta.get("drift_adapted"))
        n_chg = len(drift_meta.get("adapt_changes") or [])
        adapt_note = str(drift_meta.get("adapt_note") or "")
        drift_line = (
            f"· 证据漂移：TV类别={_fmt_num(tv_c if isinstance(tv_c, (int, float)) else None)}　"
            f"TV方向={_fmt_num(tv_d if isinstance(tv_d, (int, float)) else None)}　"
            f"drift_adapted=`{str(adapted).lower()}`"
            + (f"（改 strength {n_chg} 条）" if adapted else "")
            + "  \n"
        )
        if adapt_note:
            drift_line += f"· 漂移适应说明：{adapt_note}  \n"

    st.info(
        f"**本次分析审计**  \n"
        + (
            format_recommend_audit_zh(
                (st.session_state.get("start_cfg") or {}).get("algo_recommend")
            )
            + "  \n"
            if is_simple_setup_mode(
                (st.session_state.get("start_cfg") or {}).get("setup_mode")
            )
            and (st.session_state.get("start_cfg") or {}).get("algo_recommend")
            else (
                "· 算法来源：专家设置（手选，非系统推荐）  \n"
                if (st.session_state.get("start_cfg") or {}).get("setup_mode")
                == "expert"
                else ""
            )
        )
        + f"· peak_engine：`{peak_eng}`  \n"
        f"· jump_model：`{jump_mdl}`　jump_compensate=`{jump_comp}`  \n"
        f"· variance_reduction：`"
        f"{(st.session_state.get('start_cfg') or {}).get('variance_reduction') or diag.get('variance_reduction') or news_meta.get('variance_reduction') or '—'}`  \n"
        f"· 参数来源：{cal_zh}  \n"
        f"{oos_line}"
        f"{agree_line}"
        f"{human_line}"
        f"{ll_line}"
        f"{drift_line}"
        f"· 证据分 S={diag.get('score_S', 0):+.3f}　"
        f"μ_shift={diag.get('mu_annual_shift', 0):+.4f}　"
        f"σ×={diag.get('sigma_mult_extra', 1):.3f}  \n"
        f"· 情景权重（调整后）：{weight_bits}  \n"
        f"· evidence_n={counts.get('evidence_n', 0)}　"
        f"cluster_n={counts.get('cluster_n', news_meta.get('cluster_n', 0))}　"
        f"raw={counts.get('evidence_raw_n', news_meta.get('evidence_raw_n', counts.get('evidence_n', 0)))}　"
        f"dedup={bool(counts.get('cluster_dedup_applied') or news_meta.get('cluster_dedup_applied'))}　"
        f"cluster_method=`{news_meta.get('cluster_method') or counts.get('cluster_method') or (st.session_state.get('start_cfg') or {}).get('cluster_method') or 'jaccard'}`　"
        f"fetched/kept/classified="
        f"{counts.get('fetched', 0)}/{counts.get('kept', 0)}/{counts.get('classified', 0)}　"
        f"fallback_templates={fb}　mode=`{mode_used}`　quality=`{eq}`  \n"
        f"· {note}"
    )
    if cluster_warnings:
        st.warning(
            "**聚类/证据告警**\n\n"
            + "\n\n".join(f"· {w}" for w in cluster_warnings)
        )
    # Thin References / evidence tip (LLM alone does not invent URLs)
    try:
        from fx_report.config.api_config import has_news_api, load_config
        from fx_report.model.label_audit import thin_refs_message

        _stmt_n = len((diag.get("statements") or []) or [])
        _thin = thin_refs_message(
            evidence_n=int(counts.get("evidence_n") or 0),
            fetched=int(counts.get("fetched") or 0),
            news_keys_present=has_news_api(load_config()),
            statements_n=_stmt_n or None,
        )
        if _thin:
            st.warning(_thin)
    except Exception:
        pass
    st.markdown(t("audit.jump_full"))

    # Labeling immediately after audit — before long charts / 900px report HTML
    render_label_audit_section(
        pair=diag["market"]["pair"],
        bullish=bullish,
        evidence_rows=st.session_state.get("last_auto_evidence") or [],
        news_meta=news_meta,
        diag=diag,
    )

    st.bar_chart(
        pd.DataFrame({"区间": list(probs), "概率": list(probs.values())}).set_index("区间")
    )

    with st.expander("完整报告（FX Analyse 格式）", expanded=False):
        pdf_bytes = st.session_state.get("last_pdf_bytes")
        html_doc = st.session_state.get("last_report_html")
        c1, c2, c3 = st.columns(3)
        pair_safe = diag["market"]["pair"].replace("/", "")
        if pdf_bytes:
            c1.download_button(
                "下载 PDF（FX Analyse）",
                pdf_bytes,
                file_name=f"{pair_safe}_fx_analyse.pdf",
                mime="application/pdf",
            )
        elif st.session_state.get("last_pdf_error"):
            c1.caption(f"PDF 生成失败：{st.session_state['last_pdf_error']}")
        if html_doc:
            c2.download_button(
                "下载 HTML",
                html_doc.encode("utf-8"),
                file_name=f"{pair_safe}_fx_analyse.html",
                mime="text/html",
            )
        c3.download_button(
            "下载 Markdown（调试）",
            report.encode("utf-8"),
            file_name=f"{pair_safe}_mc_report.md",
            mime="text/markdown",
        )
        audit_csv = st.session_state.get("last_label_audit_csv")
        if audit_csv:
            st.caption(
                "证据标注请到上方审计下的「证据人工标注」填写；此处可下载当前 CSV。"
            )
            st.download_button(
                "下载证据标注 CSV（label_audit）",
                audit_csv.encode("utf-8"),
                file_name=f"{pair_safe}_label_audit.csv",
                mime="text/csv",
                help="列含 statement_id/title/url/model_*；在「证据人工标注」填写 human_* 与 agree",
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
