"""Tests for bilingual report strings, stance summaries, and cost estimates."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from fx_report.model.weights import EvidenceItem
from fx_report.news.summarize import (
    apply_stance_summaries,
    extractive_stance_summary,
)
from fx_report.report.cost_estimate import (
    cost_table_rows_zh,
    estimate_live_report_cost,
    estimate_stance_summary_usd,
)
from fx_report.report.evidence_refs import (
    evidence_stance_summary_meta,
    format_reference_markdown_row,
)
from fx_report.report.strings import (
    DEFAULT_REPORT_LANG,
    DEFAULT_REPORT_MODE,
    LANG_BOTH,
    LANG_EN,
    LANG_ZH,
    L,
    normalize_report_lang,
    normalize_report_mode,
    pair_phrase,
    report_lang_suffix,
    report_langs_for_mode,
)
from fx_report.report.torchcast import TorchcastReport, render_html


def _ev(**kwargs) -> EvidenceItem:
    base = dict(
        id="U-1",
        title="Fed signals higher for longer",
        direction=1,
        strength=1.2,
        freshness=1.0,
        unpriced=0.7,
        category="fed",
        support_quote="The Fed kept rates unchanged and signaled patience on cuts.",
        support_quote_quality="support",
        summary="Fed holds; markets price fewer cuts.",
        url="https://www.reuters.com/example",
    )
    base.update(kwargs)
    return EvidenceItem(**base)


def test_normalize_report_lang_defaults_zh():
    assert normalize_report_lang(None) == DEFAULT_REPORT_LANG == LANG_ZH
    assert normalize_report_lang("EN") == LANG_EN
    assert normalize_report_lang("中文") == LANG_ZH
    assert normalize_report_lang("both") == LANG_ZH  # primary for single-lang helpers


def test_normalize_report_mode_bilingual_default():
    assert normalize_report_mode(None) == DEFAULT_REPORT_MODE == LANG_BOTH
    assert normalize_report_mode("中英双语") == LANG_BOTH
    assert normalize_report_mode("bilingual") == LANG_BOTH
    assert normalize_report_mode("EN") == LANG_EN
    assert normalize_report_mode("中文") == LANG_ZH
    assert report_langs_for_mode("both") == [LANG_ZH, LANG_EN]
    assert report_langs_for_mode("en") == [LANG_EN]
    assert report_lang_suffix("zh") == "_zh"
    assert report_lang_suffix("en") == "_en"


def test_labels_zh_en_distinct():
    assert L("stance_summary", lang="zh") == "总结"
    assert L("stance_summary", lang="en") == "Summary"
    assert L("support", lang="zh") == "支撑引用"
    assert L("support", lang="en") == "Support quote"
    assert "美元" in pair_phrase("USD/AUD", lang="zh")
    assert "US Dollar" in pair_phrase("USD/AUD", lang="en")


def test_extractive_stance_summary_no_invention():
    e = _ev(summary="", support_quote="", title="RBA holds cash rate")
    zh = extractive_stance_summary(e, lang="zh")
    en = extractive_stance_summary(e, lang="en")
    assert "RBA holds" in zh
    assert "上行" in zh or "Higher" in zh
    assert "RBA holds" in en
    assert "Source indicates" in en


def test_apply_stance_summaries_cheap_historical_skips_llm():
    items = [_ev(id="U-1"), _ev(id="U-2", direction=-1, title="AUD soft")]
    fake_cfg = type("Cfg", (), {"api_key": "sk-test"})()
    with patch(
        "fx_report.news.summarize._llm_batch_stance_summaries",
        side_effect=AssertionError("LLM should not run on cheap historical"),
    ):
        meta = apply_stance_summaries(
            items,
            lang="zh",
            pair="USD/AUD",
            llm_cfg=fake_cfg,
            cheap_historical=True,
        )
    assert meta["method"] == "extractive"
    assert meta["n_llm"] == 0
    assert all((e.stance_summary or "").strip() for e in items)


def test_apply_stance_summaries_bilingual_extractive_fills_i18n():
    items = [_ev()]
    meta = apply_stance_summaries(
        items, langs=["zh", "en"], llm_cfg=None, cheap_historical=False
    )
    assert meta["bilingual"] is True
    assert meta["method"] == "extractive"
    bag = items[0].stance_summary_i18n
    assert "zh" in bag and "en" in bag
    assert "该来源称" in bag["zh"] or "上行" in bag["zh"]
    assert "Source indicates" in bag["en"]
    zh_meta = evidence_stance_summary_meta(items[0], lang="zh")
    en_meta = evidence_stance_summary_meta(items[0], lang="en")
    assert zh_meta["text"]
    assert "Source indicates" in en_meta["text"]


def test_apply_stance_summaries_force_extractive_without_key():
    items = [_ev()]
    meta = apply_stance_summaries(
        items, lang="en", llm_cfg=None, cheap_historical=False
    )
    assert meta["method"] == "extractive"
    assert "Source indicates" in (items[0].stance_summary or "")


def test_reference_markdown_includes_stance_and_support():
    e = _ev(
        stance_summary="该来源称美联储偏鹰，支撑上行判定。",
        support_quote="The Fed signaled higher for longer.",
        support_quote_quality="support",
    )
    line = format_reference_markdown_row(e, index=1, lang="zh")
    assert "总结" in line
    assert "支撑引用" in line
    assert "higher for longer" in line.lower() or "Higher for longer" in line
    assert "美联储" in line or "上行" in line
    assert "reuters.com" in line


def test_html_zh_renders_stance_summary_label():
    e = _ev(stance_summary="来源支持美元上行路径。")
    report = TorchcastReport(
        pair="USD/AUD",
        question="测试问题",
        forecast_date="2026-08-06",
        n_evidence=1,
        n_buckets=5,
        probs={"a": 0.6, "b": 0.4},
        top_bucket="a",
        top_prob=0.6,
        upside_bullets=["x"],
        downside_bullets=["y"],
        executive_summary="摘要",
        narratives=[],
        higher_evidence=[e],
        lower_evidence=[],
        context_evidence=[],
        watches=[],
        spot=1.5,
        lang="zh",
    )
    html = render_html(report)
    assert 'lang="zh-CN"' in html
    assert "总结" in html
    assert "支撑引用" in html
    assert "来源支持美元上行" in html
    assert "概率分布" in html or "情报报告" in html


def test_html_en_chrome():
    e = _ev(stance_summary="Source backs a higher USD peak.")
    report = TorchcastReport(
        pair="USD/AUD",
        question="Where will the peak land?",
        forecast_date="2026-08-06",
        n_evidence=1,
        n_buckets=5,
        probs={"a": 0.5, "b": 0.5},
        top_bucket="a",
        top_prob=0.5,
        upside_bullets=["x"],
        downside_bullets=["y"],
        executive_summary="sum",
        narratives=[],
        higher_evidence=[e],
        lower_evidence=[],
        context_evidence=[],
        watches=[],
        spot=1.5,
        lang="en",
    )
    html = render_html(report)
    assert 'lang="en"' in html
    assert "Summary" in html
    assert "Support quote" in html
    assert "Probability Distribution" in html
    assert "总结" not in html
    assert "支撑引用" not in html
    assert "概率分布" not in html


def test_build_torchcast_and_markdown_en_headings():
    """Key EN chrome must appear when lang=en (no ZH section titles)."""
    from datetime import date

    import numpy as np

    from fx_report.market.fetch_data import MarketSnapshot
    from fx_report.model.monte_carlo import MCResult
    from fx_report.model.weights import ModelWeights, default_scenarios
    from fx_report.report.text import build_report_markdown
    from fx_report.report.torchcast import build_torchcast_report

    market = MarketSnapshot(
        asof="2026-08-06",
        pair="USD/AUD",
        spot=1.55,
        provider_raw=1.55,
        sigma_daily=0.006,
        sigma_annual=0.095,
        mean_daily_return=0.0,
        n_returns=60,
        lookback_days=60,
        history_start="2025-01-01",
        history_end="2026-08-01",
        source="test",
        brent=None,
        dxy_proxy=None,
        notes=["unit test"],
        history_ticker="TEST",
        spot_ticker="TEST",
    )
    edges = (1.50, 1.55, 1.60, 1.65)
    labels = ["< 1.500000", "1.500000 to 1.550000", "1.550000 to 1.600000", "1.600000 to 1.650000", ">= 1.650000"]
    probs = {lab: 0.2 for lab in labels}
    probs[labels[2]] = 0.4
    probs[labels[0]] = 0.0
    weights = ModelWeights()
    weights.evidence = [
        _ev(id="U-1", direction=1),
        _ev(id="D-1", direction=-1, title="AUD soft on China data", category="china_growth"),
    ]
    weights.evidence[0].stance_summary_i18n = {
        "zh": "该来源称偏鹰。",
        "en": "Source indicates a hawkish Fed bias.",
    }
    weights.evidence[0].stance_summary = "该来源称偏鹰。"
    scens = default_scenarios("USD/AUD")
    mc = MCResult(
        maxima=np.array([1.56, 1.58]),
        bucket_labels=list(labels),
        raw_probs=dict(probs),
        scenario_counts={"baseline": 2},
        n_sims=2,
        spot=1.55,
        sigma_daily_base=0.006,
        trading_days=20,
        percentiles={"p50": 1.56, "p90": 1.60, "p95": 1.62},
        peak_engine="path_max",
    )
    tc = build_torchcast_report(
        market,
        weights,
        scens,
        mc,
        probs,
        score=0.5,
        mu_shift=0.01,
        sigma_extra=1.05,
        horizon_start=date(2026, 8, 6),
        horizon_end=date(2026, 9, 5),
        bucket_edges=edges,
        bullish_currency="USD",
        lang="en",
    )
    html = render_html(tc)
    for needle in (
        "Probability Distribution",
        "Executive Summary",
        "Evidence Base",
        "What to Watch",
        "Higher",
        "Lower",
        "Summary",
        "Support quote",
        "Mathematical Floor",
        "If → Then",
    ):
        assert needle in html, needle
    for zh in ("概率分布", "执行摘要", "证据库", "关注事项", "支撑引用", "数学地板", "若→则"):
        assert zh not in html, zh
    assert "Source indicates a hawkish Fed bias" in html

    md = build_report_markdown(
        market,
        weights,
        scens,
        mc,
        probs,
        score=0.5,
        mu_shift=0.01,
        sigma_extra=1.05,
        horizon_label="2026-08-06 to 2026-09-05",
        bucket_edges=edges,
        lang="en",
    )
    for needle in (
        "Probability distribution",
        "Market anchors",
        "Executive summary",
        "Scenario weights",
        "Raw MC frequencies",
        "Strength rubric",
        "What to Watch",
        "How strength is scored",
        "Risk-on / safe-haven",
        "contrib",
        "Summary",
    ):
        assert needle in md, needle
    for zh in (
        "概率分布",
        "行情锚点",
        "执行摘要",
        "情景权重",
        "原始 MC",
        "信息强弱如何判定",
        "风险升高",
        "贡献",
        "计分",
    ):
        assert zh not in md, zh


def test_pipeline_preface_en_localized():
    """Markdown preface/appendix chrome must be English when report_lang=en."""
    from datetime import date

    import numpy as np

    from fx_report.market.fetch_data import MarketSnapshot
    from fx_report.model.monte_carlo import MCResult
    from fx_report.model.weights import ModelWeights, default_scenarios
    from fx_report.pipeline import InfoNeed, WeightedEvidence, step7_build_report

    market = MarketSnapshot(
        asof="2026-08-06",
        pair="USD/AUD",
        spot=1.55,
        provider_raw=1.55,
        sigma_daily=0.006,
        sigma_annual=0.095,
        mean_daily_return=0.0,
        n_returns=60,
        lookback_days=60,
        history_start="2025-01-01",
        history_end="2026-08-01",
        source="test",
        brent=None,
        dxy_proxy=None,
        notes=[],
    )
    edges = (1.50, 1.55, 1.60, 1.65)
    labels = ["a", "b", "c", "d", "e"]
    probs = {lab: 0.2 for lab in labels}
    weights = ModelWeights()
    e = _ev()
    e.stance_summary_i18n = {"en": "Source indicates Fed patience.", "zh": "偏鹰"}
    weights.evidence = [e]
    weighted = [
        WeightedEvidence(evidence=e, impact_note="中文不应出现", weight_contrib=0.5)
    ]
    scens = default_scenarios("USD/AUD")
    mc = MCResult(
        maxima=np.array([1.56]),
        bucket_labels=labels,
        raw_probs=probs,
        scenario_counts={"baseline": 1},
        n_sims=1,
        spot=1.55,
        sigma_daily_base=0.006,
        trading_days=20,
        percentiles={"p50": 1.56, "p90": 1.60, "p95": 1.62},
    )
    needs = [
        InfoNeed(
            id="fed",
            need="FOMC 决议",
            why="美元利率",
            sources="Fed",
            driver="fed",
        )
    ]
    md, html, tc, diag, horizon = step7_build_report(
        market=market,
        weights=weights,
        scenarios=scens,
        mc=mc,
        probs=probs,
        score=0.1,
        mu_shift=0.0,
        sigma_extra=1.0,
        edges=edges,
        info_needs=needs,
        statements=[],
        weighted=weighted,
        stage_log=[],
        headlines=[],
        news_meta={"max_news": 10},
        bullish_currency="USD",
        as_of_date=date(2026, 8, 6),
        report_lang="en",
    )
    assert "Analysis pipeline" in md
    assert "Information needs" in md
    assert "Weight contributions" in md
    assert "References / Evidence base" in md
    assert "FOMC decisions" in md  # localized from DRIVER_CATALOG_EN
    assert "Lifts upper tail" in md or "Caps peak" in md or "Neutral" in md
    assert "分析流程" not in md
    assert "需要什么" not in md
    assert "生成时间" not in md
    assert "Probability Distribution" in html
    assert tc.lang == "en"
    assert "to" in horizon


def test_scenario_narrative_and_impact_note_i18n():
    from fx_report.report.strings import format_impact_note, scenario_narrative

    assert "Risk-on" in scenario_narrative("escalation", "USD/AUD", lang="en")
    assert "风险升高" in scenario_narrative("escalation", "USD/AUD", lang="zh")
    e = _ev(direction=1, category="fed")
    en = format_impact_note(e, 0.5, lang="en")
    zh = format_impact_note(e, 0.5, lang="zh")
    assert "Lifts upper tail" in en
    assert "推高" in zh
    assert "中文不应" not in en


def test_stance_meta_distinct_from_support_quote():
    e = _ev(
        stance_summary="总结句：偏鹰。",
        support_quote="Verbatim Fed quote about rates.",
    )
    stance = evidence_stance_summary_meta(e, lang="zh")
    assert stance["text"].startswith("总结句")
    assert "Verbatim" not in stance["text"]


def test_cost_table_has_baseline_and_ref_tiers():
    rows = cost_table_rows_zh()
    blob = "\n".join(r["场景"] + r["约 USD/份"] for r in rows)
    assert "基线" in blob
    assert "30" in blob and "80" in blob and "100" in blob
    assert "2–3" in blob or "2-3" in blob
    assert "双语" in blob or "不重跑" in blob
    base = estimate_live_report_cost(n_refs=80, include_stance_summaries=False)
    with_sum = estimate_live_report_cost(n_refs=80, include_stance_summaries=True)
    assert with_sum["total_usd"] > base["total_usd"]
    assert estimate_stance_summary_usd(80) > 0
    # Template ZH switch is free
    assert (
        estimate_live_report_cost(report_lang_via_template=True)["parts_usd"][
            "report_language"
        ]
        == 0.0
    )


def test_pipeline_result_save_bilingual_filenames(tmp_path: Path):
    """Dual-lang artifacts get ``_zh`` / ``_en`` stems; news/MC not re-run here."""
    from fx_report.pipeline import PipelineResult
    from fx_report.market.fetch_data import MarketSnapshot
    from fx_report.model.monte_carlo import MCResult
    from fx_report.model.weights import ModelWeights

    tc_zh = TorchcastReport(
        pair="USD/AUD",
        question="中文问",
        forecast_date="2026-08-06",
        n_evidence=0,
        n_buckets=2,
        probs={"a": 1.0},
        top_bucket="a",
        top_prob=1.0,
        upside_bullets=[],
        downside_bullets=[],
        executive_summary="中文摘要",
        narratives=[],
        higher_evidence=[],
        lower_evidence=[],
        context_evidence=[],
        watches=[],
        spot=1.5,
        lang="zh",
    )
    tc_en = TorchcastReport(
        pair="USD/AUD",
        question="EN Q",
        forecast_date="2026-08-06",
        n_evidence=0,
        n_buckets=2,
        probs={"a": 1.0},
        top_bucket="a",
        top_prob=1.0,
        upside_bullets=[],
        downside_bullets=[],
        executive_summary="EN summary",
        narratives=[],
        higher_evidence=[],
        lower_evidence=[],
        context_evidence=[],
        watches=[],
        spot=1.5,
        lang="en",
    )
    market = MagicMock(spec=MarketSnapshot)
    mc = MagicMock(spec=MCResult)
    result = PipelineResult(
        pair="USD/AUD",
        stage_log=[],
        info_needs=[],
        market=market,
        statements=[],
        weighted=[],
        score=0.0,
        mu_shift=0.0,
        sigma_extra=1.0,
        scenarios=[],
        edges=(1.0, 1.1, 1.2, 1.3),
        mc=mc,
        probs={"a": 1.0},
        weights=ModelWeights(),
        report_md="# zh",
        report_html="<html lang=zh>",
        torchcast=tc_zh,
        diagnostics={},
        news_meta={"report_lang_mode": "both", "report_langs": ["zh", "en"]},
        horizon_label="h",
        reports_by_lang={
            "zh": {"md": "# zh", "html": "<zh/>", "torchcast": tc_zh, "horizon": "h"},
            "en": {"md": "# en", "html": "<en/>", "torchcast": tc_en, "horizon": "h"},
        },
    )
    with patch("fx_report.pipeline.export_torchcast") as exp:
        def _fake_export(report, out_dir, *, stem):
            out = Path(out_dir)
            html = out / f"{stem}_fx_analyse.html"
            pdf = out / f"{stem}_fx_analyse.pdf"
            html.write_text("ok", encoding="utf-8")
            pdf.write_bytes(b"%PDF")
            return {"html": html, "pdf": pdf}

        exp.side_effect = _fake_export
        paths = result.save(tmp_path)

    assert paths["report_zh"].name == "USDAUD_zh_report.md"
    assert paths["report_en"].name == "USDAUD_en_report.md"
    assert paths["pdf_zh"].name == "USDAUD_zh_fx_analyse.pdf"
    assert paths["pdf_en"].name == "USDAUD_en_fx_analyse.pdf"
    assert paths["html_zh"].name.endswith("_zh_fx_analyse.html")
    assert "# zh" in paths["report_zh"].read_text(encoding="utf-8")
    assert "# en" in paths["report_en"].read_text(encoding="utf-8")
    assert exp.call_count == 2
