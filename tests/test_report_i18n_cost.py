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
