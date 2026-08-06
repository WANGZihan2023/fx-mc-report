"""Tests for Evidence Base support-quote helpers and URL hygiene."""

from __future__ import annotations

from unittest.mock import patch

from fx_report.model.weights import EvidenceItem
from fx_report.news.summarize import extract_support_quote
from fx_report.news.urls import (
    is_fragile_url,
    prefer_stable_url,
    provider_stability_rank,
    sanitize_evidence_urls,
)
from fx_report.report.evidence_refs import (
    LABEL_SUPPORT_WEAK_ZH,
    LABEL_SUPPORT_ZH,
    evidence_link_meta,
    evidence_quote,
    evidence_support_meta,
    format_reference_markdown_row,
)
from fx_report.ui.ux_helpers import refs_statement_cap, step3_pool_size


def _ev(**kwargs) -> EvidenceItem:
    base = dict(
        id="U-1",
        title="Fed signals higher for longer",
        direction=1,
        strength=1.2,
        freshness=1.0,
        unpriced=0.7,
        category="fed",
    )
    base.update(kwargs)
    return EvidenceItem(**base)


def test_extract_support_quote_picks_stance_sentence():
    """Higher (+1) item should prefer hawkish/hike sentence over SEO junk."""
    text = (
        "Subscribe to our newsletter for daily market alerts. "
        "The Federal Reserve signaled it may hike rates again if inflation stays sticky. "
        "Click here for more sports news from Sydney."
    )
    quote, quality = extract_support_quote(
        text,
        title="Fed higher for longer",
        direction=1,
        category="fed",
        pair="USD/AUD",
        max_sentences=1,
    )
    assert "Federal Reserve" in quote or "hike" in quote.lower()
    assert "Subscribe" not in quote
    assert "Click here" not in quote
    assert quality in {"support", "weak"}


def test_extract_support_quote_bearish_prefers_cut_language():
    text = (
        "Local weather remained mild through the weekend. "
        "Iron ore prices slumped on weak China demand, pressuring the Aussie. "
        "A cooking blog shared a new pasta recipe."
    )
    quote, quality = extract_support_quote(
        text,
        title="AUD soft on iron ore",
        direction=-1,
        category="china_iron",
        pair="AUD/USD",
        max_sentences=1,
    )
    assert "iron ore" in quote.lower() or "Aussie" in quote or "China" in quote
    assert "weather" not in quote.lower()
    assert "pasta" not in quote.lower()
    assert quality in {"support", "weak"}


def test_extract_support_quote_no_invent_from_empty():
    quote, quality = extract_support_quote(
        "",
        title="RBA holds cash rate",
        direction=0,
        category="rba",
    )
    assert quote == "RBA holds cash rate"
    assert quality == "title"


def test_evidence_quote_prefers_support_quote_field():
    e = _ev(
        support_quote="The Fed kept rates unchanged and signaled patience on cuts.",
        support_quote_quality="support",
        summary="Generic SEO blurb about markets today.",
        note="https://example.com/a｜extra",
        title="Short title",
    )
    meta = evidence_support_meta(e)
    assert "Fed kept rates" in meta["quote"]
    assert "SEO" not in meta["quote"]
    assert meta["label_zh"] == LABEL_SUPPORT_ZH
    assert evidence_quote(e) == meta["quote"]


def test_evidence_quote_falls_back_to_title():
    e = _ev(summary="", note="", title="Iron ore softens AUD pressure", support_quote="")
    assert "Iron ore" in evidence_quote(e)


def test_format_reference_markdown_includes_support_label_and_url():
    e = _ev(
        support_quote="AUD firmer after RBA minutes.",
        support_quote_quality="support",
        url="https://think.ing.com/real-article",
    )
    line = format_reference_markdown_row(e, index=1)
    assert "U-1" in line
    assert LABEL_SUPPORT_ZH in line
    assert "「" in line and "」" in line
    assert "think.ing.com" in line


def test_weak_support_label():
    e = _ev(
        support_quote="Markets opened mixed in Asia.",
        support_quote_quality="weak",
    )
    meta = evidence_support_meta(e)
    assert meta["label_zh"] == LABEL_SUPPORT_WEAK_ZH


def test_dead_link_meta_clears_url_display():
    e = _ev(url="", note="链接可能失效｜was dead")
    meta = evidence_link_meta(e)
    assert meta["dead_marked"] is True
    assert meta["url"] == ""


def test_is_fragile_google_news():
    assert is_fragile_url("https://news.google.com/rss/articles/CBMiabc")
    assert not is_fragile_url("https://www.reuters.com/markets/usd")


def test_prefer_stable_extracts_url_query():
    wrapped = "https://news.google.com/rss/search?url=https%3A%2F%2Fwww.reuters.com%2Ffoo"
    out = prefer_stable_url(wrapped, timeout=0.1)
    assert "reuters.com" in out
    assert "news.google" not in out


def test_sanitize_marks_maybe_dead(monkeypatch):
    e = _ev(url="https://example.com/gone", note="https://example.com/gone")
    with patch("fx_report.news.urls.soft_check_url", return_value="maybe_dead"):
        meta = sanitize_evidence_urls([e], soft_check=True, max_checks=5, timeout=0.5)
    assert meta["maybe_dead"] == 1
    assert e.url == ""
    assert "链接可能失效" in (e.note or "")


def test_provider_rank_prefers_tavily_over_google():
    assert provider_stability_rank("tavily") < provider_stability_rank("google_news_rss")
    assert provider_stability_rank("whitelist") < provider_stability_rank("tavily")


def test_live_caps_raised():
    from fx_report.ui.ux_helpers import (
        clamp_live_max_news,
        default_live_max_news,
        refs_statement_cap,
        step3_pool_size,
    )

    assert step3_pool_size(10, historical=False) == 40
    assert step3_pool_size(40, historical=False) == 120
    assert step3_pool_size(100, historical=False) == 200
    assert step3_pool_size(10, historical=True) >= 10
    assert refs_statement_cap(30) == 100
    assert refs_statement_cap(80) == 100
    assert refs_statement_cap(100) == 100
    assert refs_statement_cap(120) == 120
    assert refs_statement_cap(None) == 120
    assert clamp_live_max_news(200) == 120
    assert clamp_live_max_news(1) == 3
    with patch(
        "fx_report.news.ai_research.search_hands_available",
        return_value={"tavily": True, "brave": False},
    ):
        assert default_live_max_news() == 80
    with patch(
        "fx_report.news.ai_research.search_hands_available",
        return_value={"tavily": False, "brave": False},
    ):
        assert default_live_max_news() == 30


def test_live_research_budget_tavily():
    from fx_report.news.ai_research import live_research_budget

    with patch(
        "fx_report.news.ai_research.search_hands_available",
        return_value={"tavily": True, "brave": False, "newsapi": True, "google_news_rss": True},
    ):
        b = live_research_budget({})
    assert b["target_keep"] >= 80
    assert b["max_rounds"] >= 8
    assert b["max_headlines"] >= 80


def test_html_evidence_renders_support_quote():
    from fx_report.report.torchcast import TorchcastReport, render_html

    e = _ev(
        support_quote="USD strength persists amid rate differentials.",
        support_quote_quality="support",
        url="https://www.reuters.com/example",
    )
    report = TorchcastReport(
        pair="USD/AUD",
        question="q",
        forecast_date="2026-07-31",
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
        lang="zh",
    )
    html = render_html(report)
    assert "支撑引用" in html
    assert "USD strength" in html
    assert "证据库" in html or "References" in html
    assert "reuters.com" in html
