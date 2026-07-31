"""Tests for Evidence Base quote helpers and URL hygiene."""

from __future__ import annotations

from unittest.mock import patch

from fx_report.model.weights import EvidenceItem
from fx_report.news.urls import (
    is_fragile_url,
    prefer_stable_url,
    provider_stability_rank,
    sanitize_evidence_urls,
)
from fx_report.report.evidence_refs import (
    evidence_link_meta,
    evidence_quote,
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


def test_evidence_quote_prefers_summary():
    e = _ev(
        summary="The Federal Reserve kept rates unchanged and signaled patience.",
        note="https://example.com/a｜extra",
        title="Short title",
    )
    q = evidence_quote(e)
    assert "Federal Reserve" in q
    assert "https://" not in q


def test_evidence_quote_falls_back_to_title():
    e = _ev(summary="", note="", title="Iron ore softens AUD pressure")
    assert "Iron ore" in evidence_quote(e)


def test_format_reference_markdown_includes_quote_and_url():
    e = _ev(
        summary="AUD firmer after RBA minutes.",
        url="https://think.ing.com/real-article",
    )
    line = format_reference_markdown_row(e, index=1)
    assert "U-1" in line
    assert "「" in line and "」" in line
    assert "think.ing.com" in line


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
    assert step3_pool_size(10, historical=False) == 40
    assert step3_pool_size(40, historical=False) == 120
    assert step3_pool_size(10, historical=True) >= 10
    assert refs_statement_cap(30) == 60
    assert refs_statement_cap(None) == 80


def test_live_research_budget_tavily():
    from fx_report.news.ai_research import live_research_budget

    with patch(
        "fx_report.news.ai_research.search_hands_available",
        return_value={"tavily": True, "brave": False, "newsapi": True, "google_news_rss": True},
    ):
        b = live_research_budget({})
    assert b["target_keep"] >= 30
    assert b["max_rounds"] >= 6


def test_html_evidence_renders_quote():
    from fx_report.report.torchcast import TorchcastReport, render_html

    e = _ev(
        summary="USD strength persists amid rate differentials.",
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
    )
    html = render_html(report)
    assert "引用" in html
    assert "USD strength" in html
    assert "Evidence Base" in html
    assert "reuters.com" in html
