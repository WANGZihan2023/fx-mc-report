"""Tests for GDELT DOC historical news (mocked HTTP; no network / no NewsAPI key)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from fx_report.news.fetch import Headline, fetch_historical_headlines_for_pair
from fx_report.news.gdelt import (
    GDELT_MAX_HISTORY_DAYS,
    clamp_gdelt_from_date,
    fetch_gdelt_doc,
    gdelt_earliest_searchable_date,
    gdelt_query_for_pair,
    _parse_gdelt_seendate,
)


def test_gdelt_query_covers_aud_pair_terms():
    q = gdelt_query_for_pair("USD/AUD")
    assert "RBA" in q
    assert "iron ore" in q or "AUDUSD" in q
    assert "Fed" in q
    assert gdelt_query_for_pair("AUD/USD")


def test_clamp_gdelt_from_date_shortens_lookback():
    today = date(2026, 7, 28)
    as_of = date(2026, 7, 13)
    requested = as_of - timedelta(days=120)
    clamped, was_clamped = clamp_gdelt_from_date(requested, as_of, today=today)
    assert was_clamped is True
    assert clamped == gdelt_earliest_searchable_date(today=today)
    assert (today - clamped).days == GDELT_MAX_HISTORY_DAYS


def test_clamp_gdelt_skips_when_as_of_too_old():
    today = date(2026, 7, 28)
    as_of = date(2025, 1, 10)
    clamped, was_clamped = clamp_gdelt_from_date(
        as_of - timedelta(days=14),
        as_of,
        today=today,
    )
    assert was_clamped is True
    assert clamped is None


def test_parse_gdelt_seendate():
    dt = _parse_gdelt_seendate("20260710T153000Z")
    assert dt is not None
    assert dt.date() == date(2026, 7, 10)
    assert dt.hour == 15
    assert _parse_gdelt_seendate(None) is None


def test_fetch_gdelt_doc_parses_artlist(monkeypatch, tmp_path):
    from fx_report.news.gdelt import _GDELT_MEM_CACHE

    monkeypatch.setenv("FX_GDELT_CACHE", str(tmp_path))
    _GDELT_MEM_CACHE.clear()
    payload = {
        "articles": [
            {
                "url": "https://example.com/aud-1",
                "title": "RBA holds rates; AUDUSD steady as iron ore firms",
                "seendate": "20260710T120000Z",
                "domain": "reuters.com",
                "language": "English",
                "sourcecountry": "United States",
            },
            {
                "url": "https://example.com/aud-old",
                "title": "Too old article",
                "seendate": "20260501T120000Z",
                "domain": "example.com",
            },
        ]
    }

    def fake_request(url, timeout, *, max_retries=2):
        assert "startdatetime=20260629000000" in url
        assert "enddatetime=20260713235959" in url
        assert "mode=ArtList" in url
        return payload, None, 200

    monkeypatch.setattr("fx_report.news.gdelt._gdelt_request_json", fake_request)
    meta: dict = {}
    headlines = fetch_gdelt_doc(
        gdelt_query_for_pair("USD/AUD"),
        start_date=date(2026, 6, 29),
        end_date=date(2026, 7, 13),
        limit=10,
        call_meta=meta,
    )
    assert meta["error"] is None
    assert meta["raw_count"] == 2
    assert len(headlines) == 1
    assert headlines[0].provider == "gdelt"
    assert headlines[0].source == "reuters.com"
    assert "RBA" in headlines[0].title


def test_fetch_gdelt_doc_surfaces_429(monkeypatch, tmp_path):
    from fx_report.news.gdelt import _GDELT_MEM_CACHE

    monkeypatch.setenv("FX_GDELT_CACHE", str(tmp_path))
    _GDELT_MEM_CACHE.clear()

    def fake_request(url, timeout, *, max_retries=2):
        return None, "HTTP 429: Too Many Requests", 429

    monkeypatch.setattr("fx_report.news.gdelt._gdelt_request_json", fake_request)
    meta: dict = {}
    headlines = fetch_gdelt_doc(
        "RBA",
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 13),
        call_meta=meta,
    )
    assert headlines == []
    assert meta["http_status"] == 429
    assert "429" in str(meta["error"])


def test_historical_date_filtered_from_gdelt_alone(monkeypatch):
    """No NewsAPI key: GDELT hits alone should yield date_filtered."""
    today = date(2026, 7, 28)
    as_of = date(2026, 7, 13)

    def fake_gdelt(query, *, start_date=None, end_date=None, limit=25, timeout=20, call_meta=None):
        if call_meta is not None:
            call_meta.clear()
            call_meta.update(
                {
                    "error": None,
                    "http_status": 200,
                    "query": query,
                    "start": start_date.isoformat() if hasattr(start_date, "isoformat") else str(start_date),
                    "end": end_date.isoformat() if hasattr(end_date, "isoformat") else str(end_date),
                    "raw_count": 1,
                    "url_host": "api.gdeltproject.org",
                }
            )
        return [
            Headline(
                title="Aussie climbs after Fed cut bets; RBA on hold",
                summary="",
                source="afr.com",
                url="https://example.com/gdelt-aud",
                published=datetime(2026, 7, 11, 9, 0, tzinfo=timezone.utc),
                provider="gdelt",
            )
        ]

    monkeypatch.setattr("fx_report.news.gdelt.fetch_gdelt_doc", fake_gdelt)
    monkeypatch.setattr("fx_report.news.fetch.fetch_inbox_headlines", lambda limit=12: [])
    monkeypatch.setattr("fx_report.news.fetch.load_config", lambda: {})
    monkeypatch.setattr("fx_report.news.fetch.is_set", lambda cfg, key: False)

    headlines, meta = fetch_historical_headlines_for_pair(
        "USD/AUD",
        as_of_date=as_of,
        lookback_days=60,
        max_items=25,
        today=today,
    )
    assert meta["newsapi_enabled"] is False
    assert meta["newsapi_hits"] == 0
    assert meta["gdelt_hits"] == 1
    assert meta["historical_news_quality"] == "date_filtered"
    assert "GDELT" in meta["limitation"]
    assert any(h.provider == "gdelt" for h in headlines)


def test_historical_surfaces_gdelt_error_note(monkeypatch):
    today = date(2026, 7, 28)
    as_of = date(2026, 7, 13)

    def fake_gdelt(query, *, start_date=None, end_date=None, limit=25, timeout=20, call_meta=None):
        if call_meta is not None:
            call_meta.clear()
            call_meta.update(
                {
                    "error": "HTTP 429: Too Many Requests",
                    "http_status": 429,
                    "query": query,
                    "start": None,
                    "end": None,
                    "raw_count": 0,
                    "url_host": "api.gdeltproject.org",
                }
            )
        return []

    monkeypatch.setattr("fx_report.news.gdelt.fetch_gdelt_doc", fake_gdelt)
    monkeypatch.setattr("fx_report.news.fetch.fetch_inbox_headlines", lambda limit=12: [])
    monkeypatch.setattr("fx_report.news.fetch.load_config", lambda: {})
    monkeypatch.setattr("fx_report.news.fetch.is_set", lambda cfg, key: False)

    _hl, meta = fetch_historical_headlines_for_pair(
        "USD/AUD",
        as_of_date=as_of,
        lookback_days=14,
        max_items=10,
        today=today,
    )
    assert meta["gdelt_hits"] == 0
    assert meta["historical_news_quality"] == "limited"
    assert meta["gdelt_http_status"] == 429
    assert "GDELT" in meta["limitation"]
    assert "429" in meta["limitation"]


def test_historical_inbox_dated_counts_as_date_filtered(monkeypatch):
    today = date(2026, 7, 28)
    as_of = date(2026, 7, 13)

    monkeypatch.setattr(
        "fx_report.news.fetch.fetch_inbox_headlines",
        lambda limit=12: [
            Headline(
                title="[inbox] note.md: AUD notes",
                summary="local research",
                source="inbox:note.md",
                url="file:///tmp/note.md",
                published=datetime(2026, 7, 10, tzinfo=timezone.utc),
                provider="inbox",
            )
        ],
    )
    monkeypatch.setattr("fx_report.news.gdelt.fetch_gdelt_doc", lambda *a, **k: [])
    monkeypatch.setattr("fx_report.news.fetch.load_config", lambda: {})
    monkeypatch.setattr("fx_report.news.fetch.is_set", lambda cfg, key: False)

    _hl, meta = fetch_historical_headlines_for_pair(
        "USD/AUD",
        as_of_date=as_of,
        lookback_days=14,
        today=today,
    )
    assert meta["inbox_dated_hits"] == 1
    assert meta["gdelt_hits"] == 0
    assert meta["newsapi_hits"] == 0
    assert meta["historical_news_quality"] == "date_filtered"
