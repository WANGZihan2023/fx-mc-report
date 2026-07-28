"""Regression: historical NewsAPI from-date must respect plan window."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from fx_report.news.fetch import (
    NEWSAPI_MAX_HISTORY_DAYS,
    Headline,
    clamp_newsapi_from_date,
    fetch_historical_headlines_for_pair,
    newsapi_earliest_searchable_date,
)


def test_clamp_newsapi_from_date_shortens_vol_lookback_window():
    today = date(2026, 7, 28)
    as_of = date(2026, 7, 13)
    requested = as_of - timedelta(days=60)  # 2026-05-14 — outside plan window
    clamped, was_clamped = clamp_newsapi_from_date(requested, as_of, today=today)
    assert was_clamped is True
    assert clamped == newsapi_earliest_searchable_date(today=today)
    assert clamped == date(2026, 6, 29)
    assert (today - clamped).days == NEWSAPI_MAX_HISTORY_DAYS


def test_clamp_newsapi_from_date_skips_when_as_of_too_old():
    today = date(2026, 7, 28)
    as_of = date(2024, 3, 20)
    clamped, was_clamped = clamp_newsapi_from_date(
        as_of - timedelta(days=14),
        as_of,
        today=today,
    )
    assert was_clamped is True
    assert clamped is None


def test_fetch_historical_clamps_lookback_60_and_keeps_hits(monkeypatch):
    """Replay default lookback=60 must not zero-out NewsAPI for recent as_of."""
    today = date(2026, 7, 28)
    as_of = date(2026, 7, 13)
    captured: dict[str, object] = {}

    def fake_fetch_newsapi(query, cfg, limit=15, *, start_date=None, end_date=None):
        captured["start_date"] = start_date
        captured["end_date"] = end_date
        captured["query"] = query
        return [
            Headline(
                title="RBA holds rates as Australian dollar steadies",
                summary="AUDUSD iron ore",
                source="Reuters",
                url="https://example.com/rba",
                published=datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc),
                provider="newsapi",
            )
        ]

    monkeypatch.setattr(
        "fx_report.news.fetch.fetch_newsapi",
        fake_fetch_newsapi,
    )
    monkeypatch.setattr(
        "fx_report.news.fetch.fetch_inbox_headlines",
        lambda limit=12: [],
    )
    monkeypatch.setattr(
        "fx_report.news.fetch.load_config",
        lambda: {"NEWSAPI_KEY": "test-key-not-secret"},
    )
    monkeypatch.setattr(
        "fx_report.news.fetch.is_set",
        lambda cfg, key: bool(cfg.get(key)),
    )

    headlines, meta = fetch_historical_headlines_for_pair(
        "USD/AUD",
        as_of_date=as_of,
        lookback_days=60,
        max_items=25,
        today=today,
    )

    assert captured["start_date"] == date(2026, 6, 29)
    assert captured["end_date"] == as_of
    assert meta["newsapi_from_clamped"] is True
    assert meta["newsapi_from_requested"] == "2026-05-14"
    assert meta["newsapi_hits"] == 1
    assert meta["historical_news_quality"] == "date_filtered"
    assert meta["historical_lookback_days"] == 60
    assert meta["historical_lookback_days_effective"] == 14
    assert len(headlines) == 1


def test_pipeline_step3_propagates_clamped_news_quality(monkeypatch):
    """Wiring: step3 historical path should surface date_filtered when NewsAPI hits."""
    from types import SimpleNamespace

    from fx_report.market.pairs import get_pair
    from fx_report.pipeline import step3_collect_and_store_statements

    as_of = date(2026, 7, 13)
    fake_market = SimpleNamespace(
        spot=1.44,
        sigma_daily=0.01,
        sigma_annual=0.16,
        source="test",
        asof="2026-07-13",
        notes=[],
        to_dict=lambda: {},
    )

    monkeypatch.setattr(
        "fx_report.pipeline.fetch_market",
        lambda *a, **k: fake_market,
    )

    def fake_hist(spec, *, as_of_date, lookback_days, max_items):
        assert lookback_days == 60
        return (
            [
                Headline(
                    title="RBA and Australian dollar update",
                    summary="AUDUSD",
                    source="Reuters",
                    url="https://example.com/a",
                    published=datetime(2026, 7, 12, tzinfo=timezone.utc),
                    provider="newsapi",
                )
            ],
            {
                "historical_mode": True,
                "historical_as_of": str(as_of_date),
                "historical_lookback_days": lookback_days,
                "historical_lookback_days_effective": 14,
                "newsapi_from_clamped": True,
                "newsapi_hits": 1,
                "historical_news_quality": "date_filtered",
                "limitation": "clamped",
            },
        )

    monkeypatch.setattr(
        "fx_report.pipeline.fetch_historical_headlines_for_pair",
        fake_hist,
    )

    _market, _stmts, headlines, meta = step3_collect_and_store_statements(
        get_pair("USD/AUD"),
        [],
        lookback_days=60,
        max_items=30,
        ai_research=True,
        as_of_date=as_of,
    )
    assert len(headlines) == 1
    assert meta["historical_news_quality"] == "date_filtered"
    assert meta["newsapi_hits"] == 1
    assert meta["newsapi_from_clamped"] is True
    assert meta["ai_research"]["historical_disabled"] is True
