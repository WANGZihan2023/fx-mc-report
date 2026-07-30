"""Regression: historical NewsAPI from-date must respect plan window."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from urllib.error import HTTPError

from fx_report.news.fetch import (
    NEWSAPI_MAX_HISTORY_DAYS,
    Headline,
    clamp_newsapi_from_date,
    fetch_historical_headlines_for_pair,
    fetch_newsapi,
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

    def fake_fetch_newsapi(
        query, cfg, limit=15, *, start_date=None, end_date=None, call_meta=None
    ):
        captured["start_date"] = start_date
        captured["end_date"] = end_date
        captured["query"] = query
        if call_meta is not None:
            call_meta.clear()
            call_meta.update(
                {
                    "error": None,
                    "http_status": 200,
                    "from_cache": False,
                    "used_domains": True,
                    "total_results": 1,
                }
            )
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
        "fx_report.news.gdelt.fetch_gdelt_doc",
        lambda *a, **k: [],
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
    # GDELT ~90d window can keep the full requested lookback; NewsAPI is still clamped.
    assert meta["historical_lookback_days_effective"] == 60
    assert meta["newsapi_from"] == "2026-06-29"
    assert meta["gdelt_from"] == "2026-05-14"
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


def test_fetch_newsapi_retries_without_domains_when_empty(monkeypatch, tmp_path):
    """ok-but-empty with domains must fall through to unrestricted search."""
    calls: list[str] = []

    def fake_http_json(url: str, timeout: int = 20):
        calls.append(url)
        if "domains=" in url:
            return {"status": "ok", "totalResults": 0, "articles": []}
        return {
            "status": "ok",
            "totalResults": 1,
            "articles": [
                {
                    "title": "RBA keeps cash rate steady",
                    "description": "AUDUSD",
                    "source": {"name": "Reuters"},
                    "url": "https://example.com/rba2",
                    "publishedAt": "2026-07-12T10:00:00Z",
                }
            ],
        }

    monkeypatch.setattr("fx_report.news.fetch._http_json", fake_http_json)
    monkeypatch.setattr("fx_report.news.fetch._NEWSAPI_MEM_CACHE", {})
    monkeypatch.setattr(
        "fx_report.news.fetch._newsapi_cache_dir",
        lambda: tmp_path / "newsapi_cache",
    )

    meta: dict = {}
    out = fetch_newsapi(
        "RBA",
        {"NEWSAPI_KEY": "test-key"},
        limit=5,
        start_date=date(2026, 6, 29),
        end_date=date(2026, 7, 13),
        call_meta=meta,
    )
    assert len(out) == 1
    assert meta["used_domains"] is False
    assert meta["error"] is None
    assert any("domains=" in u for u in calls)
    assert any("domains=" not in u.split("&apiKey=")[0] for u in calls)


def test_fetch_newsapi_surfaces_429(monkeypatch, tmp_path):
    def boom(url: str, timeout: int = 20):
        raise HTTPError(url, 429, "Too Many Requests", hdrs=None, fp=None)  # type: ignore[arg-type]

    monkeypatch.setattr("fx_report.news.fetch._http_json", boom)
    monkeypatch.setattr("fx_report.news.fetch._NEWSAPI_SLEEP", lambda _s: None)
    monkeypatch.setattr("fx_report.news.fetch._NEWSAPI_MEM_CACHE", {})
    monkeypatch.setattr(
        "fx_report.news.fetch._newsapi_cache_dir",
        lambda: tmp_path / "newsapi_cache",
    )

    meta: dict = {}
    out = fetch_newsapi(
        "RBA",
        {"NEWSAPI_KEY": "test-key"},
        limit=5,
        start_date=date(2026, 6, 29),
        end_date=date(2026, 7, 13),
        call_meta=meta,
    )
    assert out == []
    assert meta["http_status"] == 429
    assert meta["error"] and "429" in str(meta["error"])


def test_historical_surfaces_newsapi_error_in_limitation(monkeypatch):
    today = date(2026, 7, 28)
    as_of = date(2026, 7, 13)

    def fake_fetch_newsapi(query, cfg, limit=15, *, start_date=None, end_date=None, call_meta=None):
        if call_meta is not None:
            call_meta.clear()
            call_meta.update(
                {
                    "error": "HTTP 429: Too Many Requests",
                    "http_status": 429,
                    "from_cache": False,
                    "used_domains": None,
                    "total_results": None,
                }
            )
        return []

    monkeypatch.setattr("fx_report.news.fetch.fetch_newsapi", fake_fetch_newsapi)
    monkeypatch.setattr("fx_report.news.fetch.fetch_inbox_headlines", lambda limit=12: [])
    monkeypatch.setattr("fx_report.news.gdelt.fetch_gdelt_doc", lambda *a, **k: [])
    monkeypatch.setattr(
        "fx_report.news.fetch.load_config",
        lambda: {"NEWSAPI_KEY": "test-key-not-secret"},
    )
    monkeypatch.setattr(
        "fx_report.news.fetch.is_set",
        lambda cfg, key: bool(cfg.get(key)),
    )

    _hl, meta = fetch_historical_headlines_for_pair(
        "USD/AUD",
        as_of_date=as_of,
        lookback_days=60,
        max_items=25,
        today=today,
    )
    assert meta["newsapi_hits"] == 0
    assert meta["historical_news_quality"] == "limited"
    assert meta["newsapi_http_status"] == 429
    assert "429" in str(meta["newsapi_error"])
    assert "429" in meta["limitation"]


def test_pipeline_evidence_n_positive_for_clamped_as_of_20260713(monkeypatch):
    """End-to-end: lookback=60 + clamp + one NewsAPI hit → evidence_n>0 date_filtered."""
    from types import SimpleNamespace

    from fx_report.market.pairs import get_pair
    from fx_report.model.weights import ModelWeights
    from fx_report.pipeline import step4_evaluate_impact

    as_of = date(2026, 7, 13)
    today = date(2026, 7, 28)
    fake_market = SimpleNamespace(
        spot=1.44,
        sigma_daily=0.01,
        sigma_annual=0.16,
        source="test",
        asof="2026-07-13",
        notes=[],
        ret_1d=0.0,
        ret_5d=0.0,
        to_dict=lambda: {},
    )

    monkeypatch.setattr(
        "fx_report.news.fetch.fetch_inbox_headlines",
        lambda limit=12: [],
    )
    monkeypatch.setattr(
        "fx_report.news.gdelt.fetch_gdelt_doc",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "fx_report.news.fetch.load_config",
        lambda: {"NEWSAPI_KEY": "test-key-not-secret"},
    )
    monkeypatch.setattr(
        "fx_report.news.fetch.is_set",
        lambda cfg, key: bool(cfg.get(key)),
    )

    def fake_fetch_newsapi(
        query, cfg, limit=15, *, start_date=None, end_date=None, call_meta=None
    ):
        assert start_date == date(2026, 6, 29)
        assert end_date == as_of
        if call_meta is not None:
            call_meta.clear()
            call_meta.update(
                {
                    "error": None,
                    "http_status": 200,
                    "from_cache": False,
                    "used_domains": True,
                    "total_results": 1,
                }
            )
        return [
            Headline(
                title="RBA hawkish hold lifts Australian dollar as USD softens",
                summary="AUDUSD rallies after RBA signal; iron ore steady",
                source="Reuters",
                url="https://example.com/rba-aud",
                published=datetime(2026, 7, 12, 8, 0, tzinfo=timezone.utc),
                provider="newsapi",
            )
        ]

    monkeypatch.setattr("fx_report.news.fetch.fetch_newsapi", fake_fetch_newsapi)

    headlines, hist_meta = fetch_historical_headlines_for_pair(
        get_pair("USD/AUD"),
        as_of_date=as_of,
        lookback_days=60,
        max_items=25,
        today=today,
    )
    assert hist_meta["newsapi_from_clamped"] is True
    assert hist_meta["historical_news_quality"] == "date_filtered"
    assert hist_meta["newsapi_hits"] == 1
    assert len(headlines) == 1

    evidence, news_meta = step4_evaluate_impact(
        headlines,
        get_pair("USD/AUD"),
        fake_market,
        ModelWeights(),
        mode="rules",
        max_items=10,
        template_policy="off",
        fetch_fulltext=False,
        as_of_date=as_of,
    )
    news_meta.update({k: hist_meta[k] for k in hist_meta if k not in news_meta})
    assert news_meta["historical_news_quality"] == "date_filtered"
    assert len(evidence) > 0
    assert news_meta["evidence_n"] > 0


def test_newsapi_failed_response_short_ttl(tmp_path, monkeypatch):
    """429/errors are negative-cached briefly; after TTL we retry HTTP."""
    import json
    from io import BytesIO

    from fx_report.news.fetch import (
        NEWSAPI_CACHE_TTL_ERROR_S,
        NEWSAPI_CACHE_TTL_OK_S,
        _NEWSAPI_MEM_CACHE,
        _newsapi_cache_key,
    )

    hits = {"n": 0}

    def boom(url: str, timeout: int = 20):
        hits["n"] += 1
        raise HTTPError(
            url,
            429,
            "Too Many Requests",
            hdrs=None,
            fp=BytesIO(b'{"code":"rateLimited"}'),
        )

    monkeypatch.setattr("fx_report.news.fetch._http_json", boom)
    monkeypatch.setattr("fx_report.news.fetch._NEWSAPI_SLEEP", lambda _s: None)
    _NEWSAPI_MEM_CACHE.clear()
    monkeypatch.setattr(
        "fx_report.news.fetch._newsapi_cache_dir",
        lambda: tmp_path / "newsapi_cache",
    )

    start, end = date(2026, 7, 1), date(2026, 7, 13)
    meta1: dict = {}
    assert (
        fetch_newsapi(
            "RBA",
            {"NEWSAPI_KEY": "test-key"},
            limit=5,
            start_date=start,
            end_date=end,
            call_meta=meta1,
        )
        == []
    )
    assert hits["n"] >= 1
    first_hits = hits["n"]

    meta2: dict = {}
    assert (
        fetch_newsapi(
            "RBA",
            {"NEWSAPI_KEY": "test-key"},
            limit=5,
            start_date=start,
            end_date=end,
            call_meta=meta2,
        )
        == []
    )
    assert hits["n"] == first_hits  # served from error cache
    assert meta2.get("from_cache") is True

    key = _newsapi_cache_key("RBA", start=start, end=end, domains=False)
    path = tmp_path / "newsapi_cache" / f"{key}.json"
    env = json.loads(path.read_text(encoding="utf-8"))
    assert env["status"] == "error"
    env["cached_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    path.write_text(json.dumps(env), encoding="utf-8")
    _NEWSAPI_MEM_CACHE.clear()

    meta3: dict = {}
    assert (
        fetch_newsapi(
            "RBA",
            {"NEWSAPI_KEY": "test-key"},
            limit=5,
            start_date=start,
            end_date=end,
            call_meta=meta3,
        )
        == []
    )
    assert hits["n"] > first_hits
    assert NEWSAPI_CACHE_TTL_ERROR_S < NEWSAPI_CACHE_TTL_OK_S
