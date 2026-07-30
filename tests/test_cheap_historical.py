"""Cost-control: historical / replay forces cheap path unless explicit override."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import patch

from fx_report.news.gdelt import (
    _GDELT_MEM_CACHE,
    _gdelt_cache_key,
    fetch_gdelt_doc,
)
from fx_report.pipeline import step3_collect_and_store_statements
from fx_report.ui.ux_helpers import cheap_historical_mode, resolve_replay_ai_research


def test_resolve_replay_ai_research_default_cheap():
    # Sidebar AI ON must not leak into historical replay.
    assert resolve_replay_ai_research(allow_historical_ai=False, sidebar_ai_research=True) is False
    assert resolve_replay_ai_research(allow_historical_ai=False, sidebar_ai_research=False) is False
    assert resolve_replay_ai_research(allow_historical_ai=False) is False


def test_resolve_replay_ai_research_expensive_override():
    assert resolve_replay_ai_research(allow_historical_ai=True) is True
    assert resolve_replay_ai_research(allow_historical_ai=True, sidebar_ai_research=True) is True
    assert resolve_replay_ai_research(allow_historical_ai=True, sidebar_ai_research=False) is False


def test_cheap_historical_mode_helper():
    assert cheap_historical_mode(as_of_date=None) is False
    assert cheap_historical_mode(as_of_date=date(2026, 7, 13)) is True
    assert (
        cheap_historical_mode(as_of_date=date(2026, 7, 13), allow_historical_ai=True) is False
    )


def test_step3_forces_ai_off_when_as_of_even_if_flag_true(monkeypatch):
    """Belt-and-suspenders: as_of + ai_research=True still skips AI/Tavily."""
    from types import SimpleNamespace

    from fx_report.market.pairs import get_pair

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
    monkeypatch.setattr(
        "fx_report.pipeline.fetch_historical_headlines_for_pair",
        lambda *a, **k: (
            [],
            {
                "historical_mode": True,
                "historical_news_quality": "limited",
                "limitation": "test",
            },
        ),
    )

    def boom_ai(*_a, **_k):
        raise AssertionError("run_ai_research must not be called in cheap historical")

    with patch("fx_report.news.ai_research.run_ai_research", boom_ai):
        _m, _s, _h, meta = step3_collect_and_store_statements(
            get_pair("USD/AUD"),
            [],
            skip_news=False,
            ai_research=True,
            allow_historical_ai=False,
            as_of_date=as_of,
        )
    assert meta.get("cheap_historical") is True
    assert meta["ai_research"]["historical_disabled"] is True
    assert meta["ai_research"]["enabled"] is False
    assert meta["ai_research"].get("cheap_historical") is True


def test_gdelt_disk_cache_hit_miss(tmp_path, monkeypatch):
    monkeypatch.setenv("FX_GDELT_CACHE", str(tmp_path))
    _GDELT_MEM_CACHE.clear()

    payload = {
        "articles": [
            {
                "url": "https://example.com/aud-cache",
                "title": "RBA holds; AUDUSD firm",
                "seendate": "20260710T120000Z",
                "domain": "reuters.com",
            }
        ]
    }
    hits = {"n": 0}

    def fake_request(url, timeout, *, max_retries=2):
        hits["n"] += 1
        return payload, None, 200

    monkeypatch.setattr("fx_report.news.gdelt._gdelt_request_json", fake_request)

    start, end = date(2026, 6, 29), date(2026, 7, 13)
    q = "RBA OR AUDUSD"
    meta1: dict = {}
    h1 = fetch_gdelt_doc(q, start_date=start, end_date=end, limit=10, call_meta=meta1)
    assert hits["n"] == 1
    assert meta1.get("from_cache") is False
    assert len(h1) == 1

    key = _gdelt_cache_key(q, start=start, end=end, limit=10)
    assert (tmp_path / f"{key}.json").is_file()

    _GDELT_MEM_CACHE.clear()  # force disk path
    meta2: dict = {}
    h2 = fetch_gdelt_doc(q, start_date=start, end_date=end, limit=10, call_meta=meta2)
    assert hits["n"] == 1  # no second HTTP
    assert meta2.get("from_cache") is True
    assert len(h2) == 1
    assert h2[0].title == h1[0].title


def test_gdelt_failed_response_short_ttl_not_forever(tmp_path, monkeypatch):
    """Errors get a short negative cache; after TTL expiry we re-fetch."""
    from datetime import datetime, timedelta, timezone

    import fx_report.news.gdelt as gdelt_mod

    monkeypatch.setenv("FX_GDELT_CACHE", str(tmp_path))
    _GDELT_MEM_CACHE.clear()
    hits = {"n": 0}

    def fake_request(url, timeout, *, max_retries=2):
        hits["n"] += 1
        return None, "HTTP 429: Too Many Requests", 429

    monkeypatch.setattr("fx_report.news.gdelt._gdelt_request_json", fake_request)

    start, end = date(2026, 6, 29), date(2026, 7, 13)
    q = "RBA OR AUDUSD"
    meta1: dict = {}
    assert fetch_gdelt_doc(q, start_date=start, end_date=end, limit=10, call_meta=meta1) == []
    assert hits["n"] == 1
    assert meta1.get("error")
    key = _gdelt_cache_key(q, start=start, end=end, limit=10)
    path = tmp_path / f"{key}.json"
    assert path.is_file()
    env = json.loads(path.read_text(encoding="utf-8"))
    assert env.get("status") == "error"

    # Within TTL: serve from cache (no second HTTP)
    _GDELT_MEM_CACHE.clear()
    meta2: dict = {}
    assert fetch_gdelt_doc(q, start_date=start, end_date=end, limit=10, call_meta=meta2) == []
    assert hits["n"] == 1
    assert meta2.get("from_cache") is True

    # Expire the envelope → miss → HTTP again
    env["cached_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    path.write_text(json.dumps(env), encoding="utf-8")
    _GDELT_MEM_CACHE.clear()
    meta3: dict = {}
    assert fetch_gdelt_doc(q, start_date=start, end_date=end, limit=10, call_meta=meta3) == []
    assert hits["n"] == 2
    assert meta3.get("from_cache") is False
    assert gdelt_mod.GDELT_CACHE_TTL_ERROR_S < gdelt_mod.GDELT_CACHE_TTL_OK_S


def test_format_cheap_historical_caption():
    from fx_report.ui.ux_helpers import format_cheap_historical_caption, step3_pool_size

    cap = format_cheap_historical_caption(
        {
            "cheap_historical": True,
            "gdelt_hits": 3,
            "newsapi_hits": 1,
            "inbox_dated_hits": 0,
            "gdelt_from_cache": True,
        }
    )
    assert "cheap_historical=ON" in cap
    assert "AI强制关" in cap
    assert "GDELT=3" in cap
    assert "GDELT缓存命中" in cap
    assert step3_pool_size(10, historical=False) == 30
    assert step3_pool_size(40, historical=False) == 90
    assert step3_pool_size(10, historical=True) >= 10
