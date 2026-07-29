"""Cost-control: historical / replay forces cheap path unless explicit override."""

from __future__ import annotations

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
