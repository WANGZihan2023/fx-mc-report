"""Unit tests for iterative AI researcher (mocked hands / brain)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from fx_report.market.pairs import get_pair
from fx_report.news.ai_research import (
    ResearchHit,
    _allowed_urls,
    _dedupe_hits,
    _hits_to_headlines,
    _llm_extract_outlooks,
    run_ai_research,
)
from fx_report.news.llm import LLMConfig


def test_dedupe_and_allowed_urls():
    hits = [
        ResearchHit("A", "https://a.example/1", "s", "x", "tavily"),
        ResearchHit("A2", "https://a.example/1", "s", "x", "tavily"),
        ResearchHit("B", "https://b.example/2", "s", "y", "brave"),
    ]
    uniq = _dedupe_hits(hits)
    assert len(uniq) == 2
    assert _allowed_urls(uniq) == {"https://a.example/1", "https://b.example/2"}


def test_extract_rejects_invented_urls():
    llm = LLMConfig(api_key="x", base_url="https://example.com/v1", model="m")
    hits = [
        ResearchHit(
            "ING outlook",
            "https://think.ing.com/real",
            "ING",
            "AUD stronger",
            "whitelist",
        )
    ]
    fake = {
        "items": [
            {
                "title": "Fake bank target",
                "summary": "made up",
                "source": "FakeBank",
                "url": "https://evil.example/invented",
                "bank": "FakeBank",
            },
            {
                "title": "ING real",
                "summary": "ok",
                "source": "ING",
                "url": "https://think.ing.com/real",
                "bank": "ING",
            },
        ]
    }
    with patch("fx_report.news.llm._chat_json", return_value=fake):
        out = _llm_extract_outlooks(
            get_pair("USD/AUD"),
            hits,
            llm,
            allowed_urls=_allowed_urls(hits),
        )
    assert len(out) == 1
    assert out[0].url == "https://think.ing.com/real"
    assert "Fake" not in out[0].title


def test_iterative_loop_one_query_at_a_time():
    spec = get_pair("USD/AUD")
    llm = LLMConfig(api_key="x", base_url="https://example.com/v1", model="m")

    plan_calls = {"n": 0}

    def fake_plan(*_a, **_k):
        plan_calls["n"] += 1
        if plan_calls["n"] == 1:
            return {"action": "search", "query": "AUD USD ING outlook", "reason": "banks"}
        return {"action": "stop", "query": "", "reason": "enough"}

    hit = ResearchHit(
        "ING sees AUD upside",
        "https://think.ing.com/article",
        "ING",
        "AUD/USD target",
        "tavily",
    )

    with (
        patch("fx_report.news.ai_research.load_config", return_value={}),
        patch("fx_report.news.ai_research.fetch_whitelist", return_value=[]),
        patch("fx_report.news.ai_research.resolve_llm_config", return_value=llm),
        patch("fx_report.news.ai_research._llm_plan_next_query", side_effect=fake_plan),
        patch(
            "fx_report.news.ai_research.execute_search",
            return_value=([hit], ["tavily"]),
        ) as exec_mock,
        patch("fx_report.news.ai_research._llm_select_hits", return_value=[0]),
        patch(
            "fx_report.news.ai_research._llm_extract_outlooks",
            return_value=_hits_to_headlines([hit], max_n=5),
        ),
    ):
        result = run_ai_research(
            spec,
            info_need_ids=["fed", "rba"],
            llm_cfg=llm,
            max_rounds=3,
            target_keep=5,
        )

    assert result.meta["mode"] == "iterative"
    assert result.meta["queries"] == ["AUD USD ING outlook"]
    assert exec_mock.call_count == 1
    assert result.meta["headlines_out"] >= 1
    assert all(h.url.startswith("http") for h in result.headlines)


def test_limitation_when_only_llm_no_paid_search():
    spec = get_pair("EUR/USD")
    llm = LLMConfig(api_key="x", base_url="https://api.deepseek.com/v1", model="deepseek-chat")
    with (
        patch("fx_report.news.ai_research.load_config", return_value={}),
        patch("fx_report.news.ai_research.fetch_whitelist", return_value=[]),
        patch("fx_report.news.ai_research.has_paid_search_api", return_value=False),
        patch(
            "fx_report.news.ai_research.execute_search",
            return_value=([], ["google_news_rss"]),
        ),
        patch(
            "fx_report.news.ai_research._llm_plan_next_query",
            return_value={"action": "stop", "query": "", "reason": "nothing"},
        ),
    ):
        # kept empty + stop → may still try seed; force stop by empty seed path:
        # action stop with kept empty continues to seed. Patch seed to empty via max_rounds=1
        # and plan returning stop with empty kept uses seed. Simpler: allow one seed round empty.
        result = run_ai_research(spec, llm_cfg=llm, max_rounds=1, target_keep=10)

    assert result.meta.get("limitation")
    assert "DeepSeek" in result.meta["limitation"] or "LLM" in result.meta["limitation"]
    assert result.meta["paid_search"] is False
