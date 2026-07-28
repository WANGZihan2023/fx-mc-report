"""Unit tests for replay engine A vs C compare helpers."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

from fx_report.model.replay_engine_compare import (
    CandidateDate,
    _parse_dates_list,
    _winner,
    scan_candidate_as_of,
)


def test_winner_and_parse_dates() -> None:
    assert _winner(0.1, 0.0) == "A"
    assert _winner(-0.2, 0.05) == "C"
    assert _winner(0.1, 0.1) == "tie"
    assert _parse_dates_list("2026-07-10, 2026-07-15") == [
        date(2026, 7, 10),
        date(2026, 7, 15),
    ]


def test_scan_stops_on_429() -> None:
    meta_429 = {
        "historical_news_quality": "limited",
        "newsapi_hits": 0,
        "newsapi_from_cache": False,
        "newsapi_http_status": 429,
        "newsapi_error": "HTTP 429: rateLimited",
    }
    with patch(
        "fx_report.model.replay_engine_compare.fetch_historical_headlines_for_pair",
        return_value=([], meta_429),
    ), patch(
        "fx_report.model.replay_engine_compare.get_pair",
        return_value=MagicMock(pair="USD/AUD"),
    ):
        cands, stopped, notes = scan_candidate_as_of(
            "USD/AUD",
            start_date="2026-07-01",
            end_date="2026-07-20",
            step_days=3,
            max_dates=3,
            verbose=False,
        )
    assert stopped is True
    assert cands == []
    assert any("429" in n for n in notes)


def test_scan_keeps_date_filtered() -> None:
    meta_ok = {
        "historical_news_quality": "date_filtered",
        "newsapi_hits": 4,
        "newsapi_from_cache": True,
        "newsapi_http_status": 200,
        "newsapi_error": None,
    }
    with patch(
        "fx_report.model.replay_engine_compare.fetch_historical_headlines_for_pair",
        return_value=([], meta_ok),
    ), patch(
        "fx_report.model.replay_engine_compare.get_pair",
        return_value=MagicMock(pair="USD/AUD"),
    ), patch(
        "fx_report.model.replay_engine_compare._rules_evidence_n",
        return_value=0,
    ):
        cands, stopped, _notes = scan_candidate_as_of(
            "USD/AUD",
            start_date="2026-07-10",
            end_date="2026-07-12",
            step_days=1,
            max_dates=2,
            verbose=False,
        )
    assert stopped is False
    assert len(cands) == 2
    assert all(isinstance(c, CandidateDate) for c in cands)
    assert cands[0].quality == "date_filtered"


if __name__ == "__main__":
    test_winner_and_parse_dates()
    test_scan_stops_on_429()
    test_scan_keeps_date_filtered()
    print("OK")
