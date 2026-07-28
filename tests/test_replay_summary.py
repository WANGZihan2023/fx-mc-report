from __future__ import annotations

import json

from fx_report.model.replay_summary import replay_summary_dataframe, summarize_replay_json
from fx_report.news.fetch import _parse_dt


def test_parse_dt_supports_iso_and_rfc2822():
    iso = _parse_dt("2024-01-05T12:34:56Z")
    rss = _parse_dt("Fri, 05 Jan 2024 12:34:56 GMT")

    assert iso is not None
    assert rss is not None
    assert iso.isoformat() == "2024-01-05T12:34:56+00:00"
    assert rss.isoformat() == "2024-01-05T12:34:56+00:00"


def test_replay_summary_marks_historical_news_working(tmp_path):
    payload = {
        "summary": {
            "pair": "USD/AUD",
            "analysis_pair": "USD/AUD",
            "bullish_currency": "USD",
            "n_rows": 2,
            "start_date": "2024-01-01",
            "end_date": "2024-01-15",
            "step_days": 7,
            "argmax_hit_rate": 0.5,
            "mean_brier": 0.42,
            "mean_skill_brier": 0.11,
            "generated_at": "2026-07-28T08:00:00+00:00",
        },
        "rows": [
            {
                "as_of": "2024-01-01",
                "evidence_n": 2,
                "historical_news_quality": "date_filtered",
            },
            {
                "as_of": "2024-01-08",
                "evidence_n": 0,
                "historical_news_quality": "limited",
            },
        ],
    }
    path = tmp_path / "replay_backtest_USDAUD_20240101_20240115.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    row = summarize_replay_json(path)
    df = replay_summary_dataframe(tmp_path)

    assert row["historical_news_working"] == "yes"
    assert row["date_filtered_count"] == 1
    assert row["limited_count"] == 1
    assert float(row["evidence_mean"]) == 1.0
    assert list(df["historical_news_working"]) == ["yes"]
