from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from fx_report.model.replay_backtest import run_replay_backtest


def _synthetic_series() -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=180, freq="B")
    vals = 1.42 + np.linspace(0.0, 0.12, len(idx)) + 0.01 * np.sin(np.arange(len(idx)) / 5.0)
    return pd.Series(vals, index=idx)


def test_replay_backtest_smoke_price_only(monkeypatch, tmp_path):
    series = _synthetic_series()

    def fake_fetch_history_series(*_args, **_kwargs):
        return series, "synthetic", ["test series"]

    def fake_historical_news(*_args, **_kwargs):
        return [], {
            "historical_mode": True,
            "historical_news_quality": "limited",
            "limitation": "test historical news unavailable",
        }

    monkeypatch.setattr(
        "fx_report.market.fetch_data.fetch_history_series",
        fake_fetch_history_series,
    )
    monkeypatch.setattr(
        "fx_report.model.replay_backtest.fetch_history_series",
        fake_fetch_history_series,
    )
    monkeypatch.setattr(
        "fx_report.pipeline.fetch_historical_headlines_for_pair",
        fake_historical_news,
    )

    result = run_replay_backtest(
        "USD/AUD",
        start_date=date(2024, 3, 1),
        end_date=date(2024, 3, 20),
        step_days=10,
        out_dir=tmp_path,
        sims=300,
        days=10,
        lookback=20,
        max_dates=2,
        no_fulltext=True,
        verbose=False,
    )

    assert result.n_rows == 2
    assert set(["as_of", "pred_bucket", "true_bucket", "skill_brier", "historical_news_quality"]).issubset(
        result.table.columns
    )
    assert result.summary["historical_news_quality_counts"]["limited"] == 2
    assert result.csv_path.exists()
    assert result.json_path.exists()
