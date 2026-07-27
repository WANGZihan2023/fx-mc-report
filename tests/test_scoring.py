"""Unit tests for multiclass proper scoring helpers (Goal E)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fx_report.model.scoring import (
    brier_multiclass,
    expected_calibration_error,
    frequency_baseline,
    log_loss_multiclass,
    reliability_by_prob_bins,
    reliability_per_bucket,
    skill_score,
    summarize_forecast_scores,
    uniform_brier_constant,
    uniform_probs,
)


def test_perfect_forecast_zero_brier_and_logloss() -> None:
    y = np.eye(3)
    p = y.copy()
    assert brier_multiclass(p, y) < 1e-12
    # log-loss uses eps floor → near-zero, not exact zero
    assert log_loss_multiclass(p, y) < 1e-5


def test_uniform_brier_constant() -> None:
    k = 5
    y = np.eye(k)
    p = np.tile(uniform_probs(k), (k, 1))
    b = brier_multiclass(p, y)
    assert abs(b - uniform_brier_constant(k)) < 1e-12
    assert abs(b - 0.8) < 1e-12


def test_skill_score_vs_uniform() -> None:
    # Model slightly better than uniform on 2-class
    y = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    p_model = np.array([[0.7, 0.3], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6]])
    p_base = np.tile(uniform_probs(2), (4, 1))
    sm = brier_multiclass(p_model, y)
    sb = brier_multiclass(p_base, y)
    sk = skill_score(sm, sb)
    assert sk > 0
    assert abs(sk - (1.0 - sm / sb)) < 1e-12


def test_frequency_baseline_and_summarize() -> None:
    y = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    freq = frequency_baseline(y)
    assert abs(freq.sum() - 1.0) < 1e-12
    assert abs(freq[0] - 0.5) < 1e-12
    p = np.tile(freq, (4, 1))
    out = summarize_forecast_scores(p, y, baseline_mode="frequency")
    assert out["n"] == 4.0
    assert abs(out["skill_brier"]) < 1e-9  # model == climatology
    assert out["baseline_mode"] == "frequency"
    assert len(out["reliability_buckets"]) == 3


def test_reliability_bins_and_ece() -> None:
    p_event = np.array([0.1, 0.15, 0.55, 0.6, 0.9, 0.95])
    hit = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    rows = reliability_by_prob_bins(p_event, hit, n_bins=3)
    assert len(rows) == 3
    ece = expected_calibration_error(rows)
    assert math.isfinite(ece)
    assert ece >= 0.0


def test_reliability_per_bucket() -> None:
    p = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
    y = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rows = reliability_per_bucket(p, y)
    assert len(rows) == 3
    assert abs(rows[0]["mean_p"] - 0.45) < 1e-9
    assert abs(rows[0]["emp_rate"] - 0.5) < 1e-9


def test_holdout_uses_external_baseline() -> None:
    y_hold = np.eye(3)
    p = np.full((3, 3), 1.0 / 3.0)
    train_clim = np.array([0.5, 0.3, 0.2])
    out = summarize_forecast_scores(
        p, y_hold, baseline_mode="frequency", baseline_probs=train_clim
    )
    assert abs(out["baseline_probs"][0] - 0.5) < 1e-9
    assert math.isfinite(out["skill_brier"])
