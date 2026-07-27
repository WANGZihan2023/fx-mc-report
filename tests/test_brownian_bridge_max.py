"""Unit tests for Brownian-bridge continuous max (Shreve reflection / inverse-CDF)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fx_report.model.brownian_bridge_max import (
    bridge_max_survival_prob,
    sample_bridge_log_maxima,
    simulate_bb_path_maxima,
)
from fx_report.model.gbm_vol import gbm_log_step_params


def test_inverse_cdf_matches_closed_form() -> None:
    a, b, v = 0.0, 0.1, 0.04
    rng = np.random.default_rng(0)
    u = np.array([0.01, 0.1, 0.5, 0.9, 1.0])
    m = sample_bridge_log_maxima(
        np.full(u.shape, a),
        np.full(u.shape, b),
        v,
        rng,
        u=u,
    )
    # Closed form
    disc = (a - b) ** 2 - 2.0 * v * np.log(np.clip(u, 1e-12, 1.0))
    expected = 0.5 * (a + b + np.sqrt(np.maximum(disc, 0.0)))
    expected = np.maximum(expected, max(a, b))
    assert np.allclose(m, expected)
    # U=1 → M = max(a,b)
    assert abs(m[-1] - max(a, b)) < 1e-12


def test_survival_identity_standard_bridge() -> None:
    """P(M ≥ m | 0→0) = exp(-2 m² / v) for m ≥ 0 (standard reflection identity)."""
    a = b = 0.0
    v = 1.0
    for m in (0.5, 1.0, 1.5, 2.0):
        p = float(bridge_max_survival_prob(m, a, b, v))
        assert abs(p - np.exp(-2.0 * m * m / v)) < 1e-12


def test_survival_matches_empirical_inverse_cdf() -> None:
    a, b, v = -0.02, 0.03, 0.0004
    m_level = 0.08
    assert m_level >= max(a, b)
    analytic = float(bridge_max_survival_prob(m_level, a, b, v))
    rng = np.random.default_rng(42)
    n = 80_000
    samples = sample_bridge_log_maxima(
        np.full(n, a),
        np.full(n, b),
        v,
        rng,
    )
    empirical = float(np.mean(samples >= m_level))
    # Monte Carlo SE ~ sqrt(p(1-p)/n); allow ~4 SE
    se = np.sqrt(analytic * (1.0 - analytic) / n)
    assert abs(empirical - analytic) < max(4.0 * se, 0.01)


def test_degenerate_sigma_returns_endpoint_max() -> None:
    rng = np.random.default_rng(1)
    x0 = np.array([1.0, 2.0, 0.5])
    x1 = np.array([1.2, 1.5, 0.5])
    m = sample_bridge_log_maxima(x0, x1, 0.0, rng)
    assert np.allclose(m, np.maximum(x0, x1))


def test_flat_endpoints_u_near_one() -> None:
    rng = np.random.default_rng(2)
    a = np.array([0.0, 0.0])
    b = np.array([0.0, 0.0])
    u = np.array([1.0, 0.999999])
    m = sample_bridge_log_maxima(a, b, 0.01, rng, u=u)
    assert np.all(m >= 0.0)
    assert m[0] == 0.0


def test_multi_day_peak_at_least_endpoint_max() -> None:
    spot = 1.55
    sigma_daily = 0.01
    rng = np.random.default_rng(99)
    peaks = simulate_bb_path_maxima(
        spot,
        sigma_daily,
        mu_annual=0.0,
        trading_days=20,
        n_paths=2000,
        rng=rng,
    )
    assert peaks.shape == (2000,)
    assert np.all(peaks >= spot - 1e-12)
    # Continuous peak must dominate a pure endpoint walk on the same Z draw:
    # rebuild endpoints with a fresh paired simulation and compare distributionally
    # via mean: BB mean peak > discrete endpoint-only mean peak.
    rng2 = np.random.default_rng(99)
    _dt, _sa, drift, diffusion = gbm_log_step_params(0.0, sigma_daily)
    z = rng2.standard_normal((2000, 20))
    log_x = np.empty((2000, 21))
    log_x[:, 0] = np.log(spot)
    log_x[:, 1:] = log_x[:, 0:1] + np.cumsum(drift + diffusion * z, axis=1)
    endpoint_only = np.exp(np.max(log_x, axis=1))
    # Same seed path for BB consumes Z then U — not pathwise comparable; check
    # that BB samples are never below spot and mean exceeds endpoint-only mean
    # under identical Z by reconstructing with shared Z + bridge U.
    u = rng2.random(size=(2000, 20))
    from fx_report.model.brownian_bridge_max import sample_bridge_log_maxima as _s

    seg = _s(log_x[:, :-1], log_x[:, 1:], float(diffusion) ** 2, rng2, u=u)
    bb = np.exp(np.maximum(np.max(seg, axis=1), log_x[:, 0]))
    assert np.all(bb >= endpoint_only - 1e-12)
    assert float(np.mean(bb)) >= float(np.mean(endpoint_only))


def test_shape_mismatch_raises() -> None:
    rng = np.random.default_rng(0)
    try:
        sample_bridge_log_maxima(np.zeros(3), np.zeros(2), 0.01, rng)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on shape mismatch")


if __name__ == "__main__":
    test_inverse_cdf_matches_closed_form()
    test_survival_identity_standard_bridge()
    test_survival_matches_empirical_inverse_cdf()
    test_degenerate_sigma_returns_endpoint_max()
    test_flat_endpoints_u_near_one()
    test_multi_day_peak_at_least_endpoint_max()
    test_shape_mismatch_raises()
    print("OK: brownian_bridge_max tests passed")
