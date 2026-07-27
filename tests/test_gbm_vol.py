"""Unit tests for Hull-aligned GBM / vol helpers (Goal A)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fx_report.model.gbm_vol import (
    TRADING_DAYS_PER_YEAR,
    annualize_daily_vol,
    estimate_vol,
    gbm_log_step_params,
    hull_window_vol,
    log_returns,
    resolve_mu_annual,
)
from fx_report.model.monte_carlo import run_mixture_monte_carlo
from fx_report.model.weights import default_scenarios


def test_log_returns_and_window_vol_hull_style() -> None:
    # Synthetic GBM-ish path with known daily vol ~1%
    rng = np.random.default_rng(0)
    n = 80
    sigma_d = 0.01
    closes = 100.0 * np.exp(np.cumsum(sigma_d * rng.standard_normal(n)))
    rets = log_returns(closes)
    assert len(rets) == n - 1
    sd, sa = hull_window_vol(closes)
    assert math.isfinite(sd) and sd > 0
    assert abs(sa - annualize_daily_vol(sd)) < 1e-12
    assert abs(sa - sd * math.sqrt(TRADING_DAYS_PER_YEAR)) < 1e-12
    # Sample std should be near 0.01
    assert abs(sd - float(np.std(rets, ddof=1))) < 1e-15


def test_ewma_vol_runs() -> None:
    closes = np.array([1.0, 1.01, 0.99, 1.02, 1.00, 1.03, 0.98, 1.01], dtype=float)
    sd_w, _ = estimate_vol(closes, estimator="window")
    sd_e, sa_e = estimate_vol(closes, estimator="ewma", ewma_lambda=0.94)
    assert math.isfinite(sd_e) and sd_e > 0
    assert abs(sa_e - sd_e * math.sqrt(252)) < 1e-12
    assert sd_w != sd_e  # EWMA terminal ≠ plain window in general


def test_gbm_log_step_consistency() -> None:
    mu = 0.05
    sigma_daily = 0.006
    dt, sig_ann, drift, diff = gbm_log_step_params(mu, sigma_daily)
    assert abs(dt - 1.0 / 252.0) < 1e-15
    assert abs(sig_ann - sigma_daily * math.sqrt(252)) < 1e-12
    assert abs(diff - sigma_daily) < 1e-15
    expected_drift = (mu - 0.5 * sig_ann**2) * dt
    assert abs(drift - expected_drift) < 1e-15


def test_resolve_mu_modes() -> None:
    assert abs(resolve_mu_annual(0.06, drift_mode="scenario", mu_annual_shift=0.01) - 0.07) < 1e-12
    assert abs(resolve_mu_annual(0.06, drift_mode="zero", mu_annual_shift=0.01) - 0.01) < 1e-12
    assert (
        abs(
            resolve_mu_annual(
                0.06, drift_mode="carry", carry_mu_annual=-0.02, mu_annual_shift=0.01
            )
            - (-0.01)
        )
        < 1e-12
    )


def test_mc_smoke_default_api() -> None:
    mc = run_mixture_monte_carlo(
        spot=1.55,
        sigma_daily_base=0.0065,
        scenarios=default_scenarios("USD/AUD"),
        trading_days=10,
        n_sims=2_000,
        seed=1,
        bucket_edges=(1.55, 1.58, 1.61, 1.65),
    )
    assert abs(sum(mc.raw_probs.values()) - 1.0) < 1e-9
    assert mc.maxima.min() >= 1.55 - 1e-12


def test_mc_drift_mode_zero_changes_distribution() -> None:
    sc = default_scenarios("USD/AUD")
    common = dict(
        spot=1.55,
        sigma_daily_base=0.008,
        scenarios=sc,
        trading_days=40,
        n_sims=6_000,
        seed=11,
        bucket_edges=(1.55, 1.58, 1.61, 1.70),
    )
    a = run_mixture_monte_carlo(**common, drift_mode="scenario")
    b = run_mixture_monte_carlo(**common, drift_mode="zero")
    # Same seed/scenarios but μ forced to 0 → mean peak should differ
    assert abs(a.percentiles["mean"] - b.percentiles["mean"]) > 1e-4


if __name__ == "__main__":
    test_log_returns_and_window_vol_hull_style()
    test_ewma_vol_runs()
    test_gbm_log_step_consistency()
    test_resolve_mu_modes()
    test_mc_smoke_default_api()
    test_mc_drift_mode_zero_changes_distribution()
    print("OK: gbm_vol / MC Goal A tests passed")
