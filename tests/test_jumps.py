from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fx_report.model.calibrate import apply_calibrated_params, pack_params
from fx_report.model.jumps import (
    bb_jumps_caveat_message,
    lam_daily_from_expected,
    lambda_annual_from_horizon,
    merton_compensator_drift_daily,
    merton_kappa,
    sample_merton_jumps,
)
from fx_report.model.monte_carlo import run_mixture_monte_carlo
from fx_report.model.weights import ScenarioSpec, default_weights


def _jump_scenario(*, expected_jumps: float = 2.0) -> list[ScenarioSpec]:
    return [
        ScenarioSpec(
            name="toy_jump",
            weight=1.0,
            mu_annual=0.0,
            sigma_mult=1.0,
            expected_jumps=expected_jumps,
            jump_mean=0.01,
            jump_std=0.005,
            narrative="toy",
        )
    ]


def test_annualization_identity() -> None:
    ej, td = 1.5, 66
    lam_ann = lambda_annual_from_horizon(ej, td)
    lam_daily = lam_daily_from_expected(ej, td)
    assert abs(lam_daily - lam_ann / 252.0) < 1e-12


def test_merton_kappa_and_compensator() -> None:
    mu_j, sj = 0.01, 0.005
    k = merton_kappa(mu_j, sj)
    assert k > 0
    c = merton_compensator_drift_daily(2.0, mu_j, sj)
    assert c < 0


def test_jump_sampling_deterministic_with_seed() -> None:
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    a = sample_merton_jumps(
        rng1,
        n_paths=40,
        trading_days=20,
        expected_jumps=3.0,
        jump_mean=0.01,
        jump_std=0.005,
    )
    b = sample_merton_jumps(
        rng2,
        n_paths=40,
        trading_days=20,
        expected_jumps=3.0,
        jump_mean=0.01,
        jump_std=0.005,
    )
    assert np.allclose(a, b)
    assert a.shape == (40, 20)
    # Some mass should land on jumps for this intensity
    assert float(np.mean(a != 0.0)) > 0.0


def test_mc_jump_seed_reproducible_and_none_disables() -> None:
    sc = _jump_scenario(expected_jumps=2.0)
    kw = dict(
        spot=1.55,
        sigma_daily_base=0.008,
        scenarios=sc,
        trading_days=12,
        n_sims=1500,
        seed=7,
        bucket_edges=(1.55, 1.58, 1.61, 1.65),
        peak_engine="path_max",
    )
    mc1 = run_mixture_monte_carlo(**kw, jump_model="merton")
    mc2 = run_mixture_monte_carlo(**kw, jump_model="merton")
    assert np.allclose(mc1.maxima, mc2.maxima)
    assert mc1.jump_model == "merton"
    assert mc1.jump_compensate is False

    mc_none = run_mixture_monte_carlo(**kw, jump_model="none")
    assert mc_none.jump_model == "none"
    # Turning jumps off must change the peak distribution under positive intensity
    assert not np.allclose(mc1.maxima, mc_none.maxima)


def test_compensator_changes_paths_when_enabled() -> None:
    sc = _jump_scenario(expected_jumps=2.0)
    kw = dict(
        spot=1.55,
        sigma_daily_base=0.008,
        scenarios=sc,
        trading_days=12,
        n_sims=1200,
        seed=11,
        bucket_edges=(1.55, 1.58, 1.61, 1.65),
        peak_engine="path_max",
        jump_model="merton",
    )
    off = run_mixture_monte_carlo(**kw, jump_compensate=False)
    on = run_mixture_monte_carlo(**kw, jump_compensate=True)
    assert on.jump_compensate is True
    assert not np.allclose(off.maxima, on.maxima)


def test_bb_caveat_when_jumps_active() -> None:
    sc = _jump_scenario(expected_jumps=1.0)
    msg = bb_jumps_caveat_message(
        peak_engine="brownian_bridge", jump_model="merton", scenarios=sc
    )
    assert msg is not None and "不含跳跃" in msg
    assert (
        bb_jumps_caveat_message(
            peak_engine="path_max", jump_model="merton", scenarios=sc
        )
        is None
    )
    mc = run_mixture_monte_carlo(
        spot=1.55,
        sigma_daily_base=0.008,
        scenarios=sc,
        trading_days=8,
        n_sims=400,
        seed=3,
        bucket_edges=(1.55, 1.58, 1.61, 1.65),
        peak_engine="brownian_bridge",
        jump_model="merton",
    )
    assert mc.bb_jumps_caveat is not None


def test_pack_apply_jump_keys() -> None:
    w = default_weights("USD/AUD")
    params = pack_params(w)
    assert params.get("jump_model") == "merton"
    assert params.get("jump_compensate") is False
    apply_calibrated_params(w, {"jump_model": "none", "jump_compensate": True})
    assert w.jump_model == "none" and w.jump_compensate is True


if __name__ == "__main__":
    test_annualization_identity()
    test_merton_kappa_and_compensator()
    test_jump_sampling_deterministic_with_seed()
    test_mc_jump_seed_reproducible_and_none_disables()
    test_compensator_changes_paths_when_enabled()
    test_bb_caveat_when_jumps_active()
    test_pack_apply_jump_keys()
    print("OK: jump / Cont–Tankov Goal B tests passed")
