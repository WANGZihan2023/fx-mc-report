from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fx_report.model.monte_carlo import run_mixture_monte_carlo
from fx_report.model.weights import ScenarioSpec


def _toy_diffusion_scenario() -> list[ScenarioSpec]:
    # expected_jumps=0 isolates diffusion contribution (jumps are effectively disabled).
    return [
        ScenarioSpec(
            name="toy_diffusion",
            weight=1.0,
            mu_annual=0.0,
            sigma_mult=1.0,
            expected_jumps=0.0,
            jump_mean=0.0,
            jump_std=0.01,
            narrative="toy",
        )
    ]


def _empirical_variance_for_bucket(
    *,
    peak_engine: str,
    variance_reduction: str,
    seeds: list[int],
    n_sims: int,
    trading_days: int,
    spot: float,
    sigma_daily_base: float,
    edges: tuple[float, float, float, float],
) -> float:
    scenarios = _toy_diffusion_scenario()
    # Determine a stable target bucket from one reference run.
    mc_ref = run_mixture_monte_carlo(
        spot=spot,
        sigma_daily_base=sigma_daily_base,
        scenarios=scenarios,
        trading_days=trading_days,
        n_sims=n_sims,
        seed=seeds[0],
        bucket_edges=edges,
        peak_engine=peak_engine,
        variance_reduction=variance_reduction,
    )
    labels = list(mc_ref.raw_probs.keys())
    target_label = labels[-1]  # >= last edge

    vals: list[float] = []
    for s in seeds:
        mc = run_mixture_monte_carlo(
            spot=spot,
            sigma_daily_base=sigma_daily_base,
            scenarios=scenarios,
            trading_days=trading_days,
            n_sims=n_sims,
            seed=s,
            bucket_edges=edges,
            peak_engine=peak_engine,
            variance_reduction=variance_reduction,
        )
        vals.append(float(mc.raw_probs[target_label]))

    return float(np.var(vals, ddof=1))


def test_antithetic_is_deterministic_with_seed() -> None:
    scenarios = _toy_diffusion_scenario()
    mc1 = run_mixture_monte_carlo(
        spot=1.55,
        sigma_daily_base=0.01,
        scenarios=scenarios,
        trading_days=10,
        n_sims=800,
        seed=123,
        bucket_edges=(1.55, 1.58, 1.61, 1.65),
        peak_engine="path_max",
        variance_reduction="antithetic",
    )
    mc2 = run_mixture_monte_carlo(
        spot=1.55,
        sigma_daily_base=0.01,
        scenarios=scenarios,
        trading_days=10,
        n_sims=800,
        seed=123,
        bucket_edges=(1.55, 1.58, 1.61, 1.65),
        peak_engine="path_max",
        variance_reduction="antithetic",
    )
    assert mc1.variance_reduction == "antithetic"
    assert abs(sum(mc1.raw_probs.values()) - 1.0) < 1e-9
    assert np.allclose(mc1.maxima, mc2.maxima)


def test_antithetic_reduces_empirical_variance_on_diffusion_toy() -> None:
    # Empirical variance of a bucket probability proxy should drop vs. none.
    spot = 1.55
    sigma_daily_base = 0.01
    trading_days = 15
    n_sims = 1200
    edges = (1.50, 1.53, 1.56, 1.59)
    seeds = list(range(100, 120))  # 20 replications

    for peak_engine in ("path_max", "brownian_bridge"):
        var_none = _empirical_variance_for_bucket(
            peak_engine=peak_engine,
            variance_reduction="none",
            seeds=seeds,
            n_sims=n_sims,
            trading_days=trading_days,
            spot=spot,
            sigma_daily_base=sigma_daily_base,
            edges=edges,
        )
        var_anti = _empirical_variance_for_bucket(
            peak_engine=peak_engine,
            variance_reduction="antithetic",
            seeds=seeds,
            n_sims=n_sims,
            trading_days=trading_days,
            spot=spot,
            sigma_daily_base=sigma_daily_base,
            edges=edges,
        )

        # Not all toy settings guarantee strict improvement, but for diffusion-only
        # maxima it is expected to reduce estimator variance.
        assert var_anti < var_none, (peak_engine, var_none, var_anti)


if __name__ == "__main__":
    test_antithetic_is_deterministic_with_seed()
    test_antithetic_reduces_empirical_variance_on_diffusion_toy()
    print("OK: peak variance reduction tests passed")

