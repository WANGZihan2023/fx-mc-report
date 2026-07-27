#!/usr/bin/env python3
"""Smoke: antithetic vs none on peak estimators (fast, deterministic)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fx_report.model.monte_carlo import run_mixture_monte_carlo
from fx_report.model.weights import ScenarioSpec


def main() -> int:
    spot = 1.55
    sigma = 0.01
    days = 15
    edges = (1.50, 1.53, 1.56, 1.59)

    scenarios = [
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

    seeds = list(range(50, 65))  # 15 reps
    n_sims = 800

    for peak_engine in ("path_max", "brownian_bridge"):
        def vals(vr: str) -> list[float]:
            mc0 = run_mixture_monte_carlo(
                spot=spot,
                sigma_daily_base=sigma,
                scenarios=scenarios,
                trading_days=days,
                n_sims=n_sims,
                seed=seeds[0],
                bucket_edges=edges,
                peak_engine=peak_engine,
                variance_reduction=vr,
            )
            target = list(mc0.raw_probs.keys())[-1]
            out: list[float] = []
            for s in seeds:
                mc = run_mixture_monte_carlo(
                    spot=spot,
                    sigma_daily_base=sigma,
                    scenarios=scenarios,
                    trading_days=days,
                    n_sims=n_sims,
                    seed=s,
                    bucket_edges=edges,
                    peak_engine=peak_engine,
                    variance_reduction=vr,
                )
                out.append(float(mc.raw_probs[target]))
            return out

        v_none = float(np.var(vals("none"), ddof=1))
        v_anti = float(np.var(vals("antithetic"), ddof=1))
        print(f"{peak_engine}: var none={v_none:.6g}  anti={v_anti:.6g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

