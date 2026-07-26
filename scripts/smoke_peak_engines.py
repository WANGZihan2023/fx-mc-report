#!/usr/bin/env python3
"""Smoke: path_max vs brownian_bridge on same spot/sigma/horizon → valid prob dicts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fx_report.model.monte_carlo import run_mixture_monte_carlo
from fx_report.model.weights import default_scenarios


def main() -> int:
    spot = 1.55
    sigma = 0.0065
    days = 66
    edges = (1.55, 1.58, 1.61, 1.65)
    scenarios = default_scenarios("USD/AUD")
    n_sims = 8_000
    seed = 7

    results = {}
    for engine in ("path_max", "brownian_bridge"):
        mc = run_mixture_monte_carlo(
            spot=spot,
            sigma_daily_base=sigma,
            scenarios=scenarios,
            trading_days=days,
            n_sims=n_sims,
            seed=seed,
            bucket_edges=edges,
            peak_engine=engine,
        )
        assert mc.peak_engine == engine, mc.peak_engine
        s = sum(mc.raw_probs.values())
        assert abs(s - 1.0) < 1e-9, (engine, s, mc.raw_probs)
        assert all(v >= 0 for v in mc.raw_probs.values()), mc.raw_probs
        assert mc.maxima.min() >= spot - 1e-12, (engine, mc.maxima.min())
        results[engine] = mc.raw_probs
        print(f"{engine}: sum={s:.12f}  probs={ {k: round(v, 4) for k, v in mc.raw_probs.items()} }")

    # Unknown engine must raise (no silent fallback)
    try:
        run_mixture_monte_carlo(
            spot=spot,
            sigma_daily_base=sigma,
            scenarios=scenarios,
            trading_days=days,
            n_sims=100,
            seed=seed,
            bucket_edges=edges,
            peak_engine="not_a_real_engine",
        )
    except ValueError as exc:
        print(f"unknown engine raised as expected: {exc}")
    else:
        print("ERROR: expected ValueError for unknown peak_engine")
        return 1

    print("OK: both engines produce valid probability dictionaries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
