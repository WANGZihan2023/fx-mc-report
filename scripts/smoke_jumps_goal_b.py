#!/usr/bin/env python3
"""Smoke: Goal B Merton jumps + BB caveat + pack/apply."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fx_report.model.calibrate import apply_calibrated_params, pack_params
from fx_report.model.monte_carlo import run_mixture_monte_carlo
from fx_report.model.weights import default_scenarios, default_weights


def main() -> int:
    sc = default_scenarios("USD/AUD")
    mc = run_mixture_monte_carlo(
        spot=1.55,
        sigma_daily_base=0.008,
        scenarios=sc,
        trading_days=10,
        n_sims=2_000,
        seed=5,
        bucket_edges=(1.5, 1.55, 1.6, 1.65),
        jump_model="merton",
    )
    assert mc.jump_model == "merton"
    assert abs(sum(mc.raw_probs.values()) - 1.0) < 1e-9

    mc_bb = run_mixture_monte_carlo(
        spot=1.55,
        sigma_daily_base=0.008,
        scenarios=sc,
        trading_days=10,
        n_sims=800,
        seed=5,
        bucket_edges=(1.5, 1.55, 1.6, 1.65),
        peak_engine="brownian_bridge",
        jump_model="merton",
    )
    assert mc_bb.bb_jumps_caveat is not None

    w = default_weights("USD/AUD")
    p = pack_params(w)
    assert "jump_model" in p and "jump_compensate" in p
    apply_calibrated_params(w, {"jump_model": "none"})
    assert w.jump_model == "none"
    print("OK: Goal B jump smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
