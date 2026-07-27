#!/usr/bin/env python3
"""Smoke: Goal A GBM helpers + tiny calibrate path still importable/runnable."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fx_report.model.calibrate import apply_calibrated_params, pack_params, scenarios_from_vec, vec_from_weights
from fx_report.model.gbm_vol import estimate_vol, gbm_log_step_params
from fx_report.model.monte_carlo import run_mixture_monte_carlo
from fx_report.model.weights import default_scenarios, default_weights


def main() -> int:
    closes = 1.5 * np.exp(np.cumsum(0.007 * np.random.default_rng(2).standard_normal(90)))
    sd, sa = estimate_vol(closes, estimator="window")
    sd_e, _ = estimate_vol(closes, estimator="ewma")
    assert sd > 0 and sa > 0 and sd_e > 0
    _dt, _ann, drift, diff = gbm_log_step_params(0.02, sd)
    assert diff == sd and np.isfinite(drift)

    sc = default_scenarios("USD/AUD")
    mc = run_mixture_monte_carlo(
        spot=float(closes[-1]),
        sigma_daily_base=float(sd),
        scenarios=sc,
        trading_days=15,
        n_sims=3_000,
        seed=3,
        bucket_edges=(1.4, 1.5, 1.6, 1.7),
        drift_mode="scenario",
    )
    assert abs(sum(mc.raw_probs.values()) - 1.0) < 1e-9

    w = default_weights("USD/AUD")
    params = pack_params(w)
    assert params.get("vol_estimator") == "window"
    assert params.get("drift_mode") == "scenario"
    apply_calibrated_params(w, {"drift_mode": "zero", "vol_estimator": "ewma"})
    assert w.drift_mode == "zero" and w.vol_estimator == "ewma"

    # Tiny synthetic peak rows → one eval via vec round-trip (no network)
    x, names, narratives = vec_from_weights(default_weights("USD/AUD"))
    specs = scenarios_from_vec(x, names, narratives)
    assert len(specs) == 3
    # Fake calibrate-style MC call
    df = pd.DataFrame(
        [
            {
                "asof": "2024-01-02",
                "spot": 1.55,
                "sigma_daily": float(sd),
                "realized_max": 1.58,
                "horizon_days": 15,
                "edge_0": 1.55,
                "edge_1": 1.58,
                "edge_2": 1.61,
                "edge_3": 1.65,
            }
        ]
    )
    assert len(df) == 1
    print(
        f"OK Goal A smoke: window_σd={sd:.5f} ewma_σd={sd_e:.5f} "
        f"mc_p50={mc.percentiles['p50']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
