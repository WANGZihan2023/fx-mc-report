"""
Brownian-bridge continuous peak (conditional maximum) under GBM / lognormal assumptions.

Method
------
Under GBM, X_t = log S_t is Brownian motion with drift
  ν = μ_ann − ½ σ_ann² ,  dX = ν dt + σ_ann dW.

1. Simulate discrete daily endpoints of X (diffusion only — **no jumps**).
2. Conditional on consecutive endpoints (x_i, x_{i+1}), the path is a Brownian
   bridge; drift is already absorbed into the endpoints.
3. Sample the continuous maximum of each bridge via the reflection principle
   (driftless bridge max inverse-CDF), then take the path peak as
   S₀ · exp(max of segment maxima), including the start.

Bridge max sampler (standard barrier / continuous-monitoring identity):
  For a Brownian bridge from a → b over Δt with Var(ΔX)=σ²Δt, and U~Unif(0,1),
    M = ½ (a + b + √((a−b)² − 2 σ²Δt log U)),   M ≥ max(a, b).

Approximation limits
--------------------
- Jumps (compound Poisson) are **not** included in this engine. Use path_max
  if jump risk should thicken the peak distribution.
- Discrete-day endpoints + continuous bridges approximate the continuous GBM
  peak; finer calendars reduce residual discretisation bias on the endpoints.
"""

from __future__ import annotations

import numpy as np

from fx_report.model.gbm_vol import gbm_log_step_params, resolve_mu_annual
from fx_report.model.weights import ScenarioSpec


def sample_bridge_log_maxima(
    x0: np.ndarray,
    x1: np.ndarray,
    sigma2_dt: float,
    rng: np.random.Generator,
    u: np.ndarray | None = None,
) -> np.ndarray:
    """
    Sample continuous max of BM bridges from x0 → x1 (same length arrays).

    sigma2_dt = σ_ann² · Δt = σ_daily² for one trading day.
    """
    if sigma2_dt <= 0:
        return np.maximum(x0, x1)
    if u is None:
        u = rng.random(size=x0.shape)
    u = np.clip(u, 1e-12, 1.0)
    disc = (x0 - x1) ** 2 - 2.0 * sigma2_dt * np.log(u)
    disc = np.maximum(disc, 0.0)
    m = 0.5 * (x0 + x1 + np.sqrt(disc))
    return np.maximum(m, np.maximum(x0, x1))


def simulate_bb_path_maxima(
    spot: float,
    sigma_daily: float,
    *,
    mu_annual: float,
    trading_days: int,
    n_paths: int,
    rng: np.random.Generator,
    variance_reduction: str = "none",
) -> np.ndarray:
    """
    Simulate n_paths continuous GBM peaks via daily endpoints + Brownian bridges.

    Returns array of shape (n_paths,) of path maxima in spot units.
    """
    vr = (variance_reduction or "none").strip().lower()
    if vr not in ("none", "antithetic"):
        raise ValueError("variance_reduction must be one of: none, antithetic")

    if n_paths < 1:
        return np.empty(0, dtype=np.float64)
    if spot <= 0:
        raise ValueError("spot must be positive")
    if trading_days < 1:
        raise ValueError("trading_days must be >= 1")

    _dt, _sigma_ann, drift, diffusion = gbm_log_step_params(mu_annual, sigma_daily)
    sigma2_dt = float(diffusion) ** 2  # = σ_ann² · Δt for one trading day

    if vr == "antithetic":
        base_n = (n_paths + 1) // 2
        z_base = rng.standard_normal((base_n, trading_days))
        z = np.vstack([z_base, -z_base])[:n_paths]
    else:
        z = rng.standard_normal((n_paths, trading_days))

    log_rets = drift + diffusion * z
    # log-spot path endpoints: shape (n, T+1) with column 0 = log(spot)
    log_x = np.empty((n_paths, trading_days + 1), dtype=np.float64)
    log_x[:, 0] = np.log(spot)
    log_x[:, 1:] = log_x[:, 0:1] + np.cumsum(log_rets, axis=1)

    # Continuous max on each daily bridge segment
    if vr == "antithetic":
        base_n = (n_paths + 1) // 2
        u_base = rng.random(size=(base_n, trading_days))
        u = np.vstack([u_base, 1.0 - u_base])[:n_paths]
    else:
        u = None

    seg_max = sample_bridge_log_maxima(
        log_x[:, :-1],
        log_x[:, 1:],
        sigma2_dt,
        rng,
        u=u,
    )
    log_peak = np.maximum(np.max(seg_max, axis=1), log_x[:, 0])
    return np.exp(log_peak)


def run_brownian_bridge_mixture(
    spot: float,
    sigma_daily_base: float,
    scenarios: list[ScenarioSpec],
    *,
    trading_days: int = 66,
    n_sims: int = 100_000,
    seed: int = 42,
    mu_annual_shift: float = 0.0,
    sigma_mult_extra: float = 1.0,
    drift_mode: str = "scenario",
    carry_mu_annual: float = 0.0,
    variance_reduction: str = "none",
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Mixture over scenarios: BB continuous peaks (jumps ignored).

    Returns (maxima, scenario_counts).
    """
    if spot <= 0:
        raise ValueError("spot must be positive")
    if n_sims < 1:
        raise ValueError("n_sims must be positive")
    if not scenarios:
        raise ValueError("scenarios must be non-empty")

    rng = np.random.default_rng(seed)
    weights = np.array([s.weight for s in scenarios], dtype=np.float64)
    if weights.sum() <= 0:
        raise ValueError("scenario weights must sum to a positive value")
    weights = weights / weights.sum()
    scenario_idx = rng.choice(len(scenarios), size=n_sims, p=weights)

    maxima = np.empty(n_sims, dtype=np.float64)
    scenario_counts = {s.name: int(np.sum(scenario_idx == i)) for i, s in enumerate(scenarios)}

    for i, sc in enumerate(scenarios):
        mask = scenario_idx == i
        m = int(np.sum(mask))
        if m == 0:
            continue
        mu = resolve_mu_annual(
            sc.mu_annual,
            drift_mode=drift_mode,
            carry_mu_annual=carry_mu_annual,
            mu_annual_shift=mu_annual_shift,
        )
        sigma = sigma_daily_base * sc.sigma_mult * sigma_mult_extra
        # Note: sc.expected_jumps / jump_* intentionally unused (BB = diffusion only).
        maxima[mask] = simulate_bb_path_maxima(
            spot,
            sigma,
            mu_annual=mu,
            trading_days=trading_days,
            n_paths=m,
            rng=rng,
            variance_reduction=variance_reduction,
        )

    return maxima, scenario_counts


PEAK_ENGINE_DOC = (
    "brownian_bridge: continuous GBM peak via daily endpoints + reflection-principle "
    "Brownian-bridge maxima; compound-Poisson jumps are excluded."
)
