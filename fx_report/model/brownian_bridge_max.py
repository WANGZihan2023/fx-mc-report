"""
Brownian-bridge continuous peak (conditional maximum) under GBM / lognormal assumptions.

Theory (Shreve II — formulas restated, not quoted)
-------------------------------------------------
- §3.7 Reflection principle / distribution of BM and its maximum
- §4.7 Brownian bridge as conditioned BM
- Goal A / Hull: work on X = ln S with exact log-Euler steps

Under GBM, X_t = log S_t is Brownian motion with drift
  ν = μ_ann − ½ σ_ann² ,  dX = ν dt + σ_ann dW.

1. Simulate discrete daily endpoints of X (diffusion only — **no jumps**).
2. Conditional on consecutive endpoints (a, b), the path is a Brownian bridge;
   drift is already absorbed into the endpoints (Shreve §4.7.5).
3. Sample the continuous maximum of each bridge via the reflection-principle
   inverse-CDF, then take the path peak as exp(max of segment maxima),
   including the start.

Conditional survival (m ≥ max(a, b), v = σ_ann² Δt = σ_daily²):
  P(M ≥ m | X_0=a, X_Δt=b) = exp(−2(m−a)(m−b)/v)

Inverse-CDF (U ~ Unif(0,1)):
  M = ½ (a + b + √((a−b)² − 2 v log U)),   then M ← max(M, max(a, b))

Multi-day horizon: for T trading days, draw T independent bridge maxima on
consecutive endpoint pairs; path log-peak = max over segments (and X_0).

Approximation limits
--------------------
- Jumps (compound Poisson / Merton) are **not** included. Use path_max if jump
  risk should thicken the peak distribution. When peak_engine=brownian_bridge
  and scenario expected_jumps > 0 with jump_model=merton, monte_carlo / UI /
  pipeline emit an explicit caveat (fx_report.model.jumps.bb_jumps_caveat_message).
- Discrete-day endpoints + continuous bridges approximate the continuous GBM
  peak; finer calendars reduce residual discretisation bias on the endpoints.

See docs/math_goal_D_shreve.md.
"""

from __future__ import annotations

import numpy as np

from fx_report.model.gbm_vol import gbm_log_step_params, resolve_mu_annual
from fx_report.model.weights import ScenarioSpec

# Numerical floor for Uniform draws in inverse-CDF (avoids log(0) / overflow).
_U_FLOOR = 1e-12


def bridge_max_survival_prob(
    m: np.ndarray | float,
    a: np.ndarray | float,
    b: np.ndarray | float,
    sigma2_dt: float,
) -> np.ndarray:
    """
    P(M ≥ m | endpoints a→b) for a driftless BM bridge with Var(ΔX)=sigma2_dt.

    For m < max(a, b) the probability is 1.0 (maximum always ≥ endpoints).
    When sigma2_dt ≤ 0, returns 1.0 iff m ≤ max(a, b) else 0.0.
    """
    m_b, a_b, b_b = np.broadcast_arrays(
        np.asarray(m, dtype=np.float64),
        np.asarray(a, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
    )
    endpoint_max = np.maximum(a_b, b_b)

    if sigma2_dt <= 0:
        return (m_b <= endpoint_max).astype(np.float64)

    # m ≥ max(a,b): exp(−2(m−a)(m−b)/v); else 1. Clip expo for stability.
    expo = -2.0 * (m_b - a_b) * (m_b - b_b) / float(sigma2_dt)
    surv = np.exp(np.clip(expo, -700.0, 0.0))
    return np.where(m_b >= endpoint_max, surv, 1.0)


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
    Uses reflection-principle inverse-CDF (see module docstring).
    """
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    if x0.shape != x1.shape:
        raise ValueError("x0 and x1 must have the same shape")

    endpoint_max = np.maximum(x0, x1)
    if sigma2_dt <= 0:
        return endpoint_max.copy()

    if u is None:
        u = rng.random(size=x0.shape)
    else:
        u = np.asarray(u, dtype=np.float64)
        if u.shape != x0.shape:
            raise ValueError("u must have the same shape as x0/x1")

    # Clip away 0 to keep log U finite; keep 1.0 so M can equal endpoint max.
    u = np.clip(u, _U_FLOOR, 1.0)
    disc = (x0 - x1) ** 2 - 2.0 * float(sigma2_dt) * np.log(u)
    disc = np.maximum(disc, 0.0)
    m = 0.5 * (x0 + x1 + np.sqrt(disc))
    # Flat / near-flat endpoints + floating error: never below endpoint max.
    return np.maximum(m, endpoint_max)


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

    Multi-day composition: independent bridge maxima on each consecutive pair of
    log-endpoints; path peak = exp(max over segment maxima and log(spot)).

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
    "Brownian-bridge maxima (Shreve II §3.7/§4.7); compound-Poisson jumps are excluded."
)

# Short UI / audit blurb (Chinese) when peak_engine=brownian_bridge.
BB_CONTINUOUS_MAX_NOTE_ZH = (
    "brownian_bridge：日端点之间用反射原理（Shreve）抽取连续路径最大值；"
    "复合泊松跳跃不计（需跳跃加厚尾部请用 path_max）。"
)
