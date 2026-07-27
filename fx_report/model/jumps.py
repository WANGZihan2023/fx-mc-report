"""
Cont–Tankov Merton / compound-Poisson jump helpers for FX path Monte Carlo.

Reference (local PDF, not in git): Cont & Tankov, *Financial Modelling with
Jump Processes* — Ch 2.5 (Poisson / compensator), Ch 3.2 (compound Poisson),
Ch 4.3 (Merton jump-diffusion), Ch 6.1 (simulation). Formulas restated only.

Parameter convention in this repo
---------------------------------
Scenario field ``expected_jumps`` = E[N_T] over the MC horizon (``trading_days``
steps), **not** annual intensity. Annual intensity and per-step Bernoulli rate:

  λ_ann = expected_jumps · N / trading_days ,   N = 252
  λ_daily = λ_ann · Δt = expected_jumps / trading_days ,   Δt = 1/252

Daily step uses a Bernoulli approximation (standard for small λΔt):
  P(jump on day) = λ_daily ,  J | jump ~ N(μ_J, σ_J²)  (log-return jump).

Merton compensator (optional, off by default for API stability):
  κ = E[e^J] − 1 = exp(μ_J + ½ σ_J²) − 1
  continuous log-drift per day gains  − λ_ann · κ · Δt
so that the continuous part's mean stays coherent with the compound-Poisson
mean contribution when jumps are added in log space.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from fx_report.model.gbm_vol import TRADING_DAYS_PER_YEAR, dt_trading_day

JumpModel = Literal["merton", "none"]
VALID_JUMP_MODELS: tuple[str, ...] = ("merton", "none")


def normalize_jump_model(jump_model: str | None) -> str:
    m = (jump_model or "merton").strip().lower()
    if m not in VALID_JUMP_MODELS:
        raise ValueError(
            f"unknown jump_model={jump_model!r}; expected one of {VALID_JUMP_MODELS}"
        )
    return m


def lambda_annual_from_horizon(
    expected_jumps: float,
    trading_days: int,
    *,
    trading_days_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """λ_ann (per year) from horizon expected jump count E[N_T]."""
    td = max(int(trading_days), 1)
    return float(expected_jumps) * float(trading_days_per_year) / float(td)


def lam_daily_from_expected(
    expected_jumps: float,
    trading_days: int,
) -> float:
    """
    Per-day jump probability / intensity for Bernoulli step.

    Equals λ_ann · Δt with Δt=1/252 when expected_jumps = λ_ann · (trading_days/252).
    """
    return float(expected_jumps) / float(max(int(trading_days), 1))


def merton_jump_mgf(mu_j: float, sigma_j: float) -> float:
    """E[e^J] for J ~ N(μ_J, σ_J²)."""
    sj = float(sigma_j)
    return float(np.exp(float(mu_j) + 0.5 * sj * sj))


def merton_kappa(mu_j: float, sigma_j: float) -> float:
    """κ = E[e^J] − 1 (Merton mean relative jump size)."""
    return merton_jump_mgf(mu_j, sigma_j) - 1.0


def merton_compensator_drift_daily(
    lam_annual: float,
    mu_j: float,
    sigma_j: float,
    *,
    trading_days_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Additive adjustment to daily log-drift: −λ_ann · κ · Δt.

    Use when ``jump_compensate=True`` so E[Δ ln S] from continuous part
    offsets the compound-Poisson contribution in the multiplicative model.
    """
    dt = dt_trading_day(trading_days_per_year)
    return -float(lam_annual) * merton_kappa(mu_j, sigma_j) * dt


def sample_merton_jumps(
    rng: np.random.Generator,
    *,
    n_paths: int,
    trading_days: int,
    expected_jumps: float,
    jump_mean: float,
    jump_std: float,
) -> np.ndarray:
    """
    Sample (n_paths × trading_days) log-jump increments (0 on no-jump days).

    Bernoulli occurrence with p = lam_daily; size ~ N(μ_J, σ_J).
    """
    if n_paths < 1 or trading_days < 1:
        return np.zeros((max(n_paths, 0), max(trading_days, 0)), dtype=np.float64)
    lam_daily = lam_daily_from_expected(expected_jumps, trading_days)
    if lam_daily <= 0.0 or float(jump_std) < 0.0:
        return np.zeros((n_paths, trading_days), dtype=np.float64)
    occur = rng.random((n_paths, trading_days)) < lam_daily
    sizes = rng.normal(float(jump_mean), float(jump_std), size=(n_paths, trading_days))
    return np.where(occur, sizes, 0.0)


def scenario_has_jumps(scenarios, *, jump_model: str = "merton") -> bool:
    """True if jump_model is active and any scenario has E[N_T] > 0."""
    if normalize_jump_model(jump_model) == "none":
        return False
    return any(float(getattr(s, "expected_jumps", 0.0) or 0.0) > 0.0 for s in scenarios)


def bb_jumps_caveat_message(
    *,
    peak_engine: str,
    jump_model: str,
    scenarios,
) -> str | None:
    """
    Honest caveat when brownian_bridge is selected but jump intensity > 0.

    BB engine remains continuous-max diffusion-only; jumps are ignored.
    """
    eng = (peak_engine or "path_max").strip().lower()
    if eng != "brownian_bridge":
        return None
    if not scenario_has_jumps(scenarios, jump_model=jump_model):
        return None
    return (
        "brownian_bridge 峰值引擎不含跳跃（连续 GBM 桥最大值）；"
        "当前情景 E[jumps]>0 且 jump_model=merton 时跳跃被忽略。"
        "若需跳跃加厚峰值尾部，请改用 peak_engine=path_max。"
    )
