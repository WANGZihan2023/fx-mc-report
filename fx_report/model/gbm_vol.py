"""
Hull-aligned GBM discretization helpers and historical / EWMA volatility.

References (Hull, Options Futures and Other Derivatives, 8e — scanned local PDF;
chapter numbers from English OCR of that file; formulas restated, not quoted):

- Ch 13: GBM dS = μ S dt + σ S dz; exact log Euler
    Δ ln S = (μ − ½σ²) Δt + σ √Δt Z,  Z ~ N(0,1)
- Ch 14: hist vol from log returns u_i = ln(S_i/S_{i−1}); annualize with √N
  where N ≈ 252 trading days/year (ignore calendar closed days)
- Ch 5 / 15: FX risk-neutral drift uses rate differential (carry); real-world μ
  is separate — this module exposes modes without forcing pricing measure
- Ch 22: EWMA variance σ_n² = λ σ_{n−1}² + (1−λ) u_{n−1}² (RiskMetrics λ≈0.94)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

# Hull Ch14: practitioners usually assume ~252 equity/FX trading days per year.
TRADING_DAYS_PER_YEAR: float = 252.0

VolEstimator = Literal["window", "ewma"]
DriftMode = Literal["scenario", "zero", "carry"]

VALID_VOL_ESTIMATORS: tuple[str, ...] = ("window", "ewma")
VALID_DRIFT_MODES: tuple[str, ...] = ("scenario", "zero", "carry")

# RiskMetrics daily EWMA decay (Hull Ch22 discussion of EWMA).
DEFAULT_EWMA_LAMBDA: float = 0.94


def dt_trading_day(trading_days_per_year: float = TRADING_DAYS_PER_YEAR) -> float:
    """Δt in years for one trading-day step."""
    return 1.0 / float(trading_days_per_year)


def annualize_daily_vol(
    sigma_daily: float,
    trading_days_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """σ_ann = σ_day · √N  (Hull Ch14 / Ch21 daily↔annual bridge)."""
    return float(sigma_daily) * float(np.sqrt(trading_days_per_year))


def daily_vol_from_annual(
    sigma_annual: float,
    trading_days_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    """σ_day = σ_ann / √N."""
    return float(sigma_annual) / float(np.sqrt(trading_days_per_year))


def log_returns(closes: np.ndarray) -> np.ndarray:
    """u_i = ln(S_i / S_{i−1}) (Hull Ch14 Table 14.1 style)."""
    c = np.asarray(closes, dtype=np.float64)
    if c.ndim != 1:
        raise ValueError("closes must be 1-D")
    if len(c) < 2:
        return np.empty(0, dtype=np.float64)
    if np.any(c[:-1] <= 0) or np.any(c[1:] <= 0):
        raise ValueError("closes must be positive for log returns")
    return np.log(c[1:] / c[:-1])


def hull_window_vol(
    closes: np.ndarray,
    *,
    trading_days_per_year: float = TRADING_DAYS_PER_YEAR,
) -> tuple[float, float]:
    """
    Hull Ch14 historical volatility on a fixed window of closes.

    Uses sample std of log returns (1/(n−1) form via ddof=1), then
    σ_ann = s · √N. Returns (sigma_daily, sigma_annual).
    """
    rets = log_returns(closes)
    if len(rets) < 2:
        return float("nan"), float("nan")
    sigma_daily = float(np.std(rets, ddof=1))
    return sigma_daily, annualize_daily_vol(sigma_daily, trading_days_per_year)


def hull_ewma_vol(
    closes: np.ndarray,
    *,
    lam: float = DEFAULT_EWMA_LAMBDA,
    trading_days_per_year: float = TRADING_DAYS_PER_YEAR,
) -> tuple[float, float]:
    """
    Hull Ch22-style EWMA daily vol from log returns.

    Recursion: v_n = λ v_{n−1} + (1−λ) u_{n−1}², seeded with the sample
    variance of the window (unbiased). Returns (sigma_daily, sigma_annual)
    for the terminal EWMA variance.
    """
    if not (0.0 < lam < 1.0):
        raise ValueError(f"EWMA lambda must be in (0,1), got {lam}")
    rets = log_returns(closes)
    if len(rets) < 2:
        return float("nan"), float("nan")
    # Seed with sample variance of the window (Hull often starts from long-run var).
    v = float(np.var(rets, ddof=1))
    one_m = 1.0 - lam
    for u in rets:
        v = lam * v + one_m * float(u) * float(u)
    sigma_daily = float(np.sqrt(max(v, 0.0)))
    return sigma_daily, annualize_daily_vol(sigma_daily, trading_days_per_year)


def estimate_vol(
    closes: np.ndarray,
    *,
    estimator: str = "window",
    ewma_lambda: float = DEFAULT_EWMA_LAMBDA,
    trading_days_per_year: float = TRADING_DAYS_PER_YEAR,
) -> tuple[float, float]:
    """Dispatch Hull window vs EWMA; returns (sigma_daily, sigma_annual)."""
    est = (estimator or "window").strip().lower()
    if est not in VALID_VOL_ESTIMATORS:
        raise ValueError(
            f"unknown vol estimator={estimator!r}; expected one of {VALID_VOL_ESTIMATORS}"
        )
    if est == "ewma":
        return hull_ewma_vol(
            closes, lam=ewma_lambda, trading_days_per_year=trading_days_per_year
        )
    return hull_window_vol(closes, trading_days_per_year=trading_days_per_year)


def resolve_mu_annual(
    scenario_mu: float,
    *,
    drift_mode: str = "scenario",
    carry_mu_annual: float = 0.0,
    mu_annual_shift: float = 0.0,
) -> float:
    """
    Map scenario / evidence into the μ used in the GBM drift term.

    - scenario (default): real-world mixture prior μ + evidence shift
    - zero: μ = 0 + shift (martingale-ish real-world placeholder; still Itô −½σ²)
    - carry: interest-differential placeholder (Hull FX RN drift shape) + shift;
      does not fetch live rates — caller supplies carry_mu_annual
    """
    mode = (drift_mode or "scenario").strip().lower()
    if mode not in VALID_DRIFT_MODES:
        raise ValueError(
            f"unknown drift_mode={drift_mode!r}; expected one of {VALID_DRIFT_MODES}"
        )
    shift = float(mu_annual_shift)
    if mode == "zero":
        return shift
    if mode == "carry":
        return float(carry_mu_annual) + shift
    return float(scenario_mu) + shift


def gbm_log_step_params(
    mu_annual: float,
    sigma_daily: float,
    *,
    trading_days_per_year: float = TRADING_DAYS_PER_YEAR,
) -> tuple[float, float, float, float]:
    """
    Exact GBM log-Euler coefficients for one trading-day step.

    Returns (dt, sigma_ann, drift, diffusion) where
      Δ ln S = drift + diffusion · Z,  Z~N(0,1)
      drift = (μ − ½ σ_ann²) Δt
      diffusion = σ_daily = σ_ann · √Δt
    """
    dt = dt_trading_day(trading_days_per_year)
    sigma_ann = annualize_daily_vol(sigma_daily, trading_days_per_year)
    drift = (float(mu_annual) - 0.5 * sigma_ann * sigma_ann) * dt
    diffusion = float(sigma_daily)
    return dt, sigma_ann, drift, diffusion
