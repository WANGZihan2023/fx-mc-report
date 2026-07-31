"""Mixture Monte Carlo for FX path maxima (peak daily / continuous level proxy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from fx_report.model.gbm_vol import gbm_log_step_params, resolve_mu_annual
from fx_report.model.jumps import (
    VALID_JUMP_MODELS,
    bb_jumps_caveat_message,
    lambda_annual_from_horizon,
    merton_compensator_drift_daily,
    normalize_jump_model,
    sample_merton_jumps,
)
from fx_report.model.weights import ScenarioSpec

VALID_PEAK_ENGINES = ("path_max", "brownian_bridge")
VALID_VARIANCE_REDUCTION = ("none", "antithetic")


@dataclass
class MCResult:
    maxima: np.ndarray
    bucket_labels: list[str]
    raw_probs: dict[str, float]
    scenario_counts: dict[str, int]
    n_sims: int
    spot: float
    sigma_daily_base: float
    trading_days: int
    percentiles: dict[str, float]
    peak_engine: str = "path_max"
    variance_reduction: str = "none"
    jump_model: str = "merton"
    jump_compensate: bool = False
    bb_jumps_caveat: str | None = None


def bucket_labels_from_edges(edges: Sequence[float]) -> list[str]:
    from fx_report.format_rate import format_rate

    e = list(edges)
    labels = [f"< {format_rate(e[0])}"]
    for i in range(len(e) - 1):
        labels.append(f"{format_rate(e[i])} to {format_rate(e[i + 1])}")
    labels.append(f">= {format_rate(e[-1])}")
    return labels


def assign_buckets(maxima: np.ndarray, edges: Sequence[float]) -> dict[str, float]:
    labels = bucket_labels_from_edges(edges)
    e = list(edges)
    n = len(maxima)
    counts = np.zeros(len(labels), dtype=np.float64)
    counts[0] = np.sum(maxima < e[0])
    for i in range(len(e) - 1):
        counts[i + 1] = np.sum((maxima >= e[i]) & (maxima < e[i + 1]))
    counts[-1] = np.sum(maxima >= e[-1])
    # Mathematical floor: max cannot be below spot — still report empirical
    return {lab: float(c / n) for lab, c in zip(labels, counts)}


def _simulate_path_max_mixture(
    spot: float,
    sigma_daily_base: float,
    scenarios: list[ScenarioSpec],
    *,
    trading_days: int,
    n_sims: int,
    seed: int,
    mu_annual_shift: float,
    sigma_mult_extra: float,
    drift_mode: str = "scenario",
    carry_mu_annual: float = 0.0,
    variance_reduction: str = "none",
    jump_model: str = "merton",
    jump_compensate: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Exact log-Euler GBM (Hull Ch13) + Merton compound Poisson jumps (Cont–Tankov
    Ch 3.2 / 4.3); peak = max along daily path. Δt = 1/252 yr.

    Jump sizes are Gaussian in log-space (Merton). Occurrence uses Bernoulli
    with p = λ_daily = expected_jumps / trading_days (= λ_ann · Δt).

    If jump_compensate: subtract λ_ann · (E[e^J]−1) · Δt from daily log-drift.
    Default jump_compensate=False preserves pre–Goal-B mean behaviour.
    """
    vr = (variance_reduction or "none").strip().lower()
    if vr not in VALID_VARIANCE_REDUCTION:
        raise ValueError(
            f"unknown variance_reduction={variance_reduction!r}; expected one of {VALID_VARIANCE_REDUCTION}"
        )
    jm = normalize_jump_model(jump_model)
    use_jumps = jm == "merton"

    rng = np.random.default_rng(seed)
    weights = np.array([s.weight for s in scenarios], dtype=np.float64)
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

        if vr == "antithetic":
            # Antithetic for diffusion increments (and reuse the same jump draw per pair)
            base_m = (m + 1) // 2
            z_base = rng.standard_normal((base_m, trading_days))
            z = np.vstack([z_base, -z_base])[:m]

            if use_jumps:
                jumps_base = sample_merton_jumps(
                    rng,
                    n_paths=base_m,
                    trading_days=trading_days,
                    expected_jumps=sc.expected_jumps,
                    jump_mean=sc.jump_mean,
                    jump_std=sc.jump_std,
                )
                jumps = np.vstack([jumps_base, jumps_base])[:m]
            else:
                jumps = np.zeros((m, trading_days), dtype=np.float64)
        else:
            z = rng.standard_normal((m, trading_days))
            if use_jumps:
                jumps = sample_merton_jumps(
                    rng,
                    n_paths=m,
                    trading_days=trading_days,
                    expected_jumps=sc.expected_jumps,
                    jump_mean=sc.jump_mean,
                    jump_std=sc.jump_std,
                )
            else:
                jumps = np.zeros((m, trading_days), dtype=np.float64)

        _dt, _sig_ann, drift, diffusion = gbm_log_step_params(mu, sigma)
        if use_jumps and jump_compensate and float(sc.expected_jumps) > 0.0:
            lam_ann = lambda_annual_from_horizon(sc.expected_jumps, trading_days)
            drift = drift + merton_compensator_drift_daily(
                lam_ann, sc.jump_mean, sc.jump_std
            )
        log_rets = drift + diffusion * z + jumps

        log_path = np.cumsum(log_rets, axis=1)
        path = spot * np.exp(log_path)
        path_max = np.maximum(np.max(path, axis=1), spot)
        maxima[mask] = path_max

    return maxima, scenario_counts


def run_mixture_monte_carlo(
    spot: float,
    sigma_daily_base: float,
    scenarios: list[ScenarioSpec],
    *,
    trading_days: int = 66,
    n_sims: int = 100_000,
    seed: int = 42,
    bucket_edges: Sequence[float] = (1.40, 1.43, 1.46, 1.49),
    mu_annual_shift: float = 0.0,
    sigma_mult_extra: float = 1.0,
    peak_engine: str = "path_max",
    drift_mode: str = "scenario",
    carry_mu_annual: float = 0.0,
    variance_reduction: str = "none",
    jump_model: str = "merton",
    jump_compensate: bool = False,
) -> MCResult:
    """
    Simulate scenario-mixture peaks and bucket them.

    peak_engine:
      - "path_max" (default): discrete GBM + Merton compound Poisson jumps; max along path
      - "brownian_bridge": continuous GBM peak via Brownian-bridge maxima between
        daily endpoints (jumps excluded — see brownian_bridge_max.py). Never falls
        back silently to path_max.

    jump_model:
      - "merton" (default): Cont–Tankov / Merton log-normal compound Poisson on path_max
      - "none": diffusion only (ignores scenario expected_jumps)

    jump_compensate: if True, apply −λ(E[e^J]−1)Δt to daily log-drift (default False).

    drift_mode / carry_mu_annual: see fx_report.model.gbm_vol.resolve_mu_annual
    (defaults preserve prior scenario-μ behaviour).
    """
    engine = (peak_engine or "path_max").strip().lower()
    if engine not in VALID_PEAK_ENGINES:
        raise ValueError(
            f"unknown peak_engine={peak_engine!r}; expected one of {VALID_PEAK_ENGINES}"
        )

    vr = (variance_reduction or "none").strip().lower()
    if vr not in VALID_VARIANCE_REDUCTION:
        raise ValueError(
            f"unknown variance_reduction={variance_reduction!r}; expected one of {VALID_VARIANCE_REDUCTION}"
        )

    jm = normalize_jump_model(jump_model)

    if spot <= 0:
        raise ValueError("spot must be positive")
    if n_sims < 1:
        raise ValueError("n_sims must be positive")
    if not scenarios:
        raise ValueError("scenarios must be non-empty")

    caveat = bb_jumps_caveat_message(
        peak_engine=engine, jump_model=jm, scenarios=scenarios
    )

    if engine == "brownian_bridge":
        from fx_report.model.brownian_bridge_max import run_brownian_bridge_mixture

        maxima, scenario_counts = run_brownian_bridge_mixture(
            spot,
            sigma_daily_base,
            scenarios,
            trading_days=trading_days,
            n_sims=n_sims,
            seed=seed,
            mu_annual_shift=mu_annual_shift,
            sigma_mult_extra=sigma_mult_extra,
            drift_mode=drift_mode,
            carry_mu_annual=carry_mu_annual,
            variance_reduction=vr,
        )
    else:
        maxima, scenario_counts = _simulate_path_max_mixture(
            spot,
            sigma_daily_base,
            scenarios,
            trading_days=trading_days,
            n_sims=n_sims,
            seed=seed,
            mu_annual_shift=mu_annual_shift,
            sigma_mult_extra=sigma_mult_extra,
            drift_mode=drift_mode,
            carry_mu_annual=carry_mu_annual,
            variance_reduction=vr,
            jump_model=jm,
            jump_compensate=bool(jump_compensate),
        )

    raw = assign_buckets(maxima, bucket_edges)
    pct = {
        "p50": float(np.percentile(maxima, 50)),
        "p75": float(np.percentile(maxima, 75)),
        "p90": float(np.percentile(maxima, 90)),
        "p95": float(np.percentile(maxima, 95)),
        "p99": float(np.percentile(maxima, 99)),
        "mean": float(np.mean(maxima)),
    }
    return MCResult(
        maxima=maxima,
        bucket_labels=bucket_labels_from_edges(bucket_edges),
        raw_probs=raw,
        scenario_counts=scenario_counts,
        n_sims=n_sims,
        spot=spot,
        sigma_daily_base=sigma_daily_base,
        trading_days=trading_days,
        percentiles=pct,
        peak_engine=engine,
        variance_reduction=vr,
        jump_model=jm,
        jump_compensate=bool(jump_compensate),
        bb_jumps_caveat=caveat,
    )


def enforce_math_floor(probs: dict[str, float], spot: float, edges: Sequence[float]) -> dict[str, float]:
    """
    Zero out buckets that are mathematically impossible given starting spot,
    then renormalize remaining mass.
    """
    e = list(edges)
    labels = bucket_labels_from_edges(edges)
    out = dict(probs)

    # Impossible: entire bucket strictly below spot
    # < e0 impossible if spot >= e0
    if spot >= e[0]:
        out[labels[0]] = 0.0
    for i in range(len(e) - 1):
        # bucket [e_i, e_{i+1}) impossible if spot >= e_{i+1}
        if spot >= e[i + 1]:
            out[labels[i + 1]] = 0.0

    mass = sum(out.values())
    if mass <= 0:
        for lab in labels:
            out[lab] = 0.0
        out[labels[-1]] = 1.0
        return out

    return {k: v / mass for k, v in out.items()}


# Re-export for callers / docs
__all__ = [
    "MCResult",
    "VALID_PEAK_ENGINES",
    "VALID_VARIANCE_REDUCTION",
    "VALID_JUMP_MODELS",
    "assign_buckets",
    "bucket_labels_from_edges",
    "enforce_math_floor",
    "run_mixture_monte_carlo",
]
