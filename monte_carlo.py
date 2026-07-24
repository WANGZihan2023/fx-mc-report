"""Mixture Monte Carlo for USD/AUD path maxima (peak daily level proxy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from weights import ScenarioSpec


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


def bucket_labels_from_edges(edges: Sequence[float]) -> list[str]:
    e = list(edges)
    labels = [f"< {e[0]}"]
    for i in range(len(e) - 1):
        labels.append(f"{e[i]} to {e[i+1]}")
    labels.append(f">= {e[-1]}")
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
) -> MCResult:
    """
    Simulate discrete GBM + compound Poisson jumps under a scenario mixture.

    Each path: S_{t+1} = S_t * exp((μ - 0.5 σ²)Δt + σ√Δt Z + jump)
    Peak proxy = max_t S_t  (including t=0), approximating highest daily level.
    """
    if spot <= 0:
        raise ValueError("spot must be positive")
    if n_sims < 1:
        raise ValueError("n_sims must be positive")

    rng = np.random.default_rng(seed)
    weights = np.array([s.weight for s in scenarios], dtype=np.float64)
    weights = weights / weights.sum()
    scenario_idx = rng.choice(len(scenarios), size=n_sims, p=weights)

    dt = 1.0 / 252.0
    maxima = np.empty(n_sims, dtype=np.float64)
    scenario_counts = {s.name: int(np.sum(scenario_idx == i)) for i, s in enumerate(scenarios)}

    # Simulate per scenario block for speed
    for i, sc in enumerate(scenarios):
        mask = scenario_idx == i
        m = int(np.sum(mask))
        if m == 0:
            continue

        mu = sc.mu_annual + mu_annual_shift
        sigma = sigma_daily_base * sc.sigma_mult * sigma_mult_extra
        # expected_jumps is over full horizon → daily Poisson intensity
        lam_daily = sc.expected_jumps / max(trading_days, 1)

        z = rng.standard_normal((m, trading_days))
        jump_occur = rng.random((m, trading_days)) < lam_daily
        jump_size = rng.normal(sc.jump_mean, sc.jump_std, size=(m, trading_days))
        jumps = np.where(jump_occur, jump_size, 0.0)

        # sigma is daily stdev; annual = sigma * sqrt(252)
        sigma_ann = sigma * np.sqrt(252.0)
        drift = (mu - 0.5 * sigma_ann**2) * dt
        log_rets = drift + sigma * z + jumps

        # cumulative path
        log_path = np.cumsum(log_rets, axis=1)
        path = spot * np.exp(log_path)
        # include starting spot in the maximum
        path_max = np.maximum(np.max(path, axis=1), spot)
        maxima[mask] = path_max

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
        # fallback: put all in first feasible bucket
        for lab in labels:
            out[lab] = 0.0
        # find first label where upper bound > spot or last
        for i, lab in enumerate(labels[:-1]):
            upper = e[i] if i == 0 else e[i]  # rough
            pass
        out[labels[-1]] = 1.0
        return out

    return {k: v / mass for k, v in out.items()}
