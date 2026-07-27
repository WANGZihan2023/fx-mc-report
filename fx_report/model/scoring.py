"""
Proper scoring rules for multiclass peak-bucket forecasts (Goal E / Gneiting).

Brier and log scores are strictly proper; skill scores compare against a naive
baseline (uniform or climatological frequency). Reliability helpers support
auditable calibration diagnostics (Goal J).

CRPS for continuous path-max is deferred — needs sample maxima + realized max;
see docs/math_goal_E_gneiting.md.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np

BaselineMode = Literal["uniform", "frequency"]


def as_prob_matrix(p: np.ndarray | Sequence[float]) -> np.ndarray:
    """Ensure shape (n, k). Accepts (k,) or (n, k)."""
    arr = np.asarray(p, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"probs must be 1-D or 2-D, got shape {arr.shape}")
    return arr


def normalize_probs(p: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Clip floor and row-normalize (for log-loss stability)."""
    arr = as_prob_matrix(p)
    arr = np.clip(arr, eps, None)
    row_sum = arr.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum <= 0, 1.0, row_sum)
    return arr / row_sum


def normalize_probs_soft(p: np.ndarray) -> np.ndarray:
    """Row-normalize without forcing a probability floor (Brier-friendly)."""
    arr = as_prob_matrix(p)
    arr = np.maximum(arr, 0.0)
    row_sum = arr.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum <= 0, 1.0, row_sum)
    return arr / row_sum


def uniform_probs(n_classes: int) -> np.ndarray:
    if n_classes < 1:
        raise ValueError("n_classes must be >= 1")
    return np.full(n_classes, 1.0 / n_classes, dtype=np.float64)


def frequency_baseline(y: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Climatology: mean one-hot over samples → shape (k,)."""
    ym = as_prob_matrix(y)
    if ym.shape[0] == 0:
        raise ValueError("frequency_baseline needs at least one row")
    freq = ym.mean(axis=0)
    s = float(freq.sum())
    if s <= 0:
        return uniform_probs(ym.shape[1])
    return freq / s


def uniform_brier_constant(n_classes: int) -> float:
    """
    Exact multiclass Brier of uniform forecast vs any one-hot outcome:
    (K-1)/K.
    """
    k = int(n_classes)
    if k < 1:
        return float("nan")
    return (k - 1) / k


def brier_multiclass(
    p: np.ndarray | Sequence[Sequence[float]],
    y: np.ndarray | Sequence[Sequence[float]],
) -> float:
    """Mean multiclass Brier: (1/n) Σ_i Σ_k (p_ik − y_ik)²."""
    pm = normalize_probs_soft(p)
    ym = as_prob_matrix(y)
    if pm.shape != ym.shape:
        raise ValueError(f"p/y shape mismatch: {pm.shape} vs {ym.shape}")
    if pm.shape[0] == 0:
        return float("nan")
    return float(np.mean(np.sum((pm - ym) ** 2, axis=1)))


def log_loss_multiclass(
    p: np.ndarray | Sequence[Sequence[float]],
    y: np.ndarray | Sequence[Sequence[float]],
    *,
    eps: float = 1e-6,
) -> float:
    """Mean multiclass log-loss / negative log score: −(1/n) Σ_i Σ_k y_ik log p_ik."""
    pm = normalize_probs(p, eps=eps)
    ym = as_prob_matrix(y)
    if pm.shape != ym.shape:
        raise ValueError(f"p/y shape mismatch: {pm.shape} vs {ym.shape}")
    if pm.shape[0] == 0:
        return float("nan")
    return float(np.mean(-np.sum(ym * np.log(pm), axis=1)))


def skill_score(score_model: float, score_baseline: float) -> float:
    """
    1 − score_model / score_baseline (higher better).
    NaN if baseline is non-finite or ~0 (undefined).
    """
    if not np.isfinite(score_model) or not np.isfinite(score_baseline):
        return float("nan")
    if abs(score_baseline) < 1e-15:
        return float("nan")
    return float(1.0 - score_model / score_baseline)


def hit_rate_argmax(
    p: np.ndarray | Sequence[Sequence[float]],
    y: np.ndarray | Sequence[Sequence[float]],
) -> float:
    pm = as_prob_matrix(p)
    ym = as_prob_matrix(y)
    if pm.shape[0] == 0:
        return float("nan")
    return float(np.mean(np.argmax(pm, axis=1) == np.argmax(ym, axis=1)))


def reliability_by_prob_bins(
    p_event: np.ndarray | Sequence[float],
    hit: np.ndarray | Sequence[float],
    *,
    n_bins: int = 5,
) -> list[dict[str, float]]:
    """
    Reliability table for a binary event (e.g. argmax-correct).

    Bins predicted probability; reports mean predicted vs empirical hit rate.
    """
    p = np.asarray(p_event, dtype=np.float64).ravel()
    h = np.asarray(hit, dtype=np.float64).ravel()
    if p.shape != h.shape:
        raise ValueError("p_event and hit must have the same length")
    if len(p) == 0 or n_bins < 1:
        return []

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "n": 0.0,
                    "mean_p": float("nan"),
                    "hit_rate": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "n": float(n),
                "mean_p": float(np.mean(p[mask])),
                "hit_rate": float(np.mean(h[mask])),
            }
        )
    return rows


def reliability_per_bucket(
    p: np.ndarray | Sequence[Sequence[float]],
    y: np.ndarray | Sequence[Sequence[float]],
) -> list[dict[str, float]]:
    """Per-bucket: mean predicted probability vs empirical hit frequency."""
    pm = normalize_probs_soft(p)
    ym = as_prob_matrix(y)
    if pm.shape != ym.shape or pm.shape[0] == 0:
        return []
    n, k = pm.shape
    rows: list[dict[str, float]] = []
    for j in range(k):
        rows.append(
            {
                "bucket": float(j),
                "n": float(n),
                "mean_p": float(np.mean(pm[:, j])),
                "emp_rate": float(np.mean(ym[:, j])),
            }
        )
    return rows


def expected_calibration_error(
    reliability_rows: Sequence[dict[str, float]],
) -> float:
    """Weighted |mean_p − hit_rate| over non-empty bins (argmax reliability)."""
    total_n = 0.0
    weighted = 0.0
    for row in reliability_rows:
        n = float(row.get("n") or 0.0)
        if n <= 0:
            continue
        mp = row.get("mean_p")
        hr = row.get("hit_rate")
        if mp is None or hr is None or not np.isfinite(mp) or not np.isfinite(hr):
            continue
        weighted += n * abs(float(mp) - float(hr))
        total_n += n
    if total_n <= 0:
        return float("nan")
    return float(weighted / total_n)


def resolve_baseline_probs(
    y: np.ndarray,
    *,
    mode: BaselineMode = "frequency",
    baseline_probs: np.ndarray | Sequence[float] | None = None,
) -> tuple[np.ndarray, BaselineMode]:
    """Pick baseline vector (k,). External baseline_probs overrides mode."""
    ym = as_prob_matrix(y)
    k = ym.shape[1]
    if baseline_probs is not None:
        bp = np.asarray(baseline_probs, dtype=np.float64).ravel()
        if bp.shape[0] != k:
            raise ValueError(f"baseline_probs length {bp.shape[0]} != k={k}")
        bp = np.clip(bp, 1e-6, 1.0)
        bp = bp / bp.sum()
        return bp, mode
    if mode == "uniform" or ym.shape[0] == 0:
        return uniform_probs(k), "uniform"
    return frequency_baseline(ym), "frequency"


def summarize_forecast_scores(
    p: np.ndarray | Sequence[Sequence[float]],
    y: np.ndarray | Sequence[Sequence[float]],
    *,
    baseline_mode: BaselineMode = "frequency",
    baseline_probs: np.ndarray | Sequence[float] | None = None,
    n_reliability_bins: int = 5,
) -> dict[str, Any]:
    """
    Aggregate metrics for a batch of multiclass forecasts.

    Returns backward-compatible keys (n, brier, logloss, hit_rate) plus
    skill_*, baseline_*, and compact reliability summaries.
    """
    pm = normalize_probs_soft(p)
    ym = as_prob_matrix(y)
    n = int(pm.shape[0])
    if n == 0 or pm.shape != ym.shape:
        return {
            "n": 0.0,
            "brier": float("nan"),
            "logloss": float("nan"),
            "hit_rate": float("nan"),
            "skill_brier": float("nan"),
            "skill_logloss": float("nan"),
            "baseline_brier": float("nan"),
            "baseline_logloss": float("nan"),
            "baseline_mode": baseline_mode,
            "reliability_ece": float("nan"),
            "reliability_argmax": [],
            "reliability_buckets": [],
        }

    brier = brier_multiclass(pm, ym)
    logloss = log_loss_multiclass(pm, ym)
    hit = hit_rate_argmax(pm, ym)

    bp, used_mode = resolve_baseline_probs(
        ym, mode=baseline_mode, baseline_probs=baseline_probs
    )
    base_p = np.tile(bp.reshape(1, -1), (n, 1))
    base_brier = brier_multiclass(base_p, ym)
    base_ll = log_loss_multiclass(base_p, ym)

    pred_idx = np.argmax(pm, axis=1)
    true_idx = np.argmax(ym, axis=1)
    p_max = pm[np.arange(n), pred_idx]
    hits = (pred_idx == true_idx).astype(np.float64)
    rel_argmax = reliability_by_prob_bins(p_max, hits, n_bins=n_reliability_bins)
    rel_buckets = reliability_per_bucket(pm, ym)
    ece = expected_calibration_error(rel_argmax)

    return {
        "n": float(n),
        "brier": brier,
        "logloss": logloss,
        "hit_rate": hit,
        "skill_brier": skill_score(brier, base_brier),
        "skill_logloss": skill_score(logloss, base_ll),
        "baseline_brier": base_brier,
        "baseline_logloss": base_ll,
        "baseline_mode": used_mode,
        "baseline_probs": [float(x) for x in bp.tolist()],
        "reliability_ece": ece,
        "reliability_argmax": rel_argmax,
        "reliability_buckets": rel_buckets,
    }
