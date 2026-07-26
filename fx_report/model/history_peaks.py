"""
Stage 0 — historical peak samples for MC calibration.

For each as-of date, record spot, realized path maximum over the next
`horizon_days` trading days, local sigma, and which relative bucket the
realized max hit.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from fx_report.market.fetch_data import fetch_history_series
from fx_report.market.pairs import PairSpec, edges_from_spot, get_pair
from fx_report.model.monte_carlo import assign_buckets, bucket_labels_from_edges
from fx_report.model.weights import default_weights


def _sigma_daily(closes: np.ndarray) -> float:
    if len(closes) < 3:
        return float("nan")
    rets = np.log(closes[1:] / closes[:-1])
    if len(rets) < 2:
        return float("nan")
    return float(np.std(rets, ddof=1))


def _bucket_index(realized_max: float, edges: Sequence[float]) -> int:
    e = list(edges)
    if realized_max < e[0]:
        return 0
    for i in range(len(e) - 1):
        if e[i] <= realized_max < e[i + 1]:
            return i + 1
    return len(e)


def build_peak_samples(
    pair: str | PairSpec = "USD/AUD",
    *,
    horizon_days: int = 66,
    vol_lookback: int = 60,
    history_days: int = 1500,
    step: int = 5,
    bucket_pct_cuts: tuple[float, float, float, float] | None = None,
    series: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Build rolling-window peak samples.

    Returns (dataframe, meta). On fetch failure, raises RuntimeError
    (callers / CLI should catch and report gracefully).
    """
    spec = get_pair(pair) if isinstance(pair, str) else pair
    cuts = bucket_pct_cuts or default_weights(spec).bucket_pct_cuts or spec.bucket_pct_cuts

    source = "provided"
    notes: list[str] = []
    if series is None:
        series, source, notes = fetch_history_series(spec, history_days=history_days)

    series = series.dropna().sort_index()
    if series.index.duplicated().any():
        series = series[~series.index.duplicated(keep="last")]

    values = series.values.astype(float)
    dates = series.index
    n = len(values)
    need = vol_lookback + horizon_days + 2
    if n < need:
        raise RuntimeError(
            f"{spec.pair} 历史不足：有 {n} 根，至少需要约 {need} 根 "
            f"(vol_lookback={vol_lookback} + horizon={horizon_days})。"
        )

    rows: list[dict] = []
    # Last usable asof index leaves `horizon_days` future bars
    last_asof = n - horizon_days - 1
    first_asof = vol_lookback
    labels = bucket_labels_from_edges(edges_from_spot(1.0, cuts))  # placeholder labels by pct

    for i in range(first_asof, last_asof + 1, max(step, 1)):
        spot = float(values[i])
        if not math.isfinite(spot) or spot <= 0:
            continue
        window_vol = values[i - vol_lookback : i + 1]
        sigma = _sigma_daily(window_vol)
        if not math.isfinite(sigma) or sigma <= 0:
            continue
        future = values[i : i + horizon_days + 1]
        realized_max = float(np.max(future))
        edges = edges_from_spot(spot, cuts)
        hit_idx = _bucket_index(realized_max, edges)
        hit_label = bucket_labels_from_edges(edges)[hit_idx]
        # also one-hot via assign_buckets for consistency
        probs_one = assign_buckets(np.array([realized_max]), edges)
        row = {
            "asof": str(dates[i].date()),
            "pair": spec.pair,
            "spot": spot,
            "horizon_days": horizon_days,
            "vol_lookback": vol_lookback,
            "sigma_daily": sigma,
            "realized_max": realized_max,
            "realized_max_pct": (realized_max / spot - 1.0) * 100.0,
            "edge_0": edges[0],
            "edge_1": edges[1],
            "edge_2": edges[2],
            "edge_3": edges[3],
            "bucket_hit_idx": hit_idx,
            "bucket_hit": hit_label,
            "cut_0": cuts[0],
            "cut_1": cuts[1],
            "cut_2": cuts[2],
            "cut_3": cuts[3],
        }
        for lab, p in probs_one.items():
            # sanitize column names
            safe = lab.replace(" ", "_").replace(">=", "ge").replace("<", "lt")
            row[f"hit_{safe}"] = int(p >= 0.5)
        rows.append(row)

    df = pd.DataFrame(rows)
    meta = {
        "pair": spec.pair,
        "source": source,
        "notes": notes,
        "n_samples": len(df),
        "horizon_days": horizon_days,
        "vol_lookback": vol_lookback,
        "history_days": history_days,
        "step": step,
        "bucket_pct_cuts": list(cuts),
        "bucket_labels_template": labels,
        "history_start": str(dates[0].date()) if n else None,
        "history_end": str(dates[-1].date()) if n else None,
    }
    return df, meta


def export_peak_samples(
    pair: str | PairSpec = "USD/AUD",
    *,
    out_dir: str | Path = "output",
    horizon_days: int = 66,
    vol_lookback: int = 60,
    history_days: int = 1500,
    step: int = 5,
    bucket_pct_cuts: tuple[float, float, float, float] | None = None,
) -> tuple[Path, pd.DataFrame, dict]:
    """Build samples and write `output/peak_samples_{PAIR}.csv`."""
    spec = get_pair(pair) if isinstance(pair, str) else pair
    df, meta = build_peak_samples(
        spec,
        horizon_days=horizon_days,
        vol_lookback=vol_lookback,
        history_days=history_days,
        step=step,
        bucket_pct_cuts=bucket_pct_cuts,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe = spec.pair.replace("/", "")
    path = out / f"peak_samples_{safe}.csv"
    df.to_csv(path, index=False)
    meta_path = out / f"peak_samples_{safe}_meta.json"
    import json

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    meta["csv"] = str(path)
    meta["meta_json"] = str(meta_path)
    return path, df, meta
