"""
Historical peak-bucket backtest — auditable right/wrong table.

For each as-of window in peak_samples, run MC (S=0 or calibrated params),
record predicted bucket probs vs realized bucket hit, and export metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from fx_report.market.pairs import edges_from_spot, get_pair
from fx_report.model.calibrate import (
    _cuts_from_row,
    _edges_from_row,
    _one_hot,
    apply_calibrated_params,
    load_calibrated_params,
    load_peak_samples,
)
from fx_report.model.history_peaks import _bucket_index
from fx_report.model.monte_carlo import (
    bucket_labels_from_edges,
    run_mixture_monte_carlo,
)
from fx_report.model.weights import ModelWeights, default_scenarios, default_weights


@dataclass
class BacktestResult:
    pair: str
    n_rows: int
    n_sims: int
    hit_rate_argmax: float
    mean_brier: float
    mean_logloss: float
    peak_engine: str
    params_source: str
    table: pd.DataFrame
    summary: dict[str, Any]


def _resolve_weights(
    pair: str,
    *,
    calibrated_params_path: str | Path | None = None,
) -> tuple[ModelWeights, str]:
    spec = get_pair(pair)
    base = default_weights(spec)
    if not base.scenarios:
        base.scenarios = default_scenarios(spec.pair)
    source = "default"
    if calibrated_params_path:
        path = Path(calibrated_params_path)
        if path.exists():
            apply_calibrated_params(base, load_calibrated_params(path))
            source = str(path)
    return base, source


def _predict_probs(
    spot: float,
    sigma_daily: float,
    horizon: int,
    weights: ModelWeights,
    edges: Sequence[float],
    *,
    n_sims: int,
    seed: int,
    score: float = 0.0,
    peak_engine: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    mu_shift = weights.score_to_mu_a * score
    sigma_extra = 1.0 + weights.score_to_sigma_b * abs(score)
    engine = peak_engine or getattr(weights, "peak_engine", "path_max")
    mc = run_mixture_monte_carlo(
        spot=spot,
        sigma_daily_base=sigma_daily,
        scenarios=weights.scenarios,
        trading_days=horizon,
        n_sims=n_sims,
        seed=seed,
        bucket_edges=edges,
        mu_annual_shift=mu_shift,
        sigma_mult_extra=sigma_extra,
        peak_engine=engine,
    )
    return np.array(list(mc.raw_probs.values()), dtype=np.float64), list(mc.raw_probs.keys())


def _chrono_split(df: pd.DataFrame, holdout_frac: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/holdout split by asof when available."""
    use = df
    if "asof" in df.columns:
        use = df.sort_values("asof").reset_index(drop=True)
    n = len(use)
    if n < 8 or holdout_frac <= 0:
        return use, use.iloc[0:0]
    cut = max(int(n * (1.0 - holdout_frac)), 1)
    if cut >= n:
        cut = n - 1
    return use.iloc[:cut].reset_index(drop=True), use.iloc[cut:].reset_index(drop=True)


def eval_split_metrics(
    df: pd.DataFrame,
    weights: ModelWeights,
    *,
    n_sims: int,
    seed: int,
    peak_engine: str | None = None,
    max_rows: int | None = None,
) -> dict[str, float]:
    """Mean Brier / log-loss / argmax hit-rate on a dataframe slice."""
    if df is None or len(df) == 0:
        return {"n": 0, "brier": float("nan"), "logloss": float("nan"), "hit_rate": float("nan")}
    use = df
    if max_rows is not None and len(df) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=max_rows, replace=False)
        use = df.iloc[sorted(idx)]
    cuts_fb = weights.bucket_pct_cuts
    brier_sum = 0.0
    ll_sum = 0.0
    hits = 0
    n = 0
    for j, (_, row) in enumerate(use.iterrows()):
        cuts = _cuts_from_row(row, cuts_fb)
        edges = _edges_from_row(row, cuts)
        y = _one_hot(float(row["realized_max"]), edges)
        p, _labels = _predict_probs(
            float(row["spot"]),
            float(row["sigma_daily"]),
            int(row.get("horizon_days", weights.trading_days)),
            weights,
            edges,
            n_sims=n_sims,
            seed=seed + j,
            score=0.0,
            peak_engine=peak_engine,
        )
        p = np.clip(p, 1e-6, 1.0)
        p = p / p.sum()
        brier_sum += float(np.sum((p - y) ** 2))
        ll_sum += float(-np.sum(y * np.log(p)))
        if int(np.argmax(p)) == int(np.argmax(y)):
            hits += 1
        n += 1
    return {
        "n": float(n),
        "brier": brier_sum / max(n, 1),
        "logloss": ll_sum / max(n, 1),
        "hit_rate": hits / max(n, 1),
    }


def write_calib_oos_summary(
    pair: str,
    *,
    train_metrics: dict[str, float],
    holdout_metrics: dict[str, float],
    out_dir: str | Path = "output",
    extra: dict[str, Any] | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe = pair.replace("/", "")
    path = out / f"calib_oos_summary_{safe}.json"
    payload: dict[str, Any] = {
        "pair": pair,
        "split": "chronological_asof",
        "train": train_metrics,
        "holdout": holdout_metrics,
        "note": (
            "Holdout is the last ~25% of peak samples by asof. "
            "If holdout.n==0, split was skipped (too few rows)."
        ),
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_backtest(
    pair: str = "USD/AUD",
    *,
    samples_path: str | Path | None = None,
    calibrated_params_path: str | Path | None = None,
    out_dir: str | Path = "output",
    n_sims: int = 2_000,
    max_rows: int | None = None,
    seed: int = 42,
    peak_engine: str | None = None,
    holdout_frac: float = 0.25,
    write_oos: bool = True,
    verbose: bool = True,
) -> BacktestResult:
    """
    Backtest MC bucket forecasts on historical peak samples (S=0).

    Writes:
      - output/backtest_{PAIR}.csv
      - output/backtest_{PAIR}_summary.json
      - output/calib_oos_summary_{PAIR}.json (train vs holdout metrics)
    """
    safe = pair.replace("/", "")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    path = Path(samples_path) if samples_path else out / f"peak_samples_{safe}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到峰值样本 {path}。请先运行：python run_cli.py build-peaks --pair {pair}"
        )
    df = load_peak_samples(path)
    if verbose:
        print(f"Loaded {len(df)} samples from {path}")

    weights, params_source = _resolve_weights(pair, calibrated_params_path=calibrated_params_path)
    # Prefer output/ then bundled fx_report/data/calibrated/ if not given
    if params_source == "default" and calibrated_params_path is None:
        from fx_report.model.calibrate import resolve_calibrated_params_path

        auto_cal = resolve_calibrated_params_path(pair, output_dir=out)
        if auto_cal is not None:
            weights, params_source = _resolve_weights(pair, calibrated_params_path=auto_cal)

    engine = peak_engine or getattr(weights, "peak_engine", "path_max")
    cuts_fb = weights.bucket_pct_cuts

    use = df
    if max_rows is not None and len(df) > max_rows:
        # Prefer evenly spaced chronological subsample for auditability
        if "asof" in df.columns:
            df_sorted = df.sort_values("asof").reset_index(drop=True)
        else:
            df_sorted = df.reset_index(drop=True)
        idx = np.linspace(0, len(df_sorted) - 1, num=max_rows, dtype=int)
        idx = np.unique(idx)
        use = df_sorted.iloc[idx].reset_index(drop=True)
        if verbose:
            print(f"Subsampled {len(use)} / {len(df)} rows")

    rows: list[dict[str, Any]] = []
    brier_sum = 0.0
    ll_sum = 0.0
    hits = 0

    for j, (_, row) in enumerate(use.iterrows()):
        cuts = _cuts_from_row(row, cuts_fb)
        edges = _edges_from_row(row, cuts)
        labels = bucket_labels_from_edges(edges)
        realized_max = float(row["realized_max"])
        true_idx = _bucket_index(realized_max, edges)
        true_bucket = labels[true_idx]
        y = _one_hot(realized_max, edges)

        p, pred_labels = _predict_probs(
            float(row["spot"]),
            float(row["sigma_daily"]),
            int(row.get("horizon_days", weights.trading_days)),
            weights,
            edges,
            n_sims=n_sims,
            seed=seed + j,
            score=0.0,
            peak_engine=engine,
        )
        p = np.clip(p, 1e-6, 1.0)
        p = p / p.sum()
        pred_idx = int(np.argmax(p))
        pred_bucket = pred_labels[pred_idx] if pred_idx < len(pred_labels) else labels[pred_idx]
        hit = int(pred_idx == true_idx)
        brier = float(np.sum((p - y) ** 2))
        logloss = float(-np.sum(y * np.log(p)))
        brier_sum += brier
        ll_sum += logloss
        hits += hit

        rec: dict[str, Any] = {
            "asof": row.get("asof"),
            "pair": row.get("pair", pair),
            "spot": float(row["spot"]),
            "sigma_daily": float(row["sigma_daily"]),
            "horizon_days": int(row.get("horizon_days", weights.trading_days)),
            "realized_max": realized_max,
            "pred_bucket": pred_bucket,
            "true_bucket": true_bucket,
            "pred_idx": pred_idx,
            "true_idx": true_idx,
            "hit": hit,
            "brier": brier,
            "logloss": logloss,
            "p_argmax": float(p[pred_idx]),
            "peak_engine": engine,
        }
        for i, lab in enumerate(pred_labels):
            safe_lab = lab.replace(" ", "_").replace(">=", "ge").replace("<", "lt")
            rec[f"p_{safe_lab}"] = float(p[i])
        rows.append(rec)

    table = pd.DataFrame(rows)
    n = len(table)
    mean_brier = brier_sum / max(n, 1)
    mean_ll = ll_sum / max(n, 1)
    hit_rate = hits / max(n, 1)

    # OOS: chronological split on full sample (not just subsample), metrics with same n_sims
    train_df, hold_df = _chrono_split(df, holdout_frac=holdout_frac)
    # Cap OOS eval cost
    oos_cap = min(max_rows or 40, 40)
    train_m = eval_split_metrics(
        train_df, weights, n_sims=n_sims, seed=seed, peak_engine=engine, max_rows=oos_cap
    )
    hold_m = eval_split_metrics(
        hold_df, weights, n_sims=n_sims, seed=seed + 10_000, peak_engine=engine, max_rows=oos_cap
    )

    summary: dict[str, Any] = {
        "pair": pair,
        "n_rows": n,
        "n_samples_available": len(df),
        "n_sims": n_sims,
        "hit_rate_argmax": hit_rate,
        "mean_brier": mean_brier,
        "mean_logloss": mean_ll,
        "peak_engine": engine,
        "params_source": params_source,
        "samples_path": str(path),
        "train_oos": train_m,
        "holdout_oos": hold_m,
        "note": (
            "S=0 backtest on historical peaks. hit=1 when argmax predicted bucket "
            "matches realized path-max bucket. OOS = last ~25% by asof."
        ),
    }

    csv_path = out / f"backtest_{safe}.csv"
    table.to_csv(csv_path, index=False)
    summary_path = out / f"backtest_{safe}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["csv"] = str(csv_path)
    summary["summary_json"] = str(summary_path)

    if write_oos:
        oos_path = write_calib_oos_summary(
            pair,
            train_metrics=train_m,
            holdout_metrics=hold_m,
            out_dir=out,
            extra={
                "source": "backtest",
                "n_sims": n_sims,
                "peak_engine": engine,
                "params_source": params_source,
            },
        )
        summary["calib_oos_summary"] = str(oos_path)

    if verbose:
        print(
            f"Backtest {pair}: hit_rate={hit_rate:.1%}  "
            f"brier={mean_brier:.4f}  logloss={mean_ll:.4f}  n={n}"
        )
        print(f"Wrote {csv_path}")
        print(f"Wrote {summary_path}")

    return BacktestResult(
        pair=pair,
        n_rows=n,
        n_sims=n_sims,
        hit_rate_argmax=hit_rate,
        mean_brier=mean_brier,
        mean_logloss=mean_ll,
        peak_engine=engine,
        params_source=params_source,
        table=table,
        summary=summary,
    )


def evidence_to_label_audit(
    evidence_rows: Sequence[dict[str, Any]] | Sequence[Any],
) -> pd.DataFrame:
    """
    Build label_audit CSV template from current-run evidence.

    Columns: statement_id, title, url, model_category, model_direction,
             human_direction, human_category, agree
    Human columns left empty for later labeling.
    Direction vocab: up / down / neutral (see fx_report.model.label_audit).
    """
    from fx_report.model.label_audit import evidence_rows_to_audit_df

    return evidence_rows_to_audit_df(evidence_rows)


def compare_peak_engines(
    spot: float,
    sigma_daily: float,
    weights: ModelWeights,
    edges: Sequence[float],
    *,
    n_sims: int = 8_000,
    seed: int = 42,
    score: float | None = None,
) -> dict[str, Any]:
    """
    Run MC twice (path_max vs brownian_bridge) with same spot/buckets/scenarios.
    Returns side-by-side probs and deltas. Uses reduced n_sims by default.
    """
    from fx_report.model.weights import apply_evidence_to_scenarios, evidence_score

    if score is None:
        score = evidence_score(weights.evidence) if weights.evidence else 0.0
    mu_shift = weights.score_to_mu_a * score
    sigma_extra = 1.0 + weights.score_to_sigma_b * abs(score)
    scenarios = apply_evidence_to_scenarios(
        weights.scenarios,
        score,
        logit_scale=weights.evidence_logit_scale,
        temperature=weights.scenario_temperature,
        max_shift=weights.max_scenario_shift,
    )

    results: dict[str, dict[str, float]] = {}
    for engine in ("path_max", "brownian_bridge"):
        mc = run_mixture_monte_carlo(
            spot=spot,
            sigma_daily_base=sigma_daily,
            scenarios=scenarios,
            trading_days=weights.trading_days,
            n_sims=n_sims,
            seed=seed,
            bucket_edges=edges,
            mu_annual_shift=mu_shift,
            sigma_mult_extra=sigma_extra,
            peak_engine=engine,
        )
        results[engine] = dict(mc.raw_probs)

    labels = list(results["path_max"].keys())
    # Align labels if brownian_bridge somehow differs (shouldn't)
    for lab in results["brownian_bridge"]:
        if lab not in labels:
            labels.append(lab)

    delta = {
        lab: float(results["brownian_bridge"].get(lab, 0.0) - results["path_max"].get(lab, 0.0))
        for lab in labels
    }
    table = pd.DataFrame(
        {
            "bucket": labels,
            "path_max": [results["path_max"].get(l, 0.0) for l in labels],
            "brownian_bridge": [results["brownian_bridge"].get(l, 0.0) for l in labels],
            "delta_bb_minus_pm": [delta[l] for l in labels],
        }
    )
    return {
        "n_sims": n_sims,
        "score_S": score,
        "mu_annual_shift": mu_shift,
        "sigma_mult_extra": sigma_extra,
        "path_max": results["path_max"],
        "brownian_bridge": results["brownian_bridge"],
        "delta": delta,
        "table": table,
        "note": "Compare mode uses reduced n_sims; not a full news re-run.",
    }
