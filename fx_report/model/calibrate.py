"""
Stage 1 — calibrate MC mixture params on historical peak samples (S=0 baseline).

Optimizes scenario priors (+ score_to_mu_a / score_to_sigma_b kept for later S≠0)
against realized bucket hits using Brier / log-loss. Uses reduced n_sims for speed.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from fx_report.model.monte_carlo import assign_buckets, run_mixture_monte_carlo
from fx_report.model.weights import ModelWeights, ScenarioSpec, default_scenarios, default_weights
from fx_report.market.pairs import edges_from_spot, get_pair


@dataclass
class CalibResult:
    pair: str
    params: dict[str, Any]
    loss_name: str
    loss: float
    n_samples: int
    n_sims: int
    n_iters: int
    baseline_loss: float


def load_peak_samples(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"asof", "spot", "sigma_daily", "realized_max", "horizon_days"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"peak samples missing columns: {sorted(missing)}")
    return df.dropna(subset=["spot", "sigma_daily", "realized_max"]).reset_index(drop=True)


def _cuts_from_row(row: pd.Series, fallback: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if all(c in row.index for c in ("cut_0", "cut_1", "cut_2", "cut_3")):
        return (float(row["cut_0"]), float(row["cut_1"]), float(row["cut_2"]), float(row["cut_3"]))
    return fallback


def _edges_from_row(row: pd.Series, cuts: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if all(c in row.index for c in ("edge_0", "edge_1", "edge_2", "edge_3")):
        return (float(row["edge_0"]), float(row["edge_1"]), float(row["edge_2"]), float(row["edge_3"]))
    return edges_from_spot(float(row["spot"]), cuts)


def _one_hot(realized_max: float, edges: Sequence[float]) -> np.ndarray:
    probs = assign_buckets(np.array([realized_max]), edges)
    return np.array(list(probs.values()), dtype=np.float64)


def scenarios_from_vec(
    x: np.ndarray,
    names: Sequence[str],
    narratives: Sequence[str],
) -> list[ScenarioSpec]:
    """
    x layout (len = 2 + 3*5 = 17 for 3 scenarios):
      score_to_mu_a, score_to_sigma_b,
      for each scenario: logit_weight, mu, sigma_mult, expected_jumps, jump_mean
      (jump_std fixed at prior to shrink dim)
    Softmax over logit_weight → mixture weights.
    """
    a, b = float(x[0]), float(x[1])
    n_sc = len(names)
    logits = []
    specs: list[ScenarioSpec] = []
    cursor = 2
    raw_parts: list[tuple[float, float, float, float, float]] = []
    for i in range(n_sc):
        logit_w = float(x[cursor])
        mu = float(x[cursor + 1])
        sm = float(x[cursor + 2])
        ej = float(x[cursor + 3])
        jm = float(x[cursor + 4])
        cursor += 5
        logits.append(logit_w)
        raw_parts.append((mu, sm, ej, jm, 0.004))  # jump_std placeholder filled below

    logits_arr = np.array(logits, dtype=np.float64)
    logits_arr = logits_arr - logits_arr.max()
    w = np.exp(logits_arr)
    w = w / w.sum()

    for i, name in enumerate(names):
        mu, sm, ej, jm, js = raw_parts[i]
        specs.append(
            ScenarioSpec(
                name=name,
                weight=float(w[i]),
                mu_annual=mu,
                sigma_mult=max(sm, 0.2),
                expected_jumps=max(ej, 0.0),
                jump_mean=jm,
                jump_std=max(js, 1e-4),
                narrative=narratives[i],
            )
        )
    # stash a,b on first narrative unused — caller reads separately
    specs[0].narrative = narratives[0]  # noqa: keep
    _ = a, b
    return specs


def vec_from_weights(w: ModelWeights) -> tuple[np.ndarray, list[str], list[str]]:
    names = [s.name for s in w.scenarios]
    narratives = [s.narrative for s in w.scenarios]
    parts: list[float] = [w.score_to_mu_a, w.score_to_sigma_b]
    for s in w.scenarios:
        # inverse sigmoid-ish logit from weight
        p = min(max(s.weight, 1e-4), 1 - 1e-4)
        logit = math.log(p / (1 - p))
        parts.extend([logit, s.mu_annual, s.sigma_mult, s.expected_jumps, s.jump_mean])
    return np.array(parts, dtype=np.float64), names, narratives


def pack_params(w: ModelWeights) -> dict[str, Any]:
    return {
        "score_to_mu_a": w.score_to_mu_a,
        "score_to_sigma_b": w.score_to_sigma_b,
        "evidence_logit_scale": w.evidence_logit_scale,
        "scenario_temperature": w.scenario_temperature,
        "max_scenario_shift": w.max_scenario_shift,
        "bucket_pct_cuts": list(w.bucket_pct_cuts),
        "use_relative_buckets": w.use_relative_buckets,
        "trading_days": w.trading_days,
        "vol_lookback_days": w.vol_lookback_days,
        "scenarios": [asdict(s) for s in w.scenarios],
        "peak_engine": getattr(w, "peak_engine", "path_max"),
        "calibration": {"baseline": "S=0", "version": 1},
    }


def apply_calibrated_params(base: ModelWeights, params: dict[str, Any]) -> ModelWeights:
    """Overlay calibrated JSON onto a ModelWeights instance (mutates and returns)."""
    if "score_to_mu_a" in params:
        base.score_to_mu_a = float(params["score_to_mu_a"])
    if "score_to_sigma_b" in params:
        base.score_to_sigma_b = float(params["score_to_sigma_b"])
    for key in ("evidence_logit_scale", "scenario_temperature", "max_scenario_shift"):
        if key in params:
            setattr(base, key, float(params[key]))
    if "bucket_pct_cuts" in params and params["bucket_pct_cuts"]:
        base.bucket_pct_cuts = tuple(float(x) for x in params["bucket_pct_cuts"])  # type: ignore[assignment]
        base.use_relative_buckets = bool(params.get("use_relative_buckets", True))
    if "trading_days" in params:
        base.trading_days = int(params["trading_days"])
    if "vol_lookback_days" in params:
        base.vol_lookback_days = int(params["vol_lookback_days"])
    sc = params.get("scenarios")
    if isinstance(sc, list) and sc:
        base.scenarios = [
            ScenarioSpec(
                name=str(s["name"]),
                weight=float(s["weight"]),
                mu_annual=float(s["mu_annual"]),
                sigma_mult=float(s["sigma_mult"]),
                expected_jumps=float(s["expected_jumps"]),
                jump_mean=float(s["jump_mean"]),
                jump_std=float(s["jump_std"]),
                narrative=str(s.get("narrative") or ""),
            )
            for s in sc
        ]
    if "peak_engine" in params:
        try:
            setattr(base, "peak_engine", params["peak_engine"])
        except Exception:
            pass
    return base


def load_calibrated_params(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "params" in data and isinstance(data["params"], dict):
        return data["params"]
    return data


# Bundled into the Docker image (output/ is gitignored / dockerignored).
BUNDLED_CALIBRATED_DIR = Path(__file__).resolve().parent.parent / "data" / "calibrated"


def _pair_safe(pair: str) -> str:
    return pair.replace("/", "")


def resolve_calibrated_params_path(
    pair: str,
    *,
    prefer_output: bool = True,
    output_dir: str | Path = "output",
) -> Path | None:
    """Prefer local `output/` (overnight refresh), else bundled deploy copy."""
    name = f"calibrated_params_{_pair_safe(pair)}.json"
    candidates = [Path(output_dir) / name, BUNDLED_CALIBRATED_DIR / name]
    if not prefer_output:
        candidates = list(reversed(candidates))
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_calib_oos_summary_path(
    pair: str,
    *,
    prefer_output: bool = True,
    output_dir: str | Path = "output",
) -> Path | None:
    """Prefer local `output/` OOS summary, else bundled deploy copy."""
    name = f"calib_oos_summary_{_pair_safe(pair)}.json"
    candidates = [Path(output_dir) / name, BUNDLED_CALIBRATED_DIR / name]
    if not prefer_output:
        candidates = list(reversed(candidates))
    for p in candidates:
        if p.exists():
            return p
    return None


def load_calib_oos_summary(pair: str, **kwargs: Any) -> dict[str, Any] | None:
    path = resolve_calib_oos_summary_path(pair, **kwargs)
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def discover_calib_oos_pairs(
    *,
    prefer_output: bool = True,
    output_dir: str | Path = "output",
) -> list[str]:
    """
    Pairs that have a calib_oos_summary_*.json (bundled and/or local output/).
    Returns display pairs like 'AUD/USD' when parseable, else safe token.
    """
    seen: dict[str, str] = {}  # safe -> display

    def _ingest(root: Path) -> None:
        if not root.is_dir():
            return
        for p in sorted(root.glob("calib_oos_summary_*.json")):
            safe = p.stem.replace("calib_oos_summary_", "")
            if not safe or safe in seen:
                continue
            # Prefer slash form from file contents when available
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                pair = str(data.get("pair") or "")
            except Exception:
                pair = ""
            if not pair and len(safe) == 6:
                pair = f"{safe[:3]}/{safe[3:]}"
            seen[safe] = pair or safe

    out_root = Path(output_dir)
    if prefer_output:
        _ingest(out_root)
        _ingest(BUNDLED_CALIBRATED_DIR)
    else:
        _ingest(BUNDLED_CALIBRATED_DIR)
        _ingest(out_root)
    return [seen[k] for k in sorted(seen.keys())]


def load_all_calib_oos_summaries(
    *,
    prefer_output: bool = True,
    output_dir: str | Path = "output",
) -> list[dict[str, Any]]:
    """Load OOS summaries for every pair with a summary JSON (typically 8)."""
    rows: list[dict[str, Any]] = []
    for pair in discover_calib_oos_pairs(
        prefer_output=prefer_output, output_dir=output_dir
    ):
        data = load_calib_oos_summary(
            pair, prefer_output=prefer_output, output_dir=output_dir
        )
        if not data:
            continue
        path = resolve_calib_oos_summary_path(
            pair, prefer_output=prefer_output, output_dir=output_dir
        )
        payload = dict(data)
        payload["_path"] = str(path) if path else ""
        rows.append(payload)
    return rows


def calib_oos_board_dataframe(
    *,
    prefer_output: bool = True,
    output_dir: str | Path = "output",
) -> pd.DataFrame:
    """
    Cross-pair trust table: holdout/train hit rate + Brier for UI「跨对质量」.
    """
    records: list[dict[str, Any]] = []
    for data in load_all_calib_oos_summaries(
        prefer_output=prefer_output, output_dir=output_dir
    ):
        hold = data.get("holdout") or {}
        train = data.get("train") or {}
        path = str(data.get("_path") or "")
        if "data/calibrated" in path.replace("\\", "/"):
            src = "bundled"
        elif path:
            src = "output"
        else:
            src = str(data.get("source") or "")
        records.append(
            {
                "pair": data.get("pair") or "",
                "holdout_hit": hold.get("hit_rate"),
                "holdout_brier": hold.get("brier"),
                "holdout_n": hold.get("n"),
                "train_hit": train.get("hit_rate"),
                "train_brier": train.get("brier"),
                "train_n": train.get("n"),
                "source": src,
            }
        )
    cols = [
        "pair",
        "holdout_hit",
        "holdout_brier",
        "holdout_n",
        "train_hit",
        "train_brier",
        "train_n",
        "source",
    ]
    if not records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(records)[cols]


def _predict_probs(
    spot: float,
    sigma_daily: float,
    horizon: int,
    scenarios: list[ScenarioSpec],
    edges: Sequence[float],
    *,
    n_sims: int,
    seed: int,
    score_to_mu_a: float,
    score_to_sigma_b: float,
    score: float = 0.0,
) -> np.ndarray:
    mu_shift = score_to_mu_a * score
    sigma_extra = 1.0 + score_to_sigma_b * abs(score)
    mc = run_mixture_monte_carlo(
        spot=spot,
        sigma_daily_base=sigma_daily,
        scenarios=scenarios,
        trading_days=horizon,
        n_sims=n_sims,
        seed=seed,
        bucket_edges=edges,
        mu_annual_shift=mu_shift,
        sigma_mult_extra=sigma_extra,
    )
    return np.array(list(mc.raw_probs.values()), dtype=np.float64)


def sample_loss(
    df: pd.DataFrame,
    scenarios: list[ScenarioSpec],
    *,
    score_to_mu_a: float,
    score_to_sigma_b: float,
    cuts_fallback: tuple[float, float, float, float],
    n_sims: int,
    seed: int,
    loss: str = "brier",
    max_rows: int | None = None,
) -> float:
    """Mean Brier or log-loss under S=0."""
    use = df
    if max_rows is not None and len(df) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=max_rows, replace=False)
        use = df.iloc[idx]
    total = 0.0
    n = 0
    for j, (_, row) in enumerate(use.iterrows()):
        cuts = _cuts_from_row(row, cuts_fallback)
        edges = _edges_from_row(row, cuts)
        y = _one_hot(float(row["realized_max"]), edges)
        p = _predict_probs(
            float(row["spot"]),
            float(row["sigma_daily"]),
            int(row.get("horizon_days", 66)),
            scenarios,
            edges,
            n_sims=n_sims,
            seed=seed + j,
            score_to_mu_a=score_to_mu_a,
            score_to_sigma_b=score_to_sigma_b,
            score=0.0,
        )
        p = np.clip(p, 1e-6, 1.0)
        p = p / p.sum()
        if loss == "logloss":
            total += float(-np.sum(y * np.log(p)))
        else:
            total += float(np.sum((p - y) ** 2))
        n += 1
    return total / max(n, 1)


def _bounds(n_sc: int) -> list[tuple[float, float]]:
    b: list[tuple[float, float]] = [
        (0.0, 0.05),  # score_to_mu_a
        (0.0, 0.15),  # score_to_sigma_b
    ]
    for _ in range(n_sc):
        b.extend(
            [
                (-3.0, 3.0),  # logit weight
                (-0.15, 0.20),  # mu
                (0.5, 2.0),  # sigma_mult
                (0.0, 2.5),  # expected_jumps
                (-0.02, 0.02),  # jump_mean
            ]
        )
    return b


def _clip_vec(x: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    out = x.copy()
    for i, (lo, hi) in enumerate(bounds):
        out[i] = min(max(out[i], lo), hi)
    return out


def calibrate_from_samples(
    df: pd.DataFrame,
    pair: str = "USD/AUD",
    *,
    n_sims: int = 2_000,
    n_iters: int = 40,
    seed: int = 42,
    loss: str = "brier",
    max_rows: int = 80,
    verbose: bool = True,
) -> CalibResult:
    """
    Random + coordinate search (no scipy). S=0 baseline.
    """
    spec = get_pair(pair)
    base = default_weights(spec)
    if not base.scenarios:
        base.scenarios = default_scenarios(spec.pair)
    x0, names, narratives = vec_from_weights(base)
    bounds = _bounds(len(names))
    cuts = base.bucket_pct_cuts

    def eval_x(x: np.ndarray) -> float:
        x = _clip_vec(x, bounds)
        sc = scenarios_from_vec(x, names, narratives)
        # restore jump_std from priors
        for i, prior in enumerate(base.scenarios):
            sc[i].jump_std = prior.jump_std
        return sample_loss(
            df,
            sc,
            score_to_mu_a=float(x[0]),
            score_to_sigma_b=float(x[1]),
            cuts_fallback=cuts,
            n_sims=n_sims,
            seed=seed,
            loss=loss,
            max_rows=max_rows,
        )

    baseline = eval_x(x0)
    best_x = x0.copy()
    best_loss = baseline
    if verbose:
        print(f"  baseline {loss}={baseline:.4f} (n_sims={n_sims}, max_rows={max_rows})")

    rng = np.random.default_rng(seed)
    # random search
    for it in range(max(n_iters // 2, 1)):
        cand = best_x.copy()
        # perturb a random subset of coords
        idxs = rng.choice(len(cand), size=min(5, len(cand)), replace=False)
        for i in idxs:
            lo, hi = bounds[i]
            cand[i] = float(rng.uniform(lo, hi))
        cand = _clip_vec(cand, bounds)
        val = eval_x(cand)
        if val < best_loss:
            best_loss = val
            best_x = cand
            if verbose:
                print(f"  iter {it+1}/{n_iters}: improved → {best_loss:.4f}")

    # coordinate descent polish
    for it in range(n_iters // 2, n_iters):
        i = int(rng.integers(0, len(best_x)))
        lo, hi = bounds[i]
        grid = np.linspace(lo, hi, 7)
        local_best = best_loss
        local_x = best_x.copy()
        for g in grid:
            cand = best_x.copy()
            cand[i] = float(g)
            val = eval_x(cand)
            if val < local_best:
                local_best = val
                local_x = cand
        if local_best < best_loss:
            best_loss = local_best
            best_x = local_x
            if verbose:
                print(f"  iter {it+1}/{n_iters}: coord → {best_loss:.4f}")

    sc_best = scenarios_from_vec(best_x, names, narratives)
    for i, prior in enumerate(base.scenarios):
        sc_best[i].jump_std = prior.jump_std
    calibrated = ModelWeights(
        n_sims=base.n_sims,
        seed=base.seed,
        trading_days=base.trading_days,
        vol_lookback_days=base.vol_lookback_days,
        use_relative_buckets=True,
        bucket_pct_cuts=cuts,
        score_to_mu_a=float(best_x[0]),
        score_to_sigma_b=float(best_x[1]),
        scenario_temperature=base.scenario_temperature,
        max_scenario_shift=base.max_scenario_shift,
        evidence_logit_scale=base.evidence_logit_scale,
        scenarios=sc_best,
        evidence=[],
    )
    params = pack_params(calibrated)
    return CalibResult(
        pair=spec.pair,
        params=params,
        loss_name=loss,
        loss=best_loss,
        n_samples=min(len(df), max_rows),
        n_sims=n_sims,
        n_iters=n_iters,
        baseline_loss=baseline,
    )


def save_calibrated_params(
    result: CalibResult,
    out_dir: str | Path = "output",
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe = result.pair.replace("/", "")
    path = out / f"calibrated_params_{safe}.json"
    payload = {
        "pair": result.pair,
        "loss_name": result.loss_name,
        "loss": result.loss,
        "baseline_loss": result.baseline_loss,
        "n_samples": result.n_samples,
        "n_sims": result.n_sims,
        "n_iters": result.n_iters,
        "params": result.params,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def calibrate_pair(
    pair: str = "USD/AUD",
    *,
    samples_path: str | Path | None = None,
    out_dir: str | Path = "output",
    n_sims: int = 2_000,
    n_iters: int = 40,
    seed: int = 42,
    loss: str = "brier",
    max_rows: int = 80,
    holdout_frac: float = 0.25,
    verbose: bool = True,
) -> tuple[Path, CalibResult]:
    safe = pair.replace("/", "")
    path = Path(samples_path) if samples_path else Path(out_dir) / f"peak_samples_{safe}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到峰值样本 {path}。请先运行：python run_cli.py build-peaks --pair {pair}"
        )
    df = load_peak_samples(path)
    if verbose:
        print(f"Loaded {len(df)} samples from {path}")

    # Chronological train / holdout for light OOS snippet
    if "asof" in df.columns:
        df_sorted = df.sort_values("asof").reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)
    n = len(df_sorted)
    if n >= 8 and holdout_frac > 0:
        cut = max(int(n * (1.0 - holdout_frac)), 1)
        if cut >= n:
            cut = n - 1
        train_df = df_sorted.iloc[:cut].reset_index(drop=True)
        hold_df = df_sorted.iloc[cut:].reset_index(drop=True)
    else:
        train_df, hold_df = df_sorted, df_sorted.iloc[0:0]

    result = calibrate_from_samples(
        train_df if len(train_df) else df,
        pair,
        n_sims=n_sims,
        n_iters=n_iters,
        seed=seed,
        loss=loss,
        max_rows=max_rows,
        verbose=verbose,
    )
    out = save_calibrated_params(result, out_dir=out_dir)

    # Write calib_oos_summary_{PAIR}.json (train vs holdout Brier/logloss)
    try:
        from fx_report.model.backtest import (
            eval_split_metrics,
            write_calib_oos_summary,
        )

        cal_w = default_weights(get_pair(pair))
        apply_calibrated_params(cal_w, result.params)
        oos_cap = min(max_rows, 40)
        train_m = eval_split_metrics(
            train_df if len(train_df) else df,
            cal_w,
            n_sims=n_sims,
            seed=seed,
            max_rows=oos_cap,
        )
        hold_m = eval_split_metrics(
            hold_df,
            cal_w,
            n_sims=n_sims,
            seed=seed + 10_000,
            max_rows=oos_cap,
        )
        oos_path = write_calib_oos_summary(
            pair,
            train_metrics=train_m,
            holdout_metrics=hold_m,
            out_dir=out_dir,
            extra={
                "source": "calibrate",
                "loss_name": result.loss_name,
                "calibrated_loss": result.loss,
                "baseline_loss": result.baseline_loss,
                "n_sims": n_sims,
                "params_path": str(out),
            },
        )
        if verbose:
            print(
                f"OOS summary → {oos_path}  "
                f"train_brier={train_m.get('brier', float('nan')):.4f}  "
                f"holdout_brier={hold_m.get('brier', float('nan')):.4f}"
            )
    except Exception as exc:
        if verbose:
            print(f"WARN: could not write calib_oos_summary: {exc}")

    if verbose:
        print(f"Saved {out}  ({result.loss_name} {result.baseline_loss:.4f} → {result.loss:.4f})")
    return out, result
