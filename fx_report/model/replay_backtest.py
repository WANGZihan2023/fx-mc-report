"""Historical freeze replay backtest for the full pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fx_report.market.fetch_data import fetch_history_series
from fx_report.market.pairs import get_pair, resolve_pair_for_bullish
from fx_report.model.history_peaks import _bucket_index
from fx_report.model.monte_carlo import bucket_labels_from_edges
from fx_report.pipeline import run_pipeline


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    if "T" in text:
        text = text.split("T", 1)[0]
    if " " in text:
        text = text.split(" ", 1)[0]
    return date.fromisoformat(text)


def _uniform_baseline_brier(k: int) -> float:
    if k <= 0:
        return float("nan")
    p = 1.0 / float(k)
    return (1.0 - p) ** 2 + (k - 1) * (p**2)


@dataclass
class ReplayBacktestResult:
    pair: str
    n_rows: int
    table: pd.DataFrame
    summary: dict[str, Any]
    csv_path: Path
    json_path: Path


def run_replay_backtest(
    pair: str = "USD/AUD",
    *,
    bullish_currency: str | None = None,
    start_date: date | datetime | str,
    end_date: date | datetime | str,
    step_days: int = 7,
    out_dir: str | Path = "output",
    sims: int = 2_000,
    days: int = 66,
    seed: int = 42,
    lookback: int = 60,
    peak_engine: str = "path_max",
    variance_reduction: str = "none",
    jump_model: str = "merton",
    jump_compensate: bool = False,
    mode: str = "hybrid",
    max_news: int = 10,
    keep_templates: bool = False,
    template_policy: str = "off",
    no_news: bool = False,
    no_fulltext: bool = False,
    ai_research: bool = True,
    calibrated_params_path: str | Path | None = None,
    use_label_learned_strength: bool = False,
    max_dates: int | None = None,
    verbose: bool = True,
) -> ReplayBacktestResult:
    start = _as_date(start_date)
    end = _as_date(end_date)
    if end < start:
        raise ValueError("end_date must be >= start_date")
    if step_days <= 0:
        raise ValueError("step_days must be positive")

    display_spec = get_pair(pair)
    bullish = (bullish_currency or display_spec.base).strip().upper()
    analysis_spec = resolve_pair_for_bullish(display_spec, bullish)

    history_days = max(lookback * 6, 400, days * 3)
    future_end = end + timedelta(days=max(days * 3, 120))
    series, history_source, history_notes = fetch_history_series(
        analysis_spec,
        history_days=history_days + (future_end - start).days,
        end_date=future_end,
    )
    series = series.dropna().sort_index()
    series = series[~series.index.duplicated(keep="last")]

    desired_dates: list[date] = []
    cur = start
    while cur <= end:
        desired_dates.append(cur)
        cur += timedelta(days=step_days)

    available_dates = [ts.date() for ts in series.index]
    rows: list[dict[str, Any]] = []
    used_dates = 0
    for target in desired_dates:
        if max_dates is not None and used_dates >= max_dates:
            break
        as_of = max((d for d in available_dates if d <= target), default=None)
        if as_of is None:
            continue
        idx = max(i for i, ts in enumerate(series.index) if ts.date() == as_of)
        if idx < lookback or idx + days >= len(series):
            continue

        result = run_pipeline(
            display_spec.pair,
            sims=sims,
            days=days,
            seed=seed + used_dates,
            lookback=lookback,
            peak_engine=peak_engine,
            variance_reduction=variance_reduction,
            jump_model=jump_model,
            jump_compensate=jump_compensate,
            mode=mode,  # type: ignore[arg-type]
            max_news=max_news,
            keep_templates=keep_templates,
            template_policy=template_policy,  # type: ignore[arg-type]
            no_news=no_news,
            no_fulltext=no_fulltext,
            ai_research=ai_research,
            out_dir=None,
            verbose=False,
            bullish_currency=bullish,
            calibrated_params_path=calibrated_params_path,
            use_label_learned_strength=use_label_learned_strength,
            as_of_date=as_of,
        )

        future = series.iloc[idx : idx + days + 1].values.astype(float)
        realized_max = float(np.max(future))
        true_idx = _bucket_index(realized_max, result.edges)
        labels = bucket_labels_from_edges(result.edges)
        pred_labels = list(result.probs.keys())
        p = np.array(list(result.probs.values()), dtype=np.float64)
        p = np.clip(p, 1e-9, 1.0)
        p = p / p.sum()
        pred_idx = int(np.argmax(p))
        pred_bucket = pred_labels[pred_idx]
        true_bucket = labels[true_idx]
        y = np.zeros_like(p)
        if true_idx < len(y):
            y[true_idx] = 1.0
        brier = float(np.sum((p - y) ** 2))
        base_brier = _uniform_baseline_brier(len(p))
        skill_brier = float("nan") if not np.isfinite(base_brier) or base_brier <= 0 else 1.0 - (brier / base_brier)
        evidence_n = int((result.news_meta or {}).get("evidence_n", len(result.weighted)))
        hist_quality = str((result.news_meta or {}).get("historical_news_quality") or "limited")
        row: dict[str, Any] = {
            "as_of": as_of.isoformat(),
            "requested_as_of": target.isoformat(),
            "pair": result.pair,
            "bullish_currency": bullish,
            "spot": float(result.market.spot),
            "realized_max": realized_max,
            "realized_max_pct": (realized_max / float(result.market.spot) - 1.0) * 100.0,
            "pred_bucket": pred_bucket,
            "true_bucket": true_bucket,
            "pred_idx": pred_idx,
            "true_idx": true_idx,
            "argmax_hit": int(pred_idx == true_idx),
            "brier": brier,
            "skill_brier": skill_brier,
            "evidence_n": evidence_n,
            "historical_news_quality": hist_quality,
            "historical_news_note": (result.news_meta or {}).get("limitation") or "",
            "template_policy": template_policy,
            "evidence_quality": (result.news_meta or {}).get("evidence_quality"),
            "history_source": history_source,
            "history_notes": " | ".join(history_notes[:3]),
            "peak_engine": peak_engine,
            "jump_model": jump_model,
        }
        for i, lab in enumerate(pred_labels):
            safe_lab = lab.replace(" ", "_").replace(">=", "ge").replace("<", "lt")
            row[f"p_{safe_lab}"] = float(p[i])
        rows.append(row)
        used_dates += 1
        if verbose:
            print(
                f"Replay {as_of.isoformat()} → pred={pred_bucket} true={true_bucket} "
                f"brier={brier:.4f} evidence_n={evidence_n} news={hist_quality}"
            )

    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("没有可用回放日期：请扩大区间，或检查历史价格长度与 horizon/lookback。")

    summary = {
        "pair": pair,
        "analysis_pair": analysis_spec.pair,
        "bullish_currency": bullish,
        "n_rows": int(len(table)),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "step_days": int(step_days),
        "days": int(days),
        "lookback": int(lookback),
        "n_sims": int(sims),
        "peak_engine": peak_engine,
        "jump_model": jump_model,
        "variance_reduction": variance_reduction,
        "argmax_hit_rate": float(table["argmax_hit"].mean()),
        "mean_brier": float(table["brier"].mean()),
        "mean_skill_brier": float(table["skill_brier"].mean()),
        "historical_news_quality_counts": table["historical_news_quality"].value_counts().to_dict(),
        "note": (
            "价格历史按 as_of 真冻结；历史新闻仅使用可日期过滤来源（如 NewsAPI）与本地 inbox。"
            "若 quality=limited，表示该时点新闻证据并非完整历史信息集。"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe = analysis_spec.pair.replace("/", "")
    span = f"{start.isoformat()}_{end.isoformat()}".replace("-", "")
    csv_path = out / f"replay_backtest_{safe}_{span}.csv"
    json_path = out / f"replay_backtest_{safe}_{span}.json"
    table.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["csv"] = str(csv_path)
    summary["json"] = str(json_path)
    return ReplayBacktestResult(
        pair=analysis_spec.pair,
        n_rows=len(table),
        table=table,
        summary=summary,
        csv_path=csv_path,
        json_path=json_path,
    )
