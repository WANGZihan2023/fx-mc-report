"""Aggregate replay backtest outputs into a concise dashboard table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if out != out:
            return None
        return out
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _quality_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("historical_news_quality") or "unknown").strip() or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def summarize_replay_json(json_path: str | Path) -> dict[str, Any]:
    path = Path(json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        rows = []

    evidence = [_safe_int(row.get("evidence_n")) for row in rows]
    quality_counts = _quality_counts(rows)
    hist_working = any(
        str(row.get("historical_news_quality") or "") == "date_filtered"
        and _safe_int(row.get("evidence_n")) > 0
        for row in rows
    )
    start = str(summary.get("start_date") or "")
    end = str(summary.get("end_date") or "")
    return {
        "pair": str(summary.get("pair") or ""),
        "analysis_pair": str(summary.get("analysis_pair") or summary.get("pair") or ""),
        "bullish_currency": str(summary.get("bullish_currency") or ""),
        "window": f"{start} -> {end}" if start or end else "",
        "start_date": start,
        "end_date": end,
        "step_days": _safe_int(summary.get("step_days")),
        "n_rows": _safe_int(summary.get("n_rows")),
        "argmax_hit_rate": _safe_float(summary.get("argmax_hit_rate")),
        "mean_brier": _safe_float(summary.get("mean_brier")),
        "mean_skill_brier": _safe_float(summary.get("mean_skill_brier")),
        "evidence_mean": (sum(evidence) / len(evidence)) if evidence else 0.0,
        "evidence_max": max(evidence) if evidence else 0,
        "date_filtered_count": quality_counts.get("date_filtered", 0),
        "limited_count": quality_counts.get("limited", 0),
        "unknown_count": sum(v for k, v in quality_counts.items() if k not in {"date_filtered", "limited"}),
        "historical_news_working": "yes" if hist_working else "no",
        "historical_news_quality_counts": quality_counts,
        "csv_path": str(summary.get("csv") or path.with_suffix(".csv")),
        "json_path": str(path),
        "generated_at": str(summary.get("generated_at") or ""),
        "note": str(summary.get("note") or ""),
    }


def replay_summary_dataframe(out_dir: str | Path = "output") -> pd.DataFrame:
    root = Path(out_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("replay_backtest_*.json")):
        try:
            rows.append(summarize_replay_json(path))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    sort_cols = [c for c in ("pair", "start_date", "end_date", "generated_at") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[True, True, True, False][: len(sort_cols)])
    return df.reset_index(drop=True)
