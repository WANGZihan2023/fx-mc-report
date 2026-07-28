"""
Light evidence drift monitoring (ECDA-style honesty).

Compare the current run's category / direction distribution against a
rolling baseline snapshot. Large total-variation (TV) shifts → Chinese
audit warnings. Does not invent evidence or change S.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from fx_report.model.weights import EvidenceItem

# Soft default: warn when category or direction TV ≥ this
DEFAULT_TV_WARN = 0.40
# Need at least this many non-prior items before comparing
DEFAULT_MIN_ITEMS = 3
# EMA blend when updating baseline after a run
DEFAULT_EMA_ALPHA = 0.25

_DIR_KEYS = ("up", "down", "neutral")


@dataclass
class DistSnapshot:
    n: int
    categories: dict[str, float]
    directions: dict[str, float]
    pair: str = ""
    updated_at: str = ""
    n_runs: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DistSnapshot":
        return cls(
            n=int(raw.get("n") or 0),
            categories={str(k): float(v) for k, v in dict(raw.get("categories") or {}).items()},
            directions={str(k): float(v) for k, v in dict(raw.get("directions") or {}).items()},
            pair=str(raw.get("pair") or ""),
            updated_at=str(raw.get("updated_at") or ""),
            n_runs=int(raw.get("n_runs") or 1),
        )


@dataclass
class DriftReport:
    tv_category: float
    tv_direction: float
    warn: bool
    warnings: list[str] = field(default_factory=list)
    current: DistSnapshot | None = None
    baseline: DistSnapshot | None = None
    baseline_path: str = ""
    baseline_updated: bool = False
    skipped_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tv_category": self.tv_category,
            "tv_direction": self.tv_direction,
            "warn": self.warn,
            "warnings": list(self.warnings),
            "current": self.current.to_dict() if self.current else None,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "baseline_path": self.baseline_path,
            "baseline_updated": self.baseline_updated,
            "skipped_reason": self.skipped_reason,
        }


def _dir_label(direction: int) -> str:
    if direction > 0:
        return "up"
    if direction < 0:
        return "down"
    return "neutral"


def _norm_cat(cat: str) -> str:
    c = (cat or "").strip().lower()
    return c if c else "other"


def distribution_from_evidence(
    items: Sequence[EvidenceItem],
    *,
    pair: str = "",
    include_priors: bool = False,
) -> DistSnapshot:
    cats: dict[str, float] = {}
    dirs: dict[str, float] = {k: 0.0 for k in _DIR_KEYS}
    n = 0
    for e in items:
        if e.is_prior and not include_priors:
            continue
        n += 1
        c = _norm_cat(e.category)
        cats[c] = cats.get(c, 0.0) + 1.0
        d = _dir_label(int(e.direction))
        dirs[d] = dirs.get(d, 0.0) + 1.0
    if n > 0:
        cats = {k: v / float(n) for k, v in cats.items()}
        dirs = {k: v / float(n) for k, v in dirs.items()}
    else:
        cats = {}
        dirs = {k: 0.0 for k in _DIR_KEYS}
    return DistSnapshot(
        n=n,
        categories=cats,
        directions=dirs,
        pair=pair,
        updated_at=datetime.now(timezone.utc).isoformat(),
        n_runs=1,
    )


def total_variation(a: dict[str, float], b: dict[str, float]) -> float:
    """Half L1 distance on aligned keys (range 0–1 for probability vectors)."""
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    return 0.5 * sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys)


def blend_snapshots(
    baseline: DistSnapshot,
    current: DistSnapshot,
    *,
    alpha: float = DEFAULT_EMA_ALPHA,
) -> DistSnapshot:
    """EMA update of category/direction shares."""
    a = max(0.0, min(1.0, float(alpha)))
    cats: dict[str, float] = {}
    for k in set(baseline.categories) | set(current.categories):
        cats[k] = (1.0 - a) * float(baseline.categories.get(k, 0.0)) + a * float(
            current.categories.get(k, 0.0)
        )
    dirs: dict[str, float] = {}
    for k in set(baseline.directions) | set(current.directions) | set(_DIR_KEYS):
        dirs[k] = (1.0 - a) * float(baseline.directions.get(k, 0.0)) + a * float(
            current.directions.get(k, 0.0)
        )
    return DistSnapshot(
        n=current.n,
        categories=cats,
        directions=dirs,
        pair=current.pair or baseline.pair,
        updated_at=datetime.now(timezone.utc).isoformat(),
        n_runs=int(baseline.n_runs) + 1,
    )


def baseline_path_for_pair(
    pair: str,
    *,
    out_dir: str | Path = "output",
) -> Path:
    safe = (pair or "PAIR").replace("/", "")
    return Path(out_dir) / f"evidence_drift_baseline_{safe}.json"


def load_baseline(path: str | Path) -> DistSnapshot | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        return DistSnapshot.from_dict(raw)
    except Exception:
        return None


def save_baseline(snap: DistSnapshot, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(snap.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _fmt_top(dist: dict[str, float], n: int = 3) -> str:
    items = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    if not items:
        return "∅"
    return ", ".join(f"{k}={v:.0%}" for k, v in items)


def check_evidence_drift(
    items: Sequence[EvidenceItem],
    *,
    pair: str = "",
    out_dir: str | Path = "output",
    tv_warn: float = DEFAULT_TV_WARN,
    min_items: int = DEFAULT_MIN_ITEMS,
    update_baseline: bool = True,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
    include_priors: bool = False,
) -> DriftReport:
    """
    Compare current evidence mix vs on-disk baseline.

    First run with enough items: seed baseline, no warn.
    Later runs: TV on category & direction; warn if either ≥ tv_warn.
    """
    current = distribution_from_evidence(
        items, pair=pair, include_priors=include_priors
    )
    path = baseline_path_for_pair(pair, out_dir=out_dir)
    path_s = str(path)

    if current.n < int(min_items):
        return DriftReport(
            tv_category=0.0,
            tv_direction=0.0,
            warn=False,
            current=current,
            baseline_path=path_s,
            skipped_reason=f"n={current.n}<min_items={min_items}",
        )

    baseline = load_baseline(path)
    if baseline is None or baseline.n < 1:
        if update_baseline:
            save_baseline(current, path)
        return DriftReport(
            tv_category=0.0,
            tv_direction=0.0,
            warn=False,
            current=current,
            baseline=current,
            baseline_path=path_s,
            baseline_updated=bool(update_baseline),
            skipped_reason="baseline_seeded",
        )

    tv_cat = total_variation(current.categories, baseline.categories)
    tv_dir = total_variation(current.directions, baseline.directions)
    warnings: list[str] = []
    warn = False
    thr = float(tv_warn)
    if tv_cat >= thr:
        warn = True
        warnings.append(
            f"证据类别分布漂移：TV={tv_cat:.2f}≥{thr:.2f} "
            f"（本次 {_fmt_top(current.categories)} vs 基线 {_fmt_top(baseline.categories)}）"
        )
    if tv_dir >= thr:
        warn = True
        warnings.append(
            f"证据方向分布漂移：TV={tv_dir:.2f}≥{thr:.2f} "
            f"（本次 {_fmt_top(current.directions)} vs 基线 {_fmt_top(baseline.directions)}）"
        )

    updated = False
    new_base = baseline
    if update_baseline:
        new_base = blend_snapshots(baseline, current, alpha=ema_alpha)
        save_baseline(new_base, path)
        updated = True

    return DriftReport(
        tv_category=tv_cat,
        tv_direction=tv_dir,
        warn=warn,
        warnings=warnings,
        current=current,
        baseline=new_base,
        baseline_path=path_s,
        baseline_updated=updated,
    )
