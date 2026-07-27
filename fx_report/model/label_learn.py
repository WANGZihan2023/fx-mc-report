"""
Stage 3 scaffold — learnable evidence strength from label_audit CSVs.

Minimal path (no ML stack):
  1. Load all output/label_audit_*.csv (+ optional in-memory frames)
  2. If ≥ MIN_LABELS rows have human_direction in {up,down,neutral},
     fit category → direction prior + strength multiplier
  3. Optionally apply multipliers to EvidenceItem.strength in the pipeline

When N is too small, ready=False and UI shows「标注不足，需至少 N 条」.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from fx_report.model.label_audit import (
    direction_to_int,
    load_label_audit,
    normalize_direction,
    project_output_dir,
)

# Minimum decisive human labels before learning is considered ready.
MIN_LABELS_FOR_LEARN = 20

STRENGTH_MULT_LO = 0.55
STRENGTH_MULT_HI = 1.45


@dataclass
class LabelLearnedParams:
    n_labeled: int = 0
    n_decisive: int = 0
    min_required: int = MIN_LABELS_FOR_LEARN
    ready: bool = False
    message: str = ""
    category_dir_prior: dict[str, float] = field(default_factory=dict)
    category_strength_mult: dict[str, float] = field(default_factory=dict)
    global_strength_mult: float = 1.0
    global_agree_rate: float | None = None
    as_of: str = ""
    source_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LabelLearnedParams":
        return cls(
            n_labeled=int(raw.get("n_labeled") or 0),
            n_decisive=int(raw.get("n_decisive") or 0),
            min_required=int(raw.get("min_required") or MIN_LABELS_FOR_LEARN),
            ready=bool(raw.get("ready")),
            message=str(raw.get("message") or ""),
            category_dir_prior={
                str(k): float(v) for k, v in (raw.get("category_dir_prior") or {}).items()
            },
            category_strength_mult={
                str(k): float(v)
                for k, v in (raw.get("category_strength_mult") or {}).items()
            },
            global_strength_mult=float(raw.get("global_strength_mult") or 1.0),
            global_agree_rate=(
                float(raw["global_agree_rate"])
                if raw.get("global_agree_rate") is not None
                else None
            ),
            as_of=str(raw.get("as_of") or ""),
            source_files=[str(x) for x in (raw.get("source_files") or [])],
        )


def label_learn_path(out_dir: Path | None = None) -> Path:
    root = out_dir or project_output_dir()
    return root / "label_learned_params.json"


def discover_label_audit_paths(out_dir: Path | None = None) -> list[Path]:
    root = out_dir or project_output_dir()
    if not root.is_dir():
        return []
    return sorted(root.glob("label_audit_*.csv"))


def _stack_audits(
    paths: Sequence[Path] | None = None,
    extra_frames: Sequence[pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    names: list[str] = []
    for p in paths or discover_label_audit_paths():
        try:
            df = load_label_audit(p)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        frames.append(df)
        names.append(p.name)
    for df in extra_frames or []:
        if df is not None and not getattr(df, "empty", True):
            frames.append(df)
            names.append("(session)")
    if not frames:
        return pd.DataFrame(), names
    return pd.concat(frames, ignore_index=True), names


def fit_label_learned_params(
    *,
    out_dir: Path | None = None,
    paths: Sequence[Path] | None = None,
    extra_frames: Sequence[pd.DataFrame] | None = None,
    min_labels: int = MIN_LABELS_FOR_LEARN,
) -> LabelLearnedParams:
    """
    Fit category→direction prior and strength multipliers from human labels.

    Strength multiplier heuristic (per category):
      mult = clip(0.7 + 0.6 * agree_rate_cat, LO, HI)
      if no decisive agree for that cat: 1.0 + 0.25 * |mean_dir|
    """
    df, names = _stack_audits(paths=paths, extra_frames=extra_frames)
    as_of = date.today().isoformat()
    empty = LabelLearnedParams(
        min_required=min_labels,
        as_of=as_of,
        source_files=names,
        message=f"标注不足，需至少 {min_labels} 条",
    )
    if df.empty:
        empty.message = f"尚无 label_audit CSV。标注不足，需至少 {min_labels} 条"
        return empty

    # Prefer human_category; fall back to model_category
    cats: list[str] = []
    dirs: list[int] = []
    agrees: list[str] = []
    for _, row in df.iterrows():
        hd = normalize_direction(row.get("human_direction", ""))
        d_int = direction_to_int(hd)
        if d_int is None:
            continue
        cat = str(row.get("human_category") or row.get("model_category") or "other").strip()
        if not cat:
            cat = "other"
        cats.append(cat)
        dirs.append(d_int)
        ag = str(row.get("agree") or "").strip().lower()
        agrees.append(ag)

    n_decisive = len(dirs)
    empty.n_labeled = n_decisive
    empty.n_decisive = n_decisive
    if n_decisive < min_labels:
        empty.message = f"标注不足，需至少 {min_labels} 条（当前 {n_decisive}）"
        return empty

    # Per-category aggregates
    by_cat: dict[str, list[int]] = {}
    agree_by_cat: dict[str, list[str]] = {}
    for c, d, a in zip(cats, dirs, agrees):
        by_cat.setdefault(c, []).append(d)
        agree_by_cat.setdefault(c, []).append(a)

    dir_prior: dict[str, float] = {}
    strength_mult: dict[str, float] = {}
    for c, ds in by_cat.items():
        mean_d = sum(ds) / max(len(ds), 1)
        dir_prior[c] = float(mean_d)
        ags = agree_by_cat.get(c) or []
        yes = sum(1 for a in ags if a == "yes")
        no = sum(1 for a in ags if a == "no")
        decisive = yes + no
        if decisive > 0:
            rate = yes / decisive
            mult = 0.7 + 0.6 * rate
        else:
            mult = 1.0 + 0.25 * abs(mean_d)
        strength_mult[c] = float(max(STRENGTH_MULT_LO, min(STRENGTH_MULT_HI, mult)))

    yes_g = sum(1 for a in agrees if a == "yes")
    no_g = sum(1 for a in agrees if a == "no")
    g_dec = yes_g + no_g
    g_rate = (yes_g / g_dec) if g_dec > 0 else None
    g_mult = 1.0
    if g_rate is not None:
        g_mult = float(max(STRENGTH_MULT_LO, min(STRENGTH_MULT_HI, 0.75 + 0.5 * g_rate)))

    return LabelLearnedParams(
        n_labeled=n_decisive,
        n_decisive=n_decisive,
        min_required=min_labels,
        ready=True,
        message=(
            f"已从 {n_decisive} 条标注拟合 "
            f"{len(strength_mult)} 个类别强度倍率"
            + (f"；全局同意率 {100 * g_rate:.0f}%" if g_rate is not None else "")
        ),
        category_dir_prior=dir_prior,
        category_strength_mult=strength_mult,
        global_strength_mult=g_mult,
        global_agree_rate=g_rate,
        as_of=as_of,
        source_files=names,
    )


def save_label_learned_params(
    params: LabelLearnedParams,
    path: Path | str | None = None,
) -> Path:
    path = Path(path) if path else label_learn_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(params.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_label_learned_params(path: Path | str | None = None) -> LabelLearnedParams | None:
    path = Path(path) if path else label_learn_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    return LabelLearnedParams.from_dict(raw)


def apply_label_learned_strength(
    evidence: Sequence[Any],
    params: LabelLearnedParams | None,
    *,
    blend_neutral_direction: bool = True,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Multiply EvidenceItem.strength by category / global multipliers.
    Optionally flip neutral items toward a strong category dir prior (|prior|≥0.4).
    Returns (new_evidence_list, apply_meta).
    """
    meta: dict[str, Any] = {
        "applied": False,
        "n_strength_scaled": 0,
        "n_dir_nudged": 0,
        "ready": bool(params and params.ready),
        "message": (params.message if params else "无学习参数"),
    }
    items = list(evidence)
    if not params or not params.ready or not items:
        return items, meta

    from fx_report.model.strength import label_strength
    from fx_report.model.weights import EvidenceItem

    out: list[Any] = []
    n_scaled = 0
    n_nudge = 0
    for e in items:
        if not isinstance(e, EvidenceItem):
            out.append(e)
            continue
        cat = (e.category or "other").strip() or "other"
        mult = float(params.category_strength_mult.get(cat, params.global_strength_mult))
        new_s = float(max(0.0, min(3.0, e.strength * mult)))
        note = e.note or ""
        if abs(mult - 1.0) > 1e-6:
            n_scaled += 1
            tag = f"label_learn×{mult:.2f}"
            if tag not in note:
                note = f"{tag}｜{note}" if note else tag
        direction = int(e.direction)
        if blend_neutral_direction and direction == 0:
            prior = float(params.category_dir_prior.get(cat, 0.0))
            if abs(prior) >= 0.4:
                direction = 1 if prior > 0 else -1
                n_nudge += 1
                nudge_tag = f"label_dir_prior={prior:+.2f}"
                if nudge_tag not in note:
                    note = f"{nudge_tag}｜{note}" if note else nudge_tag
        out.append(
            EvidenceItem(
                id=e.id,
                title=e.title,
                direction=direction,
                strength=new_s,
                freshness=e.freshness,
                unpriced=e.unpriced,
                category=e.category,
                note=note,
                strength_label=label_strength(new_s),
                strength_breakdown=dict(e.strength_breakdown or {}),
                source_tier=e.source_tier,
                surprise=e.surprise,
                scope=e.scope,
                statement_id=e.statement_id,
                url=e.url,
                is_prior=e.is_prior,
            )
        )
    meta["applied"] = True
    meta["n_strength_scaled"] = n_scaled
    meta["n_dir_nudged"] = n_nudge
    meta["message"] = params.message
    return out, meta
