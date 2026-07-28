"""
Pure UX helpers for Streamlit flows (testable without Streamlit runtime).

Password gate, relative-bucket bounds / -20 clamp healing, BB jump warnings.
"""

from __future__ import annotations

import os
from typing import Sequence

# Relative-cut number_input bounds (NOT the default cuts).
PCT_CUT_MIN = -20.0
PCT_CUT_MAX = 50.0

_DEFAULT_APP_PASSWORD = "uniocean"


def app_password_expected(
    *,
    environ: dict[str, str] | None = None,
    default: str = _DEFAULT_APP_PASSWORD,
) -> str:
    """Product shared gate: APP_PASSWORD / FX_REPORT_PASSWORD, else default uniocean."""
    env = environ if environ is not None else os.environ
    for key in ("APP_PASSWORD", "FX_REPORT_PASSWORD"):
        val = (env.get(key) or "").strip()
        if val:
            return val
    return default


def password_accepted(entered: str | None, expected: str) -> bool:
    """True only on exact match (empty ≠ default)."""
    return (entered or "") == expected


def clamp_pct_cut(v: float) -> float:
    return float(min(PCT_CUT_MAX, max(PCT_CUT_MIN, float(v))))


def pct_cuts_in_bounds(pcts: Sequence[float]) -> bool:
    return all(PCT_CUT_MIN <= float(p) <= PCT_CUT_MAX for p in pcts)


def abs_edges_to_pct_cuts(spot: float, edges: Sequence[float]) -> list[float]:
    if spot <= 0:
        raise ValueError("spot must be positive")
    return [(float(e) / spot - 1.0) * 100.0 for e in edges]


def seed_pct_widget_value(raw: float) -> tuple[float, bool]:
    """
    Seed a relative-% widget from stored pct.

    Returns (clamped_value, was_out_of_bounds).
    Out-of-bounds seeds are how the historic -20 clamp poison starts
    (abs edges from wrong quote → huge negative % → Streamlit min_value).
    """
    raw_f = float(raw)
    clamped = clamp_pct_cut(raw_f)
    oob = raw_f < PCT_CUT_MIN or raw_f > PCT_CUT_MAX
    return clamped, oob


def should_heal_floor_clamp(
    widget_vals: Sequence[float],
    *,
    seeded_from_oob: bool,
) -> bool:
    """
    Auto-reset to defaults only when widgets were seeded by clamping OOB %
    down to the floor — not when the user intentionally set all cuts to -20.
    """
    if not seeded_from_oob:
        return False
    if len(widget_vals) < 1:
        return False
    return all(abs(float(v) - PCT_CUT_MIN) < 1e-9 for v in widget_vals)


def bb_jump_compensate_warning(
    *,
    peak_engine: str,
    jump_compensate: bool,
) -> str | None:
    """UI warning: compensator is a no-op on brownian_bridge (no jumps)."""
    eng = (peak_engine or "path_max").strip().lower()
    if eng != "brownian_bridge":
        return None
    if not jump_compensate:
        return None
    return (
        "brownian_bridge 不含跳跃，jump_compensate 不会生效；"
        "需要补偿子时请改用 peak_engine=path_max。"
    )
