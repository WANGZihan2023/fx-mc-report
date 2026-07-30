"""
Pure UX helpers for Streamlit flows (testable without Streamlit runtime).

Password gate, relative-bucket bounds / -20 clamp healing, BB jump warnings,
start-setup required choices (no silent defaults).
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

from fx_report.ui.i18n import (
    CHOICE_PLACEHOLDERS,
    DEFAULT_LANG,
    PLACEHOLDER_ZH,
    format_missing_message,
    normalize_lang,
    start_field_label,
)

# Relative-cut number_input bounds (NOT the default cuts).
PCT_CUT_MIN = -20.0
PCT_CUT_MAX = 50.0

_DEFAULT_APP_PASSWORD = "uniocean"

# Placeholder label for select/radio when nothing is chosen yet (zh default).
START_CHOICE_PLACEHOLDER = PLACEHOLDER_ZH

# Stable order + Chinese labels for must-have start choices before Run.
# Prefer start_field_label(key, lang=…) for UI; this dict stays zh for tests / PDF.
START_REQUIRED_LABELS: dict[str, str] = {
    "pair": "货币对",
    "bullish_currency": "看涨货币",
    "peak_engine": "峰值引擎",
    "use_calibrated": "是否使用校准参数",
    "human_review": "不确定证据是否人工确认",
    "bucket_mode": "分档边界方式",
}

# 简洁（推荐）mode: only pair + bullish required in dialog; bucket at run.
START_SIMPLE_DIALOG_KEYS: tuple[str, ...] = ("pair", "bullish_currency")
START_SIMPLE_RUN_KEYS: tuple[str, ...] = ("pair", "bullish_currency", "bucket_mode")
START_EXPERT_DIALOG_KEYS: tuple[str, ...] = (
    "pair",
    "bullish_currency",
    "peak_engine",
    "use_calibrated",
    "human_review",
)

_PEAK_ENGINES = frozenset({"path_max", "brownian_bridge"})
_BUCKET_MODES = frozenset({"相对现价", "绝对价位"})


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


def is_unset_choice(value: object) -> bool:
    """True when the user has not made an explicit choice yet."""
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip()
        return (not s) or s in CHOICE_PLACEHOLDERS
    return False


def missing_start_choices(
    choices: Mapping[str, object],
    *,
    keys: Sequence[str] | None = None,
    setup_mode: str | None = None,
    include_bucket: bool = False,
    lang: str | None = None,
) -> list[str]:
    """
    Return localized labels for required start fields that are still unset.

    Expected keys (any subset; missing key counts as unset unless `keys` narrows):
      pair, bullish_currency, peak_engine, use_calibrated,
      human_review, bucket_mode

    ``setup_mode``: when ``keys`` is omitted, 简洁 mode only requires
    pair + bullish (+ bucket if ``include_bucket``); 专家 requires the
    full explicit set. Algo fields are auto-filled in 简洁 mode.

    Bool fields must be actual bool (not None). String fields must be
    non-empty and not the placeholder. peak_engine / bucket_mode must be
    one of the known options.

    ``lang`` defaults to Chinese for backward-compatible tests / CLI.
    """
    lang = normalize_lang(lang if lang is not None else DEFAULT_LANG)
    if keys is not None:
        check_keys = list(keys)
    elif setup_mode is not None:
        from fx_report.model.algo_recommend import start_keys_for_mode

        check_keys = list(
            start_keys_for_mode(setup_mode, include_bucket=include_bucket)
        )
    else:
        check_keys = list(START_REQUIRED_LABELS.keys())
    missing: list[str] = []
    for key in check_keys:
        label = start_field_label(key, lang=lang)
        if key not in choices:
            missing.append(label)
            continue
        val = choices[key]
        if key in ("use_calibrated", "human_review"):
            if not isinstance(val, bool):
                missing.append(label)
            continue
        if is_unset_choice(val):
            missing.append(label)
            continue
        if key == "peak_engine" and str(val).strip() not in _PEAK_ENGINES:
            missing.append(label)
            continue
        if key == "bucket_mode" and str(val).strip() not in _BUCKET_MODES:
            missing.append(label)
            continue
    return missing


def format_missing_start_message(
    missing_labels: Sequence[str],
    *,
    lang: str | None = None,
) -> str:
    """Popup copy listing what the user still needs to pick (localized)."""
    return format_missing_message(list(missing_labels), lang=lang)


def resolve_replay_ai_research(
    *,
    allow_historical_ai: bool = False,
    sidebar_ai_research: bool | None = None,
) -> bool:
    """Historical / freeze-replay cost policy for UI (and tests).

    Default = cheap: AI researcher + Tavily/Brave stay OFF regardless of the
    live sidebar toggle. Only an explicit expensive override may re-enable AI;
    if ``sidebar_ai_research`` is passed, override still requires it True.
    """
    if not allow_historical_ai:
        return False
    if sidebar_ai_research is None:
        return True
    return bool(sidebar_ai_research)


def cheap_historical_mode(
    *,
    as_of_date: object | None,
    allow_historical_ai: bool = False,
) -> bool:
    """True when a historical as_of run should force the quota-saving path."""
    return as_of_date is not None and not bool(allow_historical_ai)


def format_cheap_historical_caption(
    meta: Mapping[str, object] | None,
    *,
    cheap_historical: bool | None = None,
) -> str:
    """Chinese one-liner for UI / audit: cheap mode, AI, sources, cache hits."""
    m: Mapping[str, object] = meta or {}
    if cheap_historical is None:
        if "cheap_historical" in m:
            cheap = bool(m.get("cheap_historical"))
        elif m.get("historical_mode"):
            cheap = not bool(m.get("allow_historical_ai"))
        else:
            cheap = False
    else:
        cheap = bool(cheap_historical)

    if cheap:
        ai_bit = "AI强制关"
    else:
        ai_meta = m.get("ai_research")
        if isinstance(ai_meta, dict):
            ai_bit = "AI开" if ai_meta.get("enabled") else "AI关"
        elif m.get("allow_historical_ai"):
            ai_bit = "AI允许"
        else:
            ai_bit = "AI关"

    bits: list[str] = []
    gdelt_n = m.get("gdelt_hits")
    newsapi_n = m.get("newsapi_hits")
    inbox_n = m.get("inbox_dated_hits")
    if gdelt_n is not None:
        bits.append(f"GDELT={int(gdelt_n)}")  # type: ignore[arg-type]
    if newsapi_n is not None:
        bits.append(f"NewsAPI={int(newsapi_n)}")  # type: ignore[arg-type]
    if inbox_n is not None:
        bits.append(f"inbox={int(inbox_n)}")  # type: ignore[arg-type]
    providers = m.get("providers_used")
    if not bits and isinstance(providers, (list, tuple)) and providers:
        bits.append("来源=" + "+".join(str(p) for p in providers))

    cache_bits: list[str] = []
    if m.get("gdelt_from_cache"):
        cache_bits.append("GDELT缓存命中")
    if m.get("newsapi_from_cache"):
        cache_bits.append("NewsAPI缓存命中")
    cache_s = "｜".join(cache_bits) if cache_bits else "缓存未命中/未用"

    sources_s = "+".join(bits) if bits else "来源无命中"
    return (
        f"cheap_historical={'ON' if cheap else 'OFF'}｜{ai_bit}｜"
        f"{sources_s}｜{cache_s}"
    )


def step3_pool_size(max_news: int, *, historical: bool = False) -> int:
    """Headline pool for step3 relative to evidence cap (live can go higher)."""
    n = max(1, int(max_news))
    if historical:
        # Keep historical cheap/light: modest pool, still >= evidence cap.
        return max(n, min(40, max(25, n * 2)))
    # Live: raise with slider (UI up to 40/50); keep a floor of 30.
    return max(30, min(90, n * 3))


def refs_statement_cap(max_news: int | None = None) -> int:
    """Markdown References slice — align with raised live evidence caps."""
    if max_news is None:
        return 50
    return max(25, min(50, int(max_news) * 2))