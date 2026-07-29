"""
Auto-recommend MC / evidence algorithm settings for a currency pair.

Priority (highest first):
  1. Calibrated JSON for the pair (peak_engine / jump_model /
     recommended_variance_reduction when present)
  2. output/engine_compare/summary.json overall winner for that pair
  3. Product defaults: path_max + merton + antithetic, cluster jaccard,
     use_calibrated if file exists, human_review on

Pure helpers — no Streamlit dependency. See docs/algo_recommend.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from fx_report.model.calibrate import (
    load_calibrated_params,
    resolve_calibrated_params_path,
)
from fx_report.model.replay_engine_compare import ENGINE_COMBOS

# Product defaults when no calib / engine_compare signal.
DEFAULT_PEAK_ENGINE = "path_max"
DEFAULT_JUMP_MODEL = "merton"
DEFAULT_VARIANCE_REDUCTION = "antithetic"
DEFAULT_CLUSTER_METHOD = "jaccard"
DEFAULT_HUMAN_REVIEW = True

_PEAK_ENGINES = frozenset({"path_max", "brownian_bridge"})
_JUMP_MODELS = frozenset({"merton", "none"})
_VR_OPTS = frozenset({"none", "antithetic"})
_CLUSTER_OPTS = frozenset({"jaccard", "tfidf"})

SOURCE_CALIBRATED = "calibrated"
SOURCE_ENGINE_COMPARE = "engine_compare"
SOURCE_PRODUCT_DEFAULT = "product_default"

DEFAULT_ENGINE_COMPARE_SUMMARY = Path("output/engine_compare/summary.json")


@dataclass(frozen=True)
class AlgoRecommendation:
    peak_engine: str
    jump_model: str
    variance_reduction: str
    cluster_method: str
    use_calibrated: bool
    human_review: bool
    source: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reasons"] = list(self.reasons)
        return d

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> AlgoRecommendation | None:
        if not isinstance(raw, Mapping):
            return None
        try:
            reasons = raw.get("reasons") or ()
            if isinstance(reasons, str):
                reasons = (reasons,)
            return cls(
                peak_engine=str(raw.get("peak_engine") or DEFAULT_PEAK_ENGINE),
                jump_model=str(raw.get("jump_model") or DEFAULT_JUMP_MODEL),
                variance_reduction=str(
                    raw.get("variance_reduction") or DEFAULT_VARIANCE_REDUCTION
                ),
                cluster_method=str(
                    raw.get("cluster_method") or DEFAULT_CLUSTER_METHOD
                ),
                use_calibrated=bool(raw.get("use_calibrated")),
                human_review=bool(raw.get("human_review", DEFAULT_HUMAN_REVIEW)),
                source=str(raw.get("source") or SOURCE_PRODUCT_DEFAULT),
                reasons=tuple(str(x) for x in reasons),
            )
        except Exception:
            return None


def _norm_pair(pair: str) -> str:
    return (pair or "").strip().upper().replace(" ", "")


def _safe_engine(v: Any, fallback: str) -> str:
    s = str(v or "").strip().lower()
    return s if s in _PEAK_ENGINES else fallback


def _safe_jump(v: Any, fallback: str) -> str:
    s = str(v or "").strip().lower()
    return s if s in _JUMP_MODELS else fallback


def _safe_vr(v: Any, fallback: str) -> str:
    s = str(v or "").strip().lower()
    return s if s in _VR_OPTS else fallback


def _safe_cluster(v: Any, fallback: str) -> str:
    s = str(v or "").strip().lower()
    return s if s in _CLUSTER_OPTS else fallback


def _extract_calib_algo_fields(
    params: Mapping[str, Any],
) -> dict[str, str]:
    """Pull peak_engine / jump_model / VR from Stage-1 params dict."""
    out: dict[str, str] = {}
    if "peak_engine" in params and params.get("peak_engine") is not None:
        eng = _safe_engine(params.get("peak_engine"), "")
        if eng:
            out["peak_engine"] = eng
    if "jump_model" in params and params.get("jump_model") is not None:
        jm = _safe_jump(params.get("jump_model"), "")
        if jm:
            out["jump_model"] = jm
    vr = params.get("recommended_variance_reduction")
    cal_blob = params.get("calibration")
    if not vr and isinstance(cal_blob, Mapping):
        vr = cal_blob.get("recommended_variance_reduction")
    if vr is not None:
        vr_s = _safe_vr(vr, "")
        if vr_s:
            out["variance_reduction"] = vr_s
    return out


def load_engine_compare_summary(
    path: str | Path | None = None,
) -> dict[str, Any] | None:
    p = Path(path) if path is not None else DEFAULT_ENGINE_COMPARE_SUMMARY
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def overall_winner_from_summary(
    summary: Mapping[str, Any],
    *,
    pair: str | None = None,
) -> str | None:
    """
    Resolve engine-compare winner label (A/C/…).

    Prefers top-level overall_winner / winner when present and valid;
    else majority vote over row winners (ties → None).
    If ``pair`` is given, summary.pair must match (normalized).
    """
    if pair is not None:
        sp = _norm_pair(str(summary.get("pair") or ""))
        if not sp or sp != _norm_pair(pair):
            return None

    for key in ("overall_winner", "winner"):
        top = str(summary.get(key) or "").strip().upper()
        if top in ENGINE_COMBOS:
            return top

    rows = summary.get("rows") or []
    if not isinstance(rows, list):
        return None
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        w = str(row.get("winner") or "").strip().upper()
        if w in ENGINE_COMBOS:
            counts[w] = counts.get(w, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    leaders = [k for k, v in counts.items() if v == best]
    if len(leaders) != 1:
        return None
    return leaders[0]


def recommend_algorithms(
    pair: str,
    *,
    calibrated_path: str | Path | None = None,
    engine_compare_path: str | Path | None = None,
    prefer_output: bool = True,
    output_dir: str | Path = "output",
) -> AlgoRecommendation:
    """
    Recommend algorithm settings for ``pair`` (display form e.g. USD/AUD).

    ``calibrated_path`` / ``engine_compare_path`` override discovery for tests.
    """
    pair_s = (pair or "").strip()
    reasons: list[str] = []

    calib_path: Path | None
    if calibrated_path is not None:
        calib_path = Path(calibrated_path)
        if not calib_path.exists():
            calib_path = None
    else:
        calib_path = resolve_calibrated_params_path(
            pair_s,
            prefer_output=prefer_output,
            output_dir=output_dir,
        )

    use_calibrated = calib_path is not None
    if use_calibrated:
        reasons.append(
            f"校准参数文件存在（`{calib_path.name}`），默认启用 Stage-1 校准。"
        )
    else:
        reasons.append("未找到该货币对校准 JSON，使用默认先验参数。")

    # Always: cluster + HITL product defaults (not overridden by compare).
    cluster_method = DEFAULT_CLUSTER_METHOD
    human_review = DEFAULT_HUMAN_REVIEW
    reasons.append("事件聚类默认 Jaccard（轻量、无需 sklearn）。")
    reasons.append("不确定证据默认需要人工确认（可改专家设置关闭）。")

    # --- Priority 1: calibrated algo fields ---
    calib_fields: dict[str, str] = {}
    if calib_path is not None:
        try:
            params = load_calibrated_params(calib_path)
            if isinstance(params, dict):
                calib_fields = _extract_calib_algo_fields(params)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            calib_fields = {}

    if calib_fields:
        peak = calib_fields.get("peak_engine", DEFAULT_PEAK_ENGINE)
        jump = calib_fields.get("jump_model", DEFAULT_JUMP_MODEL)
        vr = calib_fields.get("variance_reduction", DEFAULT_VARIANCE_REDUCTION)
        reasons.insert(
            0,
            "优先采用校准 JSON 中的算法字段"
            + (
                f"：peak_engine=`{peak}`"
                if "peak_engine" in calib_fields
                else ""
            )
            + (
                f"，jump_model=`{jump}`"
                if "jump_model" in calib_fields
                else ""
            )
            + (
                f"，variance_reduction=`{vr}`"
                if "variance_reduction" in calib_fields
                else ""
            )
            + "。",
        )
        if "variance_reduction" not in calib_fields:
            reasons.append(
                f"校准 JSON 未写 recommended_variance_reduction，"
                f"方差缩减用产品默认 `{DEFAULT_VARIANCE_REDUCTION}`。"
            )
        return AlgoRecommendation(
            peak_engine=_safe_engine(peak, DEFAULT_PEAK_ENGINE),
            jump_model=_safe_jump(jump, DEFAULT_JUMP_MODEL),
            variance_reduction=_safe_vr(vr, DEFAULT_VARIANCE_REDUCTION),
            cluster_method=cluster_method,
            use_calibrated=use_calibrated,
            human_review=human_review,
            source=SOURCE_CALIBRATED,
            reasons=tuple(reasons),
        )

    # --- Priority 2: engine_compare winner ---
    summary = load_engine_compare_summary(engine_compare_path)
    winner = overall_winner_from_summary(summary, pair=pair_s) if summary else None
    if winner and winner in ENGINE_COMBOS:
        combo = ENGINE_COMBOS[winner]
        peak = _safe_engine(combo.get("peak_engine"), DEFAULT_PEAK_ENGINE)
        jump = _safe_jump(combo.get("jump_model"), DEFAULT_JUMP_MODEL)
        vr = _safe_vr(combo.get("variance_reduction"), DEFAULT_VARIANCE_REDUCTION)
        reasons.insert(
            0,
            f"校准 JSON 无算法字段；采用 engine_compare 胜出组合 "
            f"`{winner}`（peak_engine=`{peak}`，jump_model=`{jump}`，"
            f"variance_reduction=`{vr}`）。",
        )
        return AlgoRecommendation(
            peak_engine=peak,
            jump_model=jump,
            variance_reduction=vr,
            cluster_method=cluster_method,
            use_calibrated=use_calibrated,
            human_review=human_review,
            source=SOURCE_ENGINE_COMPARE,
            reasons=tuple(reasons),
        )

    # --- Priority 3: product defaults ---
    reasons.insert(
        0,
        "无校准算法字段且无匹配的 engine_compare 胜者，"
        f"使用产品默认：`{DEFAULT_PEAK_ENGINE}` + `{DEFAULT_JUMP_MODEL}` + "
        f"`{DEFAULT_VARIANCE_REDUCTION}`。",
    )
    return AlgoRecommendation(
        peak_engine=DEFAULT_PEAK_ENGINE,
        jump_model=DEFAULT_JUMP_MODEL,
        variance_reduction=DEFAULT_VARIANCE_REDUCTION,
        cluster_method=cluster_method,
        use_calibrated=use_calibrated,
        human_review=human_review,
        source=SOURCE_PRODUCT_DEFAULT,
        reasons=tuple(reasons),
    )


def format_recommend_audit_zh(rec: AlgoRecommendation | Mapping[str, Any] | None) -> str:
    """Chinese audit block: 「本次算法由系统推荐」+ reasons list."""
    if isinstance(rec, Mapping):
        rec = AlgoRecommendation.from_dict(rec)
    if rec is None:
        return ""
    source_zh = {
        SOURCE_CALIBRATED: "校准 JSON",
        SOURCE_ENGINE_COMPARE: "引擎对比 summary",
        SOURCE_PRODUCT_DEFAULT: "产品默认",
    }.get(rec.source, rec.source)
    lines = [
        "**本次算法由系统推荐**",
        f"· 来源：{source_zh}（`{rec.source}`）",
        f"· peak_engine=`{rec.peak_engine}`　jump_model=`{rec.jump_model}`　"
        f"variance_reduction=`{rec.variance_reduction}`",
        f"· cluster_method=`{rec.cluster_method}`　"
        f"use_calibrated=`{str(rec.use_calibrated).lower()}`　"
        f"human_review=`{str(rec.human_review).lower()}`",
        "· 推荐理由：",
    ]
    for r in rec.reasons:
        lines.append(f"  - {r}")
    return "  \n".join(lines)


def start_keys_for_mode(
    mode: str,
    *,
    include_bucket: bool = False,
) -> tuple[str, ...]:
    """
    Required start-choice keys for 简洁 / 专家 modes.

    简洁：pair + bullish (+ bucket at run when no PDF).
    专家：full START_REQUIRED set (caller may still drop bucket in dialog).
    """
    m = (mode or "").strip().lower()
    simple = m in {"simple", "简洁", "简洁（推荐）", "recommended"}
    if simple:
        keys: list[str] = ["pair", "bullish_currency"]
        if include_bucket:
            keys.append("bucket_mode")
        return tuple(keys)
    keys = [
        "pair",
        "bullish_currency",
        "peak_engine",
        "use_calibrated",
        "human_review",
    ]
    if include_bucket:
        keys.append("bucket_mode")
    return tuple(keys)


def is_simple_setup_mode(mode: str | None) -> bool:
    m = (mode or "").strip().lower()
    return m in {"simple", "简洁", "简洁（推荐）", "recommended"}
