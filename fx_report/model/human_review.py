"""
人机协同 + Active Learning 优先排序：不确定证据检测与人工方向覆盖。

不确定触发（可叠加）：
  - low_confidence：强度偏弱（SLIGHT / strength≤阈值）
  - rules_llm_conflict：LLM 结论与关键词规则方向冲突
  - cluster_direction_conflict：同事件簇内方向冲突
  - unclear_category：类别为 unclassified / other
  - near_neutral_margin：方向中性但强度不弱（边界样本）

Active Learning 选点（赋权前 top-N）：
  1) 不确定度打分（原因权重 + 弱强度 bump + 多原因叠加）
  2) 簇多样性：同簇后续候选降权，避免 HITL 全挤在同一 EVT-*
  3) 代表优先：同簇内优先问 rep / 高 |direction|

人工选项：利多(up) / 利空(down) / 中性(neutral) / 跳过(skip)
不发明证据；仅对已有条目打标与覆盖方向。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

from fx_report.model.weights import EvidenceItem

ReviewChoice = Literal["up", "down", "neutral", "skip"]

REASON_CODES: tuple[str, ...] = (
    "low_confidence",
    "rules_llm_conflict",
    "cluster_direction_conflict",
    "unclear_category",
    "near_neutral_margin",
)

REASON_ZH: dict[str, str] = {
    "low_confidence": "置信度偏低（强度偏弱）",
    "rules_llm_conflict": "规则与 LLM 方向冲突",
    "cluster_direction_conflict": "同簇内方向冲突",
    "unclear_category": "类别不清（未分类/其他）",
    "near_neutral_margin": "方向中性但强度不弱（边界样本）",
}

CHOICE_ZH: dict[str, str] = {
    "up": "利多",
    "down": "利空",
    "neutral": "中性",
    "skip": "跳过（保留模型）",
}

CHOICE_TO_DIR: dict[str, int | None] = {
    "up": 1,
    "down": -1,
    "neutral": 0,
    "skip": None,
}

DEFAULT_MAX_UNCERTAIN = 5
DEFAULT_LOW_STRENGTH = 1.0  # strength_label SLIGHT 上界
DEFAULT_NEUTRAL_STRENGTH = 1.2  # near_neutral_margin 下限
# Diversity: multiply remaining same-cluster scores by this after each pick
DEFAULT_CLUSTER_DIVERSITY = 0.35


def direction_label(direction: int) -> str:
    if direction > 0:
        return "up"
    if direction < 0:
        return "down"
    return "neutral"


def direction_zh(direction: int) -> str:
    lab = direction_label(direction)
    return {"up": "利多", "down": "利空", "neutral": "中性"}.get(lab, lab)


@dataclass
class PendingReview:
    """一条待人工确认的不确定证据（展示用，不新增事实）。"""

    evidence_id: str
    statement_id: str
    title: str
    snippet: str
    url: str
    model_direction: int
    model_direction_label: str
    model_category: str
    reasons: list[str] = field(default_factory=list)
    reasons_zh: list[str] = field(default_factory=list)
    uncertainty_score: float = 0.0
    cluster_id: str = ""
    strength: float = 0.0
    strength_label: str = ""
    rules_direction: int | None = None
    from_llm: bool = False
    priority_rank: int = 0  # 1-based AL rank after diversity selection
    al_score: float = 0.0  # final score used for pick (post-diversity)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PendingReview":
        return cls(
            evidence_id=str(raw.get("evidence_id") or ""),
            statement_id=str(raw.get("statement_id") or ""),
            title=str(raw.get("title") or ""),
            snippet=str(raw.get("snippet") or ""),
            url=str(raw.get("url") or ""),
            model_direction=int(raw.get("model_direction") or 0),
            model_direction_label=str(raw.get("model_direction_label") or "neutral"),
            model_category=str(raw.get("model_category") or ""),
            reasons=list(raw.get("reasons") or []),
            reasons_zh=list(raw.get("reasons_zh") or []),
            uncertainty_score=float(raw.get("uncertainty_score") or 0.0),
            cluster_id=str(raw.get("cluster_id") or ""),
            strength=float(raw.get("strength") or 0.0),
            strength_label=str(raw.get("strength_label") or ""),
            rules_direction=(
                int(raw["rules_direction"])
                if raw.get("rules_direction") is not None
                else None
            ),
            from_llm=bool(raw.get("from_llm", False)),
            priority_rank=int(raw.get("priority_rank") or 0),
            al_score=float(raw.get("al_score") or raw.get("uncertainty_score") or 0.0),
        )


def _is_llm_item(e: EvidenceItem) -> bool:
    note = (e.note or "").lower()
    return note.startswith("llm") or "llm精读" in note or "｜llm" in note


def _rules_guess(
    title: str,
    pair: str | Any | None,
) -> tuple[int | None, str]:
    if not pair or not (title or "").strip():
        return None, ""
    try:
        from fx_report.news.classify import rules_direction_guess

        return rules_direction_guess(title, pair)
    except Exception:
        return None, ""


def _conflict_cluster_ids(evidence: Sequence[EvidenceItem]) -> set[str]:
    by_cid: dict[str, list[EvidenceItem]] = {}
    for e in evidence:
        cid = (e.cluster_id or "").strip()
        if not cid:
            continue
        by_cid.setdefault(cid, []).append(e)
    out: set[str] = set()
    for cid, members in by_cid.items():
        if len(members) < 2:
            continue
        dirs = {int(m.direction) for m in members if int(m.direction) != 0}
        if len(dirs) >= 2:
            out.add(cid)
    return out


def _score_uncertainty(
    reasons: Sequence[str],
    *,
    strength: float,
    low_strength_threshold: float,
    direction: int = 0,
    cluster_role: str = "",
) -> float:
    """Higher = more uncertain / higher AL value; used to pick top-N."""
    base = 0.0
    weights = {
        "rules_llm_conflict": 3.0,
        "cluster_direction_conflict": 2.5,
        "unclear_category": 2.0,
        "near_neutral_margin": 1.8,
        "low_confidence": 1.5,
    }
    for r in reasons:
        base += weights.get(r, 1.0)
    # Multi-reason stack bonus (information gain proxy)
    if len(reasons) >= 2:
        base += 0.4 * (len(reasons) - 1)
    # Weaker strength → bump
    if strength <= low_strength_threshold:
        base += max(0.0, (low_strength_threshold - float(strength)) * 0.5)
    # Prefer asking the cluster representative when conflict exists
    role = (cluster_role or "").lower()
    if "cluster_direction_conflict" in reasons and role in {"rep", "solo", ""}:
        base += 0.35
    elif "cluster_direction_conflict" in reasons and role == "dup":
        base -= 0.15
    # Soft margin: decisive direction with many reasons still ranks high via abs
    base += 0.05 * abs(int(direction))
    return base


def _build_snippet(e: EvidenceItem) -> str:
    """Prefer compressed summary blurb; fall back to title + note."""
    summary = (e.summary or "").strip()
    if summary:
        return summary[:280]
    title = (e.title or "").strip()
    snippet = title[:220]
    note = (e.note or "").strip()
    if note and note not in snippet:
        snippet = f"{snippet} — {note[:120]}" if snippet else note[:220]
    return snippet[:280]


def _collect_candidates(
    evidence: Sequence[EvidenceItem],
    *,
    pair: str | Any | None,
    low_strength_threshold: float,
    neutral_strength_threshold: float,
    include_priors: bool,
) -> list[PendingReview]:
    conflict_cids = _conflict_cluster_ids(evidence)
    candidates: list[PendingReview] = []
    role_by_id = {
        str(e.id or ""): (e.cluster_role or "") for e in evidence
    }

    for e in evidence:
        if e.is_prior and not include_priors:
            continue
        cat = (e.category or "").strip().lower()
        reasons: list[str] = []
        rules_dir: int | None = None
        from_llm = _is_llm_item(e)

        if float(e.strength) <= float(low_strength_threshold) or (
            e.strength_label or ""
        ).upper() == "SLIGHT":
            reasons.append("low_confidence")

        if cat in {"unclassified", "other", ""}:
            reasons.append("unclear_category")

        cid = (e.cluster_id or "").strip()
        if cid and cid in conflict_cids:
            reasons.append("cluster_direction_conflict")

        # Boundary: model says neutral but strength is not trivial
        if int(e.direction) == 0 and float(e.strength) >= float(neutral_strength_threshold):
            if cat not in {"unclassified", ""}:  # unclassified already covered
                reasons.append("near_neutral_margin")

        if from_llm and pair is not None:
            rules_dir, _rules_cat = _rules_guess(e.title or "", pair)
            if rules_dir is not None and int(rules_dir) != int(e.direction):
                # Only flag when both sides are decisive (±1) and disagree
                if int(e.direction) != 0 and int(rules_dir) != 0:
                    reasons.append("rules_llm_conflict")
                elif int(e.direction) == 0 and int(rules_dir) != 0:
                    reasons.append("rules_llm_conflict")

        if not reasons:
            continue

        # Deduplicate reason codes preserving order
        seen: set[str] = set()
        uniq: list[str] = []
        for r in reasons:
            if r not in seen:
                seen.add(r)
                uniq.append(r)

        score = _score_uncertainty(
            uniq,
            strength=float(e.strength),
            low_strength_threshold=low_strength_threshold,
            direction=int(e.direction),
            cluster_role=role_by_id.get(str(e.id or ""), e.cluster_role or ""),
        )
        title = (e.title or "").strip()

        candidates.append(
            PendingReview(
                evidence_id=str(e.id or ""),
                statement_id=str(e.statement_id or e.id or ""),
                title=title[:180],
                snippet=_build_snippet(e),
                url=str(e.url or ""),
                model_direction=int(e.direction),
                model_direction_label=direction_label(int(e.direction)),
                model_category=cat or "other",
                reasons=uniq,
                reasons_zh=[REASON_ZH.get(r, r) for r in uniq],
                uncertainty_score=score,
                cluster_id=cid,
                strength=float(e.strength),
                strength_label=str(e.strength_label or ""),
                rules_direction=rules_dir,
                from_llm=from_llm,
                al_score=score,
            )
        )
    return candidates


def prioritize_for_hitl(
    candidates: Sequence[PendingReview],
    *,
    max_items: int = DEFAULT_MAX_UNCERTAIN,
    cluster_diversity: float = DEFAULT_CLUSTER_DIVERSITY,
) -> list[PendingReview]:
    """
    Active-learning selection: greedy top-N with cluster diversity.

    After picking an item from cluster C, remaining candidates in C are
    multiplied by `cluster_diversity` (<1) so the queue spreads across themes.
    """
    cap = max(0, int(max_items))
    if cap == 0 or not candidates:
        return []

    remaining = [
        PendingReview(
            evidence_id=c.evidence_id,
            statement_id=c.statement_id,
            title=c.title,
            snippet=c.snippet,
            url=c.url,
            model_direction=c.model_direction,
            model_direction_label=c.model_direction_label,
            model_category=c.model_category,
            reasons=list(c.reasons),
            reasons_zh=list(c.reasons_zh),
            uncertainty_score=c.uncertainty_score,
            cluster_id=c.cluster_id,
            strength=c.strength,
            strength_label=c.strength_label,
            rules_direction=c.rules_direction,
            from_llm=c.from_llm,
            al_score=float(c.uncertainty_score),
        )
        for c in candidates
    ]
    picked: list[PendingReview] = []
    div = max(0.0, min(1.0, float(cluster_diversity)))

    while remaining and len(picked) < cap:
        remaining.sort(
            key=lambda p: (
                -p.al_score,
                -abs(p.model_direction),
                -p.uncertainty_score,
                p.evidence_id,
            )
        )
        best = remaining.pop(0)
        best.priority_rank = len(picked) + 1
        picked.append(best)
        cid = (best.cluster_id or "").strip()
        if not cid or div >= 1.0:
            continue
        for p in remaining:
            if (p.cluster_id or "").strip() == cid:
                p.al_score = float(p.al_score) * div

    return picked


def detect_uncertain_evidence(
    evidence: Sequence[EvidenceItem],
    *,
    pair: str | Any | None = None,
    max_items: int = DEFAULT_MAX_UNCERTAIN,
    low_strength_threshold: float = DEFAULT_LOW_STRENGTH,
    neutral_strength_threshold: float = DEFAULT_NEUTRAL_STRENGTH,
    include_priors: bool = False,
    cluster_diversity: float = DEFAULT_CLUSTER_DIVERSITY,
) -> list[PendingReview]:
    """
    Scan evidence after classify+cluster; return top-N AL-prioritized items.
    Does not invent rows — only flags existing EvidenceItem.
    """
    candidates = _collect_candidates(
        evidence,
        pair=pair,
        low_strength_threshold=low_strength_threshold,
        neutral_strength_threshold=neutral_strength_threshold,
        include_priors=include_priors,
    )
    return prioritize_for_hitl(
        candidates,
        max_items=max_items,
        cluster_diversity=cluster_diversity,
    )


# Back-compat alias used in docs / active-learning wording
active_learning_prioritize = detect_uncertain_evidence


def normalize_review_choice(raw: Any) -> ReviewChoice | "":
    if raw is None or raw == "":
        return ""
    s = str(raw).strip().lower()
    aliases = {
        "up": "up",
        "利多": "up",
        "bullish": "up",
        "+1": "up",
        "1": "up",
        "down": "down",
        "利空": "down",
        "bearish": "down",
        "-1": "down",
        "neutral": "neutral",
        "中性": "neutral",
        "0": "neutral",
        "skip": "skip",
        "跳过": "skip",
        "pass": "skip",
        "keep": "skip",
    }
    out = aliases.get(s, "")
    return out if out in CHOICE_TO_DIR else ""  # type: ignore[return-value]


def apply_review_overrides(
    evidence: Sequence[EvidenceItem],
    choices: dict[str, Any],
) -> tuple[list[EvidenceItem], dict[str, Any]]:
    """
    Apply human choices keyed by evidence_id or statement_id.

    skip / empty → keep model direction.
    Returns (new evidence list, meta with n_overridden / applied map).
    """
    lab_map: dict[str, ReviewChoice] = {}
    for k, v in (choices or {}).items():
        key = str(k or "").strip()
        ch = normalize_review_choice(v)
        if key and ch:
            lab_map[key] = ch  # type: ignore[assignment]

    out: list[EvidenceItem] = []
    applied: dict[str, str] = {}
    n_overridden = 0
    n_skipped = 0

    for e in evidence:
        new_e = EvidenceItem(
            id=e.id,
            title=e.title,
            direction=int(e.direction),
            strength=float(e.strength),
            freshness=float(e.freshness),
            unpriced=float(e.unpriced),
            category=e.category,
            note=e.note,
            strength_label=e.strength_label,
            strength_breakdown=dict(e.strength_breakdown or {}),
            source_tier=e.source_tier,
            surprise=e.surprise,
            scope=e.scope,
            statement_id=e.statement_id,
            url=e.url,
            is_prior=bool(e.is_prior),
            cluster_id=e.cluster_id,
            cluster_size=int(e.cluster_size or 1),
            cluster_role=e.cluster_role,
            summary=str(e.summary or ""),
        )
        sid = str(e.statement_id or e.id or "")
        eid = str(e.id or "")
        ch = lab_map.get(eid) or lab_map.get(sid) or ""
        if not ch or ch == "skip":
            if ch == "skip":
                n_skipped += 1
                applied[eid or sid] = "skip"
            out.append(new_e)
            continue
        d_int = CHOICE_TO_DIR.get(ch)
        if d_int is None:
            out.append(new_e)
            continue
        new_e.direction = int(d_int)
        tag = f"human_review｜{ch}"
        new_e.note = f"{tag}｜{new_e.note}" if new_e.note else tag
        # Human override on unclassified: lift category gate if they set direction
        if (new_e.category or "").lower() == "unclassified" and d_int != 0:
            # Keep category honest; scoring still zeros unclassified in step5 —
            # bump to other so human direction can contribute.
            new_e.category = "other"
        n_overridden += 1
        applied[eid or sid] = ch
        out.append(new_e)

    return out, {
        "n_overridden": n_overridden,
        "n_skipped": n_skipped,
        "applied": applied,
        "n_choices": len(lab_map),
    }


def reviews_to_label_rows(
    pending: Sequence[PendingReview],
    choices: dict[str, Any],
    *,
    pair: str = "",
) -> list[dict[str, Any]]:
    """Build label_audit-compatible rows from HITL choices (skip→empty human)."""
    from fx_report.model.label_audit import compute_agree, normalize_direction

    rows: list[dict[str, Any]] = []
    for p in pending:
        eid = p.evidence_id
        sid = p.statement_id or eid
        ch = normalize_review_choice(choices.get(eid) or choices.get(sid) or "")
        human = "" if (not ch or ch == "skip") else ch
        model = p.model_direction_label
        rows.append(
            {
                "statement_id": sid,
                "title": p.title,
                "url": p.url,
                "model_category": p.model_category,
                "model_direction": model,
                "human_direction": normalize_direction(human) if human else "",
                "human_category": "",
                "agree": compute_agree(model, human) if human else "",
                "pair": pair,
                "hitl_reasons": "|".join(p.reasons),
                "al_score": p.al_score or p.uncertainty_score,
                "priority_rank": p.priority_rank,
            }
        )
    return rows
