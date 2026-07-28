"""
Event / topic clustering for news evidence (ECDA-style redundancy control).

Same-theme headlines must not linearly stack into evidence score S.
Pure-Python + optional numpy; no sklearn dependency.

Default production path: assign cluster_id, then keep_strongest within each cluster
when aggregating S (non-reps still visible in audit with cluster_role=dup).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from fx_report.model.weights import EvidenceItem

DedupMode = Literal["keep_strongest", "soft_avg", "soft_sqrt", "off"]

# Light English FX stopwords — keep tokens short and comparable across wires
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "as",
        "at",
        "by",
        "from",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "after",
        "over",
        "into",
        "vs",
        "says",
        "said",
        "amid",
        "near",
        "as",
        "us",
        "u",
        "s",
    }
)
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

# Near-duplicate threshold (Jaccard on title tokens)
DEFAULT_JACCARD_THRESHOLD = 0.45


def tokenize_title(text: str) -> frozenset[str]:
    raw = _TOKEN_RE.findall((text or "").lower())
    return frozenset(t for t in raw if len(t) > 1 and t not in _STOP)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))


def _abs_contrib(e: EvidenceItem) -> float:
    return abs(float(e.direction) * float(e.strength) * float(e.freshness) * float(e.unpriced))


@dataclass
class ClusterMeta:
    evidence_raw_n: int
    cluster_n: int
    cluster_dup_n: int
    cluster_dedup_applied: bool
    cluster_dedup_mode: DedupMode
    jaccard_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_raw_n": self.evidence_raw_n,
            "cluster_n": self.cluster_n,
            "cluster_dup_n": self.cluster_dup_n,
            "cluster_dedup_applied": self.cluster_dedup_applied,
            "cluster_dedup_mode": self.cluster_dedup_mode,
            "jaccard_threshold": self.jaccard_threshold,
        }


def _same_event_guard(a: EvidenceItem, b: EvidenceItem) -> bool:
    """Prefer merging only when category agrees (or either is empty/other)."""
    ca = (a.category or "").strip().lower()
    cb = (b.category or "").strip().lower()
    if not ca or not cb or ca in {"other", "unclassified"} or cb in {"other", "unclassified"}:
        return True
    return ca == cb


def assign_event_clusters(
    items: list[EvidenceItem],
    *,
    threshold: float = DEFAULT_JACCARD_THRESHOLD,
    enabled: bool = True,
) -> ClusterMeta:
    """
    Greedy single-linkage on title Jaccard; mutates items in place.

    Sets cluster_id (EVT-01…), cluster_size, cluster_role (rep|dup|solo).
    Representative = highest |contrib|; tie → lower id (earlier N-01).
    """
    n = len(items)
    if not enabled or n == 0:
        for e in items:
            e.cluster_id = ""
            e.cluster_size = 1
            e.cluster_role = ""
        return ClusterMeta(
            evidence_raw_n=n,
            cluster_n=n,
            cluster_dup_n=0,
            cluster_dedup_applied=False,
            cluster_dedup_mode="off",
            jaccard_threshold=threshold,
        )

    tokens = [tokenize_title(e.title) for e in items]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if not _same_event_guard(items[i], items[j]):
                continue
            if jaccard(tokens[i], tokens[j]) >= threshold:
                union(i, j)

    # Map root → members
    groups: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    # Stable cluster numbering by first member index
    ordered_roots = sorted(groups.keys(), key=lambda r: min(groups[r]))
    cluster_n = len(ordered_roots)
    dup_n = 0

    for k, root in enumerate(ordered_roots, start=1):
        members = groups[root]
        cid = f"EVT-{k:02d}"
        size = len(members)
        # Representative: max |contrib|; tie → lexicographically smaller id (earlier N-01)
        rep_i = min(
            members,
            key=lambda idx: (-_abs_contrib(items[idx]), items[idx].id or ""),
        )

        for idx in members:
            e = items[idx]
            e.cluster_id = cid
            e.cluster_size = size
            if size == 1:
                e.cluster_role = "solo"
            elif idx == rep_i:
                e.cluster_role = "rep"
            else:
                e.cluster_role = "dup"
                dup_n += 1

    return ClusterMeta(
        evidence_raw_n=n,
        cluster_n=cluster_n,
        cluster_dup_n=dup_n,
        cluster_dedup_applied=dup_n > 0,
        cluster_dedup_mode="keep_strongest",
        jaccard_threshold=threshold,
    )


def cluster_score_mult(e: EvidenceItem, *, mode: DedupMode = "keep_strongest") -> float:
    """Multiplier applied to signed contrib when summing S."""
    if mode == "off" or not e.cluster_id:
        return 1.0
    size = max(1, int(e.cluster_size or 1))
    role = (e.cluster_role or "").lower()
    if mode == "keep_strongest":
        if size <= 1 or role in {"", "solo", "rep"}:
            return 1.0
        return 0.0  # dup
    if mode == "soft_avg":
        return 1.0 / float(size)
    if mode == "soft_sqrt":
        return 1.0 / math.sqrt(float(size))
    return 1.0


def propagate_cluster_to_statements(
    evidence: Sequence[EvidenceItem],
    statements: Sequence[Any],
) -> None:
    """Copy cluster_id onto linked StoredStatement (mutates)."""
    by_sid = {e.statement_id: e for e in evidence if e.statement_id}
    by_url = {
        (e.url or "").strip().lower(): e
        for e in evidence
        if (e.url or "").strip()
    }
    by_title = {(e.title or "").strip().lower(): e for e in evidence if e.title}
    for s in statements:
        hit: EvidenceItem | None = None
        sid = getattr(s, "id", "") or ""
        if sid and sid in by_sid:
            hit = by_sid[sid]
        if hit is None:
            url = (getattr(s, "url", "") or "").strip().lower()
            if url:
                hit = by_url.get(url)
        if hit is None:
            stmt = (getattr(s, "statement", "") or "").split(" — ")[0].strip().lower()
            if stmt:
                hit = by_title.get(stmt)
        if hit is not None and hasattr(s, "cluster_id"):
            s.cluster_id = hit.cluster_id or ""
