"""
Event / topic clustering for news evidence (ECDA-style redundancy control).

Same-theme headlines must not linearly stack into evidence score S.
Default: pure-Python Jaccard on title tokens (no sklearn).
Optional: TF-IDF + cosine when sklearn is installed (`cluster_method=tfidf`);
falls back to Jaccard if sklearn is missing.

Default production path: assign cluster_id, then keep_strongest within each cluster
when aggregating S (non-reps still visible in audit with cluster_role=dup).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from fx_report.model.weights import EvidenceItem

DedupMode = Literal["keep_strongest", "soft_avg", "soft_sqrt", "off"]
ClusterMethod = Literal["jaccard", "tfidf"]

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
# Cosine threshold for TF-IDF titles (slightly higher — denser scores)
DEFAULT_TFIDF_COSINE_THRESHOLD = 0.55
DEFAULT_CLUSTER_METHOD: ClusterMethod = "jaccard"

# Anomaly thresholds (honest audit — no invented evidence)
HEAVY_DEDUP_RATIO = 0.50  # dup / raw ≥ this → 去重过重
NEAR_MISS_BAND = 0.08  # sim in [threshold−band, threshold) → 可能漏合
OVER_MERGE_MIN_SIZE = 2


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except ImportError:
        return False


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


def _norm_cat(e: EvidenceItem) -> str:
    return (e.category or "").strip().lower()


def _is_loose_cat(cat: str) -> bool:
    return not cat or cat in {"other", "unclassified"}


@dataclass
class ClusterMeta:
    evidence_raw_n: int
    cluster_n: int
    cluster_dup_n: int
    cluster_dedup_applied: bool
    cluster_dedup_mode: DedupMode
    jaccard_threshold: float
    cluster_warnings: list[str] = field(default_factory=list)
    cluster_method: ClusterMethod = "jaccard"
    cluster_method_requested: ClusterMethod = "jaccard"
    similarity_threshold: float = DEFAULT_JACCARD_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_raw_n": self.evidence_raw_n,
            "cluster_n": self.cluster_n,
            "cluster_dup_n": self.cluster_dup_n,
            "cluster_dedup_applied": self.cluster_dedup_applied,
            "cluster_dedup_mode": self.cluster_dedup_mode,
            "jaccard_threshold": self.jaccard_threshold,
            "cluster_warnings": list(self.cluster_warnings or []),
            "cluster_method": self.cluster_method,
            "cluster_method_requested": self.cluster_method_requested,
            "similarity_threshold": self.similarity_threshold,
        }


def _same_event_guard(a: EvidenceItem, b: EvidenceItem) -> bool:
    """Prefer merging only when category agrees (or either is empty/other)."""
    ca = _norm_cat(a)
    cb = _norm_cat(b)
    if _is_loose_cat(ca) or _is_loose_cat(cb):
        return True
    return ca == cb


def detect_cluster_warnings(
    items: Sequence[EvidenceItem],
    *,
    threshold: float = DEFAULT_JACCARD_THRESHOLD,
    heavy_dedup_ratio: float = HEAVY_DEDUP_RATIO,
    near_miss_band: float = NEAR_MISS_BAND,
    news_meta: dict[str, Any] | None = None,
    sim_matrix: list[list[float]] | None = None,
    sim_label: str = "Jaccard",
) -> list[str]:
    """
    Detect clustering / evidence anomalies after assign_event_clusters.

    Returns Chinese messages for audit UI. Does not invent evidence — only
    flags patterns on already-assigned cluster fields + titles.
    sim_matrix: optional n×n pairwise similarity (defaults to title Jaccard).
    """
    warnings: list[str] = []
    n = len(items)
    if n == 0:
        # Related empty-news / rate-limit notes (consistent style with fetch meta)
        if news_meta:
            warnings.extend(_related_news_warnings(news_meta))
        return warnings

    # Group by cluster_id (empty id → treat each as own solo)
    by_cid: dict[str, list[EvidenceItem]] = {}
    for e in items:
        cid = (e.cluster_id or "").strip() or f"__solo_{e.id or id(e)}"
        by_cid.setdefault(cid, []).append(e)

    cluster_n = len(by_cid)
    dup_n = sum(1 for e in items if (e.cluster_role or "").lower() == "dup")

    # 1) Possible over-merge: conflicting directions or distinct tight categories
    over_dir: list[str] = []
    over_cat: list[str] = []
    for cid, members in by_cid.items():
        if len(members) < OVER_MERGE_MIN_SIZE:
            continue
        dirs = {int(e.direction) for e in members if int(e.direction) != 0}
        if len(dirs) >= 2:
            over_dir.append(cid if not cid.startswith("__solo_") else members[0].id)
        # Distinct named categories in one cluster (incl. other vs fed via loose guard)
        named = {
            _norm_cat(e)
            for e in members
            if _norm_cat(e) and _norm_cat(e) != "unclassified"
        }
        if len(named) >= 2:
            over_cat.append(cid if not cid.startswith("__solo_") else members[0].id)
    if over_dir:
        warnings.append(
            f"可能过度合并：簇内方向冲突（{', '.join(over_dir[:5])}）"
            + ("…" if len(over_dir) > 5 else "")
        )
    if over_cat:
        warnings.append(
            f"可能过度合并：簇内类别不一致（{', '.join(over_cat[:5])}）"
            + ("…" if len(over_cat) > 5 else "")
        )

    # 2) Possible under-merge / near-miss: high similarity but not same cluster
    if sim_matrix is None:
        tokens = [tokenize_title(e.title) for e in items]
        sim_matrix = [
            [jaccard(tokens[i], tokens[j]) if i != j else 1.0 for j in range(n)]
            for i in range(n)
        ]
    near_miss: list[str] = []
    blocked: list[str] = []
    floor = max(0.0, threshold - near_miss_band)
    for i in range(n):
        for j in range(i + 1, n):
            ci = (items[i].cluster_id or "").strip()
            cj = (items[j].cluster_id or "").strip()
            if ci and cj and ci == cj:
                continue
            sim = float(sim_matrix[i][j])
            if sim <= 0:
                continue
            id_pair = f"{items[i].id}/{items[j].id}"
            if sim >= threshold and not _same_event_guard(items[i], items[j]):
                blocked.append(f"{id_pair}({sim_label[0]}={sim:.2f})")
            elif floor <= sim < threshold and _same_event_guard(items[i], items[j]):
                near_miss.append(f"{id_pair}({sim_label[0]}={sim:.2f})")
    if near_miss:
        warnings.append(
            f"可能漏合（阈值边缘）：标题相近但未聚类 "
            f"{', '.join(near_miss[:4])}"
            + ("…" if len(near_miss) > 4 else "")
        )
    if blocked:
        warnings.append(
            f"可能漏合（类别阻隔）：{sim_label}≥阈值但类别不同 "
            f"{', '.join(blocked[:4])}"
            + ("…" if len(blocked) > 4 else "")
        )

    # 3) Heavy dedup: large fraction dropped from S via keep_strongest
    if n >= 3 and dup_n / float(n) >= heavy_dedup_ratio:
        warnings.append(
            f"去重过重：raw={n} → 簇={cluster_n}（dup={dup_n}，"
            f"约 {100 * dup_n / float(n):.0f}% 不计入 S）"
        )

    # 4) Empty after dedup while raw > 0 (no item contributes under keep_strongest)
    scoring_n = sum(
        1
        for e in items
        if (e.category or "").lower() != "unclassified"
        and cluster_score_mult(e, mode="keep_strongest") > 0
        and int(e.direction) != 0
    )
    if n > 0 and scoring_n == 0:
        warnings.append(
            f"去重/过滤后无有效证据计入 S（raw={n}，cluster_n={cluster_n}）"
        )

    if news_meta:
        warnings.extend(_related_news_warnings(news_meta))

    return warnings


def _related_news_warnings(news_meta: dict[str, Any]) -> list[str]:
    """Optional related honesty notes — same Chinese style as fetch limitations."""
    out: list[str] = []
    eq = str(news_meta.get("evidence_quality") or "")
    fetched = int(news_meta.get("fetched") or 0)
    if eq == "news_empty_no_prior":
        if fetched > 0:
            out.append(
                f"新闻为空：抓取 {fetched} 条但未产出证据（template_policy=off，未静默填模板）"
            )
        else:
            out.append("新闻为空：未抓到标题且未使用先验证据（S≈0）")
    lim = str(news_meta.get("limitation") or "")
    if "429" in lim:
        out.append(f"新闻源限流（429）：{lim[:160]}")
    return out


def _pairwise_jaccard(items: Sequence[EvidenceItem]) -> list[list[float]]:
    tokens = [tokenize_title(e.title) for e in items]
    n = len(items)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        mat[i][i] = 1.0
        for j in range(i + 1, n):
            s = jaccard(tokens[i], tokens[j])
            mat[i][j] = s
            mat[j][i] = s
    return mat


def _pairwise_tfidf_cosine(items: Sequence[EvidenceItem]) -> list[list[float]] | None:
    """Return cosine similarity matrix, or None if sklearn unavailable."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return None
    texts = [(e.title or "").strip() or " " for e in items]
    try:
        vec = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"[a-z0-9]{2,}",
            stop_words=list(_STOP),
            min_df=1,
        )
        x = vec.fit_transform(texts)
        sim = cosine_similarity(x)
    except Exception:
        return None
    n = len(items)
    return [[float(sim[i, j]) for j in range(n)] for i in range(n)]


def resolve_cluster_similarity(
    items: Sequence[EvidenceItem],
    *,
    method: ClusterMethod = DEFAULT_CLUSTER_METHOD,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    tfidf_threshold: float = DEFAULT_TFIDF_COSINE_THRESHOLD,
) -> tuple[ClusterMethod, list[list[float]], float, str]:
    """
    Build pairwise similarity + effective threshold.

    Returns (effective_method, sim_matrix, threshold, sim_label).
    tfidf requested without sklearn → Jaccard fallback.
    """
    requested = method if method in {"jaccard", "tfidf"} else "jaccard"
    if requested == "tfidf":
        mat = _pairwise_tfidf_cosine(items)
        if mat is not None:
            return "tfidf", mat, float(tfidf_threshold), "cosine"
        # fallback
        return (
            "jaccard",
            _pairwise_jaccard(items),
            float(jaccard_threshold),
            "Jaccard",
        )
    return (
        "jaccard",
        _pairwise_jaccard(items),
        float(jaccard_threshold),
        "Jaccard",
    )


def assign_event_clusters(
    items: list[EvidenceItem],
    *,
    threshold: float | None = None,
    enabled: bool = True,
    detect_warnings: bool = True,
    news_meta: dict[str, Any] | None = None,
    cluster_method: ClusterMethod = DEFAULT_CLUSTER_METHOD,
    tfidf_threshold: float = DEFAULT_TFIDF_COSINE_THRESHOLD,
) -> ClusterMeta:
    """
    Greedy single-linkage on title similarity; mutates items in place.

    cluster_method:
      jaccard (default) — token-set Jaccard; no extra deps
      tfidf — TF-IDF + cosine if sklearn available, else Jaccard fallback

    Sets cluster_id (EVT-01…), cluster_size, cluster_role (rep|dup|solo).
    Representative = highest |contrib|; tie → lower id (earlier N-01).
    When detect_warnings=True, fills cluster_warnings with Chinese anomaly notes.
    """
    n = len(items)
    requested: ClusterMethod = (
        cluster_method if cluster_method in {"jaccard", "tfidf"} else "jaccard"
    )
    jac_thr = DEFAULT_JACCARD_THRESHOLD if threshold is None else float(threshold)

    if not enabled or n == 0:
        for e in items:
            e.cluster_id = ""
            e.cluster_size = 1
            e.cluster_role = ""
        eff_method: ClusterMethod = "jaccard"
        eff_thr = jac_thr
        if requested == "tfidf" and sklearn_available():
            eff_method = "tfidf"
            eff_thr = float(tfidf_threshold)
        meta = ClusterMeta(
            evidence_raw_n=n,
            cluster_n=n,
            cluster_dup_n=0,
            cluster_dedup_applied=False,
            cluster_dedup_mode="off",
            jaccard_threshold=jac_thr,
            cluster_method=eff_method,
            cluster_method_requested=requested,
            similarity_threshold=eff_thr,
        )
        if detect_warnings:
            meta.cluster_warnings = detect_cluster_warnings(
                items, threshold=eff_thr, news_meta=news_meta
            )
        return meta

    eff_method, sim_matrix, eff_thr, sim_label = resolve_cluster_similarity(
        items,
        method=requested,
        jaccard_threshold=jac_thr,
        tfidf_threshold=tfidf_threshold,
    )
    # Explicit threshold override applies to whichever method is active
    if threshold is not None:
        eff_thr = float(threshold)

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
            if sim_matrix[i][j] >= eff_thr:
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

    meta = ClusterMeta(
        evidence_raw_n=n,
        cluster_n=cluster_n,
        cluster_dup_n=dup_n,
        cluster_dedup_applied=dup_n > 0,
        cluster_dedup_mode="keep_strongest",
        jaccard_threshold=jac_thr,
        cluster_method=eff_method,
        cluster_method_requested=requested,
        similarity_threshold=eff_thr,
    )
    if detect_warnings:
        meta.cluster_warnings = detect_cluster_warnings(
            items,
            threshold=eff_thr,
            news_meta=news_meta,
            sim_matrix=sim_matrix,
            sim_label=sim_label,
        )
        if requested == "tfidf" and eff_method != "tfidf":
            meta.cluster_warnings.append(
                "cluster_method=tfidf 请求但未安装 sklearn，已回退 Jaccard"
            )
    return meta


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
