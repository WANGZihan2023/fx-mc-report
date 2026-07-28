"""Event/topic clustering: near-duplicate headlines must not inflate S."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fx_report.model.weights import EvidenceItem, evidence_score
from fx_report.news.cluster import (
    assign_event_clusters,
    detect_cluster_warnings,
    jaccard,
    tokenize_title,
)


def _ev(
    eid: str,
    title: str,
    *,
    direction: int = 1,
    strength: float = 0.8,
    freshness: float = 1.0,
    unpriced: float = 0.7,
    category: str = "fed",
) -> EvidenceItem:
    return EvidenceItem(
        id=eid,
        title=title,
        direction=direction,
        strength=strength,
        freshness=freshness,
        unpriced=unpriced,
        category=category,
    )


def test_jaccard_near_identical_titles() -> None:
    a = tokenize_title("Fed signals rate hike as inflation stays hot")
    b = tokenize_title("Fed signals rate hike as inflation remains hot")
    assert jaccard(a, b) >= 0.45


def test_cluster_near_identical_headlines_dedupe_score() -> None:
    items = [
        _ev("N-01", "Fed signals hawkish hike after hot CPI print", strength=0.9),
        _ev("N-02", "Fed signals hawkish hike after hot CPI print: markets", strength=0.7),
        _ev("N-03", "Fed signals hawkish hike after hot CPI reading", strength=0.6),
        _ev("N-04", "RBA holds rates steady as Aussie softens", category="rba", direction=-1),
    ]
    raw = evidence_score(items, cluster_dedup="off")
    meta = assign_event_clusters(items)
    assert meta.cluster_dedup_applied
    assert meta.cluster_n < meta.evidence_raw_n
    assert meta.cluster_n == 2  # fed cluster + rba solo
    # All three Fed-like share one EVT id
    fed_ids = {e.cluster_id for e in items if e.category == "fed"}
    assert len(fed_ids) == 1
    roles = {e.id: e.cluster_role for e in items}
    assert roles["N-01"] == "rep"  # strongest
    assert roles["N-02"] == "dup"
    assert roles["N-03"] == "dup"
    assert roles["N-04"] == "solo"

    scored = evidence_score(items)  # keep_strongest default
    # Without dedupe, three Fed items stack; with keep_strongest only N-01 + RBA
    assert abs(scored) < abs(raw)
    # Only rep + solo contribute
    only_rep = evidence_score(
        [e for e in items if e.cluster_role in {"rep", "solo"}],
        cluster_dedup="off",
    )
    assert abs(scored - only_rep) < 1e-9


def test_soft_avg_mode_divides_by_cluster_size() -> None:
    items = [
        _ev("N-01", "Dollar rises on strong payrolls data", strength=0.8),
        _ev("N-02", "Dollar rises on strong payrolls report", strength=0.8),
    ]
    assign_event_clusters(items)
    assert items[0].cluster_id == items[1].cluster_id
    soft = evidence_score(items, cluster_dedup="soft_avg")
    raw = evidence_score(items, cluster_dedup="off")
    assert abs(soft - raw / 2.0) < 1e-9


def test_distinct_topics_not_merged() -> None:
    items = [
        _ev("N-01", "Iran conflict escalates oil risk premium", category="geopolitics"),
        _ev("N-02", "China stimulus lifts iron ore and Aussie", category="china_iron", direction=-1),
    ]
    meta = assign_event_clusters(items)
    assert meta.cluster_n == 2
    assert not meta.cluster_dedup_applied
    assert items[0].cluster_id != items[1].cluster_id


def test_warning_over_merge_conflicting_directions() -> None:
    items = [
        _ev("N-01", "Fed signals hawkish hike after hot CPI print", direction=1),
        _ev("N-02", "Fed signals hawkish hike after hot CPI print: markets", direction=-1),
    ]
    meta = assign_event_clusters(items)
    assert meta.cluster_n == 1
    assert any("过度合并" in w and "方向" in w for w in meta.cluster_warnings)


def test_warning_over_merge_mixed_categories() -> None:
    # Loose category "other" allows merge with fed; then flag category inconsistency
    items = [
        _ev("N-01", "Dollar rises on strong payrolls data", category="fed"),
        _ev("N-02", "Dollar rises on strong payrolls report", category="other"),
    ]
    meta = assign_event_clusters(items)
    assert meta.cluster_n == 1
    assert any("过度合并" in w and "类别" in w for w in meta.cluster_warnings)


def test_warning_near_miss_under_merge() -> None:
    # Jaccard ≈ 0.43 — below 0.45 threshold, inside near-miss band
    a = "powell inflation cooling wages payrolls"
    b = "powell inflation cooling housing retail"
    assert 0.37 <= jaccard(tokenize_title(a), tokenize_title(b)) < 0.45
    items = [_ev("N-01", a), _ev("N-02", b)]
    meta = assign_event_clusters(items)
    assert meta.cluster_n == 2
    assert any("漏合" in w and "阈值边缘" in w for w in meta.cluster_warnings)


def test_warning_category_blocked_under_merge() -> None:
    items = [
        _ev(
            "N-01",
            "Oil prices surge on Middle East tension",
            category="geopolitics",
        ),
        _ev(
            "N-02",
            "Oil prices jump on Middle East risks",
            category="china_iron",
            direction=-1,
        ),
    ]
    assert jaccard(tokenize_title(items[0].title), tokenize_title(items[1].title)) >= 0.45
    meta = assign_event_clusters(items)
    assert meta.cluster_n == 2
    assert any("漏合" in w and "类别阻隔" in w for w in meta.cluster_warnings)


def test_warning_heavy_dedup() -> None:
    items = [
        _ev("N-01", "Fed signals hawkish hike after hot CPI print", strength=0.9),
        _ev("N-02", "Fed signals hawkish hike after hot CPI print: markets", strength=0.7),
        _ev("N-03", "Fed signals hawkish hike after hot CPI reading", strength=0.6),
        _ev("N-04", "Fed signals hawkish hike after hot CPI data", strength=0.5),
    ]
    meta = assign_event_clusters(items)
    assert meta.cluster_dup_n >= 2
    assert any("去重过重" in w for w in meta.cluster_warnings)


def test_warning_empty_after_dedup_all_neutral() -> None:
    items = [
        _ev("N-01", "Fed signals hawkish hike after hot CPI print", direction=0),
        _ev("N-02", "Fed signals hawkish hike after hot CPI print: markets", direction=0),
    ]
    meta = assign_event_clusters(items)
    assert meta.evidence_raw_n > 0
    assert any("无有效证据" in w for w in meta.cluster_warnings)


def test_warning_news_empty_and_429_related() -> None:
    empty: list[EvidenceItem] = []
    meta = assign_event_clusters(
        empty,
        news_meta={
            "evidence_quality": "news_empty_no_prior",
            "fetched": 5,
            "limitation": "NewsAPI HTTP 429: Too Many Requests",
        },
    )
    assert any("新闻为空" in w for w in meta.cluster_warnings)
    assert any("429" in w for w in meta.cluster_warnings)


def test_detect_cluster_warnings_standalone_no_invent() -> None:
    """Warnings only describe existing items — empty input without meta → no msgs."""
    assert detect_cluster_warnings([]) == []
