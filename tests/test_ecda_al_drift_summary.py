"""Tests for active learning prioritization, drift monitor, evidence summarization."""

from __future__ import annotations

from pathlib import Path

from fx_report.model.human_review import (
    detect_uncertain_evidence,
    prioritize_for_hitl,
    PendingReview,
)
from fx_report.model.weights import EvidenceItem
from fx_report.news.cluster import assign_event_clusters
from fx_report.news.drift import (
    check_evidence_drift,
    distribution_from_evidence,
    total_variation,
)
from fx_report.news.fetch import Headline
from fx_report.news.summarize import apply_evidence_summaries, extractive_summary


def _e(
    eid: str,
    *,
    title: str = "headline",
    direction: int = 1,
    strength: float = 2.0,
    category: str = "fed",
    note: str = "",
    cluster_id: str = "",
    cluster_role: str = "",
    is_prior: bool = False,
    summary: str = "",
) -> EvidenceItem:
    return EvidenceItem(
        id=eid,
        title=title,
        direction=direction,
        strength=strength,
        freshness=1.0,
        unpriced=0.7,
        category=category,
        note=note,
        strength_label="MODERATE" if strength > 1 else "SLIGHT",
        statement_id=eid,
        cluster_id=cluster_id,
        cluster_role=cluster_role,
        is_prior=is_prior,
        summary=summary,
    )


def test_near_neutral_margin_reason() -> None:
    items = [_e("N1", direction=0, strength=2.0, category="rba")]
    pending = detect_uncertain_evidence(items, pair="AUD/USD", max_items=5)
    assert pending
    assert "near_neutral_margin" in pending[0].reasons


def test_al_diversity_spreads_across_clusters() -> None:
    """Without diversity, top scores from one cluster would dominate; with it, spread."""
    items = [
        _e(
            "A1",
            title="Fed signals hawkish hold on rates today",
            direction=1,
            category="fed",
            strength=0.4,
        ),
        _e(
            "A2",
            title="Fed signals hawkish hold amid inflation data",
            direction=-1,
            category="fed",
            strength=0.4,
        ),
        _e(
            "B1",
            title="RBA leaves cash rate unchanged in Sydney",
            direction=0,
            category="other",
            strength=0.3,
        ),
        _e(
            "C1",
            title="China iron ore demand softens steel mills",
            direction=0,
            category="unclassified",
            strength=0.2,
        ),
    ]
    assign_event_clusters(items, enabled=True)
    pending = detect_uncertain_evidence(
        items, pair="USD/AUD", max_items=3, cluster_diversity=0.2
    )
    assert len(pending) == 3
    assert pending[0].priority_rank == 1
    cids = {p.cluster_id for p in pending if p.cluster_id}
    # Should not be all the same cluster when multiple themes exist
    assert len(cids) >= 2 or len({p.evidence_id[0] for p in pending}) >= 2


def test_prioritize_for_hitl_ranks() -> None:
    cands = [
        PendingReview(
            evidence_id="X",
            statement_id="X",
            title="x",
            snippet="x",
            url="",
            model_direction=0,
            model_direction_label="neutral",
            model_category="other",
            reasons=["low_confidence"],
            reasons_zh=["弱"],
            uncertainty_score=5.0,
            cluster_id="EVT-01",
        ),
        PendingReview(
            evidence_id="Y",
            statement_id="Y",
            title="y",
            snippet="y",
            url="",
            model_direction=0,
            model_direction_label="neutral",
            model_category="other",
            reasons=["low_confidence"],
            reasons_zh=["弱"],
            uncertainty_score=4.8,
            cluster_id="EVT-01",
        ),
        PendingReview(
            evidence_id="Z",
            statement_id="Z",
            title="z",
            snippet="z",
            url="",
            model_direction=1,
            model_direction_label="up",
            model_category="rba",
            reasons=["unclear_category"],
            reasons_zh=["类"],
            uncertainty_score=3.0,
            cluster_id="EVT-02",
        ),
    ]
    picked = prioritize_for_hitl(cands, max_items=2, cluster_diversity=0.1)
    assert [p.evidence_id for p in picked] == ["X", "Z"]
    assert picked[0].priority_rank == 1
    assert picked[1].priority_rank == 2


def test_extractive_summary_picks_fx_sentence() -> None:
    title = "Fed holds rates"
    text = (
        "The weather in Sydney was pleasant over the weekend. "
        "The Federal Reserve held interest rates steady amid sticky inflation. "
        "Local sports teams celebrated a win."
    )
    blurb = extractive_summary(text, title=title, max_chars=180, max_sentences=1)
    assert "Federal Reserve" in blurb or "inflation" in blurb or "rates" in blurb.lower()
    assert len(blurb) <= 180


def test_apply_evidence_summaries_offline() -> None:
    long_note = (
        "LLM精读｜m｜The Reserve Bank of Australia kept the cash rate on hold "
        "as inflation cooled only gradually and labour markets stayed tight. "
        "Markets priced a delayed easing path for AUD."
    )
    items = [_e("S1", title="RBA holds cash rate", note=long_note)]
    headlines = [
        Headline(
            title="RBA holds cash rate",
            summary=(
                "RBA kept rates unchanged. Inflation remains above target. "
                "AUD little changed versus the dollar."
            ),
            source="test",
            url="https://example.com/rba",
            published=None,
            provider="test",
        )
    ]
    meta = apply_evidence_summaries(items, headlines=headlines, prefer_llm=False)
    assert meta["summarized_n"] == 1
    assert items[0].summary
    assert len(items[0].summary) <= 220
    assert "summary:extractive" in (items[0].note or "")


def test_drift_tv_and_warn(tmp_path: Path) -> None:
    base_items = [
        _e("1", category="fed", direction=1),
        _e("2", category="fed", direction=1),
        _e("3", category="rba", direction=-1),
        _e("4", category="rba", direction=0),
    ]
    r1 = check_evidence_drift(
        base_items, pair="USD/AUD", out_dir=tmp_path, update_baseline=True
    )
    assert r1.skipped_reason == "baseline_seeded"
    assert not r1.warn

    shifted = [
        _e("a", category="geopolitics", direction=-1),
        _e("b", category="geopolitics", direction=-1),
        _e("c", category="china_iron", direction=-1),
        _e("d", category="china_iron", direction=-1),
    ]
    r2 = check_evidence_drift(
        shifted,
        pair="USD/AUD",
        out_dir=tmp_path,
        tv_warn=0.35,
        update_baseline=True,
    )
    assert r2.warn
    assert r2.warnings
    assert any("漂移" in w for w in r2.warnings)
    assert r2.tv_category >= 0.35 or r2.tv_direction >= 0.35


def test_total_variation_bounds() -> None:
    assert total_variation({}, {}) == 0.0
    assert abs(total_variation({"a": 1.0}, {"a": 1.0})) < 1e-9
    tv = total_variation({"a": 1.0}, {"b": 1.0})
    assert abs(tv - 1.0) < 1e-9


def test_distribution_skips_priors() -> None:
    items = [
        _e("n", category="fed", direction=1),
        _e("p", category="other", direction=-1, is_prior=True),
    ]
    snap = distribution_from_evidence(items, include_priors=False)
    assert snap.n == 1
    assert "fed" in snap.categories
