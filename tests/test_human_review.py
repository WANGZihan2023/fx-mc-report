"""Tests for human-in-the-loop uncertain evidence detection + overrides."""

from __future__ import annotations

from fx_report.model.human_review import (
    apply_review_overrides,
    detect_uncertain_evidence,
    normalize_review_choice,
)
from fx_report.model.weights import EvidenceItem
from fx_report.news.classify import rules_direction_guess
from fx_report.news.cluster import assign_event_clusters


def _e(
    eid: str,
    *,
    title: str = "headline",
    direction: int = 1,
    strength: float = 2.0,
    category: str = "fed",
    note: str = "",
    cluster_id: str = "",
    is_prior: bool = False,
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
        is_prior=is_prior,
    )


def test_detect_low_confidence_and_unclear_category() -> None:
    items = [
        _e("E1", strength=0.5, category="unclassified", direction=0),
        _e("E2", strength=2.5, category="rba", direction=1),
    ]
    pending = detect_uncertain_evidence(items, pair="AUD/USD", max_items=5)
    assert len(pending) == 1
    assert pending[0].evidence_id == "E1"
    assert "low_confidence" in pending[0].reasons
    assert "unclear_category" in pending[0].reasons
    assert any("置信度" in z for z in pending[0].reasons_zh)


def test_detect_cluster_direction_conflict() -> None:
    items = [
        _e(
            "A",
            title="Fed signals hawkish hold on rates",
            direction=1,
            category="fed",
            strength=2.0,
        ),
        _e(
            "B",
            title="Fed signals hawkish hold amid inflation",
            direction=-1,
            category="fed",
            strength=1.8,
        ),
    ]
    assign_event_clusters(items, enabled=True)
    # Same cluster with opposite dirs should flag members
    assert items[0].cluster_id and items[0].cluster_id == items[1].cluster_id
    pending = detect_uncertain_evidence(items, pair="USD/AUD", max_items=5)
    assert pending
    assert any("cluster_direction_conflict" in p.reasons for p in pending)


def test_detect_rules_llm_conflict() -> None:
    # LLM said AUD up; rules on "Fed hawkish" for AUD/USD typically USD up → AUD/USD down
    title = "Fed hawkish surprise lifts dollar versus aussie"
    rules_dir, _ = rules_direction_guess(title, "AUD/USD")
    # Whatever rules say, craft LLM item with opposite non-zero direction
    llm_dir = -1 if (rules_dir or 1) > 0 else 1
    if rules_dir is None:
        # Still exercise path: LLM note + force a conflict with synthetic rules via opposite
        llm_dir = 1
        # Use a title rules can parse
        title = "RBA hawkish hike supports Australian dollar"
        rules_dir, _ = rules_direction_guess(title, "AUD/USD")
        assert rules_dir is not None
        llm_dir = -int(rules_dir)

    items = [
        _e(
            "L1",
            title=title,
            direction=llm_dir,
            strength=2.2,
            category="fed" if "Fed" in title or "fed" in title.lower() else "rba",
            note="LLM精读｜deepseek｜rationale",
        )
    ]
    pending = detect_uncertain_evidence(items, pair="AUD/USD", max_items=5)
    assert any("rules_llm_conflict" in p.reasons for p in pending)


def test_cap_top_n_uncertain() -> None:
    items = [
        _e(f"U{i}", strength=0.3, category="other", direction=0) for i in range(8)
    ]
    pending = detect_uncertain_evidence(items, pair="USD/AUD", max_items=3)
    assert len(pending) == 3


def test_apply_review_overrides_up_down_skip() -> None:
    items = [
        _e("E1", direction=1, strength=0.4, category="other"),
        _e("E2", direction=-1, strength=2.0, category="rba"),
        _e("E3", direction=1, strength=2.0, category="fed"),
    ]
    out, meta = apply_review_overrides(
        items,
        {"E1": "利空", "E2": "skip", "E3": "neutral"},
    )
    assert meta["n_overridden"] == 2
    assert meta["n_skipped"] == 1
    by_id = {e.id: e for e in out}
    assert by_id["E1"].direction == -1
    assert "human_review" in (by_id["E1"].note or "")
    assert by_id["E2"].direction == -1  # unchanged
    assert by_id["E3"].direction == 0


def test_normalize_review_choice_zh() -> None:
    assert normalize_review_choice("利多") == "up"
    assert normalize_review_choice("利空") == "down"
    assert normalize_review_choice("中性") == "neutral"
    assert normalize_review_choice("跳过") == "skip"
    assert normalize_review_choice("") == ""


def test_checkpoint_roundtrip_pending() -> None:
    from fx_report.model.human_review import PendingReview
    from fx_report.pipeline import PipelineCheckpoint
    from fx_report.market.fetch_data import MarketSnapshot
    from fx_report.model.weights import ModelWeights

    market = MarketSnapshot(
        asof="2026-07-28",
        pair="AUD/USD",
        spot=0.65,
        provider_raw=0.65,
        sigma_daily=0.01,
        sigma_annual=0.16,
        mean_daily_return=0.0,
        n_returns=60,
        lookback_days=60,
        history_start="2026-01-01",
        history_end="2026-07-28",
        source="test",
        brent=None,
        dxy_proxy=None,
    )
    pending = [
        PendingReview(
            evidence_id="E1",
            statement_id="E1",
            title="weak item",
            snippet="weak item",
            url="",
            model_direction=0,
            model_direction_label="neutral",
            model_category="other",
            reasons=["low_confidence"],
            reasons_zh=["置信度偏低（强度偏弱）"],
            uncertainty_score=2.0,
        )
    ]
    ev = [_e("E1", strength=0.4, category="other", direction=0)]
    cp = PipelineCheckpoint(
        pair="AUD/USD",
        bullish_currency="AUD",
        stage_log=["t"],
        info_needs=[],
        market=market,
        statements=[],
        headlines=[],
        evidence=ev,
        news_meta={"human_review": {"n_pending": 1}},
        weights=ModelWeights(n_sims=1000),
        pending_reviews=pending,
    )
    raw = cp.to_session_dict()
    back = PipelineCheckpoint.from_session_dict(raw)
    assert back.pair == "AUD/USD"
    assert len(back.pending_reviews) == 1
    assert back.pending_reviews[0].evidence_id == "E1"
    assert len(back.evidence) == 1
