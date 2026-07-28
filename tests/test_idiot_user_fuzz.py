"""
Idiot-user / UX abuse suite — offline, no NewsAPI.

Exercises start-settings, password, order-PDF garbage, HITL skip paths,
contradictory MC settings, empty-run honesty. Pure helpers + parsers only.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from fx_report.market.pairs import get_pair, resolve_pair_for_bullish
from fx_report.model.human_review import (
    apply_review_overrides,
    detect_uncertain_evidence,
    normalize_review_choice,
)
from fx_report.model.weights import EvidenceItem, default_weights, resolve_bucket_edges
from fx_report.order_pdf import (
    ORDER_PDF_ALWAYS_MANUAL,
    parse_order_pdf,
    parse_order_text,
    preview_lines,
)
from fx_report.pipeline import step4_evaluate_impact
from fx_report.ui.ux_helpers import (
    START_CHOICE_PLACEHOLDER,
    bb_jump_compensate_warning,
    format_missing_start_message,
    missing_start_choices,
    password_accepted,
    app_password_expected,
)


def _e(eid: str, **kw) -> EvidenceItem:
    defaults = dict(
        title="t",
        direction=0,
        strength=0.4,
        freshness=1.0,
        unpriced=0.5,
        category="other",
        strength_label="SLIGHT",
        statement_id=eid,
    )
    defaults.update(kw)
    return EvidenceItem(id=eid, **defaults)


# ---------------------------------------------------------------------------
# Password abuse
# ---------------------------------------------------------------------------


def test_password_wrong_variants():
    exp = app_password_expected(environ={})
    assert password_accepted("uniocean ", exp) is False  # trailing space
    assert password_accepted("Uniocean", exp) is False  # case
    assert password_accepted("uniocean\n", exp) is False
    assert password_accepted(None, exp) is False
    assert password_accepted("", exp) is False
    assert password_accepted("admin", exp) is False
    assert password_accepted("uniocean", exp) is True


def test_password_env_blank_falls_through():
    assert app_password_expected(environ={"APP_PASSWORD": "", "FX_REPORT_PASSWORD": ""}) == "uniocean"


# ---------------------------------------------------------------------------
# Start settings: no selections / contradictory / partial
# ---------------------------------------------------------------------------


def test_no_selections_dialog_message():
    missing = missing_start_choices({})
    msg = format_missing_start_message(missing)
    assert msg.startswith("你还没有选择：")
    assert "货币对" in msg
    assert "峰值引擎" in msg
    assert "是否使用校准参数" in msg
    assert "不确定证据是否人工确认" in msg


def test_all_placeholders_still_missing():
    choices = {k: START_CHOICE_PLACEHOLDER for k in (
        "pair", "bullish_currency", "peak_engine", "bucket_mode"
    )}
    choices["use_calibrated"] = None
    choices["human_review"] = None
    missing = missing_start_choices(choices)
    assert len(missing) >= 6


def test_contradictory_bullish_not_in_pair_raises():
    spec = get_pair("USD/AUD")
    with pytest.raises(ValueError):
        resolve_pair_for_bullish(spec, "EUR")
    with pytest.raises(ValueError):
        resolve_pair_for_bullish(spec, "JPY")


def test_contradictory_bb_plus_jump_compensate_warns():
    msg = bb_jump_compensate_warning(
        peak_engine="brownian_bridge", jump_compensate=True
    )
    assert msg is not None
    assert "jump_compensate" in msg


def test_invalid_engine_string_counts_missing():
    missing = missing_start_choices(
        {
            "pair": "USD/AUD",
            "bullish_currency": "USD",
            "peak_engine": "monte_carlo_max",  # nonsense
            "use_calibrated": True,
            "human_review": False,
            "bucket_mode": "相对现价",
        }
    )
    assert "峰值引擎" in missing


def test_bool_fields_reject_string_yes():
    missing = missing_start_choices(
        {
            "pair": "USD/AUD",
            "bullish_currency": "USD",
            "peak_engine": "path_max",
            "use_calibrated": "yes",
            "human_review": "true",
            "bucket_mode": "绝对价位",
        }
    )
    assert "是否使用校准参数" in missing
    assert "不确定证据是否人工确认" in missing


# ---------------------------------------------------------------------------
# Order PDF abuse (offline, no LLM)
# ---------------------------------------------------------------------------


def test_bad_pdf_bytes_returns_ok_false():
    r = parse_order_pdf(b"not-a-pdf-at-all%%%", use_llm=False)
    assert r.ok is False
    assert r.error
    assert "PDF" in r.error or "pdf" in r.error.lower() or "读取" in r.error or "解析" in r.error


def test_empty_pdf_bytes():
    r = parse_order_pdf(b"", use_llm=False)
    assert r.ok is False


def test_empty_order_text():
    r = parse_order_text("")
    assert r.ok is False or (r.ok and r.still_needed)
    lines = preview_lines(r)
    assert lines


def test_garbage_order_text_still_needs_manual():
    r = parse_order_text("asdf qwer zxcv 12345 !@#$%")
    # Must not invent pair; failed parse still lists always-manual fields
    assert r.pair is None
    assert r.ok is False
    assert r.error
    for label in ORDER_PDF_ALWAYS_MANUAL:
        assert label in r.still_needed


def test_partial_order_never_fills_engine_calib_hitl():
    r = parse_order_text(
        "货币对 USD/AUD\n看涨：USD\nBarrier: 1.50\nStrike: 1.45\n"
    )
    assert r.ok
    assert r.pair == "USD/AUD"
    for label in ORDER_PDF_ALWAYS_MANUAL:
        assert label in r.still_needed
        assert label not in r.filled


# ---------------------------------------------------------------------------
# Empty run / evidence honesty
# ---------------------------------------------------------------------------


def test_empty_run_no_headlines_s_honest():
    from fx_report.market.fetch_data import MarketSnapshot

    spec = get_pair("USD/AUD")
    base = default_weights(spec)
    market = MarketSnapshot(
        asof="test",
        pair=spec.pair,
        spot=1.55,
        provider_raw=1.55,
        sigma_daily=0.006,
        sigma_annual=0.095,
        mean_daily_return=0.0,
        n_returns=60,
        lookback_days=60,
        history_start="2020-01-01",
        history_end="2026-01-01",
        source="test",
        brent=None,
        dxy_proxy=None,
        notes=[],
        ret_1d=0.0,
        ret_5d=0.0,
    )
    ev, meta = step4_evaluate_impact(
        [],
        spec,
        market,
        base,
        mode="rules",
        template_policy="off",
    )
    assert ev == []
    assert meta.get("evidence_quality") == "news_empty_no_prior"
    assert meta.get("fallback_templates") is False
    # Drift / summary meta present even on empty
    assert "summary_meta" in meta
    assert "drift_meta" in meta


# ---------------------------------------------------------------------------
# HITL skip / empty choices / weird labels
# ---------------------------------------------------------------------------


def test_hitl_all_skip_keeps_model():
    items = [_e("E1", direction=1, strength=0.3), _e("E2", direction=-1, strength=0.2)]
    out, meta = apply_review_overrides(items, {"E1": "skip", "E2": "跳过"})
    assert meta["n_overridden"] == 0
    assert meta["n_skipped"] == 2
    assert out[0].direction == 1
    assert out[1].direction == -1


def test_hitl_empty_choices_noop():
    items = [_e("E1", direction=1)]
    out, meta = apply_review_overrides(items, {})
    assert meta["n_overridden"] == 0
    assert out[0].direction == 1


def test_hitl_garbage_choice_ignored():
    assert normalize_review_choice("maybe") == ""
    assert normalize_review_choice("🚀") == ""
    items = [_e("E1", direction=1)]
    out, meta = apply_review_overrides(items, {"E1": "maybe", "E1b": "🚀"})
    assert meta["n_overridden"] == 0
    assert out[0].direction == 1


def test_hitl_detect_then_skip_all_path():
    items = [
        _e("U1", strength=0.2, category="unclassified", direction=0),
        _e("U2", strength=0.3, category="other", direction=0),
    ]
    pending = detect_uncertain_evidence(items, pair="USD/AUD", max_items=5)
    assert pending
    choices = {p.evidence_id: "skip" for p in pending}
    out, meta = apply_review_overrides(items, choices)
    assert meta["n_skipped"] == len(pending)
    assert all(e.direction == items[i].direction for i, e in enumerate(out))


# ---------------------------------------------------------------------------
# Bucket / weights edge abuse
# ---------------------------------------------------------------------------


def test_bucket_edges_resolve_with_extreme_spot():
    w = default_weights("USD/AUD")
    edges = resolve_bucket_edges(w, 1e-6)  # absurd tiny spot
    assert len(edges) == 4
    assert all(isinstance(x, float) for x in edges)


def test_weights_jump_model_none_vs_path_max_ok():
    w = default_weights("USD/AUD")
    w2 = replace(w, jump_model="none", peak_engine="path_max")
    assert w2.jump_model == "none"
