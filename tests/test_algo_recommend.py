"""Tests for auto algorithm recommendation priority."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fx_report.model.algo_recommend import (
    DEFAULT_CLUSTER_METHOD,
    DEFAULT_JUMP_MODEL,
    DEFAULT_PEAK_ENGINE,
    DEFAULT_VARIANCE_REDUCTION,
    SOURCE_CALIBRATED,
    SOURCE_ENGINE_COMPARE,
    SOURCE_PRODUCT_DEFAULT,
    format_recommend_audit_zh,
    is_simple_setup_mode,
    overall_winner_from_summary,
    recommend_algorithms,
    start_keys_for_mode,
)
from fx_report.ui.ux_helpers import missing_start_choices


def test_calibrated_beats_engine_compare(tmp_path: Path) -> None:
    """Priority 1: calib peak/jump/VR win over engine_compare winner."""
    calib = tmp_path / "calibrated_params_FAKEUSD.json"
    calib.write_text(
        json.dumps(
            {
                "pair": "FAKE/USD",
                "params": {
                    "peak_engine": "path_max",
                    "jump_model": "merton",
                    "recommended_variance_reduction": "antithetic",
                    "score_to_mu_a": 0.01,
                    "scenarios": [],
                },
            }
        ),
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "pair": "FAKE/USD",
                "rows": [{"winner": "C"}],
                "combos": {
                    "C": {
                        "peak_engine": "brownian_bridge",
                        "jump_model": "none",
                        "variance_reduction": "antithetic",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    rec = recommend_algorithms(
        "FAKE/USD",
        calibrated_path=calib,
        engine_compare_path=summary,
    )
    assert rec.source == SOURCE_CALIBRATED
    assert rec.peak_engine == "path_max"
    assert rec.jump_model == "merton"
    assert rec.variance_reduction == "antithetic"
    assert rec.use_calibrated is True
    assert rec.human_review is True
    assert rec.cluster_method == DEFAULT_CLUSTER_METHOD
    assert any("校准" in r for r in rec.reasons)


def test_engine_compare_when_no_calib_algo_fields(tmp_path: Path) -> None:
    """Priority 2: no calib algo fields → use summary winner for matching pair."""
    # Calib file exists but without peak/jump/VR → skip priority 1
    calib = tmp_path / "calibrated_params_ZZZUSD.json"
    calib.write_text(
        json.dumps(
            {
                "pair": "ZZZ/USD",
                "params": {
                    "score_to_mu_a": 0.01,
                    "scenarios": [],
                },
            }
        ),
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "pair": "ZZZ/USD",
                "rows": [
                    {"winner": "C"},
                    {"winner": "C"},
                    {"winner": "A"},
                ],
            }
        ),
        encoding="utf-8",
    )
    rec = recommend_algorithms(
        "ZZZ/USD",
        calibrated_path=calib,
        engine_compare_path=summary,
    )
    assert rec.source == SOURCE_ENGINE_COMPARE
    assert rec.peak_engine == "brownian_bridge"
    assert rec.jump_model == "none"
    assert rec.variance_reduction == "antithetic"
    assert rec.use_calibrated is True  # file exists
    assert any("engine_compare" in r for r in rec.reasons)


def test_engine_compare_pair_mismatch_falls_to_default(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"pair": "USD/AUD", "rows": [{"winner": "C"}]}),
        encoding="utf-8",
    )
    rec = recommend_algorithms(
        "EUR/USD",
        calibrated_path=tmp_path / "missing.json",
        engine_compare_path=summary,
    )
    # EUR/USD has bundled calib with peak_engine → priority 1 from resolve.
    # Force no calib by pointing calibrated_path to missing and not using resolve:
    # recommend_algorithms with missing calibrated_path sets calib_path=None only when
    # the explicit path doesn't exist — but then it still calls resolve for None path.
    # When calibrated_path is given and missing → calib_path=None, and it does NOT
    # fall back to resolve. Good.
    assert rec.use_calibrated is False
    assert rec.source == SOURCE_PRODUCT_DEFAULT
    assert rec.peak_engine == DEFAULT_PEAK_ENGINE
    assert rec.jump_model == DEFAULT_JUMP_MODEL
    assert rec.variance_reduction == DEFAULT_VARIANCE_REDUCTION


def test_product_defaults_no_signals(tmp_path: Path) -> None:
    rec = recommend_algorithms(
        "NO/PAIR",
        calibrated_path=tmp_path / "nope.json",
        engine_compare_path=tmp_path / "no_summary.json",
    )
    assert rec.source == SOURCE_PRODUCT_DEFAULT
    assert rec.peak_engine == "path_max"
    assert rec.jump_model == "merton"
    assert rec.variance_reduction == "antithetic"
    assert rec.cluster_method == "jaccard"
    assert rec.use_calibrated is False
    assert rec.human_review is True


def test_overall_winner_majority_and_tie() -> None:
    assert (
        overall_winner_from_summary(
            {"pair": "USD/AUD", "rows": [{"winner": "A"}, {"winner": "A"}, {"winner": "C"}]},
            pair="USD/AUD",
        )
        == "A"
    )
    assert (
        overall_winner_from_summary(
            {"pair": "USD/AUD", "rows": [{"winner": "A"}, {"winner": "C"}]},
            pair="USD/AUD",
        )
        is None
    )
    assert (
        overall_winner_from_summary(
            {"pair": "USD/AUD", "overall_winner": "C", "rows": [{"winner": "A"}]},
            pair="USD/AUD",
        )
        == "C"
    )
    assert (
        overall_winner_from_summary(
            {"pair": "EUR/USD", "rows": [{"winner": "A"}]},
            pair="USD/AUD",
        )
        is None
    )


def test_calib_vr_fallback_to_product_default(tmp_path: Path) -> None:
    calib = tmp_path / "c.json"
    calib.write_text(
        json.dumps(
            {
                "params": {
                    "peak_engine": "brownian_bridge",
                    "jump_model": "none",
                }
            }
        ),
        encoding="utf-8",
    )
    rec = recommend_algorithms(
        "X/Y",
        calibrated_path=calib,
        engine_compare_path=tmp_path / "missing.json",
    )
    assert rec.source == SOURCE_CALIBRATED
    assert rec.peak_engine == "brownian_bridge"
    assert rec.jump_model == "none"
    assert rec.variance_reduction == DEFAULT_VARIANCE_REDUCTION
    assert any("recommended_variance_reduction" in r for r in rec.reasons)


def test_audit_zh_mentions_system_recommend(tmp_path: Path) -> None:
    rec = recommend_algorithms(
        "NO/PAIR",
        calibrated_path=tmp_path / "x.json",
        engine_compare_path=tmp_path / "y.json",
    )
    text = format_recommend_audit_zh(rec)
    assert "本次算法由系统推荐" in text
    assert "产品默认" in text


def test_simple_mode_missing_keys() -> None:
    assert start_keys_for_mode("simple") == ("pair", "bullish_currency")
    assert start_keys_for_mode("简洁（推荐）", include_bucket=True) == (
        "pair",
        "bullish_currency",
        "bucket_mode",
    )
    assert is_simple_setup_mode("simple")
    assert not is_simple_setup_mode("expert")

    missing = missing_start_choices(
        {"pair": "USD/AUD", "bullish_currency": "USD"},
        setup_mode="simple",
        include_bucket=False,
    )
    assert missing == []

    missing_run = missing_start_choices(
        {"pair": "USD/AUD", "bullish_currency": "USD"},
        setup_mode="simple",
        include_bucket=True,
    )
    assert "分档边界方式" in missing_run

    # Expert still requires algo fields
    missing_ex = missing_start_choices(
        {"pair": "USD/AUD", "bullish_currency": "USD"},
        setup_mode="expert",
        include_bucket=False,
    )
    assert "峰值引擎" in missing_ex


def test_bundled_usdaud_uses_calibrated() -> None:
    """Real bundle: USD/AUD calib has peak_engine → source calibrated."""
    rec = recommend_algorithms("USD/AUD")
    assert rec.source == SOURCE_CALIBRATED
    assert rec.use_calibrated is True
    assert rec.peak_engine in {"path_max", "brownian_bridge"}
    assert rec.human_review is True
