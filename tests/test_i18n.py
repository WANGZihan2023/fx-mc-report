"""Tests for Streamlit UI i18n helper (zh / en)."""

from __future__ import annotations

from fx_report.ui.i18n import (
    CHOICE_PLACEHOLDERS,
    DEFAULT_LANG,
    LANG_EN,
    LANG_ZH,
    PLACEHOLDER_EN,
    PLACEHOLDER_ZH,
    choice_placeholder,
    format_missing_message,
    normalize_lang,
    start_field_label,
    t,
)
from fx_report.ui.ux_helpers import (
    format_missing_start_message,
    is_unset_choice,
    missing_start_choices,
)


def test_normalize_lang():
    assert normalize_lang(None) == LANG_ZH
    assert normalize_lang("zh") == LANG_ZH
    assert normalize_lang("EN") == LANG_EN
    assert normalize_lang("english") == LANG_EN
    assert normalize_lang("中文") == LANG_ZH
    assert normalize_lang("nope") == DEFAULT_LANG


def test_t_zh_en_and_fallback():
    assert t("main.run", lang=LANG_ZH) == "运行分析"
    assert t("main.run", lang=LANG_EN) == "Run analysis"
    assert t("auth.enter", lang=LANG_EN) == "Enter"
    # unknown key falls back to key itself (via zh miss)
    assert t("does.not.exist", lang=LANG_EN) == "does.not.exist"


def test_t_format_kwargs():
    s = t("side.label_audit.n", lang=LANG_EN, n=3)
    assert "3" in s
    s_zh = t("audit.filled", lang=LANG_ZH, n=1, total=2, path="out.csv")
    assert "1/2" in s_zh
    assert "out.csv" in s_zh


def test_start_field_labels_localized():
    assert start_field_label("pair", lang=LANG_ZH) == "货币对"
    assert start_field_label("pair", lang=LANG_EN) == "Currency pair"
    assert start_field_label("bullish_currency", lang=LANG_EN) == "Bullish currency"


def test_choice_placeholders():
    assert choice_placeholder(LANG_ZH) == PLACEHOLDER_ZH
    assert choice_placeholder(LANG_EN) == PLACEHOLDER_EN
    assert PLACEHOLDER_ZH in CHOICE_PLACEHOLDERS
    assert PLACEHOLDER_EN in CHOICE_PLACEHOLDERS
    assert is_unset_choice(PLACEHOLDER_EN) is True
    assert is_unset_choice(PLACEHOLDER_ZH) is True


def test_missing_start_choices_lang_en():
    missing = missing_start_choices({}, lang=LANG_EN)
    assert "Currency pair" in missing
    assert "Bullish currency" in missing
    assert "货币对" not in missing
    msg = format_missing_start_message(missing, lang=LANG_EN)
    assert msg.startswith("You still need to choose:")
    assert "Currency pair" in msg


def test_format_missing_message_empty():
    assert format_missing_message([], lang=LANG_ZH) == "你还没有选择：必选项"
    assert format_missing_message([], lang=LANG_EN) == "You still need to choose: required fields"
