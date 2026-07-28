"""Order / 单子 PDF parsing — synthetic text fixtures (no copyrighted PDFs)."""

from __future__ import annotations

import io

import pytest

from fx_report.order_pdf import (
    ORDER_PDF_ALWAYS_MANUAL,
    ORDER_PDF_FILLABLE,
    extract_pdf_text,
    order_pdf_from_dict,
    parse_order_pdf,
    parse_order_text,
    preview_lines,
)
from fx_report.ui.ux_helpers import START_REQUIRED_LABELS


SAMPLE_ORDER = """
外汇结构单子
货币对：USD/AUD
看涨：USD
现价 Spot: 1.4280
Barrier: 1.4500
Strike: 1.4800
期限：3个月
备注：障碍与行权均在现价上方
"""


SAMPLE_CN = """
单子摘要
澳元兑美元
看多 澳元
障碍价 0.7100
行权价 0.7300
"""


SAMPLE_PCT_CUTS = """
Pair AUDUSD
看涨 AUD
分档切点: 0, 2, 4, 6
"""


def test_parse_usdaud_fills_pair_bullish_absolute_mode():
    r = parse_order_text(SAMPLE_ORDER)
    assert r.ok
    assert r.pair == "USD/AUD"
    assert r.pair_mode == "目录"
    assert r.bullish_currency == "USD"
    assert r.barrier == pytest.approx(1.45)
    assert r.strike == pytest.approx(1.48)
    assert r.spot == pytest.approx(1.428)
    assert r.bucket_mode == "绝对价位"
    assert "货币对" in r.filled
    assert "看涨货币" in r.filled
    assert "分档边界方式" in r.filled
    # Engine / calib / HITL never auto-filled
    for label in ORDER_PDF_ALWAYS_MANUAL:
        assert label in r.still_needed
    assert r.tenor == "3个月"


def test_parse_chinese_pair_and_bearish_infer():
    r = parse_order_text(SAMPLE_CN)
    assert r.ok
    assert r.pair == "AUD/USD"
    assert r.bullish_currency == "AUD"
    assert r.barrier == pytest.approx(0.71)
    assert r.strike == pytest.approx(0.73)


def test_parse_relative_pct_cuts():
    r = parse_order_text(SAMPLE_PCT_CUTS)
    assert r.ok
    assert r.pair == "AUD/USD"
    assert r.bullish_currency == "AUD"
    assert r.bucket_mode == "相对现价"
    assert r.bucket_pct_cuts == [0.0, 2.0, 4.0, 6.0]


def test_parse_empty_and_garbage():
    assert parse_order_text("").ok is False
    assert "空" in (parse_order_text("").error or "")
    bad = parse_order_text("今天天气不错，没有汇率")
    assert bad.ok is False
    assert "未能" in (bad.error or "")


def test_bearish_infers_other_leg():
    text = "EUR/USD\n看跌 EUR\nBarrier: 1.10\nStrike: 1.12\n"
    r = parse_order_text(text)
    assert r.ok
    assert r.pair == "EUR/USD"
    assert r.bullish_currency == "USD"


def test_preview_and_roundtrip_dict():
    r = parse_order_text(SAMPLE_ORDER)
    lines = preview_lines(r)
    assert any("已自动填入" in x for x in lines)
    assert any("仍需你选择" in x for x in lines)
    back = order_pdf_from_dict(r.to_dict())
    assert back is not None
    assert back.pair == r.pair
    assert back.bullish_currency == r.bullish_currency


def test_fillable_vs_manual_labels():
    assert "货币对" in ORDER_PDF_FILLABLE
    assert START_REQUIRED_LABELS["peak_engine"] in ORDER_PDF_ALWAYS_MANUAL
    assert START_REQUIRED_LABELS["use_calibrated"] in ORDER_PDF_ALWAYS_MANUAL
    assert START_REQUIRED_LABELS["human_review"] in ORDER_PDF_ALWAYS_MANUAL


def test_extract_and_parse_tiny_pdf():
    pytest.importorskip("pypdf")
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
    except Exception:
        pytest.skip("reportlab not available for PDF fixture")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(72, 720, "USD/AUD bullish: USD Barrier: 1.45 Strike: 1.48")
    c.save()
    data = buf.getvalue()

    text = extract_pdf_text(data)
    assert "USD" in text.upper() or "AUD" in text.upper()
    pr = parse_order_pdf(data, use_llm=False)
    assert pr.ok
    assert pr.pair == "USD/AUD"
    assert pr.bullish_currency == "USD"
