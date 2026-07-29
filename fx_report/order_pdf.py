"""Backward-compatible re-exports. Prefer ``fx_report.order_doc``."""

from __future__ import annotations

from fx_report.order_doc import (  # noqa: F401
    ORDER_DOC_ALWAYS_MANUAL,
    ORDER_DOC_FILLABLE,
    ORDER_PDF_ALWAYS_MANUAL,
    ORDER_PDF_FILLABLE,
    OrderDocParse,
    OrderPdfParse,
    extract_image_text,
    extract_order_text,
    extract_pdf_text,
    llm_assist_order_text,
    llm_vision_order_text,
    order_doc_from_dict,
    order_pdf_from_dict,
    parse_order_document,
    parse_order_pdf,
    parse_order_text,
    preview_lines,
    sniff_document_kind,
)

__all__ = [
    "ORDER_DOC_ALWAYS_MANUAL",
    "ORDER_DOC_FILLABLE",
    "ORDER_PDF_ALWAYS_MANUAL",
    "ORDER_PDF_FILLABLE",
    "OrderDocParse",
    "OrderPdfParse",
    "extract_image_text",
    "extract_order_text",
    "extract_pdf_text",
    "llm_assist_order_text",
    "llm_vision_order_text",
    "order_doc_from_dict",
    "order_pdf_from_dict",
    "parse_order_document",
    "parse_order_pdf",
    "parse_order_text",
    "preview_lines",
    "sniff_document_kind",
]
