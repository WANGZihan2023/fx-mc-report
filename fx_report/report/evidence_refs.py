"""Shared helpers for Evidence Base / References support-quote + link display."""

from __future__ import annotations

import re
from typing import Any

from fx_report.model.weights import EvidenceItem
from fx_report.news.urls import is_fragile_url, is_http_url
from fx_report.report.strings import L, normalize_report_lang

_URL_IN_TEXT = re.compile(r"https?://\S+")
_DEAD_MARK = "链接可能失效"

DEFAULT_QUOTE_CHARS = 220

# Back-compat aliases (Chinese defaults)
LABEL_SUPPORT_ZH = "支撑引用"
LABEL_SUPPORT_WEAK_ZH = "支撑引用（弱）"
LABEL_STANCE_ZH = "总结"


def evidence_source_url(e: EvidenceItem) -> str:
    """Prefer EvidenceItem.url; else first http URL in note."""
    u = (getattr(e, "url", None) or "").strip()
    if is_http_url(u):
        return u
    note = getattr(e, "note", None) or ""
    m = _URL_IN_TEXT.search(note)
    if m:
        return m.group(0).rstrip("｜|)")
    return ""


def _clean_quote_candidate(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = _URL_IN_TEXT.sub("", t)
    t = re.sub(
        r"[｜|]\s*(prior_template|downweighted|summary:\w+|support:\w+|stance:\w+)\s*",
        " ",
        t,
        flags=re.I,
    )
    t = t.replace(_DEAD_MARK, "")
    t = re.sub(r"[｜|]{2,}", "｜", t)
    t = re.sub(r"\s{2,}", " ", t).strip(" ｜|;,.—-")
    return t


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def support_label(quality: str, *, lang: str | None = None) -> str:
    lang = normalize_report_lang(lang)
    if quality == "support":
        return L("support", lang=lang)
    return L("support_weak", lang=lang)


def evidence_support_meta(
    e: EvidenceItem,
    *,
    max_chars: int = DEFAULT_QUOTE_CHARS,
    lang: str | None = None,
) -> dict[str, Any]:
    """
    Stance-aligned support quote for References / Evidence cards.

    Fallback order (never invents text):
      1. explicit ``support_quote`` field
      2. on-the-fly stance extract from summary / note / title
      3. title (last resort)

    Returns:
      quote, quality ('support'|'weak'|'title'), label_zh, label
    """
    lang = normalize_report_lang(lang)
    quality = (getattr(e, "support_quote_quality", None) or "").strip()
    raw_sq = _clean_quote_candidate(getattr(e, "support_quote", None) or "")
    if len(raw_sq) >= 12:
        q = quality if quality in {"support", "weak", "title"} else "support"
        label = support_label(q, lang=lang)
        return {
            "quote": _truncate(raw_sq, max_chars),
            "quality": q,
            "label_zh": label if lang == "zh" else support_label(q, lang="zh"),
            "label": label,
        }

    # Lazy extract from fields already on the item (historical / pre-summary items)
    try:
        from fx_report.news.summarize import extract_support_quote

        blob_parts: list[str] = []
        for field in ("summary", "note", "title"):
            cleaned = _clean_quote_candidate(getattr(e, field, None) or "")
            if cleaned and len(cleaned) >= 12:
                # Skip pure metadata notes
                if cleaned.lower().startswith("source_tier=") and len(cleaned) < 80:
                    continue
                blob_parts.append(cleaned)
        blob = "\n".join(blob_parts)
        quote, q = extract_support_quote(
            blob,
            title=getattr(e, "title", None) or "",
            direction=int(getattr(e, "direction", 0) or 0),
            category=getattr(e, "category", None) or "",
            max_chars=max_chars,
        )
        quote = _clean_quote_candidate(quote)
        if quote:
            label = support_label(q, lang=lang)
            return {
                "quote": _truncate(quote, max_chars),
                "quality": q,
                "label_zh": label if lang == "zh" else support_label(q, lang="zh"),
                "label": label,
            }
    except Exception:
        pass

    # Last resort: title / summary dump (legacy path)
    for raw in (
        getattr(e, "summary", None) or "",
        getattr(e, "note", None) or "",
        getattr(e, "title", None) or "",
    ):
        cleaned = _clean_quote_candidate(str(raw))
        if len(cleaned) < 12:
            continue
        if cleaned.lower().startswith("source_tier=") and len(cleaned) < 80:
            continue
        label = support_label("weak", lang=lang)
        return {
            "quote": _truncate(cleaned, max_chars),
            "quality": "weak",
            "label_zh": label if lang == "zh" else support_label("weak", lang="zh"),
            "label": label,
        }
    title = _clean_quote_candidate(getattr(e, "title", None) or "")
    label = support_label("title", lang=lang)
    if not title:
        return {
            "quote": "",
            "quality": "title",
            "label_zh": label if lang == "zh" else support_label("title", lang="zh"),
            "label": label,
        }
    return {
        "quote": _truncate(title, max_chars),
        "quality": "title",
        "label_zh": label if lang == "zh" else support_label("title", lang="zh"),
        "label": label,
    }


def evidence_stance_summary_meta(
    e: EvidenceItem,
    *,
    lang: str | None = None,
    max_chars: int = 280,
) -> dict[str, Any]:
    """Report-lang 总结 for References — distinct from verbatim support quote."""
    lang = normalize_report_lang(lang)
    text = _clean_quote_candidate(getattr(e, "stance_summary", None) or "")
    if not text:
        # Soft fallback: extractive summary field (not a fake quote)
        text = _clean_quote_candidate(getattr(e, "summary", None) or "")
    if not text:
        return {"text": "", "label": L("stance_summary", lang=lang)}
    return {
        "text": _truncate(text, max_chars),
        "label": L("stance_summary", lang=lang),
    }


def evidence_quote(e: EvidenceItem, *, max_chars: int = DEFAULT_QUOTE_CHARS) -> str:
    """
    Short support excerpt for References / Evidence cards.

    Prefers stance-aligned ``support_quote``; never invents text beyond
    fields already on the evidence item.
    """
    return str(evidence_support_meta(e, max_chars=max_chars).get("quote") or "")


def evidence_link_meta(e: EvidenceItem, *, lang: str | None = None) -> dict[str, Any]:
    """
    Display metadata for a reference link.

    Returns:
      url: str (empty if none / cleared after dead check)
      dead_marked: bool
      fragile: bool
      label_zh: optional warning suffix for UI
      label: localized warning
    """
    lang = normalize_report_lang(lang)
    note = getattr(e, "note", None) or ""
    dead = _DEAD_MARK in note
    url = evidence_source_url(e)
    fragile = bool(url) and is_fragile_url(url)
    label = ""
    if dead:
        label = L("link_dead", lang=lang)
    elif fragile:
        label = L("link_fragile", lang=lang)
    return {
        "url": "" if dead else url,
        "dead_marked": dead,
        "fragile": fragile,
        "label_zh": label if lang == "zh" else (L("link_dead", lang="zh") if dead else (
            L("link_fragile", lang="zh") if fragile else ""
        )),
        "label": label,
    }


def format_reference_markdown_row(
    e: EvidenceItem,
    *,
    index: int | None = None,
    max_quote: int = DEFAULT_QUOTE_CHARS,
    lang: str | None = None,
) -> str:
    """One Markdown bullet: id · 总结 · 支撑引用 · optional URL."""
    lang = normalize_report_lang(lang)
    meta = evidence_support_meta(e, max_chars=max_quote, lang=lang)
    stance = evidence_stance_summary_meta(e, lang=lang)
    link = evidence_link_meta(e, lang=lang)
    eid = (e.id or "").strip() or "?"
    prefix = f"{index}. " if index is not None else ""
    bits = [f"**{eid}**"]
    if stance.get("text"):
        bits.append(f"{stance['label']}：{stance['text']}")
    quote = meta.get("quote") or ""
    if quote:
        bits.append(f"{meta.get('label') or L('support', lang=lang)} 「{quote}」")
    line = prefix + " · ".join(bits)
    if link["url"]:
        line += f" — {link['url']}"
    elif link.get("label"):
        line += f" — _{link['label']}_"
    return line
