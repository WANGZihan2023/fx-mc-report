"""Shared helpers for Evidence Base / References quote + link display."""

from __future__ import annotations

import re
from typing import Any

from fx_report.model.weights import EvidenceItem
from fx_report.news.urls import is_fragile_url, is_http_url

_URL_IN_TEXT = re.compile(r"https?://\S+")
_DEAD_MARK = "链接可能失效"

DEFAULT_QUOTE_CHARS = 220


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
    t = re.sub(r"[｜|]\s*(prior_template|downweighted|summary:\w+)\s*", " ", t, flags=re.I)
    t = t.replace(_DEAD_MARK, "")
    t = re.sub(r"[｜|]{2,}", "｜", t)
    t = re.sub(r"\s{2,}", " ", t).strip(" ｜|;,.—-")
    return t


def evidence_quote(e: EvidenceItem, *, max_chars: int = DEFAULT_QUOTE_CHARS) -> str:
    """
    Short quoted excerpt for References / Evidence cards.

    Preference: summary (extractive/LLM) → note without URL → title.
    Does not invent text beyond fields already on the evidence item.
    """
    for raw in (
        getattr(e, "summary", None) or "",
        getattr(e, "note", None) or "",
        getattr(e, "title", None) or "",
    ):
        cleaned = _clean_quote_candidate(str(raw))
        if len(cleaned) < 12:
            continue
        # Skip pure metadata notes
        if cleaned.lower().startswith("source_tier=") and len(cleaned) < 80:
            continue
        if len(cleaned) > max_chars:
            cleaned = cleaned[: max_chars - 1].rstrip() + "…"
        return cleaned
    title = _clean_quote_candidate(getattr(e, "title", None) or "")
    if not title:
        return ""
    return title[:max_chars]


def evidence_link_meta(e: EvidenceItem) -> dict[str, Any]:
    """
    Display metadata for a reference link.

    Returns:
      url: str (empty if none / cleared after dead check)
      dead_marked: bool
      fragile: bool
      label_zh: optional warning suffix for UI
    """
    note = getattr(e, "note", None) or ""
    dead = _DEAD_MARK in note
    url = evidence_source_url(e)
    fragile = bool(url) and is_fragile_url(url)
    label_zh = ""
    if dead:
        label_zh = _DEAD_MARK
    elif fragile:
        label_zh = "链接可能不稳定（Google News 跳转）"
    return {
        "url": "" if dead else url,
        "dead_marked": dead,
        "fragile": fragile,
        "label_zh": label_zh,
    }


def format_reference_markdown_row(
    e: EvidenceItem,
    *,
    index: int | None = None,
    max_quote: int = DEFAULT_QUOTE_CHARS,
) -> str:
    """One Markdown bullet: id · quote · optional URL."""
    quote = evidence_quote(e, max_chars=max_quote)
    link = evidence_link_meta(e)
    eid = (e.id or "").strip() or "?"
    prefix = f"{index}. " if index is not None else ""
    bits = [f"**{eid}**"]
    if quote:
        bits.append(f"「{quote}」")
    line = prefix + " · ".join(bits)
    if link["url"]:
        line += f" — {link['url']}"
    elif link["label_zh"]:
        line += f" — _{link['label_zh']}_"
    return line
