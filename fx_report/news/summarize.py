"""
Evidence summarization layer (ECDA-style): compress long article text into a
short auditable blurb before HITL / weighting.

Offline-first: extractive sentence pick (no network). Optional LLM rewrite
when an API key is present — always falls back to extractive on failure.
"""

from __future__ import annotations

import os
import re
from typing import Any, Sequence

from fx_report.model.weights import EvidenceItem
from fx_report.news.fetch import Headline

# FX / macro cue words — boost sentences that look market-relevant
_FX_CUES = frozenset(
    {
        "fed",
        "rba",
        "ecb",
        "boe",
        "boj",
        "rate",
        "rates",
        "hike",
        "cut",
        "inflation",
        "cpi",
        "gdp",
        "dollar",
        "aussie",
        "aud",
        "usd",
        "yield",
        "tariff",
        "china",
        "iron",
        "ore",
        "hawkish",
        "dovish",
        "policy",
        "currency",
        "fx",
        "forex",
        "央行",
        "加息",
        "降息",
        "通胀",
        "汇率",
        "美元",
        "澳元",
    }
)

_SENT_SPLIT = re.compile(r"(?<=[.!?。！？；;])\s+|\n+")
_WORD = re.compile(r"[a-z0-9\u4e00-\u9fff]+", re.I)

DEFAULT_MAX_CHARS = 220
DEFAULT_MAX_SENTENCES = 2


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD.findall(text or "") if len(t) > 1}


def split_sentences(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(raw) if p and p.strip()]
    # Drop ultra-short debris
    return [p for p in parts if len(p) >= 12]


def score_sentence(sent: str, *, title_tokens: set[str]) -> float:
    toks = _tokens(sent)
    if not toks:
        return 0.0
    overlap = len(toks & title_tokens) / max(1.0, float(len(title_tokens) or 1))
    cue = len(toks & _FX_CUES) / max(1.0, float(len(toks)))
    # Prefer mid-length informative sentences
    length_pen = 0.0
    n = len(sent)
    if n < 40:
        length_pen = -0.15
    elif n > 320:
        length_pen = -0.25
    return 1.2 * overlap + 0.9 * cue + length_pen


def extractive_summary(
    text: str,
    *,
    title: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
    max_sentences: int = DEFAULT_MAX_SENTENCES,
) -> str:
    """
    Pick top sentences by title overlap + FX cues. Works fully offline.
    If text is already short, return a trimmed version.
    """
    blob = (text or "").strip()
    title_s = (title or "").strip()
    if not blob:
        return title_s[:max_chars]
    if len(blob) <= max_chars and "\n" not in blob:
        # Already a short blurb — keep as-is (still cap)
        return blob[:max_chars]

    sents = split_sentences(blob)
    if not sents:
        return (blob if blob else title_s)[:max_chars]

    title_toks = _tokens(title_s) or _tokens(sents[0])
    ranked = sorted(
        enumerate(sents),
        key=lambda iv: (-score_sentence(iv[1], title_tokens=title_toks), iv[0]),
    )
    picked_idx = sorted(i for i, _ in ranked[: max(1, int(max_sentences))])
    pieces: list[str] = []
    total = 0
    for i in picked_idx:
        s = sents[i].strip()
        if total + len(s) + (2 if pieces else 0) > max_chars:
            remain = max_chars - total - (2 if pieces else 0)
            if remain >= 40:
                pieces.append(s[: remain - 1].rstrip() + "…")
            break
        pieces.append(s)
        total += len(s) + (2 if len(pieces) > 1 else 0)
    out = " ".join(pieces).strip()
    if not out:
        out = (title_s or blob)[:max_chars]
    return out[:max_chars]


def _llm_compress(text: str, *, title: str, max_chars: int) -> str | None:
    """Optional one-shot LLM rewrite. Returns None to trigger extractive fallback."""
    api_key = (
        (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
        or (os.environ.get("OPENAI_API_KEY") or "").strip()
        or (os.environ.get("GROQ_API_KEY") or "").strip()
    )
    if not api_key:
        return None
    # Keep dependency light: reuse resolve_llm_config when available
    try:
        from fx_report.news.llm import resolve_llm_config

        cfg = resolve_llm_config()
        if cfg is None:
            return None
    except Exception:
        return None

    prompt = (
        "Compress the FX news excerpt into ONE short English or Chinese blurb "
        f"(≤{max_chars} chars) for audit. Keep facts only; no new claims.\n"
        f"Title: {title[:160]}\n"
        f"Text: {text[:2500]}"
    )
    try:
        # Prefer a tiny chat call if the LLM module exposes one; else skip.
        from fx_report.news import llm as llm_mod

        chat = getattr(llm_mod, "chat_complete", None) or getattr(
            llm_mod, "simple_chat", None
        )
        if chat is None:
            return None
        raw = chat(cfg, prompt, max_tokens=120)  # type: ignore[misc]
        blurb = str(raw or "").strip().strip('"')
        if not blurb or len(blurb) < 12:
            return None
        return blurb[:max_chars]
    except Exception:
        return None


def summarize_text(
    text: str,
    *,
    title: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
    prefer_llm: bool = False,
) -> tuple[str, str]:
    """
    Returns (blurb, method) where method is 'extractive' | 'llm' | 'title'.
    """
    blob = (text or "").strip()
    title_s = (title or "").strip()
    if not blob and not title_s:
        return "", "empty"
    if prefer_llm and blob:
        llm_out = _llm_compress(blob, title=title_s, max_chars=max_chars)
        if llm_out:
            return llm_out, "llm"
    if blob:
        return (
            extractive_summary(blob, title=title_s, max_chars=max_chars),
            "extractive",
        )
    return title_s[:max_chars], "title"


def _source_text_for_item(
    e: EvidenceItem,
    *,
    headline: Headline | None = None,
) -> str:
    parts: list[str] = []
    if headline is not None:
        if (headline.summary or "").strip():
            parts.append(headline.summary.strip())
        if (headline.title or "").strip() and headline.title.strip() != (e.title or "").strip():
            parts.append(headline.title.strip())
    # Existing summary / long note tails (rationale after first pipe often long)
    if (e.summary or "").strip():
        parts.append(e.summary.strip())
    note = (e.note or "").strip()
    if note:
        # Strip provider tags; keep rationale body if present
        body = note
        for sep in ("｜", "|"):
            if sep in body:
                chunks = [c.strip() for c in body.split(sep) if c.strip()]
                # Prefer longest chunk as potential prose
                body = max(chunks, key=len) if chunks else body
                break
        if len(body) > 40:
            parts.append(body)
    if not parts and (e.title or "").strip():
        parts.append(e.title.strip())
    # De-dupe while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return "\n".join(uniq)


def apply_evidence_summaries(
    items: list[EvidenceItem],
    *,
    headlines: Sequence[Headline] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    prefer_llm: bool = False,
    skip_priors: bool = True,
) -> dict[str, Any]:
    """
    Mutate EvidenceItem.summary in place with a short auditable blurb.
    Does not invent facts — only compresses title/note/headline.summary.
    """
    by_url: dict[str, Headline] = {}
    by_title: dict[str, Headline] = {}
    for h in headlines or []:
        if (h.url or "").strip():
            by_url[h.url.strip().lower()] = h
        if (h.title or "").strip():
            by_title[h.title.strip().lower()] = h

    n_set = 0
    methods: dict[str, int] = {}
    for e in items:
        if skip_priors and e.is_prior:
            continue
        h = None
        if (e.url or "").strip():
            h = by_url.get(e.url.strip().lower())
        if h is None and (e.title or "").strip():
            h = by_title.get(e.title.strip().lower())
        src = _source_text_for_item(e, headline=h)
        blurb, method = summarize_text(
            src,
            title=e.title or "",
            max_chars=max_chars,
            prefer_llm=prefer_llm,
        )
        if not blurb:
            blurb = (e.title or "")[:max_chars]
            method = "title"
        e.summary = blurb
        # Keep method auditable: stamp into note only if not already tagged.
        tag = f"summary:{method}"
        if tag not in (e.note or ""):
            e.note = f"{tag}｜{e.note}" if e.note else tag
        n_set += 1
        methods[method] = methods.get(method, 0) + 1

    return {
        "summarized_n": n_set,
        "methods": methods,
        "max_chars": max_chars,
        "prefer_llm": bool(prefer_llm),
    }
