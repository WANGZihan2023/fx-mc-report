"""
Evidence summarization layer (ECDA-style): compress long article text into a
short auditable blurb before HITL / weighting.

Offline-first: extractive sentence pick (no network). Optional LLM rewrite
when an API key is present — always falls back to extractive on failure.

Also extracts stance-aligned **support quotes** for References — sentences that
actually back Higher/Lower/Context, not SEO leads or generic blurbs.
"""

from __future__ import annotations

import os
import re
from typing import Any, Sequence

from fx_report.market.pairs import PairSpec, get_pair
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

# Stance cues for support-quote picking (Higher / Lower / Context)
_BULLISH_CUES = frozenset(
    {
        "hike",
        "hikes",
        "hawkish",
        "rises",
        "rise",
        "gains",
        "climbs",
        "surge",
        "surges",
        "rally",
        "stronger",
        "strengthen",
        "strengthens",
        "firmer",
        "higher",
        "tightening",
        "tighten",
        "bullish",
        "hot",
        "sticky",
        "加息",
        "鹰派",
        "走强",
        "升值",
        "上行",
        "偏强",
    }
)
_BEARISH_CUES = frozenset(
    {
        "cut",
        "cuts",
        "dovish",
        "falls",
        "fall",
        "drops",
        "drop",
        "slides",
        "plunge",
        "slump",
        "weaker",
        "weaken",
        "weakens",
        "soft",
        "softens",
        "cool",
        "cools",
        "cooling",
        "miss",
        "missed",
        "lower",
        "easing",
        "ease",
        "bearish",
        "降息",
        "鸽派",
        "走弱",
        "贬值",
        "下行",
        "偏弱",
    }
)
_CONTEXT_CUES = frozenset(
    {
        "outlook",
        "await",
        "awaits",
        "monitor",
        "monitors",
        "unchanged",
        "hold",
        "holds",
        "held",
        "mixed",
        "background",
        "context",
        "weigh",
        "weighs",
        "balance",
        "steady",
        "range",
        "观望",
        "持稳",
        "背景",
        "中性",
    }
)

_CATEGORY_CUES: dict[str, frozenset[str]] = {
    "fed": frozenset({"fed", "fomc", "powell", "federal", "reserve"}),
    "rba": frozenset({"rba", "bullock", "australia", "aussie", "aud", "cash"}),
    "rbnz": frozenset({"rbnz", "ocr", "zealand", "nzd"}),
    "ecb": frozenset({"ecb", "lagarde", "eurozone", "eur"}),
    "boe": frozenset({"boe", "bailey", "sterling", "gbp"}),
    "boj": frozenset({"boj", "ueda", "yen", "jpy"}),
    "boc": frozenset({"boc", "canada", "cad"}),
    "pboc": frozenset({"pboc", "yuan", "cnh", "cny", "fixing"}),
    "cpi": frozenset({"cpi", "inflation", "pce", "prices"}),
    "oil": frozenset({"oil", "brent", "wti", "crude", "opec"}),
    "china_growth": frozenset({"china", "gdp", "stimulus", "property", "yuan"}),
    "china_iron": frozenset({"iron", "ore", "steel", "simandou", "dalian"}),
    "yields": frozenset({"yield", "treasury", "bond", "differential"}),
    "geopolitics": frozenset({"war", "sanction", "geopolit", "conflict", "iran"}),
    "growth": frozenset({"gdp", "payroll", "jobs", "pmi", "unemployment"}),
}

_JUNK_RE = re.compile(
    r"\b(subscribe|newsletter|cookie|click here|sign up|privacy policy|"
    r"terms of (use|service)|all rights reserved|javascript|enable cookies|"
    r"advertisement|sponsored|read more|celebrity|gossip|hollywood|"
    r"recipe|cooking|sports? news)\b",
    re.I,
)

_SENT_SPLIT = re.compile(r"(?<=[.!?。！？；;])\s+|\n+")
_WORD = re.compile(r"[a-z0-9\u4e00-\u9fff]+", re.I)

DEFAULT_MAX_CHARS = 220
DEFAULT_MAX_SENTENCES = 2
DEFAULT_SUPPORT_CHARS = 220
SUPPORT_QUALITY_SUPPORT = "support"
SUPPORT_QUALITY_WEAK = "weak"
SUPPORT_QUALITY_TITLE = "title"


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


def _is_junk_sentence(sent: str) -> bool:
    s = (sent or "").strip()
    if len(s) < 20:
        return True
    if _JUNK_RE.search(s):
        return True
    toks = _tokens(s)
    if not toks:
        return True
    if len(toks & _FX_CUES) == 0 and len(s) < 40:
        return True
    return False


def score_support_sentence(
    sent: str,
    *,
    direction: int,
    category: str = "",
    title_tokens: set[str] | None = None,
    pair: PairSpec | str | None = None,
) -> float:
    """
    Rank a candidate sentence by how well it *supports* the item's stance.
    Higher = better evidentiary quote for Higher/Lower/Context.
    """
    if _is_junk_sentence(sent):
        return -5.0
    toks = _tokens(sent)
    if not toks:
        return -5.0

    title_tokens = title_tokens or set()
    overlap = len(toks & title_tokens) / max(1.0, float(len(title_tokens) or 1))
    fx = len(toks & _FX_CUES) / max(1.0, float(len(toks)))
    cat_set = _CATEGORY_CUES.get((category or "").lower(), frozenset())
    cat_hit = len(toks & cat_set) / max(1.0, float(len(cat_set) or 1)) if cat_set else 0.0

    bull = len(toks & _BULLISH_CUES)
    bear = len(toks & _BEARISH_CUES)
    ctx = len(toks & _CONTEXT_CUES)
    stance = 0.0
    d = int(direction or 0)
    if d > 0:
        stance = 0.55 * bull - 0.35 * bear + 0.05 * ctx
    elif d < 0:
        stance = 0.55 * bear - 0.35 * bull + 0.05 * ctx
    else:
        stance = 0.45 * ctx + 0.15 * (bull + bear) * 0.25

    # Pair-aware: boost when classify would assign the same direction to this sentence
    align = 0.0
    if pair is not None and d != 0:
        try:
            from fx_report.news.classify import _direction_for_pair

            spec = get_pair(pair) if isinstance(pair, str) else pair
            cat = (category or "").strip() or "other"
            guessed = _direction_for_pair(spec, cat, sent)
            if guessed is not None and guessed == d:
                align = 1.6
            elif guessed is not None and guessed == -d:
                align = -1.0
        except Exception:
            align = 0.0

    n = len(sent)
    length_pen = 0.0
    if n < 40:
        length_pen = -0.2
    elif n > 360:
        length_pen = -0.3

    return (
        1.1 * stance
        + 0.9 * fx
        + 0.7 * cat_hit
        + 0.35 * overlap
        + align
        + length_pen
    )


def extract_support_quote(
    text: str,
    *,
    title: str = "",
    direction: int = 0,
    category: str = "",
    pair: PairSpec | str | None = None,
    max_chars: int = DEFAULT_SUPPORT_CHARS,
    max_sentences: int = 2,
) -> tuple[str, str]:
    """
    Pick 1–2 sentences that support the Higher/Lower/Context stance.

    Returns (quote, quality) where quality is:
      - 'support': stance-aligned evidentiary sentence from available text
      - 'weak': best FX-ish sentence but weak stance match
      - 'title': fell back to title (last resort)

    Never invents text outside ``text`` / ``title``.
    """
    blob = (text or "").strip()
    title_s = (title or "").strip()
    title_toks = _tokens(title_s)

    if not blob and not title_s:
        return "", SUPPORT_QUALITY_TITLE

    # Single short blurb: still score whether it looks supportive
    if blob and len(blob) <= max_chars and "\n" not in blob and len(split_sentences(blob)) <= 1:
        if _is_junk_sentence(blob) and title_s:
            return title_s[:max_chars], SUPPORT_QUALITY_TITLE
        sc = score_support_sentence(
            blob,
            direction=direction,
            category=category,
            title_tokens=title_toks,
            pair=pair,
        )
        q = SUPPORT_QUALITY_SUPPORT if sc >= 0.55 else SUPPORT_QUALITY_WEAK
        return blob[:max_chars], q

    sents = split_sentences(blob) if blob else []
    if not sents and title_s:
        return title_s[:max_chars], SUPPORT_QUALITY_TITLE
    if not sents:
        return (blob or "")[:max_chars], SUPPORT_QUALITY_WEAK

    ranked = sorted(
        enumerate(sents),
        key=lambda iv: (
            -score_support_sentence(
                iv[1],
                direction=direction,
                category=category,
                title_tokens=title_toks or _tokens(sents[0]),
                pair=pair,
            ),
            iv[0],
        ),
    )
    # Prefer best non-junk for quality threshold
    best_score = -99.0
    for _i, sraw in ranked:
        sc = score_support_sentence(
            sraw,
            direction=direction,
            category=category,
            title_tokens=title_toks or _tokens(sents[0]),
            pair=pair,
        )
        if not _is_junk_sentence(sraw):
            best_score = sc
            break
        best_score = max(best_score, sc)
    # If even the best is junk / anti-aligned, fall back to title
    if best_score < -1.0 and title_s:
        return title_s[:max_chars], SUPPORT_QUALITY_TITLE

    # Select top non-junk sentences by score, then emit in document order
    selected: list[tuple[int, str]] = []
    min_keep = max(-0.2, best_score - 1.25)
    for i, sraw in ranked:
        s = sraw.strip()
        if _is_junk_sentence(s):
            continue
        sc_i = score_support_sentence(
            s,
            direction=direction,
            category=category,
            title_tokens=title_toks or _tokens(sents[0]),
            pair=pair,
        )
        if selected and sc_i < min_keep:
            continue
        selected.append((i, s))
        if len(selected) >= max(1, int(max_sentences)):
            break

    if not selected:
        if title_s:
            return title_s[:max_chars], SUPPORT_QUALITY_TITLE
        # Last resort: best ranked even if weak/junk-ish
        selected = [(ranked[0][0], ranked[0][1].strip())]

    pieces: list[str] = []
    total = 0
    for _i, s in sorted(selected, key=lambda iv: iv[0]):
        if total + len(s) + (2 if pieces else 0) > max_chars:
            remain = max_chars - total - (2 if pieces else 0)
            if remain >= 40:
                pieces.append(s[: remain - 1].rstrip() + "…")
            break
        pieces.append(s)
        total += len(s) + (2 if len(pieces) > 1 else 0)

    out = " ".join(pieces).strip()
    if not out:
        if title_s:
            return title_s[:max_chars], SUPPORT_QUALITY_TITLE
        return (blob or "")[:max_chars], SUPPORT_QUALITY_WEAK

    quality = SUPPORT_QUALITY_SUPPORT if best_score >= 0.55 else SUPPORT_QUALITY_WEAK
    return out[:max_chars], quality


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
    pair: PairSpec | str | None = None,
    support_max_chars: int = DEFAULT_SUPPORT_CHARS,
) -> dict[str, Any]:
    """
    Mutate EvidenceItem.summary and support_quote in place.

    - summary: short auditable blurb (extractive / optional LLM)
    - support_quote: stance-aligned sentence(s) for References display

    Does not invent facts — only compresses / picks from title/note/headline text.
    """
    by_url: dict[str, Headline] = {}
    by_title: dict[str, Headline] = {}
    for h in headlines or []:
        if (h.url or "").strip():
            by_url[h.url.strip().lower()] = h
        if (h.title or "").strip():
            by_title[h.title.strip().lower()] = h

    n_set = 0
    n_support = 0
    methods: dict[str, int] = {}
    support_qualities: dict[str, int] = {}
    for e in items:
        if skip_priors and e.is_prior:
            continue
        h = None
        if (e.url or "").strip():
            h = by_url.get(e.url.strip().lower())
        if h is None and (e.title or "").strip():
            h = by_title.get(e.title.strip().lower())
        src = _source_text_for_item(e, headline=h)

        # Support quote from raw available text (before summary overwrite).
        # Prefer explicit support_quote if already set (e.g. LLM extract).
        if not (getattr(e, "support_quote", None) or "").strip():
            quote, quality = extract_support_quote(
                src,
                title=e.title or "",
                direction=int(e.direction or 0),
                category=e.category or "",
                pair=pair,
                max_chars=support_max_chars,
                max_sentences=2,
            )
            e.support_quote = quote
            e.support_quote_quality = quality
        else:
            quality = (getattr(e, "support_quote_quality", None) or "").strip() or (
                SUPPORT_QUALITY_SUPPORT
            )
            e.support_quote_quality = quality
        if e.support_quote:
            n_support += 1
        support_qualities[e.support_quote_quality or ""] = (
            support_qualities.get(e.support_quote_quality or "", 0) + 1
        )

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
        sq_tag = f"support:{e.support_quote_quality or 'none'}"
        if sq_tag not in (e.note or ""):
            e.note = f"{sq_tag}｜{e.note}" if e.note else sq_tag
        n_set += 1
        methods[method] = methods.get(method, 0) + 1

    return {
        "summarized_n": n_set,
        "support_quote_n": n_support,
        "support_qualities": support_qualities,
        "methods": methods,
        "max_chars": max_chars,
        "prefer_llm": bool(prefer_llm),
    }
