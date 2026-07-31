"""URL hygiene for References / Evidence Base.

Honesty: never invent replacement URLs. Soft-check may clear or mark a link;
Google News redirects are treated as fragile and resolved when cheap.
"""

from __future__ import annotations

import re
import ssl
from typing import Any, Sequence
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen

from fx_report.model.weights import EvidenceItem

# Hosts that often 404 or expire after a few days
_FRAGILE_HOST_RE = re.compile(
    r"(^|\.)news\.google\.com$|(^|\.)news\.google\.[a-z.]+$",
    re.I,
)
_DEAD_MARK = "链接可能失效"
_GOOGLE_ARTICLE_RE = re.compile(r"news\.google\.com/.+", re.I)


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


def is_http_url(url: str | None) -> bool:
    u = (url or "").strip()
    return u.startswith("http://") or u.startswith("https://")


def is_fragile_url(url: str | None) -> bool:
    """True for Google News redirect / RSS article wrappers (often expire)."""
    u = (url or "").strip()
    if not u:
        return False
    try:
        host = (urlparse(u).hostname or "").lower()
    except Exception:
        return "news.google" in u.lower()
    if _FRAGILE_HOST_RE.search(host or ""):
        return True
    return bool(_GOOGLE_ARTICLE_RE.search(u))


def _extract_url_from_google_query(url: str) -> str:
    """Best-effort pull of an embedded article URL from Google News query params."""
    try:
        qs = parse_qs(urlparse(url).query)
    except Exception:
        return ""
    for key in ("url", "q", "u"):
        vals = qs.get(key) or []
        for v in vals:
            cand = unquote(v).strip()
            if cand.startswith("http") and "news.google." not in cand.lower():
                return cand
    # Sometimes the path encodes a publisher URL after /articles/
    return ""


def resolve_google_news_url(url: str, *, timeout: float = 2.5) -> str:
    """
    Prefer a stable publisher URL when the input is a Google News wrapper.
    Follows one redirect if needed; returns original URL on failure (no invention).
    """
    u = (url or "").strip()
    if not is_http_url(u) or not is_fragile_url(u):
        return u
    embedded = _extract_url_from_google_query(u)
    if embedded:
        return embedded
    try:
        req = Request(
            u,
            method="GET",
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; FX-Report-LinkCheck/1.0)",
                "Accept": "text/html,*/*",
            },
        )
        with urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            final = (resp.geturl() or "").strip()
            if is_http_url(final) and not is_fragile_url(final):
                return final
            # Peek Location-like body meta is too heavy; stop here
    except Exception:
        pass
    return u


def soft_check_url(url: str, *, timeout: float = 2.0) -> str:
    """
    Cheap reachability probe.

    Returns: "ok" | "maybe_dead" | "skip"
    skip = not http / empty / check errored without a clear 404.
    """
    u = (url or "").strip()
    if not is_http_url(u):
        return "skip"
    try:
        req = Request(
            u,
            method="HEAD",
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; FX-Report-LinkCheck/1.0)",
                "Accept": "*/*",
            },
        )
        with urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            code = int(getattr(resp, "status", None) or resp.getcode() or 0)
            if code in {404, 410, 451}:
                return "maybe_dead"
            if 200 <= code < 400:
                return "ok"
            # Some hosts reject HEAD — retry GET lightly
            if code in {403, 405, 501}:
                raise OSError("head_rejected")
            if code >= 500:
                return "skip"
            return "ok"
    except Exception:
        try:
            req = Request(
                u,
                method="GET",
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; FX-Report-LinkCheck/1.0)",
                    "Accept": "text/html,*/*",
                    "Range": "bytes=0-0",
                },
            )
            with urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
                code = int(getattr(resp, "status", None) or resp.getcode() or 0)
                if code in {404, 410, 451}:
                    return "maybe_dead"
                if 200 <= code < 400 or code == 206:
                    return "ok"
                return "skip"
        except Exception as exc:
            msg = str(exc).lower()
            if "404" in msg or "not found" in msg or "gone" in msg:
                return "maybe_dead"
            return "skip"


def prefer_stable_url(url: str, *, timeout: float = 2.5) -> str:
    """Resolve fragile wrappers when possible; otherwise return input unchanged."""
    u = (url or "").strip()
    if not u:
        return ""
    if is_fragile_url(u):
        return resolve_google_news_url(u, timeout=timeout)
    return u


def _strip_url_from_note(note: str, url: str) -> str:
    n = (note or "").strip()
    if not n:
        return ""
    if url and url in n:
        n = n.replace(url, "")
    n = re.sub(r"https?://\S+", "", n)
    n = re.sub(r"[｜|]\s*$", "", n).strip(" ｜|")
    n = re.sub(r"\s{2,}", " ", n).strip()
    return n


def _ensure_dead_mark(note: str) -> str:
    n = (note or "").strip()
    if _DEAD_MARK in n:
        return n
    return f"{n}｜{_DEAD_MARK}" if n else _DEAD_MARK


def sanitize_evidence_urls(
    evidence: Sequence[EvidenceItem],
    *,
    soft_check: bool = True,
    max_checks: int = 40,
    timeout: float = 2.0,
) -> dict[str, Any]:
    """
    Mutate evidence in place:
      1) Prefer stable publisher URL over Google News wrappers
      2) Optional soft HEAD/GET — on clear 404, drop hyperlink and mark 链接可能失效

    Does not invent replacement URLs. Caps network checks for pipeline latency.
    """
    meta: dict[str, Any] = {
        "resolved_fragile": 0,
        "checked": 0,
        "ok": 0,
        "maybe_dead": 0,
        "skipped_check": 0,
        "unchanged": 0,
    }
    checks_left = max(0, int(max_checks)) if soft_check else 0

    for e in evidence:
        raw = (e.url or "").strip()
        if not raw:
            # Fallback: parse from note
            m = re.search(r"(https?://\S+)", e.note or "")
            if m:
                raw = m.group(1).rstrip("｜|)")
        if not is_http_url(raw):
            meta["unchanged"] += 1
            continue

        stable = prefer_stable_url(raw, timeout=timeout)
        if stable != raw:
            meta["resolved_fragile"] += 1
            # Update note URL token if present
            if e.note and raw in e.note:
                e.note = e.note.replace(raw, stable)
            e.url = stable
            raw = stable
        else:
            e.url = raw

        if not soft_check or checks_left <= 0:
            if soft_check and checks_left <= 0:
                meta["skipped_check"] += 1
            else:
                meta["unchanged"] += 1
            continue

        # Prefer checking fragile leftovers and newsapi / rss first
        priority = is_fragile_url(raw)
        if not priority and checks_left < max(8, max_checks // 4):
            # Save remaining budget for fragile ones later — still check a share
            if meta["checked"] > max_checks // 2 and not priority:
                meta["skipped_check"] += 1
                continue

        status = soft_check_url(raw, timeout=timeout)
        checks_left -= 1
        meta["checked"] += 1
        if status == "ok":
            meta["ok"] += 1
            continue
        if status == "maybe_dead":
            meta["maybe_dead"] += 1
            e.note = _ensure_dead_mark(_strip_url_from_note(e.note or "", raw))
            e.url = ""  # do not hyperlink a known-dead URL
            continue
        meta["skipped_check"] += 1

    return meta


def provider_stability_rank(provider: str | None) -> int:
    """Lower is better for References (stable URL + content preferred)."""
    p = (provider or "").strip().lower()
    order = {
        "whitelist": 0,
        "ai_research": 1,
        "tavily": 2,
        "brave": 3,
        "newsapi": 4,
        "finnhub": 5,
        "official_rss": 6,
        "rss": 7,
        "gdelt": 8,
        "google_news_rss": 9,
    }
    return order.get(p, 7)
