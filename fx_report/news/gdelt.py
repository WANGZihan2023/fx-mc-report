"""GDELT DOC 2.0 API helper (free, no API key) for date-filtered historical news.

Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/

Official searchable window is roughly the last ~3 months. Callers should clamp
start/end into that window; out-of-window requests are skipped with a note.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

from fx_report.news.fetch import Headline, _coerce_date, _ssl_context

# GDELT DOC API officially indexes roughly the last 3 months.
GDELT_MAX_HISTORY_DAYS = 90
GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MAX_RECORDS = 75  # API allows up to 250; keep replay fetches light
_GDELT_SLEEP = time.sleep

# Pair-relevant boolean queries (DOC API: OR + quoted phrases).
PAIR_GDELT_QUERIES: dict[str, str] = {
    "USD/AUD": (
        '("Australian dollar" OR AUDUSD OR RBA OR "iron ore" OR Aussie OR '
        'Fed OR FOMC OR "Reserve Bank of Australia")'
    ),
    "AUD/USD": (
        '("Australian dollar" OR AUDUSD OR RBA OR "iron ore" OR Aussie OR '
        'Fed OR FOMC OR "Reserve Bank of Australia")'
    ),
    "EUR/USD": '(ECB OR EURUSD OR "euro zone" OR Lagarde OR Fed OR FOMC)',
    "GBP/USD": '("Bank of England" OR GBPUSD OR sterling OR Bailey OR Fed)',
    "USD/JPY": '("Bank of Japan" OR USDJPY OR yen OR Ueda OR BOJ OR Fed)',
    "USD/CNH": '(PBOC OR "offshore yuan" OR CNH OR renminbi OR Fed)',
    "USD/CNY": '(PBOC OR yuan OR CNY OR renminbi OR Fed)',
    "USD/CAD": '("Bank of Canada" OR USDCAD OR loonie OR oil OR Fed)',
    "NZD/USD": '(RBNZ OR "New Zealand dollar" OR kiwi OR OCR OR Fed)',
    "USD/CHF": '(SNB OR "Swiss franc" OR USDCHF OR Fed)',
}


def gdelt_earliest_searchable_date(*, today: date | None = None) -> date:
    return (today or date.today()) - timedelta(days=GDELT_MAX_HISTORY_DAYS)


def clamp_gdelt_from_date(
    start: date,
    end: date,
    *,
    today: date | None = None,
) -> tuple[date | None, bool]:
    """Clamp GDELT start into the ~3-month DOC window.

    Returns (clamped_start, was_clamped). If ``end`` is older than the window,
    returns (None, True) so callers skip the request.
    """
    earliest = gdelt_earliest_searchable_date(today=today)
    if end < earliest:
        return None, True
    clamped = max(start, earliest)
    return clamped, clamped > start


def gdelt_query_for_pair(pair: str) -> str:
    key = (pair or "").strip().upper()
    if key in PAIR_GDELT_QUERIES:
        return PAIR_GDELT_QUERIES[key]
    # Fallback: pair tokens + common FX terms
    tokens = key.replace("/", " ").split()
    parts = [t for t in tokens if t] + ["Fed", "FX", "currency"]
    return "(" + " OR ".join(parts) + ")"


def _gdelt_stamp(d: date, *, end_of_day: bool = False) -> str:
    """Format YYYYMMDDHHMMSS for STARTDATETIME / ENDDATETIME."""
    if end_of_day:
        return f"{d.strftime('%Y%m%d')}235959"
    return f"{d.strftime('%Y%m%d')}000000"


def _parse_gdelt_seendate(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    # Typical: 20200713T183045Z
    try:
        core = text.replace("Z", "").replace("z", "")
        if len(core) >= 15 and core[8] == "T":
            dt = datetime.strptime(core[:15], "%Y%m%dT%H%M%S")
            return dt.replace(tzinfo=timezone.utc)
        if len(core) == 14 and core.isdigit():
            dt = datetime.strptime(core, "%Y%m%d%H%M%S")
            return dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    # ISO fallback
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _headlines_from_gdelt_articles(
    articles: list[Any],
    limit: int,
    *,
    start: date | None = None,
    end: date | None = None,
) -> list[Headline]:
    out: list[Headline] = []
    for a in articles:
        if not isinstance(a, dict):
            continue
        title = (a.get("title") or "").strip()
        if not title:
            continue
        published = _parse_gdelt_seendate(a.get("seendate"))
        if published is not None:
            d = published.date()
            if start is not None and d < start:
                continue
            if end is not None and d > end:
                continue
        domain = (a.get("domain") or "").strip() or "GDELT"
        out.append(
            Headline(
                title=title,
                summary="",
                source=domain,
                url=(a.get("url") or "").strip(),
                published=published,
                provider="gdelt",
            )
        )
        if len(out) >= limit:
            break
    return out


def _gdelt_request_json(
    url: str,
    timeout: int,
    *,
    max_retries: int = 2,
) -> tuple[dict[str, Any] | None, str | None, int | None]:
    """GET GDELT JSON with light 429 backoff. Returns (data, error, http_status)."""
    last_err: str | None = None
    last_status: int | None = None
    for attempt in range(max(1, int(max_retries))):
        try:
            req = Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; FXReportBot/2.0; +gdelt-historical)"
                    ),
                    "Accept": "application/json,text/plain,*/*",
                },
            )
            with urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
                raw = resp.read()
                last_status = int(getattr(resp, "status", 200) or 200)
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                return {"articles": []}, None, last_status
            # GDELT sometimes returns HTML error pages with HTTP 200
            if text[:1] == "<" or "text/html" in text[:40].lower():
                last_err = f"non_json_html_response:{text[:160]}"
                return None, last_err, last_status
            data = json.loads(text)
            if isinstance(data, dict):
                return data, None, last_status
            last_err = f"non_dict_response:{type(data).__name__}"
            return None, last_err, last_status
        except HTTPError as e:
            last_status = int(e.code)
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:240]
            except Exception:
                body = ""
            last_err = f"HTTP {e.code}: {e.reason}"
            if body:
                last_err = f"{last_err} | {body}"
            if e.code == 429 and attempt + 1 < max_retries:
                _GDELT_SLEEP(min(30.0, 2.0 * (3**attempt)))
                continue
            return None, last_err, last_status
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            return None, last_err, last_status
    return None, last_err or "request_failed", last_status


def fetch_gdelt_doc(
    query: str,
    *,
    start_date: date | datetime | str,
    end_date: date | datetime | str,
    limit: int = 25,
    timeout: int = 20,
    call_meta: dict[str, Any] | None = None,
) -> list[Headline]:
    """
    Fetch date-filtered article list from GDELT DOC 2.0 (ArtList / JSON).

    No API key required. Errors and empty results are surfaced via ``call_meta``
    (never silently swallowed).
    """
    if call_meta is not None:
        call_meta.clear()
        call_meta.update(
            {
                "error": None,
                "http_status": None,
                "query": None,
                "start": None,
                "end": None,
                "raw_count": 0,
                "url_host": "api.gdeltproject.org",
            }
        )

    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    q = (query or "").strip()
    if not q or start is None or end is None:
        if call_meta is not None:
            call_meta["error"] = "missing_query_or_dates"
        return []
    if end < start:
        if call_meta is not None:
            call_meta["error"] = "end_before_start"
        return []

    maxrecords = max(1, min(int(limit), GDELT_MAX_RECORDS))
    params = {
        "query": q,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(maxrecords),
        "startdatetime": _gdelt_stamp(start, end_of_day=False),
        "enddatetime": _gdelt_stamp(end, end_of_day=True),
        "sort": "DateDesc",
    }
    # urlencode quote_via=quote_plus for query safety
    url = f"{GDELT_DOC_ENDPOINT}?{urlencode(params, quote_via=quote_plus)}"
    if call_meta is not None:
        call_meta["query"] = q
        call_meta["start"] = start.isoformat()
        call_meta["end"] = end.isoformat()

    data, err, status = _gdelt_request_json(url, timeout, max_retries=2)
    if call_meta is not None:
        call_meta["http_status"] = status
    if err:
        if call_meta is not None:
            call_meta["error"] = err
        return []
    assert data is not None
    articles = data.get("articles") if isinstance(data, dict) else None
    if not isinstance(articles, list):
        # Empty / no-match responses may omit articles or return {}
        if call_meta is not None:
            call_meta["raw_count"] = 0
        return []
    if call_meta is not None:
        call_meta["raw_count"] = len(articles)
    return _headlines_from_gdelt_articles(
        articles, limit, start=start, end=end
    )
