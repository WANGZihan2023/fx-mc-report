"""Fetch FX headlines from official / vault sources only.

Priority:
  1) Central-bank / official RSS (Fed, RBA, ECB, BOE, …)
  2) inbox/ research PDFs / notes
  3) NewsAPI / Finnhub if vault keys are set
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import ssl
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from fx_report.config.api_config import inbox_files, is_set, load_config, timeout_s, vault_paths
from fx_report.market.pairs import PairSpec, get_pair

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


@dataclass
class Headline:
    title: str
    summary: str
    source: str
    url: str
    published: datetime | None
    provider: str  # official_rss | newsapi | gdelt | finnhub | inbox

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["published"] = self.published.isoformat() if self.published else None
        return d


# Official RSS — always attempted (no key)
OFFICIAL_RSS: list[tuple[str, str, str]] = [
    ("Fed", "https://www.federalreserve.gov/feeds/press_all.xml", "federalreserve.gov"),
    ("Fed monetary", "https://www.federalreserve.gov/feeds/press_monetary.xml", "federalreserve.gov"),
    ("RBA media", "https://www.rba.gov.au/rss/rss-cb-media-releases.xml", "rba.gov.au"),
    ("RBA speeches", "https://www.rba.gov.au/rss/rss-cb-speeches.xml", "rba.gov.au"),
    ("ECB press", "https://www.ecb.europa.eu/rss/press.html", "ecb.europa.eu"),
    ("BOE", "https://www.bankofengland.co.uk/rss/news", "bankofengland.co.uk"),
]

# Pair-specific official / public feeds (still no API key)
PAIR_EXTRA_RSS: dict[str, list[tuple[str, str, str]]] = {
    "USD/JPY": [],
    "EUR/USD": [],
    "USD/CNH": [],
    "USD/CNY": [],
    "USD/CAD": [],
    "NZD/USD": [],
    "USD/CHF": [],
    "USD/AUD": [],
    "AUD/USD": [],
    "GBP/USD": [],
}

# Free Google News RSS (no key) — query tuned per pair; filtered again by relevance
PAIR_GOOGLE_NEWS_Q: dict[str, str] = {
    "USD/AUD": 'RBA OR "Australian dollar" OR AUDUSD OR "iron ore" OR Aussie',
    "AUD/USD": 'RBA OR "Australian dollar" OR AUDUSD OR Aussie',
    "EUR/USD": 'ECB OR "euro zone" OR EURUSD OR Lagarde',
    "GBP/USD": '"Bank of England" OR GBPUSD OR sterling OR Bailey',
    "USD/JPY": '"Bank of Japan" OR USDJPY OR yen OR Ueda OR BOJ',
    "USD/CNH": 'PBOC OR "offshore yuan" OR CNH OR renminbi',
    "USD/CNY": 'PBOC OR yuan OR CNY OR renminbi',
    "USD/CAD": '"Bank of Canada" OR USDCAD OR loonie OR oil',
    "NZD/USD": 'RBNZ OR "New Zealand dollar" OR kiwi OR OCR',
    "USD/CHF": 'SNB OR "Swiss franc" OR USDCHF',
}

PAIR_NEWSAPI_QUERIES: dict[str, list[str]] = {
    "USD/AUD": ["RBA OR \"Australian dollar\" OR AUDUSD OR \"iron ore\""],
    "AUD/USD": ["RBA OR \"Australian dollar\" OR AUDUSD"],
    "EUR/USD": ["ECB OR \"euro zone\" OR EURUSD"],
    "GBP/USD": ["\"Bank of England\" OR GBPUSD OR sterling"],
    "USD/JPY": ["\"Bank of Japan\" OR USDJPY OR yen"],
    "USD/CNH": ["PBOC OR \"offshore yuan\" OR CNH OR renminbi"],
    "USD/CNY": ["PBOC OR yuan OR CNY OR renminbi"],
    "USD/CAD": ["\"Bank of Canada\" OR USDCAD"],
    "NZD/USD": ["RBNZ OR \"New Zealand dollar\""],
    "USD/CHF": ["SNB OR \"Swiss franc\""],
}

# NewsAPI developer / free plans only allow searching roughly the last month
# relative to *today*. Using vol lookback (often 60) as NewsAPI `from` pushes
# the window outside that plan limit and the API fails → empty headlines.
NEWSAPI_MAX_HISTORY_DAYS = 29


def newsapi_earliest_searchable_date(*, today: date | None = None) -> date:
    """Earliest calendar day NewsAPI developer/free plans can typically search."""
    return (today or date.today()) - timedelta(days=NEWSAPI_MAX_HISTORY_DAYS)


def clamp_newsapi_from_date(
    start: date,
    end: date,
    *,
    today: date | None = None,
) -> tuple[date | None, bool]:
    """Clamp NewsAPI `from` into the searchable window.

    Returns (clamped_start, was_clamped). If `end` itself is older than the
    searchable window, returns (None, True) so callers skip the request.
    """
    earliest = newsapi_earliest_searchable_date(today=today)
    if end < earliest:
        return None, True
    clamped = max(start, earliest)
    return clamped, clamped > start

# Soft relevance filter for official / public feeds
PAIR_KEEP_KEYWORDS: dict[str, re.Pattern[str]] = {
    "USD/AUD": re.compile(
        r"RBA|Australia|AUD|Aussie|iron ore|Fed|FOMC|inflation|CPI|rate|oil|"
        r"Iran|Hormuz|China|yield|dollar|FX|forex|currency",
        re.I,
    ),
    "AUD/USD": re.compile(
        r"RBA|Australia|AUD|Aussie|iron ore|Fed|FOMC|inflation|CPI|rate|oil|"
        r"China|yield|dollar|FX|forex",
        re.I,
    ),
    "EUR/USD": re.compile(
        r"ECB|euro|EUR|Lagarde|Fed|FOMC|inflation|CPI|yield|dollar|FX|forex", re.I
    ),
    "GBP/USD": re.compile(
        r"Bank of England|BOE|sterling|GBP|Bailey|Fed|inflation|CPI|yield|FX", re.I
    ),
    "USD/JPY": re.compile(
        r"BOJ|Japan|yen|JPY|Ueda|Fed|yield|Treasury|FX|forex|dollar", re.I
    ),
    "USD/CNH": re.compile(
        r"China|yuan|PBOC|CNH|CNY|renminbi|Fed|offshore|fixing|FX", re.I
    ),
    "USD/CNY": re.compile(
        r"China|yuan|PBOC|CNY|renminbi|Fed|onshore|fixing|FX", re.I
    ),
    "USD/CAD": re.compile(
        r"Bank of Canada|BoC|Canada|CAD|loonie|oil|crude|Fed|inflation|FX", re.I
    ),
    "NZD/USD": re.compile(
        r"RBNZ|New Zealand|NZD|kiwi|OCR|Fed|dairy|inflation|FX", re.I
    ),
    "USD/CHF": re.compile(
        r"SNB|Swiss|franc|CHF|Fed|safe.?haven|yield|FX", re.I
    ),
}


def _http_get(url: str, timeout: int = 20) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; FXReportBot/2.0; +official-sources)",
            "Accept": "application/rss+xml,application/atom+xml,application/xml,text/xml,*/*",
        },
    )
    with urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        return resp.read()


def _http_json(url: str, timeout: int = 20) -> Any:
    return json.loads(_http_get(url, timeout=timeout).decode("utf-8", errors="replace"))


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _coerce_date(value: date | datetime | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    if not text:
        return None
    if "T" in text:
        text = text.split("T", 1)[0]
    if " " in text:
        text = text.split(" ", 1)[0]
    return date.fromisoformat(text)


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _parse_rss_xml(raw: bytes, source_name: str, source_host: str) -> list[Headline]:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []

    items: list[ET.Element] = []
    # RSS 2.0
    items.extend(root.findall(".//item"))
    # Atom
    items.extend(root.findall(".//{http://www.w3.org/2005/Atom}entry"))
    # RDF RSS 1.0
    items.extend(root.findall(".//{http://purl.org/rss/1.0/}item"))

    out: list[Headline] = []
    for item in items:
        title = ""
        link = ""
        desc = ""
        pub = None
        for child in list(item):
            name = _local(child.tag).lower()
            text = (child.text or "").strip()
            if name == "title" and text:
                title = text
            elif name in {"link", "guid"} and text.startswith("http"):
                link = text
            elif name == "link" and child.get("href"):
                link = child.get("href") or link
            elif name in {"description", "summary", "content"} and text:
                desc = re.sub(r"<[^>]+>", "", text).strip()
            elif name in {"pubdate", "published", "updated", "date"} and text:
                pub = _parse_dt(text)
        if not title:
            continue
        out.append(
            Headline(
                title=title,
                summary=desc[:500],
                source=source_name,
                url=link or f"https://{source_host}",
                published=pub,
                provider="official_rss",
            )
        )
    return out


def fetch_official_rss(pair: str, limit_per_feed: int = 8) -> list[Headline]:
    feeds = list(OFFICIAL_RSS) + list(PAIR_EXTRA_RSS.get(pair, []))
    keep = PAIR_KEEP_KEYWORDS.get(pair)
    out: list[Headline] = []
    for name, url, host in feeds:
        try:
            raw = _http_get(url, timeout=15)
        except Exception:
            continue
        items = _parse_rss_xml(raw, name, host)
        for h in items[: limit_per_feed * 2]:
            blob = f"{h.title} {h.summary}"
            if keep is not None and not keep.search(blob):
                # Still keep Fed/RBA/ECB rate decisions even if filter is strict
                if not re.search(r"rate|monetary|FOMC|decision|CPI|inflation|SMP", blob, re.I):
                    continue
            out.append(h)
            if len(out) >= 40:
                return out
    return out


def fetch_google_news_rss(pair: str, limit: int = 12) -> list[Headline]:
    """
    Public Google News RSS — no API key.
    Relevance is soft-filtered with PAIR_KEEP_KEYWORDS; classify layer applies
    pair_relevance again before evidence.
    """
    q = PAIR_GOOGLE_NEWS_Q.get(pair) or pair.replace("/", " ")
    url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        raw = _http_get(url, timeout=15)
    except Exception:
        return []
    items = _parse_rss_xml(raw, "Google News", "news.google.com")
    keep = PAIR_KEEP_KEYWORDS.get(pair)
    out: list[Headline] = []
    for h in items:
        h.provider = "google_news_rss"
        blob = f"{h.title} {h.summary}"
        if keep is not None and not keep.search(blob):
            continue
        # Drop obvious non-FX junk that sometimes slips in
        if re.search(r"\b(recipe|sports|celebrity|horoscope)\b", blob, re.I):
            continue
        out.append(h)
        if len(out) >= limit:
            break
    return out


_NEWSAPI_DOMAINS = (
    "reuters.com,bloomberg.com,wsj.com,ft.com,afr.com,cnbc.com,federalreserve.gov"
)
# Mem cache stores envelope dicts (articles + status + cached_at).
_NEWSAPI_MEM_CACHE: dict[str, dict[str, Any]] = {}
# Injectable sleep for unit tests (avoid real waits on 429 backoff).
_NEWSAPI_SLEEP: Callable[[float], None] = time.sleep

# Disk/memory cache TTLs — never pin empty/error forever.
NEWSAPI_CACHE_TTL_OK_S = 7 * 24 * 3600
NEWSAPI_CACHE_TTL_EMPTY_S = 3600
NEWSAPI_CACHE_TTL_ERROR_S = 15 * 60


def _newsapi_cache_ttl_s(status: str) -> int:
    if status == "error":
        return NEWSAPI_CACHE_TTL_ERROR_S
    if status == "empty":
        return NEWSAPI_CACHE_TTL_EMPTY_S
    return NEWSAPI_CACHE_TTL_OK_S


def _newsapi_envelope_fresh(envelope: dict[str, Any]) -> bool:
    status = str(envelope.get("status") or "ok")
    cached_at_raw = envelope.get("cached_at")
    arts = envelope.get("articles")
    if not isinstance(arts, list):
        return False
    if not cached_at_raw:
        if status == "error":
            return False
        return len(arts) > 0
    try:
        cached_at = datetime.fromisoformat(str(cached_at_raw).replace("Z", "+00:00"))
        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - cached_at.astimezone(timezone.utc)).total_seconds()
    except Exception:
        return len(arts) > 0 and status != "error"
    return age <= float(_newsapi_cache_ttl_s(status))


def _newsapi_normalize_envelope(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    arts = data.get("articles")
    if not isinstance(arts, list):
        return None
    payload = [a for a in arts if isinstance(a, dict)]
    status = data.get("status")
    if status not in {"ok", "empty", "error"}:
        status = "empty" if not payload else "ok"
    return {
        "articles": payload,
        "status": status,
        "error": data.get("error"),
        "cached_at": data.get("cached_at"),
        "http_status": data.get("http_status"),
        "used_domains": data.get("used_domains"),
    }


def _newsapi_cache_dir() -> Path:
    override = (os.environ.get("FX_NEWSAPI_CACHE") or "").strip()
    if override:
        return Path(override)
    # output/ is gitignored; keeps successful pages across replay re-runs
    return Path(__file__).resolve().parents[2] / "output" / ".cache" / "newsapi"


def _newsapi_cache_key(
    query: str,
    *,
    start: date | None,
    end: date | None,
    domains: bool,
) -> str:
    # Intentionally omit pageSize/limit: pipeline uses max_items=30 while ad-hoc
    # fetches may use 25; same from/to window must share one cache entry.
    raw = "|".join(
        [
            query.strip(),
            start.isoformat() if start else "",
            end.isoformat() if end else "",
            "domains" if domains else "all",
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def _headlines_from_articles(articles: list[Any], limit: int) -> list[Headline]:
    out: list[Headline] = []
    for a in articles:
        if not isinstance(a, dict):
            continue
        title = (a.get("title") or "").strip()
        if not title or title == "[Removed]":
            continue
        src = a.get("source") or {}
        source = src.get("name") if isinstance(src, dict) else "NewsAPI"
        out.append(
            Headline(
                title=title,
                summary=(a.get("description") or "")[:500],
                source=source or "NewsAPI",
                url=(a.get("url") or "").strip(),
                published=_parse_dt(a.get("publishedAt")),
                provider="newsapi",
            )
        )
        if len(out) >= limit:
            break
    return out


def _newsapi_cache_get_ex(
    key: str,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    envelope: dict[str, Any] | None = None
    if key in _NEWSAPI_MEM_CACHE:
        envelope = _newsapi_normalize_envelope(_NEWSAPI_MEM_CACHE[key])
    else:
        path = _newsapi_cache_dir() / f"{key}.json"
        if not path.is_file():
            return None, None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None, None
        envelope = _newsapi_normalize_envelope(data)
        if envelope is not None:
            _NEWSAPI_MEM_CACHE[key] = dict(envelope)
    if envelope is None:
        return None, None
    if not _newsapi_envelope_fresh(envelope):
        _NEWSAPI_MEM_CACHE.pop(key, None)
        return None, None
    return list(envelope["articles"]), envelope


def _newsapi_cache_get(key: str) -> list[dict[str, Any]] | None:
    arts, _env = _newsapi_cache_get_ex(key)
    return arts


def _newsapi_cache_put(
    key: str,
    articles: list[dict[str, Any]],
    *,
    status: str = "ok",
    error: str | None = None,
    http_status: int | None = None,
    used_domains: bool | None = None,
) -> None:
    payload = [a for a in articles if isinstance(a, dict)]
    if status not in {"ok", "empty", "error"}:
        status = "empty" if not payload else "ok"
    if status == "ok" and not payload:
        status = "empty"
    envelope: dict[str, Any] = {
        "articles": payload,
        "status": status,
        "error": error,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "http_status": http_status,
        "used_domains": used_domains,
    }
    _NEWSAPI_MEM_CACHE[key] = dict(envelope)
    try:
        root = _newsapi_cache_dir()
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{key}.json").write_text(
            json.dumps(envelope, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def _newsapi_request_json(
    url: str,
    timeout: int,
    *,
    max_retries: int = 3,
) -> tuple[dict[str, Any] | None, str | None, int | None]:
    """GET NewsAPI JSON with 429 backoff. Returns (data, error, http_status)."""
    last_err: str | None = None
    last_status: int | None = None
    for attempt in range(max(1, int(max_retries))):
        try:
            data = _http_json(url, timeout)
            if isinstance(data, dict):
                return data, None, 200
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
            # Daily developer quota (100/24h) won't recover with short sleeps —
            # abort retries so replay does not burn remaining calls.
            if e.code == 429 and "rateLimited" in (body or ""):
                return None, last_err, last_status
            if e.code == 429 and attempt + 1 < max_retries:
                _NEWSAPI_SLEEP(min(60.0, 2.0 * (3**attempt)))
                continue
            return None, last_err, last_status
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            return None, last_err, last_status
    return None, last_err or "request_failed", last_status


def fetch_newsapi(
    query: str,
    cfg: dict[str, str],
    limit: int = 15,
    *,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    call_meta: dict[str, Any] | None = None,
) -> list[Headline]:
    """
    NewsAPI /v2/everything.

    Improvements vs silent-empty prior behavior:
      - process + disk cache (replay re-runs should not re-burn quota)
      - 429 exponential backoff
      - if domains filter returns ok-but-empty, retry without domains
      - surface last error via optional call_meta
    """
    if call_meta is not None:
        call_meta.clear()
        call_meta.update(
            {
                "error": None,
                "http_status": None,
                "from_cache": False,
                "used_domains": None,
                "total_results": None,
            }
        )
    if not is_set(cfg, "NEWSAPI_KEY"):
        if call_meta is not None:
            call_meta["error"] = "NEWSAPI_KEY not set"
        return []
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    page_size = min(int(limit), 50)
    extra = ""
    if start is not None:
        extra += f"&from={start.isoformat()}"
    if end is not None:
        extra += f"&to={end.isoformat()}"

    # Prefer cache from either domains or all-sources key.
    for use_domains in (True, False):
        key = _newsapi_cache_key(
            query, start=start, end=end, domains=use_domains
        )
        cached, envelope = _newsapi_cache_get_ex(key)
        if cached is not None and envelope is not None:
            if call_meta is not None:
                call_meta["from_cache"] = True
                call_meta["used_domains"] = envelope.get("used_domains", use_domains)
                call_meta["total_results"] = len(cached)
                call_meta["cache_status"] = envelope.get("status")
                if envelope.get("status") == "error":
                    call_meta["error"] = envelope.get("error") or "cached_error"
                    call_meta["http_status"] = envelope.get("http_status")
            if envelope.get("status") == "error":
                return []
            return _headlines_from_articles(cached, limit)

    timeout = timeout_s(cfg)
    api_key = cfg["NEWSAPI_KEY"]
    last_error: str | None = None
    last_status: int | None = None

    def build_url(*, domains: bool) -> str:
        base = (
            "https://newsapi.org/v2/everything?"
            f"q={quote_plus(query)}&language=en&sortBy=publishedAt"
            f"&pageSize={page_size}"
        )
        if domains:
            base += f"&domains={_NEWSAPI_DOMAINS}"
        return f"{base}{extra}&apiKey={api_key}"

    # 1) domains filter  2) if missing/empty/error → all sources
    for use_domains in (True, False):
        data, err, status = _newsapi_request_json(
            build_url(domains=use_domains), timeout, max_retries=3
        )
        if err:
            last_error = err
            last_status = status
            # On hard auth / daily quota errors, don't bother with the second variant.
            if status in {401, 403}:
                break
            if status == 429 and err and "rateLimited" in err:
                break
            continue
        assert data is not None
        if data.get("status") != "ok":
            last_error = (
                f"status={data.get('status')} code={data.get('code')} "
                f"msg={(data.get('message') or '')[:160]}"
            )
            last_status = status
            continue
        articles = list(data.get("articles") or [])
        total = data.get("totalResults")
        if call_meta is not None:
            call_meta["used_domains"] = use_domains
            call_meta["total_results"] = (
                int(total) if isinstance(total, int) else len(articles)
            )
        if not articles and use_domains:
            # ok-but-empty with domains: try unrestricted sources next
            continue
        key = _newsapi_cache_key(
            query, start=start, end=end, domains=use_domains
        )
        put_status = "empty" if not articles else "ok"
        _newsapi_cache_put(
            key,
            articles,
            status=put_status,
            http_status=status,
            used_domains=use_domains,
        )
        return _headlines_from_articles(articles, limit)

    if call_meta is not None:
        call_meta["error"] = last_error or "empty_or_failed"
        call_meta["http_status"] = last_status
    # Short TTL negative cache on the unrestricted key (shared miss).
    err_key = _newsapi_cache_key(query, start=start, end=end, domains=False)
    _newsapi_cache_put(
        err_key,
        [],
        status="error",
        error=last_error or "empty_or_failed",
        http_status=last_status,
        used_domains=False,
    )
    return []


def fetch_finnhub_forex(cfg: dict[str, str], limit: int = 15) -> list[Headline]:
    if not is_set(cfg, "FINNHUB_API_KEY"):
        return []
    url = f"https://finnhub.io/api/v1/news?category=forex&token={cfg['FINNHUB_API_KEY']}"
    try:
        data = _http_json(url, timeout_s(cfg))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: list[Headline] = []
    for a in data:
        title = (a.get("headline") or "").strip()
        if not title:
            continue
        ts = a.get("datetime")
        published = (
            datetime.fromtimestamp(ts, tz=timezone.utc)
            if isinstance(ts, (int, float))
            else None
        )
        out.append(
            Headline(
                title=title,
                summary=(a.get("summary") or "")[:500],
                source=(a.get("source") or "Finnhub"),
                url=(a.get("url") or "").strip(),
                published=published,
                provider="finnhub",
            )
        )
        if len(out) >= limit:
            break
    return out


def _pdf_to_headline(path: Path) -> Headline | None:
    text = ""
    if fitz is not None:
        try:
            doc = fitz.open(path)
            chunks = [page.get_text() for i, page in enumerate(doc) if i < 3]
            doc.close()
            text = "\n".join(chunks)
        except Exception:
            text = ""
    if not text.strip():
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    title = lines[0][:180] if lines else path.stem
    summary = " ".join(lines[1:12])[:600]
    return Headline(
        title=f"[inbox] {path.name}: {title}",
        summary=summary,
        source=f"inbox:{path.name}",
        url=path.resolve().as_uri(),
        published=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
        provider="inbox",
    )


def fetch_inbox_headlines(limit: int = 12) -> list[Headline]:
    out: list[Headline] = []
    for path in inbox_files():
        if path.suffix.lower() == ".pdf":
            h = _pdf_to_headline(path)
            if h:
                out.append(h)
        elif path.suffix.lower() in {".md", ".txt", ".csv"}:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            out.append(
                Headline(
                    title=f"[inbox] {path.name}: {(lines[0] if lines else path.stem)[:160]}",
                    summary=" ".join(lines[1:20])[:600],
                    source=f"inbox:{path.name}",
                    url=path.resolve().as_uri(),
                    published=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
                    provider="inbox",
                )
            )
        if len(out) >= limit:
            break
    return out[:limit]


def _dedupe_sort(headlines: list[Headline], max_items: int) -> list[Headline]:
    seen: set[str] = set()
    uniq: list[Headline] = []
    for h in headlines:
        key = re.sub(r"\W+", " ", h.title.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(h)

    def sort_key(h: Headline) -> tuple[int, float]:
        # Prefer official > free Google RSS > inbox > keyed APIs
        rank = {
            "official_rss": 0,
            "google_news_rss": 1,
            "ai_research": 2,
            "inbox": 3,
            "newsapi": 4,
            "gdelt": 5,
            "finnhub": 6,
        }.get(h.provider, 9)
        ts = h.published.timestamp() if h.published else 0.0
        return (rank, -ts)

    uniq.sort(key=sort_key)
    return uniq[:max_items]


def fetch_headlines_for_pair(
    pair: PairSpec | str,
    max_items: int = 25,
    *,
    allow_legacy: bool | None = None,  # ignored (compat)
) -> list[Headline]:
    """Official RSS + free Google News RSS + inbox + vault news APIs."""
    spec = get_pair(pair) if isinstance(pair, str) else pair
    cfg = load_config()
    queries = PAIR_NEWSAPI_QUERIES.get(spec.pair) or [spec.pair.replace("/", " ")]

    headlines: list[Headline] = []
    headlines.extend(fetch_official_rss(spec.pair))
    # Always try free public Google News RSS (helps when no NewsAPI/Finnhub key)
    headlines.extend(fetch_google_news_rss(spec.pair, limit=12))
    headlines.extend(fetch_inbox_headlines(limit=max_items))
    for q in queries[:2]:
        headlines.extend(fetch_newsapi(q, cfg, limit=12))
    headlines.extend(fetch_finnhub_forex(cfg, limit=12))

    return _dedupe_sort(headlines, max_items)


def fetch_historical_headlines_for_pair(
    pair: PairSpec | str,
    *,
    as_of_date: date | datetime | str,
    lookback_days: int = 60,
    max_items: int = 25,
    today: date | None = None,
) -> tuple[list[Headline], dict[str, Any]]:
    """
    Historical headline fetch with honest limitations.

    Only providers with real date constraints are used:
      - NewsAPI everything with from/to (free/dev ~last month)
      - GDELT DOC 2.0 ArtList with startdatetime/enddatetime (~last 3 months, free)
      - local inbox files whose mtime/published <= as_of

    We intentionally do NOT reuse current RSS / Google News RSS / Finnhub category
    feeds here because they do not provide trustworthy historical search by date.

    NewsAPI `from` is clamped into the plan searchable window (~last month from
    *today*). Replay often passes vol lookback=60; without clamping, the request
    fails silently and evidence_n stays 0 even when a shorter window has hits.
    GDELT is similarly clamped into its ~3-month DOC window.
    """
    # Lazy import avoids circular dependency (gdelt imports Headline from here).
    from fx_report.news.gdelt import (
        GDELT_MAX_HISTORY_DAYS,
        clamp_gdelt_from_date,
        fetch_gdelt_doc,
        gdelt_query_for_pair,
    )

    spec = get_pair(pair) if isinstance(pair, str) else pair
    cfg = load_config()
    as_of = _coerce_date(as_of_date)
    if as_of is None:
        raise ValueError("as_of_date is required for historical headline fetch")
    requested_lookback = max(int(lookback_days), 1)
    requested_start = as_of - timedelta(days=requested_lookback)
    newsapi_start, from_clamped = clamp_newsapi_from_date(
        requested_start,
        as_of,
        today=today,
    )
    gdelt_start, gdelt_from_clamped = clamp_gdelt_from_date(
        requested_start,
        as_of,
        today=today,
    )
    queries = PAIR_NEWSAPI_QUERIES.get(spec.pair) or [spec.pair.replace("/", " ")]

    headlines: list[Headline] = []
    inbox_all = fetch_inbox_headlines(limit=max_items * 2)
    inbox_dated = [
        h
        for h in inbox_all
        if h.published is not None and h.published.date() <= as_of
    ]
    headlines.extend(inbox_dated)
    inbox_dated_hits = len(inbox_dated)

    newsapi_hits = 0
    newsapi_skipped_outside_window = newsapi_start is None
    newsapi_errors: list[str] = []
    newsapi_http_status: int | None = None
    newsapi_from_cache = False
    # One query is enough for historical replay; extra queries burn free-tier quota.
    if is_set(cfg, "NEWSAPI_KEY") and newsapi_start is not None:
        for q in queries[:1]:
            call_meta: dict[str, Any] = {}
            batch = fetch_newsapi(
                q,
                cfg,
                limit=max_items,
                start_date=newsapi_start,
                end_date=as_of,
                call_meta=call_meta,
            )
            headlines.extend(
                [
                    h
                    for h in batch
                    if h.published is None
                    or newsapi_start <= h.published.date() <= as_of
                ]
            )
            newsapi_hits += len(batch)
            if call_meta.get("from_cache"):
                newsapi_from_cache = True
            err = call_meta.get("error")
            if err:
                newsapi_errors.append(str(err))
            status = call_meta.get("http_status")
            if isinstance(status, int):
                newsapi_http_status = status

    gdelt_hits = 0
    gdelt_skipped_outside_window = gdelt_start is None
    gdelt_errors: list[str] = []
    gdelt_http_status: int | None = None
    gdelt_from_cache = False
    gdelt_query = gdelt_query_for_pair(spec.pair)
    if gdelt_start is not None:
        gdelt_meta: dict[str, Any] = {}
        gdelt_batch = fetch_gdelt_doc(
            gdelt_query,
            start_date=gdelt_start,
            end_date=as_of,
            limit=max_items,
            timeout=timeout_s(cfg),
            call_meta=gdelt_meta,
        )
        headlines.extend(gdelt_batch)
        gdelt_hits = len(gdelt_batch)
        err = gdelt_meta.get("error")
        if err:
            gdelt_errors.append(str(err))
        status = gdelt_meta.get("http_status")
        if isinstance(status, int):
            gdelt_http_status = status
        if gdelt_meta.get("from_cache"):
            gdelt_from_cache = True

    # Prefer the widest date-filtered lookback actually used (NewsAPI or GDELT).
    effective_candidates: list[int] = []
    if newsapi_start is not None:
        effective_candidates.append((as_of - newsapi_start).days)
    if gdelt_start is not None:
        effective_candidates.append((as_of - gdelt_start).days)
    effective_lookback = max(effective_candidates) if effective_candidates else 0

    source_bits: list[str] = []
    if newsapi_hits > 0:
        source_bits.append(f"NewsAPI({newsapi_hits})")
    if gdelt_hits > 0:
        source_bits.append(f"GDELT({gdelt_hits})")
    if inbox_dated_hits > 0:
        source_bits.append(f"inbox({inbox_dated_hits})")
    sources_note = (
        "来源命中: " + "+".join(source_bits)
        if source_bits
        else "来源命中: 无"
    )

    limitation = (
        "历史新闻仅使用可日期过滤来源（NewsAPI、GDELT DOC）和本地 inbox 截止文件。"
        "当前 RSS / Google News RSS / Finnhub category / AI researcher 未用于历史回放，"
        "以避免把实时流误当成历史检索。"
        f" {sources_note}。"
    )
    if newsapi_skipped_outside_window:
        limitation += (
            f" as_of={as_of.isoformat()} 早于 NewsAPI 可检索窗口"
            f"（约近 {NEWSAPI_MAX_HISTORY_DAYS} 天），NewsAPI 无法日期过滤检索。"
        )
    elif from_clamped:
        limitation += (
            f" 已将 NewsAPI from 从 {requested_start.isoformat()} 钳制到"
            f" {newsapi_start.isoformat()}（开发者套餐约近"
            f" {NEWSAPI_MAX_HISTORY_DAYS} 天），避免 lookback={requested_lookback}"
            " 越界导致空结果。"
        )
    if newsapi_hits == 0 and newsapi_errors:
        limitation += f" NewsAPI 请求失败：{newsapi_errors[0][:180]}"
    elif (
        newsapi_hits == 0
        and is_set(cfg, "NEWSAPI_KEY")
        and not newsapi_skipped_outside_window
    ):
        limitation += " NewsAPI 在可检索窗口内返回 0 条（可能配额耗尽或该窗口无匹配）。"
    elif newsapi_hits == 0 and not is_set(cfg, "NEWSAPI_KEY"):
        limitation += " NewsAPI 未配置（无 KEY），已跳过。"

    if gdelt_skipped_outside_window:
        limitation += (
            f" as_of={as_of.isoformat()} 早于 GDELT DOC 可检索窗口"
            f"（约近 {GDELT_MAX_HISTORY_DAYS} 天），GDELT 无法日期过滤检索。"
        )
    elif gdelt_from_clamped:
        limitation += (
            f" 已将 GDELT from 从 {requested_start.isoformat()} 钳制到"
            f" {gdelt_start.isoformat()}（DOC 约近 {GDELT_MAX_HISTORY_DAYS} 天）。"
        )
    if gdelt_hits == 0 and gdelt_errors:
        limitation += f" GDELT 请求失败：{gdelt_errors[0][:180]}"
    elif gdelt_hits == 0 and not gdelt_skipped_outside_window:
        limitation += " GDELT 在可检索窗口内返回 0 条（可能限流或该窗口无匹配）。"

    date_filtered = (
        newsapi_hits > 0 or gdelt_hits > 0 or inbox_dated_hits > 0
    )

    meta: dict[str, Any] = {
        "historical_mode": True,
        "historical_as_of": as_of.isoformat(),
        "historical_lookback_days": requested_lookback,
        "historical_lookback_days_effective": int(effective_lookback),
        "newsapi_from": newsapi_start.isoformat() if newsapi_start is not None else None,
        "newsapi_from_requested": requested_start.isoformat(),
        "newsapi_from_clamped": bool(from_clamped),
        "newsapi_outside_window": bool(newsapi_skipped_outside_window),
        "providers_used": sorted({h.provider for h in headlines}),
        "newsapi_enabled": bool(is_set(cfg, "NEWSAPI_KEY")),
        "newsapi_hits": newsapi_hits,
        "newsapi_error": newsapi_errors[0] if newsapi_errors else None,
        "newsapi_http_status": newsapi_http_status,
        "newsapi_from_cache": bool(newsapi_from_cache),
        "gdelt_enabled": True,
        "gdelt_hits": gdelt_hits,
        "gdelt_from": gdelt_start.isoformat() if gdelt_start is not None else None,
        "gdelt_from_requested": requested_start.isoformat(),
        "gdelt_from_clamped": bool(gdelt_from_clamped),
        "gdelt_outside_window": bool(gdelt_skipped_outside_window),
        "gdelt_error": gdelt_errors[0] if gdelt_errors else None,
        "gdelt_http_status": gdelt_http_status,
        "gdelt_from_cache": bool(gdelt_from_cache),
        "gdelt_query": gdelt_query,
        "inbox_dated_hits": inbox_dated_hits,
        "historical_news_quality": "date_filtered" if date_filtered else "limited",
        "limitation": limitation,
    }
    return _dedupe_sort(headlines, max_items), meta


def headlines_to_json(headlines: list[Headline]) -> str:
    return json.dumps([h.to_dict() for h in headlines], ensure_ascii=False, indent=2)


def fetch_status_summary() -> str:
    cfg = load_config()
    paths = vault_paths(cfg)
    parts = [
        "market=ECB/Frankfurter(+FRED/Twelve/Alpha)",
        "news=Fed/RBA/ECB/BOE RSS + Google News RSS (free)",
        f"NewsAPI={'ON' if is_set(cfg, 'NEWSAPI_KEY') else 'off'}",
        "GDELT=ON(free)",
        f"Finnhub={'ON' if is_set(cfg, 'FINNHUB_API_KEY') else 'off'}",
        f"inbox={len(inbox_files(cfg))}",
        f"vault={paths['root'].name}",
    ]
    return " | ".join(parts)
