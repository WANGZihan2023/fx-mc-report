"""Fetch FX headlines from official / vault sources only.

Priority:
  1) Central-bank / official RSS (Fed, RBA, ECB, BOE, …)
  2) inbox/ research PDFs / notes
  3) NewsAPI / Finnhub if vault keys are set
"""

from __future__ import annotations

import json
import re
import ssl
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
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
    provider: str  # official_rss | newsapi | finnhub | inbox

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


def fetch_newsapi(
    query: str,
    cfg: dict[str, str],
    limit: int = 15,
    *,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
) -> list[Headline]:
    if not is_set(cfg, "NEWSAPI_KEY"):
        return []
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    extra = ""
    if start is not None:
        extra += f"&from={start.isoformat()}"
    if end is not None:
        extra += f"&to={end.isoformat()}"
    # Prefer top domain sources when possible
    domains = "reuters.com,bloomberg.com,wsj.com,ft.com,afr.com,cnbc.com,federalreserve.gov"
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={quote_plus(query)}&language=en&sortBy=publishedAt"
        f"&pageSize={min(limit, 50)}&domains={domains}"
        f"{extra}"
        f"&apiKey={cfg['NEWSAPI_KEY']}"
    )
    try:
        data = _http_json(url, timeout_s(cfg))
    except Exception:
        return []
    if not isinstance(data, dict) or data.get("status") != "ok":
        # retry without domain filter
        url2 = (
            "https://newsapi.org/v2/everything?"
            f"q={quote_plus(query)}&language=en&sortBy=publishedAt"
            f"&pageSize={min(limit, 50)}{extra}&apiKey={cfg['NEWSAPI_KEY']}"
        )
        try:
            data = _http_json(url2, timeout_s(cfg))
        except Exception:
            return []
    if not isinstance(data, dict) or data.get("status") != "ok":
        return []
    out: list[Headline] = []
    for a in data.get("articles") or []:
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
            "finnhub": 5,
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
) -> tuple[list[Headline], dict[str, Any]]:
    """
    Historical headline fetch with honest limitations.

    Only providers with real date constraints are used:
      - NewsAPI everything with from/to
      - local inbox files whose mtime <= as_of

    We intentionally do NOT reuse current RSS / Google News RSS / Finnhub category
    feeds here because they do not provide trustworthy historical search by date.
    """
    spec = get_pair(pair) if isinstance(pair, str) else pair
    cfg = load_config()
    as_of = _coerce_date(as_of_date)
    if as_of is None:
        raise ValueError("as_of_date is required for historical headline fetch")
    start = as_of - timedelta(days=max(int(lookback_days), 1))
    queries = PAIR_NEWSAPI_QUERIES.get(spec.pair) or [spec.pair.replace("/", " ")]

    headlines: list[Headline] = []
    inbox_all = fetch_inbox_headlines(limit=max_items * 2)
    headlines.extend(
        [
            h
            for h in inbox_all
            if h.published is not None and h.published.date() <= as_of
        ]
    )
    newsapi_hits = 0
    if is_set(cfg, "NEWSAPI_KEY"):
        for q in queries[:2]:
            batch = fetch_newsapi(
                q,
                cfg,
                limit=max_items,
                start_date=start,
                end_date=as_of,
            )
            headlines.extend(
                [
                    h
                    for h in batch
                    if h.published is None or start <= h.published.date() <= as_of
                ]
            )
            newsapi_hits += len(batch)

    meta: dict[str, Any] = {
        "historical_mode": True,
        "historical_as_of": as_of.isoformat(),
        "historical_lookback_days": int(lookback_days),
        "providers_used": sorted({h.provider for h in headlines}),
        "newsapi_enabled": bool(is_set(cfg, "NEWSAPI_KEY")),
        "newsapi_hits": newsapi_hits,
        "historical_news_quality": "date_filtered" if newsapi_hits > 0 else "limited",
        "limitation": (
            "历史新闻仅使用可日期过滤来源（NewsAPI）和本地 inbox 截止文件。"
            "当前 RSS / Google News RSS / Finnhub category / AI researcher 未用于历史回放，"
            "以避免把实时流误当成历史检索。"
        ),
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
        f"Finnhub={'ON' if is_set(cfg, 'FINNHUB_API_KEY') else 'off'}",
        f"inbox={len(inbox_files(cfg))}",
        f"vault={paths['root'].name}",
    ]
    return " | ".join(parts)
