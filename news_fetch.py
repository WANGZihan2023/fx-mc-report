"""Fetch FX-relevant headlines from Yahoo Finance + Google News RSS."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen

from pairs import PairSpec, get_pair

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


@dataclass
class Headline:
    title: str
    summary: str
    source: str
    url: str
    published: datetime | None
    provider: str  # yahoo | google_rss | other

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["published"] = self.published.isoformat() if self.published else None
        return d


# Pair → search queries / related Yahoo tickers for news
PAIR_NEWS_QUERIES: dict[str, list[str]] = {
    "USD/AUD": ["AUD USD", "Australian dollar RBA", "Aussie dollar iron ore"],
    "AUD/USD": ["AUD USD", "Australian dollar RBA", "Aussie dollar"],
    "EUR/USD": ["EUR USD", "euro ECB rates", "euro dollar"],
    "GBP/USD": ["GBP USD", "pound sterling BOE", "cable forex"],
    "USD/JPY": ["USD JPY", "yen BOJ", "dollar yen"],
    "USD/CNH": [
        "USD CNH offshore yuan",
        "PBOC yuan fixing",
        "China yuan dollar",
        "renminbi CNH",
    ],
    "USD/CNY": [
        "USD CNY yuan",
        "PBOC middle rate",
        "China currency policy",
        "renminbi onshore",
    ],
    "USD/CAD": ["USD CAD", "Canadian dollar oil", "loonie Fed"],
    "NZD/USD": ["NZD USD", "kiwi dollar RBNZ", "New Zealand dollar"],
    "USD/CHF": ["USD CHF", "Swiss franc SNB", "dollar franc"],
}

PAIR_YAHOO_NEWS_TICKERS: dict[str, list[str]] = {
    "USD/AUD": ["AUDUSD=X", "DX-Y.NYB"],
    "AUD/USD": ["AUDUSD=X", "DX-Y.NYB"],
    "EUR/USD": ["EURUSD=X", "DX-Y.NYB"],
    "GBP/USD": ["GBPUSD=X", "DX-Y.NYB"],
    "USD/JPY": ["USDJPY=X", "DX-Y.NYB"],
    "USD/CNH": ["USDCNY=X", "FXI", "DX-Y.NYB"],  # CNH ticker often has no news
    "USD/CNY": ["USDCNY=X", "FXI", "DX-Y.NYB"],
    "USD/CAD": ["USDCAD=X", "CL=F", "DX-Y.NYB"],
    "NZD/USD": ["NZDUSD=X", "DX-Y.NYB"],
    "USD/CHF": ["USDCHF=X", "DX-Y.NYB"],
}


def _http_get(url: str, timeout: int = 20) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; FXReportBot/1.0)",
            "Accept": "application/rss+xml,application/xml,text/xml,*/*",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # ISO
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


def fetch_yahoo_ticker_news(tickers: list[str], limit: int = 20) -> list[Headline]:
    if yf is None:
        return []
    out: list[Headline] = []
    seen: set[str] = set()
    for t in tickers:
        try:
            items = yf.Ticker(t).news or []
        except Exception:
            continue
        for item in items:
            c = item.get("content") if isinstance(item, dict) else None
            if not isinstance(c, dict):
                continue
            title = (c.get("title") or "").strip()
            if not title or title in seen:
                continue
            seen.add(title)
            provider = c.get("provider") or {}
            source = ""
            if isinstance(provider, dict):
                source = provider.get("displayName") or provider.get("sourceId") or ""
            url = ""
            for key in ("canonicalUrl", "clickThroughUrl"):
                u = c.get(key)
                if isinstance(u, dict) and u.get("url"):
                    url = u["url"]
                    break
                if isinstance(u, str) and u.startswith("http"):
                    url = u
                    break
            summary = (c.get("summary") or c.get("description") or "").strip()
            published = _parse_dt(c.get("pubDate") or c.get("displayTime"))
            out.append(
                Headline(
                    title=title,
                    summary=summary,
                    source=source or "Yahoo Finance",
                    url=url,
                    published=published,
                    provider="yahoo",
                )
            )
            if len(out) >= limit:
                return out
    return out


def fetch_google_news_rss(query: str, limit: int = 15) -> list[Headline]:
    url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        raw = _http_get(url)
    except Exception:
        return []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []

    out: list[Headline] = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        # Google titles often end with " - Source"
        source = ""
        if " - " in title:
            parts = title.rsplit(" - ", 1)
            title, source = parts[0].strip(), parts[1].strip()
        link = (item.findtext("link") or "").strip()
        pub = _parse_dt(item.findtext("pubDate"))
        desc = re.sub(r"<[^>]+>", "", item.findtext("description") or "").strip()
        out.append(
            Headline(
                title=title,
                summary=desc[:400],
                source=source or "Google News",
                url=link,
                published=pub,
                provider="google_rss",
            )
        )
        if len(out) >= limit:
            break
    return out


def fetch_headlines_for_pair(pair: PairSpec | str, max_items: int = 25) -> list[Headline]:
    spec = get_pair(pair) if isinstance(pair, str) else pair
    queries = PAIR_NEWS_QUERIES.get(spec.pair) or [spec.pair.replace("/", " ")]
    tickers = PAIR_YAHOO_NEWS_TICKERS.get(spec.pair) or [spec.yahoo_ticker]

    headlines: list[Headline] = []
    headlines.extend(fetch_yahoo_ticker_news(tickers, limit=max_items))
    for q in queries[:4]:
        headlines.extend(fetch_google_news_rss(q, limit=12))

    # Deduplicate by normalized title
    seen: set[str] = set()
    uniq: list[Headline] = []
    for h in headlines:
        key = re.sub(r"\W+", " ", h.title.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(h)

    def sort_key(h: Headline) -> float:
        if h.published is None:
            return 0.0
        return h.published.timestamp()

    uniq.sort(key=sort_key, reverse=True)
    return uniq[:max_items]


def headlines_to_json(headlines: list[Headline]) -> str:
    return json.dumps([h.to_dict() for h in headlines], ensure_ascii=False, indent=2)
