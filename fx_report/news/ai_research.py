"""
AI 检索员：按货币对信息需求，像人工一样搜索公开源并抽取投行/宏观展望。

能力（有什么用什么）：
  1) 公开投行/研究白名单页直接抓取
  2) NewsAPI / Tavily / Brave 搜索（.env 填 Key）
  3) LLM（Ollama / OpenAI 兼容）把材料收成结构化语句

不破解付费墙；打不开的源会跳过并记入 meta。
"""

from __future__ import annotations

import json
import re
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any
from urllib.request import Request, urlopen

from fx_report.config.api_config import is_set, load_config, timeout_s
from fx_report.news.fetch import Headline
from fx_report.news.llm import LLMConfig, resolve_llm_config
from fx_report.market.pairs import PairSpec

# 可公开尝试的投行/研究页（按货币对选）
BANK_PUBLIC_URLS: dict[str, list[tuple[str, str]]] = {
    # (机构名, URL模板或固定URL)
    "*": [
        ("ING Think FX Daily", "https://think.ing.com/articles/fx-daily-dollar-upside-risks-are-rising-rapidly/"),
        ("ING G10 FX Talking", "https://think.ing.com/articles/g10-fx-talking-july-2026/"),
        ("MUFG FX Outlook", "https://www.mufgresearch.com/fx/monthly-foreign-exchange-outlook-june-2026/"),
        ("UBS CIO Daily", "https://www.ubs.com/global/en/wealthmanagement/insights/chief-investment-office/house-view/daily.html"),
    ],
    "AUD": [
        (
            "StoneX AUD/USD Outlook",
            "https://www.stonex.com/en/insights/aud-usd-outlook-h2-2026-key-drivers-for-the-australian-dollar-in-q3/",
        ),
        (
            "AMP Aussie dollar",
            "https://www.amp.com.au/resources/insights-hub/econosights-australian-dollar",
        ),
    ],
    "EUR": [
        ("ING FX Daily", "https://think.ing.com/articles/fx-daily-dollar-upside-risks-are-rising-rapidly/"),
    ],
    "CNH": [
        ("Vanguard China", "https://corporate.vanguard.com/content/corporatesite/us/en/corp/vemo/vemo-china.html"),
    ],
}


@dataclass
class ResearchHit:
    title: str
    url: str
    source: str
    snippet: str
    provider: str  # whitelist | newsapi | tavily | brave | llm


@dataclass
class AIResearchResult:
    headlines: list[Headline] = field(default_factory=list)
    hits: list[ResearchHit] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


def _http_get(url: str, timeout: int = 20) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/json,*/*",
        },
    )
    with urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        return resp.read()


def _http_json(url: str, timeout: int = 20, data: bytes | None = None, headers: dict | None = None) -> Any:
    hdrs = {
        "User-Agent": "FX-AI-Researcher/1.0",
        "Accept": "application/json",
    }
    if headers:
        hdrs.update(headers)
    req = Request(url, data=data, headers=hdrs, method="POST" if data else "GET")
    with urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            t = data.strip()
            if t:
                self.parts.append(t)


def _html_to_text(html: str, max_chars: int = 4000) -> str:
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:
        pass
    text = re.sub(r"\s+", " ", " ".join(p.parts)).strip()
    return text[:max_chars]


def _queries_for_pair(spec: PairSpec, info_need_ids: list[str]) -> list[str]:
    pair = spec.pair
    q = [
        f"{pair} forecast outlook bank",
        f"{spec.base} {spec.quote} FX outlook MUFG OR ING OR UBS OR Goldman",
    ]
    if "oil" in info_need_ids or "geopolitics" in info_need_ids:
        q.append(f"{pair} oil geopolitics dollar")
    if "china_iron" in info_need_ids:
        q.append("iron ore price Australia dollar AUD")
    if "rba" in info_need_ids:
        q.append("RBA cash rate AUD USD outlook")
    if "fed" in info_need_ids or "cpi" in info_need_ids:
        q.append("Fed rate hike odds CPI USD")
    if "pboc" in info_need_ids:
        q.append("PBOC yuan CNH outlook")
    return q[:5]


def _whitelist_urls(spec: PairSpec) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    out.extend(BANK_PUBLIC_URLS.get("*", []))
    for ccy in (spec.base, spec.quote):
        out.extend(BANK_PUBLIC_URLS.get(ccy, []))
    # dedupe by url
    seen: set[str] = set()
    uniq: list[tuple[str, str]] = []
    for name, url in out:
        if url in seen:
            continue
        seen.add(url)
        uniq.append((name, url))
    return uniq


def search_newsapi(query: str, cfg: dict[str, str], limit: int = 8) -> list[ResearchHit]:
    if not is_set(cfg, "NEWSAPI_KEY"):
        return []
    domains = "reuters.com,bloomberg.com,cnbc.com,afr.com,ft.com,wsj.com,think.ing.com"
    url = (
        "https://newsapi.org/v2/everything?"
        + urllib.parse.urlencode(
            {
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": str(min(limit, 20)),
                "domains": domains,
                "apiKey": cfg["NEWSAPI_KEY"],
            }
        )
    )
    try:
        data = _http_json(url, timeout_s(cfg))
    except Exception:
        return []
    if not isinstance(data, dict) or data.get("status") != "ok":
        return []
    hits: list[ResearchHit] = []
    for a in data.get("articles") or []:
        title = (a.get("title") or "").strip()
        if not title or title == "[Removed]":
            continue
        src = a.get("source") or {}
        hits.append(
            ResearchHit(
                title=title,
                url=(a.get("url") or "").strip(),
                source=(src.get("name") if isinstance(src, dict) else None) or "NewsAPI",
                snippet=(a.get("description") or "")[:400],
                provider="newsapi",
            )
        )
    return hits


def search_tavily(query: str, cfg: dict[str, str], limit: int = 6) -> list[ResearchHit]:
    if not is_set(cfg, "TAVILY_API_KEY"):
        return []
    payload = json.dumps(
        {
            "api_key": cfg["TAVILY_API_KEY"],
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "max_results": limit,
        }
    ).encode("utf-8")
    try:
        data = _http_json(
            "https://api.tavily.com/search",
            timeout_s(cfg),
            data=payload,
            headers={"Content-Type": "application/json"},
        )
    except Exception:
        return []
    hits: list[ResearchHit] = []
    for r in data.get("results") or []:
        hits.append(
            ResearchHit(
                title=(r.get("title") or "").strip() or "Tavily result",
                url=(r.get("url") or "").strip(),
                source="Tavily",
                snippet=(r.get("content") or "")[:400],
                provider="tavily",
            )
        )
    return hits


def search_brave(query: str, cfg: dict[str, str], limit: int = 6) -> list[ResearchHit]:
    if not is_set(cfg, "BRAVE_SEARCH_API_KEY"):
        return []
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(
        {"q": query, "count": str(limit)}
    )
    try:
        req = Request(
            url,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": cfg["BRAVE_SEARCH_API_KEY"],
            },
        )
        with urlopen(req, timeout=timeout_s(cfg), context=_ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return []
    hits: list[ResearchHit] = []
    for r in (data.get("web") or {}).get("results") or []:
        hits.append(
            ResearchHit(
                title=(r.get("title") or "").strip(),
                url=(r.get("url") or "").strip(),
                source="Brave",
                snippet=(r.get("description") or "")[:400],
                provider="brave",
            )
        )
    return hits


def fetch_whitelist(spec: PairSpec, cfg: dict[str, str]) -> list[ResearchHit]:
    hits: list[ResearchHit] = []
    for name, url in _whitelist_urls(spec):
        try:
            raw = _http_get(url, timeout_s(cfg))
            html = raw.decode("utf-8", errors="ignore")
            text = _html_to_text(html, 3500)
            if len(text) < 80:
                continue
            title_m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
            title = re.sub(r"\s+", " ", title_m.group(1)).strip() if title_m else name
            hits.append(
                ResearchHit(
                    title=title[:180],
                    url=url,
                    source=name,
                    snippet=text[:800],
                    provider="whitelist",
                )
            )
        except Exception:
            continue
    return hits


def _llm_extract_outlooks(
    spec: PairSpec,
    hits: list[ResearchHit],
    llm: LLMConfig,
) -> list[Headline]:
    if not hits:
        return []
    # Pack materials for LLM
    materials = []
    for i, h in enumerate(hits[:12], 1):
        materials.append(
            f"[{i}] source={h.source} provider={h.provider}\n"
            f"title={h.title}\nurl={h.url}\ntext={h.snippet[:700]}\n"
        )
    system = (
        "You are an FX research assistant. Extract bank/institution outlooks and "
        "market-moving facts for the given currency pair. "
        "Return ONLY JSON: {\"items\":[{\"title\":str,\"summary\":str,\"source\":str,\"url\":str,"
        "\"bank\":str,\"relevance\":str}]}. "
        "Prefer investment-bank or official forecasts with target levels if present. "
        "If nothing useful, return {\"items\":[]}."
    )
    user = (
        f"Currency pair: {spec.pair}\nDrivers: {', '.join(spec.default_drivers)}\n\n"
        f"Materials:\n{''.join(materials)}"
    )
    # reuse news_llm chat path without importing private if possible
    from fx_report.news.llm import _chat_json

    try:
        data = _chat_json(llm, system, user)
    except Exception:
        # fallback: turn hits into headlines directly
        return [
            Headline(
                title=f"[AI research] {h.title}",
                summary=h.snippet[:400],
                source=h.source,
                url=h.url,
                published=datetime.now(timezone.utc),
                provider="ai_research",
            )
            for h in hits[:8]
        ]

    # _chat_json returns parsed model JSON object when successful
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        # sometimes wrapped
        if isinstance(data, dict) and "choices" in data:
            return []
        items = []

    out: list[Headline] = []
    for it in items[:12]:
        if not isinstance(it, dict):
            continue
        title = (it.get("title") or "").strip()
        if not title:
            continue
        bank = (it.get("bank") or "").strip()
        src = (it.get("source") or bank or "AI research").strip()
        summary = (it.get("summary") or it.get("relevance") or "").strip()
        if bank and bank.lower() not in title.lower():
            title = f"{bank}: {title}"
        out.append(
            Headline(
                title=f"[AI research] {title}"[:220],
                summary=summary[:500],
                source=src,
                url=(it.get("url") or "").strip(),
                published=datetime.now(timezone.utc),
                provider="ai_research",
            )
        )
    return out


def run_ai_research(
    spec: PairSpec,
    *,
    info_need_ids: list[str] | None = None,
    llm_cfg: LLMConfig | None = None,
    max_headlines: int = 12,
) -> AIResearchResult:
    """
    AI 检索员主入口。
    - 无 LLM 且无搜索 Key 时：仍尝试白名单页
    - 有 LLM：把材料收成投行/展望语句
    """
    cfg = load_config()
    need_ids = info_need_ids or list(spec.default_drivers)
    meta: dict[str, Any] = {
        "enabled": True,
        "queries": [],
        "whitelist_ok": 0,
        "search_hits": 0,
        "errors": [],
    }

    hits: list[ResearchHit] = []

    # 1) whitelist pages (no search key needed)
    wl = fetch_whitelist(spec, cfg)
    hits.extend(wl)
    meta["whitelist_ok"] = len(wl)

    # 2) search APIs
    queries = _queries_for_pair(spec, need_ids)
    meta["queries"] = queries
    for q in queries:
        before = len(hits)
        hits.extend(search_tavily(q, cfg, limit=5))
        hits.extend(search_brave(q, cfg, limit=5))
        hits.extend(search_newsapi(q, cfg, limit=6))
        if len(hits) == before:
            meta["errors"].append(f"no_search_hits:{q[:40]}")

    # dedupe by url/title
    seen: set[str] = set()
    uniq: list[ResearchHit] = []
    for h in hits:
        key = (h.url or h.title).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(h)
    hits = uniq
    meta["search_hits"] = len(hits)

    llm = llm_cfg or resolve_llm_config()
    meta["llm"] = bool(llm)
    if llm and hits:
        headlines = _llm_extract_outlooks(spec, hits, llm)
    else:
        headlines = [
            Headline(
                title=f"[AI research] {h.title}",
                summary=h.snippet[:400],
                source=h.source,
                url=h.url,
                published=datetime.now(timezone.utc),
                provider="ai_research",
            )
            for h in hits[:max_headlines]
        ]
        if not llm:
            meta["errors"].append("no_llm_config_used_raw_hits")

    meta["headlines_out"] = len(headlines)
    return AIResearchResult(headlines=headlines[:max_headlines], hits=hits, meta=meta)
