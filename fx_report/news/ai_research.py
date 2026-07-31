"""
AI 检索员：模仿人工「一条条」检索公开源，再抽投行/宏观展望。

分工：
  · 脑 = LLM（DeepSeek / Groq / Ollama …）——拟下一句搜索词、挑选有用标题、收成语句
  · 手 = 搜索 API / 公开 RSS —— Tavily → Brave → NewsAPI → Google News RSS（免费无 Key）
  · 另：公开投行白名单页直接抓取（不依赖搜索 Key）

诚实约定：
  · 不发明 URL；References 只保留搜索/白名单真正返回的链接
  · 只填 DeepSeek、没有搜索手时：只能吃白名单 +（若有）Google News；并在 meta 里写清限制
  · 不破解付费墙；打不开的源跳过并记入 meta
"""

from __future__ import annotations

import json
import re
import ssl
import urllib.parse
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from typing import Any
from urllib.request import Request, urlopen

from fx_report.config.api_config import SEARCH_KEYS, is_set, load_config, timeout_s
from fx_report.news.fetch import Headline
from fx_report.news.llm import LLMConfig, resolve_llm_config
from fx_report.market.pairs import PairSpec

# 可公开尝试的投行/研究页（按货币对选；URL 失效时自动跳过）
BANK_PUBLIC_URLS: dict[str, list[tuple[str, str]]] = {
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

DEFAULT_MAX_ROUNDS = 4
DEFAULT_TARGET_KEEP = 20
# When Tavily/Brave present: aim for Torchcast-like volume (still real URLs only)
TAVILY_MAX_ROUNDS = 7
TAVILY_TARGET_KEEP = 40
TAVILY_SEARCH_LIMIT = 8


@dataclass
class ResearchHit:
    title: str
    url: str
    source: str
    snippet: str
    provider: str  # whitelist | newsapi | tavily | brave | google_news_rss | llm


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


def _seed_queries(spec: PairSpec, info_need_ids: list[str]) -> list[str]:
    """Fallback query list when LLM planner unavailable — still one-at-a-time."""
    pair = spec.pair
    q = [
        f"{pair} forecast outlook bank",
        f"{spec.base} {spec.quote} FX outlook MUFG OR ING OR UBS OR Goldman",
        f"{pair} central bank policy outlook",
        f"{spec.base} {spec.quote} currency forecast Reuters OR Bloomberg",
    ]
    if "oil" in info_need_ids or "geopolitics" in info_need_ids:
        q.append(f"{pair} oil geopolitics dollar")
        q.append(f"geopolitical risk USD safe haven {spec.quote}")
    if "china_iron" in info_need_ids:
        q.append("iron ore price Australia dollar AUD")
        q.append("China steel demand iron ore AUD")
    if "china_growth" in info_need_ids:
        q.append("China stimulus growth yuan CNH FX")
    if "rba" in info_need_ids:
        q.append("RBA cash rate AUD USD outlook")
    if "fed" in info_need_ids or "cpi" in info_need_ids:
        q.append("Fed rate hike odds CPI USD")
        q.append("US inflation CPI Federal Reserve dollar")
    if "pboc" in info_need_ids:
        q.append("PBOC yuan CNH outlook")
    if "ecb" in info_need_ids:
        q.append("ECB rates EUR USD outlook")
    if "boj" in info_need_ids:
        q.append("BOJ policy JPY USD outlook")
    # Dedup preserve order
    seen: set[str] = set()
    out: list[str] = []
    for item in q:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out[:12]


def _whitelist_urls(spec: PairSpec) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    out.extend(BANK_PUBLIC_URLS.get("*", []))
    for ccy in (spec.base, spec.quote):
        out.extend(BANK_PUBLIC_URLS.get(ccy, []))
    seen: set[str] = set()
    uniq: list[tuple[str, str]] = []
    for name, url in out:
        if url in seen:
            continue
        seen.add(url)
        uniq.append((name, url))
    return uniq


def search_hands_available(cfg: dict[str, str] | None = None) -> dict[str, bool]:
    """Which search backends can act as「手」."""
    cfg = cfg or load_config()
    return {
        "tavily": is_set(cfg, "TAVILY_API_KEY"),
        "brave": is_set(cfg, "BRAVE_SEARCH_API_KEY"),
        "newsapi": is_set(cfg, "NEWSAPI_KEY"),
        "google_news_rss": True,  # free, no key
    }


def has_paid_search_api(cfg: dict[str, str] | None = None) -> bool:
    cfg = cfg or load_config()
    return any(is_set(cfg, k) for k in SEARCH_KEYS)


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


def search_google_news_rss(query: str, limit: int = 6) -> list[ResearchHit]:
    """Free Google News RSS — no API key. Used as fallback「手」."""
    url = (
        "https://news.google.com/rss/search?"
        + urllib.parse.urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    )
    try:
        raw = _http_get(url, timeout=15)
    except Exception:
        return []
    # Minimal RSS item parse (avoid importing private helpers)
    hits: list[ResearchHit] = []
    for m in re.finditer(r"<item>(.*?)</item>", raw.decode("utf-8", errors="ignore"), re.I | re.S):
        block = m.group(1)
        title_m = re.search(r"<title[^>]*><!\[CDATA\[(.*?)\]\]></title>|<title[^>]*>(.*?)</title>", block, re.I | re.S)
        link_m = re.search(r"<link[^>]*>(.*?)</link>", block, re.I | re.S)
        desc_m = re.search(r"<description[^>]*><!\[CDATA\[(.*?)\]\]></description>|<description[^>]*>(.*?)</description>", block, re.I | re.S)
        title = ""
        if title_m:
            title = (title_m.group(1) or title_m.group(2) or "").strip()
        link = (link_m.group(1).strip() if link_m else "")
        desc = ""
        if desc_m:
            desc = re.sub(r"<[^>]+>", " ", (desc_m.group(1) or desc_m.group(2) or ""))
            desc = re.sub(r"\s+", " ", desc).strip()[:400]
        if not title:
            continue
        hits.append(
            ResearchHit(
                title=title[:180],
                url=link,
                source="Google News",
                snippet=desc,
                provider="google_news_rss",
            )
        )
        if len(hits) >= limit:
            break
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


def execute_search(query: str, cfg: dict[str, str], *, limit: int = 6) -> tuple[list[ResearchHit], list[str]]:
    """
    Run one human-like search round with available hands (priority order).
    Returns (hits, providers_used).
    """
    hits: list[ResearchHit] = []
    used: list[str] = []
    for name, fn in (
        ("tavily", lambda: search_tavily(query, cfg, limit=limit)),
        ("brave", lambda: search_brave(query, cfg, limit=limit)),
        ("newsapi", lambda: search_newsapi(query, cfg, limit=limit)),
    ):
        batch = fn()
        if batch:
            hits.extend(batch)
            used.append(name)
    # Free fallback always tried if paid hands empty or thin
    if len(hits) < 3:
        gn = search_google_news_rss(query, limit=limit)
        if gn:
            hits.extend(gn)
            used.append("google_news_rss")
    return hits, used


def _dedupe_hits(hits: list[ResearchHit]) -> list[ResearchHit]:
    seen: set[str] = set()
    uniq: list[ResearchHit] = []
    for h in hits:
        key = (h.url or h.title).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(h)
    return uniq


def _allowed_urls(hits: list[ResearchHit]) -> set[str]:
    return {h.url.strip() for h in hits if (h.url or "").strip().startswith("http")}


def _rank_hits_for_refs(hits: list[ResearchHit]) -> list[ResearchHit]:
    """Prefer URL+snippet and stable providers over bare / fragile Google News links."""
    from fx_report.news.urls import is_fragile_url, provider_stability_rank

    def key(h: ResearchHit) -> tuple:
        has_url = 0 if (h.url or "").startswith("http") else 1
        snip_len = len((h.snippet or "").strip())
        has_snip = 0 if snip_len >= 40 else 1
        fragile = 1 if is_fragile_url(h.url) else 0
        return (
            has_url,
            has_snip,
            fragile,
            provider_stability_rank(h.provider),
            -snip_len,
        )

    return sorted(hits, key=key)


def live_research_budget(cfg: dict[str, str] | None = None) -> dict[str, int]:
    """
    Live-report keep/rounds budget. Historical cheap path must not call this
    with Tavily burning — pipeline already disables AI there.
    """
    cfg = cfg or load_config()
    hands = search_hands_available(cfg)
    rich = bool(hands.get("tavily") or hands.get("brave"))
    if rich:
        return {
            "max_rounds": TAVILY_MAX_ROUNDS,
            "target_keep": TAVILY_TARGET_KEEP,
            "search_limit": TAVILY_SEARCH_LIMIT,
            "max_headlines": TAVILY_TARGET_KEEP,
        }
    # NewsAPI / free RSS only — modest bump over legacy 10
    return {
        "max_rounds": DEFAULT_MAX_ROUNDS + 1,
        "target_keep": DEFAULT_TARGET_KEEP,
        "search_limit": 6,
        "max_headlines": max(DEFAULT_TARGET_KEEP, 24),
    }


def _llm_plan_next_query(
    spec: PairSpec,
    need_ids: list[str],
    kept: list[ResearchHit],
    tried_queries: list[str],
    round_i: int,
    max_rounds: int,
    llm: LLMConfig,
) -> dict[str, Any]:
    """Brain: decide next search query or stop."""
    from fx_report.news.llm import _chat_json

    found = []
    for i, h in enumerate(kept[:10], 1):
        found.append(f"[{i}] {h.source}: {h.title[:120]}")
    system = (
        "You are an FX research librarian. Mimic a human who searches one query at a time. "
        "Return ONLY JSON: "
        '{"action":"search"|"stop","query":str,"reason":str}. '
        "Prefer bank outlooks, central-bank, CPI, oil/geopolitics relevant to the pair. "
        "Do not invent URLs. If enough material exists or queries exhausted, action=stop."
    )
    user = (
        f"Pair: {spec.pair}\nDrivers needed: {', '.join(need_ids) or 'macro'}\n"
        f"Round {round_i + 1}/{max_rounds}\n"
        f"Already tried queries: {tried_queries or ['(none)']}\n"
        f"Kept headlines so far ({len(kept)}):\n"
        + ("\n".join(found) if found else "(none yet)")
        + "\n\nPropose ONE next English web search query, or stop."
    )
    try:
        data = _chat_json(llm, system, user)
    except Exception as e:
        return {"action": "search", "query": "", "reason": f"llm_plan_failed:{e}", "error": str(e)}
    if not isinstance(data, dict):
        return {"action": "stop", "query": "", "reason": "bad_plan_json"}
    action = (data.get("action") or "search").strip().lower()
    query = (data.get("query") or "").strip()
    reason = (data.get("reason") or "").strip()
    if action not in ("search", "stop"):
        action = "search" if query else "stop"
    return {"action": action, "query": query, "reason": reason}


def _llm_select_hits(
    spec: PairSpec,
    candidates: list[ResearchHit],
    llm: LLMConfig,
    *,
    already_kept: int,
    target: int,
) -> list[int]:
    """Brain: pick useful indices from this round's candidates (0-based)."""
    if not candidates:
        return []
    from fx_report.news.llm import _chat_json

    lines = []
    for i, h in enumerate(candidates):
        lines.append(
            f"[{i}] provider={h.provider} source={h.source}\n"
            f"title={h.title}\nurl={h.url}\nsnippet={h.snippet[:280]}\n"
        )
    need = max(0, target - already_kept)
    system = (
        "You select FX-relevant headlines for a research memo. "
        f"Return ONLY JSON: {{\"keep\":[int,...]}} with 0-based indices. "
        f"Keep at most {min(need + 2, 6)} items that help {spec.pair} outlook "
        "(banks, CB, CPI, geopolitics, commodities). Drop clickbait/unrelated."
    )
    user = "Candidates:\n" + "\n".join(lines)
    try:
        data = _chat_json(llm, system, user)
    except Exception:
        # Fallback: keep first few with http URLs
        return [i for i, h in enumerate(candidates) if (h.url or "").startswith("http")][: min(4, need or 4)]
    keep = data.get("keep") if isinstance(data, dict) else None
    if not isinstance(keep, list):
        return [i for i, h in enumerate(candidates) if (h.url or "").startswith("http")][: min(4, need or 4)]
    out: list[int] = []
    for x in keep:
        try:
            idx = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(candidates) and idx not in out:
            out.append(idx)
    return out[:6]


def _llm_extract_outlooks(
    spec: PairSpec,
    hits: list[ResearchHit],
    llm: LLMConfig,
    *,
    allowed_urls: set[str],
) -> list[Headline]:
    if not hits:
        return []
    from fx_report.news.llm import _chat_json

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
        "CRITICAL: url MUST be copied exactly from materials — never invent or alter URLs. "
        "If a fact has no URL in materials, omit that item. "
        "If nothing useful, return {\"items\":[]}."
    )
    user = (
        f"Currency pair: {spec.pair}\nDrivers: {', '.join(spec.default_drivers)}\n\n"
        f"Materials:\n{''.join(materials)}"
    )
    try:
        data = _chat_json(llm, system, user)
    except Exception:
        return [
            Headline(
                title=f"[AI research] {h.title}",
                summary=h.snippet[:400],
                source=h.source,
                url=h.url if h.url in allowed_urls else "",
                published=datetime.now(timezone.utc),
                provider="ai_research",
            )
            for h in hits[:8]
            if (h.url in allowed_urls) or h.provider == "whitelist"
        ]

    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        items = []

    out: list[Headline] = []
    for it in items[:12]:
        if not isinstance(it, dict):
            continue
        title = (it.get("title") or "").strip()
        if not title:
            continue
        url = (it.get("url") or "").strip()
        # Honesty: drop invented URLs
        if url and url not in allowed_urls:
            # try match by ignoring trailing slash / case
            url_norm = url.rstrip("/").lower()
            match = next((u for u in allowed_urls if u.rstrip("/").lower() == url_norm), "")
            url = match
        if not url:
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
                url=url,
                published=datetime.now(timezone.utc),
                provider="ai_research",
            )
        )
    return out


def _hits_to_headlines(hits: list[ResearchHit], *, max_n: int) -> list[Headline]:
    out: list[Headline] = []
    for h in hits:
        if not (h.url or "").startswith("http"):
            continue
        out.append(
            Headline(
                title=f"[AI research] {h.title}"[:220],
                summary=h.snippet[:400],
                source=h.source,
                url=h.url,
                published=datetime.now(timezone.utc),
                provider="ai_research",
            )
        )
        if len(out) >= max_n:
            break
    return out


def run_ai_research(
    spec: PairSpec,
    *,
    info_need_ids: list[str] | None = None,
    llm_cfg: LLMConfig | None = None,
    max_headlines: int | None = None,
    max_rounds: int | None = None,
    target_keep: int | None = None,
    as_of_date: date | datetime | str | None = None,
    allow_historical: bool = False,
) -> AIResearchResult:
    """
    AI 检索员主入口（迭代人工式）：
      白名单 → 多轮「拟 query → 搜索手 → 挑选」→ LLM 收成语句（仅真 URL）

    Historical as_of: disabled by default (no Tavily/Brave burn). Pass
    ``allow_historical=True`` only for explicit expensive overrides — search
    results are still live-web and may leak non-historical info.

    When Tavily/Brave Key present, default target_keep/rounds rise toward
    ~40 displayable refs (still no invented URLs).
    """
    cfg = load_config()
    budget = live_research_budget(cfg)
    if max_headlines is None:
        max_headlines = int(budget["max_headlines"])
    if max_rounds is None:
        max_rounds = int(budget["max_rounds"])
    if target_keep is None:
        target_keep = int(budget["target_keep"])
    search_limit = int(budget["search_limit"])
    need_ids = info_need_ids or list(spec.default_drivers)
    hands = search_hands_available(cfg)
    paid_search = has_paid_search_api(cfg)
    llm = llm_cfg or resolve_llm_config()

    if as_of_date is not None and not allow_historical:
        return AIResearchResult(
            headlines=[],
            hits=[],
            meta={
                "enabled": False,
                "historical_disabled": True,
                "mode": "historical_disabled",
                "limitation": (
                    "AI researcher 默认依赖当前白名单页/搜索结果，无法保证历史时点可追溯；"
                    "历史回放中已禁用，避免伪造当时可得证据。"
                ),
            },
        )

    meta: dict[str, Any] = {
        "enabled": True,
        "mode": "iterative",
        "brain": bool(llm),
        "llm": bool(llm),
        "hands": hands,
        "paid_search": paid_search,
        "target_keep": target_keep,
        "max_rounds": max_rounds,
        "max_headlines": max_headlines,
        "queries": [],
        "rounds": [],
        "whitelist_ok": 0,
        "search_hits": 0,
        "kept_hits": 0,
        "errors": [],
        "limitation": None,
    }
    if as_of_date is not None and allow_historical:
        meta["allow_historical"] = True
        meta["historical_warning"] = (
            "已显式允许历史时点启用 AI 检索：Tavily/Brave 仍为实时搜索，"
            "可能引入 as_of 之后的非历史信息；仅调试用，默认应关闭。"
        )
        meta["limitation"] = meta["historical_warning"]

    if llm and not paid_search:
        meta["limitation"] = (
            "仅有 LLM（如 DeepSeek）作「脑」，无 Tavily/Brave/NewsAPI 作增强「手」。"
            "本轮仍用白名单公开页 + Google News RSS（免费）；"
            "DeepSeek 不会虚构 URL。填 TAVILY_API_KEY 或 BRAVE_SEARCH_API_KEY 可搜得更全。"
        )
    elif not llm and not paid_search:
        meta["limitation"] = (
            "无 LLM 且无 Tavily/Brave/NewsAPI：仅白名单 + Google News RSS 固定词。"
            "建议填 DEEPSEEK_API_KEY（脑）+ TAVILY_API_KEY（手）。"
        )
    elif not llm and paid_search:
        meta["limitation"] = (
            "有搜索 Key 但无 LLM：按预设词一条条搜，不做智能选题/抽取。"
            "建议填 DEEPSEEK_API_KEY 或本机 Ollama。"
        )

    # 1) whitelist pages (no search key needed)
    kept: list[ResearchHit] = []
    wl = fetch_whitelist(spec, cfg)
    kept.extend(wl)
    meta["whitelist_ok"] = len(wl)

    all_hits: list[ResearchHit] = list(wl)
    tried_queries: list[str] = []
    seed = _seed_queries(spec, need_ids)
    seed_i = 0

    # 2) iterative rounds: brain proposes → hands search → brain selects
    for round_i in range(max(1, max_rounds)):
        if len(kept) >= target_keep:
            meta["rounds"].append({"round": round_i, "action": "stop", "reason": "target_reached"})
            break

        query = ""
        plan_reason = ""
        if llm:
            plan = _llm_plan_next_query(
                spec, need_ids, kept, tried_queries, round_i, max_rounds, llm
            )
            plan_reason = plan.get("reason") or ""
            if plan.get("error"):
                meta["errors"].append(f"plan:{plan['error']}")
            if plan.get("action") == "stop" and kept:
                meta["rounds"].append(
                    {"round": round_i, "action": "stop", "reason": plan_reason or "llm_stop"}
                )
                break
            query = (plan.get("query") or "").strip()

        if not query:
            # fallback: next seed query (still one-at-a-time)
            while seed_i < len(seed) and seed[seed_i] in tried_queries:
                seed_i += 1
            if seed_i >= len(seed):
                meta["rounds"].append(
                    {"round": round_i, "action": "stop", "reason": "no_more_seed_queries"}
                )
                break
            query = seed[seed_i]
            seed_i += 1
            plan_reason = plan_reason or "seed_fallback"

        if query in tried_queries:
            meta["rounds"].append(
                {"round": round_i, "action": "skip", "query": query, "reason": "duplicate_query"}
            )
            continue

        tried_queries.append(query)
        meta["queries"].append(query)

        round_hits, used = execute_search(query, cfg, limit=search_limit)
        round_hits = _dedupe_hits(round_hits)
        # drop already kept
        kept_keys = {(h.url or h.title).strip().lower() for h in kept}
        fresh = [h for h in round_hits if (h.url or h.title).strip().lower() not in kept_keys]
        # Prefer URL+snippet / stable providers when presenting to LLM or fallback keep
        fresh = _rank_hits_for_refs(fresh)
        all_hits.extend(fresh)

        selected: list[ResearchHit] = []
        if fresh and llm:
            idxs = _llm_select_hits(
                spec, fresh, llm, already_kept=len(kept), target=target_keep
            )
            selected = [fresh[i] for i in idxs if 0 <= i < len(fresh)]
        elif fresh:
            selected = [h for h in fresh if (h.url or "").startswith("http")][
                : min(6, max(4, target_keep - len(kept)))
            ]

        kept.extend(selected)
        meta["rounds"].append(
            {
                "round": round_i,
                "action": "search",
                "query": query,
                "reason": plan_reason,
                "providers": used,
                "fresh": len(fresh),
                "kept": len(selected),
            }
        )
        if not fresh:
            meta["errors"].append(f"no_search_hits:{query[:50]}")

    kept = _rank_hits_for_refs(_dedupe_hits(kept))
    all_hits = _dedupe_hits(all_hits)
    meta["search_hits"] = len(all_hits)
    meta["kept_hits"] = len(kept)

    allowed = _allowed_urls(kept) | _allowed_urls(all_hits)

    if llm and kept:
        headlines = _llm_extract_outlooks(spec, kept, llm, allowed_urls=allowed)
        if not headlines:
            # Extract returned empty (or all URLs rejected) — use raw kept with real URLs
            headlines = _hits_to_headlines(kept, max_n=max_headlines)
            meta["errors"].append("llm_extract_empty_used_raw_kept")
    else:
        headlines = _hits_to_headlines(kept or all_hits, max_n=max_headlines)
        if not llm:
            meta["errors"].append("no_llm_config_used_raw_hits")

    # Final honesty pass: strip any headline whose URL is not in allowed set
    honest: list[Headline] = []
    for h in headlines:
        if h.url and h.url not in allowed:
            h = Headline(
                title=h.title,
                summary=h.summary,
                source=h.source,
                url="",
                published=h.published,
                provider=h.provider,
            )
        # Prefer items that still have a real URL for References
        if h.url or h.provider == "ai_research":
            if h.url:
                honest.append(h)
            elif not allowed:
                # no URLs at all — still surface title for evidence text (no fake link)
                honest.append(h)
    # If honesty filter wiped everything, fall back to kept with URLs only
    if not honest:
        honest = _hits_to_headlines(kept, max_n=max_headlines)

    def _hl_key(h: Headline) -> tuple:
        from fx_report.news.urls import is_fragile_url, provider_stability_rank

        has_url = 0 if (h.url or "").startswith("http") else 1
        has_snip = 0 if len((h.summary or "").strip()) >= 40 else 1
        fragile = 1 if is_fragile_url(h.url) else 0
        return (has_url, has_snip, fragile, provider_stability_rank(h.provider))

    honest = sorted(honest, key=_hl_key)
    meta["headlines_out"] = len(honest[:max_headlines])
    return AIResearchResult(headlines=honest[:max_headlines], hits=kept, meta=meta)
