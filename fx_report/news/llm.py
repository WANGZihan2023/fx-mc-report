"""
LLM-based headline/full-text evidence classification.

Uses OpenAI-compatible Chat Completions API so it works with OpenAI,
DeepSeek, Moonshot, SiliconFlow, Azure OpenAI (compatible mode), etc.

Env / Streamlit secrets:
  OPENAI_API_KEY or LLM_API_KEY
  OPENAI_BASE_URL or LLM_BASE_URL  (default https://api.openai.com/v1)
  LLM_MODEL  (default gpt-4o-mini)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fx_report.news.fetch import Headline
from fx_report.news.classify import MIN_PAIR_RELEVANCE, pair_relevance
from fx_report.market.pairs import PairSpec, get_pair
from fx_report.model.strength import StrengthInputs, label_strength, score_strength
from fx_report.model.weights import EvidenceItem

VALID_SOURCE = {
    "primary_official",
    "primary_market",
    "tier1_wire",
    "tier1_bank",
    "tier2_media",
    "blog_social",
}
VALID_SURPRISE = {"none", "small", "medium", "large", "extreme"}
VALID_SCOPE = {"idiosyncratic", "pair_specific", "g10_macro", "systemic"}
VALID_CATEGORY = {
    "geopolitics",
    "oil",
    "cpi",
    "fed",
    "ecb",
    "boe",
    "boj",
    "rba",
    "rbnz",
    "boc",
    "snb",
    "pboc",
    "china_growth",
    "china_iron",
    "yields",
    "growth",
    "positioning",
    "other",
}


@dataclass
class LLMConfig:
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    timeout: int = 120
    max_article_chars: int = 3500


FREE_PROVIDERS = {
    "ollama": {
        "label": "Ollama 本机（免费）",
        "api_key": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "llama3.1:latest",
    },
    "groq": {
        "label": "Groq 云端免费额度",
        "api_key": "",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-8b-instant",
        "signup": "https://console.groq.com/keys",
    },
}


def ollama_available(timeout: float = 1.5) -> bool:
    try:
        req = Request("http://127.0.0.1:11434/api/tags", method="GET")
        with urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def resolve_llm_config(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    allow_ollama_auto: bool = True,
) -> LLMConfig | None:
    key = (
        api_key
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("GROQ_API_KEY")
        or ""
    ).strip()
    base = (
        base_url
        or os.environ.get("LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    ).rstrip("/")
    mdl = model or os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL") or ""

    # Auto local free path: Ollama with llama3.1
    if allow_ollama_auto and not key and not base:
        if ollama_available():
            p = FREE_PROVIDERS["ollama"]
            return LLMConfig(
                api_key=p["api_key"],
                base_url=p["base_url"],
                model=mdl or p["model"],
            )

    if not key and base and ("11434" in base or "localhost" in base or "127.0.0.1" in base):
        key = "ollama"

    if not key:
        return None

    if not base:
        if os.environ.get("GROQ_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            base = "https://api.groq.com/openai/v1"
            mdl = mdl or "llama-3.1-8b-instant"
        elif os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            base = "https://api.deepseek.com/v1"
            mdl = mdl or "deepseek-chat"
        else:
            base = "https://api.openai.com/v1"
            mdl = mdl or "gpt-4o-mini"
    if not mdl:
        if "11434" in base:
            mdl = "llama3.1:latest"
        elif "groq.com" in base:
            mdl = "llama-3.1-8b-instant"
        elif "deepseek.com" in base:
            mdl = "deepseek-chat"
        else:
            mdl = "gpt-4o-mini"
    return LLMConfig(api_key=key, base_url=base, model=mdl)


def fetch_article_text(url: str, max_chars: int = 3500) -> str:
    """Best-effort plain text extract from article URL (not a full browser)."""
    if not url or not url.startswith("http"):
        return ""
    try:
        req = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; FXReportBot/1.0)",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        with urlopen(req, timeout=12) as resp:
            raw = resp.read(600_000)
            ctype = (resp.headers.get("Content-Type") or "").lower()
        if "html" not in ctype and b"<html" not in raw[:500].lower():
            return ""
        html = raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""

    # Strip scripts/styles
    html = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", html)
    # Prefer <article> / <p>
    chunks: list[str] = []
    for m in re.finditer(r"(?is)<article[^>]*>(.*?)</article>", html):
        chunks.append(m.group(1))
    if not chunks:
        chunks = re.findall(r"(?is)<p[^>]*>(.*?)</p>", html)
    text = " ".join(chunks)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _chat_json(cfg: LLMConfig, system: str, user: str) -> dict[str, Any]:
    def _request(with_json_format: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": cfg.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        if with_json_format:
            payload["response_format"] = {"type": "json_object"}
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{cfg.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg.api_key}",
            },
            method="POST",
        )
        with urlopen(req, timeout=cfg.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    is_local = "11434" in cfg.base_url or "localhost" in cfg.base_url
    try:
        body = _request(with_json_format=not is_local)
    except HTTPError as e:
        err = e.read().decode("utf-8", errors="ignore")
        # Retry without response_format (Ollama / some providers)
        if e.code in {400, 422} or "response_format" in err:
            try:
                body = _request(with_json_format=False)
            except Exception as e2:
                raise RuntimeError(f"LLM HTTP retry failed: {e2}") from e2
        else:
            raise RuntimeError(f"LLM HTTP {e.code}: {err[:400]}") from e
    except URLError as e:
        raise RuntimeError(f"LLM network error: {e}") from e

    content = body["choices"][0]["message"]["content"]
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    # Extract first JSON object if model added prose
    if not content.startswith("{"):
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            content = m.group(0)
    return json.loads(content)


def _age_days(published: datetime | None) -> float:
    if published is None:
        return 2.0
    now = datetime.now(timezone.utc)
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    return max(0.0, (now - published).total_seconds() / 86400.0)


def _clamp_choice(value: str, allowed: set[str], default: str) -> str:
    v = (value or "").strip().lower()
    return v if v in allowed else default


SYSTEM_PROMPT = """你是外汇宏观分析助手。根据新闻标题与正文摘要，判断对指定货币对「窗口内最高价」的影响。

分析口径 pair=BASE/QUOTE：
- direction=+1：推高该报价路径最高值 = BASE 相对 QUOTE 走强
- direction=-1：压制峰值 = BASE 相对 QUOTE 走弱
- relevant=false：与该货币对基本无关则忽略

务必遵守（举例）：
- pair=USD/CNH 时：美元走强 / 加息预期升温 → direction=+1
- pair=USD/CNH 时：人民币走强 / PBOC 强化中间价 → direction=-1
- pair=AUD/USD 时：澳元走强 → direction=+1；美元走强 → direction=-1
- pair=EUR/USD 时：欧元走强 → direction=+1；美元走强 → direction=-1

必须只输出 JSON：
{
  "items": [
    {
      "id": "N-01",
      "relevant": true,
      "direction": 1,
      "category": "fed|pboc|geopolitics|cpi|oil|china_growth|yields|other|...",
      "source_tier": "primary_official|primary_market|tier1_wire|tier1_bank|tier2_media|blog_social",
      "surprise": "none|small|medium|large|extreme",
      "scope": "idiosyncratic|pair_specific|g10_macro|systemic",
      "unpriced_hint": 0.0到1.0,
      "rationale": "一句中文理由，并点明对 BASE/QUOTE 谁强"
    }
  ]
}
"""


def classify_headlines_llm(
    headlines: list[Headline],
    pair: PairSpec | str,
    cfg: LLMConfig,
    *,
    max_items: int = 12,
    fetch_fulltext: bool = True,
    unpriced_cap: float = 0.75,
) -> tuple[list[EvidenceItem], dict[str, Any]]:
    """
    Batch-classify headlines with an LLM. Optionally fetch article snippets.
    Returns (evidence_items, debug_meta).
    """
    spec = get_pair(pair) if isinstance(pair, str) else pair
    if not headlines:
        return [], {"error": None, "model": cfg.model, "n_input": 0}

    # Prefer pair-relevant headlines; drop low-relevance noise
    ranked = sorted(
        headlines,
        key=lambda h: (
            pair_relevance(f"{h.title} {h.summary}", spec.pair),
            h.published.timestamp() if h.published else 0.0,
        ),
        reverse=True,
    )
    filtered = [
        h
        for h in ranked
        if pair_relevance(f"{h.title} {h.summary}", spec.pair) >= MIN_PAIR_RELEVANCE
    ]
    selected = (filtered or ranked)[: max(max_items * 2, max_items)]
    payload_news = []
    for i, h in enumerate(selected, start=1):
        body = h.summary or ""
        if fetch_fulltext and h.url:
            extra = fetch_article_text(h.url, max_chars=cfg.max_article_chars)
            if extra:
                body = (body + "\n" + extra).strip()
        payload_news.append(
            {
                "id": f"N-{i:02d}",
                "title": h.title,
                "source": h.source,
                "url": h.url,
                "published": h.published.isoformat() if h.published else None,
                "text": body[: cfg.max_article_chars],
            }
        )

    user = (
        f"货币对 pair={spec.pair}（BASE={spec.base}, QUOTE={spec.quote}）。\n"
        f"请精读下列新闻（含正文摘录）并输出 JSON。最多选出 {max_items} 条 relevant=true。\n\n"
        + json.dumps(payload_news, ensure_ascii=False)
    )

    meta: dict[str, Any] = {
        "model": cfg.model,
        "base_url": cfg.base_url,
        "n_input": len(payload_news),
        "fetch_fulltext": fetch_fulltext,
        "error": None,
        "raw_count": 0,
    }

    try:
        raw = _chat_json(cfg, SYSTEM_PROMPT, user)
    except Exception as e:
        meta["error"] = str(e)
        return [], meta

    rows = raw.get("items") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        meta["error"] = "LLM JSON missing items[]"
        return [], meta
    meta["raw_count"] = len(rows)

    # Map id -> headline for titles
    by_id = {f"N-{i:02d}": h for i, h in enumerate(selected, start=1)}

    items: list[EvidenceItem] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("relevant") is False:
            continue
        eid = str(row.get("id") or f"N-{len(items)+1:02d}")
        try:
            direction = int(row.get("direction", 0))
        except Exception:
            continue
        if direction not in (-1, 1):
            continue

        category = _clamp_choice(str(row.get("category", "other")), VALID_CATEGORY, "other")
        source_tier = _clamp_choice(str(row.get("source_tier", "tier2_media")), VALID_SOURCE, "tier2_media")
        surprise = _clamp_choice(str(row.get("surprise", "medium")), VALID_SURPRISE, "medium")
        scope = _clamp_choice(str(row.get("scope", "pair_specific")), VALID_SCOPE, "pair_specific")
        try:
            unpriced_hint = float(row.get("unpriced_hint", 0.55))
        except Exception:
            unpriced_hint = 0.55
        unpriced_hint = max(0.0, min(unpriced_cap, unpriced_hint))

        h = by_id.get(eid)
        age = _age_days(h.published if h else None)
        scored = score_strength(
            StrengthInputs(
                source_tier=source_tier,
                surprise=surprise,
                scope=scope,
                age_days=age,
                category=category,
                unpriced_hint=unpriced_hint,
            )
        )
        rationale = str(row.get("rationale") or "").strip()
        title = (h.title if h else str(row.get("title") or eid))[:180]
        note = f"LLM精读｜{cfg.model}"
        if rationale:
            note += f"｜{rationale[:160]}"
        if h and h.url:
            note += f"｜{(h.url or '')[:100]}"

        items.append(
            EvidenceItem(
                id=eid,
                title=title,
                direction=direction,
                strength=scored.strength,
                freshness=scored.freshness,
                unpriced=scored.unpriced,
                category=category,
                note=note,
                strength_label=label_strength(scored.strength),
                strength_breakdown={**scored.breakdown, "llm": 1.0},
                source_tier=source_tier,
                surprise=surprise,
                scope=scope,
                url=(h.url if h else "") or "",
            )
        )
        if len(items) >= max_items:
            break

    meta["n_evidence"] = len(items)
    return items, meta
