"""
Classify headlines into EvidenceItem using transparent keyword rules.

Direction is always relative to the analysis pair quote:
  +1 = supports a higher path maximum for the pair
  -1 = caps / lowers the peak
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from fx_report.news.fetch import Headline
from fx_report.market.pairs import PairSpec, get_pair
from fx_report.model.strength import StrengthInputs, label_strength, score_strength
from fx_report.model.weights import EvidenceItem


# Publisher / domain → source_tier
SOURCE_MAP: list[tuple[str, str]] = [
    (r"federalreserve|ecb\.europa|boj\.or\.jp|rba\.gov|bankofengland|snb\.ch|pboc|safe\.gov|bls\.gov|abs\.gov|fred\.stlouisfed", "primary_official"),
    (r"cmegroup|cftc\.gov|reuters|bloomberg|wsj\.com|ft\.com|wall street journal|financial times", "tier1_wire"),
    (r"reuters|bloomberg", "tier1_wire"),
    (r"goldman|jpmorgan|morgan stanley|ubs|citi|barclays|hsbc|mufg|nomura|deutsche bank|stonex|ing ", "tier1_bank"),
    (r"inbox:.*\.(pdf|PDF)", "tier1_bank"),  # 投放的付费研报 PDF 按投行档处理
    (r"cnbc|bbc|guardian|afr\.com|scmp|nikkei|investing\.com|marketwatch|yahoo|newsapi|finnhub|news\.google", "tier2_media"),
]

# Surprise keywords
SURPRISE_EXTREME = re.compile(
    r"\b(crash|collapse|emergency|blockade|invasion|default|halt(ed)?|circuit.?breaker)\b",
    re.I,
)
SURPRISE_LARGE = re.compile(
    r"\b(surge[sd]?|plunge[sd]?|soar[sd]?|slump[sd]?|shock|unexpected(ly)?|miss(es|ed)?|"
    r"beat[s]? estimate|hottest|weakest|record high|record low|hawkish surprise|dovish surprise)\b",
    re.I,
)
SURPRISE_MEDIUM = re.compile(
    r"\b(rise[sd]?|fall[s]?|fell|gain[s]?|drop[s]|dropped|hike[sd]?|cut[s]|cutting|"
    r"tighten|ease[sd]?|escalat|de-?escalat|ceasefire|stimulus|intervention)\b",
    re.I,
)

# Category detectors
CATEGORY_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("geopolitics", re.compile(r"\b(war|iran|israel|hormuz|blockade|sanction|geopolit|conflict|missile|attack)\b", re.I)),
    ("oil", re.compile(r"\b(oil|brent|wti|crude|opec)\b", re.I)),
    ("cpi", re.compile(r"\b(cpi|inflation|pce|price index|consumer prices)\b", re.I)),
    ("fed", re.compile(r"\b(fed|fomc|powell|federal reserve|rate hike|rate cut|dot plot)\b", re.I)),
    ("ecb", re.compile(r"\b(ecb|lagarde|eurozone rates)\b", re.I)),
    ("boe", re.compile(r"\b(boe|bank of england|bailey)\b", re.I)),
    ("boj", re.compile(r"\b(boj|bank of japan|ueda|ycc)\b", re.I)),
    ("rba", re.compile(r"\b(rba|reserve bank of australia|bullock)\b", re.I)),
    ("rbnz", re.compile(r"\b(rbnz|ocr)\b", re.I)),
    ("boc", re.compile(r"\b(bank of canada|boc)\b", re.I)),
    ("snb", re.compile(r"\b(snb|swiss national)\b", re.I)),
    ("pboc", re.compile(r"\b(pboc|people.?s bank|middle rate|fixing|yuan fix|cny fix)\b", re.I)),
    ("china_growth", re.compile(r"\b(china gdp|china growth|china stimulus|property china|evergrande|yuan|renminbi|cnh|cny)\b", re.I)),
    ("china_iron", re.compile(r"\b(iron ore|steel|simandou|dalian)\b", re.I)),
    ("yields", re.compile(r"\b(yield|treasury|bond yield|rate differential)\b", re.I)),
    ("growth", re.compile(r"\b(gdp|payroll|jobs|unemployment|pmi|retail sales)\b", re.I)),
    ("positioning", re.compile(r"\b(positioning|speculator|cftc|crowded|sentiment)\b", re.I)),
]


def _source_tier(source: str, url: str) -> str:
    blob = f"{source} {url}".lower()
    for pat, tier in SOURCE_MAP:
        if re.search(pat, blob):
            return tier
    host = urlparse(url).netloc.lower()
    if host.endswith((".gov", ".gov.au", ".gov.cn")):
        return "primary_official"
    return "tier2_media"


def _surprise(text: str) -> str:
    if SURPRISE_EXTREME.search(text):
        return "extreme"
    if SURPRISE_LARGE.search(text):
        return "large"
    if SURPRISE_MEDIUM.search(text):
        return "medium"
    return "small"


def _category(text: str) -> str:
    # Title-like dollar macro should not be drowned by RSS related-link junk
    if re.search(r"\b(dollar index|dxy|u\.?s\.? dollar|greenback)\b", text, re.I) and not re.search(
        r"\b(war|iran|israel|hormuz|blockade|missile|attack|houthi)\b", text, re.I
    ):
        if re.search(r"\b(fed|fomc|powell|hike|cut)\b", text, re.I):
            return "fed"
        if re.search(r"\b(yield|treasury)\b", text, re.I):
            return "yields"
        return "yields"
    for cat, pat in CATEGORY_RULES:
        if pat.search(text):
            return cat
    return "other"


def _scope(category: str, text: str) -> str:
    if category in {"geopolitics"} or re.search(r"\b(global|world|systemic)\b", text, re.I):
        return "systemic" if category == "geopolitics" else "g10_macro"
    if category in {"fed", "cpi", "oil", "yields"}:
        return "g10_macro"
    if category == "other":
        return "idiosyncratic"
    return "pair_specific"


def _age_days(published: datetime | None, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    if published is None:
        return 2.0
    return max(0.0, (now - published).total_seconds() / 86400.0)


def _direction_for_pair(spec: PairSpec, category: str, text: str) -> int | None:
    """
    Return +1/-1 for the pair peak, or None if headline is not actionable.
    Heuristics are coarse; report shows them as auto-filled.
    """
    t = text.lower()
    base, quote = spec.base, spec.quote

    # USD strength language
    usd_up = bool(
        re.search(r"\b(dollar rises|dollar gains|dollar climbs|dollar surge|greenback (up|rises|gains)|dxy (up|rises)|usd strength)\b", t)
        or re.search(r"\b(hawkish fed|fed hike|rate hike odds rise)\b", t)
    )
    usd_down = bool(
        re.search(r"\b(dollar falls|dollar drops|dollar slides|dollar weak|greenback (down|falls)|dxy (down|falls))\b", t)
        or re.search(r"\b(dovish fed|fed cut|rate cut bets)\b", t)
        or (category == "cpi" and re.search(r"\b(cool|soft|miss|below|slows?)\b", t))
    )

    # Risk-off / geopolitics → usually USD up vs risk FX
    if category == "geopolitics":
        if re.search(r"\b(ceasefire|peace|de-?escalat|truce)\b", t):
            usd_up, usd_down = False, True
        else:
            usd_up = True

    if category == "oil" and re.search(r"\b(surge|jump|soar|spike|rally)\b", t):
        # CAD often benefits; AUD often hurt on supply shock / risk
        if quote == "CAD" or base == "CAD":
            # oil up → CAD stronger → USD/CAD down
            return -1 if base == "USD" else +1
        if "AUD" in (base, quote) or "NZD" in (base, quote):
            return +1 if base == "USD" else -1

    # China / yuan
    if category in {"china_growth", "pboc", "china_iron"} or re.search(r"\b(yuan|renminbi|cnh|cny)\b", t):
        cny_strong = bool(re.search(r"\b(yuan rises|yuan gains|cnh rises|stronger yuan|pboc fix stronger|appreciation)\b", t))
        cny_weak = bool(re.search(r"\b(yuan falls|yuan drops|cnh falls|weaker yuan|depreciation|fixing|sells yuan)\b", t))
        if category == "china_growth" and re.search(r"\b(slow|miss|weak|stimulus)\b", t):
            # weak China → pressure on CNH/AUD; USD/CNH up; AUD/USD down
            if quote in {"CNH", "CNY"} and base == "USD":
                return +1
            if base in {"AUD", "NZD"}:
                return -1
            if quote == "USD" and base in {"AUD", "NZD"}:
                return -1
        if cny_strong:
            if base == "USD" and quote in {"CNH", "CNY"}:
                return -1
            if quote == "USD" and base in {"CNH", "CNY"}:
                return +1
        if cny_weak:
            if base == "USD" and quote in {"CNH", "CNY"}:
                return +1
            if quote == "USD" and base in {"CNH", "CNY"}:
                return -1

    # Explicit pair mentions
    pair_flat = f"{base}/{quote}".lower()
    pair_noslash = f"{base}{quote}".lower()
    if pair_flat in t or pair_noslash in t or f"{base}{quote}".lower() in t.replace("/", ""):
        if re.search(rf"\b{re.escape(base.lower())}\b.*\b(rise|gain|climb|surge|rally|higher)\b", t):
            return +1
        if re.search(rf"\b{re.escape(base.lower())}\b.*\b(fall|drop|slide|slump|lower|weak)\b", t):
            return -1

    # Map USD up/down onto pair
    if usd_up or usd_down:
        stronger_usd = usd_up and not usd_down
        if base == "USD":
            return +1 if stronger_usd else -1
        if quote == "USD":
            return -1 if stronger_usd else +1

    # Local central bank hawkish → local currency stronger
    hawk = bool(re.search(r"\b(hawkish|hike|tightening)\b", t))
    dove = bool(re.search(r"\b(dovish|cut|easing)\b", t))
    bank_map = {
        "rba": "AUD",
        "rbnz": "NZD",
        "ecb": "EUR",
        "boe": "GBP",
        "boj": "JPY",
        "boc": "CAD",
        "snb": "CHF",
        "pboc": "CNH",
        "fed": "USD",
    }
    ccy = bank_map.get(category)
    if ccy and (hawk or dove):
        stronger = hawk and not dove
        if base == ccy:
            return +1 if stronger else -1
        if quote == ccy:
            return -1 if stronger else +1
        if ccy == "CNH" and quote == "CNY":
            return -1 if stronger else +1
        if ccy == "USD":
            if base == "USD":
                return +1 if stronger else -1
            if quote == "USD":
                return -1 if stronger else +1

    return None


def classify_headline(
    headline: Headline,
    pair: PairSpec | str,
    *,
    eid: str,
    unpriced_cap: float = 0.75,
    reference_now: datetime | None = None,
) -> EvidenceItem | None:
    spec = get_pair(pair) if isinstance(pair, str) else pair
    text_title = headline.title
    text_full = f"{headline.title}. {headline.summary}"
    # Category from title first to avoid Google RSS related-story pollution
    category = _category(text_title)
    if category == "other":
        category = _category(text_full)
    direction = _direction_for_pair(spec, category, text_full)
    if direction is None:
        direction = _direction_for_pair(spec, category, text_title)
    if direction is None:
        return None

    source_tier = _source_tier(headline.source, headline.url)
    surprise = _surprise(text_title) if _surprise(text_title) != "small" else _surprise(text_full)
    scope = _scope(category, text_title)
    age = _age_days(headline.published, now=reference_now)

    scored = score_strength(
        StrengthInputs(
            source_tier=source_tier,
            surprise=surprise,
            scope=scope,
            age_days=age,
            category=category,
            unpriced_hint=min(unpriced_cap, 0.75 if age < 3 else 0.45),
        )
    )
    note = f"自动抓取｜{headline.source}"
    if headline.url:
        note += f"｜{headline.url[:120]}"
    return EvidenceItem(
        id=eid,
        title=headline.title[:180],
        direction=direction,
        strength=scored.strength,
        freshness=scored.freshness,
        unpriced=scored.unpriced,
        category=category,
        note=note,
        strength_label=label_strength(scored.strength),
        strength_breakdown=scored.breakdown,
        source_tier=source_tier,
        surprise=surprise,
        scope=scope,
        url=headline.url or "",
    )


def pair_relevance(text: str, pair: str) -> float:
    """Higher = more relevant to this pair; used to rank / filter before classification."""
    t = text.lower()
    score = 0.0
    base, quote = pair.split("/")
    for token in (base.lower(), quote.lower(), pair.lower(), pair.replace("/", "").lower()):
        if token and token in t:
            score += 2.0
    extras = {
        "USD/CNH": ["yuan", "renminbi", "pboc", "offshore", "cnh", "china"],
        "USD/CNY": ["yuan", "renminbi", "pboc", "onshore", "cny", "china"],
        "USD/AUD": ["aussie", "australia", "rba", "iron ore"],
        "AUD/USD": ["aussie", "australia", "rba", "iron ore"],
        "EUR/USD": ["euro", "ecb", "lagarde"],
        "GBP/USD": ["sterling", "pound", "boe"],
        "USD/JPY": ["yen", "boj", "tokyo"],
        "USD/CAD": ["loonie", "canada", "boc"],
        "NZD/USD": ["kiwi", "new zealand", "rbnz"],
        "USD/CHF": ["franc", "snb", "swiss"],
    }
    for kw in extras.get(pair, []):
        if kw in t:
            score += 1.5
    # Broad FX / macro signals — help free RSS survive MIN_PAIR_RELEVANCE
    if "dollar" in t or "usd" in t or "fx" in t or "forex" in t or "currency" in t:
        score += 0.5
    if re.search(r"\b(fed|fomc|powell|inflation|cpi|yield|treasury|rate (hike|cut))\b", t):
        score += 0.4
    if re.search(r"\b(central bank|monetary policy|interest rate)\b", t):
        score += 0.3
    return score


# Backward-compatible alias
_pair_relevance = pair_relevance

# Headlines below this relevance are excluded from main evidence (no silent junk).
# Slightly lower than before so free RSS / Google News can contribute when no API keys;
# still filters obvious unrelated noise (score 0 stays out).
MIN_PAIR_RELEVANCE = 0.35


def match_drivers_from_text(
    text: str,
    *,
    allowed: list[str] | None = None,
) -> list[str]:
    """
    Map text → driver categories via CATEGORY_RULES.
    If nothing matches, return ['unclassified'] (never invent first-N drivers).
    """
    blob = text or ""
    hits: list[str] = []
    for cat, pat in CATEGORY_RULES:
        if pat.search(blob):
            if allowed is None or cat in allowed:
                hits.append(cat)
    if not hits:
        # title-only dollar macro → yields/fed already handled in _category; mirror lightly
        cat = _category(blob)
        if cat != "other" and (allowed is None or cat in allowed):
            hits.append(cat)
    return hits if hits else ["unclassified"]


def headlines_to_evidence(
    headlines: list[Headline],
    pair: PairSpec | str,
    *,
    max_items: int = 12,
    unpriced_cap: float = 0.75,
    min_relevance: float = MIN_PAIR_RELEVANCE,
    reference_now: datetime | None = None,
) -> tuple[list[EvidenceItem], dict[str, int]]:
    """
    Convert newest actionable headlines into evidence cards.
    Returns (items, counts) with fetched / kept / classified / evidence_n.
    """
    spec = get_pair(pair) if isinstance(pair, str) else pair
    fetched = len(headlines)
    ranked = sorted(
        headlines,
        key=lambda h: (
            pair_relevance(f"{h.title} {h.summary}", spec.pair),
            h.published.timestamp() if h.published else 0.0,
        ),
        reverse=True,
    )
    kept_pool: list[Headline] = []
    for h in ranked:
        rel = pair_relevance(f"{h.title} {h.summary}", spec.pair)
        if rel < min_relevance:
            continue
        kept_pool.append(h)

    items: list[EvidenceItem] = []
    classified = 0
    for i, h in enumerate(kept_pool, start=1):
        ev = classify_headline(
            h,
            pair,
            eid=f"N-{i:02d}",
            unpriced_cap=unpriced_cap,
            reference_now=reference_now,
        )
        if ev is None:
            continue
        classified += 1
        ev.url = h.url or ""
        items.append(ev)
        if len(items) >= max_items:
            break
    counts = {
        "fetched": fetched,
        "kept": len(kept_pool),
        "classified": classified,
        "evidence_n": len(items),
    }
    return items, counts
