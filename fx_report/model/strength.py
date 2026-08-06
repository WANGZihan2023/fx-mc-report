"""
Transparent rubric for evidence strength / freshness / unpriced.

IMPORTANT
---------
Strength is NOT "AI vibes". It is a scored checklist. Defaults in templates
are starting points; the UI exposes every input so you can override.

Final evidence contribution:
    contrib = direction × strength × freshness × unpriced

where strength ∈ [0, 3] is produced by score_strength(...).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Rubric tables (the "judgment rules")
# ---------------------------------------------------------------------------

SOURCE_TIER_POINTS = {
    # points toward strength (0–1.2)
    "primary_official": 1.2,  # 央行/统计局/财政部一手稿、FOMC 声明
    "primary_market": 1.0,  # CME FedWatch、交易所、CFTC 原始持仓
    "tier1_wire": 0.85,  # Reuters / Bloomberg 一手报道
    "tier1_bank": 0.75,  # 主流投行/资管正式研报
    "tier2_media": 0.45,  # 二手财经媒体转述
    "blog_social": 0.15,  # 博客/社交/传闻
}

SURPRISE_POINTS = {
    # how far the print / event is from consensus or prior path
    "none": 0.0,  # in line
    "small": 0.25,  # ~0.5σ or mild wording shift
    "medium": 0.55,  # ~1σ / clear policy pivot signal
    "large": 0.90,  # ≥1.5σ / regime-changing (blockade, emergency hike)
    "extreme": 1.20,  # crisis / market halt level
}

SCOPE_POINTS = {
    "idiosyncratic": 0.15,  # single name / thin local story
    "pair_specific": 0.40,  # clearly moves this pair's drivers
    "g10_macro": 0.70,  # Fed/ECB/oil/risk that hits many FX
    "systemic": 1.00,  # war blockade, global funding stress
}

# Freshness half-life (trading days) by category — used if you pass age_days
FRESHNESS_HALFLIFE_DAYS = {
    "geopolitics": 5.0,
    "oil": 4.0,
    "cpi": 8.0,
    "fed": 7.0,
    "ecb": 7.0,
    "boe": 7.0,
    "boj": 7.0,
    "rba": 7.0,
    "boc": 7.0,
    "rbnz": 7.0,
    "snb": 7.0,
    "pboc": 7.0,
    "china_iron": 10.0,
    "china_growth": 10.0,
    "dairy": 10.0,
    "yields": 3.0,
    "growth": 8.0,
    "positioning": 12.0,
    "other": 7.0,
}


@dataclass
class StrengthInputs:
    """Observable checklist used to judge one piece of information."""

    source_tier: str = "tier1_wire"
    surprise: str = "medium"
    scope: str = "pair_specific"
    # Optional: age in calendar/trading days for freshness
    age_days: float | None = None
    category: str = "other"
    # Has the move already shown up in spot / IV? 0=fully priced, 1=not priced
    unpriced_hint: float | None = None
    # Manual override (if set, skips formula for strength)
    strength_override: float | None = None


@dataclass
class StrengthResult:
    strength: float
    freshness: float
    unpriced: float
    breakdown: dict[str, float]
    rules_version: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def freshness_from_age(age_days: float | None, category: str) -> float:
    """Exponential decay: freshness = 0.5 ** (age / half_life)."""
    if age_days is None:
        return 1.0
    hl = FRESHNESS_HALFLIFE_DAYS.get(category, FRESHNESS_HALFLIFE_DAYS["other"])
    age = max(0.0, float(age_days))
    return float(0.5 ** (age / hl))


def score_strength(inp: StrengthInputs) -> StrengthResult:
    """
    Strength formula (capped at 3.0):

        strength = clamp( source + surprise + scope , 0, 3 )

    Mapping to Torchcast labels:
        ≤1.0  → SLIGHT
        ≤2.0  → MODERATE
        >2.0  → STRONG

    Freshness: from age_days + category half-life (or 1.0 if age unknown).
    Unpriced: use hint if given, else default 0.55 (partially priced).
    """
    if inp.strength_override is not None:
        strength = max(0.0, min(3.0, float(inp.strength_override)))
        breakdown = {"override": strength, "source": 0.0, "surprise": 0.0, "scope": 0.0}
    else:
        src = SOURCE_TIER_POINTS.get(inp.source_tier, 0.45)
        sur = SURPRISE_POINTS.get(inp.surprise, 0.55)
        sco = SCOPE_POINTS.get(inp.scope, 0.40)
        strength = max(0.0, min(3.0, src + sur + sco))
        breakdown = {"source": src, "surprise": sur, "scope": sco, "sum": src + sur + sco}

    freshness = freshness_from_age(inp.age_days, inp.category)
    unpriced = (
        max(0.0, min(1.0, float(inp.unpriced_hint)))
        if inp.unpriced_hint is not None
        else 0.55
    )
    return StrengthResult(
        strength=strength,
        freshness=freshness,
        unpriced=unpriced,
        breakdown=breakdown,
    )


def label_strength(strength: float) -> str:
    if strength <= 1.0:
        return "SLIGHT"
    if strength <= 2.0:
        return "MODERATE"
    return "STRONG"


def rubric_markdown(*, lang: str = "zh") -> str:
    """Human-readable explanation for the sidebar / report."""
    src = "\n".join(f"| `{k}` | {v:.2f} |" for k, v in SOURCE_TIER_POINTS.items())
    sur = "\n".join(f"| `{k}` | {v:.2f} |" for k, v in SURPRISE_POINTS.items())
    sco = "\n".join(f"| `{k}` | {v:.2f} |" for k, v in SCOPE_POINTS.items())
    en = str(lang or "").lower().startswith("en")
    if en:
        return f"""
### How strength is scored (rules, not vibes)

**Contribution**  
`contrib = direction × strength × freshness × unpriced`

**strength (0–3)** = source tier + surprise + scope (capped at 3)

| source_tier | pts |
|---|---|
{src}

| surprise | pts |
|---|---|
{sur}

| scope | pts |
|---|---|
{sco}

**Labels**: ≤1 SLIGHT｜≤2 MODERATE｜>2 STRONG (Torchcast-aligned)

**freshness**: `0.5 ** (age_days / half_life)` — geopolitics ~5d half-life; CPI/central bank ~7–8d.

**unpriced**: 0 = fully priced in, 1 = barely priced; default 0.55. After a large spot jump, lower this.

> Older hard-coded 3/2/1 tags were manual Torchcast labels; the same checklist now applies to every pair and can be overridden in the sidebar.
"""
    return f"""
### 信息强弱如何判定（规则版，非主观拍脑袋）

**贡献分**  
`contrib = direction × strength × freshness × unpriced`

**strength（0–3）** = 来源分 + 意外分 + 影响范围分（上限 3）

| 来源档 source_tier | 分 |
|---|---|
{src}

| 意外程度 surprise | 分 |
|---|---|
{sur}

| 影响范围 scope | 分 |
|---|---|
{sco}

**标签映射**：≤1 SLIGHT｜≤2 MODERATE｜>2 STRONG（对齐 Torchcast 用语）

**freshness**：`0.5 ** (age_days / half_life)`，地缘半衰期约 5 日，CPI/央行约 7–8 日。

**unpriced**：0=已被价格吃完，1=几乎未定价；默认 0.55。若事件后即期已大跳，应下调。

> 上一版里写死的 3/2/1，本质是人工填的 Torchcast 标签；现在同一套规则对所有货币对生效，可在侧栏用清单自动打分或手动覆盖。
"""
