"""
Model weights / scenarios / evidence templates — pair-agnostic.

Direction convention:
  +1 = pushes the analysis-quote PATH MAXIMUM higher
  -1 = caps / lowers the peak
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from fx_report.market.pairs import PairSpec, edges_from_spot, get_pair
from fx_report.model.strength import StrengthInputs, label_strength, score_strength


@dataclass
class ScenarioSpec:
    name: str
    weight: float
    mu_annual: float  # drift of analysis quote
    sigma_mult: float
    expected_jumps: float
    jump_mean: float
    jump_std: float
    narrative: str


@dataclass
class EvidenceItem:
    id: str
    title: str
    direction: int
    strength: float
    freshness: float
    unpriced: float
    category: str
    note: str = ""
    # How strength was obtained
    strength_label: str = ""
    strength_breakdown: dict[str, float] = field(default_factory=dict)
    source_tier: str = ""
    surprise: str = ""
    scope: str = ""


@dataclass
class ModelWeights:
    n_sims: int = 100_000
    seed: int = 42
    trading_days: int = 66
    vol_lookback_days: int = 60
    bucket_edges: tuple[float, float, float, float] = (1.40, 1.43, 1.46, 1.49)
    # If True, edges are rebuilt from spot × bucket_pct_cuts each run
    use_relative_buckets: bool = True
    bucket_pct_cuts: tuple[float, float, float, float] = (0.0, 2.0, 4.0, 6.0)

    score_to_mu_a: float = 0.012
    score_to_sigma_b: float = 0.035
    scenario_temperature: float = 1.0
    max_scenario_shift: float = 0.18
    evidence_logit_scale: float = 0.08

    scenarios: list[ScenarioSpec] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_scenarios(pair: str) -> list[ScenarioSpec]:
    """Generic three-regime prior; narratives mention the pair."""
    return [
        ScenarioSpec(
            name="escalation",
            weight=0.35,
            mu_annual=0.06,
            sigma_mult=1.30,
            expected_jumps=1.0,
            jump_mean=0.010,
            jump_std=0.007,
            narrative=f"风险升高 / 避险或单边冲击 → {pair} 上尾加厚",
        ),
        ScenarioSpec(
            name="baseline",
            weight=0.40,
            mu_annual=0.01,
            sigma_mult=1.00,
            expected_jumps=0.30,
            jump_mean=0.003,
            jump_std=0.004,
            narrative=f"中性胶着 → {pair} 峰值多落在中档",
        ),
        ScenarioSpec(
            name="deescalation",
            weight=0.25,
            mu_annual=-0.03,
            sigma_mult=0.90,
            expected_jumps=0.20,
            jump_mean=-0.005,
            jump_std=0.004,
            narrative=f"缓和 / 逆风消退 → {pair} 峰值受压",
        ),
    ]


def _item_from_checklist(
    *,
    eid: str,
    title: str,
    direction: int,
    category: str,
    source_tier: str,
    surprise: str,
    scope: str,
    age_days: float | None,
    unpriced_hint: float | None,
    note: str = "",
) -> EvidenceItem:
    scored = score_strength(
        StrengthInputs(
            source_tier=source_tier,
            surprise=surprise,
            scope=scope,
            age_days=age_days,
            category=category,
            unpriced_hint=unpriced_hint,
        )
    )
    return EvidenceItem(
        id=eid,
        title=title,
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
    )


def _dir_usd_base(spec: PairSpec, usd_stronger: bool) -> int:
    """If USD strengthens: +1 when USD is base, -1 when USD is quote."""
    if spec.base == "USD":
        return +1 if usd_stronger else -1
    if spec.quote == "USD":
        return -1 if usd_stronger else +1
    return +1 if usd_stronger else -1


def _dir_aud_stronger(spec: PairSpec, aud_stronger: bool) -> int:
    if spec.base == "AUD":
        return +1 if aud_stronger else -1
    if spec.quote == "AUD":
        return -1 if aud_stronger else +1
    return 0


def default_evidence_for_pair(spec: PairSpec) -> list[EvidenceItem]:
    """
    Seed a small cross-pair evidence pack using the strength rubric.
    Pair-specific titles; same scoring rules for everyone.
    """
    pair = spec.pair
    drivers = set(spec.default_drivers)
    items: list[EvidenceItem] = []

    if "geopolitics" in drivers:
        # Default: risk-off strengthens USD vs risk FX
        items.append(
            _item_from_checklist(
                eid="U-GEO",
                title=f"地缘风险升温（对 {pair}）",
                direction=_dir_usd_base(spec, usd_stronger=True),
                category="geopolitics",
                source_tier="tier1_wire",
                surprise="large",
                scope="systemic",
                age_days=1.0,
                unpriced_hint=0.65,
                note="默认按避险美元；若该货币对是避险方请改方向",
            )
        )
    if "oil" in drivers:
        # CAD often benefits from oil↑ → USD/CAD↓; AUD refined-import stress → AUD↓
        if spec.pair == "USD/CAD":
            oil_dir = -1
            oil_note = "油涨常支撑 CAD（压低 USD/CAD）"
        elif "AUD" in (spec.base, spec.quote) or "NZD" in (spec.base, spec.quote):
            oil_dir = _dir_aud_stronger(spec, aud_stronger=False) or _dir_usd_base(spec, True)
            oil_note = "澳/纽对油供给冲击常偏弱（能源进口/风险）"
        else:
            oil_dir = _dir_usd_base(spec, usd_stronger=True)
            oil_note = "默认油冲击偏美元；请按货币对改方向"
        items.append(
            _item_from_checklist(
                eid="U-OIL",
                title="油价冲击",
                direction=oil_dir,
                category="oil",
                source_tier="primary_market",
                surprise="medium",
                scope="g10_macro",
                age_days=1.0,
                unpriced_hint=0.55,
                note=oil_note,
            )
        )
    if "fed" in drivers:
        items.append(
            _item_from_checklist(
                eid="U-FED",
                title="美联储偏鹰残存定价",
                direction=_dir_usd_base(spec, usd_stronger=True),
                category="fed",
                source_tier="primary_market",
                surprise="medium",
                scope="g10_macro",
                age_days=2.0,
                unpriced_hint=0.50,
            )
        )
    if "cpi" in drivers:
        items.append(
            _item_from_checklist(
                eid="D-CPI",
                title="美国 CPI 意外偏弱",
                direction=_dir_usd_base(spec, usd_stronger=False),
                category="cpi",
                source_tier="primary_official",
                surprise="large",
                scope="g10_macro",
                age_days=1.0,
                unpriced_hint=0.70,
            )
        )
    if "china_iron" in drivers or "china_growth" in drivers:
        # Weaker China demand → weaker AUD/NZD/CNH
        if "AUD" in (spec.base, spec.quote):
            cn_dir = _dir_aud_stronger(spec, aud_stronger=False)
        elif spec.base == "NZD":
            cn_dir = -1
        elif spec.quote == "NZD":
            cn_dir = +1
        elif "CNH" in (spec.base, spec.quote):
            cn_dir = +1 if spec.base == "USD" else -1
        else:
            cn_dir = _dir_usd_base(spec, usd_stronger=True)
        items.append(
            _item_from_checklist(
                eid="U-CN",
                title="中国增长/商品需求偏弱",
                direction=cn_dir,
                category="china_growth",
                source_tier="tier1_wire",
                surprise="medium",
                scope="pair_specific",
                age_days=3.0,
                unpriced_hint=0.50,
            )
        )
    if "rba" in drivers and "AUD" in (spec.base, spec.quote):
        items.append(
            _item_from_checklist(
                eid="D-RBA",
                title="RBA 利率相对支撑",
                direction=_dir_aud_stronger(spec, aud_stronger=True),
                category="rba",
                source_tier="primary_official",
                surprise="small",
                scope="pair_specific",
                age_days=5.0,
                unpriced_hint=0.40,
            )
        )
    if "ecb" in drivers and "EUR" in (spec.base, spec.quote):
        items.append(
            _item_from_checklist(
                eid="U-ECB",
                title="ECB 政策路径相对美联储",
                direction=+1 if spec.base == "EUR" else -1,
                category="ecb",
                source_tier="tier1_bank",
                surprise="small",
                scope="pair_specific",
                age_days=4.0,
                unpriced_hint=0.45,
            )
        )
    if "boj" in drivers and "JPY" in (spec.base, spec.quote):
        items.append(
            _item_from_checklist(
                eid="U-BOJ",
                title="日银政策 / 日美利差",
                direction=+1 if spec.pair == "USD/JPY" else -1,
                category="boj",
                source_tier="tier1_wire",
                surprise="medium",
                scope="pair_specific",
                age_days=3.0,
                unpriced_hint=0.50,
            )
        )
    if "pboc" in drivers and ("CNH" in (spec.base, spec.quote) or "CNY" in (spec.base, spec.quote)):
        items.append(
            _item_from_checklist(
                eid="D-PBOC",
                title="央行中间价/稳汇率信号",
                direction=-1 if spec.base == "USD" else +1,
                category="pboc",
                source_tier="primary_official",
                surprise="small",
                scope="pair_specific",
                age_days=2.0,
                unpriced_hint=0.40,
            )
        )
    if "boc" in drivers and "CAD" in (spec.base, spec.quote):
        items.append(
            _item_from_checklist(
                eid="U-BOC",
                title="加央行政策相对美联储",
                direction=-1 if spec.pair.startswith("USD/") else +1,
                category="boc",
                source_tier="primary_official",
                surprise="small",
                scope="pair_specific",
                age_days=4.0,
                unpriced_hint=0.45,
            )
        )
    if "rbnz" in drivers and "NZD" in (spec.base, spec.quote):
        items.append(
            _item_from_checklist(
                eid="U-RBNZ",
                title="纽储行 OCR 路径",
                direction=+1 if spec.base == "NZD" else -1,
                category="rbnz",
                source_tier="primary_official",
                surprise="small",
                scope="pair_specific",
                age_days=4.0,
                unpriced_hint=0.45,
            )
        )
    if "dairy" in drivers and "NZD" in (spec.base, spec.quote):
        items.append(
            _item_from_checklist(
                eid="U-DAIRY",
                title="乳制品价格/贸易条件",
                direction=+1 if spec.base == "NZD" else -1,
                category="dairy",
                source_tier="primary_market",
                surprise="medium",
                scope="pair_specific",
                age_days=5.0,
                unpriced_hint=0.50,
            )
        )
    if "snb" in drivers and "CHF" in (spec.base, spec.quote):
        items.append(
            _item_from_checklist(
                eid="U-SNB",
                title="瑞央行政策与避险需求",
                direction=-1 if spec.base == "USD" else +1,
                category="snb",
                source_tier="primary_official",
                surprise="small",
                scope="pair_specific",
                age_days=5.0,
                unpriced_hint=0.40,
            )
        )

    items.append(
        _item_from_checklist(
            eid="K-POS",
            title="投机仓位/情绪确认项",
            direction=+1,
            category="positioning",
            source_tier="primary_market",
            surprise="small",
            scope="pair_specific",
            age_days=2.0,
            unpriced_hint=0.35,
            note="模板占位；有 CFTC/仓位数据时请覆盖",
        )
    )
    return [it for it in items if it.direction != 0]

def default_weights(pair: str | PairSpec = "USD/AUD") -> ModelWeights:
    spec = get_pair(pair) if isinstance(pair, str) else pair
    return ModelWeights(
        use_relative_buckets=True,
        bucket_pct_cuts=spec.bucket_pct_cuts,
        scenarios=default_scenarios(spec.pair),
        evidence=default_evidence_for_pair(spec),
    )


def resolve_bucket_edges(weights: ModelWeights, spot: float) -> tuple[float, float, float, float]:
    if weights.use_relative_buckets:
        return edges_from_spot(spot, weights.bucket_pct_cuts)
    return weights.bucket_edges


def evidence_score(items: list[EvidenceItem]) -> float:
    return sum(e.direction * e.strength * e.freshness * e.unpriced for e in items)


def apply_evidence_to_scenarios(
    base: list[ScenarioSpec],
    score: float,
    *,
    logit_scale: float,
    temperature: float,
    max_shift: float,
) -> list[ScenarioSpec]:
    import math

    logits = []
    for s in base:
        if s.name == "escalation":
            bump = logit_scale * score
        elif s.name == "deescalation":
            bump = -logit_scale * score
        else:
            bump = 0.05 * logit_scale * score
        p = min(max(s.weight, 1e-6), 1 - 1e-6)
        logits.append(math.log(p / (1 - p)) + bump)

    raw = []
    for s, logit in zip(base, logits):
        mass = 1.0 / (1.0 + math.exp(-logit / max(temperature, 1e-6)))
        shifted = min(max(mass, s.weight - max_shift), s.weight + max_shift)
        raw.append(max(shifted, 1e-6))

    total = sum(raw)
    out: list[ScenarioSpec] = []
    for s, r in zip(base, raw):
        out.append(
            ScenarioSpec(
                name=s.name,
                weight=r / total,
                mu_annual=s.mu_annual,
                sigma_mult=s.sigma_mult,
                expected_jumps=s.expected_jumps,
                jump_mean=s.jump_mean,
                jump_std=s.jump_std,
                narrative=s.narrative,
            )
        )
    return out
