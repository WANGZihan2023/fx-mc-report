"""
货币对 → 影响因子（驱动）→ 信息需求

设计原则：
  不同货币对的影响因子不同，因此必须先列出「本对需要什么信息」，
  再抓取、存语句、赋权、数学分析。任意 BASE/QUOTE 都可通过币种推断驱动。
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 单一驱动的信息需求定义（全市场共用词表）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DriverSpec:
    id: str
    label: str
    need: str
    why: str
    sources: str


# English overlays for report_lang=en (need / why / sources / label)
DRIVER_CATALOG_EN: dict[str, dict[str, str]] = {
    "spot_vol": {
        "label": "Spot & volatility",
        "need": "Authoritative spot and historical volatility",
        "why": "Monte Carlo paths and bucket edges",
        "sources": "ECB/Frankfurter → FRED → Twelve/Alpha",
    },
    "geopolitics": {
        "label": "Geopolitical risk",
        "need": "Conflict / shipping blockade / sanctions coverage",
        "why": "Safe-haven and USD flows",
        "sources": "Official statements, wire services, inbox topics",
    },
    "oil": {
        "label": "Oil",
        "need": "Oil levels and supply shocks",
        "why": "Energy terms of trade and risk appetite",
        "sources": "Public commodity pages, FRED oil, inbox",
    },
    "fed": {
        "label": "Federal Reserve",
        "need": "FOMC decisions, minutes, speeches, hike odds",
        "why": "USD rate path",
        "sources": "Fed RSS, FRED, inbox",
    },
    "ecb": {
        "label": "ECB",
        "need": "ECB decisions and statements",
        "why": "Euro-area rates and crosses",
        "sources": "ECB RSS, inbox",
    },
    "boe": {
        "label": "Bank of England",
        "need": "BOE policy and inflation-related copy",
        "why": "GBP rate path",
        "sources": "BOE RSS, inbox",
    },
    "boj": {
        "label": "Bank of Japan",
        "need": "BOJ policy / YCC / intervention expectations",
        "why": "JPY rates and carry",
        "sources": "BOJ public notes, inbox",
    },
    "rba": {
        "label": "RBA",
        "need": "RBA decisions, SMP, speeches",
        "why": "AUD rates and domestic narrative",
        "sources": "RBA RSS / SMP, inbox",
    },
    "rbnz": {
        "label": "RBNZ",
        "need": "RBNZ decisions and OCR path",
        "why": "NZD rates",
        "sources": "RBNZ public notes, inbox",
    },
    "boc": {
        "label": "Bank of Canada",
        "need": "BoC decisions and CA inflation/jobs",
        "why": "CAD rates",
        "sources": "BoC public notes, inbox",
    },
    "snb": {
        "label": "SNB",
        "need": "SNB policy and intervention expectations",
        "why": "CHF safe-haven and rates",
        "sources": "SNB public notes, inbox",
    },
    "pboc": {
        "label": "PBOC / FX management",
        "need": "Fixing, liquidity, FX-management signals",
        "why": "CNH/CNY policy bounds",
        "sources": "Public policy notes, inbox",
    },
    "cpi": {
        "label": "Inflation",
        "need": "US and local inflation prints vs consensus",
        "why": "Reprice the policy path",
        "sources": "BLS/national stats, NewsAPI, inbox",
    },
    "growth": {
        "label": "Growth & labour",
        "need": "GDP / PMI / jobs data",
        "why": "Cycle and risk appetite",
        "sources": "Official data releases, inbox",
    },
    "yields": {
        "label": "Yields & bonds",
        "need": "UST vs local yield differentials",
        "why": "Carry and USD direction",
        "sources": "FRED, public macro, inbox",
    },
    "china_growth": {
        "label": "China growth",
        "need": "China growth / stimulus / property",
        "why": "Risk appetite and China-export expectations",
        "sources": "Public macro notes, inbox",
    },
    "china_iron": {
        "label": "Iron ore",
        "need": "Iron-ore prices and China demand",
        "why": "AUD terms of trade",
        "sources": "Public commodity pages, inbox outlooks",
    },
    "dairy": {
        "label": "Dairy",
        "need": "Global dairy auctions / prices",
        "why": "NZD terms of trade",
        "sources": "Public commodity pages, inbox",
    },
    "positioning": {
        "label": "Positioning",
        "need": "Speculative net positioning / crowding",
        "why": "Reversal vs trend-extension risk",
        "sources": "Inbox or manual entry",
    },
    "inbox_research": {
        "label": "Sell-side research",
        "need": "Bank/broker outlooks and targets",
        "why": "Levels and scenario logic → References",
        "sources": "fx_data_apis/inbox/, paste, screenshots, public links",
    },
}


DRIVER_CATALOG: dict[str, DriverSpec] = {
    "spot_vol": DriverSpec(
        "spot_vol", "现价与波动",
        "权威现价与历史波动",
        "蒙特卡洛路径与分档边界",
        "ECB/Frankfurter → FRED → Twelve/Alpha",
    ),
    "geopolitics": DriverSpec(
        "geopolitics", "地缘风险",
        "冲突 / 航道封锁 / 制裁类信息",
        "避险与美元流向",
        "官方声明、一线通讯社、inbox 专题",
    ),
    "oil": DriverSpec(
        "oil", "油价",
        "油价水平与供给冲击",
        "能源贸易条件与风险偏好",
        "公开商品页、FRED 油价、inbox",
    ),
    "fed": DriverSpec(
        "fed", "美联储",
        "FOMC 决议、纪要、官员讲话、加息概率",
        "美元利率路径",
        "Fed RSS、FRED、inbox",
    ),
    "ecb": DriverSpec(
        "ecb", "欧央行",
        "ECB 决议与声明",
        "欧元区利率与交叉盘",
        "ECB RSS、inbox",
    ),
    "boe": DriverSpec(
        "boe", "英央行",
        "BOE 政策与通胀相关稿",
        "英镑利率路径",
        "BOE RSS、inbox",
    ),
    "boj": DriverSpec(
        "boj", "日央行",
        "BOJ 政策 / YCC / 干预预期",
        "日元利率与套息",
        "BOJ 公开稿、inbox",
    ),
    "rba": DriverSpec(
        "rba", "澳储行",
        "RBA 决议、SMP、演讲",
        "澳元利率与本土叙事",
        "RBA RSS / SMP、inbox",
    ),
    "rbnz": DriverSpec(
        "rbnz", "纽储行",
        "RBNZ 决议与 OCR 路径",
        "纽元利率",
        "RBNZ 公开稿、inbox",
    ),
    "boc": DriverSpec(
        "boc", "加央行",
        "BoC 决议与加国通胀/就业",
        "加元利率",
        "BoC 公开稿、inbox",
    ),
    "snb": DriverSpec(
        "snb", "瑞央行",
        "SNB 政策与干预预期",
        "瑞郎避险与利率",
        "SNB 公开稿、inbox",
    ),
    "pboc": DriverSpec(
        "pboc", "中国央行/汇率管理",
        "中间价、流动性、汇率管理信号",
        "CNH/CNY 政策边界",
        "公开政策稿、inbox",
    ),
    "cpi": DriverSpec(
        "cpi", "通胀",
        "美国及本币国通胀读数与意外",
        "重新定价政策路径",
        "BLS/本国统计、NewsAPI、inbox",
    ),
    "growth": DriverSpec(
        "growth", "增长与就业",
        "GDP / PMI / 就业数据",
        "周期与风险偏好",
        "官方数据稿、inbox",
    ),
    "yields": DriverSpec(
        "yields", "利差与国债",
        "美债与本币国利差",
        "套息与美元方向",
        "FRED、公开宏观、inbox",
    ),
    "china_growth": DriverSpec(
        "china_growth", "中国增长",
        "中国增长 / 刺激 / 地产",
        "风险偏好与对华出口预期",
        "公开宏观稿、inbox",
    ),
    "china_iron": DriverSpec(
        "china_iron", "铁矿石",
        "铁矿价格与中国需求",
        "澳贸易条件",
        "公开商品页、inbox 商品展望",
    ),
    "dairy": DriverSpec(
        "dairy", "乳制品",
        "全球乳制品拍卖/价格",
        "纽元贸易条件",
        "公开商品页、inbox",
    ),
    "positioning": DriverSpec(
        "positioning", "仓位情绪",
        "投机净仓 / 拥挤度",
        "反转与趋势延续风险",
        "inbox 或手动录入",
    ),
    "inbox_research": DriverSpec(
        "inbox_research", "机构研报",
        "投行/券商展望与目标价",
        "点位与情景逻辑 → References",
        "fx_data_apis/inbox/、粘贴、截图、公开链接",
    ),
}


# 币种 → 典型驱动（用于任意货币对自动组装）
CURRENCY_DRIVERS: dict[str, tuple[str, ...]] = {
    "USD": ("fed", "cpi", "yields"),
    "EUR": ("ecb", "cpi", "growth"),
    "GBP": ("boe", "cpi", "growth"),
    "JPY": ("boj", "yields"),
    "AUD": ("rba", "china_iron", "china_growth", "oil"),
    "NZD": ("rbnz", "dairy", "china_growth"),
    "CAD": ("boc", "oil", "growth"),
    "CHF": ("snb", "geopolitics"),
    "CNH": ("pboc", "china_growth", "yields"),
    "CNY": ("pboc", "china_growth", "yields"),
}

# 全局因子：几乎所有 G10/主要交叉盘都要看
GLOBAL_DRIVERS: tuple[str, ...] = ("geopolitics", "fed")


def infer_drivers(base: str, quote: str) -> tuple[str, ...]:
    """任意 BASE/QUOTE → 有序去重驱动列表。"""
    base, quote = base.upper(), quote.upper()
    ordered: list[str] = []
    for d in GLOBAL_DRIVERS:
        if d not in ordered:
            ordered.append(d)
    for ccy in (base, quote):
        for d in CURRENCY_DRIVERS.get(ccy, ()):
            if d not in ordered:
                ordered.append(d)
    # 双边都含 USD 时保留 fed/cpi；交叉盘补 growth
    if "USD" not in (base, quote) and "growth" not in ordered:
        ordered.append("growth")
    return tuple(ordered)


def localize_info_need(row: dict[str, str], *, lang: str = "zh") -> dict[str, str]:
    """Return a copy of an info-need row with EN overlays when lang=en."""
    out = dict(row)
    if not str(lang or "").lower().startswith("en"):
        return out
    did = str(out.get("driver") or out.get("id") or "")
    en = DRIVER_CATALOG_EN.get(did)
    if en:
        out.update(en)
        return out
    # Unknown driver fallback
    out["label"] = did or out.get("label") or ""
    out["need"] = f"Information related to driver «{did}»"
    out["why"] = "Pair-specific factor"
    out["sources"] = "Public sources / inbox"
    return out


def info_needs_for_drivers(
    drivers: tuple[str, ...], *, lang: str = "zh"
) -> list[dict[str, str]]:
    """
    步骤2输出：先列本对需要的信息。
    始终包含 spot_vol + inbox_research，再按驱动展开。
    """
    rows: list[dict[str, str]] = []
    for did in ("spot_vol", "inbox_research") + tuple(drivers):
        spec = DRIVER_CATALOG.get(did)
        if not spec:
            rows.append(
                localize_info_need(
                    {
                        "id": did,
                        "label": did,
                        "need": f"与驱动「{did}」相关的信息",
                        "why": "货币对影响因子",
                        "sources": "公开源 / inbox",
                        "driver": did,
                    },
                    lang=lang,
                )
            )
            continue
        rows.append(
            localize_info_need(
                {
                    "id": spec.id,
                    "label": spec.label,
                    "need": spec.need,
                    "why": spec.why,
                    "sources": spec.sources,
                    "driver": spec.id,
                },
                lang=lang,
            )
        )
    # 去重保序
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        out.append(r)
    return out


def describe_pair_factors(
    base: str, quote: str, drivers: tuple[str, ...] | None = None, *, lang: str = "zh"
) -> str:
    drvs = drivers or infer_drivers(base, quote)
    if str(lang or "").lower().startswith("en"):
        labels = [
            DRIVER_CATALOG_EN.get(d, {}).get("label")
            or (DRIVER_CATALOG[d].label if d in DRIVER_CATALOG else d)
            for d in drvs
        ]
        return f"{base}/{quote} drivers: " + ", ".join(labels)
    labels = [DRIVER_CATALOG[d].label if d in DRIVER_CATALOG else d for d in drvs]
    return f"{base}/{quote} 影响因子：" + "、".join(labels)
