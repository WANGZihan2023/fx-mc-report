"""Currency-pair registry for authoritative FX fetch (ECB / FRED / vault APIs)."""

from __future__ import annotations

from dataclasses import dataclass

from fx_report.market.pair_drivers import infer_drivers


@dataclass(frozen=True)
class PairSpec:
    """
    Analysis always uses `pair` as displayed (e.g. USD/AUD, EUR/USD, USD/CNH).

    `default_drivers` = 本货币对的影响因子清单（步骤2的输入）。
    不同货币对因子不同；未手写时由 base/quote 自动推断。
    """

    pair: str
    symbol_code: str
    invert: bool
    base: str
    quote: str
    description: str
    bucket_pct_cuts: tuple[float, float, float, float] = (0.0, 2.0, 4.0, 6.0)
    default_drivers: tuple[str, ...] = ()
    notes: str = ""
    fallback_tickers: tuple[str, ...] = ()
    spot_ticker: str | None = None


def _pair(
    pair: str,
    *,
    symbol_code: str,
    invert: bool,
    description: str,
    bucket_pct_cuts: tuple[float, float, float, float] = (0.0, 2.0, 4.0, 6.0),
    drivers: tuple[str, ...] | None = None,
    notes: str = "",
    fallback_tickers: tuple[str, ...] = (),
    spot_ticker: str | None = None,
) -> PairSpec:
    base, quote = pair.split("/")
    return PairSpec(
        pair=pair,
        symbol_code=symbol_code,
        invert=invert,
        base=base,
        quote=quote,
        description=description,
        bucket_pct_cuts=bucket_pct_cuts,
        default_drivers=drivers if drivers is not None else infer_drivers(base, quote),
        notes=notes,
        fallback_tickers=fallback_tickers,
        spot_ticker=spot_ticker,
    )


PAIR_CATALOG: dict[str, PairSpec] = {
    "USD/AUD": _pair(
        "USD/AUD",
        symbol_code="AUDUSD",
        invert=True,
        description="美元兑澳元（分析口径：1 美元可兑多少澳元）",
        drivers=("geopolitics", "oil", "fed", "rba", "china_iron", "cpi"),
        notes="分析口径：1 美元兑多少澳元。行情默认取 ECB 参考汇率。",
    ),
    "AUD/USD": _pair(
        "AUD/USD",
        symbol_code="AUDUSD",
        invert=False,
        description="澳元兑美元（市场常报价）",
        drivers=("geopolitics", "oil", "fed", "rba", "china_iron", "cpi"),
        notes="行情默认取 ECB 参考汇率。",
    ),
    "EUR/USD": _pair(
        "EUR/USD",
        symbol_code="EURUSD",
        invert=False,
        description="欧元兑美元",
        bucket_pct_cuts=(0.0, 1.5, 3.0, 4.5),
        drivers=("fed", "ecb", "cpi", "geopolitics", "growth"),
    ),
    "GBP/USD": _pair(
        "GBP/USD",
        symbol_code="GBPUSD",
        invert=False,
        description="英镑兑美元",
        bucket_pct_cuts=(0.0, 1.5, 3.0, 4.5),
        drivers=("fed", "boe", "cpi", "geopolitics", "growth"),
    ),
    "USD/JPY": _pair(
        "USD/JPY",
        symbol_code="USDJPY",
        invert=False,
        description="美元兑日元",
        drivers=("fed", "boj", "yields", "geopolitics", "cpi"),
    ),
    "USD/CNH": _pair(
        "USD/CNH",
        symbol_code="USDCNH",
        invert=False,
        description="美元兑离岸人民币",
        bucket_pct_cuts=(0.0, 1.0, 2.0, 3.5),
        drivers=("fed", "pboc", "china_growth", "geopolitics", "yields"),
        notes=(
            "ECB/FRED 通常只有在岸 CNY；抓取时会用 CNY 代理并标注。"
            "若需真正 CNH，请在 vault 配置 Twelve Data / Alpha Vantage。"
        ),
    ),
    "USD/CNY": _pair(
        "USD/CNY",
        symbol_code="USDCNY",
        invert=False,
        description="美元兑在岸人民币",
        bucket_pct_cuts=(0.0, 1.0, 2.0, 3.5),
        drivers=("fed", "pboc", "china_growth", "geopolitics", "yields"),
    ),
    "USD/CAD": _pair(
        "USD/CAD",
        symbol_code="USDCAD",
        invert=False,
        description="美元兑加元",
        bucket_pct_cuts=(0.0, 1.5, 3.0, 4.5),
        drivers=("fed", "boc", "oil", "cpi", "growth"),
    ),
    "NZD/USD": _pair(
        "NZD/USD",
        symbol_code="NZDUSD",
        invert=False,
        description="纽元兑美元",
        drivers=("fed", "rbnz", "dairy", "china_growth", "geopolitics"),
    ),
    "USD/CHF": _pair(
        "USD/CHF",
        symbol_code="USDCHF",
        invert=False,
        description="美元兑瑞郎",
        bucket_pct_cuts=(0.0, 1.5, 3.0, 4.5),
        drivers=("fed", "snb", "geopolitics", "cpi", "yields"),
    ),
}


def list_pairs() -> list[str]:
    return list(PAIR_CATALOG.keys())


def get_pair(pair: str) -> PairSpec:
    """目录内返回精调因子；目录外自动按币种推断影响因子（支持任意货币对）。"""
    key = pair.replace(" ", "").upper()
    if key in PAIR_CATALOG:
        return PAIR_CATALOG[key]
    if "/" not in key and len(key) == 6:
        key = f"{key[:3]}/{key[3:]}"
    if key in PAIR_CATALOG:
        return PAIR_CATALOG[key]
    if "/" not in key:
        raise KeyError(
            f"Unknown pair {pair!r}. Use BASE/QUOTE. Known: {', '.join(list_pairs())}"
        )
    base, quote = key.split("/", 1)
    return make_custom_pair(f"{base}/{quote}", f"{base}{quote}", invert=False)


def edges_from_spot(
    spot: float, pct_cuts: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    return tuple(spot * (1.0 + p / 100.0) for p in pct_cuts)  # type: ignore[return-value]


def make_custom_pair(
    pair: str,
    symbol_code: str,
    invert: bool,
    *,
    bucket_pct_cuts: tuple[float, float, float, float] = (0.0, 2.0, 4.0, 6.0),
    fallback_tickers: tuple[str, ...] = (),
    drivers: tuple[str, ...] | None = None,
) -> PairSpec:
    parts = pair.replace(" ", "").split("/")
    if len(parts) != 2:
        raise ValueError("pair must look like BASE/QUOTE")
    base, quote = parts[0].upper(), parts[1].upper()
    drvs = drivers if drivers is not None else infer_drivers(base, quote)
    return PairSpec(
        pair=f"{base}/{quote}",
        symbol_code=symbol_code or f"{base}{quote}",
        invert=invert,
        base=base,
        quote=quote,
        description=f"自定义 {base}/{quote}｜影响因子由币种自动推断",
        bucket_pct_cuts=bucket_pct_cuts,
        default_drivers=drvs,
        notes=f"自动影响因子：{', '.join(drvs)}",
        fallback_tickers=fallback_tickers,
    )
