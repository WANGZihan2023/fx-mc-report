"""Currency-pair registry — anything quoteable as BASE/QUOTE on Yahoo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PairSpec:
    """
    Analysis always uses `pair` as displayed (e.g. USD/AUD, EUR/USD, USD/CNH).

    Yahoo often lists the inverted ticker (AUDUSD=X). Set `invert=True` when
    Yahoo close must be inverted to obtain the analysis quote.

    `fallback_tickers`: tried in order if the primary series has too few bars
    (common for thin Yahoo FX symbols like USDCNH=X).
    """

    pair: str
    yahoo_ticker: str
    invert: bool
    base: str
    quote: str
    description: str
    bucket_pct_cuts: tuple[float, float, float, float] = (0.0, 2.0, 4.0, 6.0)
    default_drivers: tuple[str, ...] = ()
    notes: str = ""
    fallback_tickers: tuple[str, ...] = ()
    # Optional separate spot ticker when history comes from a proxy
    spot_ticker: str | None = None


PAIR_CATALOG: dict[str, PairSpec] = {
    "USD/AUD": PairSpec(
        pair="USD/AUD",
        yahoo_ticker="AUDUSD=X",
        invert=True,
        base="USD",
        quote="AUD",
        description="美元兑澳元（分析口径：1 美元可兑多少澳元）",
        bucket_pct_cuts=(0.0, 2.0, 4.0, 6.0),
        default_drivers=("geopolitics", "oil", "fed", "rba", "china_iron", "cpi"),
        notes="Yahoo 为 AUDUSD，取倒数。",
    ),
    "AUD/USD": PairSpec(
        pair="AUD/USD",
        yahoo_ticker="AUDUSD=X",
        invert=False,
        base="AUD",
        quote="USD",
        description="澳元兑美元（市场常报价）",
        bucket_pct_cuts=(0.0, 2.0, 4.0, 6.0),
        default_drivers=("geopolitics", "oil", "fed", "rba", "china_iron", "cpi"),
    ),
    "EUR/USD": PairSpec(
        pair="EUR/USD",
        yahoo_ticker="EURUSD=X",
        invert=False,
        base="EUR",
        quote="USD",
        description="欧元兑美元",
        bucket_pct_cuts=(0.0, 1.5, 3.0, 4.5),
        default_drivers=("fed", "ecb", "cpi", "geopolitics", "growth"),
    ),
    "GBP/USD": PairSpec(
        pair="GBP/USD",
        yahoo_ticker="GBPUSD=X",
        invert=False,
        base="GBP",
        quote="USD",
        description="英镑兑美元",
        bucket_pct_cuts=(0.0, 1.5, 3.0, 4.5),
        default_drivers=("fed", "boe", "cpi", "geopolitics", "growth"),
    ),
    "USD/JPY": PairSpec(
        pair="USD/JPY",
        yahoo_ticker="USDJPY=X",
        invert=False,
        base="USD",
        quote="JPY",
        description="美元兑日元",
        bucket_pct_cuts=(0.0, 2.0, 4.0, 6.0),
        default_drivers=("fed", "boj", "yields", "geopolitics", "cpi"),
    ),
    "USD/CNH": PairSpec(
        pair="USD/CNH",
        yahoo_ticker="USDCNH=X",
        invert=False,
        base="USD",
        quote="CNH",
        description="美元兑离岸人民币",
        bucket_pct_cuts=(0.0, 1.0, 2.0, 3.5),
        default_drivers=("fed", "pboc", "china_growth", "geopolitics", "yields"),
        notes=(
            "Yahoo USDCNH=X 历史极薄；自动回退 USDCNY=X 估波动，"
            "现价优先用 CNH 最新点（若有），并标注在岸代理。"
        ),
        fallback_tickers=("CNH=X", "USDCNY=X", "CNY=X"),
        spot_ticker="USDCNH=X",
    ),
    "USD/CNY": PairSpec(
        pair="USD/CNY",
        yahoo_ticker="USDCNY=X",
        invert=False,
        base="USD",
        quote="CNY",
        description="美元兑在岸人民币",
        bucket_pct_cuts=(0.0, 1.0, 2.0, 3.5),
        default_drivers=("fed", "pboc", "china_growth", "geopolitics", "yields"),
        fallback_tickers=("CNY=X",),
    ),
    "USD/CAD": PairSpec(
        pair="USD/CAD",
        yahoo_ticker="USDCAD=X",
        invert=False,
        base="USD",
        quote="CAD",
        description="美元兑加元",
        bucket_pct_cuts=(0.0, 1.5, 3.0, 4.5),
        default_drivers=("fed", "boc", "oil", "cpi", "growth"),
    ),
    "NZD/USD": PairSpec(
        pair="NZD/USD",
        yahoo_ticker="NZDUSD=X",
        invert=False,
        base="NZD",
        quote="USD",
        description="纽元兑美元",
        bucket_pct_cuts=(0.0, 2.0, 4.0, 6.0),
        default_drivers=("fed", "rbnz", "dairy", "china_growth", "geopolitics"),
    ),
    "USD/CHF": PairSpec(
        pair="USD/CHF",
        yahoo_ticker="USDCHF=X",
        invert=False,
        base="USD",
        quote="CHF",
        description="美元兑瑞郎",
        bucket_pct_cuts=(0.0, 1.5, 3.0, 4.5),
        default_drivers=("fed", "snb", "geopolitics", "cpi", "yields"),
    ),
}


def list_pairs() -> list[str]:
    return list(PAIR_CATALOG.keys())


def get_pair(pair: str) -> PairSpec:
    if pair not in PAIR_CATALOG:
        raise KeyError(f"Unknown pair {pair!r}. Known: {', '.join(list_pairs())}")
    return PAIR_CATALOG[pair]


def edges_from_spot(spot: float, pct_cuts: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return tuple(spot * (1.0 + p / 100.0) for p in pct_cuts)  # type: ignore[return-value]


def make_custom_pair(
    pair: str,
    yahoo_ticker: str,
    invert: bool,
    *,
    bucket_pct_cuts: tuple[float, float, float, float] = (0.0, 2.0, 4.0, 6.0),
    fallback_tickers: tuple[str, ...] = (),
) -> PairSpec:
    parts = pair.replace(" ", "").split("/")
    if len(parts) != 2:
        raise ValueError("pair must look like BASE/QUOTE")
    base, quote = parts[0].upper(), parts[1].upper()
    return PairSpec(
        pair=f"{base}/{quote}",
        yahoo_ticker=yahoo_ticker,
        invert=invert,
        base=base,
        quote=quote,
        description=f"自定义 {base}/{quote}",
        bucket_pct_cuts=bucket_pct_cuts,
        default_drivers=("fed", "cpi", "geopolitics", "growth"),
        notes="用户自定义货币对",
        fallback_tickers=fallback_tickers,
    )
