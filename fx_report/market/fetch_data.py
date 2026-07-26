"""Fetch market data from authoritative sources only.

Priority (fill vault key → use; empty → skip):
  1) Frankfurter / ECB reference rates — no key required
  2) FRED (Fed St. Louis) — needs FRED_API_KEY
  3) Twelve Data / Alpha Vantage — optional paid/free keys
"""

from __future__ import annotations

import json
import math
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from fx_report.config.api_config import is_set, load_config, timeout_s
from fx_report.market.pairs import PairSpec, get_pair

# FRED: series_id, invert_to_analysis_quote
FRED_SERIES: dict[str, tuple[str, bool]] = {
    "AUD/USD": ("DEXUSAL", False),
    "USD/AUD": ("DEXUSAL", True),
    "EUR/USD": ("DEXUSEU", False),
    "USD/EUR": ("DEXUSEU", True),
    "GBP/USD": ("DEXUSUK", False),
    "USD/GBP": ("DEXUSUK", True),
    "USD/JPY": ("DEXJPUS", False),
    "USD/CAD": ("DEXCAUS", False),
    "USD/CHF": ("DEXSZUS", False),
    "NZD/USD": ("DEXUSNZ", False),
    "USD/NZD": ("DEXUSNZ", True),
    "USD/CNY": ("DEXCHUS", False),
    "USD/CNH": ("DEXCHUS", False),  # onshore proxy; flagged in notes
}

FRANKFURTER_ALIAS: dict[str, str] = {
    "CNH": "CNY",  # ECB publishes CNY, not CNH
}


@dataclass
class MarketSnapshot:
    asof: str
    pair: str
    spot: float
    provider_raw: float  # raw print from ECB/FRED/Twelve/… before analysis invert
    sigma_daily: float
    sigma_annual: float
    mean_daily_return: float
    n_returns: int
    lookback_days: int
    history_start: str
    history_end: str
    source: str
    brent: float | None
    dxy_proxy: float | None
    notes: list[str] = field(default_factory=list)
    ret_1d: float | None = None
    ret_5d: float | None = None
    ret_20d: float | None = None
    sigma_20d_ann: float | None = None
    sigma_60d_ann: float | None = None
    history_ticker: str = ""
    spot_ticker: str = ""
    used_proxy: bool = False
    cnh_cny_basis: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "asof": self.asof,
            "pair": self.pair,
            "spot": self.spot,
            "provider_raw": self.provider_raw,
            "sigma_daily": self.sigma_daily,
            "sigma_annual": self.sigma_annual,
            "mean_daily_return": self.mean_daily_return,
            "n_returns": self.n_returns,
            "lookback_days": self.lookback_days,
            "history_start": self.history_start,
            "history_end": self.history_end,
            "source": self.source,
            "brent": self.brent,
            "dxy_proxy": self.dxy_proxy,
            "notes": self.notes,
            "ret_1d": self.ret_1d,
            "ret_5d": self.ret_5d,
            "ret_20d": self.ret_20d,
            "sigma_20d_ann": self.sigma_20d_ann,
            "sigma_60d_ann": self.sigma_60d_ann,
            "history_ticker": self.history_ticker,
            "spot_ticker": self.spot_ticker,
            "used_proxy": self.used_proxy,
            "cnh_cny_basis": self.cnh_cny_basis,
        }


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
        try:
            ctx.load_default_certs()
        except Exception:
            pass
        # macOS 系统 Python 常见缺 CA；官方公开 API 允许降级校验
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


def _http_json(url: str, timeout: int) -> Any:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "FXReportFetcher/2.0 (+authoritative-sources)"},
    )
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _series_from_closes(closes: list[tuple[pd.Timestamp, float]]) -> pd.Series:
    if not closes:
        raise RuntimeError("empty closes")
    idx, vals = zip(*closes)
    s = pd.Series(vals, index=pd.DatetimeIndex(idx), dtype=float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.dropna()


def _ann_vol(closes: np.ndarray) -> float | None:
    if len(closes) < 3:
        return None
    rets = np.log(closes[1:] / closes[:-1])
    if len(rets) < 2:
        return None
    return float(np.std(rets, ddof=1) * math.sqrt(252.0))


def _pct_change(series: pd.Series, n: int) -> float | None:
    if len(series) <= n:
        return None
    a, b = float(series.iloc[-1]), float(series.iloc[-1 - n])
    if b == 0:
        return None
    return a / b - 1.0


def _fetch_frankfurter_series(spec: PairSpec, cfg: dict[str, str]) -> tuple[pd.Series, list[str]] | None:
    """ECB reference rates via Frankfurter — no API key."""
    notes: list[str] = []
    base = spec.base
    quote = FRANKFURTER_ALIAS.get(spec.quote, spec.quote)
    used_proxy = quote != spec.quote
    if used_proxy:
        notes.append(f"ECB/Frankfurter 无 {spec.quote}，暂用 {quote} 代理（请知悉离岸/在岸差异）。")

    start = (date.today() - timedelta(days=400)).isoformat()
    url = f"https://api.frankfurter.dev/v1/{start}..?from={base}&to={quote}"
    invert = False
    try:
        data = _http_json(url, timeout_s(cfg))
    except Exception:
        url2 = f"https://api.frankfurter.dev/v1/{start}..?from={quote}&to={base}"
        try:
            data = _http_json(url2, timeout_s(cfg))
            invert = True
        except Exception:
            return None

    if not isinstance(data, dict) or "rates" not in data:
        return None
    rows: list[tuple[pd.Timestamp, float]] = []
    for day, vals in data["rates"].items():
        try:
            key = quote if not invert else base
            px = float(vals[key])
            if invert and px != 0:
                px = 1.0 / px
            rows.append((pd.Timestamp(day), px))
        except Exception:
            continue
    if len(rows) < 12:
        return None
    notes.insert(0, "行情来自 ECB 参考汇率（Frankfurter，无需 Key）")
    return _series_from_closes(rows), notes


def _fetch_fred_obs(series_id: str, cfg: dict[str, str], limit: int = 200) -> pd.Series | None:
    if not is_set(cfg, "FRED_API_KEY"):
        return None
    q = urllib.parse.urlencode(
        {
            "series_id": series_id,
            "api_key": cfg["FRED_API_KEY"],
            "file_type": "json",
            "sort_order": "desc",
            "limit": str(limit),
        }
    )
    url = f"https://api.stlouisfed.org/fred/series/observations?{q}"
    try:
        data = _http_json(url, timeout_s(cfg))
    except Exception:
        return None
    obs = data.get("observations") if isinstance(data, dict) else None
    if not isinstance(obs, list):
        return None
    rows: list[tuple[pd.Timestamp, float]] = []
    for o in obs:
        v = o.get("value")
        if v in (None, "."):
            continue
        try:
            rows.append((pd.Timestamp(o["date"]), float(v)))
        except Exception:
            continue
    if len(rows) < 5:
        return None
    return _series_from_closes(rows)


def _fetch_fred_fx(spec: PairSpec, cfg: dict[str, str]) -> tuple[pd.Series, list[str]] | None:
    mapped = FRED_SERIES.get(spec.pair)
    if not mapped:
        return None
    series_id, need_invert = mapped
    s = _fetch_fred_obs(series_id, cfg, limit=200)
    if s is None:
        return None
    if need_invert:
        s = 1.0 / s.replace(0, np.nan)
        s = s.dropna()
    notes = [f"行情来自 FRED `{series_id}`（美联储圣路易斯）"]
    if spec.pair == "USD/CNH":
        notes.append("FRED 为在岸 USD/CNY，用作 CNH 代理。")
    return s, notes


def _fetch_twelve_series(spec: PairSpec, cfg: dict[str, str]) -> tuple[pd.Series, list[str]] | None:
    if not is_set(cfg, "TWELVE_DATA_API_KEY"):
        return None
    symbol = f"{spec.base}/{spec.quote}"
    q = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": "1day",
            "outputsize": "120",
            "apikey": cfg["TWELVE_DATA_API_KEY"],
        }
    )
    url = f"https://api.twelvedata.com/time_series?{q}"
    try:
        data = _http_json(url, timeout_s(cfg))
    except Exception:
        return None
    if not isinstance(data, dict) or "values" not in data:
        return None
    rows: list[tuple[pd.Timestamp, float]] = []
    for row in data["values"]:
        try:
            rows.append((pd.Timestamp(row["datetime"]), float(row["close"])))
        except Exception:
            continue
    if len(rows) < 12:
        return None
    return _series_from_closes(rows), ["行情来自 Twelve Data（vault Key）"]


def _fetch_alpha_series(spec: PairSpec, cfg: dict[str, str]) -> tuple[pd.Series, list[str]] | None:
    if not is_set(cfg, "ALPHA_VANTAGE_API_KEY"):
        return None
    q = urllib.parse.urlencode(
        {
            "function": "FX_DAILY",
            "from_symbol": spec.base,
            "to_symbol": spec.quote,
            "apikey": cfg["ALPHA_VANTAGE_API_KEY"],
            "outputsize": "compact",
        }
    )
    url = f"https://www.alphavantage.co/query?{q}"
    try:
        data = _http_json(url, timeout_s(cfg))
    except Exception:
        return None
    series = data.get("Time Series FX (Daily)") if isinstance(data, dict) else None
    if not isinstance(series, dict):
        return None
    rows: list[tuple[pd.Timestamp, float]] = []
    for day, vals in series.items():
        try:
            rows.append((pd.Timestamp(day), float(vals["4. close"])))
        except Exception:
            continue
    if len(rows) < 12:
        return None
    return _series_from_closes(rows), ["行情来自 Alpha Vantage（vault Key）"]


def _fred_last(series_id: str, cfg: dict[str, str]) -> float | None:
    s = _fetch_fred_obs(series_id, cfg, limit=5)
    if s is None or s.empty:
        return None
    return float(s.iloc[-1])


def _snapshot_from_series(
    spec: PairSpec,
    series: pd.Series,
    *,
    lookback_days: int,
    source: str,
    history_ticker: str,
    notes: list[str],
    used_proxy: bool = False,
    brent: float | None = None,
    dxy: float | None = None,
) -> MarketSnapshot:
    effective_lb = min(lookback_days, len(series) - 1)
    if effective_lb < lookback_days:
        notes.append(f"可用历史仅 {len(series)} 根，波动回看降至 {effective_lb} 日。")
    window = series.iloc[-(effective_lb + 1) :]
    rets = np.log(window.values[1:] / window.values[:-1])
    sigma_daily = float(np.std(rets, ddof=1))
    sigma_annual = sigma_daily * math.sqrt(252.0)
    mean_daily = float(np.mean(rets))
    spot = float(series.iloc[-1])
    closes = series.values.astype(float)
    return MarketSnapshot(
        asof=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        pair=spec.pair,
        spot=spot,
        provider_raw=spot,
        sigma_daily=sigma_daily,
        sigma_annual=sigma_annual,
        mean_daily_return=mean_daily,
        n_returns=len(rets),
        lookback_days=effective_lb,
        history_start=str(window.index[0].date()),
        history_end=str(window.index[-1].date()),
        source=source,
        brent=brent,
        dxy_proxy=dxy,
        notes=notes,
        ret_1d=_pct_change(series, 1),
        ret_5d=_pct_change(series, 5),
        ret_20d=_pct_change(series, 20),
        sigma_20d_ann=_ann_vol(closes[-21:]) if len(closes) >= 22 else None,
        sigma_60d_ann=_ann_vol(closes[-61:]) if len(closes) >= 62 else sigma_annual,
        history_ticker=history_ticker,
        spot_ticker=history_ticker,
        used_proxy=used_proxy,
        cnh_cny_basis=None,
    )


def fetch_market(
    pair: PairSpec | str,
    lookback_days: int = 60,
) -> MarketSnapshot:
    """
    Authoritative FX only (ECB/Frankfurter → FRED → Twelve → Alpha).
    """
    spec = get_pair(pair) if isinstance(pair, str) else pair
    cfg = load_config()
    notes: list[str] = []
    if spec.notes:
        notes.append(spec.notes)

    candidates: list[tuple[str, Any]] = [
        ("ECB/Frankfurter", _fetch_frankfurter_series),
        ("FRED", _fetch_fred_fx),
        ("Twelve Data", _fetch_twelve_series),
        ("Alpha Vantage", _fetch_alpha_series),
    ]

    series: pd.Series | None = None
    source = ""
    history_ticker = ""
    used_proxy = False

    for name, fetcher in candidates:
        got = fetcher(spec, cfg)
        if got is None:
            continue
        s, extra_notes = got
        if s is None or len(s) < 12:
            continue
        series = s
        source = name
        history_ticker = name
        notes.extend(extra_notes)
        used_proxy = any("代理" in n for n in extra_notes)
        break

    if series is None:
        raise RuntimeError(
            f"无法从权威源抓取 {spec.pair}。\n"
            "已尝试：ECB/Frankfurter（免 Key）、FRED、Twelve Data、Alpha Vantage。\n"
            "请检查网络，或在 API 配置中填写 FRED / Twelve Data Key。"
        )

    brent = _fred_last("DCOILBRENTEU", cfg)
    dxy = _fred_last("DTWEXBGS", cfg)
    if brent is not None:
        notes.append(f"Brent 来自 FRED DCOILBRENTEU={brent:.2f}")
    if dxy is not None:
        notes.append(f"美元指数代理来自 FRED DTWEXBGS={dxy:.2f}")

    return _snapshot_from_series(
        spec,
        series,
        lookback_days=lookback_days,
        source=source,
        history_ticker=history_ticker,
        notes=notes,
        used_proxy=used_proxy,
        brent=brent,
        dxy=dxy,
    )


def calibrate_unpriced_from_market(ret_1d: float | None, ret_5d: float | None) -> float:
    move = 0.0
    if ret_1d is not None:
        move = max(move, abs(ret_1d))
    if ret_5d is not None:
        move = max(move, abs(ret_5d) * 0.5)
    if move >= 0.015:
        return 0.25
    if move >= 0.008:
        return 0.40
    if move >= 0.004:
        return 0.55
    return 0.70
