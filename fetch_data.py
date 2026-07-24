"""Fetch live market inputs for any FX pair (with ticker fallbacks)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from pairs import PairSpec, get_pair

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


@dataclass
class MarketSnapshot:
    asof: str
    pair: str
    spot: float
    yahoo_raw: float
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
    # Live derived features used by the report / evidence calibration
    ret_1d: float | None = None
    ret_5d: float | None = None
    ret_20d: float | None = None
    sigma_20d_ann: float | None = None
    sigma_60d_ann: float | None = None
    history_ticker: str = ""
    spot_ticker: str = ""
    used_proxy: bool = False
    cnh_cny_basis: float | None = None  # CNH - CNY if both known

    def to_dict(self) -> dict[str, Any]:
        return {
            "asof": self.asof,
            "pair": self.pair,
            "spot": self.spot,
            "yahoo_raw": self.yahoo_raw,
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


def _closes_from_ticker(ticker: str, period: str = "1y") -> pd.Series:
    if yf is None:
        raise RuntimeError("yfinance is required: pip install yfinance")
    hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if hist is None or hist.empty:
        raise RuntimeError(f"No data for {ticker} period={period}")
    s = hist["Close"].dropna().astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def _try_closes(ticker: str, min_bars: int) -> pd.Series | None:
    """Try several Yahoo periods until we have enough bars."""
    for period in ("1y", "2y", "6mo", "3mo", "1mo", "5d"):
        try:
            s = _closes_from_ticker(ticker, period=period)
        except Exception:
            continue
        if len(s) >= min_bars:
            return s
        # keep best effort if nothing else works later
        if len(s) >= 5:
            # continue seeking longer; store candidate by returning only if enough
            pass
    # last resort: return whatever we can get (>=3)
    for period in ("1y", "5d", "1mo"):
        try:
            s = _closes_from_ticker(ticker, period=period)
            if len(s) >= 3:
                return s
        except Exception:
            continue
    return None


def _safe_last(ticker: str) -> float | None:
    s = _try_closes(ticker, min_bars=1)
    if s is None or s.empty:
        return None
    return float(s.iloc[-1])


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


def fetch_market(pair: PairSpec | str, lookback_days: int = 60) -> MarketSnapshot:
    """
    Pull Yahoo series with fallback tickers.

    For USD/CNH: Yahoo USDCNH=X often has only ~1 bar of history. We then use
    USDCNY=X (onshore) for realized vol / path history, while preferring any
    available CNH last print for spot.
    """
    spec = get_pair(pair) if isinstance(pair, str) else pair
    notes: list[str] = []
    if spec.notes:
        notes.append(spec.notes)

    tickers = (spec.yahoo_ticker,) + tuple(spec.fallback_tickers)
    min_need = max(lookback_days + 2, 25)

    history: pd.Series | None = None
    history_ticker = ""
    for t in tickers:
        s = _try_closes(t, min_bars=min_need)
        if s is None:
            s = _try_closes(t, min_bars=12)
        if s is None:
            continue
        history = s
        history_ticker = t
        if len(s) >= min_need:
            break

    if history is None or len(history) < 12:
        tried = ", ".join(tickers)
        raise RuntimeError(
            f"无法抓取足够历史给 {spec.pair}。已尝试: {tried}。"
            f"Yahoo 对该符号历史不足时请换备用 ticker。"
        )

    used_proxy = history_ticker != spec.yahoo_ticker
    if used_proxy:
        notes.append(
            f"主代码 {spec.yahoo_ticker} 历史不足，已用 {history_ticker} 作为波动/路径代理。"
        )

    raw_hist = history.copy()
    series = (1.0 / raw_hist) if spec.invert else raw_hist

    # Spot: prefer dedicated spot ticker (e.g. thin CNH print) when available
    spot_ticker = spec.spot_ticker or history_ticker
    spot_raw = _safe_last(spot_ticker) if spot_ticker else None
    if spot_raw is None:
        spot_raw = float(raw_hist.iloc[-1])
        spot_ticker = history_ticker
    elif spot_ticker != history_ticker:
        notes.append(f"现价取自 {spot_ticker}={spot_raw:.5f}；波动来自 {history_ticker}。")

    yahoo_raw = float(spot_raw)
    spot = (1.0 / yahoo_raw) if spec.invert else yahoo_raw

    # If we have both CNH and CNY last prints, record basis
    cnh_cny_basis = None
    if spec.pair in {"USD/CNH", "USD/CNY"} or "CNH" in spec.quote or "CNY" in spec.quote:
        cnh = _safe_last("USDCNH=X") or _safe_last("CNH=X")
        cny = _safe_last("USDCNY=X") or _safe_last("CNY=X")
        if cnh is not None and cny is not None:
            cnh_cny_basis = cnh - cny
            notes.append(f"CNH−CNY 价差 ≈ {cnh_cny_basis:+.4f}（正=离岸更弱/美元兑离岸更高）")

    effective_lb = min(lookback_days, len(series) - 1)
    if effective_lb < lookback_days:
        notes.append(f"可用历史仅 {len(series)} 根，波动回看降至 {effective_lb} 日。")
    window = series.iloc[-(effective_lb + 1) :]
    rets = np.log(window.values[1:] / window.values[:-1])
    sigma_daily = float(np.std(rets, ddof=1))
    sigma_annual = sigma_daily * math.sqrt(252.0)
    mean_daily = float(np.mean(rets))

    closes = series.values.astype(float)
    brent = _safe_last("BZ=F")
    if brent is None:
        brent = _safe_last("CL=F")
        if brent is not None:
            notes.append("Brent unavailable; used WTI (CL=F).")
    dxy = _safe_last("DX-Y.NYB")
    if dxy is None:
        notes.append("DXY ticker unavailable.")

    invert_note = " (inverted)" if spec.invert else ""
    proxy_note = f", hist={history_ticker}" if used_proxy else ""
    source = (
        f"Yahoo Finance (spot={spot_ticker}{invert_note}{proxy_note} → {spec.pair})"
    )

    return MarketSnapshot(
        asof=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        pair=spec.pair,
        spot=spot,
        yahoo_raw=yahoo_raw,
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
        spot_ticker=spot_ticker,
        used_proxy=used_proxy,
        cnh_cny_basis=cnh_cny_basis,
    )


def calibrate_unpriced_from_market(ret_1d: float | None, ret_5d: float | None) -> float:
    """
    If the pair already moved hard, news is more priced-in → lower unpriced.
    Returns a suggested unpriced in [0.2, 0.9].
    """
    move = 0.0
    if ret_1d is not None:
        move = max(move, abs(ret_1d))
    if ret_5d is not None:
        move = max(move, abs(ret_5d) * 0.5)
    # 0.5% ~ lightly priced, 1.5%+ heavily priced for G10 FX day move
    if move >= 0.015:
        return 0.25
    if move >= 0.008:
        return 0.40
    if move >= 0.004:
        return 0.55
    return 0.70
