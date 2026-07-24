"""Fetch market inputs for any FX pair in the catalog (or custom)."""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    spot: float  # in analysis quote
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
    notes: list[str]

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
        }


def _closes_from_ticker(ticker: str, period: str = "1y") -> pd.Series:
    if yf is None:
        raise RuntimeError("yfinance is required: pip install yfinance")
    hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if hist is None or hist.empty:
        raise RuntimeError(f"No data for {ticker}")
    s = hist["Close"].dropna().astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def _safe_last(ticker: str) -> float | None:
    try:
        s = _closes_from_ticker(ticker, period="1mo")
        return float(s.iloc[-1])
    except Exception:
        return None


def fetch_market(pair: PairSpec | str, lookback_days: int = 60) -> MarketSnapshot:
    """Pull Yahoo series, optionally invert, estimate realized vol on analysis quote."""
    spec = get_pair(pair) if isinstance(pair, str) else pair
    notes: list[str] = []
    if spec.notes:
        notes.append(spec.notes)

    raw = _closes_from_ticker(spec.yahoo_ticker, period="1y")
    series = (1.0 / raw) if spec.invert else raw.copy()
    if len(series) < lookback_days + 2:
        raise RuntimeError(f"Need >{lookback_days} bars for {spec.pair}, got {len(series)}")

    window = series.iloc[-(lookback_days + 1) :]
    rets = np.log(window.values[1:] / window.values[:-1])
    sigma_daily = float(np.std(rets, ddof=1))
    sigma_annual = sigma_daily * math.sqrt(252.0)
    mean_daily = float(np.mean(rets))

    spot = float(series.iloc[-1])
    yahoo_raw = float(raw.iloc[-1])

    brent = _safe_last("BZ=F")
    if brent is None:
        brent = _safe_last("CL=F")
        if brent is not None:
            notes.append("Brent unavailable; used WTI (CL=F).")
    dxy = _safe_last("DX-Y.NYB")
    if dxy is None:
        notes.append("DXY ticker unavailable.")

    invert_note = " (inverted)" if spec.invert else ""
    source = f"Yahoo Finance ({spec.yahoo_ticker}{invert_note} → {spec.pair})"

    return MarketSnapshot(
        asof=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        pair=spec.pair,
        spot=spot,
        yahoo_raw=yahoo_raw,
        sigma_daily=sigma_daily,
        sigma_annual=sigma_annual,
        mean_daily_return=mean_daily,
        n_returns=len(rets),
        lookback_days=lookback_days,
        history_start=str(window.index[0].date()),
        history_end=str(window.index[-1].date()),
        source=source,
        brent=brent,
        dxy_proxy=dxy,
        notes=notes,
    )
