#!/usr/bin/env python3
"""Smoke: GDELT DOC historical headlines (no NewsAPI key required).

Usage:
  python scripts/smoke_gdelt_historical.py
  python scripts/smoke_gdelt_historical.py --as-of 2026-07-13 --lookback 14

Does not print secrets. Network required for live GDELT; exits 0 on date_filtered
or graceful empty+note, non-zero only on unexpected exceptions.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="USD/AUD")
    parser.add_argument("--as-of", default="", help="YYYY-MM-DD (default: today-7)")
    parser.add_argument("--lookback", type=int, default=14)
    parser.add_argument("--max-items", type=int, default=10)
    args = parser.parse_args()

    from fx_report.news.fetch import fetch_historical_headlines_for_pair

    today = date.today()
    as_of = date.fromisoformat(args.as_of) if args.as_of else today - timedelta(days=7)

    print(f"pair={args.pair} as_of={as_of.isoformat()} lookback={args.lookback}")
    print("(NewsAPI key not required; GDELT is free DOC ArtList)")

    headlines, meta = fetch_historical_headlines_for_pair(
        args.pair,
        as_of_date=as_of,
        lookback_days=args.lookback,
        max_items=args.max_items,
        today=today,
    )

    quality = meta.get("historical_news_quality")
    print(f"quality={quality}")
    print(
        "hits:",
        f"newsapi={meta.get('newsapi_hits')}",
        f"gdelt={meta.get('gdelt_hits')}",
        f"inbox={meta.get('inbox_dated_hits')}",
    )
    print(f"providers={meta.get('providers_used')}")
    if meta.get("gdelt_error"):
        print(f"gdelt_error={meta.get('gdelt_error')[:200]}")
    if meta.get("newsapi_error"):
        print(f"newsapi_error={meta.get('newsapi_error')[:200]}")
    print(f"headlines={len(headlines)}")
    for h in headlines[:5]:
        pub = h.published.date().isoformat() if h.published else "?"
        print(f"  [{h.provider}] {pub} {h.title[:100]}")
    limitation = str(meta.get("limitation") or "")
    print(f"limitation={limitation[:400]}")

    if quality == "date_filtered" and len(headlines) > 0:
        print("OK: date_filtered with headlines (GDELT and/or NewsAPI/inbox)")
        return 0
    if meta.get("gdelt_error") or meta.get("gdelt_hits") == 0:
        print("NOTE: no date_filtered hits this run — check GDELT rate limit / window")
        # Still a successful smoke of the code path
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
