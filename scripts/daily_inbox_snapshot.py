#!/usr/bin/env python3
"""Daily RSS → inbox snapshot (stub / future archiving helper).

Goal: once per day, capture official/public FX RSS (and optional Google News RSS)
into the vault inbox as dated markdown so freeze-replay can use local dated
files when NewsAPI/GDELT windows miss or rate-limit.

This is intentionally a light stub: it writes a dated note under inbox/ (or a
dry-run path) without requiring paid API keys. Wire to cron/launchd later.

Examples:
  python scripts/daily_inbox_snapshot.py --dry-run
  python scripts/daily_inbox_snapshot.py --pair USD/AUD

No secrets are printed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="USD/AUD")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned snapshot path and headline count; do not write",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Override inbox directory (default: vault inbox from config)",
    )
    args = parser.parse_args()

    from fx_report.config.api_config import load_config, vault_paths
    from fx_report.news.fetch import fetch_official_rss, fetch_google_news_rss

    cfg = load_config()
    paths = vault_paths(cfg)
    inbox = Path(args.out_dir) if args.out_dir else Path(paths.get("inbox") or (ROOT / "inbox"))
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d")
    out_path = inbox / f"rss_snapshot_{args.pair.replace('/', '')}_{stamp}.md"

    headlines = []
    try:
        headlines.extend(fetch_official_rss(args.pair))
    except Exception as e:
        print(f"official_rss_error={type(e).__name__}: {e}")
    try:
        headlines.extend(fetch_google_news_rss(args.pair, limit=12))
    except Exception as e:
        print(f"google_news_rss_error={type(e).__name__}: {e}")

    lines = [
        f"# RSS inbox snapshot — {args.pair}",
        f"",
        f"- captured_at_utc: {now.isoformat()}",
        f"- pair: {args.pair}",
        f"- n_headlines: {len(headlines)}",
        f"",
        "## Headlines",
        "",
    ]
    for h in headlines[:40]:
        pub = h.published.isoformat() if h.published else ""
        lines.append(f"- [{h.provider}] {pub} | {h.source} | {h.title}")
        if h.url:
            lines.append(f"  - url: {h.url}")

    body = "\n".join(lines) + "\n"
    print(f"pair={args.pair} headlines={len(headlines)} out={out_path}")
    if args.dry_run:
        print("dry-run: not writing file")
        print(body[:500])
        return 0

    inbox.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    print(f"wrote={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
