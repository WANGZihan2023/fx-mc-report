#!/usr/bin/env python3
"""Scan historical news evidence dates and compare peak engines A vs C.

Prerequisite (API keys):
  set -a && source railway-variables.env && set +a

Example:
  python scripts/auto_replay_engine_compare.py --pair USD/AUD
  python scripts/auto_replay_engine_compare.py --dates 2026-07-10,2026-07-15 --max-dates 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fx_report.model.replay_engine_compare import print_chinese_summary, run_engine_compare


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "自动扫描有历史新闻证据的 as_of，对比引擎 A "
            "(path_max+merton+antithetic) vs C (brownian_bridge+none+antithetic)"
        )
    )
    p.add_argument("--pair", default="USD/AUD")
    p.add_argument("--bullish", default=None, help="看涨货币；缺省=base")
    p.add_argument(
        "--start",
        default=None,
        help="扫描起点（默认 today-25；NewsAPI 约近 29 天）",
    )
    p.add_argument(
        "--end",
        default=None,
        help="扫描终点（默认 today-(days+1)，给实现窗口留余量）",
    )
    p.add_argument("--step", type=int, default=3, help="扫描步长（自然日）")
    p.add_argument(
        "--dates",
        default=None,
        help="跳过新闻扫描，直接指定 as_of 列表，逗号分隔",
    )
    p.add_argument(
        "--max-dates",
        type=int,
        default=3,
        help="最多对比几个 as_of（默认 3，控 NewsAPI 配额）",
    )
    p.add_argument("--sims", type=int, default=800)
    p.add_argument("--days", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lookback", type=int, default=14)
    p.add_argument("--max-news", type=int, default=10)
    p.add_argument("--mode", choices=["hybrid", "llm", "rules"], default="rules")
    p.add_argument("--out", default="output/engine_compare")
    p.add_argument("--calibrated-params", default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_engine_compare(
            args.pair,
            start_date=args.start,
            end_date=args.end,
            step_days=args.step,
            dates=args.dates,
            max_dates=args.max_dates,
            sims=args.sims,
            days=args.days,
            seed=args.seed,
            lookback=args.lookback,
            max_news=args.max_news,
            mode=args.mode,
            out_dir=args.out,
            bullish_currency=args.bullish,
            calibrated_params_path=args.calibrated_params,
            verbose=not args.quiet,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print_chinese_summary(result)
    return 0 if result.rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
