#!/usr/bin/env python3
"""CLI：按七步流水线运行（见 fx_report/pipeline.py）。"""

from __future__ import annotations

import argparse

from fx_report.config.api_config import status_text
from fx_report.market.pairs import list_pairs
from fx_report.pipeline import run_pipeline


def main() -> int:
    p = argparse.ArgumentParser(
        description="七步 FX 情报流水线：选对→看涨币→信息需求→存语句→影响→赋权→MC→报告"
    )
    p.add_argument("--pair", default="USD/AUD", help=f"One of: {', '.join(list_pairs())}")
    p.add_argument(
        "--bullish",
        default=None,
        metavar="CCY",
        help="看涨货币（须为 pair 的 base 或 quote）。缺省=base，并在 stage_log 写明",
    )
    p.add_argument("--ticker", default=None, help="自定义货币对时的兼容字段")
    p.add_argument("--invert", action="store_true")
    p.add_argument("--sims", type=int, default=100_000)
    p.add_argument("--days", type=int, default=66)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--out", type=str, default="output")
    p.add_argument("--no-news", action="store_true")
    p.add_argument("--keep-templates", action="store_true")
    p.add_argument("--max-news", type=int, default=10)
    p.add_argument("--mode", choices=["hybrid", "llm", "rules"], default="hybrid")
    p.add_argument("--no-fulltext", action="store_true")
    p.add_argument(
        "--ai-research",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="步骤3启用 AI 检索员（白名单投行页+搜索API+LLM抽取）；--no-ai-research 关闭",
    )
    p.add_argument("--api-status", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.api_status:
        print(status_text())
        return 0

    run_pipeline(
        args.pair,
        ticker=args.ticker,
        invert=args.invert,
        sims=args.sims,
        days=args.days,
        seed=args.seed,
        lookback=args.lookback,
        mode=args.mode,  # type: ignore[arg-type]
        max_news=args.max_news,
        keep_templates=args.keep_templates,
        no_news=args.no_news,
        no_fulltext=args.no_fulltext,
        ai_research=args.ai_research,
        out_dir=args.out,
        verbose=not args.quiet,
        bullish_currency=args.bullish,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
