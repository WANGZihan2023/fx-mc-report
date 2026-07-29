#!/usr/bin/env python3
"""CLI：七步流水线 + Stage0/1（峰值样本 / 校准）。"""

from __future__ import annotations

import argparse
from pathlib import Path

from fx_report.config.api_config import status_text
from fx_report.market.pairs import list_pairs
from fx_report.pipeline import run_pipeline


def _add_pipeline_args(p: argparse.ArgumentParser) -> None:
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
    p.add_argument(
        "--peak-engine",
        choices=["path_max", "brownian_bridge"],
        default="path_max",
        help="Peak estimator: path_max (GBM+jumps) or brownian_bridge (continuous GBM, no jumps)",
    )
    p.add_argument(
        "--variance-reduction",
        choices=["none", "antithetic"],
        default="none",
        help="Variance reduction for MC maxima: none (current) or antithetic",
    )
    p.add_argument(
        "--jump-model",
        choices=["merton", "none"],
        default="merton",
        help="Jump model on path_max: merton (Cont–Tankov/Merton compound Poisson) or none",
    )
    p.add_argument(
        "--jump-compensate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply Merton compensator −λ(E[e^J]−1)Δt to daily log-drift (default: off)",
    )
    p.add_argument("--out", type=str, default="output")
    p.add_argument("--no-news", action="store_true")
    p.add_argument("--keep-templates", action="store_true")
    p.add_argument(
        "--template-policy",
        choices=["off", "prior_only", "fallback_warn"],
        default="off",
        help="新闻证据为空时：off=不用模板；prior_only=标记降权模板；fallback_warn=调试告警回退",
    )
    p.add_argument("--max-news", type=int, default=10)
    p.add_argument("--mode", choices=["hybrid", "llm", "rules"], default="hybrid")
    p.add_argument("--no-fulltext", action="store_true")
    p.add_argument(
        "--ai-research",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "步骤3启用 AI 检索员（LLM脑 + Tavily/Brave/NewsAPI/GoogleNews手）；"
            "--no-ai-research 关闭。"
            "注意：replay-backtest / 带 as_of 的历史回放默认省钱模式，会强制关闭 AI+Tavily"
            "（除非另加 --allow-historical-ai）。"
        ),
    )
    p.add_argument(
        "--calibrated-params",
        default=None,
        help="Stage-1 JSON（如 output/calibrated_params_USDAUD.json）",
    )
    p.add_argument(
        "--use-label-learned-strength",
        action="store_true",
        help="Stage 3：若 label_audit 标注≥N 条，用类别强度倍率缩放证据",
    )
    p.add_argument(
        "--auto-skip-uncertain",
        action="store_true",
        help="不确定证据不弹窗：记入日志后保留模型方向继续（CLI 默认行为；显式声明）",
    )
    p.add_argument(
        "--max-uncertain",
        type=int,
        default=5,
        help="赋权前最多列出几条不确定证据（默认 5）",
    )
    p.add_argument("--api-status", action="store_true")
    p.add_argument("--quiet", action="store_true")


def _cmd_run(args: argparse.Namespace) -> int:
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
        peak_engine=args.peak_engine,
        variance_reduction=args.variance_reduction,
        jump_model=args.jump_model,
        jump_compensate=bool(args.jump_compensate),
        mode=args.mode,  # type: ignore[arg-type]
        max_news=args.max_news,
        keep_templates=args.keep_templates,
        template_policy=args.template_policy,
        no_news=args.no_news,
        no_fulltext=args.no_fulltext,
        ai_research=args.ai_research,
        out_dir=args.out,
        verbose=not args.quiet,
        bullish_currency=args.bullish,
        calibrated_params_path=args.calibrated_params,
        use_label_learned_strength=bool(args.use_label_learned_strength),
        human_review_mode="auto_skip",
        max_uncertain=int(getattr(args, "max_uncertain", 5) or 5),
    )
    return 0


def _cmd_build_peaks(args: argparse.Namespace) -> int:
    from fx_report.model.history_peaks import export_peak_samples

    try:
        path, df, meta = export_peak_samples(
            args.pair,
            out_dir=args.out,
            horizon_days=args.horizon,
            vol_lookback=args.lookback,
            history_days=args.history_days,
            step=args.step,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Wrote {path} ({len(df)} rows, source={meta.get('source')})")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from fx_report.model.calibrate import calibrate_pair

    try:
        out, result = calibrate_pair(
            args.pair,
            samples_path=args.samples,
            out_dir=args.out,
            n_sims=args.n_sims,
            n_iters=args.n_iters,
            seed=args.seed,
            loss=args.loss,
            max_rows=args.max_rows,
            verbose=not args.quiet,
            variance_reduction=args.variance_reduction,
            peak_engine=args.peak_engine,
            jump_model=args.jump_model,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Done → {out} ({result.loss_name} {result.baseline_loss:.4f} → {result.loss:.4f})")
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    from fx_report.model.backtest import run_backtest

    try:
        result = run_backtest(
            args.pair,
            samples_path=args.samples,
            calibrated_params_path=args.calibrated_params,
            out_dir=args.out,
            n_sims=args.n_sims,
            max_rows=args.max_rows,
            seed=args.seed,
            peak_engine=args.peak_engine,
            variance_reduction=args.variance_reduction,
            jump_model=args.jump_model,
            verbose=not args.quiet,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(
        f"Done → hit_rate={result.hit_rate_argmax:.1%}  "
        f"brier={result.mean_brier:.4f}  logloss={result.mean_logloss:.4f}  "
        f"n={result.n_rows}"
    )
    return 0


def _cmd_replay_backtest(args: argparse.Namespace) -> int:
    from fx_report.model.replay_backtest import run_replay_backtest

    allow_hist_ai = bool(getattr(args, "allow_historical_ai", False))
    # Cheap by default: ignore --ai-research unless expensive override is set.
    replay_ai = bool(args.ai_research) if allow_hist_ai else False
    try:
        result = run_replay_backtest(
            args.pair,
            bullish_currency=args.bullish,
            start_date=args.start,
            end_date=args.end,
            step_days=args.step,
            out_dir=args.out,
            sims=int(getattr(args, "n_sims", None) or args.sims),
            days=args.days,
            seed=args.seed,
            lookback=args.lookback,
            peak_engine=args.peak_engine,
            variance_reduction=args.variance_reduction,
            jump_model=args.jump_model,
            jump_compensate=bool(args.jump_compensate),
            mode=args.mode,
            max_news=args.max_news,
            keep_templates=args.keep_templates,
            template_policy=args.template_policy,
            no_news=args.no_news,
            no_fulltext=args.no_fulltext,
            ai_research=replay_ai,
            allow_historical_ai=allow_hist_ai,
            calibrated_params_path=args.calibrated_params,
            use_label_learned_strength=bool(args.use_label_learned_strength),
            max_dates=args.max_dates,
            verbose=not args.quiet,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(
        f"Done → hit_rate={result.summary.get('argmax_hit_rate', 0):.1%}  "
        f"brier={result.summary.get('mean_brier', float('nan')):.4f}  "
        f"skill_brier={result.summary.get('mean_skill_brier', float('nan')):.4f}  "
        f"n={result.n_rows}"
    )
    print(f"CSV  → {result.csv_path}")
    print(f"JSON → {result.json_path}")
    return 0


def _cmd_replay_summary(args: argparse.Namespace) -> int:
    from fx_report.model.replay_summary import replay_summary_dataframe

    try:
        df = replay_summary_dataframe(args.out)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    if df.empty:
        print("No replay_backtest_*.json found.")
        return 0
    show = df[
        [
            c
            for c in (
                "pair",
                "window",
                "n_rows",
                "argmax_hit_rate",
                "mean_brier",
                "mean_skill_brier",
                "evidence_mean",
                "evidence_max",
                "date_filtered_count",
                "limited_count",
                "historical_news_working",
            )
            if c in df.columns
        ]
    ].copy()
    for col in ("argmax_hit_rate",):
        if col in show.columns:
            show[col] = show[col].map(lambda x: f"{100 * float(x):.1f}%" if x == x else "—")
    for col in ("mean_brier", "mean_skill_brier", "evidence_mean"):
        if col in show.columns:
            show[col] = show[col].map(lambda x: f"{float(x):.4f}" if x == x else "—")
    print(show.to_string(index=False))
    return 0


def _cmd_replay_engine_compare(args: argparse.Namespace) -> int:
    from fx_report.model.replay_engine_compare import print_chinese_summary, run_engine_compare

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FX Analyse：七步流水线 / 峰值样本 / MC 校准 / 历史回测"
    )
    sub = parser.add_subparsers(dest="cmd")

    # default / run
    run_p = sub.add_parser("run", help="跑完整七步流水线（默认）")
    _add_pipeline_args(run_p)
    run_p.set_defaults(func=_cmd_run)

    peaks_p = sub.add_parser("build-peaks", help="Stage 0：历史峰值样本 CSV")
    peaks_p.add_argument("--pair", default="USD/AUD")
    peaks_p.add_argument("--out", default="output")
    peaks_p.add_argument("--horizon", type=int, default=66)
    peaks_p.add_argument("--lookback", type=int, default=60)
    peaks_p.add_argument("--history-days", type=int, default=1500)
    peaks_p.add_argument("--step", type=int, default=5)
    peaks_p.set_defaults(func=_cmd_build_peaks)

    cal_p = sub.add_parser("calibrate", help="Stage 1：校准 MC 参数（S=0）")
    cal_p.add_argument("--pair", default="USD/AUD")
    cal_p.add_argument("--out", default="output")
    cal_p.add_argument("--samples", default=None, help="peak_samples CSV；缺省读 output/")
    cal_p.add_argument("--n-sims", type=int, default=2_000)
    cal_p.add_argument("--n-iters", type=int, default=40)
    cal_p.add_argument("--max-rows", type=int, default=80)
    cal_p.add_argument("--loss", choices=["brier", "logloss"], default="brier")
    cal_p.add_argument(
        "--variance-reduction",
        choices=["none", "antithetic"],
        default="none",
        help="Variance reduction for MC maxima: none or antithetic",
    )
    cal_p.add_argument(
        "--peak-engine",
        choices=["path_max", "brownian_bridge"],
        default="path_max",
        help="Peak estimator used during calibration and saved into params",
    )
    cal_p.add_argument(
        "--jump-model",
        choices=["merton", "none"],
        default="merton",
        help="Jump model (path_max): merton or none; saved into params",
    )
    cal_p.add_argument("--seed", type=int, default=42)
    cal_p.add_argument("--quiet", action="store_true")
    cal_p.set_defaults(func=_cmd_calibrate)

    bt_p = sub.add_parser("backtest", help="历史回测：argmax hit / Brier / log-loss 表")
    bt_p.add_argument("--pair", default="USD/AUD")
    bt_p.add_argument("--out", default="output")
    bt_p.add_argument("--samples", default=None, help="peak_samples CSV；缺省读 output/")
    bt_p.add_argument(
        "--calibrated-params",
        default=None,
        help="校准 JSON；缺省若存在则自动用 output/calibrated_params_{PAIR}.json",
    )
    bt_p.add_argument("--n-sims", type=int, default=2_000)
    bt_p.add_argument("--max-rows", type=int, default=None, help="子采样行数（默认全量）")
    bt_p.add_argument(
        "--peak-engine",
        choices=["path_max", "brownian_bridge"],
        default=None,
        help="缺省跟校准参数 / 默认 path_max",
    )
    bt_p.add_argument("--seed", type=int, default=42)
    bt_p.add_argument(
        "--variance-reduction",
        choices=["none", "antithetic"],
        default="none",
        help="Variance reduction for MC maxima: none or antithetic",
    )
    bt_p.add_argument(
        "--jump-model",
        choices=["merton", "none"],
        default=None,
        help="Override jump model (default: from calibrated params / merton)",
    )
    bt_p.add_argument("--quiet", action="store_true")
    bt_p.set_defaults(func=_cmd_backtest)

    replay_p = sub.add_parser(
        "replay-backtest",
        help=(
            "历史时点冻结回放：全流水线预测 vs 后验实现"
            "（默认省钱：关闭 AI/Tavily；证据靠 GDELT+缓存）"
        ),
    )
    _add_pipeline_args(replay_p)
    replay_p.add_argument("--start", required=True, help="回放起点，如 2024-01-01")
    replay_p.add_argument("--end", required=True, help="回放终点，如 2024-06-30")
    replay_p.add_argument("--step", type=int, default=7, help="按多少个自然日抽一个 as_of")
    replay_p.add_argument("--n-sims", type=int, default=2_000)
    replay_p.add_argument("--max-dates", type=int, default=None, help="最多跑几个 as_of（UI/烟测可限速）")
    replay_p.add_argument(
        "--allow-historical-ai",
        action="store_true",
        help=(
            "昂贵覆盖：允许历史回放启用 AI 检索员/Tavily"
            "（默认关闭；可能引入非历史网页信息）"
        ),
    )
    replay_p.set_defaults(func=_cmd_replay_backtest)

    replay_sum_p = sub.add_parser("replay-summary", help="汇总 output/ 下历史冻结回放结果")
    replay_sum_p.add_argument("--out", default="output")
    replay_sum_p.set_defaults(func=_cmd_replay_summary)

    cmp_p = sub.add_parser(
        "replay-engine-compare",
        help="扫描有历史新闻证据的 as_of，对比引擎 A vs C（小样本）",
    )
    cmp_p.add_argument("--pair", default="USD/AUD")
    cmp_p.add_argument("--bullish", default=None)
    cmp_p.add_argument("--start", default=None, help="扫描起点；默认 today-25")
    cmp_p.add_argument("--end", default=None, help="扫描终点；默认 today-(days+1)")
    cmp_p.add_argument("--step", type=int, default=3)
    cmp_p.add_argument(
        "--dates",
        default=None,
        help="跳过新闻扫描，逗号分隔 as_of，如 2026-07-10,2026-07-15",
    )
    cmp_p.add_argument("--max-dates", type=int, default=3)
    cmp_p.add_argument("--sims", type=int, default=800)
    cmp_p.add_argument("--days", type=int, default=20)
    cmp_p.add_argument("--seed", type=int, default=42)
    cmp_p.add_argument("--lookback", type=int, default=14)
    cmp_p.add_argument("--max-news", type=int, default=10)
    cmp_p.add_argument("--mode", choices=["hybrid", "llm", "rules"], default="rules")
    cmp_p.add_argument("--out", default="output/engine_compare")
    cmp_p.add_argument("--calibrated-params", default=None)
    cmp_p.add_argument("--quiet", action="store_true")
    cmp_p.set_defaults(func=_cmd_replay_engine_compare)

    # Backward compatible: no subcommand → treat as `run`
    # Parse known first; if first token isn't a subcommand, inject `run`.
    import sys

    argv = list(sys.argv[1:])
    if not argv or argv[0] not in {
        "run",
        "build-peaks",
        "calibrate",
        "backtest",
        "replay-backtest",
        "replay-summary",
        "replay-engine-compare",
        "-h",
        "--help",
    }:
        # allow legacy flags like --api-status without subcommand
        argv = ["run", *argv]
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
