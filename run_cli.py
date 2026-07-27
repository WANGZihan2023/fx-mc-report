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
        help="步骤3启用 AI 检索员（白名单投行页+搜索API+LLM抽取）；--no-ai-research 关闭",
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
    bt_p.add_argument("--quiet", action="store_true")
    bt_p.set_defaults(func=_cmd_backtest)

    # Backward compatible: no subcommand → treat as `run`
    # Parse known first; if first token isn't a subcommand, inject `run`.
    import sys

    argv = list(sys.argv[1:])
    if not argv or argv[0] not in {
        "run",
        "build-peaks",
        "calibrate",
        "backtest",
        "-h",
        "--help",
    }:
        # allow legacy flags like --api-status without subcommand
        argv = ["run", *argv]
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
