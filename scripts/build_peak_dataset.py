#!/usr/bin/env python3
"""Build historical peak samples (Stage 0) and/or calibrate MC params (Stage 1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow `python scripts/build_peak_dataset.py` from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="FX Analyse — peak samples + MC calibration")
    p.add_argument("--pair", default="USD/AUD")
    p.add_argument("--out", default="output")
    p.add_argument("--horizon", type=int, default=66)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--history-days", type=int, default=1500)
    p.add_argument("--step", type=int, default=5, help="Sample every N trading days")
    p.add_argument("--calibrate", action="store_true", help="Also run Stage 1 calibration")
    p.add_argument("--n-sims", type=int, default=2_000)
    p.add_argument("--n-iters", type=int, default=40)
    p.add_argument("--max-rows", type=int, default=80)
    p.add_argument("--loss", choices=["brier", "logloss"], default="brier")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples", default=None, help="Existing peak CSV (skip rebuild)")
    args = p.parse_args()

    samples_path = Path(args.samples) if args.samples else None

    if samples_path is None:
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
            print(f"ERROR building peak samples: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote {path} ({len(df)} rows, source={meta.get('source')})")
        samples_path = path
    else:
        print(f"Using existing samples: {samples_path}")

    if args.calibrate:
        from fx_report.model.calibrate import calibrate_pair

        try:
            out, result = calibrate_pair(
                args.pair,
                samples_path=samples_path,
                out_dir=args.out,
                n_sims=args.n_sims,
                n_iters=args.n_iters,
                seed=args.seed,
                loss=args.loss,
                max_rows=args.max_rows,
                verbose=True,
            )
        except Exception as exc:
            print(f"ERROR calibrating: {exc}", file=sys.stderr)
            return 1
        print(f"Calibrated → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
