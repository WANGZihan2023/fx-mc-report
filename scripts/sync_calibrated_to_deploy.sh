#!/usr/bin/env bash
# Copy Stage-1 calibrated JSON (+ OOS summaries) from output/ into the
# tracked deploy bundle so Railway/Docker can ship them.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ROOT}/output"
DST="${ROOT}/fx_report/data/calibrated"

mkdir -p "${DST}"

n_params=0
n_oos=0
shopt -s nullglob
for f in "${SRC}"/calibrated_params_*.json; do
  cp -f "$f" "${DST}/"
  n_params=$((n_params + 1))
done
for f in "${SRC}"/calib_oos_summary_*.json; do
  cp -f "$f" "${DST}/"
  n_oos=$((n_oos + 1))
done

echo "Synced ${n_params} calibrated_params_*.json and ${n_oos} calib_oos_summary_*.json → ${DST}"
echo "Pairs present:"
for p in USDAUD AUDUSD EURUSD GBPUSD USDJPY USDCNH USDCAD NZDUSD; do
  ok=1
  [[ -f "${DST}/calibrated_params_${p}.json" ]] || ok=0
  [[ -f "${DST}/calib_oos_summary_${p}.json" ]] || ok=0
  if [[ "$ok" -eq 1 ]]; then
    echo "  OK  ${p}"
  else
    echo "  MISS ${p}"
  fi
done
echo "Next: git add fx_report/data/calibrated/ && commit && push (triggers redeploy)."
