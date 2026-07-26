#!/usr/bin/env bash
# Overnight full MC calibration: build-peaks → calibrate → optional AUD/USD.
# Usage: ./scripts/overnight_calibrate.sh [PAIR...]
# Defaults: USD/AUD then AUD/USD (bullish inverse form).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
LOG="${LOG:-$ROOT/output/overnight_calib_${DATE_TAG}.log}"
PIDFILE="${PIDFILE:-$ROOT/output/overnight_calib.pid}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

# Full (non-smoke) knobs — higher than CLI defaults (2k/40/80)
N_SIMS="${N_SIMS:-8000}"
N_ITERS="${N_ITERS:-80}"
MAX_ROWS="${MAX_ROWS:-250}"
HISTORY_DAYS="${HISTORY_DAYS:-2000}"
STEP="${STEP:-5}"
HORIZON="${HORIZON:-66}"
LOOKBACK="${LOOKBACK:-60}"

PAIRS=("$@")
if [[ ${#PAIRS[@]} -eq 0 ]]; then
  PAIRS=("USD/AUD" "AUD/USD")
fi

echo "$$" > "$PIDFILE"
exec >>"$LOG" 2>&1
echo "======== overnight_calibrate start $(date -Iseconds) ========"
echo "pid=$$ python=$PYTHON pairs=${PAIRS[*]}"
echo "n_sims=$N_SIMS n_iters=$N_ITERS max_rows=$MAX_ROWS history_days=$HISTORY_DAYS step=$STEP"

for PAIR in "${PAIRS[@]}"; do
  SAFE="${PAIR//\//}"
  echo "---- [$(date -Iseconds)] build-peaks $PAIR ----"
  "$PYTHON" run_cli.py build-peaks \
    --pair "$PAIR" \
    --out output \
    --horizon "$HORIZON" \
    --lookback "$LOOKBACK" \
    --history-days "$HISTORY_DAYS" \
    --step "$STEP"

  echo "---- [$(date -Iseconds)] calibrate $PAIR ----"
  "$PYTHON" run_cli.py calibrate \
    --pair "$PAIR" \
    --out output \
    --n-sims "$N_SIMS" \
    --n-iters "$N_ITERS" \
    --max-rows "$MAX_ROWS" \
    --loss brier \
    --seed 42

  echo "---- [$(date -Iseconds)] done $PAIR → output/calibrated_params_${SAFE}.json ----"
  ls -la "output/calibrated_params_${SAFE}.json" "output/calib_oos_summary_${SAFE}.json" 2>/dev/null || true
done

echo "======== overnight_calibrate finished $(date -Iseconds) ========"
echo "DONE"
