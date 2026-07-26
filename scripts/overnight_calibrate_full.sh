#!/usr/bin/env bash
# Full multi-pair overnight calibration (stronger data, not endless iters).
# Usage:
#   ./scripts/overnight_calibrate_full.sh [PAIR...]
#   MODE=full ./scripts/overnight_calibrate.sh   # delegates here
#
# Defaults: major product pairs (canonical forms + AUD flip like overnight).
# Logs: output/overnight_calib_full_YYYYMMDD.log
# PID:  output/overnight_calib_full.pid
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
LOG="${LOG:-$ROOT/output/overnight_calib_full_${DATE_TAG}.log}"
PIDFILE="${PIDFILE:-$ROOT/output/overnight_calib_full.pid}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

# Stronger data knobs (vs overnight smoke-ish defaults of 2000/5)
N_SIMS="${N_SIMS:-7000}"
N_ITERS="${N_ITERS:-70}"
MAX_ROWS="${MAX_ROWS:-350}"
HISTORY_DAYS="${HISTORY_DAYS:-3500}"
STEP="${STEP:-3}"
HORIZON="${HORIZON:-66}"
LOOKBACK="${LOOKBACK:-60}"

# Wait for a prior overnight job if WAIT_PID is set (or stale overnight_calib.pid still alive)
WAIT_PID="${WAIT_PID:-}"
if [[ -z "$WAIT_PID" && -f "$ROOT/output/overnight_calib.pid" ]]; then
  CAND="$(tr -d '[:space:]' < "$ROOT/output/overnight_calib.pid" || true)"
  if [[ -n "$CAND" ]] && kill -0 "$CAND" 2>/dev/null; then
    WAIT_PID="$CAND"
  fi
fi

PAIRS=("$@")
if [[ ${#PAIRS[@]} -eq 0 ]]; then
  # Product catalog majors (~8): AUD both orientations (overnight convention);
  # other pairs in UI-canonical form only.
  PAIRS=(
    "USD/AUD" "AUD/USD"
    "EUR/USD" "GBP/USD"
    "USD/JPY" "USD/CNH"
    "USD/CAD" "NZD/USD"
  )
fi

mkdir -p "$ROOT/output"
echo "$$" > "$PIDFILE"
exec >>"$LOG" 2>&1

echo "======== overnight_calibrate_full start $(date -Iseconds) ========"
echo "pid=$$ python=$PYTHON"
echo "pairs=${PAIRS[*]}"
echo "n_sims=$N_SIMS n_iters=$N_ITERS max_rows=$MAX_ROWS history_days=$HISTORY_DAYS step=$STEP horizon=$HORIZON lookback=$LOOKBACK"

if [[ -n "$WAIT_PID" ]]; then
  echo "Waiting for prior job pid=$WAIT_PID …"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
  echo "Prior job $WAIT_PID finished at $(date -Iseconds)"
fi

FAILED=0
for PAIR in "${PAIRS[@]}"; do
  SAFE="${PAIR//\//}"
  echo "==== [$(date -Iseconds)] BEGIN $PAIR ===="
  set +e
  echo "---- [$(date -Iseconds)] build-peaks $PAIR ----"
  "$PYTHON" run_cli.py build-peaks \
    --pair "$PAIR" \
    --out output \
    --horizon "$HORIZON" \
    --lookback "$LOOKBACK" \
    --history-days "$HISTORY_DAYS" \
    --step "$STEP"
  bp_rc=$?
  if [[ $bp_rc -ne 0 ]]; then
    echo "ERROR: build-peaks failed for $PAIR (rc=$bp_rc)"
    FAILED=$((FAILED + 1))
    continue
  fi

  echo "---- [$(date -Iseconds)] calibrate $PAIR ----"
  "$PYTHON" run_cli.py calibrate \
    --pair "$PAIR" \
    --out output \
    --n-sims "$N_SIMS" \
    --n-iters "$N_ITERS" \
    --max-rows "$MAX_ROWS" \
    --loss brier \
    --seed 42
  cal_rc=$?
  set -e
  if [[ $cal_rc -ne 0 ]]; then
    echo "ERROR: calibrate failed for $PAIR (rc=$cal_rc)"
    FAILED=$((FAILED + 1))
    continue
  fi

  echo "---- [$(date -Iseconds)] artifacts $PAIR ----"
  ls -la "output/calibrated_params_${SAFE}.json" "output/calib_oos_summary_${SAFE}.json" "output/peak_samples_${SAFE}.csv" 2>/dev/null || true
  if [[ -f "output/calib_oos_summary_${SAFE}.json" ]]; then
    echo "---- OOS summary $PAIR ----"
    "$PYTHON" -c "
import json
from pathlib import Path
p = Path('output/calib_oos_summary_${SAFE}.json')
d = json.loads(p.read_text(encoding='utf-8'))
train = d.get('train') or {}
hold = d.get('holdout') or {}
print(f\"  train n={train.get('n')} brier={train.get('brier')} hit={train.get('hit_rate')}\")
print(f\"  holdout n={hold.get('n')} brier={hold.get('brier')} hit={hold.get('hit_rate')}\")
print(f\"  calibrated_loss={d.get('calibrated_loss')} baseline_loss={d.get('baseline_loss')} n_sims={d.get('n_sims')}\")
"
  fi
  echo "==== [$(date -Iseconds)] END $PAIR ===="
done

echo "======== overnight_calibrate_full finished $(date -Iseconds) failed_pairs=$FAILED ========"
if [[ $FAILED -eq 0 ]]; then
  echo "DONE"
else
  echo "DONE_WITH_ERRORS count=$FAILED"
fi
exit 0
