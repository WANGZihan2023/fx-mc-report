#!/usr/bin/env bash
# Daily Stage-1 calibration (moderate knobs) → sync deploy JSON → optional git push.
#
# Usage:
#   ./scripts/daily_calibrate.sh [PAIR...]
#   MODE=primary ./scripts/daily_calibrate.sh   # USD/AUD + AUD/USD only
#   MODE=product ./scripts/daily_calibrate.sh   # 8 product pairs (default)
#   DRY_RUN=1 ./scripts/daily_calibrate.sh      # wiring smoke (no MC)
#   AUTO_GIT=0 ./scripts/daily_calibrate.sh     # skip commit/push
#
# Env knobs (defaults balanced for ~nightly window):
#   N_SIMS=4000 N_ITERS=40 MAX_ROWS=200 HISTORY_DAYS=2000 STEP=5
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p "$ROOT/output"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
LOG_DATED="$ROOT/output/daily_calib_${DATE_TAG}.log"
LOG_LATEST="$ROOT/output/daily_calib_latest.log"
STATUS="$ROOT/output/daily_calib_status.txt"
PIDFILE="$ROOT/output/daily_calib.pid"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

MODE="${MODE:-product}"
DRY_RUN="${DRY_RUN:-0}"
AUTO_GIT="${AUTO_GIT:-1}"

N_SIMS="${N_SIMS:-4000}"
N_ITERS="${N_ITERS:-40}"
MAX_ROWS="${MAX_ROWS:-200}"
HISTORY_DAYS="${HISTORY_DAYS:-2000}"
STEP="${STEP:-5}"
HORIZON="${HORIZON:-66}"
LOOKBACK="${LOOKBACK:-60}"

PAIRS=("$@")
if [[ ${#PAIRS[@]} -eq 0 ]]; then
  if [[ "$MODE" == "primary" ]]; then
    PAIRS=("USD/AUD" "AUD/USD")
  else
    # Same product catalog as overnight_calibrate_full.sh
    PAIRS=(
      "USD/AUD" "AUD/USD"
      "EUR/USD" "GBP/USD"
      "USD/JPY" "USD/CNH"
      "USD/CAD" "NZD/USD"
    )
  fi
fi

# Dated log + pointer for latest (symlink preferred)
: >"$LOG_DATED"
ln -sfn "$(basename "$LOG_DATED")" "$LOG_LATEST" 2>/dev/null || cp -f "$LOG_DATED" "$LOG_LATEST"
exec >>"$LOG_DATED" 2>&1

echo "$$" > "$PIDFILE"
write_status() {
  local state="$1"
  shift || true
  {
    echo "state=$state"
    echo "time=$(date -Iseconds)"
    echo "mode=$MODE"
    echo "pairs=${PAIRS[*]}"
    echo "n_sims=$N_SIMS n_iters=$N_ITERS max_rows=$MAX_ROWS history_days=$HISTORY_DAYS step=$STEP"
    echo "log=$LOG_DATED"
    for line in "$@"; do
      echo "$line"
    done
  } >"$STATUS"
}

cleanup() {
  rm -f "$PIDFILE" 2>/dev/null || true
}
trap cleanup EXIT

write_status "running"
echo "======== daily_calibrate start $(date -Iseconds) ========"
echo "pid=$$ python=$PYTHON mode=$MODE dry_run=$DRY_RUN auto_git=$AUTO_GIT"
echo "pairs=${PAIRS[*]}"
echo "n_sims=$N_SIMS n_iters=$N_ITERS max_rows=$MAX_ROWS history_days=$HISTORY_DAYS step=$STEP"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1 — skip build-peaks/calibrate; check sync + status only"
  if [[ -x "$ROOT/scripts/sync_calibrated_to_deploy.sh" ]]; then
    "$ROOT/scripts/sync_calibrated_to_deploy.sh" || true
  fi
  write_status "dry_run_ok" "note=skipped MC"
  echo "======== daily_calibrate dry_run finished $(date -Iseconds) ========"
  echo "DONE_DRY_RUN"
  exit 0
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
  ls -la "output/calibrated_params_${SAFE}.json" "output/calib_oos_summary_${SAFE}.json" 2>/dev/null || true
  echo "==== [$(date -Iseconds)] END $PAIR ===="
done

if [[ $FAILED -ne 0 ]]; then
  write_status "failed" "failed_pairs=$FAILED"
  echo "======== daily_calibrate finished $(date -Iseconds) failed_pairs=$FAILED ========"
  echo "DONE_WITH_ERRORS count=$FAILED"
  # Still try sync for any pairs that succeeded
fi

echo "---- sync calibrated → deploy ----"
set +e
"$ROOT/scripts/sync_calibrated_to_deploy.sh"
sync_rc=$?
set -e
if [[ $sync_rc -ne 0 ]]; then
  echo "ERROR: sync_calibrated_to_deploy.sh failed (rc=$sync_rc)"
  write_status "sync_failed" "failed_pairs=$FAILED" "sync_rc=$sync_rc"
  exit "$sync_rc"
fi

GIT_NOTE="git=skipped"
if [[ "$AUTO_GIT" == "1" ]]; then
  echo "---- optional git commit + push ----"
  set +e
  git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1
  in_git=$?
  if [[ $in_git -ne 0 ]]; then
    echo "WARN: not a git repo; skip commit/push"
    GIT_NOTE="git=not_a_repo"
  else
    git -C "$ROOT" add fx_report/data/calibrated/ 2>&1
    if git -C "$ROOT" diff --cached --quiet; then
      echo "No calibrated JSON changes to commit"
      GIT_NOTE="git=no_changes"
    else
      git -C "$ROOT" commit -m "chore: daily calibrated params" 2>&1
      commit_rc=$?
      if [[ $commit_rc -ne 0 ]]; then
        echo "ERROR: git commit failed (rc=$commit_rc) — leaving staged files"
        GIT_NOTE="git=commit_failed rc=$commit_rc"
      else
        git -C "$ROOT" push origin HEAD 2>&1
        push_rc=$?
        if [[ $push_rc -ne 0 ]]; then
          echo "ERROR: git push failed (rc=$push_rc) — commit exists locally; push manually later"
          GIT_NOTE="git=push_failed rc=$push_rc"
        else
          echo "Pushed chore: daily calibrated params"
          GIT_NOTE="git=pushed"
        fi
      fi
    fi
  fi
  set -e
fi

if [[ $FAILED -eq 0 ]]; then
  write_status "ok" "$GIT_NOTE"
  echo "======== daily_calibrate finished $(date -Iseconds) ========"
  echo "DONE"
  exit 0
else
  write_status "partial" "failed_pairs=$FAILED" "$GIT_NOTE"
  echo "======== daily_calibrate finished $(date -Iseconds) failed_pairs=$FAILED ========"
  echo "DONE_WITH_ERRORS count=$FAILED"
  exit 0
fi
