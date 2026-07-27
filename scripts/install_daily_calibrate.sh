#!/usr/bin/env bash
# Install LaunchAgent: daily calibrate at 03:00 Asia/Shanghai.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.fxreport.daily-calibrate"
PLIST_SRC="$ROOT/scripts/${LABEL}.plist.template"
PLIST_DST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
UID_NUM="$(id -u)"

mkdir -p "${HOME}/Library/LaunchAgents" "$ROOT/output"
chmod +x "$ROOT/scripts/daily_calibrate.sh" \
  "$ROOT/scripts/sync_calibrated_to_deploy.sh" \
  "$ROOT/scripts/install_daily_calibrate.sh" \
  "$ROOT/scripts/uninstall_daily_calibrate.sh"

if [[ ! -f "$PLIST_SRC" ]]; then
  echo "ERROR: missing template $PLIST_SRC" >&2
  exit 1
fi

# Unload existing if present
if launchctl print "gui/${UID_NUM}/${LABEL}" >/dev/null 2>&1; then
  launchctl bootout "gui/${UID_NUM}/${LABEL}" 2>/dev/null || true
fi
launchctl unload "$PLIST_DST" 2>/dev/null || true

sed "s|__REPO_ROOT__|${ROOT}|g" "$PLIST_SRC" > "$PLIST_DST"
chmod 644 "$PLIST_DST"

# Prefer modern bootstrap; fall back to load
if launchctl bootstrap "gui/${UID_NUM}" "$PLIST_DST" 2>/dev/null; then
  echo "bootstrapped gui/${UID_NUM}/${LABEL}"
else
  launchctl load -w "$PLIST_DST"
  echo "loaded $PLIST_DST"
fi

echo "Installed LaunchAgent:"
echo "  plist: $PLIST_DST"
echo "  schedule: 03:00 daily (TZ=Asia/Shanghai)"
echo "  script: $ROOT/scripts/daily_calibrate.sh"
echo "  logs: $ROOT/output/daily_calib_YYYYMMDD.log + daily_calib_latest.log"
echo
echo "Verify:"
launchctl list | grep -i fxreport || launchctl print "gui/${UID_NUM}/${LABEL}" 2>/dev/null | head -20 || true
echo
echo "Uninstall: $ROOT/scripts/uninstall_daily_calibrate.sh"
echo "Manual dry smoke: DRY_RUN=1 $ROOT/scripts/daily_calibrate.sh"
