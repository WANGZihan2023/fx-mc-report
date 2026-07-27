#!/usr/bin/env bash
# Remove daily calibrate LaunchAgent.
set -euo pipefail

LABEL="com.fxreport.daily-calibrate"
PLIST_DST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
UID_NUM="$(id -u)"

if launchctl print "gui/${UID_NUM}/${LABEL}" >/dev/null 2>&1; then
  launchctl bootout "gui/${UID_NUM}/${LABEL}" 2>/dev/null || true
fi
launchctl unload "$PLIST_DST" 2>/dev/null || true
rm -f "$PLIST_DST"

echo "Removed LaunchAgent $LABEL"
echo "plist gone: $PLIST_DST"
launchctl list | grep -i fxreport || echo "(no com.fxreport jobs in launchctl list)"
