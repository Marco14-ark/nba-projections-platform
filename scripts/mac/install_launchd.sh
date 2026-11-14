#!/bin/bash
# Installs the free daily scheduler on macOS (launchd, no cost). Run from the project folder:
#   bash scripts/mac/install_launchd.sh
# Verify afterwards with:  launchctl list | grep nba-platform   and   tail -50 data/logs/daily.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
python3 -m venv "$ROOT/.venv" && "$ROOT/.venv/bin/pip" install -q -e "$ROOT" scikit-learn
PL="$HOME/Library/LaunchAgents/com.nba-platform.daily.plist"
sed "s#REPLACE_WITH_PATH/nba-platform#$ROOT#" "$ROOT/scripts/com.nba-platform.daily.plist" > "$PL"
launchctl unload "$PL" 2>/dev/null || true
launchctl load "$PL"
launchctl kickstart -k "gui/$(id -u)/com.nba-platform.daily"   # run once now
sleep 20; tail -30 "$ROOT/data/logs/daily.log" || echo "no log yet: check /tmp/nba-platform.err"
