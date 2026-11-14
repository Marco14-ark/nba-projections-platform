#!/bin/bash
# launchd entrypoint (every 30 min). Uses the project's free virtualenv; logs to data/logs/daily.log
cd "$(dirname "$0")/.."
mkdir -p data/logs
PY=".venv/bin/python"; [ -x "$PY" ] || PY=python3
NBA_DATA_DIR="$PWD/data" "$PY" scripts/mac/daily_run.py >> data/logs/daily.log 2>&1
