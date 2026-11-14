#!/bin/bash
# Run once on the Mac, from the project root, after downloading nba-data-bootstrap.zip next to it.
#   bash scripts/mac/bootstrap_data.sh ~/Downloads/nba-data-bootstrap.zip
# Unpacks the historical archive (2024-25, 2025-26, 2026-27 to date) into ./data and verifies it.
set -euo pipefail
ZIP="${1:?usage: bootstrap_data.sh <path to nba-data-bootstrap.zip>}"
mkdir -p data
unzip -n "$ZIP" -d data
for f in real_games_v2.pkl real_logs_v2.pkl cdnnba_2024_segments.pkl cdnnba_2025_segments.pkl s2627/nba_2026_27_pbp_box.jsonl.gz; do
  [ -s "data/$f" ] || { echo "MISSING data/$f"; exit 1; }
done
(cd data && shasum -a 256 -c SHA256SUMS)
echo "bootstrap OK -> now run: bash scripts/mac/check_connectivity.sh"
