"""Walk-forward backtest of NBA_MINUTES_ENGINE_V1 on REAL 2025-26 games (reconstructed PBP).

Usage: python scripts/backtest_real.py START END MODE [STEP]
Each test date is predicted with a model fitted only on games strictly before it.
"""
import json
import sys
import time

import pandas as pd

sys.path.insert(0, "/home/claude/nba-platform")
from nba_platform.minutes.validate import backtest_baseline, backtest_engine  # noqa: E402

D = "/home/claude/data"
start, end, mode = sys.argv[1], sys.argv[2], sys.argv[3]
step = int(sys.argv[4]) if len(sys.argv) > 4 else 1
logs = pd.read_pickle(f"{D}/real_logs.pkl")
games = pd.read_pickle(f"{D}/real_games.pkl")
bad = set(games.loc[games["flags"].fillna("").str.contains("MINUTES_SUM|STARTERS"), "game_id"])
dates = sorted(d for d in games["game_date_et"].unique() if start <= d <= end)[::step]
t = time.time()
eng = backtest_engine(logs, games, dates, n_sims=1000, starters_mode=mode, name=f"ENGINE_{mode}")
eng.rows = eng.rows[~eng.rows["game_id"].isin(bad)]
b1 = backtest_baseline(logs, dates, name="LAST10_PLAYED")
b2 = backtest_baseline(logs, dates, name="LAST10_AVAILABLE", include_dnp=True)
for r in (b1, b2):
    r.rows = r.rows[~r.rows["game_id"].isin(bad)]
tag = f"{mode}_{start}_{end}_{step}"
for r in (eng, b1, b2):
    r.rows.to_pickle(f"{D}/bt_{tag}_{r.name}.pkl")
print(json.dumps({"dates": len(dates), "seconds": round(time.time() - t), "excluded_games": len(bad)}))
