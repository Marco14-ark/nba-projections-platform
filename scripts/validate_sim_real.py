"""Walk-forward validation of props + game markets on REAL 2025-26 games. Args: START END STEP"""
import sys, time, pandas as pd
sys.path.insert(0, "/home/claude/nba-platform")
from nba_platform.sim.validate import walk_forward
D = "/home/claude/data"
start, end, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
logs = pd.read_pickle(f"{D}/real_logs.pkl"); games = pd.read_pickle(f"{D}/real_games.pkl")
bad = set(games.loc[games["flags"].fillna("").str.contains("MINUTES_SUM|STARTERS"), "game_id"])
dates = sorted(d for d in games["game_date_et"].unique() if start <= d <= end)[::step]
t = time.time()
anchor = sys.argv[4] if len(sys.argv) > 4 else "bottom_up"
P, G = walk_forward(logs, games, dates, n_sims=4000, progress=True, anchor=anchor)
P = P[~P.game_id.isin(bad)]; G = G[~G.game_id.isin(bad)]
P.to_pickle(f"{D}/val_props_{start}_{end}_{step}_{anchor}_m11.pkl"); G.to_pickle(f"{D}/val_games_{start}_{end}_{step}_{anchor}_m11.pkl")
print("done", len(dates), "dates", len(G), "games", round(time.time() - t), "s", flush=True)
