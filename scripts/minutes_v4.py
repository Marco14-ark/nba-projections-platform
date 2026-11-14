"""Minutes V4 blowout challengers, 3 seasons, leak-free walk-forward. Args: NAME START END STEP [OFFSET]
Uses v3 data (2024-25, 2025-26, 2026-27) + PBP garbage time per game (games['garbage_min']).
Garbage time of the FORECAST game is never an input: training rows only, and the simulated margin
drives the pregame G draw."""
import sys, json, time, os, pandas as pd
sys.path.insert(0, "/home/claude/nba-platform")
from nba_platform.minutes.engine import MinutesConfig
from nba_platform.minutes.validate import backtest_engine
D = "/home/claude/data"
CFG = {
    "CHAMPION_1_2": MinutesConfig(),
    "BLOWOUT_G": MinutesConfig(blowout_feature="garbage"),
    "BLOWOUT_GD": MinutesConfig(blowout_feature="garbage", dnp_garbage=True),
    "BLOWOUT_BXD": MinutesConfig(dnp_garbage=True),
}
name, start, end, step = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
off = int(sys.argv[5]) if len(sys.argv) > 5 else 0
logs = pd.read_pickle(f"{D}/real_logs_v3.pkl")
games = pd.read_pickle(f"{D}/real_games_v3.pkl").merge(pd.read_pickle(f"{D}/garbage_by_game.pkl")[["game_id", "garbage_min"]],
                                                        on="game_id", how="left")
bad = set(games.loc[games["flags"].fillna("") != "", "game_id"])
dates = sorted(d for d in games["game_date_et"].unique() if start <= d <= end)[off::step]
t = time.time()
r = backtest_engine(logs, games, dates, cfg=CFG[name], n_sims=1000, starters_mode="EXPECTED", name=name)
r.rows = r.rows[~r.rows.game_id.isin(bad)]
r.rows.to_pickle(f"{D}/m4_{name}_{start}_{end}_{step}_{off}.pkl")
print(json.dumps({"name": name, "dates": len(dates), "rows": len(r.rows), "s": round(time.time() - t)}), flush=True)
