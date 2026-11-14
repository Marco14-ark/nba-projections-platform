"""Champion/challenger for the minutes engine on REAL 2025-26 data (EXPECTED starters)."""
import sys, json, pandas as pd
sys.path.insert(0, "/home/claude/nba-platform")
from nba_platform.minutes.engine import MinutesConfig
from nba_platform.minutes.validate import backtest_engine
D = "/home/claude/data"
name, start, end, step = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cfgs = {"CHAMPION_V1_0": MinutesConfig(),
        "CHALLENGER_A": MinutesConfig(struct_weighting="minutes_room", absent_scope="recent"),
        "CHALLENGER_B": MinutesConfig(struct_weighting="minutes_room", absent_scope="recent", dnp_context_buckets=False)}
logs = pd.read_pickle(f"{D}/real_logs.pkl"); games = pd.read_pickle(f"{D}/real_games.pkl")
bad = set(games.loc[games["flags"].fillna("").str.contains("MINUTES_SUM|STARTERS"), "game_id"])
dates = sorted(d for d in games["game_date_et"].unique() if start <= d <= end)[::step]
r = backtest_engine(logs, games, dates, cfg=cfgs[name], n_sims=1000, starters_mode="EXPECTED", name=name)
r.rows = r.rows[~r.rows.game_id.isin(bad)]
r.rows.to_pickle(f"{D}/cc_{name}_{start}_{end}_{step}.pkl")
print(json.dumps({"name": name, "dates": len(dates)}))
