"""Minutes V2 champion/challenger on REAL leak-free data (v2). Args: NAME START END STEP"""
import sys, json, time, pandas as pd
sys.path.insert(0, "/home/claude/nba-platform")
from nba_platform.minutes.engine import MinutesConfig
from nba_platform.minutes.validate import backtest_engine, backtest_baseline
D = "/home/claude/data"
CFG = {
    "CHAMPION_1_1": MinutesConfig(),
    "M_OT": MinutesConfig(ot_gamma=5.0, ot_regulation_training=True),
    "M_CLOSE": MinutesConfig(close_feature=True, prior_k_blowout=5.0),
    "M_DNP": MinutesConfig(dnp_model="competition"),
    "M_HL8": MinutesConfig(half_life_games=8.0),
    "M_HL18": MinutesConfig(half_life_games=18.0),
    "M_SIGMA": MinutesConfig(sigma_dof_correct=True),
    "M3_EMP": MinutesConfig(noise_model="empirical"),
    "M3_SIGMA_ST": MinutesConfig(sigma_dof_correct=True, sigma_dof_scope="starters"),
    "M3_EMP_SIGMA_ST": MinutesConfig(noise_model="empirical", sigma_dof_correct=True, sigma_dof_scope="starters"),
}
name, start, end, step = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
logs = pd.read_pickle(f"{D}/real_logs_v2.pkl"); games = pd.read_pickle(f"{D}/real_games_v2.pkl")
bad = set(games.loc[games["flags"].fillna("") != "", "game_id"])
import os
OFF = int(os.environ.get("DATE_OFFSET", "0"))
dates = sorted(d for d in games["game_date_et"].unique() if start <= d <= end)[OFF::step]
t = time.time()
if name == "BASELINE":
    r = backtest_baseline(logs, dates, name=name, include_dnp=True)
else:
    cfg = CFG[name] if name in CFG else MinutesConfig(**json.loads(sys.argv[5]))
    r = backtest_engine(logs, games, dates, cfg=cfg, n_sims=1000, starters_mode="EXPECTED", name=name)
r.rows = r.rows[~r.rows.game_id.isin(bad)]
r.rows.to_pickle(f"{D}/m2_{name}_{start}_{end}_{step}" + (f"_off{OFF}" if OFF else "") + ".pkl")
print(json.dumps({"name": name, "dates": len(dates), "rows": len(r.rows), "s": round(time.time() - t)}), flush=True)
