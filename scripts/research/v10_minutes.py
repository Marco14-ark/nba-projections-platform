"""V10 minutes challengers, leak-free walk-forward (engine fitted only on games before each date).
Availability = players actually active (as in all stack backtests); starters = EXPECTED (not confirmed).
Args: NAME START END STEP  -> docs/validation/v10/min_NAME_START_STEP.pkl"""
import json, os, sys, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from nba_platform.lineups.resolve import expected_starters
from nba_platform.minutes.engine import MinutesConfig, MinutesEngine
from nba_platform.minutes.validate import _day_games
from nba_platform.sim.validate import crps_samples

CFG = {"CHAMPION": {}, "MIN_V2_ROLE": {"start_mix": True}, "MIN_V2_RETURN": {"return_model": "role_decay"},
       "MIN_V2_JOINT": {"proj_weight": "var_role"},
       "MIN_V2_ALL": {"start_mix": True, "return_model": "role_decay", "proj_weight": "var_role"}}
D = os.environ.get("NBA_DATA_DIR", "/home/claude/data")
name, start, end, step = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
logs = pd.read_pickle(f"{D}/real_logs_v3.pkl"); games = pd.read_pickle(f"{D}/real_games_v3.pkl")
dates = sorted(d for d in games.game_date_et.unique() if start <= d <= end)[::step]
t, recs = time.time(), []
for d in dates:
    eng = MinutesEngine(logs[logs.game_date_et < d], games[games.game_date_et < d], d, MinutesConfig(**CFG[name]))
    dg = games[games.game_date_et == d]
    for gid, tid, active, st_act, actual in _day_games(logs, d):
        gr = dg[dg.game_id == gid]
        if not len(gr):
            continue
        h = int(gr.iloc[0].home_team_id)
        bx, n_ot, mg = eng.margin_draws(h, int(gr.iloc[0].away_team_id), 2000, 11, return_margin=True)
        st = expected_starters(active, eng.start_scores(tid, active))
        tm = eng.simulate_team(tid, active, st, {}, bx, n_ot, 11, margin=mg if tid == h else -mg)
        bench = sorted([p for p in active if p not in st], key=lambda p: -tm.params[p].a)
        for k, p in enumerate(tm.players):
            x = tm.samples[:, k]
            recs.append({"date": d, "game_id": gid, "team_id": tid, "player_id": p, "exp_starter": int(p in st),
                         "started": int(p in st_act), "bench_rank": bench.index(p) + 1 if p in bench else 0,
                         "ret": int(any(f.startswith("RETURN") for f in tm.params[p].flags)),
                         "mean": float(x.mean()), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
                         "actual": actual[p], "crps": crps_samples(x[::2], actual[p]),
                         "margin_abs": abs(float(gr.iloc[0].home_score - gr.iloc[0].away_score))})
os.makedirs(os.path.join(ROOT, "docs/validation/v10"), exist_ok=True)
pd.DataFrame(recs).to_pickle(os.path.join(ROOT, f"docs/validation/v10/min_{name}_{start}_{step}.pkl"))
print(json.dumps({"name": name, "start": start, "rows": len(recs), "s": round(time.time() - t)}), flush=True)
