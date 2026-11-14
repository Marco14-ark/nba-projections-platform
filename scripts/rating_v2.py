"""Walk-forward comparison of team ratings on REAL games (leak-free).
Champion V1 = point-differential rating. Challenger V2 = lineup RAPM x available players.
Expected minutes = pregame proxy: last-10 available-game average (DNP=0), rescaled to 240.
"""
import sys, time, json, numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, "/home/claude/nba-platform")
from nba_platform.rating.team_rating import LineupRating, PointDiffRating, RatingConfig, rest_days
from nba_platform.sim.rates import OpportunityRates
D = "/home/claude/data"
start, end = sys.argv[1], sys.argv[2]
ridge = float(sys.argv[3]) if len(sys.argv) > 3 else 3000.0
logs = pd.read_pickle(f"{D}/real_logs_v2.pkl"); games = pd.read_pickle(f"{D}/real_games_v2.pkl")
seg = pd.concat([pd.read_pickle(f"{D}/cdnnba_{s}_segments.pkl") for s in ("2024", "2025")])
rd = rest_days(games).set_index(["game_id", "team_id"])["rest"]
avail = logs[~logs["not_playing_reason"].fillna("").str.startswith("INACTIVE")].sort_values("game_date_et")
dates = sorted(d for d in games.game_date_et.unique() if start <= d <= end)
recs, model, refit_day = [], None, None
t = time.time()
for d in dates:
    if model is None or (pd.Timestamp(d) - pd.Timestamp(refit_day)).days >= 7:
        model = LineupRating(seg, games, d, RatingConfig(ridge=ridge)); refit_day = d
        rates = OpportunityRates(logs[logs.game_date_et < d], games, d)
    pdr = PointDiffRating(games, d)
    hist = avail[avail.game_date_et < d]
    last = hist.groupby("player_id").tail(10).groupby("player_id")["minutes"].mean()
    for g in games[games.game_date_et == d].itertuples():
        day = avail[avail.game_id == g.game_id]
        mins = {}
        for tid in (g.home_team_id, g.away_team_id):
            ps = day.loc[day.team_id == tid, "player_id"]
            mins[tid] = {p: float(last.get(p, 5.0)) for p in ps}
        poss = rates.expected_pace(int(g.home_team_id), int(g.away_team_id))
        pr = model.predict(mins[g.home_team_id], mins[g.away_team_id], poss,
                           rd.get((g.game_id, g.home_team_id), 2), rd.get((g.game_id, g.away_team_id), 2))
        recs.append({"date": d, "game_id": g.game_id, "actual_margin": g.home_score - g.away_score,
                     "actual_total": g.home_score + g.away_score, "v1_margin": pdr.margin(int(g.home_team_id), int(g.away_team_id)),
                     "v1_sd": pdr.sd, "v2_margin": pr["margin"], "v2_total": pr["total"], "periods": g.periods})
R = pd.DataFrame(recs); R.to_pickle(f"{D}/rating_{start}_{end}_{int(ridge)}.pkl")
print(json.dumps({"games": len(R), "s": round(time.time() - t), "b2b": model.b2b_effect, "home_off": model.home_off}))
