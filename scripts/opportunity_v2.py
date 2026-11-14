"""Opportunity-engine evaluation with ORACLE minutes & possessions (isolates opportunity error).
For each test game: expected FGA/3PA/FTA/TOV/OREB/DREB/AST per player given actual minutes,
actual team possessions and actual team misses / makes, from rates fitted strictly before.
Args: START END NAME"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/nba-platform")
from nba_platform.sim.rates import OpportunityRates, RatesConfig
D = "/home/claude/data"
start, end, name = sys.argv[1], sys.argv[2], sys.argv[3]
VAR = {"V1": dict(alpha=1.0, hl=20.0, lineup_kappa=False), "ALPHA125": dict(alpha=1.25, hl=20.0, lineup_kappa=False),
       "ALPHA15": dict(alpha=1.5, hl=20.0, lineup_kappa=False), "HL10": dict(alpha=1.0, hl=10.0, lineup_kappa=False),
       "LINEUPK": dict(alpha=1.0, hl=20.0, lineup_kappa=True), "HL30": dict(alpha=1.0, hl=30.0, lineup_kappa=False), "HL45": dict(alpha=1.0, hl=45.0, lineup_kappa=False), "HL70": dict(alpha=1.0, hl=70.0, lineup_kappa=False), "A125_HL30": dict(alpha=1.25, hl=30.0, lineup_kappa=False), "ALPHA125_HL10": dict(alpha=1.25, hl=10.0, lineup_kappa=False)}
v = VAR[name]
logs = pd.read_pickle(f"{D}/real_logs_v2.pkl"); games = pd.read_pickle(f"{D}/real_games_v2.pkl")
games = games[games["flags"].fillna("") == ""]
p = logs[logs.played == 1]
dates = sorted(d for d in games.game_date_et.unique() if start <= d <= end)
recs, R, refit = [], None, None
for d in dates:
    if R is None or (pd.Timestamp(d) - pd.Timestamp(refit)).days >= 7:
        R = OpportunityRates(logs[logs.game_date_et < d], games, d, RatesConfig(hl_player_games=v["hl"])); refit = d
    for g in games[games.game_date_et == d].itertuples():
        day = p[p.game_id == g.game_id]
        for tid, opp, poss in ((g.home_team_id, g.away_team_id, g.home_poss), (g.away_team_id, g.home_team_id, g.away_poss)):
            t = day[day.team_id == tid]; o = day[day.team_id == opp]
            if len(t) < 5: continue
            T, Dd = R.team(int(tid)), R.team(int(opp))
            rr = [R.player(x) for x in t.player_id]
            gm = 48 + 5 * (g.periods - 4)
            pon = poss * t.minutes.to_numpy() / gm
            use = np.array([x["use"] for x in rr])
            kappa = T["kappa"] if not v["lineup_kappa"] else float((use * pon).sum() / poss * 0 + T["kappa"] * 0.5 + 0.5 * (use * pon).sum() / poss)
            w = (use ** v["alpha"]) * pon; w = w / w.sum()
            U = poss * kappa
            ftov = np.array([x["f_tov"] for x in rr]) * Dd["d_tov"]; ftrip = np.array([x["f_trip"] for x in rr]) * Dd["d_trip"]
            ffga = np.clip(1 - ftov - ftrip, 0.05, None)
            fga = U * w * ffga; p3 = np.clip(np.array([x["p3"] for x in rr]) * Dd["d_p3"], 0, .97)
            fta = U * w * ftrip * np.array([x["fta_trip"] for x in rr]); tov = U * w * ftov
            tm_miss = (t.fga - t.fgm).sum(); op_miss = (o.fga - o.fgm).sum()
            orr = np.array([x["oreb"] for x in rr]) * t.minutes.to_numpy(); drr = np.array([x["dreb"] for x in rr]) * t.minutes.to_numpy()
            oreb_t = t.oreb.sum(); dreb_t = t.dreb.sum()
            # rebounds: team totals predicted from misses x team rates; allocation by rate x minutes
            po = T["oreb_t"]; pdd = T["dreb_t"]
            oreb = tm_miss * po * orr / orr.sum(); dreb = op_miss * pdd * drr / drr.sum()
            ar = np.array([x["ast"] for x in rr]) * t.minutes.to_numpy(); ast = t.fgm.sum() * T["ast_t"] * ar / ar.sum()
            for k, pid in enumerate(t.player_id):
                a = t.iloc[k]
                recs.append({"date": d, "game_id": g.game_id, "player_id": pid, "min": a.minutes,
                             "fga": fga[k], "fga_a": a.fga, "fg3a": fga[k] * p3[k], "fg3a_a": a.fg3a, "fta": fta[k], "fta_a": a.fta,
                             "tov": tov[k], "tov_a": a.tov, "oreb": oreb[k], "oreb_a": a.oreb, "dreb": dreb[k], "dreb_a": a.dreb,
                             "ast": ast[k], "ast_a": a.ast})
X = pd.DataFrame(recs); X.to_pickle(f"{D}/opp_{name}_{start}_{end}.pkl")
print(json.dumps({"name": name, "rows": len(X), **{k: round(float((X[k] - X[k + "_a"]).abs().mean()), 4) for k in ("fga", "fg3a", "fta", "tov", "oreb", "dreb", "ast")}}))
