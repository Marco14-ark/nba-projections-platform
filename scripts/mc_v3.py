"""Monte Carlo structure vs reality (champion STACK_V2_ENV config, point-diff anchor for speed)."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/nba-platform")
from nba_platform.minutes.engine import MinutesEngine, MinutesConfig
from nba_platform.sim.rates import OpportunityRates, RatesConfig
from nba_platform.sim.game_sim import GameSimulator, TeamInput
from nba_platform.lineups.resolve import expected_starters
D = "/home/claude/data"
L = pd.read_pickle(f"{D}/real_logs_v2.pkl"); G = pd.read_pickle(f"{D}/real_games_v2.pkl"); G = G[G["flags"].fillna("") == ""]
p = L[(L.played == 1) & (L.game_date_et >= "2025-10-01")].copy()
for c in ("pts", "reb", "ast", "minutes"):
    p["r_" + c] = p[c] - p.groupby("player_id")[c].transform("mean")
reg = p[p.groupby("player_id").minutes.transform("mean") >= 20]
real = {f"within_{a}_{b}": float(np.corrcoef(reg["r_" + a], reg["r_" + b])[0, 1]) for a, b in (("pts", "reb"), ("pts", "ast"), ("reb", "ast"), ("minutes", "pts"))}
st = reg[reg.started == 1]
pr = []
for _, d in st.groupby(["game_id", "team_id"]):
    r = d.r_pts.to_numpy(); pr += [(r[i], r[j]) for i in range(len(r)) for j in range(i + 1, len(r))]
pr = np.array(pr); real["teammate_starters_pts"] = float(np.corrcoef(pr[:, 0], pr[:, 1])[0, 1])
g25 = G[G.game_date_et >= "2025-10-01"]
m = g25.home_score - g25.away_score
real.update({"sd_team_pts": float(pd.concat([g25.home_score, g25.away_score]).std()), "p_ot": float((g25.periods > 4).mean()),
             "p_abs_margin_ge20": float((m.abs() >= 20).mean()), "p_abs_margin_le5": float((m.abs() <= 5).mean())})
dates = sorted(g25.game_date_et.unique())[20::12]
acc = {k: [] for k in ("pts_reb", "pts_ast", "reb_ast", "min_pts", "team", "ot", "ge20", "le5", "minsd_st")}
for d in dates:
    eng = MinutesEngine(L[L.game_date_et < d], G[G.game_date_et < d], d, MinutesConfig())
    r = OpportunityRates(L[L.game_date_et < d], G, d, RatesConfig()); sim = GameSimulator(eng, r)
    day = L[(L.game_date_et == d) & ~L.not_playing_reason.fillna("").str.startswith("INACTIVE")]
    for g in G[G.game_date_et == d].itertuples():
        ti = []
        for tid in (g.home_team_id, g.away_team_id):
            av = list(day.loc[(day.game_id == g.game_id) & (day.team_id == tid), "player_id"])
            ti.append(TeamInput(int(tid), av, expected_starters(av, eng.start_scores(int(tid), av))))
        s = sim.simulate(ti[0], ti[1], n=4000, seed=5)
        T = s.home; idx = [T.players.index(x) for x in ti[0].starters]
        S = T.stats
        def wc(a, b): return np.mean([np.corrcoef(S[a][:, k], S[b][:, k])[0, 1] for k in idx])
        acc["pts_reb"].append(wc("pts", "reb")); acc["pts_ast"].append(wc("pts", "ast")); acc["reb_ast"].append(wc("reb", "ast"))
        acc["min_pts"].append(wc("minutes", "pts"))
        P = S["pts"][:, idx].astype(float); C = np.corrcoef(P.T); acc["team"].append(C[np.triu_indices(len(idx), 1)].mean())
        mg = s.margin; acc["ot"].append((s.n_ot > 0).mean()); acc["ge20"].append((np.abs(mg) >= 20).mean()); acc["le5"].append((np.abs(mg) <= 5).mean())
        acc["minsd_st"].append(np.median(S["minutes"][:, idx].std(0)))
simv = {"within_pts_reb": np.mean(acc["pts_reb"]), "within_pts_ast": np.mean(acc["pts_ast"]), "within_reb_ast": np.mean(acc["reb_ast"]),
        "within_minutes_pts": np.mean(acc["min_pts"]), "teammate_starters_pts": np.mean(acc["team"]), "p_ot": np.mean(acc["ot"]),
        "p_abs_margin_ge20": np.mean(acc["ge20"]), "p_abs_margin_le5": np.mean(acc["le5"]), "starter_minutes_sd": np.median(acc["minsd_st"]), "games": len(acc["ot"])}
print(json.dumps({"real": real, "sim": {k: float(v) for k, v in simv.items()}}, indent=1))
