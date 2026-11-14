"""Estimate JOINT_MC_V2 variances on a TRAINING season only (default 2024-25).
form_sd^2  = mean cross-moment of normalized per-possession residuals (usage, rebounds, assists)
             -> one shared lognormal factor with equal loadings (method of moments).
team_use_sd = relative SD of team uses per possession within team (team-game level)."""
import sys, json, numpy as np, pandas as pd
D = "/home/claude/data"
season = sys.argv[1] if len(sys.argv) > 1 else "2024-25"
L = pd.read_pickle(f"{D}/real_logs_v3.pkl"); G = pd.read_pickle(f"{D}/real_games_v3.pkl")
G = G[G.season == season]
L = L[L.game_id.isin(G.game_id)]
tr = L[(L.played == 1) & (L.minutes >= 10) & (L.team_poss_on > 0)].copy()
tr["use"] = tr.fga + 0.44 * tr.fta + tr.tov
tr = tr[tr.groupby("player_id").player_id.transform("size") >= 20]
g = tr.groupby("player_id")
S = ["use", "reb", "ast"]
for k in S:
    tr["mu_" + k] = g[k].transform("sum") / g.team_poss_on.transform("sum") * tr.team_poss_on
    tr["e_" + k] = (tr[k] - tr["mu_" + k]) / tr["mu_" + k]
t = tr[(tr[[f"mu_{k}" for k in S]] > 0.5).all(axis=1)]
cross = {f"{a}_{b}": float((t["e_" + a] * t["e_" + b]).mean()) for a, b in (("use", "reb"), ("use", "ast"), ("reb", "ast"))}
form_var = max(0.0, float(np.mean(list(cross.values()))))
tl = L[L.played == 1].groupby(["game_id", "team_id"]).agg(fga=("fga", "sum"), fta=("fta", "sum"), tov=("tov", "sum")).reset_index()
tl["use"] = tl.fga + 0.44 * tl.fta + tl.tov
poss = pd.concat([G[["game_id", "home_team_id", "home_poss"]].set_axis(["game_id", "team_id", "poss"], axis=1),
                  G[["game_id", "away_team_id", "away_poss"]].set_axis(["game_id", "team_id", "poss"], axis=1)])
tl = tl.merge(poss, on=["game_id", "team_id"]); tl = tl[tl.poss > 0]
tl["k"] = tl.use / tl.poss
rel = tl.k / tl.groupby("team_id").k.transform("mean") - 1
out = {"season": season, "n_player_games": int(len(t)), "cross_moments": cross, "form_sd": round(float(np.sqrt(form_var)), 4),
       "team_use_sd": round(float(np.log1p(rel).std()), 4), "n_team_games": int(len(tl))}
json.dump(out, open("/home/claude/nba-platform/models/joint_v2_params.json", "w"), indent=1)
print(json.dumps(out, indent=1))
