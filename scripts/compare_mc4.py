"""Summarize JOINT_MC_V2 challengers vs champion on the same games (paired, game-cluster bootstrap)."""
import sys, json, numpy as np, pandas as pd
D = "/home/claude/data"
d = pd.read_pickle(sys.argv[1]); season_from, season_to = sys.argv[2], sys.argv[3]
R, Gm, C = d["rows"], d["games"], d["corr"]
rng = np.random.default_rng(1)
def ci(delta, groups):
    ug, inv = np.unique(groups, return_inverse=True); S = np.bincount(inv, delta); N = np.bincount(inv).astype(float)
    idx = rng.integers(0, len(ug), (1000, len(ug))); b = S[idx].sum(1) / N[idx].sum(1)
    return [round(float(delta.mean()), 4), [round(float(x), 4) for x in np.percentile(b, [2.5, 97.5])]]
# real structure for the same season window (players >= 20 mpg, deviations from player mean)
L = pd.read_pickle(f"{D}/real_logs_v3.pkl")
p = L[(L.played == 1) & (L.game_date_et >= season_from) & (L.game_date_et <= season_to)].copy()
for c in ("pts", "reb", "ast", "minutes"):
    p["r_" + c] = p[c] - p.groupby("player_id")[c].transform("mean")
reg = p[p.groupby("player_id").minutes.transform("mean") >= 20]
real = {f"{a}_{b}": round(float(np.corrcoef(reg["r_" + a], reg["r_" + b])[0, 1]), 3)
        for a, b in (("pts", "reb"), ("pts", "ast"), ("reb", "ast"), ("minutes", "pts"))}
st = reg[reg.started == 1]; pr = []
for _, g in st.groupby(["game_id", "team_id"]):
    r = g.r_pts.to_numpy(); pr += [(r[i], r[j]) for i in range(len(r)) for j in range(i + 1, len(r))]
pr = np.array(pr); real["teammates_pts"] = round(float(np.corrcoef(pr[:, 0], pr[:, 1])[0, 1]), 3)
out = {"games": int(Gm.game_id.nunique()), "real": real,
       "sim_corr": C.groupby("cfg")[["pts_reb", "pts_ast", "reb_ast", "min_pts", "teammates_pts"]].mean().round(3).to_dict("index"), "vs_champion": {}}
base = R[R.cfg == "CHAMPION"].set_index(["game_id", "player_id", "market"])
gb = Gm[Gm.cfg == "CHAMPION"].set_index("game_id")
for cfg in [c for c in R.cfg.unique() if c != "CHAMPION"]:
    x = R[R.cfg == cfg].set_index(["game_id", "player_id", "market"]).join(base, rsuffix="_c", how="inner")
    res = {}
    for mk, g in x.groupby(level="market"):
        gid = g.index.get_level_values(0).to_numpy()
        res[mk] = {"crps": ci((g.crps - g.crps_c).to_numpy(), gid), "brier": ci(((g.pover - g.over) ** 2 - (g.pover_c - g.over_c) ** 2).to_numpy(), gid),
                   "cov80": [round(float(g.cov_c.mean()), 3), round(float(g["cov"].mean()), 3)]}
    gg = Gm[Gm.cfg == cfg].set_index("game_id").join(gb, rsuffix="_c")
    res["MARGIN"] = {"crps": ci((gg.crps_margin - gg.crps_margin_c).to_numpy(), gg.index.to_numpy())}
    res["TOTAL"] = {"crps": ci((gg.crps_total - gg.crps_total_c).to_numpy(), gg.index.to_numpy())}
    out["vs_champion"][cfg] = res
print(json.dumps(out, indent=1))
