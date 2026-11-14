"""Monte Carlo structure vs reality on real 2025-26 games (champion stack)."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/nba-platform")
from nba_platform.minutes.engine import MinutesEngine
from nba_platform.sim.rates import OpportunityRates
from nba_platform.sim.game_sim import GameSimulator, TeamInput
from nba_platform.lineups.resolve import expected_starters
D = "/home/claude/data"
logs = pd.read_pickle(f"{D}/real_logs_v2.pkl"); games = pd.read_pickle(f"{D}/real_games_v2.pkl")
games = games[games["flags"].fillna("") == ""]
p = logs[logs.played == 1].copy()
# REAL: teammate correlation of points residuals among starters (residual vs player's season mean)
s25 = p[p.game_date_et >= "2025-10-01"].copy()
s25["res"] = s25.pts - s25.groupby("player_id").pts.transform("mean")
st = s25[s25.started == 1]
pairs = []
for (g, t), d in st.groupby(["game_id", "team_id"]):
    r = d.res.to_numpy()
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            pairs.append((r[i], r[j]))
pr = np.array(pairs); real_corr = float(np.corrcoef(pr[:, 0], pr[:, 1])[0, 1])
g25 = games[games.game_date_et >= "2025-10-01"]
real = {"teammate_starter_pts_corr": real_corr, "p_ot": float((g25.periods > 4).mean()),
        "p_abs_margin_ge20": float(((g25.home_score - g25.away_score).abs() >= 20).mean()),
        "starter_min_sd": float(st.groupby("player_id").minutes.std().median()),
        "starter_fga_sd": float(st.groupby("player_id").fga.std().median())}
# SIM
dates = sorted(g25.game_date_et.unique())[20::12]
cors, pot, pbl, msd, fsd = [], [], [], [], []
for d in dates:
    eng = MinutesEngine(logs[logs.game_date_et < d], games[games.game_date_et < d], d)
    r = OpportunityRates(logs[logs.game_date_et < d], games, d); sim = GameSimulator(eng, r)
    day = logs[(logs.game_date_et == d) & ~logs.not_playing_reason.fillna("").str.startswith("INACTIVE")]
    for g in games[games.game_date_et == d].itertuples():
        ti = []
        for tid in (g.home_team_id, g.away_team_id):
            av = list(day.loc[(day.game_id == g.game_id) & (day.team_id == tid), "player_id"])
            ti.append(TeamInput(int(tid), av, expected_starters(av, eng.start_scores(int(tid), av))))
        s = sim.simulate(ti[0], ti[1], n=3000, seed=5)
        T = s.home; idx = [T.players.index(x) for x in ti[0].starters]
        P = T.stats["pts"][:, idx].astype(float); C = np.corrcoef(P.T); cors.append(C[np.triu_indices(len(idx), 1)].mean())
        msd.append(np.median(T.stats["minutes"][:, idx].std(0))); fsd.append(np.median(T.stats["fga"][:, idx].std(0)))
        pot.append(float((s.n_ot > 0).mean())); pbl.append(float((np.abs(s.margin) >= 20).mean()))
simv = {"teammate_starter_pts_corr": float(np.mean(cors)), "p_ot": float(np.mean(pot)), "p_abs_margin_ge20": float(np.mean(pbl)),
        "starter_min_sd": float(np.median(msd)), "starter_fga_sd": float(np.median(fsd)), "games": len(cors)}
print(json.dumps({"real": real, "sim": simv}, indent=1))
