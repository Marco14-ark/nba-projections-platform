"""V10 correlation study: within-simulation correlations of each config vs the correlations of REAL
residuals (actual - champion's pregame mean) on the same games. Pregame inputs only.
Args: START END STEP -> docs/validation/v10/corr_START_STEP.json"""
import json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from nba_platform.lineups.resolve import expected_starters
from nba_platform.minutes.engine import MinutesConfig, MinutesEngine
from nba_platform.sim.game_sim import GameSimulator, JointConfig, TeamInput, accounting_check
from nba_platform.sim.rates import OpportunityRates, RatesConfig

D = os.environ.get("NBA_DATA_DIR", "/home/claude/data")
CFGS = {"CHAMPION": ({}, {}), "JOINT_MC_V2": ({}, {"form_sd": 0.1268, "team_use_sd": 0.0403, "ast_on_makes": True}),
        "JOINT_MC_V2_TEAM": ({}, {"team_use_sd": 0.0403}),
        "MIN_V2_ROLE": ({"start_mix": True}, {}),
        "MIN_V3_SCRIPT": ({}, {"script_two_pass": True}),
        "MIN_V3_COHERENT": ({}, {"script_coherent_sd": 8.0}),
        "SHOT_V3_TEAMVAR": ({}, {"team_use_sd": 0.0415, "team_tov_sd": 0.0612, "team_trip_sd": 0.0686})}
start, end, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
L = pd.read_pickle(f"{D}/real_logs_v3.pkl"); G = pd.read_pickle(f"{D}/real_games_v3.pkl"); G = G[G["flags"].fillna("") == ""]
dates = sorted(d for d in G.game_date_et.unique() if start <= d <= end)[::step]
PAIRS = ("tm_pts", "tm_reb", "ast_tm_pts", "pts_ast", "pts_reb", "fga_pts", "fg3a_fg3m", "tm_min", "min_absmargin_starters",
         "home_away_pts", "poss_team_fga")
sim = {c: {k: [] for k in PAIRS} for c in CFGS}
real = {k: [] for k in PAIRS}


def cz(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


for d in dates:
    hist, gh = L[L.game_date_et < d], G[G.game_date_et < d]
    engs = {c: MinutesEngine(hist, gh, d, MinutesConfig(**mc)) for c, (mc, _) in CFGS.items()}
    r = OpportunityRates(hist, G, d, RatesConfig(hl_player_games=45.0, league_env=True))
    day = L[(L.game_date_et == d) & ~L.not_playing_reason.fillna("").str.startswith("INACTIVE")]
    for g in G[G.game_date_et == d].itertuples():
        out = {}
        for c, (mc, jc) in CFGS.items():
            e = engs[c]
            ti = []
            for tid in (g.home_team_id, g.away_team_id):
                av = list(day.loc[(day.game_id == g.game_id) & (day.team_id == tid), "player_id"])
                ti.append(TeamInput(int(tid), av, expected_starters(av, e.start_scores(int(tid), av))))
            if min(len(t.available) for t in ti) < 5:
                break
            s = GameSimulator(e, r, JointConfig(**jc)).simulate(ti[0], ti[1], n=3000, seed=5)
            assert all(accounting_check(s).values())
            out[c] = s
        if len(out) < len(CFGS):
            continue
        act = day[day.game_id == g.game_id].set_index("player_id")
        ch = out["CHAMPION"]
        for c, s in out.items():
            R = sim[c]
            R["home_away_pts"].append(cz(s.home.score, s.away.score))
            for T in (s.home, s.away):
                S = T.stats
                rot = [k for k in range(len(T.players)) if S["minutes"][:, k].mean() >= 20]
                R["poss_team_fga"].append(cz(s.possessions, S["fga"].sum(1)))
                for k in rot:
                    R["pts_ast"].append(cz(S["pts"][:, k], S["ast"][:, k])); R["pts_reb"].append(cz(S["pts"][:, k], S["reb"][:, k]))
                    R["fga_pts"].append(cz(S["fga"][:, k], S["pts"][:, k])); R["fg3a_fg3m"].append(cz(S["fg3a"][:, k], S["fg3m"][:, k]))
                    if T.players[k] in set(T.players[:0]) or S["minutes"][:, k].mean() >= 28:
                        R["min_absmargin_starters"].append(cz(S["minutes"][:, k], np.abs(s.margin)))
                for i in rot:
                    for j in rot:
                        if i < j:
                            R["tm_pts"].append(cz(S["pts"][:, i], S["pts"][:, j])); R["tm_reb"].append(cz(S["reb"][:, i], S["reb"][:, j]))
                            R["tm_min"].append(cz(S["minutes"][:, i], S["minutes"][:, j]))
                        if i != j:
                            R["ast_tm_pts"].append(cz(S["ast"][:, i], S["pts"][:, j]))
        # real residuals vs champion pregame means (pairs pooled across games)
        real["home_away_pts"].append((g.home_score - ch.home.score.mean(), g.away_score - ch.away.score.mean()))
        for T in (ch.home, ch.away):
            S = T.stats
            rot = [k for k in range(len(T.players)) if S["minutes"][:, k].mean() >= 20 and T.players[k] in act.index]
            res = {k: {st: float(act.loc[T.players[k], col]) - S[st][:, k].mean() for st, col in
                       (("pts", "pts"), ("reb", "reb"), ("ast", "ast"), ("fga", "fga"), ("fg3a", "fg3a"), ("fg3m", "fg3m"), ("minutes", "minutes"))}
                   for k in rot}
            for k in rot:
                real["pts_ast"].append((res[k]["pts"], res[k]["ast"])); real["pts_reb"].append((res[k]["pts"], res[k]["reb"]))
                real["fga_pts"].append((res[k]["fga"], res[k]["pts"])); real["fg3a_fg3m"].append((res[k]["fg3a"], res[k]["fg3m"]))
                if S["minutes"][:, k].mean() >= 28:
                    real["min_absmargin_starters"].append((res[k]["minutes"], abs(g.home_score - g.away_score) - np.abs(ch.margin).mean()))
            for i in rot:
                for j in rot:
                    if i < j:
                        real["tm_pts"].append((res[i]["pts"], res[j]["pts"])); real["tm_reb"].append((res[i]["reb"], res[j]["reb"]))
                        real["tm_min"].append((res[i]["minutes"], res[j]["minutes"]))
                    if i != j:
                        real["ast_tm_pts"].append((res[i]["ast"], res[j]["pts"]))
            tid = T.team_id
            poss = g.home_poss if tid == g.home_team_id else g.away_poss
            tf = float(act[act.team_id == tid].fga.sum())
            real["poss_team_fga"].append((poss - ch.possessions.mean(), tf - S["fga"].sum(1).mean()))


def zmean(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return round(float(np.tanh(np.mean(np.arctanh(np.clip(v, -0.999, 0.999))))), 3) if v else None


res = {"n_games": len(real["home_away_pts"]), "real_residual_corr": {}, "sim_corr": {}}
for k, v in real.items():
    a = np.array(v, dtype=float)
    a = a[np.isfinite(a).all(1)] if len(a) else a
    res["real_residual_corr"][k] = {"r": round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 3), "n": len(a)} if len(a) > 10 else None
for c in CFGS:
    res["sim_corr"][c] = {k: zmean(v) for k, v in sim[c].items()}
os.makedirs(os.path.join(ROOT, "docs/validation/v10"), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, f"docs/validation/v10/corr_{start}_{step}.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
