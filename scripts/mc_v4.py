"""JOINT_MC_V2 challengers vs champion on the SAME games (out of sample: params fitted on 2024-25).
Args: SEASON_START SEASON_END EVERY OFFSET OUT
Per game: identical inputs/seed for every config. Records per-player CRPS/coverage (PTS, REB, AST, 3PM, PRA)
for players who played, game margin/total CRPS, and within-simulation correlation structure."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/nba-platform")
from nba_platform.minutes.engine import MinutesEngine, MinutesConfig
from nba_platform.minutes.validate import crps_samples
from nba_platform.sim.rates import OpportunityRates, RatesConfig
from nba_platform.sim.game_sim import GameSimulator, TeamInput, JointConfig, accounting_check
from nba_platform.lineups.resolve import expected_starters
D = "/home/claude/data"
start, end, every, off, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
P = json.load(open("/home/claude/nba-platform/models/joint_v2_params.json"))
CFGS = {"CHAMPION": JointConfig(),
        "JOINT_MC_V2_A": JointConfig(form_sd=P["form_sd"]),
        "JOINT_MC_V2_B": JointConfig(team_use_sd=P["team_use_sd"], ast_on_makes=True),
        "JOINT_MC_V2_C": JointConfig(form_sd=P["form_sd"], team_use_sd=P["team_use_sd"], ast_on_makes=True)}
L = pd.read_pickle(f"{D}/real_logs_v3.pkl"); G = pd.read_pickle(f"{D}/real_games_v3.pkl"); G = G[G["flags"].fillna("") == ""]
G = G.merge(pd.read_pickle(f"{D}/garbage_by_game.pkl")[["game_id", "garbage_min"]], on="game_id", how="left")
import os
MIN_GD = os.environ.get("MC_SET") == "GD"
if MIN_GD:   # diagnostic: does the blowout minutes challenger (shared starter pulls) fix teammate structure?
    CFGS = {"CHAMPION": JointConfig(), "MINUTES_GD": JointConfig(), "JOINT_MC_V2_C+GD": CFGS["JOINT_MC_V2_C"]}
dates = sorted(d for d in G.game_date_et.unique() if start <= d <= end)[off::every]
rows, games, corr = [], [], []
for d in dates:
    eng = MinutesEngine(L[L.game_date_et < d], G[G.game_date_et < d], d, MinutesConfig())
    eng_gd = MinutesEngine(L[L.game_date_et < d], G[G.game_date_et < d], d,
                           MinutesConfig(blowout_feature="garbage", dnp_garbage=True)) if MIN_GD else None
    r = OpportunityRates(L[L.game_date_et < d], G, d, RatesConfig())
    day = L[(L.game_date_et == d) & ~L.not_playing_reason.fillna("").str.startswith("INACTIVE")]
    for g in G[G.game_date_et == d].itertuples():
        ti = []
        for tid in (g.home_team_id, g.away_team_id):
            av = list(day.loc[(day.game_id == g.game_id) & (day.team_id == tid), "player_id"])
            ti.append(TeamInput(int(tid), av, expected_starters(av, eng.start_scores(int(tid), av))))
        if min(len(ti[0].available), len(ti[1].available)) < 5:
            continue
        act = day[day.game_id == g.game_id].set_index("player_id")
        for name, jc in CFGS.items():
            e_ = eng_gd if (MIN_GD and name != "CHAMPION") else eng
            if e_ is eng_gd:
                ti = [TeamInput(t.team_id, t.available, expected_starters(t.available, eng_gd.start_scores(t.team_id, t.available))) for t in ti]
            s = GameSimulator(e_, r, jc).simulate(ti[0], ti[1], n=4000, seed=5)
            assert all(accounting_check(s).values())
            mg, tt = s.margin.astype(float), s.total.astype(float)
            games.append({"cfg": name, "game_id": g.game_id, "date": d,
                          "crps_margin": crps_samples(mg, g.home_score - g.away_score),
                          "crps_total": crps_samples(tt, g.home_score + g.away_score)})
            for T, tin in ((s.home, ti[0]), (s.away, ti[1])):
                S = T.stats
                for k, p in enumerate(T.players):
                    if p not in act.index or act.loc[p, "played"] != 1:
                        continue
                    a = act.loc[p]
                    for mk, x, y in (("PTS", S["pts"][:, k], a.pts), ("REB", S["reb"][:, k], a.reb),
                                     ("AST", S["ast"][:, k], a.ast), ("3PM", S["fg3m"][:, k], a.fg3m),
                                     ("PRA", S["pts"][:, k] + S["reb"][:, k] + S["ast"][:, k], a.pts + a.reb + a.ast)):
                        x = x.astype(float)
                        q10, q90 = np.percentile(x, [10, 90])
                        rows.append({"cfg": name, "game_id": g.game_id, "player_id": p, "market": mk,
                                     "started": int(p in tin.starters), "crps": crps_samples(x[::4], y),
                                     "ae": abs(x.mean() - y), "cov": int(q10 <= y <= q90),
                                     "pover": float(np.mean(x > np.floor(np.median(x)) + 0.5)),
                                     "over": int(y > np.floor(np.median(x)) + 0.5)})
                idx = [T.players.index(x) for x in tin.starters]
                def wc(a, b):
                    return float(np.nanmean([np.corrcoef(S[a][:, k], S[b][:, k])[0, 1] for k in idx]))
                Pm = S["pts"][:, idx].astype(float); C = np.corrcoef(Pm.T)
                corr.append({"cfg": name, "game_id": g.game_id, "pts_reb": wc("pts", "reb"), "pts_ast": wc("pts", "ast"),
                             "reb_ast": wc("reb", "ast"), "min_pts": wc("minutes", "pts"),
                             "teammates_pts": float(np.nanmean(C[np.triu_indices(len(idx), 1)]))})
    print(d, len(games), flush=True)
pd.to_pickle({"rows": pd.DataFrame(rows), "games": pd.DataFrame(games), "corr": pd.DataFrame(corr), "params": P}, out)
