"""Paired comparison CHAMPION vs STACK_V2 on identical player-markets and games. Args: TAG"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/nba-platform")
from nba_platform.sim.calibration import MarketCalibrator
D = "/home/claude/data"; tag = sys.argv[1]
A = sys.argv[2] if len(sys.argv) > 2 else "CHAMPION"; B = sys.argv[3] if len(sys.argv) > 3 else "STACK_V2"
Pc, Gc = pd.read_pickle(f"{D}/st_{A}_{tag}_props.pkl"), pd.read_pickle(f"{D}/st_{A}_{tag}_games.pkl")
Ps, Gs = pd.read_pickle(f"{D}/st_{B}_{tag}_props.pkl"), pd.read_pickle(f"{D}/st_{B}_{tag}_games.pkl")
print(A, "->", B)
rng = np.random.default_rng(7)
def ci(d, games):
    ug, inv = np.unique(games, return_inverse=True); S = np.bincount(inv, d); C = np.bincount(inv).astype(float)
    idx = rng.integers(0, len(ug), (2000, len(ug))); b = S[idx].sum(1) / C[idx].sum(1)
    return np.round(np.percentile(b, [2.5, 97.5]), 4).tolist()
k = ["game_id", "player_id", "market"]
M = Pc.merge(Ps, on=k, suffixes=("_c", "_s"))
print(f"PLAYER MARKETS ({tag}): rows per market ~", int(len(M) / M.market.nunique()))
mins = M[M.market == "PTS"]
dm = (np.abs(mins.minutes_pred_s - mins.minutes_actual_s) - np.abs(mins.minutes_pred_c - mins.minutes_actual_c)).to_numpy()
print(f"  MIN  MAE {np.abs(mins.minutes_pred_c-mins.minutes_actual_c).mean():.3f} -> {np.abs(mins.minutes_pred_s-mins.minutes_actual_s).mean():.3f}  diff {dm.mean():+.4f} CI {ci(dm, mins.game_id.values)}  (players who played)")
for mk, d in M.groupby("market"):
    ae = (np.abs(d.mean_s - d.actual_s) - np.abs(d.mean_c - d.actual_c)).to_numpy()
    cr = (d.crps_s - d.crps_c).to_numpy()
    b = d.dropna(subset=["p_over_c", "p_over_s"])
    br = ((b.p_over_s - b.over_s) ** 2 - (b.p_over_c - b.over_c) ** 2).to_numpy()
    cov = lambda s: ((d["actual_" + s] >= d["p10_" + s]) & (d["actual_" + s] <= d["p90_" + s])).mean()
    print(f"  {mk:4s} MAE {np.abs(d.mean_c-d.actual_c).mean():.3f}->{np.abs(d.mean_s-d.actual_s).mean():.3f} d{ae.mean():+.4f} CI{ci(ae, d.game_id.values)} | "
          f"CRPS {d.crps_c.mean():.3f}->{d.crps_s.mean():.3f} d{cr.mean():+.4f} CI{ci(cr, d.game_id.values)} | "
          f"Brier {((b.p_over_c-b.over_c)**2).mean():.4f}->{((b.p_over_s-b.over_s)**2).mean():.4f} d{br.mean():+.4f} CI{ci(br, b.game_id.values)} | cov80 {cov('c'):.3f}->{cov('s'):.3f}")
G = Gc.merge(Gs, on="game_id", suffixes=("_c", "_s"))
hw = (G.actual_margin_c > 0).astype(float)
for nm, col, act in (("margin", "model_margin", "actual_margin"), ("total", "model_total", "actual_total")):
    d = (np.abs(G[col + "_s"] - G[act + "_c"]) - np.abs(G[col + "_c"] - G[act + "_c"])).to_numpy()
    print(f"GAME {nm} MAE {np.abs(G[col+'_c']-G[act+'_c']).mean():.3f} -> {np.abs(G[col+'_s']-G[act+'_c']).mean():.3f} d{d.mean():+.4f} CI {ci(d, G.game_id.values)}")
bd = ((G.p_home_s - hw) ** 2 - (G.p_home_c - hw) ** 2).to_numpy()
print(f"GAME win Brier {((G.p_home_c-hw)**2).mean():.4f} -> {((G.p_home_s-hw)**2).mean():.4f} d{bd.mean():+.4f} CI {ci(bd, G.game_id.values)}  | games {len(G)}")
print(f"GAME cov80 margin {G.cov80_margin_c.mean():.3f}->{G.cov80_margin_s.mean():.3f} total {G.cov80_total_c.mean():.3f}->{G.cov80_total_s.mean():.3f} | accounting ok {G.accounting_ok_s.mean():.3f}")
