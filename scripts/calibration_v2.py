import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/nba-platform")
from nba_platform.sim.calibration_v2 import long_format, walk_forward, ece, brier, logloss
D = "/home/claude/data"
P = pd.concat([pd.read_pickle(f"{D}/st_STACK_V2_ENV_V3DATA_v2_2025-11-15_2026-04-12_3_props.pkl"),
               pd.read_pickle(f"{D}/st_STACK_V2_ENV_REPRO_v3_2026-10-27_2026-11-12_1_props.pkl")], ignore_index=True)
L = long_format(P)
blocks = [("2025-12-01", "2026-01-01"), ("2026-01-01", "2026-02-01"), ("2026-02-01", "2026-03-01"),
          ("2026-03-01", "2026-04-01"), ("2026-04-01", "2026-05-01"), ("2026-10-01", "2026-12-01")]
M = ["RAW", "PLATT_GLOBAL", "PLATT_MKT", "BETA_MKT", "ISO_MKT", "HIER"]
R = walk_forward(L, blocks, M)
R.to_pickle(f"{D}/calib_v2_walkforward.pkl")
rng = np.random.default_rng(3)
ug, inv = np.unique(R.game_id.values, return_inverse=True)
def ci(d):
    S = np.bincount(inv, d); C = np.bincount(inv).astype(float); idx = rng.integers(0, len(ug), (1000, len(ug)))
    b = S[idx].sum(1) / C[idx].sum(1); return np.round(np.percentile(b, [2.5, 97.5]), 5).tolist()
y = R.over.values
print("TEST rows", len(R), "games", len(ug), "blocks", sorted(R.block.unique()))
for m in M:
    p = R["p_" + m].values
    db = brier(p, y) - brier(R.p_RAW.values, y); dl = logloss(p, y) - logloss(R.p_RAW.values, y)
    print(f"{m:13s} Brier {brier(p,y).mean():.5f} LogLoss {logloss(p,y).mean():.5f} ECE {ece(p,y):.4f} | vs RAW dBrier {db.mean():+.5f} {ci(db)} dLL {dl.mean():+.5f} {ci(dl)}")
print("per market (Brier / LogLoss / ECE):")
for mk, d in R.groupby("market"):
    print(" ", mk, len(d), {m: (round(brier(d['p_'+m], d.over).mean(), 4), round(logloss(d['p_'+m], d.over).mean(), 4), round(ece(d['p_'+m], d.over), 4)) for m in M})
s26 = R[R.block == "2026-10-01"]
print("2026-27 block only:", len(s26), {m: (round(brier(s26['p_'+m], s26.over).mean(), 5), round(logloss(s26['p_'+m], s26.over).mean(), 5), round(ece(s26['p_'+m], s26.over), 4)) for m in M})
