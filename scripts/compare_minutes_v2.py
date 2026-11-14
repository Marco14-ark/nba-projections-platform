import sys, os, glob, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/nba-platform")
from nba_platform.minutes.validate import summarize, compare, BacktestResult
D = "/home/claude/data"; tag = "2025-11-01_2026-04-12_2"
CH = os.environ.get("CHAMP", "CHAMPION_1_1"); champ = BacktestResult(CH, pd.read_pickle(f"{D}/m2_{CH}_{tag}.pkl"), {})
champ.metrics = summarize(champ.rows)
base = pd.read_pickle(f"{D}/m2_BASELINE_{tag}.pkl"); B = BacktestResult("BASE", base, summarize(base))
g = pd.read_pickle(f"{D}/real_games_v2.pkl")[["game_id", "periods", "home_score", "away_score"]]
f = lambda m: {k: round(m[k], 3) for k in ("n", "mae", "crps", "cov80", "bias")}
print("BASELINE", f(B.metrics["ALL"])); print(CH, f(champ.metrics["ALL"]), "| starters", f(champ.metrics["started=1"]))
names = sys.argv[1:] or [p.split("m2_")[1].split("_2025")[0] for p in sorted(glob.glob(f"{D}/m2_M_*_{tag}.pkl"))]
for n in names:
    r = pd.read_pickle(f"{D}/m2_{n}_{tag}.pkl"); X = BacktestResult(n, r, summarize(r))
    c = compare(champ, X, n_boot=1000)
    ot = r.merge(g, on="game_id"); co = champ.rows.merge(g, on="game_id")
    so = lambda d: round(float((d[(d.periods > 4) & (d.started == 1)]["mean"] - d[(d.periods > 4) & (d.started == 1)].actual).mean()), 2)
    print(n, f(X.metrics["ALL"]), "| starters", f(X.metrics["started=1"]))
    print("   vs champion: dMAE", round(c["mae_diff"], 4), np.round(c["mae_ci95"], 4), " dCRPS", round(c["crps_diff"], 4),
          np.round(c["crps_ci95"], 4), "->", c["decision"], "| OT-starter bias champ/chal", so(co), so(ot))
