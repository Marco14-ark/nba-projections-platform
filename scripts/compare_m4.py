"""Paired comparison of minutes challengers vs champion (same player-games), game-cluster bootstrap CIs.
Args: CHAMP CHALLENGER [season specs...]  -> uses every m4_<name>_<spec>.pkl present for both."""
import sys, glob, json, numpy as np, pandas as pd
D = "/home/claude/data"
A, B = sys.argv[1], sys.argv[2]
GB = pd.read_pickle(f"{D}/garbage_by_game.pkl")[["game_id", "garbage_min"]]
out = {}
for fa in sorted(glob.glob(f"{D}/m4_{A}_*.pkl")):
    spec = fa.split(f"m4_{A}_")[1]
    fb = f"{D}/m4_{B}_{spec}"
    try:
        a, b = pd.read_pickle(fa), pd.read_pickle(fb)
    except FileNotFoundError:
        continue
    k = ["game_id", "player_id"]
    m = a.merge(b, on=k, suffixes=("_a", "_b")).merge(GB, on="game_id", how="left")
    for x in ("a", "b"):
        m[f"ae_{x}"] = (m[f"mean_{x}"] - m["actual_a"]).abs()
        m[f"cov_{x}"] = ((m["actual_a"] >= m[f"p10_{x}"]) & (m["actual_a"] <= m[f"p90_{x}"])).astype(float)
    g = m.groupby("game_id")
    rng = np.random.default_rng(0)
    ids = m["game_id"].unique()

    def boot(col_a, col_b, sub=None):
        mm = m if sub is None else m[sub]
        s = mm.groupby("game_id").agg(da=(col_a, "sum"), db=(col_b, "sum"), n=(col_a, "size"))
        d = (s.db - s.da).to_numpy(); n = s.n.to_numpy()
        est = d.sum() / n.sum()
        bs = []
        for _ in range(1000):
            i = rng.integers(0, len(d), len(d)); bs.append(d[i].sum() / n[i].sum())
        return round(float(est), 4), [round(float(x), 4) for x in np.percentile(bs, [2.5, 97.5])]
    res = {"rows": int(len(m)), "games": int(len(ids)),
           "mae": [round(m.ae_a.mean(), 3), round(m.ae_b.mean(), 3)], "crps": [round(m.crps_a.mean(), 3), round(m.crps_b.mean(), 3)],
           "cov80": [round(m.cov_a.mean(), 3), round(m.cov_b.mean(), 3)],
           "d_mae": boot("ae_a", "ae_b"), "d_crps": boot("crps_a", "crps_b")}
    for name, sub in (("starters", m.started_a == 1), ("bench", m.started_a == 0),
                      ("G=0", m.garbage_min == 0), ("G>=6", m.garbage_min >= 6)):
        res[name] = {"n": int(sub.sum()), "d_mae": boot("ae_a", "ae_b", sub), "d_crps": boot("crps_a", "crps_b", sub),
                     "bias_a": round(float((m.loc[sub, "mean_a"] - m.loc[sub, "actual_a"]).mean()), 3),
                     "bias_b": round(float((m.loc[sub, "mean_b"] - m.loc[sub, "actual_a"]).mean()), 3)}
    out[spec.replace(".pkl", "")] = res
print(json.dumps(out, indent=1))
