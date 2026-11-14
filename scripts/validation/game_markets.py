"""V8 historical validation of game markets (Moneyline, Spread/margin, Total, Team totals).

Inputs: walk-forward 'games' frames produced by the stacked backtests (pregame information only;
each date is forecast from data strictly before it). Output: JSON + Markdown report.
No market odds are used anywhere in here."""
from __future__ import annotations
import glob, json, os, re, sys
import numpy as np
import pandas as pd

DATA = os.environ.get("NBA_DATA_DIR", "/home/claude/data")
OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/validation"


def absences(logs: pd.DataFrame) -> pd.DataFrame:
    """Rotation absences per team-game: players averaging >=20 min over the team's previous 10 games
    (absent games count as 0; strictly before the game) who do not appear in the game's box score.
    Pre-game information only (the box only tells who was missing)."""
    out = []
    for tid, l in logs[logs["played"] == 1].groupby("team_id"):
        order = l[["game_id", "game_date_et"]].drop_duplicates().sort_values(["game_date_et", "game_id"])["game_id"]
        M = l.pivot_table(index="game_id", columns="player_id", values="minutes", aggfunc="sum").reindex(order).fillna(0.0)
        prior = M.shift(1).rolling(10, min_periods=5).mean()
        miss = (prior >= 20) & (M == 0)
        out.append(pd.DataFrame({"game_id": M.index, "team_id": tid, "rot_out": miss.sum(axis=1).values}))
    return pd.concat(out, ignore_index=True)


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot, table = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        tot += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
        table.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(m.sum()),
                      "p_mean": round(float(p[m].mean()), 4), "y_rate": round(float(y[m].mean()), 4)})
    return float(tot), table


def metrics(d: pd.DataFrame) -> dict:
    y = (d["actual_margin"] > 0).astype(float).values
    p = np.clip(d["p_home"].values, 1e-4, 1 - 1e-4)
    e, table = ece(p, y)
    hm = (d["model_total"] + d["model_margin"]) / 2
    am = (d["model_total"] - d["model_margin"]) / 2
    ha = (d["actual_total"] + d["actual_margin"]) / 2
    aa = (d["actual_total"] - d["actual_margin"]) / 2
    fav_home = d["model_margin"] > 0
    fav_err = np.where(fav_home, d["actual_margin"] - d["model_margin"], -(d["actual_margin"] - d["model_margin"]))
    out = {
        "n_games": int(len(d)), "dates": [str(d["date"].min()), str(d["date"].max())],
        "moneyline": {"brier": float(np.mean((p - y) ** 2)), "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                      "ece": e, "accuracy": float(np.mean((p > .5) == (y == 1))), "calibration": table,
                      "brier_climatology": float(np.mean((y.mean() - y) ** 2))},
        "spread": {"margin_mae": float(np.mean(np.abs(d["actual_margin"] - d["model_margin"]))),
                   "margin_rmse": float(np.sqrt(np.mean((d["actual_margin"] - d["model_margin"]) ** 2))),
                   "margin_bias_home": float(np.mean(d["actual_margin"] - d["model_margin"])),
                   "crps_margin": float(d["crps_margin"].mean()), "coverage80_margin": float(d["cov80_margin"].mean()),
                   "favorite_margin_bias": float(np.mean(fav_err)),
                   "note": "favorite_margin_bias > 0 means favourites won by MORE than projected"},
        "total": {"mae": float(np.mean(np.abs(d["actual_total"] - d["model_total"]))),
                  "bias": float(np.mean(d["actual_total"] - d["model_total"])),
                  "crps": float(d["crps_total"].mean()), "coverage80": float(d["cov80_total"].mean())},
        "team_totals": {"mae_home": float(np.mean(np.abs(ha - hm))), "mae_away": float(np.mean(np.abs(aa - am))),
                        "bias_home": float(np.mean(ha - hm)), "bias_away": float(np.mean(aa - am))},
    }
    if "base_p_home" in d:
        pb = np.clip(d["base_p_home"].values, 1e-4, 1 - 1e-4)
        out["moneyline"]["brier_rating_baseline"] = float(np.mean((pb - y) ** 2))
        out["moneyline"]["logloss_rating_baseline"] = float(-np.mean(y * np.log(pb) + (1 - y) * np.log(1 - pb)))
    # cohorts
    coh = {}
    big = np.abs(d["model_margin"]) >= 8
    for name, m in (("strong_favourite(|proj|>=8)", big), ("close(|proj|<8)", ~big)):
        if m.sum() >= 20:
            pp, yy = p[m.values], y[m.values]
            coh[name] = {"n": int(m.sum()), "brier": float(np.mean((pp - yy) ** 2)),
                         "fav_margin_bias": float(np.mean(fav_err[m.values])),
                         "fav_win_pred": float(np.mean(np.maximum(pp, 1 - pp))),
                         "fav_win_actual": float(np.mean(np.where(pp >= .5, yy, 1 - yy)))}
    if "rot_out" in d:
        for name, m in (("many_absences(>=3 rotation out)", d["rot_out"] >= 3), ("few_absences(<3)", d["rot_out"] < 3)):
            if m.sum() >= 20:
                coh[name] = {"n": int(m.sum()), "brier": float(np.mean((p[m.values] - y[m.values]) ** 2)),
                             "margin_mae": float(np.mean(np.abs(d["actual_margin"] - d["model_margin"])[m])),
                             "total_mae": float(np.mean(np.abs(d["actual_total"] - d["model_total"])[m])),
                             "total_bias": float(np.mean((d["actual_total"] - d["model_total"])[m]))}
    out["cohorts"] = coh
    return out


def main():
    logs = pd.concat([pd.read_pickle(f"{DATA}/cdnnba_2025_logs.pkl")] +
                     ([pd.read_pickle(f"{DATA}/cdnnba_2026_logs.pkl")] if os.path.exists(f"{DATA}/cdnnba_2026_logs.pkl") else []))
    ab = absences(logs).groupby("game_id")["rot_out"].sum().rename("rot_out")
    res = {}
    for f in sorted(glob.glob(f"{DATA}/st_*_games.pkl")):
        m = re.match(r"st_(.+)_(v\d)_(\d{4}-\d\d-\d\d)_(\d{4}-\d\d-\d\d)_(\d)_games\.pkl", os.path.basename(f))
        if not m:
            continue
        name, ver, a, b, _ = m.groups()
        d = pd.read_pickle(f)
        if len(d) < 50:
            continue
        d = d.merge(ab, left_on="game_id", right_index=True, how="left")
        res[f"{name}|{ver}|{a}..{b}"] = metrics(d)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/game_markets_v8.json", "w"), indent=1)
    lines = ["# V8 — Validación histórica de mercados de juego", "",
             "Walk-forward, solo información pregame. Sin cuotas de mercado.", "",
             "| Modelo | Periodo | N | ML Brier | ML LogLoss | ML ECE | Margen MAE | Cob80 margen | Total MAE | Total CRPS | Cob80 total | TT MAE (L/V) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        n, v, per = k.split("|")
        ml, sp, tt, te = r["moneyline"], r["spread"], r["total"], r["team_totals"]
        lines.append(f"| {n} ({v}) | {per} | {r['n_games']} | {ml['brier']:.4f} | {ml['logloss']:.4f} | {ml['ece']:.3f} | "
                     f"{sp['margin_mae']:.2f} | {sp['coverage80_margin']:.2f} | {tt['mae']:.2f} | {tt['crps']:.2f} | "
                     f"{tt['coverage80']:.2f} | {te['mae_home']:.2f}/{te['mae_away']:.2f} |")
    open(f"{OUT}/game_markets_v8.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
