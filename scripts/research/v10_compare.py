"""V10 paired comparisons (same games, same seeds, same temporal cuts). Cluster bootstrap by game.
Writes docs/validation/v10_report.json"""
import glob, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V = os.path.join(ROOT, "docs/validation/v10")
SEASON = {"2026-10-27": "2026-27", "2025-11-15": "2025-26"}


def r(x, k=3):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def cboot(d, g, n=1000, seed=3):
    ug, inv = np.unique(g, return_inverse=True)
    S, C = np.bincount(inv, weights=d), np.bincount(inv).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(ug), (n, len(ug)))
    b = S[idx].sum(1) / C[idx].sum(1)
    return [r(np.percentile(b, 2.5), 3), r(np.percentile(b, 97.5), 3)]


def min_profile(x):
    if x.ret: return "regresa de lesión"
    if x.exp_starter and x.started: return "titular (acertado)"
    if x.exp_starter and not x.started: return "titular esperado que no inició"
    if x.bench_rank == 1: return "sexto hombre (1º de banca)"
    if x.bench_rank <= 3: return "banca 2º-3º"
    return "banca profunda (4º+)"


def minutes():
    out = {}
    for f in sorted(glob.glob(f"{V}/min_CHAMPION_*.pkl")):
        tag = f.split("min_CHAMPION_")[1][:-4]
        st = tag.split("_")[0]
        C = pd.read_pickle(f)
        C["profile"] = C.apply(min_profile, axis=1)
        sea = SEASON[st]
        res = {"CHAMPION": {"mae": r((C["mean"] - C.actual).abs().mean()), "crps": r(C.crps.mean()),
                            "bias": r((C["mean"] - C.actual).mean()), "cov80": r(((C.p10 <= C.actual) & (C.actual <= C.p90)).mean()),
                            "extreme_gt15": r(((C["mean"] - C.actual).abs() > 15).mean())},
               "by_profile_champion": {p: {"n": int(len(d)), "mae": r((d["mean"] - d.actual).abs().mean(), 2),
                                           "bias": r((d["mean"] - d.actual).mean(), 2)} for p, d in C.groupby("profile")}}
        for g in sorted(glob.glob(f"{V}/min_*_{tag}.pkl")):
            name = os.path.basename(g)[4:-len(tag) - 5]
            if name == "CHAMPION":
                continue
            X = pd.read_pickle(g)
            m = C.merge(X[["game_id", "player_id", "mean", "crps", "p10", "p90"]], on=["game_id", "player_id"], suffixes=("", "_x"))
            dm = (m["mean_x"] - m.actual).abs() - (m["mean"] - m.actual).abs()
            dc = m.crps_x - m.crps
            prof = {}
            for p, d in m.groupby("profile"):
                dd = (d["mean_x"] - d.actual).abs() - (d["mean"] - d.actual).abs()
                prof[p] = {"n": int(len(d)), "mae_x": r((d["mean_x"] - d.actual).abs().mean(), 2), "bias_x": r((d["mean_x"] - d.actual).mean(), 2),
                           "d_mae": r(dd.mean(), 3), "ci": cboot(dd.values, d.game_id.values)}
            res[name] = {"mae": r((m["mean_x"] - m.actual).abs().mean()), "crps": r(m.crps_x.mean()), "bias": r((m["mean_x"] - m.actual).mean()),
                         "cov80": r(((m.p10_x <= m.actual) & (m.actual <= m.p90_x)).mean()),
                         "extreme_gt15": r(((m["mean_x"] - m.actual).abs() > 15).mean()),
                         "d_mae": r(dm.mean()), "d_mae_ci": cboot(dm.values, m.game_id.values),
                         "d_crps": r(dc.mean()), "d_crps_ci": cboot(dc.values, m.game_id.values), "by_profile": prof}
        out[sea] = res
    return out


def stack():
    out = {}
    for f in sorted(glob.glob(f"{V}/st_CHAMPION_*_props.pkl")):
        tag = f.split("st_CHAMPION_")[1][:-len("_props.pkl")]
        sea = SEASON[tag.split("_")[0]]
        C, CG = pd.read_pickle(f), pd.read_pickle(f.replace("_props", "_games"))
        res = {}
        for g in sorted(glob.glob(f"{V}/st_*_{tag}_props.pkl")):
            name = os.path.basename(g)[3:-len(tag) - len("_props.pkl") - 1]
            X, XG = pd.read_pickle(g), pd.read_pickle(g.replace("_props", "_games"))
            k = ["game_id", "player_id", "market"]
            m = C.merge(X[k + ["mean", "crps", "p10", "p90", "p_over", "minutes_pred"]], on=k, suffixes=("", "_x"))
            pm = {}
            for mk, d in m.groupby("market"):
                dd = (d["mean_x"] - d.actual).abs() - (d["mean"] - d.actual).abs()
                dc = d.crps_x - d.crps
                row = {"n": int(len(d)), "mae": r((d["mean_x"] - d.actual).abs().mean()), "crps": r(d.crps_x.mean()),
                       "bias": r((d["mean_x"] - d.actual).mean()), "cov80": r(((d.p10_x <= d.actual) & (d.actual <= d.p90_x)).mean())}
                ok = d.line.notna()
                if ok.any():
                    row["brier"] = r(((d.p_over_x - d.over) ** 2)[ok].mean(), 4)
                if name != "CHAMPION":
                    row.update({"d_mae": r(dd.mean()), "d_mae_ci": cboot(dd.values, d.game_id.values),
                                "d_crps": r(dc.mean()), "d_crps_ci": cboot(dc.values, d.game_id.values)})
                    if ok.any():
                        db = ((d.p_over_x - d.over) ** 2 - (d.p_over - d.over) ** 2)[ok]
                        row["d_brier"] = r(db.mean(), 5)
                        row["d_brier_ci"] = cboot(db.values, d.game_id[ok].values)
                pm[mk] = row
            gm = CG.merge(XG, on="game_id", suffixes=("", "_x"))
            y = (gm.actual_margin > 0).astype(float)
            gres = {"n": int(len(gm)),
                    "margin_mae": r((gm.model_margin_x - gm.actual_margin).abs().mean()),
                    "total_mae": r((gm.model_total_x - gm.actual_total).abs().mean()),
                    "total_bias": r((gm.model_total_x - gm.actual_total).mean(), 2),
                    "crps_margin": r(gm.crps_margin_x.mean()), "crps_total": r(gm.crps_total_x.mean()),
                    "ml_brier": r(((gm.p_home_x - y) ** 2).mean(), 4),
                    "cov80_total": r(gm.cov80_total_x.mean()), "cov80_margin": r(gm.cov80_margin_x.mean())}
            if name != "CHAMPION":
                for key, a, b in (("d_margin_mae", (gm.model_margin_x - gm.actual_margin).abs(), (gm.model_margin - gm.actual_margin).abs()),
                                  ("d_total_mae", (gm.model_total_x - gm.actual_total).abs(), (gm.model_total - gm.actual_total).abs()),
                                  ("d_crps_total", gm.crps_total_x, gm.crps_total), ("d_crps_margin", gm.crps_margin_x, gm.crps_margin),
                                  ("d_ml_brier", (gm.p_home_x - y) ** 2, (gm.p_home - y) ** 2)):
                    d = (a - b).values
                    gres[key] = r(d.mean(), 4)
                    gres[key + "_ci"] = cboot(d, gm.game_id.values)
            res[name] = {"props": pm, "games": gres,
                         "minutes_mae": r((m.minutes_pred_x - m.minutes_actual)[m.market == "PTS"].abs().mean())}
        out[sea] = res
    return out


def decomposition():
    """Oracle experiments, now separating the random part of shot volume (Poisson-like) from the
    systematic part, so not all residual is attributed to luck."""
    out = {}
    L = pd.read_pickle(os.environ.get("NBA_DATA_DIR", "/home/claude/data") + "/real_logs_v3.pkl")
    box = L[L.played == 1][["game_id", "player_id", "fga", "fta"]]
    for f in sorted(glob.glob(f"{V}/st_*_props.pkl")):
        name = os.path.basename(f)[3:].rsplit("_", 3)[0]
        tag = f.split(f"st_{name}_")[1][:-len("_props.pkl")]
        sea = SEASON[tag.split("_")[0]]
        P = pd.read_pickle(f)
        pts = P[(P.market == "PTS") & (P.minutes_pred >= 3) & (P.minutes_actual > 0)].copy()
        fga = P[(P.market == "FGA") & (P.minutes_pred >= 3) & (P.minutes_actual > 0)].set_index(["game_id", "player_id"])
        if not len(fga):
            continue
        pts = pts.join(fga[["mean", "actual"]].rename(columns={"mean": "fga_pred", "actual": "fga_act"}), on=["game_id", "player_id"]).dropna(subset=["fga_pred"])
        base = (pts["mean"] - pts.actual).abs().mean()
        orc_min = (pts["mean"] * pts.minutes_actual / pts.minutes_pred - pts.actual).abs().mean()
        # FGA error given actual minutes: systematic vs random (Poisson expected abs dev sqrt(2*lam/pi))
        lam = pts.fga_pred * pts.minutes_actual / pts.minutes_pred
        fga_err = (lam - pts.fga_act).abs().mean()
        fga_noise = np.sqrt(2 * lam.clip(lower=0) / np.pi).mean()
        out.setdefault(sea, {})[name] = {"pts_mae": r(base), "pts_mae_oracle_minutes": r(orc_min),
                                         "minutes_share": r(1 - orc_min / base),
                                         "fga_mae_given_actual_minutes": r(fga_err), "fga_mae_expected_pure_noise": r(fga_noise),
                                         "fga_systematic_excess": r(fga_err - fga_noise)}
    return out


if __name__ == "__main__":
    rep = {"minutes": minutes(), "stack": stack(), "decomposition": decomposition()}
    for f in glob.glob(f"{V}/corr_*.json"):
        rep.setdefault("correlations", {})[SEASON.get(os.path.basename(f)[5:15], f)] = json.load(open(f))
    rep["decision"] = [
        "Champion STACK_V2_ENV se conserva (sin muestra prospectiva de ningún challenger todavía).",
        "MIN_V2_ROLE (titulares inciertos simulados) → SHADOW STACK_V6_MINROLE: mejora minutos en ambas temporadas y props en 2025-26; neutral en 2026-27; no toca game markets.",
        "STACK_V5_PACE sigue en SHADOW: mejora totales en 2026-27, neutral en 2025-26.",
        "JOINT_MC_V2 / JOINT_MC_V2_TEAM: cambios pequeños y no consistentes entre temporadas → descartados por ahora.",
        "MIN_V2_JOINT (reparto por rol): empeora MAE de minutos → descartado. MIN_V2_RETURN: efecto no significativo → descartado.",
        "SHOT_V2_USAGE (half-life corto de usage): empeora PTS/FTA en 2026-27 → descartado. SHOT_V2_LINEUP: elasticidad de usage estimada 1.06 (≈ proporcional, igual que el Champion) → sin cambio. SHOT_V2_BUDGET: el Champion ya asigna tiros dentro de un presupuesto de equipo.",
    ]
    json.dump(rep, open(os.path.join(ROOT, "docs/validation/v10_report.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, ensure_ascii=False)[:6000])
