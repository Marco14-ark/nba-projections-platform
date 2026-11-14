"""V9 quantitative research (retrospective diagnostics; actual results are used ONLY to measure error,
never as forecast inputs). Champion = STACK_V2_ENV walk-forward reconstructions (backtests), kept
separate from prospective (real frozen) forecasts.
Outputs docs/validation/v9_research.json"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import norm
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from nba_platform.sim.rates import OpportunityRates, RatesConfig

D = os.environ.get("NBA_DATA_DIR", "/home/claude/data")
OUT = os.path.join(ROOT, "docs", "validation", "v9_research.json")
SEASONS = {"2025-26": ("v3", "2025-11-15_2026-04-12_3"), "2026-27": ("v3", "2026-10-27_2026-11-12_1")}
L = pd.read_pickle(f"{D}/real_logs_v3.pkl")
G = pd.read_pickle(f"{D}/real_games_v3.pkl")
L = L[L.played == 1]


def r(x, k=3):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def load(kind, season, model="STACK_V2_ENV"):
    v, per = SEASONS[season]
    return pd.read_pickle(f"{D}/st_{model}_{v}_{per}_{kind}.pkl")


# ------------------------------------------------------------------ pre-game player context (from earlier games only)
def pregame_features():
    rows = []
    for tid, l in L.groupby("team_id"):
        order = l[["game_id", "game_date_et"]].drop_duplicates().sort_values(["game_date_et", "game_id"])
        gi = {g: i for i, g in enumerate(order.game_id)}
        M = l.pivot_table(index="game_id", columns="player_id", values="minutes", aggfunc="sum").reindex(order.game_id).fillna(0.0)
        S = l.pivot_table(index="game_id", columns="player_id", values="started", aggfunc="max").reindex(order.game_id)
        prior = M.shift(1).rolling(10, min_periods=3).mean()
        played = (M > 0)
        miss = (prior >= 15) & ~played
        absent = (prior.where(miss, 0)).sum(axis=1)
        last_idx = played.astype(float).replace(0, np.nan).mul(np.arange(len(M)), axis=0).ffill().shift(1)
        st5 = S.shift(1).rolling(5, min_periods=1).mean()
        cnt = played.cumsum().shift(1).fillna(0)
        for g in M.index:
            i = gi[g]
            for pid in M.columns[played.loc[g].values]:
                li = last_idx.at[g, pid]
                rows.append((g, pid, tid, float(st5.at[g, pid]) if pd.notna(st5.at[g, pid]) else 0.0,
                             int(cnt.at[g, pid]), int(pd.notna(li) and i - li > 3), float(absent.at[g])))
    return pd.DataFrame(rows, columns=["game_id", "player_id", "team_id", "start_rate5", "n_prior_team", "flag_return_f", "absent_rot_min_f"])


FEAT = None


def with_features(P):
    global FEAT
    if FEAT is None:
        FEAT = pregame_features()
    P = P.merge(FEAT, on=["game_id", "player_id"], how="left")
    P["exp_starter"] = (P.start_rate5 >= 0.5).astype(int)
    P["n_games"] = P.n_prior_team.fillna(0)
    P["flag_return"] = P.flag_return_f.fillna(0)
    P["absent_rot_min"] = P.absent_rot_min_f.fillna(0)
    return P


# ------------------------------------------------------------------ A. minutes audit by profile
def minutes_audit(season):
    P = with_features(load("props", season).drop(columns=["exp_starter", "n_games", "flag_return", "absent_rot_min", "team_id"], errors="ignore"))
    m = P[P.market == "PTS"].copy()
    gm = G.set_index("game_id")
    m["blowout"] = m.game_id.map(lambda g: abs(gm.at[g, "home_score"] - gm.at[g, "away_score"]) >= 20 if g in gm.index else False)
    m["bench_rank"] = m[m.exp_starter == 0].groupby(["game_id", "team_id"]).minutes_pred.rank(ascending=False)
    def prof(x):
        if x.flag_return:
            return "regresa de lesión"
        if x.n_games <= 25:
            return "pocas muestras con el equipo (<=25 juegos)"
        if x.exp_starter == 1:
            return "titular"
        if x.bench_rank == 1:
            return "sexto hombre"
        if x.minutes_pred >= 15:
            return "suplente habitual (>=15 min)"
        return "rotación variable (<15 min)"
    m["profile"] = m.apply(prof, axis=1)
    m["err"] = m.minutes_pred - m.minutes_actual
    out = {}
    for name, d in list(m.groupby("profile")) + [("partidos con blowout (|margen|>=20)", m[m.blowout]), ("todos", m)]:
        out[name] = {"n": int(len(d)), "mae": r(d.err.abs().mean(), 2), "bias_pred_minus_actual": r(d.err.mean(), 2),
                     "extreme_err_gt15_share": r((d.err.abs() > 15).mean(), 3),
                     "share_of_total_abs_error": r(d.err.abs().sum() / m.err.abs().sum(), 3)}
    # blowout effect by role
    for role, sel in (("titulares en blowout", (m.exp_starter == 1) & m.blowout), ("suplentes en blowout", (m.exp_starter == 0) & m.blowout)):
        d = m[sel]
        out[role] = {"n": int(len(d)), "mae": r(d.err.abs().mean(), 2), "bias_pred_minus_actual": r(d.err.mean(), 2)}
    return out, m


def minutes_challenger(m):
    """MINUTES_PROFILE_RECAL: per-profile multiplicative recalibration of predicted minutes, learned
    walk-forward (strictly earlier dates of the same season, >= 30 cases, shrunk toward 1)."""
    m = m.sort_values("date").reset_index(drop=True)
    out = {}
    for prof in ("regresa de lesión", "sexto hombre", "titular", "suplente habitual (>=15 min)"):
        sel = (m.profile == prof).values
        adj = m.minutes_pred.values.copy()
        for dt in sorted(m.date.unique()):
            tr, cur = sel & (m.date < dt).values, sel & (m.date == dt).values
            if cur.sum() == 0 or tr.sum() < 30:
                continue
            k = (m.minutes_actual[tr].sum() + 30) / (m.minutes_pred[tr].sum() + 30)
            adj[cur] = m.minutes_pred.values[cur] * k
        b = np.abs(m.minutes_pred[sel] - m.minutes_actual[sel]).values
        e = np.abs(adj[sel] - m.minutes_actual[sel].values)
        d = b - e
        bs = [np.mean(np.random.default_rng(i).choice(d, len(d))) for i in range(1000)]
        out[prof] = {"n": int(sel.sum()), "mae_champion": r(b.mean(), 2), "mae_challenger": r(e.mean(), 2),
                     "gain_ci95": [r(np.percentile(bs, 2.5), 2), r(np.percentile(bs, 97.5), 2)]}
    return out


# ------------------------------------------------------------------ B. props error decomposition (oracle experiments)
def prior_rates(logs):
    """Walk-forward per player: exponentially weighted prior FG2%, FG3%, FT% (strictly before each game)."""
    l = logs.sort_values(["player_id", "game_date_et"]).copy()
    l["fg2a"], l["fg2m"] = l.fga - l.fg3a, l.fgm - l.fg3m
    out = []
    for pid, d in l.groupby("player_id"):
        cs = d[["fg2a", "fg2m", "fg3a", "fg3m", "fta", "ftm"]].cumsum().shift(1).fillna(0)
        p2 = (cs.fg2m + 0.53 * 60) / (cs.fg2a + 60)
        p3 = (cs.fg3m + 0.36 * 60) / (cs.fg3a + 60)
        pf = (cs.ftm + 0.78 * 30) / (cs.fta + 30)
        out.append(pd.DataFrame({"game_id": d.game_id.values, "player_id": pid, "p2": p2.values, "p3": p3.values, "pf": pf.values}))
    return pd.concat(out)


def decomposition(season, pr):
    P = load("props", season)
    box = L[["game_id", "player_id", "fga", "fg3a", "fta", "fgm", "fg3m", "ftm", "pts"]]
    res = {}
    for mk in ("PTS", "REB", "AST", "3PM"):
        d = P[(P.market == mk) & (P.minutes_pred >= 3) & (P.minutes_actual > 0)].copy()
        base = (d["mean"] - d.actual).abs()
        orc_min = (d["mean"] * d.minutes_actual / d.minutes_pred - d.actual).abs()
        row = {"n": int(len(d)), "mae_champion": r(base.mean()), "mae_oracle_minutes": r(orc_min.mean()),
               "share_of_error_from_minutes": r(1 - orc_min.mean() / base.mean())}
        if mk == "PTS":
            d = d.merge(box, on=["game_id", "player_id"]).merge(pr, on=["game_id", "player_id"])
            fg2a = d.fga - d.fg3a
            orc_opp = fg2a * 2 * d.p2 + d.fg3a * 3 * d.p3 + d.fta * d.pf
            e_opp = (orc_opp - d.actual).abs()
            # irreducible make/miss noise given the actual attempts (binomial, MAE ~ 0.8 sd)
            sd = np.sqrt(fg2a * 4 * d.p2 * (1 - d.p2) + d.fg3a * 9 * d.p3 * (1 - d.p3) + d.fta * d.pf * (1 - d.pf))
            row.update({"mae_oracle_minutes_and_attempts": r(e_opp.mean()),
                        "expected_mae_pure_shooting_luck": r((0.8 * sd).mean()),
                        "decomposition_share": {
                            "minutes": r(1 - orc_min.mean() / base.mean()),
                            "opportunities (shot volume given minutes)": r((orc_min.mean() - e_opp.mean()) / base.mean()),
                            "efficiency skill misestimation": r(max(0.0, e_opp.mean() - (0.8 * sd).mean()) / base.mean()),
                            "unpredictable shooting variance": r(min(e_opp.mean(), (0.8 * sd).mean()) / base.mean())}})
        res[mk] = row
    return res


def injury_effects(m_min, season):
    """Do teammates of absent important players get the extra minutes/points the model predicted?"""
    P = m_min
    out = {}
    for name, sel in (("equipo con >=60 min de rotación ausentes", P.absent_rot_min >= 60),
                      ("equipo con 25-60 min ausentes", (P.absent_rot_min >= 25) & (P.absent_rot_min < 60)),
                      ("equipo sin ausencias relevantes (<25)", P.absent_rot_min < 25)):
        d = P[sel]
        mm = d[d.market == "PTS"]
        out[name] = {"n_player_games": int(len(mm)),
                     "minutes_bias_pred_minus_actual": r((mm.minutes_pred - mm.minutes_actual).mean(), 2),
                     "minutes_mae": r((mm.minutes_pred - mm.minutes_actual).abs().mean(), 2),
                     "pts_bias": r((mm["mean"] - mm.actual).mean(), 2), "pts_mae": r((mm["mean"] - mm.actual).abs().mean(), 2)}
    return out


# ------------------------------------------------------------------ C. game totals: possessions vs efficiency
def totals(season):
    g = load("games", season).merge(G[["game_id", "home_team_id", "away_team_id", "home_poss", "away_poss", "periods", "home_score", "away_score"]], on="game_id")
    rows = []
    for dt, d in g.groupby("date"):
        Lb = L[L.game_date_et < dt]
        R = OpportunityRates(Lb, G, dt)
        R15 = OpportunityRates(Lb, G, dt, RatesConfig(k_env_games=15))
        for x in d.itertuples():
            pace = R.expected_pace(int(x.home_team_id), int(x.away_team_id))
            p15 = R15.expected_pace(int(x.home_team_id), int(x.away_team_id))
            th, ta = R.team(int(x.home_team_id))["pace48"], R.team(int(x.away_team_id))["pace48"]
            lgw = float(np.mean([v["pace48"] for v in R.T.values()]))
            rows.append({"game_id": x.game_id, "pace_v2": th * ta / lgw, "pace_pred": pace, "env_pace": R.env_pace, "env_eff": R.env_eff,
                         "adj_k15": (R15.env_pace * R15.env_eff) / (R.env_pace * R.env_eff), "home_eff": R.home_eff})
    P = g.merge(pd.DataFrame(rows), on="game_id")
    gmin = 48 + 5 * (P.periods - 4)
    P["pace_act"] = (P.home_poss + P.away_poss) / 2 * 48 / gmin
    P["ppp_act"] = P.actual_total / (P.home_poss + P.away_poss)
    # model-implied efficiency at regulation pace (the sim adds OT time on top: compare regulation-equivalent)
    P["exp_min"] = 48  # pace48 * game minutes; OT share in sims ~6% adds ~0.3 min on average
    P["ppp_pred"] = P.model_total / (2 * P.pace_pred * 1.006)
    P["total_reg_act"] = P.actual_total * 48 / gmin
    bias = (P.model_total - P.actual_total)
    lp = np.log(P.pace_pred * 1.006 / (P.pace_act * gmin / 48))
    le = np.log(P.ppp_pred / P.ppp_act)
    out = {"n": int(len(P)), "total_bias_pred_minus_actual": r(bias.mean(), 2),
           "pace_pred_mean": r(P.pace_pred.mean(), 2), "pace_actual_mean_per48": r(P.pace_act.mean(), 2),
           "ppp_pred_mean": r(P.ppp_pred.mean(), 4), "ppp_actual_mean": r(P.ppp_act.mean(), 4),
           "bias_share_possessions": r(lp.mean() / (lp.mean() + le.mean()) if (lp.mean() + le.mean()) else None, 3),
           "bias_points_from_possessions": r(P.model_total.mean() * lp.mean(), 2),
           "bias_points_from_efficiency": r(P.model_total.mean() * le.mean(), 2),
           "overtime_games_share": r((P.periods > 4).mean(), 3),
           "bias_regulation_only": r(bias[P.periods == 4].mean(), 2),
           "env_eff_mean": r(P.env_eff.mean(), 4), "env_pace_mean": r(P.env_pace.mean(), 4),
           "home_bias": r(((P.model_total + P.model_margin) / 2 - P.home_score).mean(), 2),
           "away_bias": r(((P.model_total - P.model_margin) / 2 - P.away_score).mean(), 2)}
    # early season vs later
    P["month"] = P.date.str[:7]
    out["bias_by_month"] = {k: {"n": int(len(v)), "bias": r((v.model_total - v.actual_total).mean(), 2)} for k, v in P.groupby("month")}
    return out, P


def team_totals(P):
    rows = []
    for side in ("home", "away"):
        pred = (P.model_total + (1 if side == "home" else -1) * P.model_margin) / 2
        rows.append(pd.DataFrame({"team": P[f"{side}_team_id"], "err": pred - P[f"{side}_score"], "side": side, "pred": pred}))
    T = pd.concat(rows)
    by = T.groupby("team").err.agg(["mean", "count"])
    q = T.pred.quantile([.25, .75]).values
    return {"by_team_bias_range": [r(by["mean"].min(), 1), r(by["mean"].max(), 1)],
            "teams_with_abs_bias_gt5": int((by["mean"].abs() > 5).sum()),
            "strong_offense_bias (pred top quartile)": r(T[T.pred >= q[1]].err.mean(), 2),
            "weak_offense_bias (pred bottom quartile)": r(T[T.pred <= q[0]].err.mean(), 2),
            "home_bias": r(T[T.side == "home"].err.mean(), 2), "away_bias": r(T[T.side == "away"].err.mean(), 2)}


# ------------------------------------------------------------------ D. game challengers (walk-forward, out of sample)
def challengers(P_all):
    P = P_all.sort_values("date").reset_index(drop=True)
    y_total, y_margin = P.actual_total.values, P.actual_margin.values
    res = {}
    # 1) TOTAL_ENV_K15: faster league-environment update (root cause candidate)
    t1 = P.model_total * P.adj_k15
    # 2) TOTAL_ONLINE_BIAS: shrunken running mean of past residuals, current season only (k=40 games)
    t2 = np.empty(len(P))
    for i, row in P.iterrows():
        past = P[(P.date < row.date) & (P.season == row.season)]
        b = (past.model_total - past.actual_total).sum() / (len(past) + 40) if len(past) else 0.0
        t2[i] = row.model_total - b
    # 2b) PACE_NO_DOUBLE_COUNT: team pace factors already carry the current season (recency weights);
    #     the champion multiplies by the league-environment factor again and normalises by an unweighted
    #     multi-season mean. Challenger: pace = home*away / mean(team factors), efficiency unchanged.
    t3 = P.model_total * P.pace_v2 / P.pace_pred
    # 3) MARGIN_STRETCH: margin * k, k fitted on strictly earlier dates (min 100 games)
    m3 = np.full(len(P), np.nan)
    for dt in sorted(P.date.unique()):
        tr = P[P.date < dt]
        if len(tr) < 100:
            continue
        k = float((tr.model_margin * tr.actual_margin).sum() / (tr.model_margin ** 2).sum())
        m3[(P.date == dt).values] = P.model_margin[(P.date == dt).values] * min(max(k, 0.8), 1.3)
    for season in ("2025-26", "2026-27"):
        s = (P.season == season).values
        base_t = np.abs(P.model_total[s] - y_total[s])
        for name, t in (("TOTAL_ENV_K15", t1), ("TOTAL_ONLINE_BIAS", t2), ("PACE_NO_DOUBLE_COUNT", t3)):
            e = np.abs(t[s] - y_total[s])
            d = base_t - e
            bs = [np.mean(np.random.default_rng(i).choice(d, len(d))) for i in range(1000)]
            res.setdefault(name, {})[season] = {"n": int(s.sum()), "mae_champion": r(base_t.mean(), 2), "mae_challenger": r(e.mean(), 2),
                                                "bias_challenger": r((t[s] - y_total[s]).mean(), 2),
                                                "gain_ci95": [r(np.percentile(bs, 2.5), 2), r(np.percentile(bs, 97.5), 2)]}
        ok = s & ~np.isnan(m3)
        b = np.abs(P.model_margin[ok] - y_margin[ok]); e = np.abs(m3[ok] - y_margin[ok]); d = b - e
        bs = [np.mean(np.random.default_rng(i).choice(d, len(d))) for i in range(1000)] if len(d) else [0]
        res.setdefault("MARGIN_STRETCH", {})[season] = {"n": int(ok.sum()), "mae_champion": r(b.mean(), 2), "mae_challenger": r(e.mean(), 2),
                                                        "gain_ci95": [r(np.percentile(bs, 2.5), 2), r(np.percentile(bs, 97.5), 2)]}
    return res


# ------------------------------------------------------------------ E. teammate correlations: real vs simulated
def correlations():
    out = {}
    for f, season in (("mc4_2526.pkl", "2025-26"), ("mc4_2627.pkl", "2026-27")):
        p = f"{D}/{f}"
        if not os.path.exists(p):
            continue
        d = pd.read_pickle(p)
        C = d["corr"]
        out[season] = {"simulated_mean_corr": C.groupby("cfg")[["pts_reb", "pts_ast", "reb_ast", "min_pts", "teammates_pts"]].mean().round(3).to_dict("index")}
    # real within-team correlation of points residuals (vs player season-to-date mean) — data, not model
    l = L.sort_values(["player_id", "game_date_et"]).copy()
    l["exp_pts"] = l.groupby("player_id").pts.transform(lambda s: s.shift(1).expanding(5).mean())
    l["exp_min"] = l.groupby("player_id").minutes.transform(lambda s: s.shift(1).expanding(5).mean())
    l = l.dropna(subset=["exp_pts"])
    l = l[l.exp_min >= 20]
    l["res"] = l.pts - l.exp_pts
    l["mres"] = l.minutes - l.exp_min
    pairs, mpairs = [], []
    for (g, t), d in l.groupby(["game_id", "team_id"]):
        v, mv = d.res.values, d.mres.values
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                pairs.append((v[i], v[j])); mpairs.append((mv[i], mv[j]))
    pr, mp = np.array(pairs), np.array(mpairs)
    out["real_teammates_pts_corr (rotation players >=20 mpg)"] = r(np.corrcoef(pr[:, 0], pr[:, 1])[0, 1])
    out["real_teammates_minutes_corr"] = r(np.corrcoef(mp[:, 0], mp[:, 1])[0, 1])
    # opposing teams: game-script correlation of team scores (regulation games)
    g = G[(G.periods == 4)].dropna(subset=["home_score"])
    out["real_home_away_score_corr"] = r(np.corrcoef(g.home_score, g.away_score)[0, 1])
    return out


def main():
    rep = {"note": "Backtests reconstruidos walk-forward (no prospectivos). Resultados reales usados solo para medir error."}
    pr = prior_rates(L)
    Ps = []
    for season in SEASONS:
        a, m = minutes_audit(season)
        rep.setdefault("minutes_audit", {})[season] = a
        rep.setdefault("minutes_challenger_profile_recal", {})[season] = minutes_challenger(m)
        rep.setdefault("props_error_decomposition", {})[season] = decomposition(season, pr)
        rep.setdefault("injury_context", {})[season] = injury_effects(m, season)
        t, P = totals(season)
        P["season"] = season
        rep.setdefault("totals_decomposition", {})[season] = t
        rep.setdefault("team_totals", {})[season] = team_totals(P)
        Ps.append(P)
    rep["game_challengers"] = challengers(pd.concat(Ps, ignore_index=True))
    rep["correlations"] = correlations()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rep, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
