"""Walk-forward validation of the joint simulation: player props + game markets (spec §47-52).

For every test date: minutes engine + opportunity rates are fitted on games strictly before it;
availability = the game's active list (proxy for the official inactive list known pre-tip);
starters = EXPECTED by the model (no lineup leakage).

Player markets are scored on players who actually played, using the model distribution
conditional on playing (books void DNP props). Without free historical sportsbook lines,
probability calibration is tested on PSEUDO-LINES: line = floor(player's last-10 average)+0.5,
a line built only from pre-game information, scored for both the model and the baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..lineups.resolve import expected_starters
from ..minutes.engine import MinutesEngine
from ..minutes.validate import crps_samples
from .game_sim import GameSimulator, TeamInput, accounting_check
from .rates import OpportunityRates

MARKETS = {"PTS": ["pts"], "REB": ["reb"], "AST": ["ast"], "3PM": ["fg3m"], "PRA": ["pts", "reb", "ast"]}
# extra pseudo-line offsets (calibration across the probability range); all lines end in .5
LINE_OFFSETS = {"PTS": (-4, -2, 2, 4), "REB": (-2, -1, 1, 2), "AST": (-2, -1, 1, 2), "3PM": (-1, 1), "PRA": (-6, -3, 3, 6)}


def _stat(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return sum(df[c].to_numpy(float) for c in cols)


def walk_forward(logs: pd.DataFrame, games: pd.DataFrame, dates: list[str], n_sims: int = 4000,
                 seed: int = 17, last_n: int = 10, progress: bool = False, anchor: str = "bottom_up",
                 minutes_cfg=None, rates_cfg=None, rating_blend: tuple | None = None, segments=None,
                 joint=None, markets: dict | None = None):
    """rating_blend=(w_v1, w_v2): anchor the script on w_v1*point-diff + w_v2*lineup-RAPM rating
    (NBA_TEAM_RATING_V2) computed from available players and pregame minutes proxies."""
    from ..rating.team_rating import LineupRating, PointDiffRating, RatingConfig, rest_days
    rd = rest_days(games).set_index(["game_id", "team_id"])["rest"] if rating_blend else None
    lr, lr_day = None, None
    prow, grow = [], []
    reason = logs["not_playing_reason"].fillna("").astype(str)
    inactive = reason.str.startswith("INACTIVE")
    played_all = logs[logs["played"] == 1].sort_values("game_date_et")
    for di, d in enumerate(dates):
        hist = logs[logs["game_date_et"] < d]
        ghist = games[games["game_date_et"] < d]
        eng = MinutesEngine(hist, ghist, d, minutes_cfg)
        rates = OpportunityRates(hist, ghist, d, rates_cfg)
        if rating_blend and (lr is None or (pd.Timestamp(d) - pd.Timestamp(lr_day)).days >= 7):
            lr, lr_day = LineupRating(segments, games, d, RatingConfig(ridge=1500.0)), d
        pdr = PointDiffRating(games, d) if rating_blend else None
        avail_hist = hist[~hist["not_playing_reason"].fillna("").astype(str).str.startswith("INACTIVE")]
        mproxy = avail_hist.sort_values("game_date_et").groupby("player_id").tail(10).groupby("player_id")["minutes"].mean() \
            if rating_blend else None
        sim = GameSimulator(eng, rates, joint)
        ph = played_all[played_all["game_date_et"] < d]
        last = {p: g.tail(last_n) for p, g in ph.groupby("player_id")}
        tl = _team_last(ghist, last_n)
        for g in games[games["game_date_et"] == d].itertuples():
            day = logs[(logs["game_id"] == g.game_id) & ~inactive]
            ti = []
            for tid in (g.home_team_id, g.away_team_id):
                av = list(day.loc[day["team_id"] == tid, "player_id"])
                ti.append(TeamInput(int(tid), av, expected_starters(av, eng.start_scores(int(tid), av))))
            am = None
            if rating_blend:
                mins = [{p: float(mproxy.get(p, 5.0)) for p in t_.available} for t_ in ti]
                pr = lr.predict(mins[0], mins[1], rates.expected_pace(int(g.home_team_id), int(g.away_team_id)),
                                rd.get((g.game_id, g.home_team_id), 2), rd.get((g.game_id, g.away_team_id), 2))
                am = rating_blend[0] * pdr.margin(int(g.home_team_id), int(g.away_team_id)) + rating_blend[1] * pr["margin"]
            s = sim.simulate(ti[0], ti[1], n=n_sims, seed=seed + di, anchor=anchor, anchor_margin=am)
            acc = all(accounting_check(s).values())
            m = s.margin.astype(float)
            mu_r = eng.ratings.get(int(g.home_team_id), 0) - eng.ratings.get(int(g.away_team_id), 0) + eng.home_edge
            th, ta = tl.get(int(g.home_team_id)), tl.get(int(g.away_team_id))
            base_total = (th[0] + th[1] + ta[0] + ta[1]) / 2 if th and ta else np.nan
            am = g.home_score - g.away_score
            grow.append({"date": d, "game_id": g.game_id, "actual_margin": am, "actual_total": g.home_score + g.away_score,
                         "model_margin": m.mean(), "model_total": s.total.mean(), "p_home": float(np.mean(m > 0) + 0.5 * np.mean(m == 0)),
                         "crps_margin": crps_samples(m[::2], am), "crps_total": crps_samples(s.total[::2].astype(float), g.home_score + g.away_score),
                         "cov80_margin": float(np.percentile(m, 10) <= am <= np.percentile(m, 90)),
                         "cov80_total": float(np.percentile(s.total, 10) <= g.home_score + g.away_score <= np.percentile(s.total, 90)),
                         "base_margin": mu_r, "base_p_home": float(norm.cdf(mu_r / eng.sd_margin)), "base_total": base_total,
                         "accounting_ok": acc, "ess": s.meta["ess"]})
            for T, tin in ((s.home, ti[0]), (s.away, ti[1])):
                act = day[(day["team_id"] == T.team_id) & (day["played"] == 1)].set_index("player_id")
                rot = eng.rot_players.get(T.team_id, {})
                scope = set(eng._absent_scope(T.team_id))
                abs_min = float(sum(m for q, m in rot.items() if q in scope and q not in tin.available))
                for k, pid in enumerate(T.players):
                    if pid not in act.index:
                        continue
                    mins = T.stats["minutes"][:, k]
                    keep = mins > 0
                    if keep.sum() < 50:
                        continue
                    L = last.get(pid)
                    for mk, cols in (markets or MARKETS).items():
                        x = sum(T.stats[c][:, k] for c in cols).astype(float)[keep]
                        y = float(sum(act.loc[pid, c] for c in cols))
                        prm = eng.player_params(pid, T.team_id, tin.available, pid in tin.starters)
                        rec = {"date": d, "game_id": g.game_id, "player_id": pid, "market": mk, "actual": y,
                               "minutes_actual": float(act.loc[pid, "minutes"]), "minutes_pred": float(mins[keep].mean()),
                               "minutes_sd": float(mins.std()), "p_dnp_pred": float(np.mean(mins <= 0)),
                               "exp_starter": int(pid in tin.starters), "absent_rot_min": abs_min,
                               "n_games": int(rates.player(pid).get("games", 0)),
                               "flag_return": int(any(f.startswith("RETURN") for f in prm.flags)),
                               "flag_new_team": int("NEW_TEAM" in prm.flags),
                               "mean": x.mean(), "median": float(np.median(x)), "p10": np.percentile(x, 10),
                               "p90": np.percentile(x, 90), "crps": crps_samples(x[::2], y)}
                        if L is not None and len(L) >= 5:
                            b = _stat(L, cols)
                            line = np.floor(b.mean()) + 0.5
                            rec.update({"b_mean": b.mean(), "b_p10": np.percentile(b, 10), "b_p90": np.percentile(b, 90),
                                        "b_crps": crps_samples(b, y), "line": line, "over": float(y > line),
                                        "p_over": float(np.mean(x > line)),
                                        "b_p_over": float(((b > line).sum() + 1) / (len(b) + 2))})
                            for o in LINE_OFFSETS.get(mk, ()):
                                ln = line + o
                                if ln > 0:
                                    rec[f"p_over_{o:+d}"] = float(np.mean(x > ln))
                                    rec[f"over_{o:+d}"] = float(y > ln)
                        prow.append(rec)
        if progress:
            print(f"  {d}: {len(grow)} games", flush=True)
    return pd.DataFrame(prow), pd.DataFrame(grow)


def _team_last(games: pd.DataFrame, n: int) -> dict[int, tuple[float, float]]:
    rows = pd.concat([
        pd.DataFrame({"t": games["home_team_id"], "d": games["game_date_et"], "f": games["home_score"], "a": games["away_score"]}),
        pd.DataFrame({"t": games["away_team_id"], "d": games["game_date_et"], "f": games["away_score"], "a": games["home_score"]})])
    out = {}
    for t, g in rows.sort_values("d").groupby("t"):
        g = g.tail(n)
        out[int(t)] = (float(g["f"].mean()), float(g["a"].mean()))
    return out


def _brier(p, y):
    return float(np.mean((p - y) ** 2))


def _logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def summarize_props(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mk, d in P.groupby("market"):
        b = d.dropna(subset=["b_mean"])
        rows.append({"market": mk, "n": len(d), "n_paired": len(b),
                     "mae_model": float(np.mean(np.abs(b["mean"] - b["actual"]))),
                     "mae_base": float(np.mean(np.abs(b["b_mean"] - b["actual"]))),
                     "crps_model": float(b["crps"].mean()), "crps_base": float(b["b_crps"].mean()),
                     "cov80_model": float(np.mean((b["actual"] >= b["p10"]) & (b["actual"] <= b["p90"]))),
                     "cov80_base": float(np.mean((b["actual"] >= b["b_p10"]) & (b["actual"] <= b["b_p90"]))),
                     "brier_model": _brier(b["p_over"], b["over"]), "brier_base": _brier(b["b_p_over"], b["over"]),
                     "logloss_model": _logloss(b["p_over"], b["over"]), "logloss_base": _logloss(b["b_p_over"], b["over"]),
                     "bias_model": float(np.mean(b["mean"] - b["actual"]))})
    return pd.DataFrame(rows)


def reliability(P: pd.DataFrame, col: str = "p_over", bins=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)) -> pd.DataFrame:
    d = P.dropna(subset=[col]).copy()
    d["bin"] = pd.cut(d[col], bins, include_lowest=True)
    return d.groupby(["market", "bin"], observed=True).agg(n=("over", "size"), predicted=(col, "mean"),
                                                            observed=("over", "mean")).reset_index()


def summarize_games(G: pd.DataFrame) -> dict:
    d = G.dropna(subset=["base_total"])
    hw = (d["actual_margin"] > 0).astype(float)
    return {"n_games": len(d),
            "margin_mae_model": float(np.mean(np.abs(d["model_margin"] - d["actual_margin"]))),
            "margin_mae_base": float(np.mean(np.abs(d["base_margin"] - d["actual_margin"]))),
            "total_mae_model": float(np.mean(np.abs(d["model_total"] - d["actual_total"]))),
            "total_mae_base": float(np.mean(np.abs(d["base_total"] - d["actual_total"]))),
            "total_bias_model": float(np.mean(d["model_total"] - d["actual_total"])),
            "brier_home_model": _brier(d["p_home"], hw), "brier_home_base": _brier(d["base_p_home"], hw),
            "logloss_home_model": _logloss(d["p_home"], hw), "logloss_home_base": _logloss(d["base_p_home"], hw),
            "cov80_margin": float(d["cov80_margin"].mean()), "cov80_total": float(d["cov80_total"].mean()),
            "crps_margin": float(d["crps_margin"].mean()), "crps_total": float(d["crps_total"].mean()),
            "accounting_ok_rate": float(d["accounting_ok"].mean())}
