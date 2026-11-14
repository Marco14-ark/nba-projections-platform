"""NBA_PLAYER_PROPS_V1 + NBA_GAME_MARKETS_V1 — everything is read from ONE joint simulation.

Combos (PRA, PR, PA, RA) are summed per simulation, never by adding independent distributions
(spec §22). Game markets come from the same simulated scores as the player props (spec §27).
No sportsbook input is used anywhere here; `price_line` only converts OUR probabilities into
fair odds, and `compare_to_book` (Phase 4) is the only place a book price is accepted.
"""
from __future__ import annotations

import numpy as np

from .game_sim import GameSim

PROP_MARKETS = {
    "PTS": ("pts",), "REB": ("reb",), "AST": ("ast",), "3PM": ("fg3m",), "FGA": ("fga",), "FGM": ("fgm",),
    "3PA": ("fg3a",), "FTA": ("fta",), "FTM": ("ftm",), "TOV": ("tov",), "MIN": ("minutes",),
    "PRA": ("pts", "reb", "ast"), "PR": ("pts", "reb"), "PA": ("pts", "ast"), "RA": ("reb", "ast"),
}


def american(p: float) -> float | None:
    if p <= 0 or p >= 1:
        return None
    return round(-100 * p / (1 - p), 1) if p >= 0.5 else round(100 * (1 - p) / p, 1)


def line_probs(x: np.ndarray, line: float) -> dict:
    return {"p_over": float(np.mean(x > line)), "p_under": float(np.mean(x < line)),
            "p_push": float(np.mean(x == line))}


def dist_summary(x: np.ndarray) -> dict:
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return {"mean": float(x.mean()), "median": float(q[2]), "p10": float(q[0]), "p25": float(q[1]),
            "p75": float(q[3]), "p90": float(q[4]), "sd": float(x.std())}


def player_samples(g: GameSim, pid: str, market: str) -> np.ndarray:
    for T in (g.home, g.away):
        if pid in T.players:
            k = T.players.index(pid)
            return sum(T.stats[s][:, k] for s in PROP_MARKETS[market]).astype(float)
    raise KeyError(pid)


def player_props(g: GameSim, markets=("PTS", "REB", "AST", "3PM", "PRA", "PR", "PA", "RA"),
                 lines: dict | None = None) -> list[dict]:
    """lines: optional {(pid, market): line}. Without a line we still publish the distribution
    and a model 'fair line' (median, half-point) — never presented as a sportsbook line."""
    out = []
    for side, T in (("home", g.home), ("away", g.away)):
        mins = T.stats["minutes"]
        for k, pid in enumerate(T.players):
            for mk in markets:
                x = sum(T.stats[s][:, k] for s in PROP_MARKETS[mk]).astype(float)
                d = dist_summary(x)
                fair = np.floor(d["median"]) + 0.5
                rec = {"player_id": pid, "team_id": T.team_id, "side": side, "market": mk, **d,
                       "minutes_mean": float(mins[:, k].mean()), "minutes_median": float(np.median(mins[:, k])),
                       "p_dnp": float(np.mean(mins[:, k] <= 0)), "model_fair_line": float(fair),
                       **{f"fair_{k2}": v for k2, v in line_probs(x, fair).items()}}
                if lines and (pid, mk) in lines:
                    ln = lines[(pid, mk)]
                    rec.update({"line": ln, **line_probs(x, ln)})
                out.append(rec)
    return out


def game_markets(g: GameSim, spreads=None, totals=None) -> dict:
    m, t = g.margin.astype(float), g.total.astype(float)
    ties = float(np.mean(m == 0))
    p_home = float(np.mean(m > 0) + 0.5 * ties)
    res = {"home_team_id": g.home.team_id, "away_team_id": g.away.team_id,
           "projected_home": float(g.home.score.mean()), "projected_away": float(g.away.score.mean()),
           "median_home": float(np.median(g.home.score)), "median_away": float(np.median(g.away.score)),
           "p_home_win": p_home, "p_away_win": 1 - p_home, "fair_ml_home": american(p_home),
           "fair_ml_away": american(1 - p_home), "p_tie_end_of_sim": ties, "p_overtime": float(np.mean(g.n_ot > 0)),
           "margin": dist_summary(m), "total": dist_summary(t),
           "team_total_home": dist_summary(g.home.score.astype(float)),
           "team_total_away": dist_summary(g.away.score.astype(float)),
           "model_fair_spread_home": float(-np.round(np.median(m) * 2) / 2),
           "model_fair_total": float(np.round(np.median(t) * 2) / 2)}
    sp = spreads if spreads is not None else np.arange(-20.5, 21.0, 1.0)
    res["alt_spreads_home"] = [{"spread": float(s), "p_cover": float(np.mean(m + s > 0)),
                                "p_push": float(np.mean(m + s == 0))} for s in sp]
    tl = totals if totals is not None else np.round(np.median(t)) + np.arange(-15.5, 16.0, 1.0)
    res["alt_totals"] = [{"total": float(x), **line_probs(t, x)} for x in tl]
    return res


def compare_to_book(p_model: float, price_side: int, price_other: int | None = None) -> dict:
    """Phase 4 hook. Needs a REAL book price; returns EV of the side at that price, plus no-vig
    probability when both sides are given. Never called with invented prices."""
    def implied(a: int) -> float:
        return 100 / (a + 100) if a > 0 else -a / (-a + 100)
    dec = 1 + (price_side / 100 if price_side > 0 else 100 / -price_side)
    out = {"p_model": p_model, "implied": implied(price_side), "ev_per_unit": p_model * dec - 1}
    if price_other is not None:
        a, b = implied(price_side), implied(price_other)
        out["no_vig"] = a / (a + b)
        out["gap"] = p_model - out["no_vig"]
    return out
