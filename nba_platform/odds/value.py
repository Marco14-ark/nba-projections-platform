"""Market value engine.

Model probabilities come ONLY from the frozen Monte Carlo distributions; book prices are used to
compute implied / no-vig probabilities, edge and EV, never to change the model's forecast.

Decision ladder (first match wins):
  NO_MARKET_DATA  no real price for the side
  NO_BET          any blocking reason (market not validated against real lines, stale odds,
                  game not pregame, late status conflict, unmatched player, model flag, conservative EV <= 0)
  VALUE           conservative EV > 0 and the market passed validation
"""
from __future__ import annotations
import math
from datetime import datetime

import numpy as np

from .pricing import (american_to_decimal, ev_per_unit, fair_american, fair_american_with_push, implied_prob,
                      mc_se, no_vig_pair)

STALE_MINUTES = 90


def dist_probs(lo: int, counts, line: float) -> dict:
    """P(X>line), P(X<line), P(X==line) from exact integer counts."""
    c = np.asarray(counts, dtype=float)
    v = lo + np.arange(len(c))
    n = c.sum()
    return {"over": float(c[v > line].sum() / n), "under": float(c[v < line].sum() / n),
            "push": float(c[v == line].sum() / n), "n": int(n)}


def side_probs(market: str, side: str, line: float | None, dists: dict) -> dict | None:
    """Model win/push probability for one side of a market.
    dists: {'MARGIN':(lo,counts), 'TOTAL':..., 'HOME_PTS':..., 'AWAY_PTS':..., or player market}.
    Spread `line` is from the side's own perspective (home -3.5 => home covers if margin - 3.5 > 0)."""
    if market == "ML":
        lo, c = dists["MARGIN"]
        d = dist_probs(lo, c, 0)
        tie = d["push"]          # simulated end-of-game ties are split (OT resolves them)
        p = d["over"] + tie / 2 if side == "home" else d["under"] + tie / 2
        return {"win": p, "push": 0.0, "n": d["n"]}
    if market == "SPREAD":
        lo, c = dists["MARGIN"]
        if side == "home":       # covers if margin + line > 0  <=> margin > -line
            d = dist_probs(lo, c, -line)
            return {"win": d["over"], "push": d["push"], "n": d["n"]}
        d = dist_probs(lo, c, line)   # away covers if -margin + line > 0 <=> margin < line
        return {"win": d["under"], "push": d["push"], "n": d["n"]}
    if market in ("TOTAL", "TEAM_TOTAL") or market in dists:
        key = market
        if market == "TEAM_TOTAL":
            key = "HOME_PTS" if side.startswith("home") else "AWAY_PTS"
        if key not in dists:
            return None
        lo, c = dists[key]
        d = dist_probs(lo, c, line)
        ou = side.split(":")[-1]
        return {"win": d[ou], "push": d["push"], "n": d["n"]}
    return None


def beta_calibrate(params: dict | None, market: str, q: float, line: float) -> float | None:
    """BETA_MKT calibration (validated walk-forward) — only for half-point lines of calibrated markets."""
    if not params or abs(line * 2 - round(line * 2)) > 1e-9 or abs(line - math.floor(line) - 0.5) > 1e-9:
        return None
    m = (params.get("markets") or {}).get(market)
    if not m:
        return None
    eps = params.get("eps", 1e-4)
    x = min(max(q, eps), 1 - eps)
    z = m["a"] * math.log(x) - m["b"] * math.log(1 - x) + m["c"]
    return 1 / (1 + math.exp(-z))


def evaluate(p_win: float, p_push: float, n_sims: int, price: int, price_other: int | None,
             uncertainty: float, blockers: list[str], age_min: float | None) -> dict:
    """Full value record for one priced side."""
    out = {"price": price, "decimal": round(american_to_decimal(price), 4), "implied": implied_prob(price),
           "p_model": p_win, "p_push": p_push, "fair_odds": fair_american_with_push(p_win, p_push)}
    q = 1 - p_push
    p_cond = p_win / q if q > 0 else None
    if price_other is not None:
        nv, nv_o, over = no_vig_pair(price, price_other)
        out.update({"no_vig": nv, "overround": over, "edge": (p_cond - nv) if p_cond is not None else None})
    else:
        out.update({"no_vig": None, "overround": None, "edge": None})
    out["ev"] = ev_per_unit(p_win, p_push, price)
    # conservative probability: subtract measured calibration error + 2 Monte Carlo standard errors
    pad = uncertainty + 2 * mc_se(p_win, n_sims)
    p_c = max(0.0, p_win - pad)
    out["p_conservative"] = p_c
    out["ev_conservative"] = ev_per_unit(p_c, p_push, price)
    out["odds_age_min"] = age_min
    b = list(blockers)
    if age_min is not None and age_min > STALE_MINUTES:
        b.append("STALE_ODDS")
    if out["ev_conservative"] <= 0:
        b.append("NO_CONSERVATIVE_EDGE")
    out["blockers"] = b
    out["decision"] = "VALUE" if not b else "NO_BET"
    return out


def age_minutes(ts: str | None, now: datetime) -> float | None:
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (now - t).total_seconds() / 60
