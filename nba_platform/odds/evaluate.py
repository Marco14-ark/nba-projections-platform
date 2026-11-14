"""Pair real quotes (over/under, home/away) and evaluate them against the frozen model distributions."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime

from .value import age_minutes, beta_calibrate, evaluate, side_probs

# Calibration uncertainty (absolute probability) used for the conservative EV, with its evidence.
UNCERTAINTY = {
    "ML": (0.042, "ECE moneyline walk-forward 2025-26 (n=363)"),
    "SPREAD": (0.04, "margin cov80 0.83 / no market-line calibration yet"),
    "TOTAL": (0.04, "total cov80 0.79 / no market-line calibration yet"),
    "TEAM_TOTAL": (0.05, "team totals: only mean error validated"),
    "PROPS_CAL": (0.01, "BETA_MKT walk-forward ECE 0.004 (PTS/REB/AST/3PM/PRA, half lines)"),
    "PROPS_RAW": (0.03, "uncalibrated market / integer line"),
}
OPPOSITE = {"over": "under", "under": "over", "home": "away", "away": "home"}


def _gate_blockers(gate: dict, market: str) -> list[str]:
    g = (gate or {}).get(market) or {}
    return [] if g.get("status") == "VALIDATED" else ["NOT_VALIDATED_VS_MARKET"]


def build(quotes: list[dict], game_dists: dict | None, player_dists: dict, cal: dict | None, gate: dict,
          now: datetime, pregame: bool, late_players: set, conditional: bool, player_names: dict) -> list[dict]:
    """quotes: current matched quotes for one game. Returns one row per priced side."""
    by = defaultdict(dict)
    for q in quotes:
        k = (q["provider"], q["book"], q["kind"], q["market"], q.get("team_side") or "", q.get("player_id") or "")
        by[k][q["side"]] = q
    rows = []
    for (prov, book, kind, market, ts, pid), sides in by.items():
        for side, q in sides.items():
            other = sides.get(OPPOSITE.get(side))
            # a two-way pair must be the same line (spread lines are opposite in sign)
            if other is not None:
                if market == "SPREAD" and (q["line"] is None or other["line"] is None or q["line"] != -other["line"]):
                    other = None
                elif market not in ("ML", "SPREAD") and other["line"] != q["line"]:
                    other = None
            blockers = []
            if q.get("match_status") not in ("MATCHED", "MATCHED_FUZZY"):
                blockers.append("UNMATCHED")
            if not pregame:
                blockers.append("NOT_PREGAME")
            if conditional:
                blockers.append("CONDITIONAL_FORECAST")
            if pid and pid in late_players:
                blockers.append("LATE_STATUS_CHANGE")
            if other is None:
                blockers.append("ONE_SIDED_PRICE")
            blockers += _gate_blockers(gate, market)
            if kind == "PLAYER":
                d = (player_dists or {}).get(pid) or {}
                if market not in d:
                    rows.append(_row(q, None, "NO_MODEL_DISTRIBUTION", player_names))
                    continue
                pr = side_probs(market, side, q["line"], {market: d[market]})
                p_raw = pr["win"]
                cal_p = None
                if pr["push"] == 0:
                    over_raw = side_probs(market, "over", q["line"], {market: d[market]})["win"]
                    co = beta_calibrate(cal, market, over_raw, q["line"])
                    if co is not None:
                        cal_p = co if side == "over" else 1 - co
                p_use = cal_p if cal_p is not None else p_raw
                unc, src = UNCERTAINTY["PROPS_CAL" if cal_p is not None else "PROPS_RAW"]
            else:
                if not game_dists:
                    rows.append(_row(q, None, "NO_MODEL_DISTRIBUTION", player_names))
                    continue
                s = f"{ts}:{side}" if market == "TEAM_TOTAL" else side
                pr = side_probs(market, s, q["line"], game_dists)
                p_raw, cal_p, p_use = pr["win"], None, pr["win"]
                unc, src = UNCERTAINTY[market]
            ev = evaluate(p_use, pr["push"], pr["n"], q["price"], other["price"] if other else None, unc, blockers,
                          age_minutes(q["captured_at"], now))
            ev.update({"p_model_raw": p_raw, "p_model_calibrated": cal_p, "uncertainty": unc, "uncertainty_source": src})
            rows.append(_row(q, ev, None, player_names))
    return rows


def _r(x, k=4):
    return None if x is None else round(float(x), k)


def _row(q, ev, missing, names):
    out = {"provider": q["provider"], "book": q["book"], "kind": q["kind"], "market": q["market"],
           "team_side": q.get("team_side") or None, "player_id": q.get("player_id"),
           "player": names.get(q.get("player_id")) or q.get("player_name"), "side": q["side"], "line": q["line"],
           "price": q["price"], "captured_at": q["captured_at"], "book_updated_at": q.get("book_updated_at"),
           "match_status": q.get("match_status")}
    if ev is None:
        out.update({"decision": "NO_BET", "blockers": [missing]})
        return out
    for k in ("implied", "no_vig", "edge", "ev", "ev_conservative", "p_model", "p_model_raw", "p_model_calibrated",
              "p_push", "p_conservative", "overround", "uncertainty"):
        out[k] = _r(ev.get(k))
    out["fair_odds"] = None if ev["fair_odds"] is None else round(ev["fair_odds"])
    out["odds_age_min"] = _r(ev["odds_age_min"], 0)
    out["uncertainty_source"] = ev["uncertainty_source"]
    out["decision"], out["blockers"] = ev["decision"], ev["blockers"]
    return out
