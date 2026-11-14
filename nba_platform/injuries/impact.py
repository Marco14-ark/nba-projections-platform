"""injury-impact-ranking-v1

Ranks the absences of a team for ONE game by the absent player's real role, using only box scores of
games played BEFORE the game. Purely a presentation/triage layer: the model itself always processes
every verified absence; this module only decides what deserves space on the page.

Role metrics (all from the player's recent played games with his current team, within 45 days):
  role_min     average minutes when available (last <=10 played games)
  start_rate   share of those games started
  usage        (FGA + 0.44*FTA + TOV) / team possessions on court
  prod36       (PTS + 1.5*AST + 1.2*REB) per 36 minutes
  team ranks   rank of role_min / usage / impact score inside the team's own recent rotation
Recency:
  games_missed consecutive team games missed before this game. >= LONG_ABSENCE -> the absence is
               already part of the team's normal rotation (not presented as a new loss).
Categories (adapted to each team through within-team ranks):
  CRITICAL  role_min >= 28 and (team #1 by impact score or usage >= 0.26), or role_min >= 24 as the
            team's #1 with usage >= 0.28 (stars on reduced minutes)
  HIGH      role_min >= 24, or start_rate >= 0.6 with role_min >= 18, or team top-3 by impact score with role_min >= 18
  MEDIUM    role_min >= 12 (rotation player)
  LOW       everything else (marginal minutes / no recent role)
MEDIUM is shown on the main view only when its absence is consequential for this game: the team is
already missing a CRITICAL/HIGH player, it is the team's only rotation player at his position group,
or the model gives a teammate >= 4 extra minutes for this game."""
from __future__ import annotations
import numpy as np
import pandas as pd

VERSION = "injury-impact-ranking-v1"
LONG_ABSENCE = 8
WINDOW_DAYS = 45
ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
STATUS_ORDER = {"OUT": 0, "SUSPENDED": 0, "NOT_WITH_TEAM": 0, "DOUBTFUL": 1, "QUESTIONABLE": 2, "GTD": 2, "PROBABLE": 3, "AVAILABLE": 4}


def pos_group(pos: str | None) -> str:
    p = (pos or "").upper()
    if p.startswith("C") or p == "F-C":
        return "BIG"
    if p.startswith("G"):
        return "GUARD"
    return "WING"


def team_roles(logs: pd.DataFrame, team_id: int, game_date: str) -> tuple[pd.DataFrame, list[str]]:
    """Recent role table for every player who played for the team in the window before game_date."""
    l = logs[(logs.team_id == team_id) & (logs.game_date_et < game_date) & (logs.played == 1)]
    order = l[["game_id", "game_date_et"]].drop_duplicates().sort_values(["game_date_et", "game_id"])
    gids = list(order.game_id)
    start = (pd.Timestamp(game_date) - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    rec = l[l.game_date_et >= start].sort_values("game_date_et")
    rows = []
    for pid, d in rec.groupby("player_id"):
        d = d.tail(10)
        last = gids.index(d.game_id.iloc[-1]) if d.game_id.iloc[-1] in gids else len(gids) - 1
        poss = d.team_poss_on.sum() if "team_poss_on" in d else np.nan
        mins = d.minutes.sum()
        rows.append({"player_id": pid, "role_min": float(d.minutes.mean()), "start_rate": float(d.started.mean()),
                     "usage": float((d.fga.sum() + 0.44 * d.fta.sum() + d.tov.sum()) / poss) if poss and poss > 0 else np.nan,
                     "prod36": float((d.pts.sum() + 1.5 * d.ast.sum() + 1.2 * d.reb.sum()) / mins * 36) if mins > 0 else 0.0,
                     "pts": float(d.pts.mean()), "n_recent": int(len(d)),
                     "games_missed": int(len(gids) - 1 - last)})
    R = pd.DataFrame(rows)
    if not len(R):
        return R, gids
    R["usage"] = R["usage"].fillna(0.0)
    R["score"] = (0.5 * (R.role_min / 36).clip(upper=1) + 0.3 * (R.usage / 0.30).clip(upper=1)
                  + 0.2 * (R.prod36 / max(R.prod36.max(), 1e-9)))
    rot = R[R.games_missed < LONG_ABSENCE]           # current rotation, for within-team ranks
    R["rank_score"] = R.score.rank(ascending=False, method="min")
    R["rank_score_rot"] = R.player_id.map(dict(zip(rot.player_id, rot.score.rank(ascending=False, method="min")))).fillna(99)
    return R, gids


def categorize(row) -> str:
    rm, top = row.role_min, row.rank_score_rot
    if (rm >= 28 and (top == 1 or row.usage >= 0.26)) or (rm >= 24 and top == 1 and row.usage >= 0.28):
        return "CRITICAL"
    if rm >= 24 or (row.start_rate >= 0.6 and rm >= 18) or (top <= 3 and rm >= 18):
        return "HIGH"
    if rm >= 12:
        return "MEDIUM"
    return "LOW"


def rank_absences(logs: pd.DataFrame, team_id: int, game_date: str, absences: list[dict], positions: dict,
                  projected: dict | None = None, recent_team_min: dict | None = None) -> list[dict]:
    """absences: [{player_id, status, reason, report_ts, ...}] — official statuses for this game.
    projected: {player_id: {"min": model minutes, "pts": model pts}} of AVAILABLE teammates in the frozen snapshot.
    Returns every absence with category, display flags and model-based beneficiaries (never invented)."""
    R, _ = team_roles(logs, team_id, game_date)
    info = R.set_index("player_id").to_dict("index") if len(R) else {}
    out = []
    for a in absences:
        r = info.get(a["player_id"])
        if r is None:          # no recent games with this team: no measurable role
            item = {**a, "category": "LOW", "role": None, "long_absence": True, "games_missed": None}
        else:
            cat = categorize(pd.Series(r))
            item = {**a, "category": cat, "games_missed": r["games_missed"],
                    "long_absence": r["games_missed"] >= LONG_ABSENCE,
                    "role": {"min": round(r["role_min"], 1), "start_rate": round(r["start_rate"], 2),
                             "usage": round(r["usage"], 3), "pts": round(r["pts"], 1), "n_recent_games": r["n_recent"],
                             "team_rank": int(r["rank_score"])}}
        item["pos_group"] = pos_group(positions.get(a["player_id"]))
        item["score"] = (r or {}).get("score", 0.0)
        out.append(item)
    # beneficiaries: available teammates whose MODEL minutes for this game exceed their recent role
    ben = []
    if projected and recent_team_min is not None:
        for pid, pr in projected.items():
            base = (info.get(pid) or {}).get("role_min")
            if base is None or base < 0:
                base = recent_team_min.get(pid, 0.0)
            dm = (pr.get("min") or 0) - (base or 0)
            if dm >= 2.5:
                ben.append({"player_id": pid, "delta_min": round(dm, 1), "proj_min": pr.get("min"), "base_min": round(base or 0, 1),
                            "delta_pts": round((pr.get("pts") or 0) - ((info.get(pid) or {}).get("pts") or 0), 1),
                            "pos_group": pos_group(positions.get(pid))})
    ben.sort(key=lambda x: -x["delta_min"])
    out_status = {"OUT", "DOUBTFUL", "SUSPENDED", "NOT_WITH_TEAM"}
    big_missing = any(o["category"] in ("CRITICAL", "HIGH") and o["status"] in out_status and not o["long_absence"] for o in out)
    rot_groups = {}
    for pid, r in info.items():
        if r["games_missed"] < LONG_ABSENCE and r["role_min"] >= 12:
            rot_groups.setdefault(pos_group(positions.get(pid)), []).append(pid)
    for o in out:
        same = [b for b in ben if b["pos_group"] == o["pos_group"]] or ben
        o["beneficiaries"] = same[:3] if o["category"] != "LOW" and not o["long_absence"] else []
        only_at_pos = rot_groups.get(o["pos_group"], []) == [o["player_id"]]
        consequential = big_missing or only_at_pos or any(b["delta_min"] >= 4 for b in same)
        o["show_main"] = (o["status"] != "AVAILABLE" and not o["long_absence"] and
                          (o["category"] in ("CRITICAL", "HIGH") or
                           (o["category"] == "MEDIUM" and consequential and o["status"] != "PROBABLE")))
        o["show_reason"] = ("rol principal" if o["category"] in ("CRITICAL", "HIGH") else
                            "profundidad: ya falta un titular" if big_missing else
                            "único jugador de rotación en su posición" if only_at_pos else
                            "el modelo reasigna minutos" if consequential else None)
    out.sort(key=lambda o: (not o["show_main"], o["long_absence"], ORDER[o["category"]], STATUS_ORDER.get(o["status"], 5), -o["score"]))
    return out
