"""Load NBA CDN play-by-play + boxscore captured as JSON lines ({gameId, actions, box}).

Produced by the free capture step (browser/Mac, see pipeline docs). The boxscore gives the
OFFICIAL minutes, starters and inactive list (with notPlayingReason), used to validate the
play-by-play reconstruction and as leak-free availability for 2026-27.
"""
from __future__ import annotations

import gzip
import json

import pandas as pd

from ..core.timeutil import game_date_et, iso_duration_minutes, parse_utc
from .pbp_reconstruct import PBP_COLS, reconstruct_game


def iter_games(path: str):
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def actions_frame(g: dict) -> pd.DataFrame:
    df = pd.DataFrame(g["actions"])
    df["gameId"] = int(g["gameId"])
    for c in PBP_COLS:
        if c not in df:
            df[c] = None
    for c in ("personId", "possession", "assistPersonId", "stealPersonId", "blockPersonId", "foulDrawnPersonId",
              "teamId", "scoreHome", "scoreAway", "period", "orderNumber", "shotDistance", "isFieldGoal"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("personId", "possession", "assistPersonId", "stealPersonId", "blockPersonId", "foulDrawnPersonId"):
        df[c] = df[c].fillna(0)
    df["scoreHome"] = df["scoreHome"].ffill().fillna(0)
    df["scoreAway"] = df["scoreAway"].ffill().fillna(0)
    return df[PBP_COLS]


def official_box(g: dict) -> list[dict]:
    b = g["box"]
    tip = parse_utc(b["gameTimeUTC"])
    rows = []
    for side in ("homeTeam", "awayTeam"):
        t = b[side]
        for p in t["players"]:
            st = p.get("statistics") or {}
            rows.append({"game_id": b["gameId"], "player_id": f"P{p['personId']}", "team_id": int(t["teamId"]),
                         "game_date_et": game_date_et(tip).isoformat(), "official_status": p.get("status"),
                         "official_starter": int(str(p.get("starter", "0")) == "1"),
                         "official_played": int(str(p.get("played", "0")) == "1"),
                         "official_minutes": iso_duration_minutes(st.get("minutes")),
                         "official_pts": st.get("points"), "not_playing_reason": p.get("notPlayingReason"),
                         "name": p.get("nameI")})
    return rows


def build_frames(path: str, before_utc=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(games, logs incl. OFFICIAL availability, official box rows, lineup segments) from the archive."""
    from .pbp_reconstruct import segments_frame
    games, logs, box, segs = [], [], [], []
    for g in iter_games(path):
        b = g["box"]
        tip = parse_utc(b["gameTimeUTC"])
        if int(b.get("gameStatus", 0)) != 3 or (before_utc is not None and tip >= before_utc):
            continue
        r = reconstruct_game(actions_frame(g))
        d = game_date_et(tip).isoformat()
        h, a = int(b["homeTeam"]["teamId"]), int(b["awayTeam"]["teamId"])
        games.append({"game_id": r.game_id, "season": "2026-27", "game_date_et": d, "tip_utc": tip.isoformat(),
                      "home_team_id": h, "away_team_id": a, "home_score": int(b["homeTeam"]["score"]),
                      "away_score": int(b["awayTeam"]["score"]), "periods": int(b["period"]), "status": 3,
                      "home_poss": r.possessions.get(h, 0), "away_poss": r.possessions.get(a, 0), "flags": ";".join(r.flags)})
        for pid, s in r.players.items():
            logs.append({"game_id": r.game_id, "player_id": f"P{pid}", "team_id": s["team_id"], "game_date_et": d,
                         "started": s["started"], "played": 1, "minutes": s["sec"] / 60, "not_playing_reason": None,
                         "reb": s["oreb"] + s["dreb"], **{k: v for k, v in s.items() if k not in ("team_id", "started")}})
        ob = official_box(g)
        box += ob
        for o in ob:
            if o["official_played"] == 0:
                s_ = str(o["not_playing_reason"] or "")
                lab = ("INACTIVE_OFFICIAL:" + s_) if (s_.startswith(("INACTIVE", "NWT")) or s_ in
                                                      ("DND_INJURY", "DND_REST", "DND_ILLNESS", "DND_PERSONAL")) else ("DNP_OFFICIAL:" + (s_ or "COACH"))
                logs.append({"game_id": o["game_id"], "player_id": o["player_id"], "team_id": o["team_id"],
                             "game_date_et": o["game_date_et"], "started": 0, "played": 0, "minutes": 0.0,
                             "not_playing_reason": lab})
        segs.append(segments_frame(r))
    return (pd.DataFrame(games), pd.DataFrame(logs), pd.DataFrame(box),
            pd.concat(segs, ignore_index=True) if segs else pd.DataFrame())
