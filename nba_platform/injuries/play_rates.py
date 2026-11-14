"""Empirical P(plays | official status at time-before-tip) from our own archive.

This is the only defensible way to put probabilities on QUESTIONABLE/DOUBTFUL scenarios
(spec §34): the rate is measured on past official reports we captured, as they stood at a
comparable time before tip, joined to whether the player actually played. Below `min_n`
observations the rate is None and scenarios are shown separately without probabilities.
"""
from __future__ import annotations

from datetime import timedelta

from ..core.timeutil import parse_utc
from ..db.store import Store

UNCERTAIN = ("QUESTIONABLE", "DOUBTFUL", "PROBABLE", "GTD")


def fit_status_play_rates(store: Store, minutes_before_tip: int = 90, min_n: int = 40,
                          before_date: str | None = None) -> dict[str, dict]:
    games = {g["game_id"]: g for g in store.q(
        "SELECT game_id, tip_utc, game_date_et FROM games WHERE status=3"
        + (" AND game_date_et < ?" if before_date else ""), (before_date,) if before_date else ())}
    played = {(r["game_id"], r["player_id"]): r["played"] for r in store.q(
        "SELECT game_id, player_id, played FROM player_game_logs")}
    counts: dict[str, list[int]] = {s: [0, 0] for s in UNCERTAIN}
    for gid, g in games.items():
        cutoff = (parse_utc(g["tip_utc"]) - timedelta(minutes=minutes_before_tip)).isoformat()
        for pid, row in store.injury_state_asof(gid, cutoff).items():
            if row["origin"] != "OFFICIAL" or row["status"] not in counts:
                continue
            key = (gid, pid)
            if key not in played:
                continue
            counts[row["status"]][0] += 1
            counts[row["status"]][1] += int(played[key])
    out = {}
    for s, (n, k) in counts.items():
        out[s] = {"n": n, "p_play": (k / n) if n >= min_n else None}
    return out
