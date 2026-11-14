"""TEST-ONLY synthetic league with a KNOWN data-generating process.

This is not NBA data and is never written to a production database. It exists so tests can
check that the minutes engine recovers structure we planted on purpose:
  * each player's depth-chart backup absorbs the largest share (40%) of his minutes when he is
    out and replaces him in the starting lineup; the rest spreads over the rotation
  * coaches differ in how hard they pull starters in blowouts
  * a long injury followed by a reduced-minutes return
  * a mid-season trade
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

POS = ["G", "G", "F", "F", "C", "G", "F", "C", "G", "F", "C", "G", "F"]
BASE = [34, 33, 32, 31, 29, 22, 19, 16, 12, 8, 5, 3, 2]
# backup index for each slot (same position, next in depth)
BACKUP = {0: 5, 1: 8, 2: 6, 3: 9, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12, 10: 7, 11: 8, 12: 9}
COACH_PULL = [0.9, 0.25, 0.5, 0.5, 0.15, 0.7]   # starter minutes lost per point of blowout excess (team total share)


def make_league(n_teams: int = 6, n_dates: int = 46, seed: int = 1, start: date = date(2026, 10, 21)):
    rng = np.random.default_rng(seed)
    teams = [1610612700 + t for t in range(n_teams)]
    roster = {t: [f"T{t}P{k:02d}" for k in range(13)] for t in range(n_teams)}
    base = {roster[t][k]: float(BASE[k]) for t in range(n_teams) for k in range(13)}
    positions = {roster[t][k]: POS[k] for t in range(n_teams) for k in range(13)}
    strength = rng.normal(0, 4, n_teams)
    logs, games = [], []
    long_inj = (roster[0][0], 14, 26)      # team0 star out dates 14..25
    short_inj = (roster[1][4], 30, 34)
    trade_day, traded = 24, roster[2][6]
    gnum = 0
    for di in range(n_dates):
        d = (start + timedelta(days=di)).isoformat()
        order = list(rng.permutation(n_teams))
        if di == trade_day:      # player moves from team 2 to team 3, takes slot 6-ish role there
            roster[2].remove(traded)
            roster[3].append(traded)
            base[traded] = 20.0
        for h, a in zip(order[0::2], order[1::2]):
            gnum += 1
            gid = f"00226{gnum:05d}"
            margin = strength[h] - strength[a] + 2.5 + rng.normal(0, 12)
            periods = 5 if abs(margin) < 1.0 else 4
            if periods == 5:
                margin = rng.choice([-1, 1]) * rng.integers(1, 8)
            hs = int(110 + margin / 2)
            as_ = int(hs - round(margin))
            games.append({"game_id": gid, "game_date_et": d, "home_team_id": teams[h], "away_team_id": teams[a],
                          "home_score": hs, "away_score": as_, "periods": periods, "status": 3,
                          "tip_utc": f"{d}T23:30:00+00:00", "season": "2026-27"})
            bx = max(0.0, abs(margin) - 12) if periods == 4 else 0.0
            for t in (h, a):
                logs.extend(_team_game(rng, t, teams[t], roster[t], base, gid, d, di, bx, periods,
                                       long_inj, short_inj))
    return pd.DataFrame(logs), pd.DataFrame(games), positions, roster, teams


def _team_game(rng, t, team_id, players, base, gid, d, di, bx, periods, long_inj, short_inj):
    out = set()
    for p in players:
        if rng.random() < 0.04:
            out.add(p)
    for pid, a, b in (long_inj, short_inj):
        if pid in players and a <= di < b:
            out.add(pid)
    avail = [p for p in players if p not in out]
    mins = {p: base[p] for p in avail}
    # first game back after the long injury: reduced minutes (ramp)
    if long_inj[0] in avail and di == long_inj[2]:
        mins[long_inj[0]] *= 0.75
    starters = [p for p in players[:5] if p in avail]
    for p in players[:13]:
        if p in out and p in players:
            k = players.index(p) if players.index(p) < 13 else None
            bk = BACKUP.get(k) if k is not None else None
            give = base[p]
            if bk is not None and bk < len(players) and players[bk] in avail:
                mins[players[bk]] += 0.4 * give
                give *= 0.6
                if k < 5 and players[bk] not in starters:
                    starters.append(players[bk])
            rot = [q for q in avail if base[q] >= 10]
            for q in rot:
                mins[q] += give / len(rot)
    for p in avail:   # fill starters if still short
        if len(starters) >= 5:
            break
        if p not in starters:
            starters.append(p)
    starters = starters[:5]
    pull = COACH_PULL[t % len(COACH_PULL)] * bx
    for p in starters:
        mins[p] -= pull
    bench = [p for p in avail if p not in starters]
    for p in bench:
        mins[p] += pull * 5 / max(1, len(bench))
    played = {}
    for p in avail:
        sd = 2.5 if p in starters else 3.0
        m = mins[p] + rng.normal(0, sd)
        if base[p] < 6 and p not in starters and rng.random() < 0.55 - 0.08 * bx:
            m = 0.0
        played[p] = max(0.0, min(48.0, m))
    total = 240.0 + 25.0 * (periods - 4)
    pos = [p for p in avail if played[p] > 0]
    s = sum(played[p] for p in pos)
    for p in pos:
        played[p] = played[p] * total / s
    rows = []
    for p in players:
        if p in out:
            rows.append({"game_id": gid, "player_id": p, "team_id": team_id, "game_date_et": d, "started": 0,
                         "played": 0, "minutes": 0.0, "not_playing_reason": "INACTIVE_INJURY"})
        else:
            rows.append({"game_id": gid, "player_id": p, "team_id": team_id, "game_date_et": d,
                         "started": int(p in starters), "played": int(played[p] > 0), "minutes": played[p],
                         "not_playing_reason": None if played[p] > 0 else "DNP_COACH"})
    return rows
