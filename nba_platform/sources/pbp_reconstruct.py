"""Reconstruct boxscores, minutes, starters, stints and possessions from NBA CDN play-by-play.

Input: the CDN live play-by-play format (`game.actions[]` from cdn.nba.com, or the same columns
as distributed in the open, Apache-2.0 dataset github.com/shufinskiy/nba_data `cdnnba_<season>`).

Everything here is DERIVED from observed events. Each game gets quality checks:
  * minutes_ok : every team's on-court seconds == 5 x regulation/OT length (within 1 s/player)
  * points_ok  : sum of player points == final team score
  * starters_ok: exactly 5 inferred period-1 starters per team
Games failing a check are kept but flagged; consumers choose whether to use them.

Known limitation (the PBP has no roster/inactive list): a player who did not enter the game
leaves no trace. Availability of non-participants must be inferred elsewhere
(see `infer_absences`), and those inferences are labelled as such.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.timeutil import game_date_et, parse_utc, season_for_date

PBP_COLS = ["gameId", "actionNumber", "orderNumber", "clock", "timeActual", "period", "periodType",
            "actionType", "subType", "qualifiers", "personId", "teamId", "possession", "scoreHome",
            "scoreAway", "shotResult", "isFieldGoal", "area", "shotDistance", "assistPersonId",
            "stealPersonId", "blockPersonId", "foulDrawnPersonId", "x", "y"]


NON_PRESENCE_ACTIONS = {"ejection", "timeout", "instantreplay", "stoppage", "memo", "game", "period"}


def clock_seconds(c: str) -> float:
    # 'PT11M36.00S'
    c = c[2:]
    m, s = c.split("M")
    return int(m) * 60 + float(s[:-1])


def period_length(period: int) -> float:
    return 720.0 if period <= 4 else 300.0


@dataclass
class GameRecon:
    game_id: str
    tip_utc: str
    home_team_id: int
    away_team_id: int
    home_score: int
    away_score: int
    periods: int
    players: dict = field(default_factory=dict)           # pid -> stat dict
    stints: list = field(default_factory=list)            # (team, period, start_elapsed, end_elapsed, lineup)
    segments: list = field(default_factory=list)          # both lineups: (period, a, b, home5, away5, hp, ap, hposs, aposs)
    possessions: dict = field(default_factory=dict)       # team -> count
    flags: list = field(default_factory=list)


_STAT_KEYS = ["sec", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "tov", "pf",
              "stl", "blk", "rim_fga", "rim_fgm", "mid_fga", "c3_fga", "ab3_fga",
              "team_fga_on", "team_fgm_on", "opp_fga_on", "team_miss_on", "opp_miss_on", "team_poss_on",
              "team_pts_on", "opp_pts_on", "ft_trips"]


def _blank() -> dict:
    return {k: 0.0 for k in _STAT_KEYS}


def reconstruct_game(ev: pd.DataFrame) -> GameRecon:
    ev = ev.sort_values(["period", "orderNumber"], kind="stable")
    gid = f"{int(ev['gameId'].iloc[0]):010d}"
    tip = str(ev["timeActual"].iloc[0])
    # team ids and home/away (score deltas on made shots tell us which side is home)
    teams = [int(t) for t in pd.unique(ev["teamId"].dropna())]
    home = away = None
    prev_h = prev_a = 0
    for r in ev.itertuples(index=False):
        if r.scoreHome > prev_h and not pd.isna(r.teamId):
            home = int(r.teamId)
        elif r.scoreAway > prev_a and not pd.isna(r.teamId):
            away = int(r.teamId)
        prev_h, prev_a = r.scoreHome, r.scoreAway
        if home and away:
            break
    if home is None or away is None:
        other = [t for t in teams if t not in (home, away)]
        home = home or (other.pop(0) if other else 0)
        away = away or (other.pop(0) if other else 0)
    last = ev.iloc[-1]
    g = GameRecon(gid, tip, home, away, int(last["scoreHome"]), int(last["scoreAway"]), int(ev["period"].max()))

    # player -> team map
    p_team: dict[int, int] = {}
    for pid, tid in zip(ev["personId"], ev["teamId"]):
        if pid and pid > 0 and not pd.isna(tid):
            p_team.setdefault(int(pid), int(tid))
    P = defaultdict(_blank)
    end_lineup: dict[int, set] = {home: set(), away: set()}
    starters: dict[int, set] = {}

    for per, pe in ev.groupby("period", sort=True):
        plen = period_length(int(per))
        # period-start lineup. A player's first SUBSTITUTION event decides it when he has one
        # ('out' => was on court at the start; 'in' => was not, even if an earlier event such as a
        # bench technical or an out-of-order log references him). Players with no substitution
        # in the period are on court if they have any non-technical event.
        first_sub: dict[int, str] = {}
        has_evt: set[int] = set()
        for r in pe.itertuples(index=False):
            if r.personId and r.personId > 0:
                pid0 = int(r.personId)
                if r.actionType == "substitution":
                    first_sub.setdefault(pid0, r.subType)
                elif r.actionType in NON_PRESENCE_ACTIONS or (
                        r.actionType == "foul" and isinstance(r.subType, str) and "technical" in r.subType):
                    pass   # bench ejections / technicals / timeouts are not evidence of being on court
                else:
                    has_evt.add(pid0)
            for col in (r.assistPersonId, r.stealPersonId, r.blockPersonId, r.foulDrawnPersonId):
                if not pd.isna(col) and col > 0:
                    has_evt.add(int(col))
        first = {p: "evt" for p in has_evt} | {p: ("evt" if k == "out" else "sub_in") for p, k in first_sub.items()}
        on = {t: set() for t in (home, away)}
        for pid, kind in first.items():
            t = p_team.get(pid)
            if kind == "evt" and t in on:
                on[t].add(pid)
        for t in (home, away):
            if len(on[t]) < 5 and per > 1:
                # players on court all period without any event: carry from previous period end
                for pid in end_lineup[t]:
                    if len(on[t]) >= 5:
                        break
                    if pid not in first:
                        on[t].add(pid)
            if len(on[t]) != 5:
                g.flags.append(f"LINEUP_{len(on[t])}:{t}:P{per}")
        if per == 1:
            starters = {t: set(on[t]) for t in on}
        # walk events, credit time, track stints and on-court context
        last_clock = plen
        stint_start = {t: 0.0 for t in on}
        poss_team = None
        seg = {"a": 0.0, "pts": {home: 0, away: 0}, "poss": {home: 0, away: 0}}

        def close_seg(at_elapsed):
            if at_elapsed > seg["a"] or any(seg["pts"].values()) or any(seg["poss"].values()):
                g.segments.append((int(per), seg["a"], at_elapsed, tuple(sorted(on[home])), tuple(sorted(on[away])),
                                   seg["pts"][home], seg["pts"][away], seg["poss"][home], seg["poss"][away]))
            seg["a"] = at_elapsed
            seg["pts"] = {home: 0, away: 0}
            seg["poss"] = {home: 0, away: 0}
        for r in pe.itertuples(index=False):
            c = clock_seconds(r.clock)
            dt = last_clock - c
            if dt > 0:
                for t in on:
                    for pid in on[t]:
                        P[pid]["sec"] += dt
                last_clock = c
            at, st = r.actionType, r.subType
            pid = int(r.personId) if r.personId and r.personId > 0 else None
            tid = int(r.teamId) if not pd.isna(r.teamId) else None
            # possession counting: a new possession starts whenever the possession team changes
            pt = int(r.possession) if r.possession and r.possession > 0 else None
            if pt is not None and pt != poss_team and pt in on:
                poss_team = pt
                for q in on[pt]:
                    P[q]["team_poss_on"] += 1
                g.possessions[pt] = g.possessions.get(pt, 0) + 1
                seg["poss"][pt] += 1
            if at == "substitution" and tid in on and pid:
                elapsed = plen - c
                close_seg(elapsed)
                if st == "out" and pid in on[tid]:
                    g.stints.append((tid, int(per), stint_start[tid], elapsed, tuple(sorted(on[tid]))))
                    stint_start[tid] = elapsed
                    on[tid].discard(pid)
                elif st == "in":
                    if stint_start[tid] != elapsed and len(on[tid]) == 5:
                        g.stints.append((tid, int(per), stint_start[tid], elapsed, tuple(sorted(on[tid]))))
                        stint_start[tid] = elapsed
                    on[tid].add(pid)
                continue
            if pid is None or tid not in on:
                continue
            S = P[pid]
            opp = away if tid == home else home
            if at in ("2pt", "3pt"):
                made = r.shotResult == "Made"
                S["fga"] += 1
                S["fgm"] += made
                pts = (3 if at == "3pt" else 2) * made
                S["pts"] += pts
                seg["pts"][tid] += pts
                dist = r.shotDistance if not pd.isna(r.shotDistance) else 99
                area = r.area if isinstance(r.area, str) else ""
                if at == "3pt":
                    S["fg3a"] += 1
                    S["fg3m"] += made
                    if "Corner" in area:
                        S["c3_fga"] += 1
                    else:
                        S["ab3_fga"] += 1
                elif dist <= 4 or "Restricted" in area:
                    S["rim_fga"] += 1
                    S["rim_fgm"] += made
                else:
                    S["mid_fga"] += 1
                for q in on[tid]:
                    P[q]["team_fga_on"] += 1
                    P[q]["team_fgm_on"] += made
                    P[q]["team_miss_on"] += (not made)
                    P[q]["team_pts_on"] += pts
                for q in on[opp]:
                    P[q]["opp_fga_on"] += 1
                    P[q]["opp_miss_on"] += (not made)
                    P[q]["opp_pts_on"] += pts
                if made and not pd.isna(r.assistPersonId) and r.assistPersonId > 0:
                    P[int(r.assistPersonId)]["ast"] += 1
                if not pd.isna(r.blockPersonId) and r.blockPersonId > 0:
                    P[int(r.blockPersonId)]["blk"] += 1
            elif at == "freethrow":
                made = r.shotResult == "Made"
                S["fta"] += 1
                S["ftm"] += made
                S["pts"] += made
                seg["pts"][tid] += made
                if isinstance(st, str) and st.startswith("1 of"):
                    S["ft_trips"] += 1
                for q in on[tid]:
                    P[q]["team_pts_on"] += made
                for q in on[opp]:
                    P[q]["opp_pts_on"] += made
            elif at == "rebound":
                S["oreb" if st == "offensive" else "dreb"] += 1
            elif at == "turnover":
                S["tov"] += 1
                if not pd.isna(r.stealPersonId) and r.stealPersonId > 0:
                    P[int(r.stealPersonId)]["stl"] += 1
            elif at == "foul" and st in ("personal", "offensive", "shooting", "loose ball", "flagrant",
                                         "clear path", "transition take", "away from play"):
                S["pf"] += 1
        if last_clock > 0:
            for t in on:
                for pid in on[t]:
                    P[pid]["sec"] += last_clock
        close_seg(plen)
        for t in on:
            if stint_start[t] < plen:
                g.stints.append((t, int(per), stint_start[t], plen, tuple(sorted(on[t]))))
        end_lineup = {t: set(on[t]) for t in on}

    for pid, s in P.items():
        t = p_team.get(pid)
        if t is None:
            continue
        g.players[pid] = {**s, "team_id": t, "started": int(pid in starters.get(t, set()))}
    # quality checks
    total_sec = sum(period_length(p) for p in range(1, g.periods + 1)) * 5
    for t, score in ((home, g.home_score), (away, g.away_score)):
        sec = sum(s["sec"] for s in g.players.values() if s["team_id"] == t)
        pts = sum(s["pts"] for s in g.players.values() if s["team_id"] == t)
        if abs(sec - total_sec) > 5:
            g.flags.append(f"MINUTES_SUM:{t}:{(sec - total_sec) / 60:+.2f}")
        if pts != score:
            g.flags.append(f"POINTS_SUM:{t}:{pts - score:+.0f}")
        if sum(s["started"] for s in g.players.values() if s["team_id"] == t) != 5:
            g.flags.append(f"STARTERS:{t}")
    return g


def segments_frame(g: "GameRecon") -> pd.DataFrame:
    return pd.DataFrame([{"game_id": g.game_id, "period": a, "start": b, "end": c,
                          "home5": ",".join(f"P{x}" for x in h5), "away5": ",".join(f"P{x}" for x in a5),
                          "home_pts": hp, "away_pts": ap, "home_poss": hq, "away_poss": aq}
                         for (a, b, c, h5, a5, hp, ap, hq, aq) in g.segments])


def reconstruct_season(csv_path: str, progress: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (games, player_logs, stints) DataFrames for one season file."""
    df = pd.read_csv(csv_path, usecols=lambda c: c in PBP_COLS, low_memory=False)
    for c in ("personId", "possession", "assistPersonId", "stealPersonId", "blockPersonId", "foulDrawnPersonId"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    games, logs, stints = [], [], []
    for i, (gid, ev) in enumerate(df.groupby("gameId", sort=True)):
        g = reconstruct_game(ev)
        tip = parse_utc(g.tip_utc)
        d = game_date_et(tip).isoformat()
        games.append({"game_id": g.game_id, "season": season_for_date(game_date_et(tip)), "game_date_et": d,
                      "tip_utc": tip.isoformat(), "home_team_id": g.home_team_id, "away_team_id": g.away_team_id,
                      "home_score": g.home_score, "away_score": g.away_score, "periods": g.periods, "status": 3,
                      "home_poss": g.possessions.get(g.home_team_id, 0), "away_poss": g.possessions.get(g.away_team_id, 0),
                      "flags": ";".join(g.flags)})
        for pid, s in g.players.items():
            logs.append({"game_id": g.game_id, "player_id": f"P{pid}", "team_id": s["team_id"], "game_date_et": d,
                         "started": s["started"], "played": 1, "minutes": s["sec"] / 60.0, "not_playing_reason": None,
                         "reb": s["oreb"] + s["dreb"], **{k: v for k, v in s.items() if k not in ("team_id", "started")}})
        for (t, per, a, b, lu) in g.stints:
            stints.append({"game_id": g.game_id, "team_id": t, "period": per, "start": a, "end": b,
                           "lineup": ",".join(f"P{x}" for x in lu)})
        if progress and i % 200 == 0:
            print(f"  {i} games", flush=True)
    return pd.DataFrame(games), pd.DataFrame(logs), pd.DataFrame(stints)


def infer_absences(logs: pd.DataFrame, games: pd.DataFrame, rotation_min: float = 12.0,
                   lookback: int = 10) -> pd.DataFrame:
    """Add rows for rostered players who did not appear in a game (PBP has no inactive list).

    Roster membership for a team = between the player's first and last appearance for that team
    (handles trades). A non-appearance is labelled:
      INACTIVE_INFERRED : the player was a rotation player (recent mean minutes >= rotation_min)
                          -> treated as unavailable (injury/rest), the usual case for rotation players
      DNP_INFERRED      : otherwise -> treated as available but not used (coach decision)
    These are INFERENCES, not official designations.
    """
    team_games = {}
    tg = pd.concat([games[["game_id", "game_date_et", "home_team_id"]].rename(columns={"home_team_id": "team_id"}),
                    games[["game_id", "game_date_et", "away_team_id"]].rename(columns={"away_team_id": "team_id"})])
    for t, g in tg.sort_values(["game_date_et", "game_id"]).groupby("team_id"):
        team_games[int(t)] = list(zip(g["game_id"], g["game_date_et"]))
    have = set(zip(logs["game_id"], logs["player_id"]))
    out = []
    for (pid, t), g in logs.sort_values("game_date_et").groupby(["player_id", "team_id"]):
        first, lastd = g["game_date_et"].min(), g["game_date_et"].max()
        hist = list(zip(g["game_date_et"], g["minutes"]))
        k = 0
        recent: list[float] = []
        for gid, d in team_games.get(int(t), []):
            if d < first or d > lastd:
                continue
            while k < len(hist) and hist[k][0] < d:
                recent.append(hist[k][1])
                k += 1
            if (gid, pid) in have:
                continue
            m = float(np.mean(recent[-lookback:])) if recent else 0.0
            out.append({"game_id": gid, "player_id": pid, "team_id": int(t), "game_date_et": d, "started": 0,
                        "played": 0, "minutes": 0.0,
                        "not_playing_reason": "INACTIVE_INFERRED" if m >= rotation_min else "DNP_INFERRED"})
    return pd.concat([logs, pd.DataFrame(out)], ignore_index=True)


def infer_absences_asof(logs: pd.DataFrame, games: pd.DataFrame, rotation_min: float = 12.0,
                        lookback: int = 10, max_gap_team_games: int = 20) -> pd.DataFrame:
    """Leak-free version of `infer_absences` (audit 2026-11-13).

    Roster of team T before game g = players whose most recent appearance STRICTLY BEFORE g's
    date was for T and happened within the last `max_gap_team_games` games of T. Nothing after
    the game date is used (the old version used the player's final appearance of the season).
    Labels are still inferences: INACTIVE_INFERRED (recent mean >= rotation_min) / DNP_INFERRED.
    """
    tg = pd.concat([games[["game_id", "game_date_et", "home_team_id"]].rename(columns={"home_team_id": "team_id"}),
                    games[["game_id", "game_date_et", "away_team_id"]].rename(columns={"away_team_id": "team_id"})])
    tg = tg.sort_values(["game_date_et", "game_id"])
    played = logs[logs["played"] == 1]
    by_game = {g: d for g, d in played.groupby("game_id")}
    last_team: dict[str, int] = {}
    last_idx: dict[str, int] = {}
    hist: dict[str, list[float]] = {}
    team_count: dict[int, int] = {}
    out = []
    for d, day in tg.groupby("game_date_et", sort=True):
        # 1) absences use only state built from earlier dates
        for gid, t in zip(day["game_id"], day["team_id"]):
            t = int(t)
            n = team_count.get(t, 0)
            present = by_game.get(gid)
            here = set(present.loc[present["team_id"] == t, "player_id"]) if present is not None else set()
            for pid, lt in last_team.items():
                if lt != t or pid in here or n - last_idx[pid] > max_gap_team_games:
                    continue
                m = float(np.mean(hist[pid][-lookback:])) if hist.get(pid) else 0.0
                out.append({"game_id": gid, "player_id": pid, "team_id": t, "game_date_et": d, "started": 0,
                            "played": 0, "minutes": 0.0,
                            "not_playing_reason": "INACTIVE_INFERRED" if m >= rotation_min else "DNP_INFERRED"})
        # 2) then update state with this date's games
        for gid, t in zip(day["game_id"], day["team_id"]):
            t = int(t)
            team_count[t] = team_count.get(t, 0) + 1
            present = by_game.get(gid)
            if present is None:
                continue
            for pid, mins in zip(present.loc[present["team_id"] == t, "player_id"],
                                 present.loc[present["team_id"] == t, "minutes"]):
                last_team[pid] = t
                last_idx[pid] = team_count[t]
                hist.setdefault(pid, []).append(float(mins))
    return pd.concat([logs, pd.DataFrame(out)], ignore_index=True)
