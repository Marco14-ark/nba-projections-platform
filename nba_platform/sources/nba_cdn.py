"""Adapters for the public NBA CDN JSON feeds (cdn.nba.com).

These are the endpoints the nba.com site itself uses. Unlike stats.nba.com they are not
blocked from cloud IPs (per community reports), so they are the free backbone for:
  - schedule (all games of the season, tip times in UTC)
  - today's scoreboard (status changes)
  - boxscore per game: active/inactive players, starters, minutes, stats (post-game truth)
  - play-by-play per game: substitutions -> stints (Phase 2 rotation model)

The JSON contracts below follow the structure documented by the nba_api project's `live`
module. They are undocumented by the NBA and can change: every parser validates the fields it
needs and raises `SchemaDrift` instead of silently producing wrong data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.timeutil import game_date_et, iso_duration_minutes, parse_utc, season_for_date

CDN = "https://cdn.nba.com/static/json"
SCHEDULE_URL = f"{CDN}/staticData/scheduleLeagueV2.json"
SCOREBOARD_URL = f"{CDN}/liveData/scoreboard/todaysScoreboard_00.json"
BOXSCORE_URL = CDN + "/liveData/boxscore/boxscore_{game_id}.json"
PBP_URL = CDN + "/liveData/playbyplay/playbyplay_{game_id}.json"
HEADSHOT_URL = "https://cdn.nba.com/headshots/nba/latest/1040x760/{person_id}.png"

GAME_TYPES = {"001": "PRESEASON", "002": "REGULAR", "003": "ALLSTAR", "004": "PLAYOFFS",
              "005": "PLAYIN", "006": "CUP_FINAL"}


class SchemaDrift(ValueError):
    """Upstream JSON no longer has the structure we rely on."""


def _req(d: dict, *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            raise SchemaDrift(f"missing key path {'.'.join(keys)}")
        cur = cur[k]
    return cur


def game_type(game_id: str) -> str:
    return GAME_TYPES.get(game_id[:3], "OTHER")


@dataclass
class Game:
    game_id: str
    tip_utc: str
    home_team_id: int
    away_team_id: int
    home_tricode: str
    away_tricode: str
    status: int
    home_score: int | None = None
    away_score: int | None = None
    period: int | None = None

    def to_row(self) -> dict:
        tip = parse_utc(self.tip_utc)
        d = game_date_et(tip)
        return {"game_id": self.game_id, "season": season_for_date(d), "game_date_et": d.isoformat(),
                "tip_utc": tip.isoformat(), "home_team_id": self.home_team_id,
                "away_team_id": self.away_team_id, "status": self.status,
                "home_score": self.home_score, "away_score": self.away_score,
                "periods": self.period}


def parse_schedule(doc: dict, include_types: tuple[str, ...] = ("REGULAR", "PLAYIN", "PLAYOFFS", "CUP_FINAL")) -> list[Game]:
    out: list[Game] = []
    for gd in _req(doc, "leagueSchedule", "gameDates"):
        for g in gd.get("games", []):
            gid = _req(g, "gameId")
            if game_type(gid) not in include_types:
                continue
            ht, at = _req(g, "homeTeam"), _req(g, "awayTeam")
            if not ht.get("teamId") or not at.get("teamId"):
                continue  # TBD playoff slots
            out.append(Game(gid, _req(g, "gameDateTimeUTC"), int(ht["teamId"]), int(at["teamId"]),
                            ht.get("teamTricode", ""), at.get("teamTricode", ""),
                            int(g.get("gameStatus", 1)), ht.get("score"), at.get("score")))
    return out


def parse_scoreboard(doc: dict) -> list[Game]:
    out = []
    for g in _req(doc, "scoreboard", "games"):
        ht, at = _req(g, "homeTeam"), _req(g, "awayTeam")
        out.append(Game(_req(g, "gameId"), _req(g, "gameTimeUTC"), int(_req(ht, "teamId")),
                        int(_req(at, "teamId")), ht.get("teamTricode", ""), at.get("teamTricode", ""),
                        int(_req(g, "gameStatus")), ht.get("score"), at.get("score"), g.get("period")))
    return out


@dataclass
class BoxPlayer:
    person_id: int
    team_id: int
    first_name: str
    family_name: str
    position: str | None
    jersey: str | None
    status: str                 # ACTIVE | INACTIVE
    starter: bool
    played: bool
    minutes: float
    not_playing_reason: str | None
    not_playing_description: str | None
    stats: dict = field(default_factory=dict)


@dataclass
class BoxScore:
    game_id: str
    tip_utc: str
    status: int
    period: int
    home_team_id: int
    away_team_id: int
    home_score: int
    away_score: int
    players: list[BoxPlayer]


_STAT_MAP = {"points": "pts", "reboundsTotal": "reb", "assists": "ast", "threePointersMade": "fg3m",
             "threePointersAttempted": "fg3a", "fieldGoalsAttempted": "fga", "fieldGoalsMade": "fgm",
             "freeThrowsAttempted": "fta", "freeThrowsMade": "ftm", "reboundsOffensive": "oreb",
             "reboundsDefensive": "dreb", "turnovers": "tov", "foulsPersonal": "pf",
             "plusMinusPoints": "plus_minus"}


def parse_boxscore(doc: dict) -> BoxScore:
    g = _req(doc, "game")
    players: list[BoxPlayer] = []
    for side in ("homeTeam", "awayTeam"):
        t = _req(g, side)
        tid = int(_req(t, "teamId"))
        for p in _req(t, "players"):
            st = p.get("statistics") or {}
            players.append(BoxPlayer(
                person_id=int(_req(p, "personId")), team_id=tid,
                first_name=p.get("firstName", ""), family_name=p.get("familyName", ""),
                position=p.get("position") or None, jersey=p.get("jerseyNum"),
                status=p.get("status", "ACTIVE"), starter=str(p.get("starter", "0")) == "1",
                played=str(p.get("played", "0")) == "1",
                minutes=iso_duration_minutes(st.get("minutes")),
                not_playing_reason=p.get("notPlayingReason"),
                not_playing_description=p.get("notPlayingDescription"),
                stats={v: st.get(k) for k, v in _STAT_MAP.items()}))
    return BoxScore(_req(g, "gameId"), _req(g, "gameTimeUTC"), int(_req(g, "gameStatus")),
                    int(g.get("period", 4)), int(g["homeTeam"]["teamId"]), int(g["awayTeam"]["teamId"]),
                    int(g["homeTeam"].get("score", 0)), int(g["awayTeam"].get("score", 0)), players)


def boxscore_to_logs(bs: BoxScore, player_id_of: callable) -> list[dict]:
    """Convert a FINAL boxscore into player_game_logs rows. Refuses non-final games."""
    if bs.status != 3:
        raise ValueError(f"boxscore {bs.game_id} not final (status={bs.status})")
    d = game_date_et(parse_utc(bs.tip_utc)).isoformat()
    rows = []
    for p in bs.players:
        rows.append({"game_id": bs.game_id, "player_id": player_id_of(p), "team_id": p.team_id,
                     "game_date_et": d, "started": int(p.starter), "played": int(p.played),
                     "minutes": round(p.minutes, 3), "not_playing_reason": p.not_playing_reason,
                     **p.stats})
    return rows


def boxscore_minutes_check(bs: BoxScore) -> dict[int, float]:
    """Team accounting check: sum of player minutes per team should be 240 + 25*OT."""
    expected = 240 + 25 * max(0, bs.period - 4)
    sums: dict[int, float] = {}
    for p in bs.players:
        sums[p.team_id] = sums.get(p.team_id, 0.0) + p.minutes
    return {tid: s - expected for tid, s in sums.items()}
