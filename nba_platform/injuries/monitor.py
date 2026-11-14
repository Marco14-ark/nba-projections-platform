"""NBA_INJURY_MONITOR_V1 — append-only availability observations + change detection.

  * Official NBA report rows are stored with origin OFFICIAL; third-party feeds with
    LICENSED_FEED; anything the model infers with MODEL_INFERENCE (spec §5).
  * Names that cannot be matched to exactly one rostered player are returned as `unresolved`
    and are NOT stored (no invented injuries).
  * Minutes restrictions are only recorded when a source states one explicitly.
  * `diff` compares two as-of states and labels each change by relevance so the scheduler can
    recompute only when something material changed (spec §57).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..core.enums import AvailabilityStatus as S, StatusOrigin
from ..core.timeutil import ET, utcnow
from ..db.store import Store
from ..roster.sync import IdResolver
from ..sources.injury_report import InjuryRow

MONITOR_NAME = "NBA_INJURY_MONITOR_V1"

# Ordinal "probability-of-playing class" used only to decide whether a change is material.
_PLAY_CLASS = {S.AVAILABLE: 0, S.PROBABLE: 0, S.MINUTES_RESTRICTION: 1, S.GTD: 2, S.QUESTIONABLE: 2,
               S.DOUBTFUL: 3, S.OUT: 4, S.REST: 4, S.SUSPENDED: 4, S.NOT_WITH_TEAM: 4, S.UNKNOWN: 2}


@dataclass
class IngestResult:
    stored: int = 0
    unresolved: list[InjuryRow] = field(default_factory=list)
    unmatched_games: list[str] = field(default_factory=list)
    teams_not_submitted: set[int] = field(default_factory=set)
    teams_submitted: set[int] = field(default_factory=set)


@dataclass
class StatusChange:
    player_id: str
    game_id: str
    old: S | None
    new: S
    relevance: str      # HIGH | MEDIUM | LOW


def _game_index(store: Store, game_date_et: str) -> dict[tuple[int, int], str]:
    idx = {}
    for g in store.games_on(game_date_et):
        idx[(g["away_team_id"], g["home_team_id"])] = g["game_id"]
    return idx


class InjuryMonitor:
    NAME = MONITOR_NAME

    def __init__(self, store: Store, directory=None):
        self.s = store
        self.ids = IdResolver(store)
        self.directory = directory      # callable(name_key, team_id) -> player_id | None

    def ingest_official(self, rows: list[InjuryRow], report_ts_et: datetime,
                        observed_at: str | None = None) -> IngestResult:
        from ..sources.teams import BY_TRICODE

        res = IngestResult()
        observed_at = observed_at or utcnow().isoformat()
        idx_cache: dict[str, dict] = {}
        for r in rows:
            m, d, y = r.game_date.split("/")
            gdate = f"{y}-{m}-{d}"
            idx = idx_cache.setdefault(gdate, _game_index(self.s, gdate))
            try:
                away, home = r.matchup.split("@")
                gid = idx.get((BY_TRICODE[away][0], BY_TRICODE[home][0]))
            except (ValueError, KeyError):
                gid = None
            if gid is None:
                res.unmatched_games.append(f"{r.game_date} {r.matchup}")
                continue
            if r.team_id:
                self.s.x("INSERT OR REPLACE INTO injury_report_coverage VALUES (?,?,?,?,?)",
                         (gid, r.team_id, 0 if r.not_yet_submitted else 1,
                          report_ts_et.astimezone(ET).isoformat(), observed_at))
            if r.not_yet_submitted:
                if r.team_id:
                    res.teams_not_submitted.add(r.team_id)
                continue
            if r.team_id:
                res.teams_submitted.add(r.team_id)
            pid = self.ids.match_name(r.name_key, r.team_id, gdate)
            if pid is None and self.directory is not None:
                # the official report is itself a roster source: resolve by name + team history
                pid = self.directory(r.name_key, r.team_id)
                if pid is not None:
                    from ..roster.sync import RosterSync
                    from ..roster.sync import team_of as _team_of
                    if _team_of(self.s, pid, gdate) != r.team_id:
                        RosterSync(self.s).apply_transaction(pid, __import__("nba_platform.core.enums", fromlist=["x"]).TransactionType.ROSTER_ADD_DETECTED,
                                                             gdate, r.team_id, "nba_injury_report", detail="listed on official report")
            if pid is None:
                res.unresolved.append(r)
                continue
            self.s.insert_injury_obs({
                "player_id": pid, "team_id": r.team_id, "game_id": gid, "status": r.status.value,
                "origin": StatusOrigin.OFFICIAL.value, "reason": r.reason, "source": "nba_injury_report",
                "report_ts": report_ts_et.astimezone(ET).isoformat(), "observed_at": observed_at})
            res.stored += 1
        self.s.commit()
        return res

    def ingest_licensed(self, items: list[dict], game_for_team: dict[int, str], source: str,
                        observed_at: str | None = None) -> IngestResult:
        """items: dicts with player_id (already resolved), team_id, status (enum), reason,
        optional minutes_cap (only if the provider states an explicit limit)."""
        res = IngestResult()
        observed_at = observed_at or utcnow().isoformat()
        for it in items:
            gid = game_for_team.get(it["team_id"])
            if gid is None:
                continue
            self.s.insert_injury_obs({
                "player_id": it["player_id"], "team_id": it["team_id"], "game_id": gid,
                "status": S(it["status"]).value, "origin": StatusOrigin.LICENSED_FEED.value,
                "reason": it.get("reason"), "source": source, "observed_at": observed_at,
                "minutes_cap": it.get("minutes_cap")})
            res.stored += 1
        self.s.commit()
        return res

    def record_inactives(self, game_id: str, team_id: int, inactive_ids: list[str], source: str,
                         observed_at: str | None = None) -> None:
        """Official game-day inactive list (e.g. from the pregame boxscore feed)."""
        observed_at = observed_at or utcnow().isoformat()
        for pid in inactive_ids:
            self.s.insert_injury_obs({"player_id": pid, "team_id": team_id, "game_id": game_id,
                                      "status": S.OUT.value, "origin": StatusOrigin.OFFICIAL.value,
                                      "reason": "INACTIVE", "source": source, "observed_at": observed_at})
        self.s.commit()

    def state(self, game_id: str, as_of: str) -> dict[str, S]:
        return {pid: S(r["status"]) for pid, r in self.s.injury_state_asof(game_id, as_of).items()}

    def minutes_caps(self, game_id: str, as_of: str) -> dict[str, float]:
        return {pid: r["minutes_cap"] for pid, r in self.s.injury_state_asof(game_id, as_of).items()
                if r["minutes_cap"] is not None}

    @staticmethod
    def diff(old: dict[str, S], new: dict[str, S], game_id: str,
             core_players: set[str] | None = None) -> list[StatusChange]:
        core_players = core_players or set()
        out = []
        for pid in set(old) | set(new):
            o, n = old.get(pid), new.get(pid, S.AVAILABLE)
            if o == n or (o is None and n in (S.AVAILABLE,)):
                continue
            oc = _PLAY_CLASS.get(o, 0) if o is not None else 0
            nc = _PLAY_CLASS[n]
            crosses = (oc < 4) != (nc < 4)
            if crosses and pid in core_players:
                rel = "HIGH"
            elif crosses or abs(nc - oc) >= 2:
                rel = "MEDIUM" if pid not in core_players else "HIGH"
            else:
                rel = "LOW"
            out.append(StatusChange(pid, game_id, o, n, rel))
        return sorted(out, key=lambda c: ("HIGH", "MEDIUM", "LOW").index(c.relevance))
