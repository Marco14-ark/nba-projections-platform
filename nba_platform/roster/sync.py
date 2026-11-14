"""NBA_ROSTER_SYNC_V1 — effective-dated rosters, stable ids, transaction log.

Rules (spec §4, §76-77):
  * One internal id per human. NBA personId is the anchor ('P<personId>'); other sources map
    through player_id_xref. Unmatched / ambiguous names are reported, never guessed.
  * Team membership is a set of stints with start/end dates; a move closes one stint and opens
    another. Nothing is deleted, so 'which team was X on at date D' is always answerable.
  * Roster changes emit events with the affected team ids so dependent forecasts are invalidated.
  * The sync never copies a player's old role or usage into the new team: that is the minutes
    engine's job, which also widens uncertainty for players with few games on the new team.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from ..core.enums import TransactionType
from ..core.hashing import sha
from ..core.names import name_key
from ..core.timeutil import utcnow
from ..db.store import Store

ROSTER_SYNC_NAME = "NBA_ROSTER_SYNC_V1"


@dataclass
class RosterEntry:
    team_id: int
    first_name: str
    last_name: str
    nba_person_id: int | None = None
    source_id: str | None = None
    contract_type: str = "STANDARD"
    position: str | None = None


@dataclass
class RosterEvent:
    player_id: str
    txn_type: TransactionType
    from_team_id: int | None
    to_team_id: int | None
    effective_date: str


@dataclass
class SyncResult:
    events: list[RosterEvent] = field(default_factory=list)
    unresolved: list[RosterEntry] = field(default_factory=list)
    affected_team_ids: set[int] = field(default_factory=set)


class IdResolver:
    def __init__(self, store: Store):
        self.s = store

    def ensure_player(self, e: RosterEntry, source: str) -> str | None:
        now = utcnow().isoformat()
        if e.source_id is not None:
            r = self.s.q("SELECT player_id FROM player_id_xref WHERE source=? AND source_id=?",
                         (source, str(e.source_id)))
            if r:
                return r[0]["player_id"]
        if e.nba_person_id is not None:
            pid = f"P{e.nba_person_id}"
            self.s.x("""INSERT OR IGNORE INTO players(player_id, nba_person_id, full_name, first_name,
                        last_name, name_key, position, created_at) VALUES (?,?,?,?,?,?,?,?)""",
                     (pid, e.nba_person_id, f"{e.first_name} {e.last_name}", e.first_name, e.last_name,
                      name_key(e.first_name, e.last_name), e.position, now))
            self._xref("nba", str(e.nba_person_id), pid)
            if e.source_id is not None:
                self._xref(source, str(e.source_id), pid)
            return pid
        # no NBA id: match by normalised name, disambiguated by current team if needed
        pid = self.match_name(name_key(e.first_name, e.last_name), e.team_id)
        if pid and e.source_id is not None:
            self._xref(source, str(e.source_id), pid)
        return pid

    def match_name(self, key: str, team_id: int | None = None, on_date: str | None = None) -> str | None:
        cands = [r["player_id"] for r in self.s.q("SELECT player_id FROM players WHERE name_key=?", (key,))]
        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1 and team_id is not None:
            d = on_date or date.today().isoformat()
            on_team = [c for c in cands if team_of(self.s, c, d) == team_id]
            if len(on_team) == 1:
                return on_team[0]
        return None  # unknown or ambiguous: caller reports it

    def _xref(self, source: str, sid: str, pid: str) -> None:
        self.s.x("INSERT OR IGNORE INTO player_id_xref VALUES (?,?,?)", (source, sid, pid))


def team_of(store: Store, player_id: str, on_date: str) -> int | None:
    r = store.q("""SELECT team_id FROM roster_stints WHERE player_id=? AND start_date<=?
                   AND (end_date IS NULL OR end_date>=?) ORDER BY start_date DESC LIMIT 1""",
                (player_id, on_date, on_date))
    return r[0]["team_id"] if r else None


def roster_on(store: Store, team_id: int, on_date: str) -> list[str]:
    return [r["player_id"] for r in store.q(
        """SELECT player_id FROM roster_stints WHERE team_id=? AND start_date<=?
           AND (end_date IS NULL OR end_date>=?) ORDER BY player_id""", (team_id, on_date, on_date))]


def contract_on(store: Store, player_id: str, on_date: str) -> str | None:
    r = store.q("""SELECT contract_type FROM roster_stints WHERE player_id=? AND start_date<=?
                   AND (end_date IS NULL OR end_date>=?) ORDER BY start_date DESC LIMIT 1""",
                (player_id, on_date, on_date))
    return r[0]["contract_type"] if r else None


class RosterSync:
    NAME = ROSTER_SYNC_NAME

    def __init__(self, store: Store):
        self.s = store
        self.ids = IdResolver(store)

    def ingest_snapshot(self, source: str, as_of: str, entries: list[RosterEntry],
                        covered_team_ids: set[int] | None = None) -> SyncResult:
        """Apply a roster snapshot effective `as_of` (ET date).

        covered_team_ids: teams the snapshot is complete for. Drops are only inferred for these
        teams (a partial feed must never silently remove players from other teams).
        """
        res = SyncResult()
        covered = covered_team_ids if covered_team_ids is not None else {e.team_id for e in entries}
        seen: dict[str, RosterEntry] = {}
        for e in entries:
            pid = self.ids.ensure_player(e, source)
            if pid is None:
                res.unresolved.append(e)
                continue
            if pid in seen and seen[pid].team_id != e.team_id:
                res.unresolved.append(e)   # same human on two teams in one snapshot: reject
                continue
            seen[pid] = e
        prev_day = (date.fromisoformat(as_of) - timedelta(days=1)).isoformat()
        for pid, e in seen.items():
            cur = self._open_stint(pid)
            if cur is None:
                self._open(pid, e, as_of, source)
                self._event(res, pid, TransactionType.ROSTER_ADD_DETECTED, None, e.team_id, as_of, source)
            elif cur["team_id"] != e.team_id:
                self._close(cur, prev_day)
                self._open(pid, e, as_of, source)
                self._event(res, pid, TransactionType.TEAM_CHANGE_DETECTED, cur["team_id"], e.team_id,
                            as_of, source)
            elif cur["contract_type"] != e.contract_type:
                self._close(cur, prev_day)
                self._open(pid, e, as_of, source)
                self._event(res, pid, TransactionType.TWO_WAY if e.contract_type == "TWO_WAY"
                            else TransactionType.SIGNING, cur["team_id"], e.team_id, as_of, source)
        # drops: open stints on covered teams that are absent from the snapshot
        for r in self.s.q("SELECT * FROM roster_stints WHERE end_date IS NULL"):
            if r["team_id"] in covered and r["player_id"] not in seen and r["start_date"] <= as_of:
                self._close(r, prev_day)
                self._event(res, r["player_id"], TransactionType.ROSTER_DROP_DETECTED, r["team_id"], None,
                            as_of, source)
        self.s.commit()
        return res

    def apply_transaction(self, player_id: str, txn: TransactionType, effective_date: str,
                          to_team_id: int | None, source: str, contract_type: str = "STANDARD",
                          detail: str = "") -> SyncResult:
        """Explicit transaction (trade, waiver, signing...) from a transactions feed."""
        res = SyncResult()
        cur = self._open_stint(player_id)
        prev_day = (date.fromisoformat(effective_date) - timedelta(days=1)).isoformat()
        if cur is not None and (to_team_id is None or cur["team_id"] != to_team_id
                                or cur["contract_type"] != contract_type):
            self._close(cur, prev_day)
        if to_team_id is not None and (cur is None or cur["team_id"] != to_team_id
                                       or cur["contract_type"] != contract_type):
            self.s.x("INSERT OR IGNORE INTO roster_stints VALUES (?,?,?,?,NULL,?,?)",
                     (player_id, to_team_id, contract_type, effective_date, source, utcnow().isoformat()))
        self._event(res, player_id, txn, cur["team_id"] if cur else None, to_team_id, effective_date,
                    source, detail)
        self.s.commit()
        return res

    # -- helpers --------------------------------------------------------------------------------
    def _open_stint(self, pid: str):
        r = self.s.q("SELECT * FROM roster_stints WHERE player_id=? AND end_date IS NULL", (pid,))
        return r[0] if r else None

    def _open(self, pid: str, e: RosterEntry, d: str, source: str) -> None:
        self.s.x("INSERT OR IGNORE INTO roster_stints VALUES (?,?,?,?,NULL,?,?)",
                 (pid, e.team_id, e.contract_type, d, source, utcnow().isoformat()))

    def _close(self, stint, end: str) -> None:
        end = max(end, stint["start_date"])
        self.s.x("UPDATE roster_stints SET end_date=? WHERE player_id=? AND team_id=? AND start_date=?",
                 (end, stint["player_id"], stint["team_id"], stint["start_date"]))

    def _event(self, res: SyncResult, pid: str, t: TransactionType, frm: int | None, to: int | None,
               d: str, source: str, detail: str = "") -> None:
        ev = RosterEvent(pid, t, frm, to, d)
        res.events.append(ev)
        res.affected_team_ids.update(x for x in (frm, to) if x is not None)
        txn_id = sha({"p": pid, "t": t.value, "f": frm, "to": to, "d": d})
        self.s.x("INSERT OR IGNORE INTO transactions VALUES (?,?,?,?,?,?,?,?,?)",
                 (txn_id, pid, t.value, frm, to, d, source, detail, utcnow().isoformat()))
