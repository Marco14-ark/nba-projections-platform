"""Live ingestion jobs (wire real feeds into the store). Each job is idempotent.

  sync_schedule        cdn.nba.com schedule  -> games
  ingest_final_boxes   cdn.nba.com boxscores -> player_game_logs, final scores, per-team rosters
  ingest_injury_report official PDF          -> injury_status (OFFICIAL) + coverage
  ingest_pregame_box   cdn.nba.com boxscore before tip -> CONFIRMED starters / inactives, if posted

NOTE: none of these jobs could be executed against the live endpoints from the build sandbox
(egress blocked). Parsers are tested on fixtures that follow the documented structure; the first
production run must be watched (SchemaDrift is raised, never swallowed).
"""
from __future__ import annotations

import json
from datetime import datetime

from ..core.timeutil import ET, utcnow
from ..db.store import Store
from ..injuries.monitor import InjuryMonitor
from ..roster.sync import RosterEntry, RosterSync
from ..sources import nba_cdn
from ..sources.http import FetchError, Fetcher
from ..sources.injury_report import candidate_urls, parse_pdf
from ..sources.teams import TEAMS


def make_fetcher(store: Store) -> Fetcher:
    return Fetcher(archive=lambda src, url, body, ts: store.archive_payload(src, url, body, ts))


def ensure_teams(store: Store) -> None:
    for t in TEAMS:
        store.upsert_team(*t)
    store.commit()


def sync_schedule(store: Store, f: Fetcher) -> int:
    doc = json.loads(f.get(nba_cdn.SCHEDULE_URL, "nba_cdn").body)
    games = nba_cdn.parse_schedule(doc)
    for g in games:
        store.upsert_game(g.to_row())
    store.commit()
    return len(games)


def _pid(p: nba_cdn.BoxPlayer) -> str:
    return f"P{p.person_id}"


def ingest_final_boxes(store: Store, f: Fetcher, game_date_et: str) -> dict:
    out = {"games": 0, "accounting_errors": []}
    rs = RosterSync(store)
    for g in store.games_on(game_date_et):
        if g["status"] == 3 and store.q("SELECT 1 FROM player_game_logs WHERE game_id=? LIMIT 1", (g["game_id"],)):
            continue
        res = f.get(nba_cdn.BOXSCORE_URL.format(game_id=g["game_id"]), "nba_cdn", ok_404=True)
        if res is None:
            continue
        bs = nba_cdn.parse_boxscore(json.loads(res.body))
        if bs.status != 3:
            continue
        for tid, err in nba_cdn.boxscore_minutes_check(bs).items():
            if abs(err) > 0.5:
                out["accounting_errors"].append((bs.game_id, tid, err))
        for team_id in (bs.home_team_id, bs.away_team_id):
            ents = [RosterEntry(team_id, p.first_name, p.family_name, p.person_id, position=p.position)
                    for p in bs.players if p.team_id == team_id]
            rs.ingest_snapshot("nba_boxscore", game_date_et, ents, covered_team_ids={team_id})
        for row in nba_cdn.boxscore_to_logs(bs, _pid):
            store.upsert_log(row)
        store.upsert_game({**dict(g), "status": 3, "home_score": bs.home_score, "away_score": bs.away_score,
                           "periods": bs.period})
        out["games"] += 1
    store.commit()
    return out


def ingest_injury_report(store: Store, f: Fetcher, now_et: datetime | None = None) -> dict:
    now_et = now_et or utcnow().astimezone(ET)
    for url in candidate_urls(now_et):
        try:
            res = f.get(url, "nba_injury_report", ok_404=True)
        except FetchError:
            continue
        if res is None:
            continue
        stamp = url.rsplit("Injury-Report_", 1)[1].removesuffix(".pdf")       # 2026-11-13_06_00PM
        report_ts = datetime.strptime(stamp, "%Y-%m-%d_%I_%M%p").replace(tzinfo=ET)
        if store.q("SELECT 1 FROM injury_status WHERE source='nba_injury_report' AND report_ts=? LIMIT 1",
                   (report_ts.isoformat(),)):
            return {"url": url, "status": "ALREADY_INGESTED"}
        r = InjuryMonitor(store).ingest_official(parse_pdf(res.body), report_ts, res.fetched_at)
        return {"url": url, "stored": r.stored, "unresolved": [x.player_raw for x in r.unresolved],
                "unmatched_games": r.unmatched_games, "not_submitted": sorted(r.teams_not_submitted)}
    return {"status": "NO_REPORT_FOUND"}


def ingest_pregame_box(store: Store, f: Fetcher, game_id: str) -> dict:
    """If the league has posted the game's box before tip, starters are CONFIRMED and the
    inactive list is official. Not posted -> nothing recorded (lineups stay EXPECTED)."""
    res = f.get(nba_cdn.BOXSCORE_URL.format(game_id=game_id), "nba_cdn", ok_404=True)
    if res is None:
        return {"status": "NOT_POSTED"}
    bs = nba_cdn.parse_boxscore(json.loads(res.body))
    mon = InjuryMonitor(store)
    out = {}
    for tid in (bs.home_team_id, bs.away_team_id):
        ps = [p for p in bs.players if p.team_id == tid]
        starters = [_pid(p) for p in ps if p.starter]
        if len(starters) == 5:
            store.insert_lineup(game_id, tid, "CONFIRMED", starters, "nba_cdn_box", res.fetched_at)
        inactive = [_pid(p) for p in ps if p.status == "INACTIVE"]
        if inactive:
            mon.record_inactives(game_id, tid, inactive, "nba_cdn_box", res.fetched_at)
        out[tid] = {"starters": len(starters), "inactive": len(inactive)}
    store.commit()
    return out
