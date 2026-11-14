"""Ingestion jobs run end-to-end against a fake HTTP session (no network)."""
import json
from datetime import datetime

from nba_platform.core.timeutil import ET
from nba_platform.db.store import Store
from nba_platform.pipeline import ingest
from nba_platform.sources.http import Fetcher
from tests.test_sources import BOX, _make_pdf


class FakeResp:
    def __init__(self, code, body):
        self.status_code, self.content = code, body


class FakeSession:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, **kw):
        self.calls.append(url)
        for k, v in self.routes.items():
            if k in url:
                return FakeResp(*v) if isinstance(v, tuple) else FakeResp(200, v)
        return FakeResp(404, b"")


SCHED = {"leagueSchedule": {"gameDates": [{"games": [
    {"gameId": "0022600123", "gameDateTimeUTC": "2026-11-14T00:30:00Z",
     "homeTeam": {"teamId": 1610612747, "teamTricode": "LAL"}, "awayTeam": {"teamId": 1610612738, "teamTricode": "BOS"}},
    {"gameId": "0022600124", "gameDateTimeUTC": "2026-02-28T00:00:00Z",
     "homeTeam": {"teamId": 1610612765, "teamTricode": "DET"}, "awayTeam": {"teamId": 1610612739, "teamTricode": "CLE"}}]}]}}


def test_schedule_finals_and_rosters_from_box():
    s = Store()
    ingest.ensure_teams(s)
    sess = FakeSession({"scheduleLeagueV2": json.dumps(SCHED).encode(), "boxscore_0022600123": json.dumps(BOX).encode()})
    f = Fetcher(session=sess, archive=lambda *a: s.archive_payload(*a), sleep=lambda x: None)
    assert ingest.sync_schedule(s, f) == 2
    out = ingest.ingest_final_boxes(s, f, "2026-11-13")
    assert out["games"] == 1 and out["accounting_errors"] == []
    assert s.q("SELECT status, home_score, periods FROM games WHERE game_id='0022600123'")[0][:] == (3, 120, 5)
    assert s.q("SELECT COUNT(*) FROM player_game_logs")[0][0] == 4
    assert s.q("SELECT COUNT(*) FROM roster_stints WHERE end_date IS NULL")[0][0] == 4
    assert s.q("SELECT COUNT(*) FROM raw_payloads")[0][0] == 2             # every response archived
    # rerun is idempotent and does not refetch the final game
    n_calls = len(sess.calls)
    assert ingest.ingest_final_boxes(s, f, "2026-11-13")["games"] == 0 and len(sess.calls) == n_calls


def test_injury_report_job_walks_back_to_latest_pdf():
    s = Store()
    ingest.ensure_teams(s)
    s.upsert_game({"game_id": "0022600124", "season": "2025-26", "game_date_et": "2026-02-27",
                   "tip_utc": "2026-02-28T00:00:00+00:00", "home_team_id": 1610612765, "away_team_id": 1610612739})
    from nba_platform.roster.sync import RosterEntry, RosterSync
    RosterSync(s).ingest_snapshot("nba", "2026-01-01", [RosterEntry(1610612739, "Kyle", "Ellis", 1),
                                                        RosterEntry(1610612739, "Julian", "Strand", 2),
                                                        RosterEntry(1610612765, "Peyton", "Watts", 3)])
    sess = FakeSession({"Injury-Report_2026-02-27_02_30PM.pdf": _make_pdf()})
    f = Fetcher(session=sess, sleep=lambda x: None)
    r = ingest.ingest_injury_report(s, f, datetime(2026, 2, 27, 14, 52, tzinfo=ET))
    assert r["stored"] == 3 and r["unresolved"] == [] and r["url"].endswith("02_30PM.pdf")
    assert sess.calls[0].endswith("02_45PM.pdf")                           # tried newest first
    assert ingest.ingest_injury_report(s, f, datetime(2026, 2, 27, 14, 52, tzinfo=ET))["status"] == "ALREADY_INGESTED"
