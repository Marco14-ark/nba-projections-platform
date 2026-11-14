"""End-to-end Phase-1 pipeline on a SQLite store loaded with the synthetic league history."""
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from nba_platform.core.enums import AvailabilityStatus as S, RefreshWindow as W
from nba_platform.db.store import Store
from nba_platform.injuries.monitor import InjuryMonitor
from nba_platform.pipeline.autopilot import Autopilot, due_window
from nba_platform.roster.sync import RosterEntry, RosterSync
from tests.synthetic import make_league

UTC = timezone.utc
TIP = datetime(2026, 12, 6, 0, 30, tzinfo=UTC)      # 2026-12-05 19:30 ET
GID = "0022699999"


@pytest.fixture
def env():
    logs, games, pos, roster, teams = make_league(n_dates=45)
    s = Store()
    for t in teams:
        s.upsert_team(t, f"T{t % 100}", "Synth", str(t))
    entries = []
    for ti, t in enumerate(teams):
        for p in roster[ti]:
            entries.append(RosterEntry(t, p, "", source_id=p))
            s.x("""INSERT OR IGNORE INTO players(player_id, full_name, first_name, last_name, name_key, position,
                   created_at) VALUES (?,?,?,?,?,?,?)""", (p, p, p, "", p.lower() + "|", pos[p], "x"))
            s.x("INSERT OR IGNORE INTO player_id_xref VALUES ('synthetic', ?, ?)", (p, p))
    RosterSync(s).ingest_snapshot("synthetic", "2026-10-01", entries)
    for r in games.to_dict("records"):
        s.upsert_game(r)
    for r in logs.to_dict("records"):
        s.upsert_log(r)
    s.upsert_game({"game_id": GID, "season": "2026-27", "game_date_et": "2026-12-05", "tip_utc": TIP.isoformat(),
                   "home_team_id": teams[0], "away_team_id": teams[1], "status": 1})
    s.commit()
    return s, teams


def test_due_window_logic():
    assert due_window(TIP, TIP - timedelta(hours=10), set()) is W.INITIAL
    assert due_window(TIP, TIP - timedelta(hours=10), {"INITIAL"}) is None
    assert due_window(TIP, TIP - timedelta(hours=5), {"INITIAL"}) is W.T_6H
    assert due_window(TIP, TIP - timedelta(minutes=25), {"INITIAL"}) is W.T_30M    # missed windows skipped
    assert due_window(TIP, TIP + timedelta(minutes=1), set()) is None


def test_autopilot_end_to_end(env):
    s, teams = env
    ap = Autopilot(s, n_sims=2000)
    r = ap.tick(TIP - timedelta(hours=10), "2026-12-05")
    assert [x.status for x in r] == ["COMPUTED"]
    snap = s.latest_forecast(GID)
    assert snap["flag"] == "PARTIAL_DATA"                 # no official injury report ingested yet
    rows = s.q("SELECT * FROM minutes_forecasts WHERE snapshot_id=?", (snap["snapshot_id"],))
    for t in teams[:2]:
        tot = sum(x["mean"] for x in rows if x["team_id"] == t and x["scenario"] == "BASE")
        assert abs(tot - (240 + 25 * 0.0)) < 3.0          # means include OT sims -> slightly above 240
    # 6H window: nothing changed -> no Monte Carlo
    r = ap.tick(TIP - timedelta(hours=5), "2026-12-05")
    assert [x.status for x in r] == ["SKIPPED_NO_CHANGE"]
    assert ap.tick(TIP - timedelta(hours=4), "2026-12-05") == []
    assert ap.public_status(GID, TIP - timedelta(hours=4))["flag"] == "PARTIAL_DATA"
    # a core starter is ruled OUT -> public view must not present the old forecast as current
    star = "T0P00"
    InjuryMonitor(s).ingest_licensed([{"player_id": star, "team_id": teams[0], "status": "OUT"}], {teams[0]: GID},
                                     "test_feed", observed_at=(TIP - timedelta(hours=3)).isoformat())
    assert ap.public_status(GID, TIP - timedelta(hours=2))["flag"] == "UNAVAILABLE"
    r = ap.tick(TIP - timedelta(minutes=85), "2026-12-05")
    assert [x.status for x in r] == ["COMPUTED"] and "MINUTES" in r[0].plan
    new = s.latest_forecast(GID)
    assert new["snapshot_id"] != snap["snapshot_id"]
    old = s.q("SELECT superseded_by FROM forecast_snapshots WHERE snapshot_id=?", (snap["snapshot_id"],))[0][0]
    assert old == new["snapshot_id"]                       # history kept, linked
    star_rows = s.q("SELECT * FROM minutes_forecasts WHERE snapshot_id=? AND player_id=?", (new["snapshot_id"], star))
    assert star_rows == []                                 # OUT player not in any minutes distribution
    assert ap.public_status(GID, TIP - timedelta(minutes=80))["flag"] in ("OK", "PARTIAL_DATA")


def test_snapshots_are_immutable(env):
    s, _ = env
    Autopilot(s, n_sims=500).tick(TIP - timedelta(hours=10), "2026-12-05")
    sid = s.latest_forecast(GID)["snapshot_id"]
    with pytest.raises(sqlite3.IntegrityError):
        s.x("UPDATE forecast_snapshots SET flag='OK' WHERE snapshot_id=?", (sid,))
    with pytest.raises(sqlite3.IntegrityError):
        s.x("UPDATE minutes_forecasts SET mean=0 WHERE snapshot_id=?", (sid,))
    with pytest.raises(sqlite3.IntegrityError):
        s.x("DELETE FROM forecast_snapshots WHERE snapshot_id=?", (sid,))


def test_questionable_scenarios_reach_snapshot(env):
    s, teams = env
    InjuryMonitor(s).ingest_licensed([{"player_id": "T1P01", "team_id": teams[1], "status": "QUESTIONABLE"}],
                                     {teams[1]: GID}, "test_feed", observed_at=(TIP - timedelta(hours=12)).isoformat())
    Autopilot(s, n_sims=500).tick(TIP - timedelta(hours=10), "2026-12-05")
    snap = s.latest_forecast(GID)
    sc = {r["scenario"] for r in s.q("SELECT DISTINCT scenario FROM minutes_forecasts WHERE snapshot_id=? AND team_id=?",
                                     (snap["snapshot_id"], teams[1]))}
    assert sc == {"S0", "S1"}                               # plays / out shown separately, no invented probability
    plays = s.q("SELECT * FROM minutes_forecasts WHERE snapshot_id=? AND player_id='T1P01'", (snap["snapshot_id"],))
    assert [r["scenario"] for r in plays] == ["S0"]


def test_backtest_rows_never_use_same_day(env):
    s, _ = env
    rows = s.logs_before("2026-11-20")
    assert rows and max(r["game_date_et"] for r in rows) < "2026-11-20"
