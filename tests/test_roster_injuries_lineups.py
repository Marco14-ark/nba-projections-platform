from datetime import datetime

import pytest

from nba_platform.core.enums import AvailabilityStatus as S, LineupState, TransactionType as T
from nba_platform.core.hashing import HashTree, recompute_plan
from nba_platform.core.timeutil import ET
from nba_platform.db.store import Store
from nba_platform.injuries.monitor import InjuryMonitor
from nba_platform.lineups.resolve import build_scenarios, expected_starters, resolve_team
from nba_platform.roster.sync import RosterEntry, RosterSync, roster_on, team_of
from nba_platform.sources.injury_report import parse_text_lines
from nba_platform.sources.teams import TEAMS

LAL, BOS, WAS, POR = 1610612747, 1610612738, 1610612764, 1610612757


@pytest.fixture
def store():
    s = Store()
    for t in TEAMS:
        s.upsert_team(*t)
    return s


def test_roster_add_trade_history_and_no_duplicates(store):
    rs = RosterSync(store)
    r1 = rs.ingest_snapshot("nba", "2026-10-15", [RosterEntry(LAL, "Ann", "Able", 101), RosterEntry(LAL, "Ben", "Baker", 102),
                                                  RosterEntry(BOS, "Cal", "Cole", 103)])
    assert len(r1.events) == 3 and all(e.txn_type is T.ROSTER_ADD_DETECTED for e in r1.events)
    r2 = rs.ingest_snapshot("nba", "2026-12-01", [RosterEntry(BOS, "Ann", "Able", 101), RosterEntry(LAL, "Ben", "Baker", 102),
                                                  RosterEntry(BOS, "Cal", "Cole", 103)])
    assert [(e.player_id, e.txn_type) for e in r2.events] == [("P101", T.TEAM_CHANGE_DETECTED)]
    assert r2.affected_team_ids == {LAL, BOS}
    assert team_of(store, "P101", "2026-11-30") == LAL and team_of(store, "P101", "2026-12-01") == BOS
    assert roster_on(store, LAL, "2026-12-05") == ["P102"]
    assert store.q("SELECT COUNT(*) c FROM players")[0]["c"] == 3
    # idempotent
    assert rs.ingest_snapshot("nba", "2026-12-01", [RosterEntry(BOS, "Ann", "Able", 101), RosterEntry(LAL, "Ben", "Baker", 102),
                                                    RosterEntry(BOS, "Cal", "Cole", 103)]).events == []


def test_partial_snapshot_never_drops_other_teams(store):
    rs = RosterSync(store)
    rs.ingest_snapshot("nba", "2026-10-15", [RosterEntry(LAL, "Ann", "Able", 101), RosterEntry(BOS, "Cal", "Cole", 103)])
    res = rs.ingest_snapshot("nba", "2026-10-20", [RosterEntry(LAL, "Ann", "Able", 101)], covered_team_ids={LAL})
    assert res.events == [] and team_of(store, "P103", "2026-10-21") == BOS
    res = rs.ingest_snapshot("nba", "2026-10-22", [RosterEntry(LAL, "Ann", "Able", 101)], covered_team_ids={LAL, BOS})
    assert [e.txn_type for e in res.events] == [T.ROSTER_DROP_DETECTED] and team_of(store, "P103", "2026-10-22") is None


def test_third_party_ids_map_by_name_and_ambiguity_is_not_guessed(store):
    rs = RosterSync(store)
    rs.ingest_snapshot("nba", "2026-10-15", [RosterEntry(LAL, "Ann", "Able", 101), RosterEntry(BOS, "Sam", "Smith", 201),
                                             RosterEntry(WAS, "Sam", "Smith", 202)])
    r = rs.ingest_snapshot("balldontlie", "2026-10-16", [RosterEntry(LAL, "Ann", "Able", source_id="bdl-9"),
                                                         RosterEntry(BOS, "Sam", "Smith", source_id="bdl-7")], covered_team_ids=set())
    assert not r.unresolved
    assert store.q("SELECT player_id FROM player_id_xref WHERE source='balldontlie' AND source_id='bdl-7'")[0][0] == "P201"
    assert rs.ids.match_name("smith|sam") is None                              # two Sam Smiths, no team: refuse
    r = rs.ingest_snapshot("x", "2026-10-16", [RosterEntry(BOS, "Zed", "Unknown", source_id="u1")], covered_team_ids=set())
    assert len(r.unresolved) == 1 and store.q("SELECT COUNT(*) c FROM players")[0]["c"] == 3


def test_explicit_trade_and_two_way(store):
    rs = RosterSync(store)
    rs.ingest_snapshot("nba", "2026-10-15", [RosterEntry(LAL, "Ann", "Able", 101)])
    rs.apply_transaction("P101", T.TRADE, "2027-01-10", WAS, "nba_transactions")
    assert team_of(store, "P101", "2027-01-09") == LAL and team_of(store, "P101", "2027-01-10") == WAS
    rs.apply_transaction("P101", T.TWO_WAY, "2027-02-01", WAS, "nba_transactions", contract_type="TWO_WAY")
    assert store.q("SELECT contract_type FROM roster_stints WHERE end_date IS NULL")[0][0] == "TWO_WAY"


def _setup_game(store):
    RosterSync(store).ingest_snapshot("nba", "2026-10-01", [
        RosterEntry(WAS, "Andre", "Alpha", 1), RosterEntry(WAS, "Kelly", "Beta Jr.", 2), RosterEntry(WAS, "Chris", "Gamma", 3),
        RosterEntry(WAS, "Will", "Wash", 4), RosterEntry(WAS, "Xan", "Xu", 5), RosterEntry(WAS, "Yul", "Yi", 6),
        RosterEntry(POR, "Pat", "Port", 10), RosterEntry(POR, "Quin", "Q", 11)])
    store.upsert_game({"game_id": "0022600500", "season": "2026-27", "game_date_et": "2027-01-27",
                       "tip_utc": "2027-01-28T00:00:00+00:00", "home_team_id": WAS, "away_team_id": POR})
    store.commit()


REPORT = """Injury Report: 01/27/27 10:45 AM
Game Date Game Time Matchup Team Player Name Current Status Reason
01/27/2027 07:00 (ET) POR@WAS Portland Trail Blazers NOT YET SUBMITTED
Washington Wizards Alpha, Andre Out Injury/Illness - Right Knee; Sprain
Beta Jr., Kelly Questionable Injury/Illness - Left Ankle; Sprain
Ghost, Nobody Out Injury/Illness - Back; Spasms
"""


def test_official_ingest_asof_and_unresolved(store):
    _setup_game(store)
    mon = InjuryMonitor(store)
    res = mon.ingest_official(parse_text_lines(REPORT), datetime(2027, 1, 27, 10, 45, tzinfo=ET),
                              observed_at="2027-01-27T15:50:00+00:00")
    assert res.stored == 2 and [r.player_raw for r in res.unresolved] == ["Ghost, Nobody"]
    assert res.teams_not_submitted == {POR}
    assert mon.state("0022600500", "2027-01-27T15:49:00+00:00") == {}            # nothing known before we saw it
    st = mon.state("0022600500", "2027-01-27T16:00:00+00:00")
    assert st == {"P1": S.OUT, "P2": S.QUESTIONABLE}
    # licensed feed at the same instant cannot override official
    mon.ingest_licensed([{"player_id": "P1", "team_id": WAS, "status": "AVAILABLE"}], {WAS: "0022600500"},
                        "balldontlie", observed_at="2027-01-27T15:50:00+00:00")
    assert mon.state("0022600500", "2027-01-27T16:00:00+00:00")["P1"] is S.OUT


def test_diff_relevance():
    ch = InjuryMonitor.diff({"a": S.QUESTIONABLE, "b": S.PROBABLE}, {"a": S.OUT, "b": S.AVAILABLE, "c": S.DOUBTFUL},
                            "g", core_players={"a"})
    rel = {c.player_id: c.relevance for c in ch}
    assert rel == {"a": "HIGH", "b": "LOW", "c": "MEDIUM"}


def test_lineup_resolution_scenarios_and_confirmed(store):
    _setup_game(store)
    InjuryMonitor(store).ingest_official(parse_text_lines(REPORT), datetime(2027, 1, 27, 10, 45, tzinfo=ET),
                                         observed_at="2027-01-27T15:50:00+00:00")
    ctx = resolve_team(store, "0022600500", WAS, "2027-01-27", "2027-01-27T16:00:00+00:00")
    assert "P1" in ctx.out and "P1" not in ctx.available()
    assert ctx.lineup_state is LineupState.EXPECTED
    sc = build_scenarios(ctx, {"P2": 30.0})
    assert len(sc) == 2 and all(s.prob is None for s in sc)
    assert all("P1" not in ctx.available(s) for s in sc)
    sc = build_scenarios(ctx, {"P2": 30.0}, {"QUESTIONABLE": {"n": 300, "p_play": 0.6}})
    assert sorted(round(s.prob, 2) for s in sc) == [0.4, 0.6]
    # confirmed lineup containing the questionable player resolves his uncertainty
    store.insert_lineup("0022600500", WAS, "CONFIRMED", ["P2", "P3", "P4", "P5", "P6"], "nba", "2027-01-27T23:35:00+00:00")
    ctx = resolve_team(store, "0022600500", WAS, "2027-01-27", "2027-01-27T23:40:00+00:00")
    assert ctx.lineup_state is LineupState.CONFIRMED and ctx.uncertain == {}
    # a confirmed starter later ruled OUT is removed and flagged
    InjuryMonitor(store).record_inactives("0022600500", WAS, ["P3"], "nba_boxscore_pregame", "2027-01-27T23:45:00+00:00")
    ctx = resolve_team(store, "0022600500", WAS, "2027-01-27", "2027-01-27T23:50:00+00:00")
    assert "P3" not in ctx.confirmed_starters and any(f.startswith("CONFIRMED_STARTER_NOW_OUT") for f in ctx.flags)
    assert expected_starters(ctx.available(), {"P1": 1.0}, ctx.confirmed_starters)[:4] == ["P2", "P4", "P5", "P6"]


def test_hash_tree_dependency_aware_plan():
    def tree(inj, mkt):
        h = HashTree()
        for l, v in (("ROSTER", 1), ("INJURY", inj), ("STARTERS", 1), ("MODEL", 1), ("MARKET", mkt)):
            h.set_input(l, v)
        return h.resolved()
    base = tree("a", "m1")
    assert recompute_plan(base, tree("a", "m1")) == []
    assert recompute_plan(base, tree("a", "m2")) == ["MARKET_COMPARISON"]          # odds only: no Monte Carlo
    assert recompute_plan(base, tree("b", "m1")) == ["ROTATION", "MINUTES", "OPPORTUNITY", "DISTRIBUTION",
                                                     "MARKET_COMPARISON"]
