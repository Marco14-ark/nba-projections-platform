"""V8: game distributions, game-market math, odds normalisation, value engine, odds history.
All odds used here are SYNTHETIC test fixtures (tests/fixtures/odds_synthetic) and never published."""
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from nba_platform.odds import pricing as P
from nba_platform.odds.budget import allowance, can_spend
from nba_platform.odds.evaluate import build
from nba_platform.odds.match import match_event, match_player, player_key_from_provider, team_index
from nba_platform.odds.providers import SportsGameOdds, TheOddsAPI, FetchResult
from nba_platform.odds.store import OddsStore
from nba_platform.odds.value import dist_probs, side_probs, evaluate
from nba_platform.pipeline.autopilot import Autopilot
from nba_platform.sources.teams import TEAMS

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "odds_synthetic")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _counts(x):
    x = np.asarray(x, dtype=int)
    lo = int(x.min())
    return lo, np.bincount(x - lo).tolist()


# ------------------------------------------------------------------ distributions in snapshots
@pytest.fixture(scope="module")
def synth_snapshot():
    from nba_platform.db.store import Store
    from nba_platform.roster.sync import RosterEntry, RosterSync
    from tests.synthetic import make_league
    from tests.test_sim import add_box_stats
    logs, games, pos, roster, teams = make_league(n_dates=40)
    L, G = add_box_stats(logs, games)
    s = Store()
    for t in teams:
        s.upsert_team(t, f"T{t % 100}", "Synth", str(t))
    ents = [RosterEntry(t, p, "", source_id=p) for ti, t in enumerate(teams) for p in roster[ti]]
    for e in ents:
        s.x("""INSERT OR IGNORE INTO players(player_id, full_name, first_name, last_name, name_key, position, created_at)
               VALUES (?,?,?,?,?,?,?)""", (e.first_name, e.first_name, e.first_name, "", e.first_name.lower(), pos[e.first_name], "x"))
        s.x("INSERT OR IGNORE INTO player_id_xref VALUES ('synthetic', ?, ?)", (e.first_name, e.first_name))
    RosterSync(s).ingest_snapshot("synthetic", "2026-10-01", ents)
    for r in G.to_dict("records"):
        s.upsert_game(r)
    for r in L.to_dict("records"):
        s.upsert_log(r)
    tip = datetime(2026, 12, 1, 0, 30, tzinfo=timezone.utc)
    s.upsert_game({"game_id": "0022677778", "season": "2026-27", "game_date_et": "2026-11-30", "tip_utc": tip.isoformat(),
                   "home_team_id": teams[0], "away_team_id": teams[1], "status": 1})
    s.commit()
    sid = Autopilot(s, n_sims=500, with_sim=True, sim_n=800).tick(tip - timedelta(hours=8), "2026-11-30")[0].snapshot_id
    return s, sid


def test_game_distributions_stored_and_coherent(synth_snapshot):
    s, sid = synth_snapshot
    payload = json.loads(s.q("SELECT payload FROM game_market_forecasts WHERE snapshot_id=?", (sid,))[0]["payload"])
    rows = s.q("SELECT * FROM game_distributions WHERE snapshot_id=?", (sid,))
    by = {}
    for r in rows:
        by.setdefault(r["scenario"], {})[r["market"]] = (r["lo"], json.loads(r["counts"]), r["n"])
    assert set(by) == set(payload)
    for lb, d in by.items():
        assert set(d) == {"MARGIN", "TOTAL", "HOME_PTS", "AWAY_PTS"}
        m = payload[lb]["markets"]
        for k, (lo, c, n) in d.items():
            assert sum(c) == n
        mean = lambda k: float(np.dot(np.arange(d[k][0], d[k][0] + len(d[k][1])), d[k][1]) / d[k][2])
        assert np.isclose(mean("HOME_PTS"), m["projected_home"]) and np.isclose(mean("AWAY_PTS"), m["projected_away"])
        assert np.isclose(mean("TOTAL"), mean("HOME_PTS") + mean("AWAY_PTS"))
        assert np.isclose(mean("MARGIN"), mean("HOME_PTS") - mean("AWAY_PTS"))
        ml = side_probs("ML", "home", None, {"MARGIN": d["MARGIN"][:2]})["win"]
        assert np.isclose(ml, m["p_home_win"])
        assert np.isclose(ml + side_probs("ML", "away", None, {"MARGIN": d["MARGIN"][:2]})["win"], 1.0)
        # props and team totals come from the same simulations: player points add up to the team score
        for side, tkey in (("HOME", "projected_home"), ("AWAY", "projected_away")):
            tid = m["home_team_id"] if side == "HOME" else m["away_team_id"]
            sp = s.q("SELECT SUM(mean) v FROM prop_forecasts WHERE snapshot_id=? AND scenario=? AND market='PTS' AND team_id=?",
                     (sid, lb, tid))[0]["v"]
            assert np.isclose(sp, m[tkey], atol=1e-6)
        for a in m["alt_spreads_home"]:
            assert np.isclose(side_probs("SPREAD", "home", a["spread"], {"MARGIN": d["MARGIN"][:2]})["win"], a["p_cover"])
    with pytest.raises(sqlite3.DatabaseError):
        s.x("DELETE FROM game_distributions WHERE snapshot_id=?", (sid,))
    with pytest.raises(sqlite3.DatabaseError):
        s.x("UPDATE game_distributions SET n=1 WHERE snapshot_id=?", (sid,))


# ------------------------------------------------------------------ probability identities
@pytest.mark.parametrize("line", [-7, -6.5, -0.5, 0, 0.5, 3, 3.5, 10, 220, 221.5])
def test_over_under_push_sum_to_one(line):
    rng = np.random.default_rng(1)
    for x in (rng.integers(-30, 30, 5000), rng.integers(190, 250, 5000)):
        d = dist_probs(*_counts(x), line)
        assert np.isclose(d["over"] + d["under"] + d["push"], 1.0)
        assert np.isclose(d["over"], np.mean(x > line)) and np.isclose(d["push"], np.mean(x == line))
        if line != int(line):
            assert d["push"] == 0


def test_spread_signs_and_perspective():
    margin = {"MARGIN": _counts(np.full(100, 5))}     # home always wins by 5
    sp = lambda side, line: side_probs("SPREAD", side, line, margin)
    assert sp("home", -3.5)["win"] == 1 and sp("home", -6.5)["win"] == 0
    assert sp("home", -5)["push"] == 1 and sp("home", -5)["win"] == 0
    assert sp("away", 3.5)["win"] == 0 and sp("away", 5.5)["win"] == 1 and sp("away", 5)["push"] == 1
    assert sp("home", 2.5)["win"] == 1                 # home as an underdog line still covers
    rng = np.random.default_rng(3)
    m = {"MARGIN": _counts(rng.integers(-25, 26, 20000))}
    for line in (-7.5, -3, 0, 2.5, 6):                 # home -L and away +L partition the outcomes
        h, a = side_probs("SPREAD", "home", line, m), side_probs("SPREAD", "away", -line, m)
        assert np.isclose(h["win"] + a["win"] + h["push"], 1) and h["push"] == a["push"]


def test_team_totals_and_total_use_their_own_distributions():
    rng = np.random.default_rng(4)
    h, a = rng.integers(95, 130, 5000), rng.integers(90, 125, 5000)
    dd = {"HOME_PTS": _counts(h), "AWAY_PTS": _counts(a), "TOTAL": _counts(h + a), "MARGIN": _counts(h - a)}
    assert np.isclose(side_probs("TEAM_TOTAL", "home:over", 110.5, dd)["win"], np.mean(h > 110.5))
    assert np.isclose(side_probs("TEAM_TOTAL", "away:under", 108, dd)["win"], np.mean(a < 108))
    assert np.isclose(side_probs("TOTAL", "over", 221.5, dd)["win"], np.mean(h + a > 221.5))


# ------------------------------------------------------------------ pricing / EV
def test_pricing_math():
    assert np.isclose(P.american_to_decimal(-110), 1 + 100 / 110) and np.isclose(P.american_to_decimal(150), 2.5)
    assert np.isclose(P.implied_prob(-110), 110 / 210) and np.isclose(P.implied_prob(+150), 0.4)
    a, b, over = P.no_vig_pair(-110, -110)
    assert np.isclose(a, .5) and np.isclose(b, .5) and np.isclose(over, 220 / 210 - 1)
    a, b, _ = P.no_vig_pair(-150, 130)
    assert np.isclose(a + b, 1) and a > b
    for p in (0.2, 0.5, 0.61, 0.9):
        assert np.isclose(P.implied_prob(P.fair_american(p)), p)
        assert np.isclose(P.ev_per_unit(p, 0, P.fair_american(p)), 0, atol=1e-12)
    assert np.isclose(P.ev_per_unit(0.55, 0, -110), 0.55 * (100 / 110) - 0.45)
    assert np.isclose(P.ev_per_unit(0.50, 0.10, -110), 0.5 * (100 / 110) - 0.40)      # push refunds
    assert np.isclose(P.ev_per_unit(0.45, 0.10, P.fair_american_with_push(0.45, 0.10)), 0, atol=1e-12)
    with pytest.raises(ValueError):
        P.american_to_decimal(50)


def test_value_engine_decisions():
    e = evaluate(0.60, 0, 10000, -110, -110, 0.01, [], 10)
    assert e["decision"] == "VALUE" and e["edge"] > 0 and e["ev_conservative"] < e["ev"]
    e = evaluate(0.60, 0, 10000, -110, -110, 0.01, ["NOT_VALIDATED_VS_MARKET"], 10)
    assert e["decision"] == "NO_BET" and "NOT_VALIDATED_VS_MARKET" in e["blockers"]
    assert "STALE_ODDS" in evaluate(0.60, 0, 10000, -110, -110, 0.01, [], 500)["blockers"]
    assert "NO_CONSERVATIVE_EDGE" in evaluate(0.53, 0, 10000, -110, -110, 0.02, [], 5)["blockers"]
    assert evaluate(0.6, 0, 10000, -110, None, 0.01, [], 5)["no_vig"] is None


# ------------------------------------------------------------------ providers: normalisation, nothing invented
def test_sgo_parse_synthetic_fixture():
    ev = json.load(open(os.path.join(FIX, "SYNTHETIC_sgo_event.json")))
    qs = SportsGameOdds.parse_event(ev)
    got = {(q.book, q.market, q.team_side, q.side, q.line, q.price) for q in qs}
    assert ("draftkings", "ML", None, "home", None, -150) in got and ("draftkings", "ML", None, "away", None, 130) in got
    assert ("fanduel", "SPREAD", None, "home", -3.5, -110) in got and ("fanduel", "SPREAD", None, "away", 3.5, -110) in got
    assert ("fanduel", "TOTAL", None, "over", 221.5, -105) in got and ("betmgm", "TEAM_TOTAL", "home", "over", 112.5, -110) in got
    assert ("draftkings", "PTS", None, "over", 20.5, -120) in got and ("draftkings", "PTS", None, "under", 20.5, 100) in got
    assert not any(q.book == "bovada" for q in qs)          # empty price -> no quote
    assert not any(q.book == "caesars" for q in qs)         # invalid American odds -> no quote
    assert not any(q.line == 5.5 or q.line == 1.5 for q in qs)   # 1Q and unsupported stats ignored
    assert len(qs) == 9


def test_the_odds_api_parse_synthetic_fixture():
    ev = json.load(open(os.path.join(FIX, "SYNTHETIC_toa_event.json")))
    qs = TheOddsAPI.parse_event(ev)
    got = {(q.market, q.team_side, q.side, q.line, q.price) for q in qs}
    assert ("ML", None, "home", None, -150) in got and ("SPREAD", None, "away", 3.5, -110) in got
    assert ("TEAM_TOTAL", "away", "over", 108.5, -110) in got and ("PTS", None, "under", 20.5, 100) in got
    fg = [q for q in qs if q.market == "FGM"]
    assert fg and "FGM_ASSUMED_FROM_player_field_goals" in fg[0].notes
    assert not any(q.market == "BLK" for q in qs)


def test_providers_without_key_fetch_nothing(monkeypatch):
    monkeypatch.delenv("SGO_API_KEY", raising=False)
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    for cls in (SportsGameOdds, TheOddsAPI):
        p = cls()
        assert not p.configured
        r = p.fetch("2026-12-01T00:00:00+00:00", "2026-12-01T06:00:00+00:00")
        assert r.error == "NO_KEY" and r.quotes == [] and r.cost_units == 0


def test_provider_http_error_is_contained():
    class Resp:
        status_code, content, headers = 429, b"", {}
        def json(self): return {}
    class Sess:
        def get(self, *a, **k): return Resp()
    r = SportsGameOdds(key="x", session=Sess()).fetch("a", "b")
    assert not r.ok and r.error == "HTTP_429" and r.quotes == []


# ------------------------------------------------------------------ matching
def test_matching_teams_and_players():
    idx = team_index([{"team_id": t[0], "tricode": t[1], "city": t[2], "name": t[3]} for t in TEAMS])
    lac = next(t[0] for t in TEAMS if t[1] == "LAC")
    bos = next(t[0] for t in TEAMS if t[1] == "BOS")
    games = [{"game_id": "G1", "home_team_id": lac, "away_team_id": bos, "tip_utc": "2026-12-01T03:30:00+00:00"}]
    assert match_event("Los Angeles Clippers", "Boston Celtics", "2026-12-01T03:30:00Z", games, idx) == ("G1", "MATCHED")
    assert match_event("LOS_ANGELES_CLIPPERS_NBA", "BOSTON_CELTICS_NBA", "2026-12-01T03:00:00Z", games, idx)[0] == "G1"
    assert match_event("Boston Celtics", "Los Angeles Clippers", "2026-12-01T03:30:00Z", games, idx)[0] is None
    assert match_event("Los Angeles Clippers", "Boston Celtics", "2026-12-05T03:30:00Z", games, idx)[0] is None
    roster = {"playerone|synth": "P1", "jackson|jaren": "P2", "jackson|reggie": "P3"}
    roster["claxton|nicolas"] = "P4"
    assert match_player(player_key_from_provider("Nic Claxton", None), roster) == ("P4", "MATCHED_FUZZY")
    assert match_player(player_key_from_provider("Jaren Jackson Jr.", None), roster) == ("P2", "MATCHED")
    assert match_player(player_key_from_provider(None, "JAREN_JACKSON_1_NBA"), roster) == ("P2", "MATCHED")
    assert match_player(player_key_from_provider("J. Jackson", None), roster)[0] is None    # initial only -> never guessed


# ------------------------------------------------------------------ odds history store
def _res(provider="sportsgameodds", ts="2026-12-01T00:00:00+00:00"):
    return FetchResult(provider, "x", ts, 200, True, None, 3, 1)


def _q(price, line=20.5, side="over", book="draftkings"):
    return {"provider": "sportsgameodds", "book": book, "provider_event_id": "E", "kind": "PLAYER", "market": "PTS",
            "team_side": None, "player_key": "K", "player_id": "P1", "player_name": "x", "side": side, "line": line,
            "price": price, "available": True, "game_id": "G1", "match_status": "MATCHED"}


def test_odds_store_change_log_and_append_only():
    st = OddsStore()
    ev = [{"provider_event_id": "E", "game_id": "G1", "match_status": "MATCHED"}]
    assert st.record(_res(), [_q(-110), _q(-110, side="under")], ev)["n_changes"] == 2
    assert st.record(_res(ts="2026-12-01T01:00:00+00:00"), [_q(-110), _q(-110, side="under")], ev)["n_changes"] == 0
    assert st.record(_res(ts="2026-12-01T02:00:00+00:00"), [_q(-125, 21.5)], ev)["n_changes"] == 2   # move + withdrawn under
    cur = st.current()
    assert len(cur) == 1 and cur[0]["price"] == -125 and cur[0]["line"] == 21.5
    old = st.current(at="2026-12-01T01:30:00+00:00")
    assert {(r["side"], r["price"]) for r in old} == {("over", -110), ("under", -110)}
    assert st.usage("sportsgameodds", "2026-12") == 9
    with pytest.raises(sqlite3.DatabaseError):
        st.c.execute("UPDATE odds_quotes SET price=1")
    with pytest.raises(sqlite3.DatabaseError):
        st.c.execute("DELETE FROM odds_fetches")


def test_budget():
    now = datetime(2026, 11, 21, 12, tzinfo=timezone.utc)      # 10 days left in November
    a = allowance(2500, used_month=1200, used_today=0, now=now)
    assert np.isclose(a["daily_share"], (2250 - 1200) / 10)
    assert can_spend(a, 10) and not can_spend(a, 200)
    a = allowance(500, used_month=450, used_today=0, now=now)
    assert not can_spend(a, 3)


def test_evaluate_rows_never_value_without_validation():
    rng = np.random.default_rng(5)
    pts = rng.poisson(22, 10000)
    quotes = [{**_q(-110), "captured_at": "2026-12-01T00:00:00+00:00"}, {**_q(-110, side="under"), "captured_at": "2026-12-01T00:00:00+00:00"}]
    now = datetime(2026, 12, 1, 0, 30, tzinfo=timezone.utc)
    rows = build(quotes, None, {"P1": {"PTS": _counts(pts)}}, None, {}, now, True, set(), False, {"P1": "Synth"})
    assert len(rows) == 2 and all(r["decision"] == "NO_BET" for r in rows)
    assert all("NOT_VALIDATED_VS_MARKET" in r["blockers"] for r in rows)
    over = next(r for r in rows if r["side"] == "over")
    assert np.isclose(over["p_model"], np.mean(pts > 20.5), atol=1e-4)
    assert np.isclose(over["no_vig"], 0.5)
    rows = build(quotes, None, {"P1": {"PTS": _counts(pts)}}, None, {"PTS": {"status": "VALIDATED"}}, now, True, set(), False, {})
    assert any(r["decision"] == "VALUE" for r in rows) or all("NO_CONSERVATIVE_EDGE" in r["blockers"] for r in rows)
    rows = build(quotes, None, {"P1": {"PTS": _counts(pts)}}, None, {"PTS": {"status": "VALIDATED"}}, now, False, {"P1"}, False, {})
    assert all({"NOT_PREGAME", "LATE_STATUS_CHANGE"} <= set(r["blockers"]) for r in rows)


def test_odds_cycle_without_keys_records_no_key(tmp_path):
    from nba_platform.db.store import Store
    (tmp_path / "live").mkdir()
    Store(str(tmp_path / "live" / "prospective.db")).commit()
    env = {**os.environ, "NBA_DATA_DIR": str(tmp_path), "SGO_API_KEY": "", "ODDS_API_KEY": ""}
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "odds_cycle.py")], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    st = json.load(open(tmp_path / "odds" / "odds_status.json"))
    assert all(v["state"] == "NO_KEY" for v in st["providers"].values())
    c = sqlite3.connect(tmp_path / "odds" / "odds.db")
    assert c.execute("SELECT COUNT(*) FROM odds_quotes").fetchone()[0] == 0
