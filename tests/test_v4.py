"""V4: scenario integrity (publication <-> settlement), data integrity, blowout + joint MC switches."""
import gzip
import json
import sqlite3
from datetime import timedelta

import numpy as np
import pytest

from nba_platform.injuries.monitor import InjuryMonitor
from nba_platform.minutes.garbage import GarbageSampler, game_garbage
from nba_platform.pipeline import publication as pub
from nba_platform.pipeline.autopilot import Autopilot
from nba_platform.pipeline.integrity import archive_integrity, box_coherence
from nba_platform.pipeline.settle import category_of, settle_published
from nba_platform.sim.game_sim import GameSimulator, JointConfig, TeamInput, accounting_check
from nba_platform.sim.mixture import mix_game_sims
from tests.test_pipeline import GID, TIP, env  # noqa: F401
from tests.test_sim import setup  # noqa: F401

Q = "T1P01"          # bench player of the away team made QUESTIONABLE in several tests


def _questionable(s, teams, status="QUESTIONABLE"):
    InjuryMonitor(s).ingest_licensed([{"player_id": Q, "team_id": teams[1], "status": status}],
                                     {teams[1]: GID}, "test_feed", observed_at=(TIP - timedelta(hours=12)).isoformat())


def _official(s, sid, q_available: bool):
    rows = []
    for r in s.q("SELECT DISTINCT player_id, team_id FROM minutes_forecasts WHERE snapshot_id=?", (sid,)):
        avail = q_available if r["player_id"] == Q else True
        rows.append({"player_id": r["player_id"], "team_id": r["team_id"], "available": int(avail),
                     "played": int(avail), "minutes": 20.0 if avail else 0.0, "pts": 8, "reb": 3, "ast": 2, "fg3m": 1})
    return rows


def _rates(_d):
    return {"QUESTIONABLE": {"n": 120, "p_play": 0.6}}


_rates.source = "test_rates"


# ------------------------------------------------------------------ 1. scenario publication
def test_questionable_without_rates_is_conditional_not_mixture(env):  # noqa: F811
    s, teams = env
    _questionable(s, teams)
    r = Autopilot(s, n_sims=500).tick(TIP - timedelta(hours=10), "2026-12-05")
    sid = r[0].snapshot_id
    p = {(x["component"], x["team_id"]): dict(x) for x in s.q("SELECT * FROM forecast_publications WHERE snapshot_id=?", (sid,))}
    assert p[("MINUTES", teams[1])]["publication"] == "CONDITIONAL"
    assert p[("MINUTES", teams[1])]["published_label"] is None and p[("MINUTES", teams[1])]["weights_hash"] is None
    assert p[("MINUTES", teams[0])]["publication"] == "SINGLE"
    assert not s.q("SELECT 1 FROM minutes_forecasts WHERE snapshot_id=? AND scenario='MIXTURE'", (sid,))


def test_questionable_with_justified_rates_publishes_frozen_mixture(env):  # noqa: F811
    s, teams = env
    _questionable(s, teams)
    r = Autopilot(s, n_sims=500, play_rates_fn=_rates).tick(TIP - timedelta(hours=10), "2026-12-05")
    sid = r[0].snapshot_id
    row = dict(s.q("SELECT * FROM forecast_publications WHERE snapshot_id=? AND component='MINUTES' AND team_id=?",
                   (sid, teams[1]))[0])
    assert row["publication"] == "MIXTURE" and row["published_label"] == "MIXTURE"
    scen = json.loads(row["scenarios"])
    assert abs(sum(x["weight"] for x in scen) - 1) < 1e-9 and {x["weight"] for x in scen} == {0.6, 0.4}
    assert row["weights_hash"] == pub.weights_hash(scen) and row["weights_source"] == "play_rates:test_rates"
    # mixture minutes: the QUESTIONABLE player's mean is below his PLAYS-scenario mean (0 when OUT)
    m = {x["scenario"]: x["mean"] for x in s.q("SELECT * FROM minutes_forecasts WHERE snapshot_id=? AND player_id=?", (sid, Q))}
    assert m["MIXTURE"] < m["S0"]


def test_record_rejects_weights_not_summing_to_one(env):  # noqa: F811
    s, _ = env
    with pytest.raises(ValueError):
        pub.record(s, "x", "MINUTES", 1, [{"label": "S0", "conditions": {}, "weight": 0.7},
                                          {"label": "S1", "conditions": {}, "weight": 0.2}], "MIXTURE", "t", "now")


def test_publication_records_are_immutable(env):  # noqa: F811
    s, teams = env
    r = Autopilot(s, n_sims=500).tick(TIP - timedelta(hours=10), "2026-12-05")
    with pytest.raises(sqlite3.DatabaseError):
        s.x("UPDATE forecast_publications SET publication='MIXTURE' WHERE snapshot_id=?", (r[0].snapshot_id,))
    with pytest.raises(sqlite3.DatabaseError):
        s.x("DELETE FROM forecast_publications WHERE snapshot_id=?", (r[0].snapshot_id,))


def test_parse_and_match_conditions():
    assert pub.parse_conditions("P1:PLAYS+P2:OUT") == {"P1": "PLAYS", "P2": "OUT"}
    assert pub.parse_conditions("BASE") == {}
    with pytest.raises(ValueError):
        pub.parse_conditions("P1:MAYBE")
    p = {"scenarios": [{"label": "S0", "conditions": {"A": "PLAYS"}}, {"label": "S1", "conditions": {"A": "OUT"}}]}
    assert pub.matching_label(p, {"A": True}) == "S0"
    assert pub.matching_label(p, {"A": False}) == "S1"
    assert pub.matching_label(p, {}) is None                # unknown availability -> never guessed


# ------------------------------------------------------------------ 2. settlement of what was published
@pytest.mark.parametrize("q_available,label", [(True, "S0"), (False, "S1")])
def test_conditional_settles_only_the_scenario_that_happened(env, q_available, label):  # noqa: F811
    s, teams = env
    _questionable(s, teams)
    sid = Autopilot(s, n_sims=500).tick(TIP - timedelta(hours=10), "2026-12-05")[0].snapshot_id
    s.x("UPDATE games SET status=3 WHERE game_id=?", (GID,))
    out = settle_published(s, GID, _official(s, sid, q_available), 101, 99)
    assert out["status"] == "SETTLED" and out["category"] == "PROSPECTIVE"
    rows = s.q("SELECT DISTINCT settled_label, settle_method FROM settlements WHERE snapshot_id=? AND component='MINUTES'"
               " AND player_id IN (SELECT player_id FROM minutes_forecasts WHERE snapshot_id=? AND team_id=?)",
               (sid, sid, teams[1]))
    assert [(r[0], r[1]) for r in rows] == [(label, "CONDITIONAL_MATCHED")]
    assert settle_published(s, GID, [], 0, 0)["status"] == "ALREADY_SETTLED"
    with pytest.raises(sqlite3.DatabaseError):
        s.x("UPDATE settlements SET actual=0 WHERE snapshot_id=?", (sid,))


def test_mixture_is_settled_as_published_with_its_weights(env):  # noqa: F811
    s, teams = env
    _questionable(s, teams)
    sid = Autopilot(s, n_sims=500, play_rates_fn=_rates).tick(TIP - timedelta(hours=10), "2026-12-05")[0].snapshot_id
    s.x("UPDATE games SET status=3 WHERE game_id=?", (GID,))
    settle_published(s, GID, _official(s, sid, False), 101, 99)          # outcome must not change what is settled
    r = s.q("""SELECT DISTINCT settled_label, settle_method, weights_hash FROM settlements WHERE snapshot_id=? AND
               component='MINUTES' AND player_id=?""", (sid, Q))
    w = s.q("SELECT weights_hash FROM forecast_publications WHERE snapshot_id=? AND component='MINUTES' AND team_id=?",
            (sid, teams[1]))[0][0]
    assert [(x[0], x[1], x[2]) for x in r] == [("MIXTURE", "PUBLISHED_MIXTURE", w)]


def test_unknown_availability_is_not_settled(env):  # noqa: F811
    s, teams = env
    _questionable(s, teams)
    sid = Autopilot(s, n_sims=500).tick(TIP - timedelta(hours=10), "2026-12-05")[0].snapshot_id
    s.x("UPDATE games SET status=3 WHERE game_id=?", (GID,))
    official = [r for r in _official(s, sid, True) if r["player_id"] != Q]   # questionable player missing from box
    settle_published(s, GID, official, 100, 90)
    ev = s.q("SELECT status FROM settlement_events WHERE snapshot_id=? AND component='MINUTES' AND team_id=?", (sid, teams[1]))
    assert [e[0] for e in ev] == ["NO_MATCHING_SCENARIO"]


def test_post_tip_snapshot_is_never_settled(env):  # noqa: F811
    s, teams = env
    sid = Autopilot(s, n_sims=500).tick(TIP - timedelta(hours=10), "2026-12-05")[0].snapshot_id
    snap = dict(s.q("SELECT * FROM forecast_snapshots WHERE snapshot_id=?", (sid,))[0])
    late = {**snap, "snapshot_id": "LATE", "created_at": (TIP + timedelta(minutes=5)).isoformat(),
            "hashes": json.loads(snap["hashes"]), "context": json.loads(snap["context"]), "refresh_window": "LATE"}
    s.insert_forecast(late, [])
    s.x("UPDATE games SET status=3 WHERE game_id=?", (GID,))
    out = settle_published(s, GID, _official(s, sid, True), 100, 90)
    assert out["snapshot_id"] == sid


def test_categories_are_kept_apart():
    assert category_of({"mode": "PROSPECTIVE", "stack": "STACK_V2_ENV"}) == "PROSPECTIVE"
    assert category_of({"mode": "PROSPECTIVE_SHADOW", "stack": "STACK_V3_SIGMA"}) == "SHADOW"
    assert category_of({"mode": "REPLAY"}) == "REPLAY"
    assert category_of({"mode": "PROSPECTIVE", "stack": "STACK_V3_SIGMA"}) == "SHADOW"


# ------------------------------------------------------------------ 3. OUT players / confirmed starters
def test_out_player_absent_everywhere_and_confirmed_starters_used(env):  # noqa: F811
    s, teams = env
    InjuryMonitor(s).ingest_licensed([{"player_id": "T0P00", "team_id": teams[0], "status": "OUT"}], {teams[0]: GID},
                                     "test_feed", observed_at=(TIP - timedelta(hours=12)).isoformat())
    confirmed = ["T0P01", "T0P02", "T0P03", "T0P04", "T0P09"]
    s.insert_lineup(GID, teams[0], "CONFIRMED", confirmed, "test", (TIP - timedelta(hours=11)).isoformat())
    s.commit()
    sid = Autopilot(s, n_sims=500).tick(TIP - timedelta(hours=10), "2026-12-05")[0].snapshot_id
    assert not s.q("SELECT 1 FROM minutes_forecasts WHERE snapshot_id=? AND player_id='T0P00'", (sid,))
    ctx = json.loads(s.q("SELECT context FROM forecast_snapshots WHERE snapshot_id=?", (sid,))[0][0])
    assert ctx["confirmed_starters"]["home"] == confirmed
    assert sorted(ctx["scenarios"]["home"][0]["starters"]) == sorted(confirmed)


# ------------------------------------------------------------------ 4. data integrity
def test_archive_integrity_detects_newline_and_duplicates(tmp_path):
    p = tmp_path / "a.jsonl.gz"
    with gzip.open(p, "wt") as f:
        f.write(json.dumps({"gameId": "1"}) + "\n" + json.dumps({"gameId": "1"}) + "\n" + json.dumps({"gameId": "2"}))
    r = archive_integrity(p)
    assert r == {"records": 3, "unique_games": 2, "duplicates": 1, "bad_json": 0, "missing_trailing_newline": True}


def _box(home_min, home_pts, score, period=4):
    pl = lambda m, p: {"statistics": {"minutes": f"PT{int(m)}M{int(round((m % 1) * 60)):02d}.00S", "points": p}}  # noqa: E731
    return {"period": period, "homeTeam": {"score": score, "players": [pl(home_min / 5, home_pts // 5)] * 5},
            "awayTeam": {"score": 100, "players": [pl(48 + 5 * (period - 4), 20)] * 5}}


def test_box_coherence_minutes_and_points():
    assert box_coherence(_box(240, 100, 100)) == []
    assert any("MINUTES_SUM" in x for x in box_coherence(_box(230, 100, 100)))
    assert any("POINTS_SUM" in x for x in box_coherence(_box(240, 100, 104)))
    assert box_coherence(_box(265, 100, 100, period=5)) == []


# ------------------------------------------------------------------ 5. mixture + joint MC + blowout switches
def test_mix_game_sims_exact_and_accounting(setup):  # noqa: F811
    sim, g, teams, h, a, r = setup
    g2 = sim.simulate(TeamInput(teams[0], h[1:], h[1:6]), TeamInput(teams[1], a, a[:5]), n=3000, seed=11)
    mix, pick = mix_game_sims([g, g2], [0.7, 0.3], seed=3)
    assert all(accounting_check(mix).values())
    assert abs(np.mean(pick == 0) - 0.7) < 0.03
    k = mix.home.players.index(h[0])
    assert np.all(mix.home.stats["minutes"][pick == 1, k] == 0)      # OUT in scenario 2 -> zero
    assert np.allclose(mix.margin[pick == 0], g.margin[pick == 0])


def test_joint_switches_off_reproduce_champion_and_on_keep_accounting(setup):  # noqa: F811
    sim, g, teams, h, a, r = setup
    base = GameSimulator(sim.m, r, JointConfig()).simulate(TeamInput(teams[0], h, h[:5]), TeamInput(teams[1], a, a[:5]),
                                                            n=3000, seed=11)
    assert np.array_equal(base.home.stats["pts"], g.home.stats["pts"])
    c = GameSimulator(sim.m, r, JointConfig(form_sd=0.13, team_use_sd=0.04, ast_on_makes=True)).simulate(
        TeamInput(teams[0], h, h[:5]), TeamInput(teams[1], a, a[:5]), n=3000, seed=11)
    assert all(accounting_check(c).values())
    assert c.meta["joint"] == "f0.130_t0.040_a1"


def test_garbage_measure_and_sampler():
    import pandas as pd
    seg = pd.DataFrame({"game_id": ["g"] * 3, "period": [3, 4, 4], "start": [0, 0, 360.0], "end": [720, 360.0, 720],
                        "home_pts": [30, 2, 2], "away_pts": [2, 0, 0], "home5": "", "away5": ""})
    gb = game_garbage(seg)
    assert gb.garbage_min.iloc[0] == pytest.approx(12.0)        # lead 28 from the start of Q4
    gs = GarbageSampler(np.array([2.0] * 40 + [30.0] * 40), np.array([0.0] * 40 + [10.0] * 40))
    rng = np.random.default_rng(0)
    d = gs.draw(np.array([1.0, 31.0, 31.0]), np.array([0, 0, 1]), rng)
    assert d[0] == 0.0 and d[1] == 10.0 and d[2] == 0.0          # OT games never garbage time


def test_props_and_game_mixture_published_with_sim():
    from datetime import datetime, timezone
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
    gid = "0022688888"
    s.upsert_game({"game_id": gid, "season": "2026-27", "game_date_et": "2026-11-30", "tip_utc": tip.isoformat(),
                   "home_team_id": teams[0], "away_team_id": teams[1], "status": 1})
    InjuryMonitor(s).ingest_licensed([{"player_id": Q, "team_id": teams[1], "status": "QUESTIONABLE"}], {teams[1]: gid},
                                     "test_feed", observed_at=(tip - timedelta(hours=12)).isoformat())
    s.commit()
    sid = Autopilot(s, n_sims=500, with_sim=True, sim_n=800, play_rates_fn=_rates).tick(
        tip - timedelta(hours=8), "2026-11-30")[0].snapshot_id
    p = {x["component"]: dict(x) for x in s.q("SELECT * FROM forecast_publications WHERE snapshot_id=? AND team_id=0", (sid,))}
    assert p["PROPS"]["publication"] == "MIXTURE" and p["PROPS"]["published_label"] == "MIX"
    assert p["GAME"]["weights_hash"] == p["PROPS"]["weights_hash"] is not None
    payload = json.loads(s.q("SELECT payload FROM game_market_forecasts WHERE snapshot_id=?", (sid,))[0][0])
    assert set(payload) == {"H0A0", "H0A1", "MIX"} and payload["MIX"]["scenario"]["weights"] == pytest.approx([0.6, 0.4])
    mixq = s.q("SELECT p_dnp FROM prop_forecasts WHERE snapshot_id=? AND player_id=? AND scenario='MIX' AND market='PTS'", (sid, Q))
    assert 0.35 < mixq[0][0] < 0.5                     # OUT in ~40% of mixture draws (+ small DNP chance)


def test_prop_distributions_stored_and_exact():
    """V6: the stored counts reproduce the published summary (median/percentiles/fair-line probs) exactly."""
    from datetime import datetime, timezone
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
    s.upsert_game({"game_id": "0022677777", "season": "2026-27", "game_date_et": "2026-11-30", "tip_utc": tip.isoformat(),
                   "home_team_id": teams[0], "away_team_id": teams[1], "status": 1})
    s.commit()
    sid = Autopilot(s, n_sims=500, with_sim=True, sim_n=800).tick(tip - timedelta(hours=8), "2026-11-30")[0].snapshot_id
    rows = s.q("SELECT * FROM prop_distributions WHERE snapshot_id=?", (sid,))
    assert {r["market"] for r in rows} >= {"PTS", "REB", "AST", "3PM", "PRA", "PR", "PA", "RA", "FGA", "FGM", "3PA"}
    for r in rows[:60]:
        c = np.array(json.loads(r["counts"])); assert c.sum() == r["n"]
        x = np.repeat(np.arange(r["lo"], r["lo"] + len(c)), c).astype(float)
        p = s.q("SELECT * FROM prop_forecasts WHERE snapshot_id=? AND player_id=? AND market=? AND scenario=?",
                (sid, r["player_id"], r["market"], r["scenario"]))
        if p:
            p = p[0]
            assert np.isclose(np.percentile(x, 50), p["median"]) and np.isclose(np.percentile(x, 90), p["p90"])
            assert np.isclose(x.mean(), p["mean"]) and np.isclose(np.mean(x > p["model_fair_line"]), p["p_over_fair"])
    with pytest.raises(sqlite3.DatabaseError):
        s.x("DELETE FROM prop_distributions WHERE snapshot_id=?", (sid,))
