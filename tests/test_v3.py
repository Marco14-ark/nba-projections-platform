"""V3 tests: real (no-space) injury-report layout, glued name suffixes, tip-off refusal,
prospective settlement (pre-tip only, idempotent, immutable) and production calibration."""
import sqlite3
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from nba_platform.core.enums import AvailabilityStatus as S
from nba_platform.core.names import name_key_from_last_first
from nba_platform.pipeline.autopilot import Autopilot
from nba_platform.pipeline.settle import pregame_snapshot, settle_game
from nba_platform.sim.calibration_v2 import Calibrator, apply_production, brier, ece, fit_production, long_format
from nba_platform.sources.injury_report import classify
from nba_platform.sources.teams import team_id_from_full_name
from tests.test_pipeline import GID, TIP, env  # noqa: F401  (fixture re-export)


# ---------------------------------------------------------------- injury report, real layout
def test_glued_suffix_names():
    assert name_key_from_last_first("HolmesII,DaRon") == name_key_from_last_first("Holmes II, DaRon")
    assert name_key_from_last_first("JacksonJr.,Jaren") == name_key_from_last_first("Jackson Jr., Jaren")
    assert name_key_from_last_first("Williams,Jalen") == "williams|jalen"


def test_team_names_without_spaces():
    assert team_id_from_full_name("OrlandoMagic") == team_id_from_full_name("Orlando Magic") is not None
    assert team_id_from_full_name("LosAngelesLakers") == team_id_from_full_name("Los Angeles Lakers")


def test_classify_reasons_without_spaces():
    assert classify("Out", "GLeague-Two-Way") is not S.AVAILABLE
    assert classify("Out", "Injury/Illness-LeftAnkle;Sprain") is S.OUT
    assert classify("Questionable", "Injury/Illness-RightKnee;Soreness") is S.QUESTIONABLE
    assert classify("Out", "Suspension") is S.SUSPENDED


# ---------------------------------------------------------------- tip-off refusal + settlement
def test_autopilot_refuses_after_tip(env):  # noqa: F811
    s, _ = env
    ap = Autopilot(s, n_sims=500)
    # due_window returns None after tip, so tick never computes; nothing is written
    assert ap.tick(TIP + timedelta(minutes=1), "2026-12-05") == []
    assert s.q("SELECT COUNT(*) FROM forecast_snapshots WHERE game_id=?", (GID,))[0][0] == 0


def test_settlement_uses_last_pregame_snapshot_and_is_idempotent(env):  # noqa: F811
    s, teams = env
    ap = Autopilot(s, n_sims=1000)
    assert [r.status for r in ap.tick(TIP - timedelta(hours=10), "2026-12-05")] == ["COMPUTED"]
    snap = pregame_snapshot(s, GID, TIP.isoformat())
    assert snap is not None and snap["created_at"] < TIP.isoformat()
    # a snapshot created after tip must never be the one settled
    assert pregame_snapshot(s, GID, snap["created_at"]) is None
    players = [r["player_id"] for r in s.q("SELECT DISTINCT player_id FROM minutes_forecasts WHERE snapshot_id=?",
                                           (snap["snapshot_id"],))]
    official = [{"player_id": p, "played": int(i % 4 != 3), "minutes": 20.0, "pts": 10, "reb": 4, "ast": 2,
                 "fg3m": 1} for i, p in enumerate(players)]
    r1 = settle_game(s, GID, official, 110, 104)
    assert r1["status"] == "SETTLED" and r1["snapshot_id"] == snap["snapshot_id"] and r1["rows"] > 0
    rows = s.q("SELECT market, actual FROM forecast_results WHERE snapshot_id=?", (snap["snapshot_id"],))
    mins = [r["actual"] for r in rows if r["market"] == "MIN"]
    assert 0.0 in mins and 20.0 in mins                     # DNP settled as 0 minutes
    assert {r["market"] for r in rows} >= {"MIN"}
    r2 = settle_game(s, GID, official, 110, 104)
    assert r2["status"] == "ALREADY_SETTLED"
    assert s.q("SELECT COUNT(*) FROM forecast_results WHERE snapshot_id=?", (snap["snapshot_id"],))[0][0] == r1["rows"]
    with pytest.raises(sqlite3.DatabaseError):
        s.x("UPDATE forecast_results SET actual=0 WHERE snapshot_id=?", (snap["snapshot_id"],))


def test_settlement_without_pregame_snapshot(env):  # noqa: F811
    s, _ = env
    assert settle_game(s, GID, [], 100, 99)["status"] == "NO_PREGAME_SNAPSHOT"
    assert settle_game(s, "nope", [], 100, 99)["status"] == "UNKNOWN_GAME"


# ---------------------------------------------------------------- calibration V2
def _synthetic_props(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    p_true = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < p_true).astype(int)
    # raw model is over-confident: pushes probabilities away from 0.5
    raw = 1 / (1 + np.exp(-1.6 * np.log(p_true / (1 - p_true))))
    dates = pd.date_range("2025-10-21", periods=100).strftime("%Y-%m-%d")
    return pd.DataFrame({"date": rng.choice(dates, n), "market": rng.choice(["PTS", "REB"], n),
                         "p_over": raw, "over": y})


def test_beta_calibration_fixes_overconfidence(tmp_path):
    L = _synthetic_props()
    params = fit_production(L, "2026-01-01", path=str(tmp_path / "cal.json"))
    assert params["version"] == "BETA_MKT@2026-01-01" and set(params["markets"]) == {"PTS", "REB"}
    test = _synthetic_props(seed=1)
    cal = np.concatenate([apply_production(m, d.p_over, params) for m, d in test.groupby("market")])
    y = np.concatenate([d.over.to_numpy() for _, d in test.groupby("market")])
    raw = np.concatenate([d.p_over.to_numpy() for _, d in test.groupby("market")])
    assert brier(cal, y).mean() < brier(raw, y).mean() and ece(cal, y) < ece(raw, y)
    # unknown market passes through unchanged
    assert np.allclose(apply_production("STL", [0.3, 0.7], params), [0.3, 0.7])


def test_production_fit_never_uses_dates_at_or_after_cutoff(tmp_path):
    L = _synthetic_props()
    cut = sorted(L.date.unique())[50]
    params = fit_production(L, cut, path=str(tmp_path / "cal.json"))
    assert params["n_rows"] == int((L.date < cut).sum())


def test_capture_appends_safely_to_archive_without_trailing_newline(tmp_path):
    import gzip, json
    from nba_platform.pipeline import capture

    arch = tmp_path / "a.jsonl.gz"
    with gzip.open(arch, "wt") as f:
        f.write(json.dumps({"gameId": "0022600001", "actions": [], "box": {}}))   # no trailing newline

    class R:
        def __init__(self, body): self.body = body

    class F:
        def get(self, url, src):
            if "schedule" in url.lower():
                return R(json.dumps({"leagueSchedule": {"gameDates": [{"games": [
                    {"gameId": "0022600001", "gameStatus": 3}, {"gameId": "0022600002", "gameStatus": 3}]}]}}))
            return R(json.dumps({"game": {"actions": [], "gameId": "0022600002"}}))

    out = capture.capture_finals(F(), arch)
    assert out["added"] == 1 and out["already"] == 1
    ids = [json.loads(l)["gameId"] for l in gzip.open(arch, "rt") if l.strip()]
    assert ids == ["0022600001", "0022600002"]
