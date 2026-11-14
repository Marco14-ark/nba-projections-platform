"""V2 tests: data audit fixes, leak-free absences, segments, team rating, minutes V2 switches."""
import numpy as np
import pandas as pd
import pytest

from nba_platform.minutes.engine import MinutesConfig, MinutesEngine
from nba_platform.rating.team_rating import LineupRating, PointDiffRating, rest_days
from nba_platform.sources.pbp_reconstruct import infer_absences_asof, reconstruct_game, segments_frame
from tests.synthetic import make_league


def _ev(rows):
    cols = ["gameId", "actionNumber", "orderNumber", "clock", "timeActual", "period", "periodType", "actionType",
            "subType", "qualifiers", "personId", "teamId", "possession", "scoreHome", "scoreAway", "shotResult",
            "isFieldGoal", "area", "shotDistance", "assistPersonId", "stealPersonId", "blockPersonId",
            "foulDrawnPersonId", "x", "y"]
    out = []
    for i, r in enumerate(rows):
        base = {c: np.nan for c in cols}
        base.update({"gameId": 22699001, "actionNumber": i, "orderNumber": i, "timeActual": "2026-11-01T00:00:00Z",
                     "periodType": "REGULAR", "personId": 0, "possession": 0, "assistPersonId": 0, "stealPersonId": 0,
                     "blockPersonId": 0, "foulDrawnPersonId": 0})
        base.update(r)
        out.append(base)
    return pd.DataFrame(out)


def _game_rows(extra_bench_ejection: bool):
    H, A = 1610612747, 1610612738
    rows = []
    sh = sa = 0
    for per in (1, 2, 3, 4):
        for k in range(5):     # every starter of each team scores once per period -> evidence of presence
            sh += 2
            rows.append({"period": per, "clock": f"PT{11 - k:02d}M00.00S", "actionType": "2pt", "subType": "Layup",
                         "personId": 100 + k, "teamId": H, "possession": H, "scoreHome": sh, "scoreAway": sa,
                         "shotResult": "Made", "isFieldGoal": 1, "shotDistance": 2})
            sa += 2
            rows.append({"period": per, "clock": f"PT{10 - k:02d}M30.00S", "actionType": "2pt", "subType": "Layup",
                         "personId": 200 + k, "teamId": A, "possession": A, "scoreHome": sh, "scoreAway": sa,
                         "shotResult": "Made", "isFieldGoal": 1, "shotDistance": 2})
        if extra_bench_ejection and per == 4:
            rows.append({"period": 4, "clock": "PT01M00.00S", "actionType": "ejection", "subType": "other",
                         "personId": 199, "teamId": H, "scoreHome": sh, "scoreAway": sa})
    return rows


def test_bench_ejection_is_not_presence():
    g = reconstruct_game(_ev(_game_rows(True)))
    assert not any(f.startswith("LINEUP_6") for f in g.flags), g.flags
    assert sum(s["sec"] for s in g.players.values() if s["team_id"] == 1610612747) == pytest.approx(4 * 720 * 5)
    seg = segments_frame(g)
    assert seg.home_pts.sum() == g.home_score and seg.away_pts.sum() == g.away_score


def test_absence_inference_uses_no_future_information():
    logs, games, *_ = make_league(n_dates=30)
    played = logs[logs.played == 1]
    full = infer_absences_asof(played, games)
    cut = sorted(games.game_date_et.unique())[15]
    part = infer_absences_asof(played[played.game_date_et < cut], games[games.game_date_et < cut])
    a = full[(full.game_date_et < cut) & (full.played == 0)].sort_values(["game_id", "player_id"]).reset_index(drop=True)
    b = part[part.played == 0].sort_values(["game_id", "player_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[["game_id", "player_id", "not_playing_reason"]], b[["game_id", "player_id", "not_playing_reason"]])


def test_rating_ignores_future_segments_and_rest():
    logs, games, *_ = make_league(n_dates=30)
    g = games.sort_values("game_date_et")
    past, fut = g.game_id.iloc[0], g.game_id.iloc[-1]
    seg = pd.DataFrame([
        {"game_id": past, "period": 1, "start": 0, "end": 60, "home5": "a,b,c,d,e", "away5": "f,g,h,i,j",
         "home_pts": 3, "away_pts": 2, "home_poss": 2, "away_poss": 2},
        {"game_id": fut, "period": 1, "start": 0, "end": 60, "home5": "FUT1,b,c,d,e", "away5": "f,g,h,i,j",
         "home_pts": 30, "away_pts": 2, "home_poss": 2, "away_poss": 2}])
    m = LineupRating(seg, games, g.game_date_et.iloc[len(g) // 2])
    assert "FUT1" not in m.players and "a" in m.players
    r = rest_days(games)
    assert r.rest.min() >= 1 and r.rest.max() <= 7
    p = PointDiffRating(games, "2026-11-20")
    assert np.isfinite(p.margin(int(games.home_team_id.iloc[0]), int(games.away_team_id.iloc[0])))


def test_minutes_v2_competition_dnp_keeps_team_constraint():
    logs, games, pos, *_ = make_league()
    d = "2026-11-30"
    eng = MinutesEngine(logs[logs.game_date_et < d], games[games.game_date_et < d], d,
                        MinutesConfig(dnp_model="competition", ot_gamma=5.0), pos)
    roster = [f"T0P{k:02d}" for k in range(13)]
    bx, n_ot, mg = eng.margin_draws(1610612700, 1610612701, 3000, 3, return_margin=True)
    tm = eng.simulate_team(1610612700, roster, roster[:5], {}, bx, n_ot, 3, margin=mg)
    assert np.allclose(tm.samples.sum(1), 240 + 25 * n_ot)
    deep = tm.players.index("T0P12")
    assert np.mean(tm.samples[:, deep] == 0) > np.mean(tm.samples[:, tm.players.index("T0P05")] == 0)
