"""V9: injury-impact-ranking-v1, pace challenger config, champion fingerprint stability."""
from dataclasses import asdict

import numpy as np
import pandas as pd

from nba_platform.core.hashing import sha
from nba_platform.injuries import impact
from nba_platform.sim.rates import RatesConfig


def _logs():
    """Synthetic team: STAR (35 min, high usage), STARTER (30), SIXTH (24), BENCH (16), FRINGE (4),
    LONGOUT (was a 30-min starter but missed the last 10 games). 20 team games before 2026-12-01."""
    rows = []
    dates = pd.date_range("2026-11-05", periods=20).strftime("%Y-%m-%d")
    roles = {"STAR": (35, 1, 22, 8), "STARTER": (30, 1, 12, 2), "SIXTH": (24, 0, 10, 2), "BENCH": (16, 0, 5, 1),
             "FRINGE": (4, 0, 1, 0), "LONGOUT": (30, 1, 12, 2), "ROOKIE": (0, 0, 0, 0)}
    for i, d in enumerate(dates):
        for pid, (m, st, fga, fta) in roles.items():
            if pid == "LONGOUT" and i >= 10:
                continue
            if pid == "ROOKIE" and i < 18:
                continue
            mm = 20 if pid == "ROOKIE" else m
            rows.append({"game_id": f"G{i:02d}", "game_date_et": d, "team_id": 1, "player_id": pid, "played": 1,
                         "minutes": float(mm), "started": st, "fga": fga if pid != "ROOKIE" else 8, "fta": fta, "tov": 2 if pid == "STAR" else 1,
                         "pts": fga * 1.1 + fta * .8, "ast": 3, "reb": 4, "team_poss_on": mm * 2.1})
    return pd.DataFrame(rows)


POS = {"STAR": "G", "STARTER": "F", "SIXTH": "G", "BENCH": "F", "FRINGE": "G", "LONGOUT": "C", "ROOKIE": "F"}


def _ab(*pids, status="OUT"):
    return [{"player_id": p, "player": p, "status": status, "reason": "x"} for p in pids]


def test_categories_follow_real_role():
    out = {o["player_id"]: o for o in impact.rank_absences(_logs(), 1, "2026-12-01", _ab("STAR", "SIXTH", "BENCH", "FRINGE"), POS)}
    assert out["STAR"]["category"] == "CRITICAL"
    assert out["SIXTH"]["category"] == "HIGH"
    assert out["BENCH"]["category"] == "MEDIUM"
    assert out["FRINGE"]["category"] == "LOW" and not out["FRINGE"]["show_main"]
    assert out["STAR"]["show_main"] and out["SIXTH"]["show_main"]
    assert out["BENCH"]["show_main"]          # MEDIUM shown because a CRITICAL/HIGH player is also out


def test_ordering_by_impact_not_alphabetical():
    ranked = impact.rank_absences(_logs(), 1, "2026-12-01", _ab("BENCH", "FRINGE", "STAR"), POS)
    assert [o["player_id"] for o in ranked][0] == "STAR" and ranked[-1]["player_id"] == "FRINGE"


def test_long_absence_is_not_a_new_loss():
    o = impact.rank_absences(_logs(), 1, "2026-12-01", _ab("LONGOUT"), POS)[0]
    assert o["long_absence"] and o["games_missed"] >= impact.LONG_ABSENCE and not o["show_main"]
    assert o["category"] in ("HIGH", "CRITICAL")          # role is still known, just already incorporated


def test_medium_alone_hidden_unless_consequential():
    o = impact.rank_absences(_logs(), 1, "2026-12-01", _ab("BENCH"), POS)[0]
    assert o["category"] == "MEDIUM" and not o["show_main"]
    # the model reassigns >= 4 minutes to a teammate -> consequential -> shown
    o = impact.rank_absences(_logs(), 1, "2026-12-01", _ab("BENCH"), POS, projected={"SIXTH": {"min": 29.0, "pts": 13}}, recent_team_min={})[0]
    assert o["show_main"] and o["beneficiaries"][0]["player_id"] == "SIXTH" and np.isclose(o["beneficiaries"][0]["delta_min"], 5.0)


def test_rookie_new_role_counts():
    """A player who just entered the rotation is ranked by his CURRENT role, not by season averages."""
    o = impact.rank_absences(_logs(), 1, "2026-12-01", _ab("ROOKIE"), POS)[0]
    assert o["role"]["min"] == 20.0 and o["category"] == "MEDIUM"


def test_probable_medium_not_on_main_view_and_statuses_kept():
    ranked = impact.rank_absences(_logs(), 1, "2026-12-01", _ab("STAR") + _ab("BENCH", status="PROBABLE"), POS)
    b = next(o for o in ranked if o["player_id"] == "BENCH")
    assert b["status"] == "PROBABLE" and not b["show_main"]


def test_no_invented_beneficiaries_without_model_projection():
    o = impact.rank_absences(_logs(), 1, "2026-12-01", _ab("STAR"), POS)[0]
    assert o["beneficiaries"] == []


def test_unknown_player_has_no_role_and_is_low():
    o = impact.rank_absences(_logs(), 1, "2026-12-01", _ab("NOBODY"), POS)[0]
    assert o["category"] == "LOW" and o["role"] is None and not o["show_main"]


def test_champion_rates_fingerprint_unchanged_and_challenger_differs():
    d = asdict(RatesConfig())
    d.pop("pace_norm"); d.pop("hl_use_games")
    assert RatesConfig().digest() == sha(d)
    assert RatesConfig(pace_norm="team_mean").digest() != RatesConfig().digest()


def test_team_mean_pace_has_no_double_count():
    from nba_platform.sim.rates import OpportunityRates
    r = OpportunityRates.__new__(OpportunityRates)
    r.cfg = RatesConfig(pace_norm="team_mean")
    r.T = {1: {"pace48": 102.0}, 2: {"pace48": 98.0}, 3: {"pace48": 100.0}}
    r.LT = {"pace48": 99.0}
    r.env_pace = 1.05
    assert np.isclose(r.expected_pace(1, 2), 102 * 98 / 100.0)      # env factor not applied again
    r.cfg = RatesConfig()
    assert np.isclose(r.expected_pace(1, 2), 99 * (102 / 99) * (98 / 99) * 1.05)
