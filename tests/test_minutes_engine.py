import numpy as np
import pandas as pd
import pytest

from nba_platform.minutes.engine import MinutesEngine, project_minutes
from nba_platform.minutes.validate import backtest_baseline, backtest_engine, compare, crps_samples
from tests.synthetic import make_league


@pytest.fixture(scope="module")
def league():
    return make_league()


def _engine(league, d):
    logs, games, pos, *_ = league
    return MinutesEngine(logs[logs.game_date_et < d], games[games.game_date_et < d], d, positions=pos)


def test_leakage_guard_rejects_future_logs(league):
    logs, games, pos, *_ = league
    with pytest.raises(ValueError, match="leakage"):
        MinutesEngine(logs, games, "2026-11-01", positions=pos)


def test_team_minutes_sum_exactly_240_or_ot(league):
    eng = _engine(league, "2026-11-30")
    roster = [f"T0P{k:02d}" for k in range(13)]
    bx, n_ot = eng.margin_draws(1610612700, 1610612701, 4000, seed=5)
    tm = eng.simulate_team(1610612700, roster, roster[:5], {}, bx, n_ot, seed=5)
    sums = tm.samples.sum(1)
    assert np.allclose(sums, 240 + 25 * n_ot, atol=1e-6)
    assert (n_ot > 0).any(), "overtime should occur in a few thousand sims"
    assert (tm.samples >= 0).all() and (tm.samples <= (48 + 5 * n_ot)[:, None] + 1e-9).all()


def test_out_player_gets_zero_and_caps_respected(league):
    eng = _engine(league, "2026-11-30")
    roster = [f"T1P{k:02d}" for k in range(13)]
    avail = [p for p in roster if p != "T1P00"]          # T1P00 OUT -> excluded entirely
    bx, n_ot = eng.margin_draws(1610612701, 1610612702, 3000, seed=1)
    tm = eng.simulate_team(1610612701, avail, avail[:5], {"T1P01": 24.0}, bx, n_ot, seed=1)
    assert "T1P00" not in tm.players
    k = tm.players.index("T1P01")
    assert tm.samples[:, k].max() <= 24.0 + 1e-9
    assert np.allclose(tm.samples.sum(1), 240 + 25 * n_ot)


def test_reproducible_with_seed(league):
    eng = _engine(league, "2026-11-30")
    roster = [f"T2P{k:02d}" for k in range(12)]
    a = eng.simulate_team(1610612702, roster, roster[:5], {}, *eng.margin_draws(1610612702, 1610612703, 500, 9), seed=9)
    b = eng.simulate_team(1610612702, roster, roster[:5], {}, *eng.margin_draws(1610612702, 1610612703, 500, 9), seed=9)
    c = eng.simulate_team(1610612702, roster, roster[:5], {}, *eng.margin_draws(1610612702, 1610612703, 500, 10), seed=10)
    assert np.array_equal(a.samples, b.samples)
    assert not np.array_equal(a.samples, c.samples)


def test_backup_absorbs_minutes_when_starter_out(league):
    """Planted: T3P05 (G backup) absorbs ~75% of starter T3P00's minutes."""
    eng = _engine(league, "2026-12-03")
    roster = [p for p in [f"T3P{k:02d}" for k in range(13)]]
    with_star = eng.player_params("T3P05", 1610612703, roster, started=False)
    without = [p for p in roster if p != "T3P00"]
    no_star = eng.player_params("T3P05", 1610612703, without, started=True)
    other = eng.player_params("T3P07", 1610612703, without, started=False)   # C backup: not the absorber
    gain = no_star.a - with_star.a
    other_gain = other.a - eng.player_params("T3P07", 1610612703, roster, started=False).a
    assert 8 < gain < 22, gain                      # planted ~13.6 + share of the rest
    assert gain > 2 * max(other_gain, 0.5)
    assert no_star.a <= 48


def test_expected_starters_pick_the_real_replacement(league):
    eng = _engine(league, "2026-12-03")
    roster = [f"T3P{k:02d}" for k in range(13)]
    avail = [p for p in roster if p != "T3P00"]
    from nba_platform.lineups.resolve import expected_starters
    st = expected_starters(avail, eng.start_scores(1610612703, avail))
    assert "T3P05" in st and "T3P00" not in st


def test_coach_blowout_behaviour_learned(league):
    """Team 0 coach pulls starters hard (0.9), team 4 barely (0.15)."""
    eng = _engine(league, "2026-12-05")
    assert eng.team_slope[(1610612700, 1)] < eng.team_slope[(1610612704, 1)]
    assert eng.team_slope[(1610612700, 1)] < -0.3


def test_return_from_injury_flagged(league):
    eng = _engine(league, "2026-11-16")     # T0P00 out from day 14 -> streak before return on day 26
    p = eng.player_params("T0P00", 1610612700, [f"T0P{k:02d}" for k in range(13)], started=True)
    assert any(f.startswith("RETURN_FROM_INJURY") for f in p.flags)


def test_traded_player_new_team_uncertainty(league):
    eng = _engine(league, "2026-11-17")      # trade happened on day 24 = 2026-11-14
    roster3 = [f"T3P{k:02d}" for k in range(13)] + ["T2P06"]
    p = eng.player_params("T2P06", 1610612703, roster3, started=False)
    assert "NEW_TEAM" in p.flags
    assert p.sigma > eng.sd_bench


def test_projection_feasibility_activates_players():
    raw = np.array([[30.0, 30, 30, 30, 0, 0]])
    plays = np.array([[True, True, True, True, False, False]])   # only 4 players drawn to play
    x = project_minutes(raw, plays, np.full((1, 6), 48.0), np.array([240.0]), np.ones(6))
    assert np.isclose(x.sum(), 240) and (x > 0).sum() >= 5


def test_crps_matches_definition():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 4000)
    # analytic CRPS of N(0,1) at y=0 is 2*phi(0) - 1/sqrt(pi) ... = 0.2337
    assert abs(crps_samples(x, 0.0) - 0.2337) < 0.02


def test_walk_forward_beats_naive_baseline(league):
    logs, games, pos, *_ = league
    dates = sorted(logs.game_date_et.unique())[-6:]
    champ = backtest_engine(logs, games, dates, n_sims=800, positions=pos)
    base = backtest_baseline(logs, dates)
    m, b = champ.metrics["ALL"], base.metrics["ALL"]
    assert m["n"] == b["n"] > 100
    assert m["mae"] < b["mae"]
    assert m["crps"] < b["crps"]
    assert 0.6 <= m["cov80"] <= 0.97
    # with an absent rotation player the gap should be clearer
    if "injury_ctx=1" in champ.metrics and "injury_ctx=1" in base.metrics:
        assert champ.metrics["injury_ctx=1"]["mae"] < base.metrics["injury_ctx=1"]["mae"]
    cmp_ = compare(base, champ, n_boot=500)
    assert cmp_["mae_diff"] < 0 and cmp_["crps_ci95"][1] < 0
    assert cmp_["decision"] == "PROMOTE"
    # a model compared with itself must never be promoted
    assert compare(champ, champ, n_boot=200)["decision"] == "KEEP_CHAMPION"
