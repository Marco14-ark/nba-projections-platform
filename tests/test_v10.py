"""V10: MIN_V2 challengers (start-mix, return projection weights, role-weighted projection), SHOT_V2_USAGE,
champion fingerprints unchanged, minute budget respected in every simulation."""
from dataclasses import asdict

import numpy as np
import pytest

from nba_platform.core.hashing import sha
from nba_platform.minutes.engine import MinutesConfig, MinutesEngine
from nba_platform.sim.rates import RatesConfig
from tests.synthetic import make_league

T0 = 1610612700
ROSTER = [f"T0P{k:02d}" for k in range(13)]


@pytest.fixture(scope="module")
def league():
    return make_league()


def _eng(league, cfg=None, d="2026-11-30"):
    logs, games, pos, *_ = league
    return MinutesEngine(logs[logs.game_date_et < d], games[games.game_date_et < d], d, cfg, positions=pos)


def test_champion_minutes_fingerprint_unchanged():
    d = asdict(MinutesConfig())
    for k in ("start_mix", "start_mix_power", "proj_weight", "return_model"):
        d.pop(k)
    assert MinutesConfig().digest() == sha(d)
    for kw in ({"start_mix": True}, {"proj_weight": "var_role"}, {"return_model": "role_decay"}):
        assert MinutesConfig(**kw).digest() != MinutesConfig().digest()


def test_champion_rates_fingerprint_unchanged_with_usage_field():
    d = asdict(RatesConfig())
    d.pop("pace_norm"); d.pop("hl_use_games")
    assert RatesConfig().digest() == sha(d)
    assert RatesConfig(hl_use_games=15.0).digest() != RatesConfig().digest()


@pytest.mark.parametrize("kw", [{}, {"start_mix": True}, {"proj_weight": "var_role"}, {"return_model": "role_decay"},
                                {"start_mix": True, "proj_weight": "var_role", "return_model": "role_decay"}])
def test_budget_respected_every_simulation(league, kw):
    eng = _eng(league, MinutesConfig(**kw))
    bx, n_ot = eng.margin_draws(T0, T0 + 1, 3000, seed=5)
    tm = eng.simulate_team(T0, ROSTER, ROSTER[:5], {}, bx, n_ot, seed=5)
    assert np.allclose(tm.samples.sum(1), 240 + 25 * n_ot, atol=1e-6)        # never exceeds, OT only where it happens
    assert (tm.samples >= 0).all() and (tm.samples <= (48 + 5 * n_ot)[:, None] + 1e-9).all()


def test_start_mix_widens_only_when_lineup_unconfirmed(league):
    base = _eng(league)
    mix = _eng(league, MinutesConfig(start_mix=True))
    bx, n_ot = base.margin_draws(T0, T0 + 1, 3000, seed=7)
    st = ROSTER[:5]
    a = base.simulate_team(T0, ROSTER, st, {}, bx, n_ot, seed=7)
    b = mix.simulate_team(T0, ROSTER, st, {}, bx, n_ot, seed=7)
    c = mix.simulate_team(T0, ROSTER, st, {}, bx, n_ot, seed=7, starters_confirmed=True)
    assert np.array_equal(a.samples, c.samples)          # confirmed lineup -> identical to champion
    assert not np.array_equal(a.samples, b.samples)
    # expected starters keep most of the minutes, but some mass moves to plausible alternatives
    assert b.samples[:, :5].sum(1).mean() <= a.samples[:, :5].sum(1).mean() + 1e-6


def test_return_projection_scale_only_with_challenger(league):
    eng = _eng(league, MinutesConfig(return_model="role_decay"))
    p = eng.player_params(ROSTER[0], T0, ROSTER, True)
    assert p.proj_scale == 1.0 or any(f.startswith("RETURN") for f in p.flags)
    eng0 = _eng(league)
    assert all(eng0.player_params(q, T0, ROSTER, i < 5).proj_scale == 1.0 for i, q in enumerate(ROSTER))


def test_export_check_accepts_box_inactives_but_not_untimed_reports():
    import re
    src = open("scripts/check_web_export.py").read()
    assert 'need_report_ts = i["source"] == "nba_injury_report"' in src
