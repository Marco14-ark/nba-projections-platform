"""Invariant tests for rates + joint Monte Carlo + markets (TEST-ONLY synthetic box stats)."""
import numpy as np
import pandas as pd
import pytest

from nba_platform.minutes.engine import MinutesEngine
from nba_platform.sim.game_sim import GameSimulator, TeamInput, accounting_check
from nba_platform.sim.markets import compare_to_book, game_markets, line_probs, player_props, player_samples
from nba_platform.sim.rates import OpportunityRates
from tests.synthetic import make_league


def add_box_stats(logs: pd.DataFrame, games: pd.DataFrame, seed: int = 4):
    """Attach plausible event counts to the synthetic minutes (test fixture only)."""
    rng = np.random.default_rng(seed)
    L = logs.copy()
    G = games.copy()
    G["home_poss"] = rng.normal(100, 4, len(G)).round()
    G["away_poss"] = G["home_poss"] + rng.integers(-1, 2, len(G))
    poss = dict(zip(G.game_id, G.home_poss))
    skill = {p: rng.uniform(0.6, 1.6) for p in L.player_id.unique()}
    L["team_poss_on"] = [poss[g] * m / 48 for g, m in zip(L.game_id, L.minutes)]
    lam = np.array([skill[p] for p in L.player_id]) * L["team_poss_on"] * 0.19
    L["fga"] = rng.poisson(lam)
    L["fg3a"] = rng.binomial(L["fga"], 0.4)
    L["rim_fga"] = rng.binomial(L["fga"] - L["fg3a"], 0.5)
    L["mid_fga"] = L["fga"] - L["fg3a"] - L["rim_fga"]
    L["fg3m"] = rng.binomial(L["fg3a"], 0.36)
    L["rim_fgm"] = rng.binomial(L["rim_fga"], 0.65)
    L["fgm"] = L["fg3m"] + L["rim_fgm"] + rng.binomial(L["mid_fga"], 0.43)
    L["ft_trips"] = rng.poisson(lam * 0.14)
    L["fta"] = L["ft_trips"] * 2
    L["ftm"] = rng.binomial(L["fta"], 0.78)
    L["tov"] = rng.poisson(lam * 0.15)
    L["pts"] = 2 * (L["fgm"] - L["fg3m"]) + 3 * L["fg3m"] + L["ftm"]
    L["team_miss_on"] = L["team_poss_on"] * 0.5
    L["opp_miss_on"] = L["team_poss_on"] * 0.5
    L["team_fgm_on"] = L["team_poss_on"] * 0.42
    L["oreb"] = rng.poisson(L["team_miss_on"] * 0.05)
    L["dreb"] = rng.poisson(L["opp_miss_on"] * 0.14)
    L["reb"] = L["oreb"] + L["dreb"]
    L["ast"] = rng.poisson(L["team_fgm_on"] * 0.12)
    for c in L.columns:
        if L[c].dtype.kind in "fi" and c not in ("minutes",):
            L.loc[L["played"] == 0, c] = 0 if c not in ("team_id",) else L.loc[L["played"] == 0, c]
    return L, G


@pytest.fixture(scope="module")
def setup():
    logs, games, pos, roster, teams = make_league()
    L, G = add_box_stats(logs, games)
    d = "2026-12-01"
    eng = MinutesEngine(L[L.game_date_et < d], G[G.game_date_et < d], d, positions=pos)
    r = OpportunityRates(L[L.game_date_et < d], G, d)
    h = [f"T0P{k:02d}" for k in range(13)]
    a = [f"T1P{k:02d}" for k in range(13)]
    sim = GameSimulator(eng, r)
    g = sim.simulate(TeamInput(teams[0], h, h[:5]), TeamInput(teams[1], a, a[:5]), n=3000, seed=11)
    return sim, g, teams, h, a, r


def test_rates_leakage_guard():
    logs, games, *_ = make_league()
    L, G = add_box_stats(logs, games)
    with pytest.raises(ValueError, match="leakage"):
        OpportunityRates(L, G, "2026-11-01")


def test_accounting_holds_in_every_simulation(setup):
    _, g, *_ = setup
    chk = accounting_check(g)
    assert all(chk.values()), chk


def test_reproducible(setup):
    sim, g, teams, h, a, _ = setup
    g2 = sim.simulate(TeamInput(teams[0], h, h[:5]), TeamInput(teams[1], a, a[:5]), n=3000, seed=11)
    assert np.array_equal(g.home.score, g2.home.score) and np.array_equal(g.home.stats["ast"], g2.home.stats["ast"])


def test_calibrated_spread_matches_targets(setup):
    _, g, *_, r = setup
    assert abs(g.margin.std() - r.sd_margin_t) < 1.5
    assert abs(g.total.std() - r.sd_total_t) < 1.5
    assert g.meta["ess"] > 1000


def test_combos_come_from_same_simulation(setup):
    _, g, _, h, *_ = setup
    pra = player_samples(g, h[0], "PRA")
    parts = player_samples(g, h[0], "PTS") + player_samples(g, h[0], "REB") + player_samples(g, h[0], "AST")
    assert np.array_equal(pra, parts)
    # correlation is preserved: PRA variance != sum of independent variances in general
    props = {(p["player_id"], p["market"]): p for p in player_props(g)}
    assert props[(h[0], "PRA")]["mean"] == pytest.approx(pra.mean())


def test_out_player_absent_and_minutes_drive_props(setup):
    sim, _, teams, h, a, _ = setup
    h2 = [p for p in h if p != "T0P00"]
    g = sim.simulate(TeamInput(teams[0], h2, h2[:5]), TeamInput(teams[1], a, a[:5]), n=2000, seed=3)
    assert "T0P00" not in g.home.players
    assert all(accounting_check(g).values())


def test_market_probabilities_and_push():
    x = np.array([10, 11, 12, 12, 13], float)
    p = line_probs(x, 12)
    assert p["p_push"] == pytest.approx(0.4) and p["p_over"] + p["p_under"] + p["p_push"] == pytest.approx(1)
    assert line_probs(x, 11.5)["p_push"] == 0


def test_game_markets_consistent(setup):
    _, g, *_ = setup
    gm = game_markets(g)
    assert gm["p_home_win"] + gm["p_away_win"] == pytest.approx(1)
    assert gm["projected_home"] + gm["projected_away"] == pytest.approx(gm["total"]["mean"])
    cov = [s["p_cover"] for s in gm["alt_spreads_home"]]
    assert all(b >= a for a, b in zip(cov, cov[1:]))           # bigger handicap -> easier cover


def test_book_comparison_requires_real_prices():
    out = compare_to_book(0.55, -110, -110)
    assert out["no_vig"] == pytest.approx(0.5) and out["gap"] == pytest.approx(0.05)
    assert out["ev_per_unit"] == pytest.approx(0.55 * (1 + 100 / 110) - 1)


def test_autopilot_stores_props_and_markets_snapshot():
    from datetime import datetime, timedelta, timezone
    import json, sqlite3
    from nba_platform.db.store import Store
    from nba_platform.pipeline.autopilot import Autopilot
    from nba_platform.roster.sync import RosterEntry, RosterSync
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
    s.upsert_game({"game_id": "0022688888", "season": "2026-27", "game_date_et": "2026-11-30", "tip_utc": tip.isoformat(),
                   "home_team_id": teams[0], "away_team_id": teams[1], "status": 1})
    s.commit()
    ap = Autopilot(s, n_sims=500, with_sim=True, sim_n=800)
    res = ap.tick(tip - timedelta(hours=8), "2026-11-30")
    assert [r.status for r in res] == ["COMPUTED"], res
    sid = res[0].snapshot_id
    props = s.q("SELECT * FROM prop_forecasts WHERE snapshot_id=?", (sid,))
    assert {r["market"] for r in props} >= {"PTS", "REB", "AST", "3PM", "PRA"}
    gm = json.loads(s.q("SELECT payload FROM game_market_forecasts WHERE snapshot_id=?", (sid,))[0][0])
    assert abs(gm["BASE"]["markets"]["p_home_win"] + gm["BASE"]["markets"]["p_away_win"] - 1) < 1e-9
    with pytest.raises(sqlite3.IntegrityError):
        s.x("UPDATE prop_forecasts SET mean=0 WHERE snapshot_id=?", (sid,))
